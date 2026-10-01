"""Offline tests for episodes, clustering, scoring and the LLM intent labeller (LLM faked)."""

import json
from types import SimpleNamespace

import pytest

from skillminer.mining import collapse_repeats, load_episodes, mine, score
from skillminer.mining import intents as intents_mod
from skillminer.mining.cluster import group_by_intent
from skillminer.mining.intents import IntentLabeler, _parse_json


def trace(session, t, message, tools, workflow=None):
    return {
        "session_id": session, "started_at": f"2026-10-01T00:00:{t:02d}+00:00", "user_message": message,
        "final_response": "ok", "meta": {"workflow": workflow} if workflow else {},
        "steps": [{"tool": name, "status": status, "args": {}, "result": {}} for name, status in tools],
    }


@pytest.fixture
def trace_dir(tmp_path):
    rows = []
    for i in range(4):
        rows.append(trace(f"r{i}", 1, "hi", []))
        rows.append(trace(f"r{i}", 2, f"I want a refund, item {i} is broken, order A10{i}",
                          [("lookup_order", "ok"), ("check_refund_eligibility", "ok"), ("issue_refund", "ok")], "refund"))
        rows.append(trace(f"t{i}", 1, f"where is my parcel A20{i}, when will it arrive",
                          [("lookup_order", "ok"), ("track_shipment", "ok")], "track"))
        rows.append(trace(f"a{i}", 1, f"please change the delivery address of order A30{i}",
                          [("lookup_order", "ok"), ("update_address", "ok")], "address"))
    rows.append(trace("idle", 1, "hello?", []))
    path = tmp_path / "app" / "day.jsonl"
    path.parent.mkdir()
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path


def test_collapse_repeats():
    assert collapse_repeats(["a!", "a!", "a!", "b", "a!"]) == ["a!", "b", "a!"]


def test_turns_are_joined_into_one_episode_per_session(trace_dir):
    episodes = {e.session_id: e for e in load_episodes(trace_dir)}
    assert len(episodes) == 13
    refund = episodes["r0"]
    assert refund.user_messages[0] == "hi" and "refund" in refund.user_messages[1]
    assert refund.path == ["lookup_order", "check_refund_eligibility", "issue_refund"]
    assert refund.meta == {"workflow": "refund"}
    assert not episodes["idle"].has_actions


def test_tfidf_mining_recovers_clear_workflows(trace_dir):
    result = mine(load_episodes(trace_dir), k=3)
    assert [e.session_id for e in result.unclustered] == ["idle"]
    s = score(result, lambda e: e.meta.get("workflow"))
    assert s["purity"] == 1.0 and s["ari"] == 1.0


def test_group_by_intent_and_branches(trace_dir):
    episodes = load_episodes(trace_dir)
    labels = {e.session_id: e.session_id[0] for e in episodes}  # r/t/a prefix = intent
    result = group_by_intent(episodes, labels, {"r": "refunds"})
    by_name = {c.keywords[0]: c for c in result.clusters}
    assert set(by_name) == {"r", "t", "a"}
    assert by_name["r"].keywords == ["r", "refunds"]
    assert by_name["t"].path_counts == [(("lookup_order", "track_shipment"), 4)]


def test_parse_json_tolerates_fences_and_noise():
    assert _parse_json('```json\n{"intent": "x"}\n```') == {"intent": "x"}
    assert _parse_json("no json here") == {}


async def test_labeller_reuses_intents_and_caches(trace_dir, tmp_path, monkeypatch):
    prompts = []

    async def fake_completion(model, messages, **kwargs):
        prompt = messages[0]["content"]
        prompts.append(prompt)
        text = prompt.split("Customer messages")[1]
        name = "request_refund" if "refund" in text else "Track Order!"  # messy name gets normalised
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"intent": name, "description": f"about {name}"})))])

    monkeypatch.setattr(intents_mod.litellm, "acompletion", fake_completion)
    episodes = [e for e in load_episodes(trace_dir) if e.session_id in ("r0", "r1", "t0")]
    cache = tmp_path / "intents.json"

    labeler = IntentLabeler("fake", cache_path=cache)
    labels = await labeler.label_all(episodes)
    assert labels == {"r0": "request_refund", "r1": "request_refund", "t0": "track_order"}
    assert "request_refund: about request_refund" in prompts[1]  # the 2nd call sees the intent created by the 1st

    calls_before = len(prompts)
    again = IntentLabeler("fake", cache_path=cache)
    assert await again.label_all(episodes) == labels
    assert len(prompts) == calls_before  # everything came from the cache
