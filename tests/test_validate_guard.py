"""Offline tests for the Privacy Guard, the Validator's statistics and decisions, and skill status."""

from skillminer.guard import collect_case_values, guard, scan
from skillminer.mining.episodes import Episode, Step
from skillminer.validate import Verdict, compare, mcnemar_exact
from skillminer.writer import set_status, skill_name


def test_guard_redacts_case_values_but_keeps_policy_text():
    steps = [{"args": {"order_id": "A1001", "to": "priya@example.com", "new_address": "12 Park Lane, Leeds"},
              "result": {"status": "ok", "item": "Wireless earbuds", "reason": "Outside the 30-day refund window"}}]
    values = collect_case_values(steps)
    text = "Refund A1001 (wireless earbuds) for priya@example.com at 12 Park Lane, Leeds. Outside the 30-day refund window."
    clean, report = guard(text, values)
    assert clean == "Refund <ORDER_ID> (<ITEM>) for <EMAIL> at <ADDRESS>. Outside the 30-day refund window."
    assert report.clean and report.redacted_values == 4


def test_guard_pattern_scan_catches_unseen_personal_data():
    clean, report = guard("Call me on +44 7700 900123 or mail bob@corp.io, card 4111 1111 1111 1111", {})
    kinds = {f.kind for f in report.pattern_findings}
    assert {"PHONE", "EMAIL", "CARD_NUMBER"} <= kinds
    assert not report.clean
    assert "bob@corp.io" not in clean and "4111" not in clean
    assert scan("Use <EMAIL> and <ORDER_ID>") == []


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(8, 0) < 0.01
    assert mcnemar_exact(3, 3) == 1.0


def _ep(sid, ok):
    return Episode(sid, ["hi"], ["ok"], [Step("lookup_order", True)], meta={"scenario_id": sid, "workflow": "w", "ok": ok})


def _judge(e):
    return Verdict(compliant=e.meta["ok"], violation=False)


def test_compare_approves_a_clear_improvement():
    a = {f"s{i}": _ep(f"s{i}", False) for i in range(10)}
    b = {f"s{i}": _ep(f"s{i}", True) for i in range(10)}
    report = compare(a, b, _judge)
    assert (report.fixed, report.broken) == (10, 0)
    assert report.decision == "approve"


def test_compare_rejects_a_regression_and_waits_on_small_samples():
    good = {f"s{i}": _ep(f"s{i}", True) for i in range(4)}
    bad = {f"s{i}": _ep(f"s{i}", False) for i in range(4)}
    assert compare(good, bad, _judge).decision == "reject"
    a = {"s0": _ep("s0", False), "s1": _ep("s1", True)}
    b = {"s0": _ep("s0", True), "s1": _ep("s1", True)}
    assert compare(a, b, _judge).decision == "promising"


def test_skill_status_round_trip(tmp_path):
    d = tmp_path / "request-refund"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: request-refund\ndescription: x\nmetadata:\n  status: candidate\n---\n\nBody\n")
    (d / "skillminer.json").write_text('{"status": "candidate"}')
    set_status(d, "approved", {"validation": {"p_value": 0.01}})
    assert "status: approved" in (d / "SKILL.md").read_text()
    assert "Body" in (d / "SKILL.md").read_text()
    assert skill_name("Request_Refund!") == "request-refund"


def test_drift_watcher_flags_removed_and_changed_tools(tmp_path):
    import json

    from skillminer.drift import check_all

    for name, tools in {"a": {"lookup_order": ["order_id"]},
                        "b": {"issue_refund": ["order_id", "amount", "reason"]},
                        "c": {"track_shipment": ["order_id"]}}.items():
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: x\n---\nbody\n")
        (d / "skillminer.json").write_text(json.dumps({"tools": tools}))

    current = {"lookup_order": ["order_id"], "issue_refund": ["order_id", "amount"]}  # changed + one removed
    results = {r.skill: r for r in check_all(tmp_path, current)}
    assert results["a"].status == "ok"
    assert results["b"].status == "stale" and "parameters changed" in results["b"].problems[0]
    assert results["c"].status == "stale" and "no longer exists" in results["c"].problems[0]
