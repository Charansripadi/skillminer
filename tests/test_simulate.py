"""Offline tests for the simulation loop: scripted agent model + scripted customer."""

import json

import litellm
from google.adk import Agent
from google.adk.apps import App

from skillminer.capture import TraceRecorderPlugin
from skillminer.simulate import Scenario, UserSimulator, run_scenarios

from fakes import ScriptedLlm, call, say


class ScriptedCustomer(UserSimulator):
    """Replays fixed customer messages; can raise a rate-limit error on the first call."""

    def __init__(self, messages, fail_first_with=None):
        super().__init__(model="scripted")
        self.messages = list(messages)
        self.fail_first_with = fail_first_with
        self.seen_transcripts = []

    async def next_message(self, scenario, transcript):
        if self.fail_first_with is not None:
            error, self.fail_first_with = self.fail_first_with, None
            raise error
        self.seen_transcripts.append(list(transcript))
        return self.messages.pop(0) if self.messages else None


def lookup_order(order_id: str) -> dict:
    """Look up an order."""
    return {"status": "ok", "order_id": order_id}


def track_shipment(order_id: str) -> dict:
    """Track a shipment."""
    return {"status": "ok", "eta": "2 days"}


def make_app(tmp_path, script):
    agent = Agent(name="shop", model=ScriptedLlm(model="scripted", script=script), instruction="test",
                  tools=[lookup_order, track_shipment])
    return App(name="shop", root_agent=agent, plugins=[TraceRecorderPlugin(trace_dir=tmp_path)])


def read_traces(tmp_path):
    [file] = tmp_path.rglob("*.jsonl")
    return [json.loads(line) for line in file.read_text().splitlines()]


SCENARIO = Scenario(id="t1", goal="track A1002", persona="terse",
                    label={"workflow": "track_order", "expected_path": ["lookup_order", "track_shipment"]})


async def test_multi_turn_conversation_records_traces_with_labels(tmp_path):
    app = make_app(tmp_path, [
        say("Which order?"),                                         # turn 1: agent asks for the ID
        call("lookup_order", order_id="A1002"),                      # turn 2: agent works
        call("track_shipment", order_id="A1002"),
        say("It arrives in 2 days."),
    ])
    customer = ScriptedCustomer(["where is my stuff", "A1002"])
    resets = []

    [result] = await run_scenarios(app, [SCENARIO], customer, reset_world=lambda: resets.append(1))

    assert result.ok and result.turns == 2
    assert resets == [1]
    # The customer sees the agent's previous reply before writing turn 2.
    assert customer.seen_transcripts[1] == [("customer", "where is my stuff"), ("agent", "Which order?")]

    turn1, turn2 = read_traces(tmp_path)
    assert turn1["steps"] == [] and turn1["final_response"] == "Which order?"
    assert [s["tool"] for s in turn2["steps"]] == ["lookup_order", "track_shipment"]
    assert turn2["session_id"] == turn1["session_id"]
    assert turn2["meta"]["workflow"] == "track_order"
    assert turn2["meta"]["scenario_id"] == "t1"


async def test_rate_limit_is_retried(tmp_path):
    app = make_app(tmp_path, [say("Hi! How can I help?")])
    rate_limited = litellm.RateLimitError("429 slow down", llm_provider="groq", model="x")
    customer = ScriptedCustomer(["hello"], fail_first_with=rate_limited)

    [result] = await run_scenarios(app, [SCENARIO], customer, base_delay=0)

    assert result.ok and result.turns == 1


async def test_other_errors_fail_the_scenario_but_not_the_run(tmp_path):
    app = make_app(tmp_path, [say("Hi!")])
    customer = ScriptedCustomer(["hello"], fail_first_with=ValueError("boom"))

    results = await run_scenarios(app, [SCENARIO, Scenario(id="t2", goal="g", persona="p")], customer, base_delay=0)

    assert [r.ok for r in results] == [False, True]
    assert "boom" in results[0].error


def test_trailing_done_sends_the_message_then_ends():
    sim = UserSimulator(model="unused")
    sc = Scenario(id="d1", goal="g", persona="p")
    assert sim._handle_done(sc, "The new address is 5 Oak Street, Leeds.DONE") == "The new address is 5 Oak Street, Leeds."
    assert sc.id in sim._finishing
    assert sim._handle_done(Scenario(id="d2", goal="g", persona="p"), "DONE") is None
    assert sim._handle_done(Scenario(id="d3", goal="g", persona="p"), "Where is my order?") == "Where is my order?"


async def test_finishing_scenario_stops_without_calling_the_model():
    sim = UserSimulator(model="this-model-does-not-exist")
    sc = Scenario(id="d4", goal="g", persona="p")
    sim._finishing.add(sc.id)
    assert await sim.next_message(sc, [("customer", "bye"), ("agent", "ok")]) is None
    assert sc.id not in sim._finishing
