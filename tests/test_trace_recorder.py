"""Run a real ADK Runner with a scripted fake model and check the recorded trace.

No API calls: the fake model replays a fixed refund conversation, so the test is free,
offline and deterministic.
"""

import json
from typing import AsyncGenerator

from google.adk import Agent, Runner
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types

from skillminer.capture import TraceRecorderPlugin
from skillminer.capture.show import tool_path


def call(name: str, **args) -> types.Content:
    return types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))])


def say(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


class ScriptedLlm(BaseLlm):
    """Returns the next scripted response on every model call."""

    script: list[types.Content]
    calls: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        content = self.script[self.calls]
        self.calls += 1
        yield LlmResponse(content=content)


def lookup_order(order_id: str) -> dict:
    """Look up an order."""
    return {"status": "ok", "order_id": order_id, "customer": "priya@example.com"}


def check_refund_eligibility(order_id: str) -> dict:
    """Check refund eligibility."""
    return {"status": "ok", "eligible": True}


def issue_refund(order_id: str, amount: float, reason: str) -> dict:
    """Issue a refund."""
    return {"status": "error", "message": "payment gateway down"}


async def run_once(tmp_path, script, message):
    agent = Agent(
        name="test_agent",
        model=ScriptedLlm(model="scripted", script=script),
        instruction="test",
        tools=[lookup_order, check_refund_eligibility, issue_refund],
    )
    app = App(name="test_app", root_agent=agent, plugins=[TraceRecorderPlugin(trace_dir=tmp_path)])
    sessions = InMemorySessionService()
    runner = Runner(app=app, session_service=sessions)
    session = await sessions.create_session(app_name="test_app", user_id="u1")
    async for _ in runner.run_async(user_id="u1", session_id=session.id,
                                    new_message=types.Content(role="user", parts=[types.Part(text=message)])):
        pass
    files = list(tmp_path.rglob("*.jsonl"))
    assert len(files) == 1, files
    return [json.loads(line) for line in files[0].read_text().splitlines()]


async def test_records_tool_path_args_errors_and_final_reply(tmp_path):
    script = [
        call("lookup_order", order_id="A1001"),
        call("check_refund_eligibility", order_id="A1001"),
        call("issue_refund", order_id="A1001", amount=59.99, reason="broken"),
        say("Sorry, the refund failed. Please try again later."),
    ]
    [trace] = await run_once(tmp_path, script, "Refund A1001 please")

    assert trace["app_name"] == "test_app"
    assert trace["agent"] == "test_agent"
    assert trace["user_message"] == "Refund A1001 please"
    assert [s["tool"] for s in trace["steps"]] == ["lookup_order", "check_refund_eligibility", "issue_refund"]
    assert trace["steps"][0]["args"] == {"order_id": "A1001"}
    assert [s["status"] for s in trace["steps"]] == ["ok", "ok", "error"]
    assert trace["final_response"].startswith("Sorry, the refund failed")
    assert trace["outcome"] == "completed"
    assert tool_path(trace) == "lookup_order -> check_refund_eligibility -> issue_refund!"


async def test_run_without_tools_is_recorded(tmp_path):
    [trace] = await run_once(tmp_path, [say("Hello! How can I help?")], "hi")
    assert trace["steps"] == []
    assert trace["final_response"] == "Hello! How can I help?"
    assert tool_path(trace) == "(no tools)"
