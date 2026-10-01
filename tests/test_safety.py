"""Retries must never repeat side effects, and failed runs must still be recorded."""

import json

import litellm
import pytest
from google.adk import Agent, Runner
from google.adk.apps import App
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.genai import types

from skillminer.capture import TraceRecorderPlugin
from skillminer.llm import RetryingLiteLlm

from fakes import ScriptedLlm, call


class CrashAfterToolLlm(ScriptedLlm):
    """Calls a tool, then the next model call fails (like a provider error mid-turn)."""

    async def generate_content_async(self, llm_request, stream=False):
        if self.calls >= len(self.script):
            raise RuntimeError("provider exploded")
        async for r in super().generate_content_async(llm_request, stream):
            yield r


async def test_failed_run_is_still_recorded_with_its_tool_calls(tmp_path):
    refunds = []

    def issue_refund(order_id: str) -> dict:
        """Refund."""
        refunds.append(order_id)
        return {"status": "ok"}

    agent = Agent(name="a", model=CrashAfterToolLlm(model="s", script=[call("issue_refund", order_id="A1")]),
                  instruction="t", tools=[issue_refund])
    app = App(name="a", root_agent=agent, plugins=[TraceRecorderPlugin(trace_dir=tmp_path)])
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name="a", user_id="u")
    with pytest.raises(RuntimeError):
        async for _ in Runner(app=app, session_service=sessions).run_async(
                user_id="u", session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text="refund A1")])):
            pass

    [line] = next(tmp_path.rglob("*.jsonl")).read_text().splitlines()
    trace = json.loads(line)
    assert refunds == ["A1"]
    assert [s["tool"] for s in trace["steps"]] == ["issue_refund"]
    assert trace["outcome"] == "error" and "provider exploded" in trace["error"]


def _rate_limit(msg="429 slow down"):
    return litellm.RateLimitError(msg, llm_provider="groq", model="m")


async def _collect(model):
    return [r async for r in model.generate_content_async(llm_request=None)]


async def test_model_call_is_retried_on_per_minute_limits(monkeypatch):
    attempts = []

    async def flaky(self, llm_request, stream=False):
        attempts.append(1)
        if len(attempts) < 3:
            raise _rate_limit()
        yield "response"

    monkeypatch.setattr(LiteLlm, "generate_content_async", flaky)
    model = RetryingLiteLlm(model="groq/x", base_delay=0)
    assert await _collect(model) == ["response"]
    assert len(attempts) == 3


async def test_daily_limits_and_partial_responses_are_not_retried(monkeypatch):
    async def daily(self, llm_request, stream=False):
        raise _rate_limit("Rate limit reached on tokens per day (TPD)")
        yield  # pragma: no cover

    monkeypatch.setattr(LiteLlm, "generate_content_async", daily)
    with pytest.raises(litellm.RateLimitError):
        await _collect(RetryingLiteLlm(model="groq/x", base_delay=0))

    attempts = []

    async def fails_mid_stream(self, llm_request, stream=False):
        attempts.append(1)
        yield "partial"
        raise _rate_limit()

    monkeypatch.setattr(LiteLlm, "generate_content_async", fails_mid_stream)
    with pytest.raises(litellm.RateLimitError):
        await _collect(RetryingLiteLlm(model="groq/x", base_delay=0))
    assert len(attempts) == 1  # something was already returned, so retrying could duplicate it
