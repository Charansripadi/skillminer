"""Drive an ADK app with simulated customers. Traces are written by the app's TraceRecorderPlugin."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

from google.adk import Runner
from google.adk.apps import App
from google.adk.sessions import InMemorySessionService
from google.genai import types

from ..capture.trace_recorder import META_STATE_KEY
from .user_simulator import Scenario, UserSimulator

log = logging.getLogger(__name__)


@dataclass
class SessionResult:
    scenario_id: str
    turns: int
    ok: bool
    error: str | None = None


def _is_rate_limit(error: Exception) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return "ratelimit" in text or "rate limit" in text or "429" in text


async def _with_backoff(make_call, *, retries: int, base_delay: float):
    """Await make_call(), retrying with growing delays when the provider rate-limits us."""
    for attempt in range(retries + 1):
        try:
            return await make_call()
        except Exception as e:  # providers raise different exception types for 429s
            if not _is_rate_limit(e) or attempt == retries:
                raise
            delay = base_delay * (attempt + 1)
            log.warning("Rate limited, waiting %.0fs (attempt %d/%d)", delay, attempt + 1, retries)
            await asyncio.sleep(delay)


async def run_scenarios(
    app: App,
    scenarios: list[Scenario],
    simulator: UserSimulator,
    *,
    reset_world: Callable[[], None] | None = None,
    retries: int = 4,
    base_delay: float = 20.0,
    on_result: Callable[[SessionResult], None] | None = None,
) -> list[SessionResult]:
    """Run every scenario as a fresh session. One failed scenario does not stop the rest."""
    sessions = InMemorySessionService()
    runner = Runner(app=app, session_service=sessions)
    results = []

    for scenario in scenarios:
        if reset_world:
            reset_world()
        session = await sessions.create_session(
            app_name=app.name, user_id=f"sim-{scenario.id}",
            state={META_STATE_KEY: {"scenario_id": scenario.id, "persona": scenario.persona, **scenario.label}},
        )
        transcript: list[tuple[str, str]] = []
        try:
            while True:
                message = await _with_backoff(lambda: simulator.next_message(scenario, transcript),
                                              retries=retries, base_delay=base_delay)
                if message is None:
                    break
                transcript.append(("customer", message))
                reply = await _with_backoff(lambda: _agent_turn(runner, session, message),
                                            retries=retries, base_delay=base_delay)
                transcript.append(("agent", reply))
            result = SessionResult(scenario.id, turns=len(transcript) // 2, ok=True)
        except Exception as e:
            log.error("Scenario %s failed: %s", scenario.id, e)
            result = SessionResult(scenario.id, turns=len(transcript) // 2, ok=False, error=str(e)[:200])

        results.append(result)
        if on_result:
            on_result(result)
    return results


async def _agent_turn(runner: Runner, session, message: str) -> str:
    """Send one customer message to the agent and return its final text reply."""
    reply = ""
    async for event in runner.run_async(
        user_id=session.user_id, session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=message)]),
    ):
        if event.author != "user" and event.is_final_response() and event.content:
            reply = "".join(p.text for p in event.content.parts if getattr(p, "text", None)).strip()
    return reply
