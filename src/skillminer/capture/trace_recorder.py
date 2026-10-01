"""ADK plugin that records every agent run as one JSON line.

One trace = one invocation = one user message and everything the agent did to answer it:
the tools it called (in order), their arguments, whether each succeeded, and the final reply.
Traces are appended to ``<trace_dir>/<app_name>/<YYYY-MM-DD>.jsonl``.

Attach it to any ADK app without touching the agent's own code:

    app = App(name="my_agent", root_agent=root_agent,
              plugins=[TraceRecorderPlugin(trace_dir="traces")])
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.adk.plugins.base_plugin import BasePlugin

SCHEMA_VERSION = 1
META_STATE_KEY = "skillminer_meta"  # session.state key whose dict is copied into trace["meta"]
MAX_RESULT_CHARS = 2000  # keep trace files small; full tool results are rarely needed for mining


@dataclass
class _RunState:
    """Everything collected for one invocation while it is running."""

    started_at: str
    started_monotonic: float
    user_message: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    tool_started: dict[str, float] = field(default_factory=dict)
    final_response: str = ""


def _text_of(content: Any) -> str:
    """Join the text parts of a google.genai Content object."""
    parts = getattr(content, "parts", None) or []
    return "".join(p.text for p in parts if getattr(p, "text", None)).strip()


def _truncate(value: Any) -> Any:
    """Return the value unchanged if it is small, else a truncated JSON string."""
    text = json.dumps(value, default=str)
    if len(text) <= MAX_RESULT_CHARS:
        return value
    return text[:MAX_RESULT_CHARS] + "...[truncated]"


def _step_status(result: Any) -> str:
    """Our tools signal failure with {"status": "error"}; anything else counts as ok."""
    if isinstance(result, dict) and result.get("status") == "error":
        return "error"
    return "ok"


class TraceRecorderPlugin(BasePlugin):
    """Records each run's tool trajectory to JSONL so SkillMiner can mine it later."""

    def __init__(self, trace_dir: str | Path = "traces", name: str = "trace_recorder"):
        super().__init__(name=name)
        self.trace_dir = Path(trace_dir)
        self._runs: dict[str, _RunState] = {}  # keyed by invocation_id

    # --- lifecycle hooks (called by ADK) ------------------------------------------------

    async def before_run_callback(self, *, invocation_context):
        # ADK fires on_user_message_callback before this hook, so the run may already exist.
        # Reuse it rather than replacing it, or the user message would be lost.
        self._run(invocation_context.invocation_id)
        return None

    async def on_user_message_callback(self, *, invocation_context, user_message):
        run = self._run(invocation_context.invocation_id)
        run.user_message = _text_of(user_message)
        return None

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        run = self._run(tool_context.invocation_id)
        run.tool_started[tool_context.function_call_id or tool.name] = time.monotonic()
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        self._add_step(tool, tool_args, tool_context, status=_step_status(result), result=result)
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        self._add_step(tool, tool_args, tool_context, status="exception",
                       result={"error": f"{type(error).__name__}: {error}"})
        return None  # let ADK handle the error as it normally would

    async def on_event_callback(self, *, invocation_context, event):
        if event.author != "user" and event.is_final_response():
            text = _text_of(event.content)
            if text:
                self._run(invocation_context.invocation_id).final_response = text
        return None

    async def after_run_callback(self, *, invocation_context):
        run = self._runs.pop(invocation_context.invocation_id, None)
        if run is None:
            return
        session = invocation_context.session
        agent = invocation_context.agent
        trace = {
            "schema_version": SCHEMA_VERSION,
            "trace_id": invocation_context.invocation_id,
            "app_name": session.app_name,
            "session_id": session.id,
            "user_id": session.user_id,
            "agent": agent.name,
            "model": str(getattr(getattr(agent, "model", None), "model", getattr(agent, "model", ""))),
            "started_at": run.started_at,
            "duration_ms": round((time.monotonic() - run.started_monotonic) * 1000),
            "user_message": run.user_message,
            "steps": run.steps,
            "final_response": run.final_response,
            "outcome": "completed" if run.final_response else "incomplete",
            # Optional labels set by whoever created the session (e.g. the simulator's ground-truth
            # intent). SkillMiner never mines these; they are only used to score the miner.
            "meta": dict(session.state.get(META_STATE_KEY) or {}),
        }
        self._write(trace)

    # --- helpers ------------------------------------------------------------------------

    def _run(self, invocation_id: str) -> _RunState:
        # Tool hooks can fire for an invocation we did not see start (e.g. plugin added mid-run).
        if invocation_id not in self._runs:
            self._runs[invocation_id] = _RunState(
                started_at=datetime.now(timezone.utc).isoformat(),
                started_monotonic=time.monotonic(),
            )
        return self._runs[invocation_id]

    def _add_step(self, tool, tool_args, tool_context, *, status: str, result: Any) -> None:
        run = self._run(tool_context.invocation_id)
        started = run.tool_started.pop(tool_context.function_call_id or tool.name, None)
        run.steps.append({
            "index": len(run.steps),
            "tool": tool.name,
            "args": dict(tool_args or {}),
            "status": status,
            "result": _truncate(result),
            "latency_ms": round((time.monotonic() - started) * 1000) if started else None,
        })

    def _write(self, trace: dict[str, Any]) -> None:
        day = trace["started_at"][:10]
        path = self.trace_dir / trace["app_name"] / f"{day}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace, default=str) + "\n")
