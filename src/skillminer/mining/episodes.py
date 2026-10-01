"""Turn raw per-message traces into episodes: one episode = one whole conversation.

A workflow often spans several messages (the customer gives the order ID only when asked),
so mining per message would cut workflows into pieces. We join every turn of a session and
then simplify the tool path so that noise does not hide the pattern.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..capture.show import iter_traces


@dataclass
class Step:
    tool: str
    ok: bool
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None

    @property
    def token(self) -> str:
        """'issue_refund' for a success, 'issue_refund!' for a failure."""
        return self.tool if self.ok else self.tool + "!"


@dataclass
class Episode:
    session_id: str
    user_messages: list[str]
    agent_messages: list[str]
    steps: list[Step]
    meta: dict[str, Any] = field(default_factory=dict)  # ground-truth labels, if any; never used for mining

    @property
    def path(self) -> list[str]:
        """Simplified tool path: consecutive repeats of the same call collapsed into one."""
        return collapse_repeats([s.token for s in self.steps])

    @property
    def request_text(self) -> str:
        """What the customer asked for, across the whole conversation."""
        return " ".join(self.user_messages)

    @property
    def has_actions(self) -> bool:
        return bool(self.steps)


def collapse_repeats(tokens: list[str]) -> list[str]:
    """['lookup_order!', 'lookup_order!', 'track_shipment'] -> ['lookup_order!', 'track_shipment']."""
    out: list[str] = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def load_episodes(path: str | Path) -> list[Episode]:
    """Group every trace under ``path`` by session, in time order, into episodes."""
    by_session: dict[str, list[dict]] = defaultdict(list)
    for trace in iter_traces(path):
        by_session[trace["session_id"]].append(trace)

    episodes = []
    for session_id, turns in by_session.items():
        turns.sort(key=lambda t: t["started_at"])
        episodes.append(Episode(
            session_id=session_id,
            user_messages=[t["user_message"] for t in turns if t["user_message"]],
            agent_messages=[t["final_response"] for t in turns if t["final_response"]],
            steps=[Step(s["tool"], s["status"] == "ok", s.get("args") or {}, s.get("result"))
                   for t in turns for s in t["steps"]],
            meta=next((t["meta"] for t in turns if t.get("meta")), {}),
        ))
    episodes.sort(key=lambda e: e.session_id)
    return episodes
