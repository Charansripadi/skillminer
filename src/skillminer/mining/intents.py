"""LLM intent labeller: decide what each conversation's customer wanted.

Word-count clustering groups conversations by surface words (product names, street names).
An LLM reads the request for its meaning instead. Intents are induced online: for each
conversation the model either picks one of the intents it has already created or creates a
new one, so the set of intents grows only when something genuinely new appears and nobody has
to fix the number of intents in advance.

Labels are cached per session in a JSON file, so re-running costs no API calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import litellm

from .episodes import Episode

PROMPT = """You label customer support conversations by the customer's goal (their intent).

Known intents so far:
{known}

Customer messages from one conversation:
{messages}

Pick the known intent that matches the customer's main goal. Only if none fits, create a new
intent: a short snake_case name for the general goal (not for a specific product, order or
outcome), and a one-sentence description. Whether the request succeeded does not matter.

Reply with JSON only: {{"intent": "<name>", "description": "<one sentence, only for a new intent>"}}"""


@dataclass
class Intent:
    name: str
    description: str


class IntentLabeler:
    def __init__(self, model: str, cache_path: str | Path | None = None, **completion_kwargs):
        self.model = model
        self.completion_kwargs = {"temperature": 0, **completion_kwargs}
        self.cache_path = Path(cache_path) if cache_path else None
        self.intents: dict[str, Intent] = {}
        self.labels: dict[str, str] = {}  # session_id -> intent name
        self._load_cache()

    async def label_all(self, episodes: list[Episode], on_label=None) -> dict[str, str]:
        """Label every episode (cached ones are skipped). Returns session_id -> intent name."""
        for episode in episodes:
            if episode.session_id not in self.labels:
                self.labels[episode.session_id] = await self._label(episode)
                self._save_cache()
            if on_label:
                on_label(episode, self.labels[episode.session_id])
        return {e.session_id: self.labels[e.session_id] for e in episodes}

    async def _label(self, episode: Episode) -> str:
        known = "\n".join(f"- {i.name}: {i.description}" for i in self.intents.values()) or "(none yet)"
        messages = "\n".join(f"- {m}" for m in episode.user_messages[:4]) or "(no messages)"
        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": PROMPT.format(known=known, messages=messages)}],
            **self.completion_kwargs,
        )
        parsed = _parse_json(response.choices[0].message.content or "")
        name = _snake(parsed.get("intent", "")) or "unknown"
        if name not in self.intents:
            self.intents[name] = Intent(name, parsed.get("description", "").strip())
        return name

    def _load_cache(self) -> None:
        if self.cache_path and self.cache_path.exists():
            data = json.loads(self.cache_path.read_text())
            self.intents = {n: Intent(n, d) for n, d in data.get("intents", {}).items()}
            self.labels = data.get("labels", {})

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps({
                "model": self.model,
                "intents": {n: i.description for n, i in self.intents.items()},
                "labels": self.labels,
            }, indent=2))


def _parse_json(text: str) -> dict:
    """Parse the model's JSON, tolerating code fences or text around it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
