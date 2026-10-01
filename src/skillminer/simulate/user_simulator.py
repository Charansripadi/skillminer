"""An LLM that role-plays a customer, so we can generate realistic sessions without real users."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import litellm

DONE = "DONE"


@dataclass
class Scenario:
    """One simulated conversation.

    ``goal`` and ``persona`` are shown to the simulated customer. ``label`` is the hidden ground
    truth (e.g. which workflow the agent should follow); it is stored in the trace for scoring
    the miner later and is never shown to the agent or the miner.
    """

    id: str
    goal: str
    persona: str
    label: dict[str, Any] = field(default_factory=dict)
    max_turns: int = 4


SYSTEM_PROMPT = """You are role-playing a customer who is contacting an online shop's support chat.

Your goal: {goal}
Your personality: {persona}

Rules:
- Write ONLY the customer's next chat message: 1 to 3 short sentences, no quotes, no labels.
- Stay in character. Never mention that you are an AI or that this is a simulation.
- Only share details (like your order ID) that your goal gives you. Do not invent new ones.
- When your goal has been achieved, or the agent has clearly refused or cannot help, reply with exactly: {done}
"""


class UserSimulator:
    """Generates the customer's side of a conversation one message at a time."""

    def __init__(self, model: str, **completion_kwargs: Any):
        self.model = model
        self.completion_kwargs = completion_kwargs
        self._finishing: set[str] = set()  # scenarios whose last message carried a trailing DONE

    async def next_message(self, scenario: Scenario, transcript: list[tuple[str, str]]) -> str | None:
        """Return the customer's next message, or None when the conversation should end.

        ``transcript`` is a list of ("customer" | "agent", text) pairs so far.
        """
        if scenario.id in self._finishing:  # previous message ended with DONE: stop without calling the model
            self._finishing.discard(scenario.id)
            return None
        user_turns = sum(1 for who, _ in transcript if who == "customer")
        if user_turns >= scenario.max_turns:
            return None

        # From the simulator's point of view the support agent is the "user" it replies to.
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(
            goal=scenario.goal, persona=scenario.persona, done=DONE)}]
        # Always open with a user turn: some providers reject a conversation that starts with "assistant".
        messages.append({"role": "user", "content": "(The support chat has opened. Write your first message.)"})
        for who, text in transcript:
            messages.append({"role": "assistant" if who == "customer" else "user", "content": text})

        response = await litellm.acompletion(model=self.model, messages=messages, **self.completion_kwargs)
        text = (response.choices[0].message.content or "").strip().strip('"')
        return self._handle_done(scenario, text)

    def _handle_done(self, scenario: Scenario, text: str) -> str | None:
        """Interpret the DONE signal, including when the model glues it onto a final message.

        "DONE"                       -> end now
        "Thanks, that's all. DONE"   -> send "Thanks, that's all." and end after the agent replies
        """
        if text.upper().endswith(DONE):
            text = text[: -len(DONE)].strip()
            if text:
                self._finishing.add(scenario.id)
        if not text or text.upper().startswith(DONE):
            return None
        return text
