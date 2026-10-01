"""Provider quirks and safe retries, in one place."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from google.adk.models.lite_llm import LiteLlm

log = logging.getLogger(__name__)


def provider_kwargs(model: str) -> dict:
    """Extra completion arguments a model needs.

    Groq's reasoning models return their reasoning by default. ADK then sends it back on the next
    turn, which Groq rejects, and it also burns through the free tier's daily token limit.
    """
    if model.startswith("groq/openai/gpt-oss"):
        return {"include_reasoning": False}
    if model.startswith("groq/qwen/"):
        return {"reasoning_effort": "none"}
    return {}


def _rate_limited(error: Exception) -> tuple[bool, bool]:
    """(is a rate limit, is a daily limit)"""
    text = f"{type(error).__name__} {error}".lower()
    limited = "ratelimit" in text or "rate limit" in text or "429" in text
    daily = "per day" in text or "(tpd)" in text or "(rpd)" in text
    return limited, daily


class RetryingLiteLlm(LiteLlm):
    """LiteLlm that waits and retries a single model call when the provider rate-limits it.

    Retrying here, before any response reaches the agent, is safe. Retrying a whole agent turn
    is not: tools that already ran (a refund, an email) would run a second time.
    """

    max_retries: int = 5
    base_delay: float = 15.0

    async def generate_content_async(self, llm_request, stream: bool = False) -> AsyncGenerator:
        for attempt in range(self.max_retries + 1):
            yielded = False
            try:
                async for response in super().generate_content_async(llm_request, stream):
                    yielded = True
                    yield response
                return
            except Exception as e:
                limited, daily = _rate_limited(e)
                if yielded or not limited or daily or attempt == self.max_retries:
                    raise
                delay = self.base_delay * (attempt + 1)
                log.warning("Model rate limited, retrying this call in %.0fs (%d/%d)", delay, attempt + 1, self.max_retries)
                await asyncio.sleep(delay)


def adk_model(model_id: str):
    """What to pass as an ADK Agent's model: Gemini IDs natively, everything else via LiteLLM."""
    if model_id.startswith("gemini"):
        return model_id
    return RetryingLiteLlm(model=model_id, **provider_kwargs(model_id))
