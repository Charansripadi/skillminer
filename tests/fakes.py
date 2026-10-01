"""Test doubles: a scripted model so ADK agents can run offline and deterministically."""

from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types


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
