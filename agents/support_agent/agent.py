import os
from pathlib import Path

from google.adk import Agent
from google.adk.apps import App
from google.adk.models.lite_llm import LiteLlm

from skillminer.capture import TraceRecorderPlugin

from .tools import (
    check_refund_eligibility,
    issue_refund,
    lookup_order,
    send_email,
    track_shipment,
    update_address,
)

# Set SKILLMINER_MODEL in .env. Gemini IDs (e.g. "gemini-2.5-flash") go to Google directly;
# anything with a provider prefix (e.g. "openrouter/...") goes through LiteLLM.
MODEL_ID = os.getenv("SKILLMINER_MODEL", "groq/openai/gpt-oss-120b")

# Groq's gpt-oss models return their reasoning, ADK sends it back on the next turn, and Groq
# rejects it ("reasoning_content is unsupported"). Asking Groq not to return it avoids that.
extra = {"include_reasoning": False} if MODEL_ID.startswith("groq/openai/gpt-oss") else {}
model = MODEL_ID if MODEL_ID.startswith("gemini") else LiteLlm(model=MODEL_ID, **extra)

root_agent = Agent(
    name="support_agent",
    model=model,
    description="Customer support agent for an online shop.",
    instruction=(
        "You are a customer support agent for an online shop. "
        "Use the tools to look up orders, check refund eligibility, issue refunds, "
        "track shipments and update delivery addresses. "
        "Always look up the order before acting on it. "
        "After any refund or address change, send the customer a confirmation email. "
        "Be concise and polite."
    ),
    tools=[
        lookup_order,
        check_refund_eligibility,
        issue_refund,
        track_shipment,
        update_address,
        send_email,
    ],
)

# ADK picks up `app` before `root_agent`. Its name must match this folder's name.
# Traces go to <project root>/traces/, whichever folder adk web is started from.
TRACE_DIR = Path(__file__).resolve().parents[2] / "traces"

app = App(
    name="support_agent",
    root_agent=root_agent,
    plugins=[TraceRecorderPlugin(trace_dir=TRACE_DIR)],
)
