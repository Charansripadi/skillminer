import os
from pathlib import Path

from google.adk import Agent
from google.adk.apps import App

from skillminer.capture import TraceRecorderPlugin
from skillminer.llm import adk_model

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

# adk_model() handles provider quirks (e.g. Groq reasoning output, which ADK would send back and
# Groq rejects) and retries a rate-limited model call safely, before any tool runs twice.
model = adk_model(MODEL_ID)

TOOLS = [lookup_order, check_refund_eligibility, issue_refund, track_shipment, update_address, send_email]

INSTRUCTION = (
    "You are a customer support agent for an online shop. "
    "Use the tools to look up orders, check refund eligibility, issue refunds, "
    "track shipments and update delivery addresses. "
    "Always look up the order before acting on it. "
    "After any refund or address change, send the customer a confirmation email. "
    "Be concise and polite."
)

# Traces go to <project root>/traces/, whichever folder adk web is started from.
TRACE_DIR = Path(__file__).resolve().parents[2] / "traces"


def build_agent(skills_dir: str | Path | None = None) -> Agent:
    """The support agent, optionally with SkillMiner skills available through ADK's SkillToolset."""
    tools = list(TOOLS)
    if skills_dir:
        from google.adk.skills import load_skills_from_dir
        from google.adk.tools.skill_toolset import SkillToolset

        skills = load_skills_from_dir(skills_dir)
        if skills:
            tools.append(SkillToolset(skills=skills))
    return Agent(
        name="support_agent",
        model=model,
        description="Customer support agent for an online shop.",
        instruction=INSTRUCTION,
        tools=tools,
    )


def build_app(skills_dir: str | Path | None = None, trace_dir: str | Path = TRACE_DIR) -> App:
    # The app name must match this folder's name for adk web.
    return App(
        name="support_agent",
        root_agent=build_agent(skills_dir),
        plugins=[TraceRecorderPlugin(trace_dir=trace_dir)],
    )


# ADK picks up `app` before `root_agent`. Set SKILLMINER_SKILLS_DIR to give the agent skills.
app = build_app(skills_dir=os.getenv("SKILLMINER_SKILLS_DIR") or None)
root_agent = app.root_agent
