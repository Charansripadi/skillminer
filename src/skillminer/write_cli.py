"""Write a candidate SKILL.md for every mined intent with enough support.

    skillminer-write --tools support_agent.agent:TOOLS --import-path agents --env-file agents/support_agent/.env
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
from pathlib import Path

import litellm
from dotenv import load_dotenv

from .llm import provider_kwargs
from .simulate.session_runner import _with_backoff
from .writer import all_case_values, describe_tools, write_skill


def _load_tools(spec: str | None, import_path: str | None) -> dict:
    """'package.module:ATTR' -> tool descriptions, so the writer only uses tools that exist."""
    if not spec:
        return {}
    if import_path:
        sys.path.insert(0, str(Path(import_path).resolve()))
    module_name, attr = spec.split(":")
    return describe_tools(getattr(importlib.import_module(module_name), attr))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write candidate skills from mined intents.")
    parser.add_argument("--mining", default="runs/mining.json", help="output of skillminer-mine")
    parser.add_argument("--skills-dir", default="skills")
    parser.add_argument("--min-support", type=int, default=5, help="skip intents seen fewer times")
    parser.add_argument("--tools", default=None, help="module:ATTR holding the agent's tool functions")
    parser.add_argument("--import-path", default=None, help="folder to add to sys.path for --tools")
    parser.add_argument("--model", default=None, help="writer LLM (default: $SKILLMINER_WRITER_MODEL or $SKILLMINER_MODEL)")
    parser.add_argument("--env-file", default=None)
    args = parser.parse_args(argv)

    load_dotenv(args.env_file) if args.env_file else load_dotenv()
    litellm.suppress_debug_info = True
    for noisy in ("LiteLLM", "litellm", "httpx"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    model = args.model or os.getenv("SKILLMINER_WRITER_MODEL") or os.getenv("SKILLMINER_MODEL", "groq/openai/gpt-oss-120b")
    kwargs = {"temperature": 0.2, **provider_kwargs(model)}

    mining = json.loads(Path(args.mining).read_text())
    tools = _load_tools(args.tools, args.import_path)
    case_values = all_case_values(mining)
    skills_dir = Path(args.skills_dir)
    skills_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writer model: {model} | tools known: {len(tools)} | case values to redact: {len(case_values)}")

    async def run():
        for cluster in mining["clusters"]:
            intent = cluster["keywords"][0]
            if cluster["size"] < args.min_support:
                print(f"- skip {intent}: only {cluster['size']} conversations (< {args.min_support})")
                continue
            description = cluster["keywords"][1] if len(cluster["keywords"]) > 1 else ""
            skill = await _with_backoff(lambda c=cluster, d=description: write_skill(
                c, model=model, tools=tools, skills_dir=skills_dir, case_values=case_values,
                intent_description=d, completion_kwargs=kwargs), retries=4, base_delay=20)
            state = "BLOCKED by Privacy Guard" if skill.blocked else "candidate"
            print(f"- {skill.name}: {state} | support {skill.support} | "
                  f"{skill.guard.redacted_values} values redacted, {len(skill.guard.pattern_findings)} pattern findings "
                  f"-> {skill.path}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
