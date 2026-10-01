"""Skill Writer: turn one mined intent into an Agent Skill (SKILL.md) that ADK can load.

For each intent we show the LLM every distinct way the agent handled it (how often, plus one
example with the real tool results), with personal data already redacted. The LLM writes a
procedure that copies the variants that ended correctly and states MUST/NEVER rules wherever a
variant shows the agent going wrong. The result is scanned again by the Privacy Guard before it
is saved, and saved as a "candidate": only the Validator can approve it.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import litellm
import yaml

from .guard import GuardReport, collect_case_values, guard

PROMPT = """You are writing an Agent Skill (a SKILL.md file) for a customer support agent. The skill is
learned from how the agent actually handled real conversations with this intent.

Intent: {intent} - {intent_description}
Seen in {support} conversations. Below are the distinct ways the agent handled it, how often each
happened, and one example of each with the real tool results. Personal data has been replaced
with placeholders like <ORDER_ID> or <EMAIL>.

Available tools:
{tools}

Observed variants:
{variants}

Write the skill so that an agent following it handles this intent correctly every time:
- Build the procedure from the variants whose tool results show they ended correctly. Where the
  agent was inconsistent, pick the behaviour the tool results show to be correct.
- Cover every branch the evidence shows (for example when a check fails or an order is not
  found) and say exactly what to do in each.
- In "Rules", list hard MUST / NEVER rules supported by the evidence, especially where a variant
  shows the agent breaking one (for example acting although a check said no). Name the variant.
- Use only the tools listed. Refer to values with placeholders, never real data.
- Do not invent facts, timelines, policies or options that the tool results and replies do not
  show (no "5-7 business days", no "goodwill gestures", no escalation paths). If something is
  unknown, tell the agent to say so rather than guess.
- At most 350 words.

Reply in exactly this format and nothing else:
DESCRIPTION: <one sentence: when should an agent use this skill>
---
## When to use
...
## Procedure
...
## Rules
...
## Replying to the customer
..."""


@dataclass
class WrittenSkill:
    name: str
    path: Path | None
    support: int
    guard: GuardReport
    blocked: bool = False
    notes: list[str] = field(default_factory=list)


def describe_tools(tools: list[Callable]) -> dict[str, dict[str, Any]]:
    """name -> {params, doc} for plain-function tools (what the agent can actually call)."""
    out = {}
    for fn in tools:
        if callable(fn) and hasattr(fn, "__name__"):
            out[fn.__name__] = {
                "params": list(inspect.signature(fn).parameters),
                "doc": (inspect.getdoc(fn) or "").split("\n")[0],
            }
    return out


def skill_name(intent: str) -> str:
    """ADK skill names are lowercase kebab-case and must match the folder name."""
    return re.sub(r"[^a-z0-9]+", "-", intent.lower()).strip("-")[:64] or "skill"


def _fmt_call(step: dict, case_values: dict[str, str]) -> str:
    args = ", ".join(f"{k}={v}" for k, v in (step.get("args") or {}).items())
    result = json.dumps(step.get("result"), default=str)
    if len(result) > 220:
        result = result[:220] + "..."
    status = "ok" if step.get("ok", True) else "FAILED"
    text = f"{step['tool']}({args}) -> {status}: {result}"
    return guard(text, case_values)[0]


def build_evidence(cluster: dict, case_values: dict[str, str], max_variants: int = 6) -> str:
    """One block per tool-path variant: count, the customer's request, the calls, the final reply."""
    blocks = []
    for i, variant in enumerate(cluster["paths"][:max_variants], start=1):
        path = variant["path"]
        example = next(e for e in cluster["episodes"] if e["path"] == path)
        customer = " / ".join(m[:160] for m in example["user_messages"][:2])
        calls = "\n".join(f"    {n}. {_fmt_call(s, case_values)}" for n, s in enumerate(example["steps"], 1))
        reply = (example["agent_messages"] or [""])[-1][:220]
        block = (f"Variant {i}: seen {variant['count']} times: {' -> '.join(path) or '(no tools)'}\n"
                 f"  Customer: {customer}\n  Calls:\n{calls or '    (none)'}\n  Agent's final reply: {reply}")
        blocks.append(guard(block, case_values)[0])
    hidden = len(cluster["paths"]) - max_variants
    if hidden > 0:
        blocks.append(f"(+{hidden} rarer variants not shown)")
    return "\n\n".join(blocks)


def _parse_reply(text: str) -> tuple[str, str]:
    match = re.search(r"DESCRIPTION:\s*(.+?)\n-{3,}\s*\n(.*)", text, re.DOTALL)
    if not match:
        raise ValueError("Writer reply did not follow the DESCRIPTION / --- / body format")
    return match.group(1).strip(), match.group(2).strip()


async def write_skill(cluster: dict, *, model: str, tools: dict[str, dict], skills_dir: Path,
                      case_values: dict[str, str], intent_description: str = "",
                      completion_kwargs: dict | None = None,
                      blocked_dir: Path = Path("runs/blocked_skills")) -> WrittenSkill:
    intent = cluster["keywords"][0]
    name = skill_name(intent)
    tool_text = "\n".join(f"- {n}({', '.join(t['params'])}): {t['doc']}" for n, t in tools.items())
    prompt = PROMPT.format(intent=intent, intent_description=intent_description or intent,
                           support=cluster["size"], tools=tool_text or "(not provided)",
                           variants=build_evidence(cluster, case_values))

    response = await litellm.acompletion(model=model, messages=[{"role": "user", "content": prompt}],
                                         **(completion_kwargs or {}))
    description, body = _parse_reply(response.choices[0].message.content or "")

    # Second pass of the Privacy Guard on what the LLM wrote.
    body, report = guard(body, case_values)
    description, desc_report = guard(description, case_values)
    report.pattern_findings += desc_report.pattern_findings
    report.redacted_values += desc_report.redacted_values

    used_tools = sorted(t for t in tools if t in body)
    frontmatter = {
        "name": name,
        "description": description[:1024],
        "metadata": {"source": "skillminer", "status": "candidate", "support": cluster["size"]},
    }
    provenance = {
        "intent": intent,
        "intent_description": intent_description,
        "support": cluster["size"],
        "variants": cluster["paths"],
        "tools": {t: tools[t]["params"] for t in used_tools},  # for the Drift Watcher
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "candidate",
        "guard": {"redacted_values": report.redacted_values,
                  "pattern_findings": [f.__dict__ for f in report.pattern_findings]},
    }

    if not report.clean:
        # Never publish a skill that still looked personal after redaction; keep it for review.
        out = Path(blocked_dir) / name
        blocked = True
    else:
        out = skills_dir / name
        blocked = False
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "SKILL.md").write_text(f"---\n{yaml.safe_dump(frontmatter, sort_keys=False)}---\n\n{body}\n")
    (out / "skillminer.json").write_text(json.dumps(provenance, indent=2))
    return WrittenSkill(name=name, path=out, support=cluster["size"], guard=report, blocked=blocked)


def load_provenance(skill_dir: Path) -> dict:
    return json.loads((Path(skill_dir) / "skillminer.json").read_text())


def set_status(skill_dir: Path, status: str, extra: dict | None = None) -> None:
    """Update a skill's status in both SKILL.md frontmatter and skillminer.json."""
    skill_dir = Path(skill_dir)
    text = (skill_dir / "SKILL.md").read_text()
    _, fm, body = text.split("---", 2)
    data = yaml.safe_load(fm)
    data.setdefault("metadata", {})["status"] = status
    (skill_dir / "SKILL.md").write_text(f"---\n{yaml.safe_dump(data, sort_keys=False)}---{body}")
    prov = load_provenance(skill_dir)
    prov["status"] = status
    prov.update(extra or {})
    (skill_dir / "skillminer.json").write_text(json.dumps(prov, indent=2))


def all_case_values(mining: dict) -> dict[str, str]:
    steps = [s for c in mining["clusters"] for e in c["episodes"] for s in e["steps"]]
    return collect_case_values(steps)
