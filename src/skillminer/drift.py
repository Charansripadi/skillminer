"""Drift Watcher: retire skills whose tools changed since the skill was written.

A skill says "call issue_refund(order_id, amount, reason)". If that tool is renamed, removed or
gets different parameters, following the skill now produces failed calls. When a skill is written
we store the exact tool signatures it relies on (skillminer.json); this module compares them with
the agent's current tools and marks drifted skills "stale" so they stop being used.

    skillminer-drift --tools support_agent.agent:TOOLS --import-path agents [--apply]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from .writer import describe_tools, load_provenance, set_status


@dataclass
class DriftResult:
    skill: str
    status: str  # "ok" or "stale"
    problems: list[str] = field(default_factory=list)


def check_skill(skill_dir: Path, current: dict[str, list[str]]) -> DriftResult:
    expected = load_provenance(skill_dir).get("tools", {})
    problems = []
    for tool, params in expected.items():
        if tool not in current:
            problems.append(f"tool '{tool}' no longer exists")
        elif list(current[tool]) != list(params):
            problems.append(f"tool '{tool}' parameters changed: {params} -> {current[tool]}")
    return DriftResult(skill_dir.name, "stale" if problems else "ok", problems)


def check_all(skills_dir: str | Path, current: dict[str, list[str]]) -> list[DriftResult]:
    return [check_skill(d, current) for d in sorted(Path(skills_dir).iterdir())
            if (d / "SKILL.md").exists() and (d / "skillminer.json").exists()]


def main(argv: list[str] | None = None) -> None:
    from .write_cli import _load_tools

    parser = argparse.ArgumentParser(description="Detect skills whose tools have changed.")
    parser.add_argument("--skills-dir", default="skills")
    parser.add_argument("--tools", required=True, help="module:ATTR holding the agent's current tool functions")
    parser.add_argument("--import-path", default=None)
    parser.add_argument("--apply", action="store_true", help="mark drifted skills as 'stale'")
    args = parser.parse_args(argv)

    current = {name: info["params"] for name, info in _load_tools(args.tools, args.import_path).items()}
    results = check_all(args.skills_dir, current)
    for r in results:
        print(f"{r.status.upper():6s} {r.skill}" + "".join(f"\n       - {p}" for p in r.problems))
        if args.apply and r.status == "stale":
            set_status(Path(args.skills_dir) / r.skill, "stale", {"drift": r.problems})
    stale = sum(r.status == "stale" for r in results)
    print(f"\n{len(results) - stale} ok, {stale} stale" + ("" if args.apply or not stale else " (re-run with --apply to retire them)"))


if __name__ == "__main__":
    main()
