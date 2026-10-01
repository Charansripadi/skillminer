"""A/B test the mined skills on the demo support agent.

    python scripts/validate.py --n 10                      # 10 paired scenarios (20 conversations)
    python scripts/validate.py --n 30 --name big-run       # resumes: finished pairs are skipped

Arm A: the agent alone. Arm B: the same agent and model plus skills/ through ADK's SkillToolset.
Scenarios focus on the workflows where the baseline agent was weakest. Each pair runs A then B,
so a run cut short by a daily quota still leaves complete pairs. Re-running the same --name
continues where it stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agents" / "support_agent"
POLITE = "Polite and clear. You give all the details up front."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10, help="paired scenarios to run")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--workflows", default="refund,refund_denied,change_address,track_order")
    parser.add_argument("--skills-dir", default=str(ROOT / "skills"))
    parser.add_argument("--model", default=None, help="agent model for BOTH arms (default: $SKILLMINER_VALIDATE_MODEL or $SKILLMINER_MODEL)")
    parser.add_argument("--all-personas", action="store_true", help="use every persona (default: polite only, fewer turns and tokens)")
    parser.add_argument("--name", default="pilot", help="run name; re-use it to resume")
    args = parser.parse_args()

    load_dotenv(AGENT_DIR / ".env")
    model = args.model or os.getenv("SKILLMINER_VALIDATE_MODEL") or os.getenv("SKILLMINER_MODEL")
    os.environ["SKILLMINER_MODEL"] = model  # agent.py reads this at import time
    sys.path.insert(0, str(ROOT / "agents"))
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    for noisy in ("LiteLLM", "litellm", "google_adk", "google.adk", "httpx"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    import litellm
    from support_agent.agent import build_app
    from support_agent.judge import judge
    from support_agent.scenarios import build_scenarios
    from support_agent.tools import reset_orders

    from skillminer.llm import provider_kwargs
    from skillminer.simulate import UserSimulator, run_scenarios
    from skillminer.validate import compare, episodes_by_scenario, save_report
    from skillminer.writer import set_status

    litellm.suppress_debug_info = True
    wanted = set(args.workflows.split(","))
    scenarios = [s for s in build_scenarios(args.n * 8, seed=args.seed) if s.label["workflow"] in wanted][: args.n]
    if not args.all_personas:
        for s in scenarios:
            s.persona = POLITE
            s.max_turns = 3

    run_dir = ROOT / "runs" / "validation" / args.name
    dirs = {"baseline": run_dir / "baseline", "with_skills": run_dir / "with_skills"}
    progress_file = run_dir / "done.json"
    done = json.loads(progress_file.read_text()) if progress_file.exists() else {"baseline": [], "with_skills": []}
    apps = {"baseline": build_app(None, trace_dir=dirs["baseline"]),
            "with_skills": build_app(args.skills_dir, trace_dir=dirs["with_skills"])}

    sim_model = os.getenv("SKILLMINER_SIM_MODEL", "groq/openai/gpt-oss-20b")
    simulator = UserSimulator(sim_model, temperature=0.9, **provider_kwargs(sim_model))
    skills = sorted(p.name for p in Path(args.skills_dir).iterdir() if (p / "SKILL.md").exists())
    print(f"A/B validation '{args.name}': {len(scenarios)} paired scenarios | agent model {model} | "
          f"skills: {', '.join(skills)}\n")

    async def run() -> None:
        start = time.monotonic()
        for i, scenario in enumerate(scenarios, 1):
            for arm in ("baseline", "with_skills"):
                if scenario.id in done[arm]:
                    continue
                [result] = await run_scenarios(apps[arm], [scenario], simulator, reset_world=reset_orders)
                if not result.ok:
                    print(f"  {scenario.id} {arm}: failed ({result.error[:120]})")
                    if "per day" in (result.error or "").lower():
                        print("\nDaily quota reached. Re-run the same command later to continue.")
                        return
                    continue
                done[arm].append(scenario.id)
                run_dir.mkdir(parents=True, exist_ok=True)
                progress_file.write_text(json.dumps(done))
            print(f"[{i}/{len(scenarios)}] {scenario.id} {scenario.label['workflow']:15s} "
                  f"({time.monotonic() - start:.0f}s)")

    asyncio.run(run())

    complete = set(done["baseline"]) & set(done["with_skills"])
    a = {k: v for k, v in episodes_by_scenario(dirs["baseline"]).items() if k in complete}
    b = {k: v for k, v in episodes_by_scenario(dirs["with_skills"]).items() if k in complete}
    report = compare(a, b, judge)
    save_report(report, run_dir / "report.json")

    print(f"\n{report.pairs} complete pairs")
    print(f"  compliance  without skills: {report.baseline.compliant}/{report.baseline.n} "
          f"({report.baseline.compliance_rate:.0%})   with skills: {report.with_skills.compliant}/{report.with_skills.n} "
          f"({report.with_skills.compliance_rate:.0%})")
    print(f"  violations  without skills: {report.baseline.violations}   with skills: {report.with_skills.violations}")
    print(f"  pairs fixed by skills: {report.fixed}   pairs broken by skills: {report.broken}   exact McNemar p={report.p_value}")
    for wf, s in report.by_workflow.items():
        print(f"    {wf:22s} {s['baseline']}/{s['n']} -> {s['with_skills']}/{s['n']}")
    print(f"\nDecision: {report.decision.upper()}: {report.reason}")

    status = {"approve": "approved", "reject": "rejected"}.get(report.decision, "candidate")
    for name in skills:
        set_status(Path(args.skills_dir) / name, status,
                   {"validation": {"run": args.name, "decision": report.decision, "p_value": report.p_value,
                                   "pairs": report.pairs}})
    print(f"Skill status set to '{status}'. Report: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
