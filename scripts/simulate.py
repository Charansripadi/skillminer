"""Generate support-agent sessions with simulated customers.

    python scripts/simulate.py --n 5             # quick check (~5 conversations)
    python scripts/simulate.py --n 60 --seed 1   # a dataset for mining

Traces land in traces/support_agent/ (written by the agent's TraceRecorderPlugin).
The agent uses SKILLMINER_MODEL and the simulated customer uses SKILLMINER_SIM_MODEL,
both read from agents/support_agent/.env. Using two different Groq models gives each its
own rate-limit quota.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agents" / "support_agent"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=5, help="number of conversations")
    parser.add_argument("--seed", type=int, default=0, help="scenario sampling seed")
    args = parser.parse_args()

    # Load .env before importing the agent: agent.py reads SKILLMINER_MODEL at import time.
    load_dotenv(AGENT_DIR / ".env")
    sys.path.insert(0, str(ROOT / "agents"))
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    for noisy in ("LiteLLM", "litellm", "google_adk", "google.adk", "httpx"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    import litellm
    from support_agent.agent import app
    from support_agent.scenarios import build_scenarios
    from support_agent.tools import reset_orders

    from skillminer.simulate import UserSimulator, run_scenarios

    litellm.suppress_debug_info = True
    sim_model = os.getenv("SKILLMINER_SIM_MODEL", "groq/openai/gpt-oss-20b")
    sim_kwargs = {"include_reasoning": False} if sim_model.startswith("groq/openai/gpt-oss") else {}
    simulator = UserSimulator(sim_model, temperature=0.9, **sim_kwargs)

    scenarios = build_scenarios(args.n, seed=args.seed)
    print(f"Running {len(scenarios)} simulated conversations")
    print(f"  agent model:     {os.getenv('SKILLMINER_MODEL')}")
    print(f"  simulator model: {sim_model}\n")

    start = time.monotonic()
    by_id = {s.id: s for s in scenarios}

    def report(result):
        s = by_id[result.scenario_id]
        status = "ok " if result.ok else "ERR"
        print(f"[{status}] {result.scenario_id}  {s.label['workflow']:22s} turns={result.turns}"
              + (f"  {result.error}" if result.error else ""))

    results = asyncio.run(run_scenarios(app, scenarios, simulator, reset_world=reset_orders, on_result=report))

    ok = sum(r.ok for r in results)
    print(f"\n{ok}/{len(results)} conversations completed in {time.monotonic() - start:.0f}s")
    print("Workflows simulated:", dict(Counter(s.label["workflow"] for s in scenarios)))
    print("View them with:  skillminer-traces traces")


if __name__ == "__main__":
    main()
