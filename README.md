# SkillMiner

Mine recurring workflows from agent session traces and turn them into validated Agent Skills (`SKILL.md`).

Agents keep re-deriving the same procedures in every conversation, and small or busy models skip
steps while doing it. SkillMiner watches what an agent actually does, finds the workflows that
recur, and (next steps) writes them up as skills, strips private data, and keeps a skill only if
an A/B test shows it helps. It is a working answer to
[google/adk-python#5398](https://github.com/google/adk-python/issues/5398).

## Pipeline

| Stage | What it does | Status |
|---|---|---|
| Capture | `TraceRecorderPlugin`: an ADK plugin that records every run as JSONL, no agent changes | done |
| Simulate | LLM customers with personas drive the demo agent; each scenario carries a hidden ground-truth label | done |
| Mine | Join turns into episodes, label each conversation's intent with an LLM, list each intent's branches | done |
| Write | Draft a `SKILL.md` per intent from its best episodes | next |
| Guard | Remove personal data before a skill is saved | planned |
| Validate | A/B test the agent with and without the skill; keep it only if it helps | planned |
| Drift | Retire skills whose tools changed | planned |

## Results so far (demo support agent, Groq `gpt-oss-120b`, 65 simulated conversations)

**Baseline: the agent without skills follows the correct procedure inconsistently.**

| Workflow | Followed the expected procedure |
|---|---|
| refund | 3/8 |
| refund denied | 1/4 |
| change address | 5/8 |
| track order | 6/8 |

Two policy violations: a refund issued after the eligibility check said no, and a refund issued
with no eligibility check at all.

**Mining: LLM intent labelling recovers the real workflows far better than word clustering.**

| Method | Purity | ARI | NMI |
|---|---|---|---|
| TF-IDF + agglomerative clustering (baseline) | 0.97 | 0.48 | 0.67 |
| LLM intent labelling (`gpt-oss-20b`, online intent induction) | **1.00** | **0.93** | **0.94** |

Word clustering split conversations by product and street names; the LLM groups them by goal.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# put GROQ_API_KEY and SKILLMINER_MODEL in agents/support_agent/.env
python scripts/simulate.py --n 40 --seed 1          # generate conversations
skillminer-traces traces                             # look at them
skillminer-mine traces --env-file agents/support_agent/.env   # find the workflows
pytest
```
