# SkillMiner

**Mine the workflows your agent keeps repeating, turn them into Agent Skills (`SKILL.md`), and
keep a skill only if an A/B test shows it makes the agent better.**

Agents re-derive the same procedures in every conversation, and they don't do it reliably: they
skip steps and sometimes break policy. SkillMiner records what an agent actually does, finds the
recurring workflows, writes each one up as a skill that ADK's own `SkillToolset` can load,
removes personal data, and checks with a paired A/B test whether the skill helps.

It is a working answer to [google/adk-python#5398](https://github.com/google/adk-python/issues/5398)
(generate skills from recurring patterns across sessions), built on Google ADK 2.10 and running on
any model through LiteLLM.

## How it works

```
 ADK agent ──► TraceRecorderPlugin ──► traces/*.jsonl
                                            │
              ┌─────────────────────────────┘
              ▼
   Pattern Miner ──► Skill Writer ──► Privacy Guard ──► Validator (A/B) ──► Drift Watcher
   (LLM intents       (SKILL.md per     (redact before     (paired, McNemar)   (retire skills
    + branches)        intent)           and after LLM)                         whose tools changed)
```

| Component | Agentic concept | File |
|---|---|---|
| Trace capture | ADK plugin / lifecycle callbacks, no change to the agent | `capture/trace_recorder.py` |
| Simulated customers | Agent-to-agent simulation with personas and hidden ground truth | `simulate/` |
| Pattern Miner | Episodes, online LLM intent induction, intent → branches | `mining/` |
| Skill Writer | Episodic → procedural memory; evidence-grounded generation | `writer.py` |
| Privacy Guard | Guardrails: value-based redaction from traces + pattern scan | `guard.py` |
| Validator | LLM-agent A/B testing, deterministic judge, exact McNemar | `validate.py` |
| Drift Watcher | Tool-signature monitoring, skill lifecycle (`candidate → approved / rejected / stale`) | `drift.py` |
| Safe retries | Retry a rate-limited *model call*, never a whole agent turn | `llm.py` |

## Results (demo support agent, all numbers from this repo)

**1. The baseline agent is unreliable.** 40 simulated conversations, Groq `gpt-oss-120b`, judged
against hidden labels: 26/40 followed the correct procedure, with **2 policy violations**: a refund
issued after the eligibility check said no, and a refund with no eligibility check at all.

**2. Mining recovers the real workflows without labels.**

| Method | Purity | ARI | NMI |
|---|---|---|---|
| TF-IDF + agglomerative clustering (baseline) | 0.97 | 0.48 | 0.67 |
| LLM intent labelling, online intent induction | **1.00** | **0.93** | **0.94** |

Word clustering split conversations by product and street names; the LLM groups them by goal.

**3. The written skills turn observed failures into rules.** From the traces alone, the refund skill
learned: *"NEVER call `issue_refund` if `check_refund_eligibility` returns `eligible: false`
(Variant 6 showed an agent issuing a refund despite ineligibility)."* See [`skills/`](skills).
All three skills load with ADK's own `load_skills_from_dir`.

**4. Pilot A/B: skills help, but the sample is still small.** 8 paired scenarios, same model in both
arms (Qwen 3.8 27B on Groq):

| | Without skills | With mined skills |
|---|---|---|
| Followed the correct procedure | 6/8 (75%) | **8/8 (100%)** |
| Policy violations | 1 | **0** |
| Pairs fixed / broken by skills | | 2 / 0 |

Exact McNemar p = 0.5 (only 2 disagreeing pairs), so the Validator labels this **promising, not
approved**. Skills stay `candidate` until a larger run is significant:
`python scripts/validate.py --n 30 --name big-run` (resumable across days of free-tier quota).

## Lessons worth knowing

- **Mine conversations, not messages.** 13 of 40 conversations needed several turns before the agent
  acted; per-message mining cut those workflows apart.
- **Never retry an agent turn.** A rate-limited final model call once caused the whole turn to be
  re-sent, so the tools ran again and the second, tool-less run was the one recorded. With a refund
  that means paying twice. Retries now happen per model call, before anything is returned, and
  failed runs are recorded with their tool calls.
- **Skills must not invent policy.** The first draft promised "5–7 business days". The writer is now
  told to use only what the evidence shows, and the regenerated skills forbid invented timelines.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# agents/support_agent/.env needs: GROQ_API_KEY=..., SKILLMINER_MODEL=groq/openai/gpt-oss-120b

python scripts/simulate.py --n 40 --seed 1                        # 1. generate conversations
skillminer-traces traces                                           #    look at them
skillminer-mine traces --env-file agents/support_agent/.env        # 2. find workflows
skillminer-write --tools support_agent.agent:TOOLS --import-path agents \
                 --env-file agents/support_agent/.env              # 3. write + guard skills
python scripts/validate.py --n 30 --name big-run                   # 4. A/B test them
skillminer-drift --tools support_agent.agent:TOOLS --import-path agents   # 5. detect drift
pytest                                                             # 23 offline tests, no API calls
```

Use the skills in any ADK agent:

```python
from google.adk.skills import load_skills_from_dir
from google.adk.tools.skill_toolset import SkillToolset

agent = Agent(..., tools=[*my_tools, SkillToolset(skills=load_skills_from_dir("skills"))])
```

Record traces from any ADK app with one line:

```python
from skillminer.capture import TraceRecorderPlugin
app = App(name="my_agent", root_agent=agent, plugins=[TraceRecorderPlugin(trace_dir="traces")])
```

## Limitations

- The demo domain is a small simulated shop; real traffic is messier.
- The A/B pilot is 8 pairs; significance needs roughly 30+.
- The judge is rule-based for this demo; other domains need their own judge (or an LLM judge).
- Free-tier limits (Groq: 8K tokens/min, 200K tokens/day per model) make large runs slow.

Built by Sri Charan Sripadi · Apache-2.0
