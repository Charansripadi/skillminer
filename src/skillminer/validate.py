"""Validator: keep a skill only if a paired A/B test shows it helps.

Arm A is the agent alone, arm B the same agent with the candidate skills. Both arms see the same
scenarios with the same model, so differences come from the skills rather than from harder cases.
A judge marks every conversation (did it follow the expected procedure? did it break a policy?)
and an exact McNemar test on the paired verdicts decides whether the difference is real.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from scipy.stats import binomtest

from .mining.episodes import Episode, load_episodes


@dataclass
class Verdict:
    compliant: bool  # followed the expected procedure
    violation: bool  # broke a hard rule (e.g. refunded an ineligible order)


@dataclass
class ArmStats:
    n: int = 0
    compliant: int = 0
    violations: int = 0
    tool_calls: int = 0

    @property
    def compliance_rate(self) -> float:
        return self.compliant / self.n if self.n else 0.0


@dataclass
class ValidationReport:
    pairs: int
    baseline: ArmStats
    with_skills: ArmStats
    fixed: int  # failed without skills, passed with them
    broken: int  # passed without skills, failed with them
    p_value: float
    decision: str
    reason: str
    by_workflow: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["baseline"]["compliance_rate"] = round(self.baseline.compliance_rate, 3)
        d["with_skills"]["compliance_rate"] = round(self.with_skills.compliance_rate, 3)
        return d


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar test: only the pairs where the arms disagree carry information."""
    n = fixed + broken
    if n == 0:
        return 1.0
    return float(binomtest(min(fixed, broken), n, 0.5).pvalue)


def episodes_by_scenario(trace_dir: str | Path) -> dict[str, Episode]:
    """Latest episode per scenario id (scenarios are run once per arm)."""
    if not Path(trace_dir).exists():
        return {}
    out = {}
    for e in load_episodes(trace_dir):
        sid = e.meta.get("scenario_id")
        if sid:
            out[sid] = e
    return out


def compare(baseline: dict[str, Episode], with_skills: dict[str, Episode],
            judge: Callable[[Episode], Verdict], alpha: float = 0.05) -> ValidationReport:
    shared = sorted(set(baseline) & set(with_skills))
    a, b = ArmStats(), ArmStats()
    fixed = broken = 0
    by_workflow: dict[str, dict] = {}

    for sid in shared:
        va, vb = judge(baseline[sid]), judge(with_skills[sid])
        for stats, v, ep in ((a, va, baseline[sid]), (b, vb, with_skills[sid])):
            stats.n += 1
            stats.compliant += v.compliant
            stats.violations += v.violation
            stats.tool_calls += len(ep.steps)
        fixed += (not va.compliant) and vb.compliant
        broken += va.compliant and (not vb.compliant)
        wf = by_workflow.setdefault(baseline[sid].meta.get("workflow", "?"),
                                    {"n": 0, "baseline": 0, "with_skills": 0})
        wf["n"] += 1
        wf["baseline"] += va.compliant
        wf["with_skills"] += vb.compliant

    p = mcnemar_exact(fixed, broken)
    if not shared:
        decision, reason = "inconclusive", "no paired conversations yet"
    elif b.violations > a.violations or b.compliance_rate < a.compliance_rate:
        decision, reason = "reject", "the skills made the agent worse (more violations or lower compliance)"
    elif b.compliance_rate > a.compliance_rate and p < alpha:
        decision, reason = "approve", f"significant improvement (exact McNemar p={p:.3f})"
    elif b.compliance_rate > a.compliance_rate or b.violations < a.violations:
        decision, reason = ("promising", f"better but not yet significant (p={p:.3f}, {fixed + broken} "
                                         "disagreeing pairs); run more scenarios")
    else:
        decision, reason = "inconclusive", "no measurable difference"
    return ValidationReport(len(shared), a, b, fixed, broken, round(p, 4), decision, reason, by_workflow)


def save_report(report: ValidationReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))
