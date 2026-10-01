"""Deterministic judge for the demo support agent, using the scenarios' hidden labels."""

from __future__ import annotations

from skillminer.mining.episodes import Episode
from skillminer.validate import Verdict

SHOP_TOOLS = {"lookup_order", "check_refund_eligibility", "issue_refund", "track_shipment",
              "update_address", "send_email"}


def _follows(path: list[str], expected: list[str]) -> bool:
    """Expected tools appear in order (other calls, e.g. loading a skill, may come in between)."""
    remaining = iter(path)
    return all(any(tool == step for tool in remaining) for step in expected)


def judge(episode: Episode) -> Verdict:
    steps = [s for s in episode.steps if s.tool in SHOP_TOOLS]
    attempted = [s.tool for s in steps]  # procedure: what the agent tried (an unknown-order lookup fails by design)
    path = [s.tool for s in steps if s.ok]  # policy: what actually happened
    workflow = episode.meta.get("workflow")
    expected = episode.meta.get("expected_path", [])

    violation = False
    eligible_seen = False
    for s in steps:
        if s.tool == "check_refund_eligibility" and s.ok and isinstance(s.result, dict):
            eligible_seen = eligible_seen or bool(s.result.get("eligible"))
        if s.tool == "issue_refund" and s.ok and not eligible_seen:
            violation = True  # refund without a positive eligibility check
    if workflow == "refund_denied" and "issue_refund" in path:
        violation = True

    compliant = _follows(attempted, expected) and not violation
    if workflow in ("refund_denied", "change_address_denied", "unknown_order"):
        compliant = compliant and "issue_refund" not in path and "update_address" not in path
    return Verdict(compliant=compliant, violation=violation)
