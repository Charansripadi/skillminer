"""Simulated customer scenarios for the demo support agent.

Each scenario has a hidden ``workflow`` label: the procedure a well-behaved agent should follow.
SkillMiner never sees it. We use it afterwards to measure how well the miner rediscovered the
real workflows from traces alone.
"""

from __future__ import annotations

import random

from skillminer.simulate import Scenario

from .tools import _SEED_ORDERS, REFUND_WINDOW_DAYS

PERSONAS = [
    "Polite and clear. You give all the details up front.",
    "Terse. You write very short messages with no greetings.",
    "Frustrated and a bit rude, but you still cooperate.",
    "Vague. Your first message does not include the order ID; you give it only when asked.",
    "Chatty. You add some irrelevant small talk but stay on topic eventually.",
    "Not a native English speaker. Simple sentences, small grammar mistakes.",
]

NEW_ADDRESSES = ["5 Oak Street, Leeds", "77 River Rd, Bath", "10 Market Sq, Durham", "31 Green Lane, Hull"]
UNKNOWN_IDS = ["A9999", "B1234", "A0007", "A2001"]

# Expected tool paths per workflow, used only for scoring (never shown to the miner).
EXPECTED_PATHS = {
    "refund": ["lookup_order", "check_refund_eligibility", "issue_refund", "send_email"],
    "refund_denied": ["lookup_order", "check_refund_eligibility"],
    "track_order": ["lookup_order", "track_shipment"],
    "change_address": ["lookup_order", "update_address", "send_email"],
    "change_address_denied": ["lookup_order"],
    "unknown_order": ["lookup_order"],
}

# Share of conversations per workflow. Allowed and denied variants are both well represented
# (only 3 of the 12 orders are refundable, so sampling orders uniformly would starve "refund").
WORKFLOW_SHARES = {
    "refund": 0.20,
    "refund_denied": 0.12,
    "track_order": 0.20,
    "change_address": 0.18,
    "change_address_denied": 0.12,
    "unknown_order": 0.18,
}

REFUND_REASONS = ["it arrived broken", "it stopped working", "you changed your mind", "it is the wrong size"]


def _refund_workflow(order: dict) -> str:
    eligible = order["status"] == "delivered" and order["days_since_delivery"] <= REFUND_WINDOW_DAYS
    return "refund" if eligible else "refund_denied"


def _orders_for(workflow: str) -> list[str]:
    """Order IDs for which the correct outcome is this workflow."""
    if workflow in ("refund", "refund_denied"):
        return [k for k, o in _SEED_ORDERS.items() if _refund_workflow(o) == workflow]
    if workflow == "track_order":
        return [k for k, o in _SEED_ORDERS.items() if o["status"] in ("shipped", "processing")]
    if workflow == "change_address":
        return [k for k, o in _SEED_ORDERS.items() if o["status"] == "processing"]
    if workflow == "change_address_denied":
        return [k for k, o in _SEED_ORDERS.items() if o["status"] == "shipped"]
    return UNKNOWN_IDS


def _workflow_quota(n: int, rng: random.Random) -> list[str]:
    """Exactly n workflows in the configured shares (stratified), in random order."""
    quota = [w for w, share in WORKFLOW_SHARES.items() for _ in range(int(n * share))]
    leftovers = rng.choices(list(WORKFLOW_SHARES), weights=list(WORKFLOW_SHARES.values()), k=n - len(quota))
    quota += leftovers
    rng.shuffle(quota)
    return quota


def _goal(workflow: str, order_id: str, rng: random.Random) -> str:
    if workflow == "unknown_order":
        return f"You want an update on your order {order_id}. (This ID is wrong, but you don't know that.)"
    item = _SEED_ORDERS[order_id]["item"]
    if workflow in ("refund", "refund_denied"):
        return f"You want a full refund for your {item} (order {order_id}) because {rng.choice(REFUND_REASONS)}."
    if workflow == "track_order":
        return f"You want to know where your {item} (order {order_id}) is and when it will arrive."
    return f"You want to change the delivery address of your {item} (order {order_id}) to {rng.choice(NEW_ADDRESSES)}."


def build_scenarios(n: int, seed: int = 0) -> list[Scenario]:
    """Build n scenarios with a guaranteed mix of workflows. The same seed gives the same list."""
    rng = random.Random(seed)
    scenarios = []
    for i, workflow in enumerate(_workflow_quota(n, rng)):
        order_id = rng.choice(_orders_for(workflow))
        scenarios.append(Scenario(
            id=f"s{seed}-{i:03d}",
            goal=_goal(workflow, order_id, rng),
            persona=rng.choice(PERSONAS),
            label={"workflow": workflow, "order_id": order_id, "expected_path": EXPECTED_PATHS[workflow]},
        ))
    return scenarios
