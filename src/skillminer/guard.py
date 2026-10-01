"""Privacy Guard: keep personal and case-specific data out of skills.

Skills are shared with every future conversation (and maybe other teams), so they must not
contain one customer's email, address or order. Two layers:

1. Value-based redaction. Every concrete value that appeared in a tool call's arguments or
   result is case data by definition. We collect those values from the traces and replace them
   with typed placeholders (<EMAIL>, <ADDRESS>, <ORDER_ID>, ...). This catches things no regex
   would, like a product name or a street name.
2. Pattern scan. Regexes for emails, phone numbers, card numbers and UK postcodes catch anything
   that was never in a tool call (e.g. typed only in a chat message).

The same scan runs on the finished SKILL.md; a skill with any remaining finding is blocked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "CARD_NUMBER": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "PHONE": re.compile(r"(?<!\w)\+?\d[\d ()-]{8,}\d\b"),
    "UK_POSTCODE": re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}\b"),
}

# Map argument/result keys to placeholder types. Anything else becomes <VALUE>.
KEY_TYPES = [
    (("email", "to", "customer"), "EMAIL"),
    (("address",), "ADDRESS"),
    (("order_id", "refund_id", "id"), "ORDER_ID"),
    (("item", "product"), "ITEM"),
    (("name",), "NAME"),
    (("amount", "max_refund", "price"), "AMOUNT"),
    (("reason",), "REASON"),
]
# Values that are not case data even though they appear in tool results.
GENERIC_VALUES = {"ok", "error", "true", "false", "none", "delivered", "shipped", "processing",
                  "already delivered", "2 days", "5 days"}


@dataclass
class Finding:
    kind: str
    text: str


@dataclass
class GuardReport:
    redacted_values: int = 0
    pattern_findings: list[Finding] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.pattern_findings


def _type_for_key(key: str) -> str:
    k = key.lower()
    for keys, kind in KEY_TYPES:
        if any(k == x or k.endswith("_" + x) for x in keys):
            return kind
    return "VALUE"


# From tool *results* we only take clearly personal fields: results also carry system text such
# as "Outside the 30-day refund window", which is policy the skill should keep.
RESULT_KINDS = {"EMAIL", "ADDRESS", "ORDER_ID", "ITEM", "NAME"}


def collect_case_values(steps: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Concrete values seen in tool args/results -> placeholder type. Steps are dicts with args/result."""
    values: dict[str, str] = {}

    def visit(key: str, value: Any, from_result: bool) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                visit(k, v, from_result)
        elif isinstance(value, (list, tuple)):
            for v in value:
                visit(key, v, from_result)
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            text = str(value).strip()
            kind = _type_for_key(key)
            numeric = isinstance(value, (int, float)) or text.replace(".", "", 1).isdigit()
            if text.lower() in GENERIC_VALUES or len(text) < 3 or (numeric and kind == "VALUE"):
                return
            if from_result and kind not in RESULT_KINDS:
                return
            values.setdefault(text, kind)

    for step in steps:
        visit("args", step.get("args") or {}, from_result=False)
        visit("result", step.get("result") or {}, from_result=True)
    return values


def redact(text: str, case_values: dict[str, str]) -> tuple[str, int]:
    """Replace every known case value (longest first, case-insensitive) with its placeholder."""
    count = 0
    for value in sorted(case_values, key=len, reverse=True):
        pattern = re.compile(re.escape(value), re.IGNORECASE)
        text, n = pattern.subn(f"<{case_values[value]}>", text)
        count += n
    return text, count


def scan(text: str) -> list[Finding]:
    """Pattern findings left in a text (placeholders like <EMAIL> are ignored)."""
    findings = []
    for kind, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append(Finding(kind, match.group(0)))
    return findings


def guard(text: str, case_values: dict[str, str]) -> tuple[str, GuardReport]:
    """Redact known case values, then scan for anything personal that is left."""
    clean, n = redact(text, case_values)
    findings = scan(clean)
    for f in findings:  # pattern hits are redacted too, but still reported so a human can look
        clean = clean.replace(f.text, f"<{f.kind}>")
    return clean, GuardReport(redacted_values=n, pattern_findings=findings)
