"""Print recorded traces as one-line tool paths.

    skillminer-traces traces/
    skillminer-traces traces/support_agent/2026-10-01.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterator


def iter_traces(path: str | Path) -> Iterator[dict]:
    """Yield every trace from a .jsonl file or from all .jsonl files under a directory."""
    path = Path(path)
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        with file.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def tool_path(trace: dict) -> str:
    """'lookup_order -> check_refund_eligibility -> issue_refund', marking failed steps with '!'."""
    names = [s["tool"] + ("" if s["status"] == "ok" else "!") for s in trace["steps"]]
    return " -> ".join(names) if names else "(no tools)"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show recorded agent traces as tool paths.")
    parser.add_argument("path", nargs="?", default="traces", help="trace file or directory")
    args = parser.parse_args(argv)

    traces = list(iter_traces(args.path))
    if not traces:
        print(f"No traces found under {args.path}")
        return

    for t in traces:
        print(f"[{t['started_at'][11:19]}] {t['user_message'][:50]!r}")
        print(f"    {tool_path(t)}   ({t['outcome']}, {t['duration_ms']} ms)")

    print(f"\n{len(traces)} traces. Most common tool paths:")
    for path, count in Counter(tool_path(t) for t in traces).most_common(5):
        print(f"  {count:3d} x  {path}")


if __name__ == "__main__":
    main()
