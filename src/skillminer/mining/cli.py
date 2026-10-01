"""Mine recurring workflows from traces and print a report.

    skillminer-mine traces
    skillminer-mine traces --k 3
    skillminer-mine traces --score-key workflow --label-map refund=refund_request,refund_denied=refund_request

Writes runs/mining.json (git-ignored: it contains raw customer data) for the Skill Writer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .cluster import MiningResult, group_by_intent, mine
from .episodes import load_episodes
from .evaluate import score


def _parse_label_map(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    return dict(pair.split("=", 1) for pair in text.split(","))


def to_json(result: MiningResult) -> dict:
    return {
        "k": result.k,
        "silhouette_by_k": result.silhouette_by_k,
        "unclustered_sessions": [e.session_id for e in result.unclustered],
        "clusters": [
            {
                "id": c.id,
                "size": c.size,
                "keywords": c.keywords,
                "paths": [{"path": list(p), "count": n} for p, n in c.path_counts],
                "episodes": [
                    {"session_id": e.session_id, "path": e.path, "user_messages": e.user_messages,
                     "agent_messages": e.agent_messages, "steps": [asdict(s) for s in e.steps]}
                    for e in c.episodes
                ],
            }
            for c in result.clusters
        ],
    }


def _mine_with_llm(episodes, args) -> MiningResult:
    import asyncio
    import logging
    import os

    import litellm
    from dotenv import load_dotenv

    from ..simulate.session_runner import _with_backoff
    from ..llm import provider_kwargs
    from .intents import IntentLabeler

    load_dotenv(args.env_file) if args.env_file else load_dotenv()
    litellm.suppress_debug_info = True
    for noisy in ("LiteLLM", "litellm", "httpx"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    model = args.model or os.getenv("SKILLMINER_SIM_MODEL", "groq/openai/gpt-oss-20b")
    labeler = IntentLabeler(model, cache_path=args.cache, **provider_kwargs(model))
    active = [e for e in episodes if e.has_actions]
    todo = sum(e.session_id not in labeler.labels for e in active)
    print(f"Labelling intents with {model}: {todo} new conversations, {len(active) - todo} cached")

    async def run():
        for e in active:  # one at a time so each label can reuse intents created before it
            if e.session_id not in labeler.labels:
                await _with_backoff(lambda e=e: labeler.label_all([e]), retries=4, base_delay=20)
        return labeler.labels

    labels = asyncio.run(run())
    descriptions = {n: i.description for n, i in labeler.intents.items()}
    return group_by_intent(episodes, labels, descriptions)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mine recurring workflows from agent traces.")
    parser.add_argument("path", nargs="?", default="traces", help="trace file or directory")
    parser.add_argument("--k", type=int, default=None, help="number of clusters (default: chosen by silhouette)")
    parser.add_argument("--text-weight", type=float, default=0.6, help="weight of request text vs tool path")
    parser.add_argument("--score-key", default=None, help="meta key holding ground-truth labels (simulated data)")
    parser.add_argument("--label-map", default=None, help="merge labels, e.g. refund_denied=refund,a=b")
    parser.add_argument("--out", default="runs/mining.json", help="where to save the result")
    parser.add_argument("--method", choices=["llm", "tfidf"], default="llm",
                        help="llm: an LLM labels each conversation's intent; tfidf: word/tool-path clustering baseline")
    parser.add_argument("--model", default=None, help="LLM for intent labelling (default: $SKILLMINER_SIM_MODEL)")
    parser.add_argument("--env-file", default=None, help=".env file with API keys")
    parser.add_argument("--cache", default="runs/intents.json", help="intent label cache (re-runs cost nothing)")
    args = parser.parse_args(argv)

    episodes = load_episodes(args.path)
    if args.method == "tfidf":
        result = mine(episodes, k=args.k, text_weight=args.text_weight)
    else:
        result = _mine_with_llm(episodes, args)

    print(f"{len(episodes)} conversations: {len(episodes) - len(result.unclustered)} with tool calls, "
          f"{len(result.unclustered)} without (set aside)")
    if result.silhouette_by_k:
        sil = ", ".join(f"k={k}: {s:.2f}" for k, s in result.silhouette_by_k.items())
        print(f"Silhouette by k: {sil}  ->  chose k={result.k}")

    label_map = _parse_label_map(args.label_map)
    label_of = None
    if args.score_key:
        def label_of(e):
            raw = e.meta.get(args.score_key)
            return label_map.get(raw, raw) if raw is not None else None

    for c in result.clusters:
        print(f"\n=== Cluster {c.id}  ({c.size} conversations)   keywords: {', '.join(c.keywords)}")
        if label_of:
            print("    true labels:", dict(sorted(
                {lab: sum(label_of(e) == lab for e in c.episodes) for lab in {label_of(e) for e in c.episodes}}.items(),
                key=lambda kv: -kv[1])))
        for path, count in c.path_counts[:5]:
            print(f"    {count:3d} x  {' -> '.join(path)}")
        if len(c.path_counts) > 5:
            print(f"    ... {len(c.path_counts) - 5} more path variants")

    if label_of:
        s = score(result, label_of)
        print(f"\nScore vs ground truth ({s['labelled']} labelled conversations): "
              f"purity={s.get('purity')}  ARI={s.get('ari')}  NMI={s.get('nmi')}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_json(result), indent=2, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
