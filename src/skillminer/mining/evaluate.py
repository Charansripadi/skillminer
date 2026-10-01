"""Score clusters against hidden ground-truth labels (only possible on simulated data)."""

from __future__ import annotations

from collections import Counter
from typing import Callable

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .cluster import MiningResult
from .episodes import Episode


def score(result: MiningResult, label_of: Callable[[Episode], str | None]) -> dict:
    """Purity, ARI and NMI of the clustering, using only episodes that have a label.

    purity: share of episodes that sit in a cluster whose majority label matches their own.
    ARI:    agreement with the true grouping, corrected for chance (1 = perfect, ~0 = random).
    NMI:    shared information between clusters and labels (1 = perfect, 0 = none).
    """
    true, pred, majority = [], [], {}
    for cluster in result.clusters:
        labels = [label_of(e) for e in cluster.episodes]
        labelled = [lab for lab in labels if lab is not None]
        if labelled:
            majority[cluster.id] = Counter(labelled).most_common(1)[0]
        for lab in labelled:
            true.append(lab)
            pred.append(cluster.id)

    if not true:
        return {"labelled": 0}
    purity = sum(count for _, count in majority.values()) / len(true)
    return {
        "labelled": len(true),
        "purity": round(purity, 3),
        "ari": round(adjusted_rand_score(true, pred), 3),
        "nmi": round(normalized_mutual_info_score(true, pred), 3),
        "cluster_majority": {cid: lab for cid, (lab, _) in majority.items()},
    }
