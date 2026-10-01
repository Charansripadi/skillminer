"""Group episodes into recurring workflows.

Each episode becomes one vector built from two views:
  * what the customer asked for: TF-IDF of their words (IDs and numbers removed), and
  * what the agent did: TF-IDF of its tool steps and step pairs (so order matters a little).
Episodes are clustered with average-linkage agglomerative clustering on cosine distance.
The number of clusters is picked automatically by silhouette score unless given.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import hstack
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from .episodes import Episode

WORDS_ONLY = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"  # drops order IDs like A1001 and numbers


@dataclass
class Cluster:
    id: int
    episodes: list[Episode]
    keywords: list[str]
    path_counts: list[tuple[tuple[str, ...], int]]  # most common simplified paths first

    @property
    def size(self) -> int:
        return len(self.episodes)


@dataclass
class MiningResult:
    clusters: list[Cluster]
    unclustered: list[Episode]  # episodes where the agent never called a tool
    k: int
    silhouette_by_k: dict[int, float] = field(default_factory=dict)


def _tool_ngrams(path: list[str]) -> list[str]:
    return path + [f"{a}>{b}" for a, b in zip(path, path[1:])]


def featurize(episodes: list[Episode], text_weight: float = 0.6):
    """Return (feature matrix, fitted text vectorizer)."""
    text_vec = TfidfVectorizer(token_pattern=WORDS_ONLY, stop_words="english", sublinear_tf=True)
    tool_vec = TfidfVectorizer(analyzer=_tool_ngrams)
    x_text = normalize(text_vec.fit_transform([e.request_text for e in episodes]))
    x_tool = normalize(tool_vec.fit_transform([e.path for e in episodes]))
    x = normalize(hstack([text_weight * x_text, (1 - text_weight) * x_tool]).tocsr())
    return x, text_vec, x_text


def _cluster(x: np.ndarray, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(x)


def choose_k(x: np.ndarray, k_min: int = 2, k_max: int = 10) -> tuple[int, dict[int, float]]:
    """Try each k and keep the one whose clusters are best separated (highest silhouette)."""
    n = x.shape[0]
    scores = {}
    for k in range(k_min, min(k_max, n - 1) + 1):
        labels = _cluster(x, k)
        if len(set(labels)) > 1:
            scores[k] = float(silhouette_score(x, labels, metric="cosine"))
    if not scores:
        return 1, {}
    return max(scores, key=scores.get), scores


def group_by_intent(episodes: list[Episode], labels: dict[str, str],
                    descriptions: dict[str, str] | None = None) -> MiningResult:
    """One cluster per LLM-assigned intent; the tool paths inside it are that intent's branches."""
    active = [e for e in episodes if e.has_actions]
    idle = [e for e in episodes if not e.has_actions]
    by_intent: dict[str, list[Episode]] = {}
    for e in active:
        by_intent.setdefault(labels.get(e.session_id, "unknown"), []).append(e)

    clusters = []
    for name, members in sorted(by_intent.items(), key=lambda kv: -len(kv[1])):
        keywords = [name] + ([descriptions[name]] if descriptions and descriptions.get(name) else [])
        paths = Counter(tuple(e.path) for e in members).most_common()
        clusters.append(Cluster(id=len(clusters), episodes=members, keywords=keywords, path_counts=paths))
    return MiningResult(clusters=clusters, unclustered=idle, k=len(clusters))


def mine(episodes: list[Episode], k: int | None = None, text_weight: float = 0.6) -> MiningResult:
    """Cluster the episodes that contain tool calls. Episodes without any tool call are set aside."""
    active = [e for e in episodes if e.has_actions]
    idle = [e for e in episodes if not e.has_actions]
    if len(active) < 3:
        return MiningResult(clusters=[], unclustered=episodes, k=0)

    x_sparse, text_vec, x_text = featurize(active, text_weight)
    x = x_sparse.toarray()
    silhouettes: dict[int, float] = {}
    if k is None:
        k, silhouettes = choose_k(x)
    labels = _cluster(x, k) if k > 1 else np.zeros(len(active), dtype=int)

    vocab = np.array(text_vec.get_feature_names_out())
    clusters = []
    for cid in sorted(set(labels), key=lambda c: -int(np.sum(labels == c))):
        members = [e for e, lab in zip(active, labels) if lab == cid]
        centroid = np.asarray(x_text[labels == cid].mean(axis=0)).ravel()
        keywords = vocab[np.argsort(centroid)[::-1][:6]].tolist()
        paths = Counter(tuple(e.path) for e in members).most_common()
        clusters.append(Cluster(id=len(clusters), episodes=members, keywords=keywords, path_counts=paths))
    return MiningResult(clusters=clusters, unclustered=idle, k=k, silhouette_by_k=silhouettes)
