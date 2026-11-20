"""Small statistics used by the analysis: macro-F1, a cluster bootstrap and a rank trend."""
from __future__ import annotations

import numpy as np


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Unweighted mean F1 over all classes; a class with no support and no predictions scores 0."""
    scores = []
    for c in range(n_classes):
        tp = np.sum((y == c) & (pred == c))
        fp = np.sum((y != c) & (pred == c))
        fn = np.sum((y == c) & (pred != c))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def cluster_bootstrap(clusters: np.ndarray, statistic, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """Resample whole clusters with replacement and recompute ``statistic(row_index)`` each time.

    Rows that share a cluster (a spatial block, a trained model, a workload) are not
    independent, so they are kept or dropped together.
    """
    rng = np.random.default_rng(seed)
    ids, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(ids))]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        out.append(statistic(np.concatenate([members[k] for k in pick])))
    return np.array(out)


def interval(samples: np.ndarray, level: float = 0.95) -> list:
    lo, hi = np.nanpercentile(samples, [50 * (1 - level), 100 - 50 * (1 - level)], axis=0)
    return [np.asarray(lo).tolist(), np.asarray(hi).tolist()]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for val in np.unique(v):                      # average ranks over ties
            m = v == val
            r[m] = r[m].mean()
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
