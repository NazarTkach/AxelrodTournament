from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score

from clustering import ClusterConfig, hierarchical_cluster_labels


@dataclass
class StabilityConfig:
    B: int = 100          # bootstrap repeats
    seed: int = 0


def _labels_on_overlap(
    idx_a: np.ndarray,
    lab_a: np.ndarray,
    idx_b: np.ndarray,
    lab_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given bootstrap samples idx_a, idx_b (with replacement), compute ARI on the overlap of unique indices.
    """
    ua = np.unique(idx_a)
    ub = np.unique(idx_b)
    overlap = np.intersect1d(ua, ub)
    if overlap.size < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    # For each original index in overlap, assign its cluster label from that bootstrap.
    # If duplicates exist, they all share same label; take first occurrence.
    map_a = {}
    for pos, orig in enumerate(idx_a):
        if orig in overlap and orig not in map_a:
            map_a[int(orig)] = int(lab_a[pos])

    map_b = {}
    for pos, orig in enumerate(idx_b):
        if orig in overlap and orig not in map_b:
            map_b[int(orig)] = int(lab_b[pos])

    y_a = np.array([map_a[int(i)] for i in overlap], dtype=int)
    y_b = np.array([map_b[int(i)] for i in overlap], dtype=int)
    return y_a, y_b


def bootstrap_stability_curve(
    X: np.ndarray,
    K_values: List[int],
    cluster_cfg: ClusterConfig,
    stab_cfg: StabilityConfig,
) -> Dict[int, Dict[str, float]]:
    """
    Returns {K: {'mean_ari': ..., 'std_ari': ..., 'pairs': ...}}
    """
    rng = np.random.default_rng(stab_cfg.seed)
    n = X.shape[0]

    out: Dict[int, Dict[str, float]] = {}

    for K in K_values:
        aris: List[float] = []
        for _ in range(stab_cfg.B):
            idx1 = rng.integers(0, n, size=n)
            idx2 = rng.integers(0, n, size=n)

            X1 = X[idx1]
            X2 = X[idx2]

            lab1 = hierarchical_cluster_labels(X1, K=K, cfg=cluster_cfg)
            lab2 = hierarchical_cluster_labels(X2, K=K, cfg=cluster_cfg)

            y1, y2 = _labels_on_overlap(idx1, lab1, idx2, lab2)
            if y1.size >= 2:
                aris.append(float(adjusted_rand_score(y1, y2)))

        if len(aris) == 0:
            out[K] = {"mean_ari": float("nan"), "std_ari": float("nan"), "pairs": 0.0}
        else:
            out[K] = {
                "mean_ari": float(np.mean(aris)),
                "std_ari": float(np.std(aris)),
                "pairs": float(len(aris)),
            }

    return out
