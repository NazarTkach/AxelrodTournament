import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def compute_family_stats(
    names: List[str],
    labels: np.ndarray,
    X: np.ndarray,
    keys: List[str],
    top_features: int = 8,
) -> Dict:
    """
    Per-family:
      - size
      - feature means
      - top distinguishing features (by abs z-score between family mean and global mean)
    """
    out = {"families": {}}

    global_mean = X.mean(axis=0) if X.size else np.array([])
    global_std = X.std(axis=0) + 1e-12 if X.size else np.array([])

    for cid in sorted(set(int(x) for x in labels)):
        idx = np.where(labels == cid)[0]
        Xc = X[idx]
        mean = Xc.mean(axis=0) if Xc.size else np.array([])
        z = (mean - global_mean) / global_std if X.size else np.array([])

        # top abs-z features
        if z.size:
            order = np.argsort(-np.abs(z))[:top_features]
            top = [(keys[int(j)], float(mean[int(j)]), float(z[int(j)])) for j in order]
        else:
            top = []

        out["families"][str(cid)] = {
            "size": int(idx.size),
            "members": [names[int(i)] for i in idx[:50]],  # cap for readability
            "top_features": top,
        }

    return out


def save_family_stats(stats: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
