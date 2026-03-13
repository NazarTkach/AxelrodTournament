import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import RobustScaler


@dataclass
class ClusterConfig:
    method: str = "average"     # average is fine with cosine
    metric: str = "cosine"      # cosine or correlation are typical
    scaler: str = "robust"      # robust scaling to reduce outlier effects


def scale_features(X: np.ndarray, scaler: str = "robust") -> np.ndarray:
    if X.size == 0:
        return X
    if scaler == "robust":
        return RobustScaler().fit_transform(X)
    if scaler == "none":
        return X
    raise ValueError("scaler must be 'robust' or 'none'")


def hierarchical_cluster_labels(
    X: np.ndarray,
    K: int,
    cfg: ClusterConfig,
) -> np.ndarray:
    """
    Returns integer cluster labels in {1..K}.
    """
    Xs = scale_features(X, cfg.scaler)
    Z = linkage(Xs, method=cfg.method, metric=cfg.metric)
    labels = fcluster(Z, t=K, criterion="maxclust")
    return labels


def labels_to_families(labels: np.ndarray) -> Dict[int, str]:
    """
    Map cluster id -> family name (placeholder names).
    You can rename later based on stats.
    """
    uniq = sorted(set(int(x) for x in labels))
    return {cid: f"Family_{i+1}" for i, cid in enumerate(uniq)}


def export_family_labels(
    names: List[str],
    labels: np.ndarray,
    out_path: str,
    family_names: Optional[Dict[int, str]] = None,
) -> None:
    if family_names is None:
        family_names = labels_to_families(labels)

    mapping = {name: family_names[int(lbl)] for name, lbl in zip(names, labels)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
