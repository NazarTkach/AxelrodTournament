import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass
class VectorizeResult:
    names: List[str]
    keys: List[str]
    X: np.ndarray


def load_fingerprints(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_matrix(
    records: List[Dict],
    key_mode: str = "intersection",  # "intersection" or "union"
    fill_value: float = 0.0,
    min_presence: float = 1.0,       # used for union: keep keys present in >= fraction of records
) -> VectorizeResult:
    """
    records: list of dicts produced by fingerprint_strategy(), each with:
      - strategy_name
      - flat: {feature_key: value}

    key_mode:
      - intersection: keep keys present in all records (safest)
      - union: keep keys present in >= min_presence fraction, fill missing
    """
    names = [r.get("strategy_name", f"strategy_{i}") for i, r in enumerate(records)]
    flats = [r.get("flat", {}) for r in records]

    if not flats:
        return VectorizeResult(names=[], keys=[], X=np.zeros((0, 0), dtype=float))

    key_sets = [set(d.keys()) for d in flats]

    if key_mode == "intersection":
        keys = sorted(set.intersection(*key_sets))
    elif key_mode == "union":
        all_keys = sorted(set.union(*key_sets))
        # keep only sufficiently common keys
        keep = []
        n = len(flats)
        for k in all_keys:
            cnt = sum(1 for d in flats if k in d)
            if cnt / n >= min_presence:
                keep.append(k)
        keys = keep
    else:
        raise ValueError("key_mode must be 'intersection' or 'union'")

    X = np.zeros((len(flats), len(keys)), dtype=float)
    for i, d in enumerate(flats):
        for j, k in enumerate(keys):
            X[i, j] = float(d.get(k, fill_value))

    return VectorizeResult(names=names, keys=keys, X=X)
