import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import axelrod as axl
from base import StrategySpec, build_spec


@dataclass
class CatalogConfig:
    target_size: int = 300
    include_stochastic: bool = True
    include_deterministic: bool = True
    max_memory_depth: Optional[int] = None
    seed: int = 0


def _passes_filters(spec: StrategySpec, cfg: CatalogConfig) -> bool:
    if spec.is_stochastic is not None:
        if spec.is_stochastic and not cfg.include_stochastic:
            return False
        if (spec.is_stochastic is False) and not cfg.include_deterministic:
            return False

    if cfg.max_memory_depth is not None and spec.memory_depth is not None:
        try:
            if spec.memory_depth > cfg.max_memory_depth:
                return False
        except Exception:
            pass

    return True


def build_catalog(cfg: CatalogConfig) -> List[StrategySpec]:
    rng = random.Random(cfg.seed)

    pool = list(axl.all_strategies)  # factories in your version
    specs = []

    for obj in pool:
        s = build_spec(obj)
        if s is not None:
            specs.append(s)

    # Deduplicate by qualname (safe; no module names stored)
    uniq = {}
    for s in specs:
        uniq[s.qualname] = s
    specs = list(uniq.values())

    specs = [s for s in specs if _passes_filters(s, cfg)]

    print(f"[catalog] pool={len(pool)} valid={len(specs)}")

    specs.sort(key=lambda s: (s.name.lower(), s.qualname))
    if len(specs) > cfg.target_size:
        specs = rng.sample(specs, cfg.target_size)
        specs.sort(key=lambda s: s.name.lower())

    return specs


def save_catalog(specs: List[StrategySpec], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in specs], f, indent=2, ensure_ascii=False)

def load_catalog(path: str):
    """
    Load frozen strategy catalog (Layer 1 output).
    Returns list[dict].
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
