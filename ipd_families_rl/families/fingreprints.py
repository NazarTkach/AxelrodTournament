import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import axelrod as axl

from environment import EnvConfig
from batch import BatteryConfig, run_pair_battery
from strategies.probes import probe_strategy_classes  # if you keep probes in strategies/, adjust path

Action = axl.Action


@dataclass
class FingerprintConfig:
    # Environment settings for fingerprinting should be fixed & reported.
    p_end: float = 0.1
    max_rounds: int = 30
    tremble_eps: float = 0.0

    seeds_per_pair: int = 30
    base_seed: int = 0


def _basic_match_stats(results) -> Dict[str, float]:
    """
    Aggregate interpretable behavior stats from MatchResult logs.
    """
    n = len(results)
    if n == 0:
        return {}

    coop_rate_self = 0.0
    coop_rate_opp = 0.0
    mutual_c = 0.0
    mutual_d = 0.0
    rounds = 0.0

    for res in results:
        rounds += res.rounds
        for (a, b) in res.actions:
            coop_rate_self += 1.0 if a == axl.Action.C else 0.0
            coop_rate_opp += 1.0 if b == axl.Action.C else 0.0
            mutual_c += 1.0 if (a == axl.Action.C and b == axl.Action.C) else 0.0
            mutual_d += 1.0 if (a == axl.Action.D and b == axl.Action.D) else 0.0

    denom = max(rounds, 1.0)
    return {
        "self_coop_rate": coop_rate_self / denom,
        "opp_coop_rate": coop_rate_opp / denom,
        "mutual_C_rate": mutual_c / denom,
        "mutual_D_rate": mutual_d / denom,
        "mean_rounds": rounds / n,
    }


def fingerprint_strategy(strategy_obj, fp_cfg: FingerprintConfig) -> Dict:
    """
    Compute probe-based fingerprint for one strategy (class or factory).
    Returns a dict with per-probe stats and a flattened vector-friendly dict.
    """
    env_cfg = EnvConfig(
        p_end=fp_cfg.p_end,
        max_rounds=fp_cfg.max_rounds,
        tremble_eps=fp_cfg.tremble_eps,
        seed=fp_cfg.base_seed,
    )
    bat_cfg = BatteryConfig(seeds_per_pair=fp_cfg.seeds_per_pair, base_seed=fp_cfg.base_seed)

    probes = probe_strategy_classes()
    per_probe: Dict[str, Dict[str, float]] = {}
    flat: Dict[str, float] = {}

    # name resolution
    try:
        s_name = strategy_obj.name
    except Exception:
        s_name = strategy_obj().name if callable(strategy_obj) else "UNKNOWN"

    for probe_cls in probes:
        agg, logs = run_pair_battery(strategy_obj, probe_cls, env_cfg, bat_cfg)
        stats = _basic_match_stats(logs)

        p_name = probe_cls.name
        per_probe[p_name] = stats

        # Flatten with stable keys for clustering
        for k, v in stats.items():
            flat[f"{p_name}__{k}"] = float(v)

        # Also include payoff-level aggregate (useful for clustering)
        flat[f"{p_name}__score_per_move"] = float(agg.mean_score_per_move_a)

    return {
        "strategy_name": str(s_name),
        "per_probe": per_probe,
        "flat": flat,
        "meta": {
            "p_end": fp_cfg.p_end,
            "max_rounds": fp_cfg.max_rounds,
            "tremble_eps": fp_cfg.tremble_eps,
            "seeds_per_pair": fp_cfg.seeds_per_pair,
        },
    }


def save_fingerprints(records: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
