from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import random

import axelrod as axl

from environment import EnvConfig
from match import play_match, MatchResult

Action = axl.Action


@dataclass
class BatteryConfig:
    seeds_per_pair: int = 20
    # If you want per-seed new env rng but stable:
    base_seed: int = 0


@dataclass
class PairAggregate:
    a_name: str
    b_name: str
    n: int
    mean_score_per_move_a: float
    mean_score_per_move_b: float
    mean_total_a: float
    mean_total_b: float
    mean_rounds: float


def run_pair_battery(
    strat_a,
    strat_b,
    env_cfg: EnvConfig,
    bat_cfg: BatteryConfig,
) -> Tuple[PairAggregate, List[MatchResult]]:
    """
    Run multiple matches with different seeds for one ordered pair (A vs B).
    `strat_a` and `strat_b` can be Player classes OR PostInitCaller factories.
    """
    results: List[MatchResult] = []

    total_spm_a = 0.0
    total_spm_b = 0.0
    total_a = 0.0
    total_b = 0.0
    total_rounds = 0.0

    for i in range(bat_cfg.seeds_per_pair):
        seed = bat_cfg.base_seed + i
        res = play_match(strat_a, strat_b, cfg=env_cfg, seed=seed)
        results.append(res)

        spm_a, spm_b = res.score_per_move()
        total_spm_a += spm_a
        total_spm_b += spm_b
        total_a += res.total[0]
        total_b += res.total[1]
        total_rounds += res.rounds

    n = bat_cfg.seeds_per_pair
    a_name = getattr(strat_a, "name", None) or getattr(strat_a(), "name", "A") if callable(strat_a) else "A"
    b_name = getattr(strat_b, "name", None) or getattr(strat_b(), "name", "B") if callable(strat_b) else "B"

    agg = PairAggregate(
        a_name=str(a_name),
        b_name=str(b_name),
        n=n,
        mean_score_per_move_a=total_spm_a / n,
        mean_score_per_move_b=total_spm_b / n,
        mean_total_a=total_a / n,
        mean_total_b=total_b / n,
        mean_rounds=total_rounds / n,
    )
    return agg, results


def run_battery_against_list(
    focal,
    opponents: Sequence,
    env_cfg: EnvConfig,
    bat_cfg: BatteryConfig,
) -> List[PairAggregate]:
    """
    Evaluate one focal strategy against many opponents (ordered focal vs opp).
    Returns aggregates only (logs optional; keep logs only when needed).
    """
    aggs: List[PairAggregate] = []
    for opp in opponents:
        agg, _ = run_pair_battery(focal, opp, env_cfg, bat_cfg)
        aggs.append(agg)
    return aggs
