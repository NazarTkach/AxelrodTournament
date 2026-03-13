from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type
import random
import axelrod as axl

from environment import EnvConfig, IPDEnv
from payoff import PDPayoff
from base import StrategyAdapter  # flat import if strategies/ is in PYTHONPATH

Action = axl.Action


@dataclass
class MatchResult:
    actions: List[Tuple[Action, Action]]          # (a, b) after noise
    raw_actions: List[Tuple[Action, Action]]      # (a_raw, b_raw) before noise
    rewards: List[Tuple[float, float]]
    total: Tuple[float, float]
    rounds: int

    def score_per_move(self) -> Tuple[float, float]:
        if self.rounds == 0:
            return 0.0, 0.0
        return self.total[0] / self.rounds, self.total[1] / self.rounds


def play_match(
    strat_a: Type[axl.Player],
    strat_b: Type[axl.Player],
    cfg: EnvConfig,
    payoff: Optional[PDPayoff] = None,
    seed: int = 0,
) -> MatchResult:
    """
    Runs one IPD match with geometric termination and optional trembling-hand noise.
    """
    payoff = payoff or PDPayoff()
    env = IPDEnv(cfg=cfg, payoff=payoff)
    env.reset(seed=seed)

    A = StrategyAdapter(strat_a, seed=seed)
    B = StrategyAdapter(strat_b, seed=seed + 99991)

    actions: List[Tuple[Action, Action]] = []
    raw_actions: List[Tuple[Action, Action]] = []
    rewards: List[Tuple[float, float]] = []

    total_a = 0.0
    total_b = 0.0

    done = False
    while not done:
        a_raw = A.act()
        b_raw = B.act()

        obs, (r_a, r_b), done, info = env.step(a_raw, b_raw)

        a = info["a"]
        b = info["b"]

        # update both players with realized actions (after noise!)
        A.record(a, b)
        B.record(b, a)

        raw_actions.append((a_raw, b_raw))
        actions.append((a, b))
        rewards.append((r_a, r_b))

        total_a += r_a
        total_b += r_b

    return MatchResult(
        actions=actions,
        raw_actions=raw_actions,
        rewards=rewards,
        total=(total_a, total_b),
        rounds=len(actions),
    )
