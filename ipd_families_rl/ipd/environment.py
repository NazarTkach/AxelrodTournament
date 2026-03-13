from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import random
import axelrod as axl

from payoff import PDPayoff
from noise import TremblingHand

Action = axl.Action


@dataclass
class EnvConfig:
    p_end: float = 0.1          # termination probability after each round
    max_rounds: int = 30        # hard cap
    tremble_eps: float = 0.0    # action flip prob
    seed: int = 0


class IPDEnv:
    """
    Environment controls:
      - termination (geometric with prob p_end, capped at max_rounds)
      - trembling-hand noise
      - payoff matrix
    It does NOT know about tournaments or learning.
    """

    def __init__(self, cfg: EnvConfig, payoff: Optional[PDPayoff] = None):
        self.cfg = cfg
        self.payoff = payoff or PDPayoff()
        self.rng = random.Random(cfg.seed)
        self.noise = TremblingHand(cfg.tremble_eps)
        self.t = 0

    def reset(self, seed: Optional[int] = None) -> Dict:
        if seed is not None:
            self.rng.seed(seed)
        self.t = 0
        return {"p_end": self.cfg.p_end, "t": self.t, "max_rounds": self.cfg.max_rounds}

    def step(self, a_raw: Action, b_raw: Action) -> Tuple[Dict, Tuple[float, float], bool, Dict]:
        """
        One simultaneous-move step.
        Returns: obs, (r_a, r_b), done, info
        """
        a = self.noise.apply(a_raw, self.rng)
        b = self.noise.apply(b_raw, self.rng)

        r_a, r_b = self.payoff.payoff(a, b)

        self.t += 1
        done = False

        # terminate after applying rewards
        if self.t >= self.cfg.max_rounds:
            done = True
        else:
            if self.rng.random() < self.cfg.p_end:
                done = True

        obs = {"p_end": self.cfg.p_end, "t": self.t, "max_rounds": self.cfg.max_rounds}
        info = {"a_raw": a_raw, "b_raw": b_raw, "a": a, "b": b}
        return obs, (r_a, r_b), done, info
