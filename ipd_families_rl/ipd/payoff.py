from dataclasses import dataclass
from typing import Tuple
import axelrod as axl

Action = axl.Action  # C/D


@dataclass(frozen=True)
class PDPayoff:
    """
    Standard Prisoner's Dilemma payoff:
      T > R > P > S and 2R > T+S (for "proper" PD)
    """
    R: float = 3.0  # reward for mutual cooperation
    T: float = 5.0  # temptation to defect
    S: float = 0.0  # sucker's payoff
    P: float = 1.0  # punishment for mutual defection

    def payoff(self, a: Action, b: Action) -> Tuple[float, float]:
        if a == axl.Action.C and b == axl.Action.C:
            return self.R, self.R
        if a == axl.Action.C and b == axl.Action.D:
            return self.S, self.T
        if a == axl.Action.D and b == axl.Action.C:
            return self.T, self.S
        # D, D
        return self.P, self.P
