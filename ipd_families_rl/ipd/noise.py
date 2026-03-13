from dataclasses import dataclass
import random
import axelrod as axl

Action = axl.Action


@dataclass
class TremblingHand:
    """
    With probability eps, flip the chosen action C<->D.
    """
    eps: float = 0.0

    def apply(self, action: Action, rng: random.Random) -> Action:
        if self.eps <= 0:
            return action
        if rng.random() < self.eps:
            return axl.Action.D if action == axl.Action.C else axl.Action.C
        return action
