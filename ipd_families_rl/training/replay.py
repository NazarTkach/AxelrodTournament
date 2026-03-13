from dataclasses import dataclass
from typing import Any, List
import random

@dataclass
class Transition:
    s: Any
    a: int
    r: float
    s2: Any
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.data: List[Transition] = []
        self.i = 0

    def add(self, tr: Transition):
        if len(self.data) < self.capacity:
            self.data.append(tr)
        else:
            self.data[self.i] = tr
            self.i = (self.i + 1) % self.capacity

    def sample(self, batch_size: int):
        return self.rng.sample(self.data, batch_size)

    def __len__(self):
        return len(self.data)
