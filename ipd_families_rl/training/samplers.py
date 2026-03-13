import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable

import axelrod as axl
from base import resolve_strategy  # returns PostInitCaller factory

@dataclass
class OpponentSamplerConfig:
    seed: int = 0
    mode: str = "uniform_family"  # uniform_family | uniform_strategy | curriculum
    curriculum_power: float = 1.0 # >1 biases towards "hard" families when you add hardness scores


class OpponentSampler:
    def __init__(self, catalog_path: str, labels_path: str, cfg: OpponentSamplerConfig):
        self.rng = random.Random(cfg.seed)
        self.cfg = cfg
        self.catalog = json.load(open(catalog_path, "r", encoding="utf-8"))
        self.labels = json.load(open(labels_path, "r", encoding="utf-8"))

        # Build family -> list of strategy qualnames
        fam2q = {}
        for item in self.catalog:
            name = item["name"]
            qn = item["qualname"]
            fam = self.labels.get(name, "UNKNOWN")
            fam2q.setdefault(fam, []).append(qn)

        self.families = sorted(fam2q.keys())
        self.fam2q = fam2q

    def sample(self) -> Tuple[str, Callable]:
        """
        Returns (family_name, opponent_factory).
        """
        if self.cfg.mode == "uniform_family":
            fam = self.rng.choice(self.families)
            qn = self.rng.choice(self.fam2q[fam])
            return fam, resolve_strategy(qn)

        if self.cfg.mode == "uniform_strategy":
            item = self.rng.choice(self.catalog)
            fam = self.labels.get(item["name"], "UNKNOWN")
            return fam, resolve_strategy(item["qualname"])

        raise ValueError(f"Unknown sampling mode: {self.cfg.mode}")
