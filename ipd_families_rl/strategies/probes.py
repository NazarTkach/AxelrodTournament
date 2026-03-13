# strategies/probes.py
from __future__ import annotations

from typing import List, Type
import axelrod as axl


def probe_strategy_classes() -> List[Type[axl.Player]]:
    """
    A small, diverse probe set used for *agent-independent* fingerprints.
    Keep it stable. Add more later if needed.
    """
    probes: List[Type[axl.Player]] = [
        axl.Cooperator,
        axl.Defector,
        axl.TitForTat,
        axl.Grudger,              # grim trigger-like
        axl.WinStayLoseShift,     # WSLS / Pavlov
        axl.Random,               # stochastic baseline
        axl.Alternator,
        axl.GTFT,                 # Generous TFT
        axl.SuspiciousTitForTat,
        axl.Forgiver,
    ]
    return probes
