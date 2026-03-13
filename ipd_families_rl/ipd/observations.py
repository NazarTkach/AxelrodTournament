import numpy as np
import axelrod as axl
from typing import List, Tuple, Sequence

Action = axl.Action

# Joint actions: CC, CD, DC, DD -> 0..3
def _encode_joint(a: Action, b: Action) -> int:
    if a == axl.Action.C and b == axl.Action.C:
        return 0
    if a == axl.Action.C and b == axl.Action.D:
        return 1
    if a == axl.Action.D and b == axl.Action.C:
        return 2
    return 3


def make_tokens(history: List[Tuple[Action, Action]], max_len: int) -> np.ndarray:
    """
    Returns (L,) int64 tokens.
    0 = PAD, 1..4 = (CC,CD,DC,DD)+1
    """
    tokens = np.zeros(max_len, dtype=np.int64)
    h = history[-max_len:]
    for i, (a, b) in enumerate(h):
        tokens[i] = _encode_joint(a, b) + 1
    return tokens


def _rates(window: Sequence[Tuple[Action, Action]]) -> np.ndarray:
    if len(window) == 0:
        return np.zeros(4, dtype=np.float32)
    n = float(len(window))
    cc = sum(1 for a, b in window if a == axl.Action.C and b == axl.Action.C) / n
    cd = sum(1 for a, b in window if a == axl.Action.C and b == axl.Action.D) / n
    dc = sum(1 for a, b in window if a == axl.Action.D and b == axl.Action.C) / n
    dd = sum(1 for a, b in window if a == axl.Action.D and b == axl.Action.D) / n
    return np.array([cc, cd, dc, dd], dtype=np.float32)


def make_aux(history: List[Tuple[Action, Action]], windows=(5, 10, 20, 50)) -> np.ndarray:
    """
    Returns float32 aux vector.
    For each window: [cc,cd,dc,dd] + global [cc,cd,dc,dd]
    aux_dim = 4*len(windows) + 4
    """
    feats = []
    for w in windows:
        feats.append(_rates(history[-w:]))
    feats.append(_rates(history))
    return np.concatenate(feats, axis=0).astype(np.float32)


def make_tokens_aux(history: List[Tuple[Action, Action]], max_len: int, windows=(5, 10, 20, 50)):
    return {
        "tokens": make_tokens(history, max_len=max_len),
        "aux": make_aux(history, windows=windows),
    }
