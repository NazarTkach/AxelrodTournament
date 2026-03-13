import numpy as np
import axelrod as axl
from typing import List, Tuple

Action = axl.Action

# Joint action encoding:
# CC=0, CD=1, DC=2, DD=3
def encode_joint(a: Action, b: Action) -> int:
    if a == axl.Action.C and b == axl.Action.C:
        return 0
    if a == axl.Action.C and b == axl.Action.D:
        return 1
    if a == axl.Action.D and b == axl.Action.C:
        return 2
    return 3


def make_sequence_obs(
    history: List[Tuple[Action, Action]],
    max_len: int = 200,
) -> np.ndarray:
    """
    Returns a fixed-length sequence of joint-action tokens.
    Shape: (max_len,)
    Padding value: -1
    """
    seq = np.zeros(max_len, dtype=np.int64)  # 0 = PAD

    h = history[-max_len:]
    for i, (a, b) in enumerate(h):
        seq[i] = encode_joint(a, b) + 1          # shift to 1..4

    return seq
