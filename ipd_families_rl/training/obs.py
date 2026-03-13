import numpy as np
import axelrod as axl
from typing import List, Tuple

Action = axl.Action

def encode_action(a: Action) -> int:
    return 1 if a == axl.Action.C else 0

def make_multiscale_obs(
    history: List[Tuple[Action, Action]],
    max_len: int = 200,
    windows = (1, 5, 10, 20, 50),
) -> np.ndarray:
    """
    Returns a fixed-size feature vector:
      - last max_len joint actions (padded) as two binary sequences
      - multi-window averages of cooperation rates (self & opp)
    """
    h = history[-max_len:]
    L = len(h)

    self_seq = np.zeros(max_len, dtype=np.float32)
    opp_seq  = np.zeros(max_len, dtype=np.float32)
    for i, (a, b) in enumerate(h):
        self_seq[i] = encode_action(a)
        opp_seq[i]  = encode_action(b)

    feats = []
    # multi-scale averages over last w steps
    for w in windows:
        w = min(w, L) if L > 0 else 0
        if w == 0:
            feats += [0.0, 0.0]
        else:
            chunk = h[-w:]
            sc = sum(1 for a, _ in chunk if a == axl.Action.C) / w
            oc = sum(1 for _, b in chunk if b == axl.Action.C) / w
            feats += [sc, oc]

    # Final vector: sequences + summary features
    return np.concatenate([self_seq, opp_seq, np.array(feats, dtype=np.float32)], axis=0)
