import numpy as np
import torch
import torch.nn.functional as F


def _stack_obs(obs_list, device):
    """
    Supports:
      - numpy arrays (vector or token seq)
      - dict obs: {"tokens": np.int64[L], "aux": np.float32[d]}
    Returns torch tensor or dict of torch tensors.
    """
    first = obs_list[0]

    if isinstance(first, dict):
        tokens = np.stack([o["tokens"] for o in obs_list], axis=0)
        aux = np.stack([o["aux"] for o in obs_list], axis=0)
        return {
            "tokens": torch.as_tensor(tokens, dtype=torch.long, device=device),
            "aux": torch.as_tensor(aux, dtype=torch.float32, device=device),
        }

    # array-like
    arr = np.stack(obs_list, axis=0)  # avoids PyTorch slow list-of-arrays warning
    if np.issubdtype(arr.dtype, np.integer):
        return torch.as_tensor(arr, dtype=torch.long, device=device)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def ddqn_loss(model_online, model_target, batch, gamma: float, device: str = "cpu"):
    S = _stack_obs([b.s for b in batch], device)
    S2 = _stack_obs([b.s2 for b in batch], device)

    a = torch.as_tensor([b.a for b in batch], dtype=torch.int64, device=device)
    r = torch.as_tensor([b.r for b in batch], dtype=torch.float32, device=device)
    done = torch.as_tensor([b.done for b in batch], dtype=torch.float32, device=device)

    q_sa = model_online.forward_obs(S).gather(1, a.view(-1, 1)).squeeze(1)

    with torch.no_grad():
        a2 = torch.argmax(model_online.forward_obs(S2), dim=1)
        q_s2 = model_target.forward_obs(S2).gather(1, a2.view(-1, 1)).squeeze(1)
        y = r + gamma * (1.0 - done) * q_s2

    return F.smooth_l1_loss(q_sa, y)
