import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),  # C/D
        )

    def forward(self, x):
        return self.net(x)


def ddqn_loss(q_online, q_target, batch, gamma: float):
    s = torch.tensor([b.s for b in batch], dtype=torch.float32)
    a = torch.tensor([b.a for b in batch], dtype=torch.int64)
    r = torch.tensor([b.r for b in batch], dtype=torch.float32)
    s2 = torch.tensor([b.s2 for b in batch], dtype=torch.float32)
    done = torch.tensor([b.done for b in batch], dtype=torch.float32)

    q = q_online(s).gather(1, a.view(-1, 1)).squeeze(1)

    with torch.no_grad():
        a2 = torch.argmax(q_online(s2), dim=1)
        q2 = q_target(s2).gather(1, a2.view(-1, 1)).squeeze(1)
        y = r + gamma * (1.0 - done) * q2

    return F.smooth_l1_loss(q, y)
