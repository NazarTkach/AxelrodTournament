import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetMLP(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, obs_dim)
        returns: (B, 2)
        """
        return self.net(obs)

    def forward(self, x):
        return self.forward_obs(x)
