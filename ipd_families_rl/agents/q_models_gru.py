import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4,
        embed_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=5, embedding_dim=embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, L) int64 token sequences
        """
        x = self.embed(obs)        # (B, L, E)
        _, h = self.gru(x)         # h: (layers, B, H)
        h_last = h[-1]             # (B, H)
        return self.head(h_last)
