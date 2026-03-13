import torch
import torch.nn as nn


class QNetMultiScale(nn.Module):
    """
    DDQN Q-network with:
      - token embedding + positional embedding
      - transformer encoder
      - aux MLP
      - Q head
    Expects dict obs: {"tokens": LongTensor[B,L], "aux": FloatTensor[B,aux_dim]}
    """

    def __init__(
        self,
        *,
        max_len: int,
        aux_dim: int,
        vocab_size: int = 5,       # 0 pad + 1..4
        embed_dim: int = 64,
        ff_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.q_head = nn.Sequential(
            nn.Linear(embed_dim + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward_obs(self, batch):
        if not isinstance(batch, dict):
            raise TypeError(f"QNetMultiScale expects dict obs with keys 'tokens','aux', got {type(batch)}")
        tokens = batch["tokens"]
        aux = batch["aux"]
        # (B,aux_dim) float

        B, L = tokens.shape
        x = self.token_embed(tokens) + self.pos_embed[:, :L, :]

        # Optional attention mask: ignore pads (0)
        pad_mask = (tokens == 0)  # (B,L) True where pad
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # pooling: mean of non-pad tokens
        nonpad = (~pad_mask).float().unsqueeze(-1)  # (B,L,1)
        denom = nonpad.sum(dim=1).clamp(min=1.0)
        seq_repr = (x * nonpad).sum(dim=1) / denom  # (B,embed_dim)

        aux_repr = self.aux_mlp(aux)
        z = torch.cat([seq_repr, aux_repr], dim=1)
        return self.q_head(z)
