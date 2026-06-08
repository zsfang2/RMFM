from __future__ import annotations

import math

import torch
from torch import nn


class SparseObservationTokenizer(nn.Module):
    def __init__(
        self,
        token_dim: int = 256,
        fourier_bands: int = 32,
        transformer_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.fourier_bands = fourier_bands
        self.register_buffer(
            "freq_bands",
            (2.0 ** torch.arange(fourier_bands, dtype=torch.float32)) * math.pi,
            persistent=False,
        )

        coord_dim = 2 + 4 * fourier_bands
        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.type_embed = nn.Parameter(torch.zeros(1, 2, token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.norm = nn.LayerNorm(token_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.type_embed, std=0.02)

    def fourier_encode(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.clamp(0.0, 1.0)
        angles = coords[..., None] * self.freq_bands.to(device=coords.device, dtype=coords.dtype)
        sin = torch.sin(angles).flatten(start_dim=-2)
        cos = torch.cos(angles).flatten(start_dim=-2)
        return torch.cat([coords, sin, cos], dim=-1)

    def forward(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if coords.dim() != 3 or coords.shape[-1] != 2:
            raise ValueError(f"coords must have shape [B, K, 2], got {tuple(coords.shape)}")
        if values.dim() != 3 or values.shape[-1] != 1:
            raise ValueError(f"values must have shape [B, K, 1], got {tuple(values.shape)}")
        if mask.dim() != 2:
            raise ValueError(f"mask must have shape [B, K], got {tuple(mask.shape)}")

        mask = mask.bool()
        point_tokens = self.coord_mlp(self.fourier_encode(coords)) + self.value_mlp(values)
        point_tokens = point_tokens + self.type_embed[:, 1:2]
        point_tokens = point_tokens * mask.unsqueeze(-1).to(dtype=point_tokens.dtype)

        cls = self.cls_token.to(dtype=point_tokens.dtype).expand(coords.shape[0], -1, -1)
        cls = cls + self.type_embed[:, 0:1].to(dtype=point_tokens.dtype)
        tokens = torch.cat([cls, point_tokens], dim=1)
        obs_mask = torch.cat([torch.ones(mask.shape[0], 1, device=mask.device, dtype=torch.bool), mask], dim=1)

        tokens = self.encoder(tokens, src_key_padding_mask=~obs_mask)
        tokens = self.norm(tokens)
        tokens = tokens * obs_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        return tokens, obs_mask
