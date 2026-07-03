from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from torch import nn


TX_RELATIVE_CHANNELS = 5
SOURCE_RESIDUAL_TOKEN_FEATURES = 10


def _norm_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PriorNet(nn.Module):
    """Predict a source-aware coarse propagation prior from building and Tx geometry."""

    def __init__(
        self,
        in_channels: int = 2 + TX_RELATIVE_CHANNELS,
        out_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.enc2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.mid = ConvBlock(base_channels * 4, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, building: torch.Tensor, source: torch.Tensor, tx_maps: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = torch.cat([building, source, tx_maps], dim=1)
        f1 = self.enc1(x)
        f2 = self.enc2(self.down1(f1))
        f3 = self.mid(self.down2(f2))
        u2 = self.up2(f3)
        if u2.shape[-2:] != f2.shape[-2:]:
            u2 = F.interpolate(u2, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, f2], dim=1))
        u1 = self.up1(d2)
        if u1.shape[-2:] != f1.shape[-2:]:
            u1 = F.interpolate(u1, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, f1], dim=1))
        return self.out(d1), [f1, f2, f3]


class SingularityNet(nn.Module):
    """Lightweight physics-structure branch for propagation-sensitive regions."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        building: torch.Tensor,
        source: torch.Tensor,
        tx_maps: torch.Tensor,
        prior_map: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([building, source, tx_maps, prior_map], dim=1))


class SourceResidualObservationTokenizer(nn.Module):
    """Tokenize sparse measurements as residual evidence relative to the prior map."""

    def __init__(
        self,
        token_dim: int = 256,
        fourier_bands: int = 32,
        feature_dim: int = SOURCE_RESIDUAL_TOKEN_FEATURES,
        transformer_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.fourier_bands = fourier_bands
        self.feature_dim = feature_dim
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
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, token_dim),
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
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if coords.dim() != 3 or coords.shape[-1] != 2:
            raise ValueError(f"coords must have shape [B, K, 2], got {tuple(coords.shape)}")
        if features.dim() != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [B, K, {self.feature_dim}], got {tuple(features.shape)}"
            )
        if mask.dim() != 2:
            raise ValueError(f"mask must have shape [B, K], got {tuple(mask.shape)}")

        mask = mask.bool()
        point_tokens = self.coord_mlp(self.fourier_encode(coords)) + self.feature_mlp(features)
        point_tokens = point_tokens + self.type_embed[:, 1:2]
        point_tokens = point_tokens * mask.unsqueeze(-1).to(dtype=point_tokens.dtype)

        cls = self.cls_token.to(dtype=point_tokens.dtype).expand(coords.shape[0], -1, -1)
        cls = cls + self.type_embed[:, 0:1].to(dtype=point_tokens.dtype)
        tokens = torch.cat([cls, point_tokens], dim=1)
        obs_mask = torch.cat(
            [torch.ones(mask.shape[0], 1, device=mask.device, dtype=torch.bool), mask],
            dim=1,
        )

        tokens = self.encoder(tokens, src_key_padding_mask=~obs_mask)
        tokens = self.norm(tokens)
        tokens = tokens * obs_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        return tokens, obs_mask


def build_tx_relative_maps(source_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return rel-x, rel-y, normalized distance, sin(angle), cos(angle)."""

    if source_map.dim() != 4 or source_map.shape[1] != 1:
        raise ValueError(f"source_map must have shape [B, 1, H, W], got {tuple(source_map.shape)}")

    batch, _, height, width = source_map.shape
    dtype = source_map.dtype
    device = source_map.device
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.view(1, 1, height, width)
    yy = yy.view(1, 1, height, width)

    weights = source_map.clamp_min(0.0)
    denom = weights.flatten(1).sum(dim=1).view(batch, 1, 1, 1)
    center_x = (weights * xx).flatten(1).sum(dim=1).view(batch, 1, 1, 1)
    center_y = (weights * yy).flatten(1).sum(dim=1).view(batch, 1, 1, 1)
    center_x = (center_x + eps * 0.5) / (denom + eps)
    center_y = (center_y + eps * 0.5) / (denom + eps)

    rel_x = xx - center_x
    rel_y = yy - center_y
    dist = torch.sqrt(rel_x.square() + rel_y.square() + eps)
    dist_norm = dist / math.sqrt(2.0)
    angle_sin = rel_y / dist.clamp_min(eps)
    angle_cos = rel_x / dist.clamp_min(eps)
    return torch.cat([rel_x, rel_y, dist_norm, angle_sin, angle_cos], dim=1)


def sample_dense_features(feature_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if feature_map.dim() != 4:
        raise ValueError(f"feature_map must have shape [B, C, H, W], got {tuple(feature_map.shape)}")
    if coords.dim() != 3 or coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape [B, K, 2], got {tuple(coords.shape)}")
    if feature_map.shape[0] != coords.shape[0]:
        raise ValueError("feature_map and coords must have the same batch size")

    batch, channels, _, _ = feature_map.shape
    k = coords.shape[1]
    if k == 0:
        return feature_map.new_zeros((batch, 0, channels))

    grid = coords.clamp(0.0, 1.0).mul(2.0).sub(1.0)
    grid = grid.view(batch, k, 1, 2)
    sampled = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.squeeze(-1).transpose(1, 2)


def build_source_residual_token_unet_config(
    image_size: int = 256,
    data_channels: int = 3,
    cond_channels: int = 2,
    base_channels: int = 64,
    channel_mults: tuple[int, ...] = (1, 2, 4, 4),
    layers_per_block: int = 2,
    norm_num_groups: int = 32,
    token_dim: int = 256,
    tokenizer_layers: int = 2,
    tokenizer_heads: int = 4,
    fourier_bands: int = 32,
    attention_head_dim: int = 8,
    transformer_layers_per_block: int = 1,
    prior_base_channels: int = 32,
    singularity_base_channels: int = 32,
) -> dict[str, Any]:
    block_out_channels = tuple(base_channels * mult for mult in channel_mults)
    n_blocks = len(block_out_channels)
    if n_blocks < 2:
        raise ValueError("SourceResidualTokenUNet needs at least two resolution blocks")

    down_block_types = ["DownBlock2D"]
    down_block_types.extend("CrossAttnDownBlock2D" for _ in range(n_blocks - 1))
    up_block_types = ["CrossAttnUpBlock2D" for _ in range(n_blocks - 1)]
    up_block_types.append("UpBlock2D")

    main_in_channels = data_channels + data_channels + 1 + 2 + TX_RELATIVE_CHANNELS
    return {
        "backbone_version": "source_residual_v2",
        "image_size": image_size,
        "data_channels": data_channels,
        "cond_channels": cond_channels,
        "token_dim": token_dim,
        "tokenizer_layers": tokenizer_layers,
        "tokenizer_heads": tokenizer_heads,
        "fourier_bands": fourier_bands,
        "tx_relative_channels": TX_RELATIVE_CHANNELS,
        "residual_token_feature_dim": SOURCE_RESIDUAL_TOKEN_FEATURES,
        "prior": {
            "in_channels": 2 + TX_RELATIVE_CHANNELS,
            "out_channels": data_channels,
            "base_channels": prior_base_channels,
        },
        "singularity": {
            "in_channels": 2 + TX_RELATIVE_CHANNELS + data_channels,
            "out_channels": 1,
            "base_channels": singularity_base_channels,
        },
        "unet": {
            "sample_size": image_size,
            "in_channels": main_in_channels,
            "out_channels": data_channels,
            "layers_per_block": layers_per_block,
            "block_out_channels": block_out_channels,
            "down_block_types": tuple(down_block_types),
            "mid_block_type": "UNetMidBlock2DCrossAttn",
            "up_block_types": tuple(up_block_types),
            "norm_num_groups": norm_num_groups,
            "act_fn": "silu",
            "cross_attention_dim": token_dim,
            "attention_head_dim": attention_head_dim,
            "transformer_layers_per_block": transformer_layers_per_block,
        },
    }


class SourceResidualTokenUNetFlowModel(nn.Module):
    """Source-conditioned residual flow matching backbone.

    The public forward signature mirrors TokenUNetFlowModel. The caller may keep
    constructing full-state x_t and target velocity as before; the model converts
    x_t to R_t = x_t - M0 internally and predicts residual velocity.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.data_channels = int(config["data_channels"])
        self.cond_channels = int(config.get("cond_channels", 2))
        self.tx_relative_channels = int(config.get("tx_relative_channels", TX_RELATIVE_CHANNELS))
        self.residual_token_feature_dim = int(
            config.get("residual_token_feature_dim", SOURCE_RESIDUAL_TOKEN_FEATURES)
        )

        self.prior_net = PriorNet(**config["prior"])
        self.singularity_net = SingularityNet(**config["singularity"])
        self.tokenizer = SourceResidualObservationTokenizer(
            token_dim=int(config["token_dim"]),
            fourier_bands=int(config["fourier_bands"]),
            feature_dim=self.residual_token_feature_dim,
            transformer_layers=int(config["tokenizer_layers"]),
            heads=int(config["tokenizer_heads"]),
        )
        self.unet = UNet2DConditionModel(**config["unet"])

    def _building_source(self, dense_condition: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if dense_condition.dim() != 4:
            raise ValueError(
                f"dense_condition must have shape [B, C, H, W], got {tuple(dense_condition.shape)}"
            )
        if dense_condition.shape[1] < 2:
            pad = dense_condition.new_zeros(
                (
                    dense_condition.shape[0],
                    2 - dense_condition.shape[1],
                    dense_condition.shape[2],
                    dense_condition.shape[3],
                )
            )
            dense_condition = torch.cat([dense_condition, pad], dim=1)
        dense_condition = dense_condition.to(dtype=dtype)
        return dense_condition[:, 0:1], dense_condition[:, 1:2]

    def compute_prior(
        self,
        dense_condition: torch.Tensor,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if dtype is None:
            dtype = dense_condition.dtype
        building, source = self._building_source(dense_condition, dtype=dtype)
        tx_maps = build_tx_relative_maps(source)
        prior_map, prior_features = self.prior_net(building, source, tx_maps)
        singularity_map = self.singularity_net(building, source, tx_maps, prior_map)
        return building, source, tx_maps, prior_map, singularity_map, prior_features

    def build_residual_observation_features(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        building: torch.Tensor,
        tx_maps: torch.Tensor,
        prior_map: torch.Tensor,
        singularity_map: torch.Tensor,
    ) -> torch.Tensor:
        if obs_values.dim() != 3 or obs_values.shape[-1] != 1:
            raise ValueError(f"obs_values must have shape [B, K, 1], got {tuple(obs_values.shape)}")

        prior_values = sample_dense_features(prior_map[:, :1], obs_coords)
        residual_values = obs_values - prior_values
        singularity_values = sample_dense_features(singularity_map, obs_coords)
        tx_values = sample_dense_features(tx_maps, obs_coords)
        building_values = sample_dense_features(building, obs_coords)
        return torch.cat(
            [
                obs_values,
                prior_values,
                residual_values,
                singularity_values,
                tx_values,
                building_values,
            ],
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dense_condition: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        building, source, tx_maps, prior_map, singularity_map, _ = self.compute_prior(
            dense_condition,
            dtype=x.dtype,
        )
        residual_state = x - prior_map

        obs_coords = obs_coords.to(device=x.device, dtype=x.dtype)
        obs_values = obs_values.to(device=x.device, dtype=x.dtype)
        obs_features = self.build_residual_observation_features(
            obs_coords=obs_coords,
            obs_values=obs_values,
            building=building,
            tx_maps=tx_maps,
            prior_map=prior_map,
            singularity_map=singularity_map,
        )
        tokens, token_mask = self.tokenizer(obs_coords, obs_features, obs_mask.to(device=x.device))

        model_input = torch.cat(
            [
                residual_state,
                prior_map,
                singularity_map,
                building,
                source,
                tx_maps,
            ],
            dim=1,
        )
        pred_v = self.unet(
            model_input,
            t,
            encoder_hidden_states=tokens,
            encoder_attention_mask=token_mask,
        ).sample
        if not return_aux:
            return pred_v
        return {
            "velocity": pred_v,
            "prior_map": prior_map,
            "singularity_map": singularity_map,
            "residual_state": residual_state,
        }


def create_source_residual_token_unet_flow_model(config: dict[str, Any]) -> SourceResidualTokenUNetFlowModel:
    return SourceResidualTokenUNetFlowModel(config)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    model_config: dict[str, Any],
    step: int,
    extra: dict[str, Any] | None = None,
    scaler: torch.amp.GradScaler | None = None,
    rng_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "step": step,
        "model_type": "source_residual_token_unet_flow",
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None and scaler.is_enabled():
        payload["scaler_state_dict"] = scaler.state_dict()
    if rng_state is not None:
        payload["rng_state"] = rng_state
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_from_checkpoint(
    path: Path,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[SourceResidualTokenUNetFlowModel, dict[str, Any], dict[str, Any]]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    config = checkpoint["model_config"]
    model = create_source_residual_token_unet_flow_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, config, checkpoint


build_token_unet_v2_config = build_source_residual_token_unet_config
create_token_unet_v2_model = create_source_residual_token_unet_flow_model
