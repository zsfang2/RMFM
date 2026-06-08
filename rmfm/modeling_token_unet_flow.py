from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from diffusers import UNet2DConditionModel

from .obs_tokenizer import SparseObservationTokenizer


def build_token_unet_config(
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
) -> dict[str, Any]:
    block_out_channels = tuple(base_channels * mult for mult in channel_mults)
    n_blocks = len(block_out_channels)
    if n_blocks < 2:
        raise ValueError("TokenUNet needs at least two resolution blocks")

    down_block_types = ["DownBlock2D"]
    down_block_types.extend("CrossAttnDownBlock2D" for _ in range(n_blocks - 1))
    up_block_types = ["CrossAttnUpBlock2D" for _ in range(n_blocks - 1)]
    up_block_types.append("UpBlock2D")

    return {
        "image_size": image_size,
        "data_channels": data_channels,
        "cond_channels": cond_channels,
        "token_dim": token_dim,
        "tokenizer_layers": tokenizer_layers,
        "tokenizer_heads": tokenizer_heads,
        "fourier_bands": fourier_bands,
        "unet": {
            "sample_size": image_size,
            "in_channels": data_channels + cond_channels,
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


class TokenUNetFlowModel(torch.nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.data_channels = int(config["data_channels"])
        self.cond_channels = int(config["cond_channels"])
        self.tokenizer = SparseObservationTokenizer(
            token_dim=int(config["token_dim"]),
            fourier_bands=int(config["fourier_bands"]),
            transformer_layers=int(config["tokenizer_layers"]),
            heads=int(config["tokenizer_heads"]),
        )
        self.unet = UNet2DConditionModel(**config["unet"])

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dense_condition: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        if dense_condition.shape[1] < self.cond_channels:
            pad = dense_condition.new_zeros(
                (
                    dense_condition.shape[0],
                    self.cond_channels - dense_condition.shape[1],
                    dense_condition.shape[2],
                    dense_condition.shape[3],
                )
            )
            dense_condition = torch.cat([dense_condition, pad], dim=1)
        dense_condition = dense_condition[:, : self.cond_channels]
        model_input = torch.cat([x, dense_condition.to(dtype=x.dtype)], dim=1)
        tokens, token_mask = self.tokenizer(
            obs_coords.to(dtype=x.dtype),
            obs_values.to(dtype=x.dtype),
            obs_mask,
        )
        return self.unet(
            model_input,
            t,
            encoder_hidden_states=tokens,
            encoder_attention_mask=token_mask,
        ).sample


def create_token_unet_flow_model(config: dict[str, Any]) -> TokenUNetFlowModel:
    return TokenUNetFlowModel(config)


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
        "model_type": "token_unet_flow",
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
) -> tuple[TokenUNetFlowModel, dict[str, Any], dict[str, Any]]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    config = checkpoint["model_config"]
    model = create_token_unet_flow_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, config, checkpoint
