from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from diffusers import UNet2DModel


def build_unet_config(
    image_size: int = 256,
    data_channels: int = 3,
    cond_channels: int = 3,
    base_channels: int = 64,
    channel_mults: tuple[int, ...] = (1, 2, 4, 4),
    layers_per_block: int = 2,
    norm_num_groups: int = 32,
) -> dict[str, Any]:
    block_out_channels = tuple(base_channels * mult for mult in channel_mults)
    n_blocks = len(block_out_channels)
    return {
        "sample_size": image_size,
        "in_channels": data_channels + cond_channels,
        "out_channels": data_channels,
        "layers_per_block": layers_per_block,
        "block_out_channels": block_out_channels,
        "down_block_types": tuple("DownBlock2D" for _ in range(n_blocks)),
        "up_block_types": tuple("UpBlock2D" for _ in range(n_blocks)),
        "norm_num_groups": norm_num_groups,
        "act_fn": "silu",
    }


def create_unet_flow_model(config: dict[str, Any]) -> UNet2DModel:
    return UNet2DModel(**config)


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
) -> tuple[UNet2DModel, dict[str, Any], dict[str, Any]]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    config = checkpoint["model_config"]
    model = create_unet_flow_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, config, checkpoint
