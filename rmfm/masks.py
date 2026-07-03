from __future__ import annotations

import hashlib

import numpy as np
import torch


def stable_int_seed(base_seed: int, *parts: str) -> int:
    h = hashlib.sha256()
    h.update(str(base_seed).encode("utf-8"))
    for part in parts:
        h.update(b"::")
        h.update(part.encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little", signed=False) % (2**32)


def make_exact_ratio_mask(
    height: int,
    width: int,
    sampling_rate: float,
    seed: int,
) -> np.ndarray:
    if not (0.0 <= sampling_rate <= 1.0):
        raise ValueError(f"sampling_rate must be in [0, 1], got {sampling_rate}")
    total = height * width
    n_known = int(round(total * sampling_rate))
    if n_known == 0:
        return np.zeros((height, width), dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=n_known, replace=False)
    mask = np.zeros(total, dtype=np.float32)
    mask[idx] = 1.0
    return mask.reshape(height, width)


def make_exact_k_mask(
    height: int,
    width: int,
    k: int,
    seed: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if valid_mask is None:
        candidates = np.arange(height * width)
    else:
        if valid_mask.shape != (height, width):
            raise ValueError(
                f"valid_mask shape {valid_mask.shape} does not match {(height, width)}"
            )
        candidates = np.flatnonzero(valid_mask.astype(bool).reshape(-1))

    if k > len(candidates):
        raise ValueError(f"Cannot sample k={k} points from only {len(candidates)} valid pixels")

    mask = np.zeros(height * width, dtype=np.float32)
    if k == 0:
        return mask.reshape(height, width)

    rng = np.random.default_rng(seed)
    selected = rng.choice(candidates, size=k, replace=False)
    mask[selected] = 1.0
    return mask.reshape(height, width)


def mask_to_tensor(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    return torch.from_numpy(mask.astype(np.float32))[None, None].to(device)


def build_measurement(
    image: torch.Tensor,
    mask: torch.Tensor,
    noise_std: float,
    seed: int,
) -> torch.Tensor:
    measurement = image * mask
    if noise_std > 0:
        generator = torch.Generator(device=image.device)
        generator.manual_seed(seed)
        noise = torch.randn(image.shape, device=image.device, generator=generator, dtype=image.dtype)
        measurement = measurement + noise_std * noise * mask
    return measurement.clamp(-1.0, 1.0)
