from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().float().cpu()
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.shape[0] == 1:
        arr = tensor[0].numpy()
        arr = (arr + 1.0) * 0.5
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    arr = tensor[:3].permute(1, 2, 0).numpy()
    arr = (arr + 1.0) * 0.5
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def tensor_to_float01(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor_to_uint8_image(tensor).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return arr


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor_to_uint8_image(tensor)
    Image.fromarray(arr).save(path)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask.astype(np.float32))
