from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


@lru_cache(maxsize=8192)
def load_binary_mask(path: str, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.NEAREST)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.float32)


@lru_cache(maxsize=8192)
def load_source_heatmap(path: str, image_size: int, sigma: float) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.NEAREST)
    arr = np.asarray(image, dtype=np.uint8)
    ys, xs = np.where(arr > 128)
    if len(xs) == 0:
        return np.zeros((image_size, image_size), dtype=np.float32)

    x = float(xs.mean())
    y = float(ys.mean())
    coords = np.arange(image_size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))
    return heatmap.astype(np.float32)


def resolve_required(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path
