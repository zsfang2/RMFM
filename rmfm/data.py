from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .heatmap import load_binary_mask, load_source_heatmap, resolve_required


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass(frozen=True)
class RadioMapSeerSample:
    gain_path: Path
    antenna_path: Path
    building_path: Path
    car_path: Path | None
    mode: str
    city_id: str
    source_id: str


def parse_city_source(path: Path) -> tuple[str, str]:
    parts = path.stem.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected file name like city_source.png, got {path.name}")
    return parts[0], parts[1]


def list_radiomapseer_samples(
    dataset_root: Path,
    gain_modes: Iterable[str],
    require_car: bool = False,
) -> list[RadioMapSeerSample]:
    samples: list[RadioMapSeerSample] = []
    for mode in gain_modes:
        gain_dir = dataset_root / "gain" / mode
        if not gain_dir.exists():
            raise FileNotFoundError(f"Gain mode directory not found: {gain_dir}")
        for ext in IMAGE_EXTENSIONS:
            for gain_path in sorted(gain_dir.glob(f"*{ext}")):
                city_id, source_id = parse_city_source(gain_path)
                antenna_path = dataset_root / "png" / "antennas" / gain_path.name
                building_path = dataset_root / "png" / "buildings_complete" / f"{city_id}.png"
                car_path = dataset_root / "png" / "cars" / f"{city_id}.png"
                resolve_required(antenna_path, "Antenna image")
                resolve_required(building_path, "Building mask")
                if require_car:
                    resolve_required(car_path, "Car mask")
                samples.append(
                    RadioMapSeerSample(
                        gain_path=gain_path,
                        antenna_path=antenna_path,
                        building_path=building_path,
                        car_path=car_path if car_path.exists() else None,
                        mode=mode,
                        city_id=city_id,
                        source_id=source_id,
                    )
                )
    return sorted(samples, key=lambda x: (x.mode, int(x.city_id), int(x.source_id)))


def split_city_ids(
    samples: list[RadioMapSeerSample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    city_ids = sorted({sample.city_id for sample in samples}, key=lambda x: int(x))
    rng = random.Random(seed)
    rng.shuffle(city_ids)
    n_total = len(city_ids)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    test_ids = sorted(city_ids[:n_test], key=lambda x: int(x))
    val_ids = sorted(city_ids[n_test : n_test + n_val], key=lambda x: int(x))
    train_ids = sorted(city_ids[n_test + n_val :], key=lambda x: int(x))
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def filter_samples_by_city(
    samples: list[RadioMapSeerSample],
    city_ids: Iterable[str],
    max_samples: int = -1,
) -> list[RadioMapSeerSample]:
    city_set = set(city_ids)
    selected = [sample for sample in samples if sample.city_id in city_set]
    if max_samples > 0:
        selected = selected[:max_samples]
    return selected


def select_samples_per_city(
    samples: list[RadioMapSeerSample],
    samples_per_city: int,
    seed: int,
) -> list[RadioMapSeerSample]:
    if samples_per_city <= 0:
        return samples

    grouped: dict[str, list[RadioMapSeerSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.city_id, []).append(sample)

    selected: list[RadioMapSeerSample] = []
    for city_id in sorted(grouped, key=lambda x: int(x)):
        city_samples = sorted(grouped[city_id], key=lambda x: (x.mode, int(x.source_id)))
        if len(city_samples) <= samples_per_city:
            selected.extend(city_samples)
            continue

        rng = random.Random(f"{seed}:{city_id}")
        indices = sorted(rng.sample(range(len(city_samples)), samples_per_city))
        selected.extend(city_samples[index] for index in indices)
    return selected


class RadioMapSeerFlowDataset(Dataset):
    def __init__(
        self,
        samples: list[RadioMapSeerSample],
        image_size: int = 256,
        data_channels: int = 3,
        heatmap_sigma: float = 20.0,
        third_channel: str = "auto",
    ) -> None:
        if data_channels not in (1, 3):
            raise ValueError("data_channels must be 1 or 3")
        if third_channel not in ("auto", "ones", "zeros", "cars"):
            raise ValueError("third_channel must be one of auto, ones, zeros, cars")
        self.samples = samples
        self.image_size = image_size
        self.data_channels = data_channels
        self.heatmap_sigma = heatmap_sigma
        self.third_channel = third_channel

    def __len__(self) -> int:
        return len(self.samples)

    def _load_target(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        if self.data_channels == 3:
            arr = np.repeat(arr[None, :, :], 3, axis=0)
        else:
            arr = arr[None, :, :]
        return torch.from_numpy(arr.astype(np.float32))

    def _load_condition(self, sample: RadioMapSeerSample) -> torch.Tensor:
        building = load_binary_mask(str(sample.building_path), self.image_size)
        source = load_source_heatmap(str(sample.antenna_path), self.image_size, self.heatmap_sigma)

        use_car = self.third_channel == "cars" or (
            self.third_channel == "auto" and "cars" in sample.mode.lower()
        )
        if use_car and sample.car_path is not None:
            third = load_binary_mask(str(sample.car_path), self.image_size)
        elif self.third_channel == "zeros":
            third = np.zeros_like(building, dtype=np.float32)
        else:
            third = np.ones_like(building, dtype=np.float32)

        cond = np.stack([building, source, third], axis=0).astype(np.float32)
        return torch.from_numpy(cond)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        return {
            "image": self._load_target(sample.gain_path),
            "condition": self._load_condition(sample),
            "gain_path": str(sample.gain_path),
            "mode": sample.mode,
            "city_id": sample.city_id,
            "source_id": sample.source_id,
        }
