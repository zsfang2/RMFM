from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


METRIC_KEYS = ["psnr", "ssim", "mse", "nmse", "rmse", "mae"]


def compute_metrics(recon: np.ndarray, label: np.ndarray) -> dict[str, float]:
    if recon.ndim == 2:
        recon = np.repeat(recon[:, :, None], 3, axis=2)
    if label.ndim == 2:
        label = np.repeat(label[:, :, None], 3, axis=2)
    recon = recon.astype(np.float64)
    label = label.astype(np.float64)
    diff = recon - label
    mse = float(np.mean(diff**2))
    nmse = float(np.sum(diff**2) / (np.sum(label**2) + 1e-12))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    psnr = float(psnr_fn(label, recon, data_range=1.0))
    ssim = float(ssim_fn(label, recon, data_range=1.0, channel_axis=2))
    return {"psnr": psnr, "ssim": ssim, "mse": mse, "nmse": nmse, "rmse": rmse, "mae": mae}


def summarize_metrics(rows: list[dict]) -> dict[str, float]:
    summary: dict[str, float] = {"n_frames": len(rows)}
    for key in METRIC_KEYS:
        vals = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        summary[f"{key}_std"] = float(vals.std()) if len(vals) else float("nan")
    return summary


def write_metrics_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = ["frame", "mode", "city_id", "source_id", "sampling_rate", "condition_mode", "time_sec"]
    extra = sorted({key for row in rows for key in row} - set(preferred) - set(METRIC_KEYS))
    fields = [key for key in preferred if any(key in row for row in rows)] + extra + METRIC_KEYS
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
