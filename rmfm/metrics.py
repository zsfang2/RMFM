from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


METRIC_KEYS = ["psnr", "ssim", "mse", "nmse", "rmse", "mae"]
MASKED_METRIC_KEYS = ["psnr", "mse", "nmse", "rmse", "mae"]


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


def compute_masked_metrics(
    recon: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    if recon.ndim == 2:
        recon = np.repeat(recon[:, :, None], 3, axis=2)
    if label.ndim == 2:
        label = np.repeat(label[:, :, None], 3, axis=2)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != recon.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {recon.shape[:2]}")

    recon = recon.astype(np.float64)
    label = label.astype(np.float64)
    selected = mask.astype(bool)
    result = {f"{prefix}_{key}": float("nan") for key in MASKED_METRIC_KEYS}
    result[f"{prefix}_pixel_count"] = int(selected.sum())
    if not np.any(selected):
        return result

    selected_3c = np.repeat(selected[:, :, None], recon.shape[2], axis=2)
    diff = recon - label
    diff_vals = diff[selected_3c]
    label_vals = label[selected_3c]
    mse = float(np.mean(diff_vals**2))
    result[f"{prefix}_mse"] = mse
    result[f"{prefix}_nmse"] = float(np.sum(diff_vals**2) / (np.sum(label_vals**2) + 1e-12))
    result[f"{prefix}_rmse"] = float(np.sqrt(mse))
    result[f"{prefix}_mae"] = float(np.mean(np.abs(diff_vals)))
    result[f"{prefix}_psnr"] = float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))
    return result


def summarize_metric_keys(rows: list[dict], keys: list[str]) -> dict[str, float]:
    summary: dict[str, float] = {"n_frames": len(rows)}
    for key in keys:
        vals = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if not len(vals):
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_std"] = float("nan")
        elif np.isinf(vals).any():
            if np.isposinf(vals).any() and not np.isneginf(vals).any():
                summary[f"{key}_mean"] = float("inf")
            elif np.isneginf(vals).any() and not np.isposinf(vals).any():
                summary[f"{key}_mean"] = float("-inf")
            else:
                summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_std"] = float("nan")
        else:
            summary[f"{key}_mean"] = float(vals.mean())
            summary[f"{key}_std"] = float(vals.std())
    return summary


def summarize_metrics(rows: list[dict]) -> dict[str, float]:
    return summarize_metric_keys(rows, METRIC_KEYS)


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
