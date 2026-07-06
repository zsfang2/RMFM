#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rmfm.data import (  # noqa: E402
    RadioMapSeerFlowDataset,
    filter_samples_by_city,
    list_radiomapseer_samples,
    select_samples_per_city,
)
from rmfm.device import add_gpu_argument, resolve_torch_device  # noqa: E402
from rmfm.flowdps import parse_dtype  # noqa: E402
from rmfm.io import save_mask, save_tensor_image, tensor_to_float01  # noqa: E402
from rmfm.masks import build_measurement, make_exact_k_mask, mask_to_tensor, stable_int_seed  # noqa: E402
from rmfm.metrics import (  # noqa: E402
    MASKED_METRIC_KEYS,
    compute_masked_metrics,
    compute_metrics,
    summarize_metric_keys,
    write_metrics_csv,
    write_summary_json,
)
from rmfm.modeling_unet_flow import load_model_from_checkpoint  # noqa: E402
from rmfm.paths import DEFAULT_DATASET_ROOT, DEFAULT_UNET_RESULT_ROOT  # noqa: E402


def count_tag(value: int) -> str:
    return f"k_{value:04d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample C1 Raster-Sparse FM-UNet with exact-K sparse conditions.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_UNET_RESULT_ROOT / "raster_sparse_unet_flow_c1")
    parser.add_argument("--gain_mode", type=str, default="DPM")
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--data_channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--heatmap_sigma", type=float, default=20.0)
    parser.add_argument("--third_channel", choices=["auto", "ones", "zeros", "cars"], default="auto")
    parser.add_argument("--sampling_count", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--samples_per_city", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measurement_noise_std", type=float, default=0.0)
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--solver", choices=["euler", "heun"], default="heun")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    add_gpu_argument(parser)
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def filter_by_split(samples: list, args: argparse.Namespace) -> list:
    if args.split == "all":
        return samples
    split_file = args.split_file
    if split_file is None:
        split_file = args.checkpoint.parent / "city_splits.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}. Pass --split_file or use --split all.")
    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    if args.split not in splits:
        raise KeyError(f"Split '{args.split}' not found in {split_file}")
    return filter_samples_by_city(samples, splits[args.split])


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_memory_stats(device: torch.device) -> dict[str, float | str]:
    if device.type != "cuda":
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "cuda_device_index": index,
        "cuda_device_name": torch.cuda.get_device_name(index),
        "cuda_memory_allocated_mb": torch.cuda.memory_allocated(index) / (1024**2),
        "cuda_memory_reserved_mb": torch.cuda.memory_reserved(index) / (1024**2),
        "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated(index) / (1024**2),
        "cuda_max_memory_reserved_mb": torch.cuda.max_memory_reserved(index) / (1024**2),
    }


class RasterSparseUNetSampler:
    def __init__(self, checkpoint: Path, device: torch.device, dtype: str) -> None:
        self.device = device
        self.model_dtype = parse_dtype(dtype, device)
        self.model, self.model_config, self.checkpoint = load_model_from_checkpoint(
            checkpoint, device=device, dtype=self.model_dtype
        )
        self.data_channels = int(self.model_config["out_channels"])
        self.cond_channels = int(self.model_config["in_channels"]) - self.data_channels
        if self.cond_channels != 3:
            raise ValueError(f"C1 sampler expects 3 condition channels, got {self.cond_channels}")

    @torch.no_grad()
    def predict_velocity(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        model_in = torch.cat([x, condition.to(device=x.device, dtype=x.dtype)], dim=1).to(dtype=self.model_dtype)
        timesteps = (t * 1000.0).to(device=x.device)
        return self.model(model_in, timesteps).sample.float()

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        shape: torch.Size,
        num_steps: int,
        solver: str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=self.device, dtype=torch.float32, generator=generator)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            t_batch = t.expand(x.shape[0])
            v = self.predict_velocity(x, t_batch, condition)
            if solver == "heun" and i < num_steps - 1:
                x_euler = (x + dt * v).clamp(-1.0, 1.0)
                v_next = self.predict_velocity(x_euler, t_next.expand(x.shape[0]), condition)
                x = x + 0.5 * dt * (v + v_next)
            else:
                x = x + dt * v
            x = x.clamp(-1.0, 1.0)
        return x.clamp(-1.0, 1.0)


def main() -> None:
    args = parse_args()
    device = resolve_torch_device(args.device, args.gpu)

    samples = list_radiomapseer_samples(args.dataset_root, [args.gain_mode])
    samples = filter_by_split(samples, args)
    samples = select_samples_per_city(samples, args.samples_per_city, args.seed)
    end_index = None if args.num_samples < 0 else args.start_index + args.num_samples
    samples = samples[args.start_index:end_index]
    if not samples:
        raise RuntimeError("No samples selected.")

    dataset = RadioMapSeerFlowDataset(
        samples,
        image_size=args.image_size,
        data_channels=args.data_channels,
        heatmap_sigma=args.heatmap_sigma,
        third_channel=args.third_channel,
    )
    sampler = RasterSparseUNetSampler(args.checkpoint, device=device, dtype=args.dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    out_root = args.output_dir / args.gain_mode / count_tag(args.sampling_count)
    input_dir = out_root / "input"
    raw_dir = out_root / "raw"
    projected_dir = out_root / "projected"
    label_dir = out_root / "label"
    mask_dir = out_root / "masks"
    rows = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        image = item["image"].unsqueeze(0).to(device)
        dense_condition = item["condition"].unsqueeze(0).to(device)
        building = dense_condition[:, :1]
        valid_np = (building[0, 0].detach().float().cpu().numpy() <= 0.5)

        mask_seed = stable_int_seed(args.seed, item["gain_path"], f"k={args.sampling_count}")
        noise_seed = stable_int_seed(args.seed, item["gain_path"], "measurement_noise")
        sample_seed = stable_int_seed(args.seed, item["gain_path"], "c1_flow_sample")
        mask_np = make_exact_k_mask(args.image_size, args.image_size, args.sampling_count, mask_seed, valid_mask=valid_np)
        mask = mask_to_tensor(mask_np, device=device)
        measurement = build_measurement(image, mask, args.measurement_noise_std, noise_seed)
        sparse_gain = measurement[:, :1] * mask
        condition = torch.cat([building, sparse_gain, mask], dim=1).float()

        generator = torch.Generator(device=device)
        generator.manual_seed(sample_seed)
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        raw = sampler.sample(
            condition=condition,
            shape=image.shape,
            num_steps=args.num_steps,
            solver=args.solver,
            generator=generator,
        )
        synchronize_if_cuda(device)
        elapsed = time.perf_counter() - t0
        projected = raw * (1.0 - mask) + measurement * mask

        frame = Path(item["gain_path"]).stem
        sparse_display = measurement * mask + (-1.0) * (1.0 - mask)
        save_tensor_image(sparse_display, input_dir / f"{frame}.png")
        save_tensor_image(raw, raw_dir / f"{frame}.png")
        save_tensor_image(projected, projected_dir / f"{frame}.png")
        save_tensor_image(image, label_dir / f"{frame}.png")
        save_mask(mask_np, mask_dir / f"{frame}.npy")

        raw_np = tensor_to_float01(raw)
        projected_np = tensor_to_float01(projected)
        image_np = tensor_to_float01(image)
        measurement_np = tensor_to_float01(measurement)
        observed_np = (mask_np > 0.5) & valid_np
        unobserved_np = (mask_np <= 0.5) & valid_np

        metrics = {f"raw_{key}": value for key, value in compute_metrics(raw_np, image_np).items()}
        metrics.update({f"projected_{key}": value for key, value in compute_metrics(projected_np, image_np).items()})
        metrics.update(compute_masked_metrics(raw_np, image_np, observed_np, "raw_observed"))
        metrics.update(compute_masked_metrics(raw_np, image_np, unobserved_np, "raw_unobserved"))
        metrics.update(compute_masked_metrics(raw_np, measurement_np, observed_np, "raw_measurement"))
        metrics.update(compute_masked_metrics(projected_np, image_np, observed_np, "projected_observed"))
        metrics.update(compute_masked_metrics(projected_np, image_np, unobserved_np, "projected_unobserved"))
        metrics.update(compute_masked_metrics(projected_np, measurement_np, observed_np, "projected_measurement"))

        rows.append(
            {
                "frame": frame,
                "mode": item["mode"],
                "city_id": item["city_id"],
                "source_id": item["source_id"],
                "sampling_count": args.sampling_count,
                "valid_pixel_count": int(valid_np.sum()),
                "time_sec": elapsed,
                **cuda_memory_stats(device),
                **metrics,
            }
        )
        print(
            f"[{idx + 1}/{len(dataset)}] {frame} "
            f"raw_unobserved_RMSE={metrics['raw_unobserved_rmse']:.5f} "
            f"time={elapsed:.2f}s"
        )

    metric_keys = [f"raw_{key}" for key in ("psnr", "ssim", "mse", "nmse", "rmse", "mae")]
    metric_keys += [f"projected_{key}" for key in ("psnr", "ssim", "mse", "nmse", "rmse", "mae")]
    metric_keys += [
        f"{prefix}_{key}"
        for prefix in (
            "raw_observed",
            "raw_unobserved",
            "raw_measurement",
            "projected_observed",
            "projected_unobserved",
            "projected_measurement",
        )
        for key in MASKED_METRIC_KEYS
    ]
    metric_keys += ["time_sec"]
    summary = summarize_metric_keys(rows, metric_keys)
    if rows:
        summary["total_time_sec"] = float(sum(float(row["time_sec"]) for row in rows))
    summary.update(cuda_memory_stats(device))
    summary.update(
        {
            "model_type": "raster_sparse_unet_flow_c1",
            "gain_mode": args.gain_mode,
            "sampling_count": args.sampling_count,
            "checkpoint": str(args.checkpoint),
            "num_steps": args.num_steps,
            "solver": args.solver,
            "measurement_noise_std": args.measurement_noise_std,
            "n_frames": len(rows),
        }
    )
    write_metrics_csv(rows, out_root / "metrics.csv")
    write_summary_json(summary, out_root / "summary.json")
    print(f"Saved results to {out_root}")


if __name__ == "__main__":
    main()
