#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rmfm.device import add_gpu_argument, resolve_torch_device  # noqa: E402
from rmfm.data import (  # noqa: E402
    RadioMapSeerFlowDataset,
    filter_samples_by_city,
    list_radiomapseer_samples,
    select_samples_per_city,
)
from rmfm.flowdps import RadioMapUNetFlowDPS  # noqa: E402
from rmfm.io import save_mask, save_tensor_image, tensor_to_float01  # noqa: E402
from rmfm.masks import (  # noqa: E402
    build_measurement,
    make_exact_k_mask,
    make_exact_ratio_mask,
    mask_to_tensor,
    stable_int_seed,
)
from rmfm.metrics import (  # noqa: E402
    MASKED_METRIC_KEYS,
    compute_masked_metrics,
    compute_metrics,
    summarize_metric_keys,
    write_metrics_csv,
    write_summary_json,
)
from rmfm.paths import DEFAULT_DATASET_ROOT, DEFAULT_UNET_RESULT_ROOT  # noqa: E402


CONDITION_MODES = (
    "full",
    "no_building",
    "no_source",
    "building_only",
    "source_only",
    "no_building_source",
    "zero_all",
)


def ratio_tag(value: float) -> str:
    return f"sr_{value:.4f}".replace(".", "p")


def count_tag(value: int) -> str:
    return f"k_{value:04d}"


def sampling_tag(sampling_rate: float, sampling_count: int | None) -> str:
    if sampling_count is not None:
        return count_tag(sampling_count)
    return ratio_tag(sampling_rate)


def apply_condition_mode(condition: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "full":
        return condition
    if condition.shape[1] < 2:
        raise ValueError(f"Condition ablation requires at least 2 condition channels, got {condition.shape[1]}")

    ablated = condition.clone()
    if mode in ("no_building", "source_only", "no_building_source"):
        ablated[:, 0] = 0.0
    if mode in ("no_source", "building_only", "no_building_source"):
        ablated[:, 1] = 0.0
    if mode == "building_only":
        ablated[:, 2:] = 0.0
    elif mode == "source_only":
        ablated[:, 2:] = 0.0
    elif mode == "zero_all":
        ablated.zero_()
    elif mode not in CONDITION_MODES:
        raise ValueError(f"Unknown condition_mode: {mode}")
    return ablated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sparse RadioMap recovery with a U-Net flow prior.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_UNET_RESULT_ROOT / "sample_flowdps")
    parser.add_argument("--gain_mode", type=str, default="DPM")
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--data_channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--heatmap_sigma", type=float, default=20.0)
    parser.add_argument("--third_channel", choices=["auto", "ones", "zeros", "cars"], default="auto")
    parser.add_argument("--condition_mode", choices=CONDITION_MODES, default="full")

    parser.add_argument("--sampling_rate", type=float, default=0.01)
    parser.add_argument(
        "--sampling_count",
        type=int,
        default=None,
        help="If set, sample exactly this many valid non-building pixels instead of using sampling_rate.",
    )
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--samples_per_city",
        type=int,
        default=0,
        help="If > 0, select this many samples from each city after split filtering.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measurement_noise_std", type=float, default=0.03)

    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--step_size", type=float, default=50.0)
    parser.add_argument("--dc_iters", type=int, default=3)
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
        raise FileNotFoundError(
            f"Split file not found: {split_file}. Pass --split_file or use --split all."
        )
    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    if args.split not in splits:
        raise KeyError(f"Split '{args.split}' not found in {split_file}")
    return filter_samples_by_city(samples, splits[args.split])


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
    sampler = RadioMapUNetFlowDPS(
        checkpoint=args.checkpoint,
        device=device,
        dtype=args.dtype,
    )

    out_root = args.output_dir / args.gain_mode / sampling_tag(args.sampling_rate, args.sampling_count)
    input_dir = out_root / "input"
    recon_dir = out_root / "recon"
    label_dir = out_root / "label"
    mask_dir = out_root / "masks"
    rows = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        image = item["image"].unsqueeze(0).to(device)
        full_condition = item["condition"].unsqueeze(0).to(device)
        condition = apply_condition_mode(full_condition, args.condition_mode)
        sampling_key = (
            f"k={args.sampling_count}"
            if args.sampling_count is not None
            else f"sr={args.sampling_rate:.8f}"
        )
        mask_seed = stable_int_seed(args.seed, item["gain_path"], sampling_key)
        noise_seed = stable_int_seed(args.seed, item["gain_path"], "measurement_noise")
        sample_seed = stable_int_seed(args.seed, item["gain_path"], "flow_sample")

        building_np = full_condition[0, 0].detach().float().cpu().numpy()
        valid_np = building_np <= 0.5
        if args.sampling_count is not None:
            mask_np = make_exact_k_mask(
                args.image_size,
                args.image_size,
                args.sampling_count,
                mask_seed,
                valid_mask=valid_np,
            )
        else:
            mask_np = make_exact_ratio_mask(args.image_size, args.image_size, args.sampling_rate, mask_seed)
        mask = mask_to_tensor(mask_np, device=device)
        measurement = build_measurement(image, mask, args.measurement_noise_std, noise_seed)

        generator = torch.Generator(device=device)
        generator.manual_seed(sample_seed)

        t0 = time.perf_counter()
        recon = sampler.sample(
            measurement=measurement,
            mask=mask,
            condition=condition,
            num_steps=args.num_steps,
            step_size=args.step_size,
            dc_iters=args.dc_iters,
            generator=generator,
            show_progress=not args.no_progress,
        )
        elapsed = time.perf_counter() - t0

        frame = Path(item["gain_path"]).stem
        sparse_display = measurement * mask + (-1.0) * (1.0 - mask)
        save_tensor_image(sparse_display, input_dir / f"{frame}.png")
        save_tensor_image(recon, recon_dir / f"{frame}.png")
        save_tensor_image(image, label_dir / f"{frame}.png")
        save_mask(mask_np, mask_dir / f"{frame}.npy")

        recon_np = tensor_to_float01(recon)
        image_np = tensor_to_float01(image)
        measurement_np = tensor_to_float01(measurement)
        observed_np = (mask_np > 0.5) & valid_np
        unobserved_np = (mask_np <= 0.5) & valid_np
        metrics = compute_metrics(recon_np, image_np)
        metrics.update(compute_masked_metrics(recon_np, image_np, observed_np, "observed"))
        metrics.update(compute_masked_metrics(recon_np, image_np, unobserved_np, "unobserved"))
        metrics.update(compute_masked_metrics(recon_np, measurement_np, observed_np, "measurement"))
        rows.append(
            {
                "frame": frame,
                "mode": item["mode"],
                "city_id": item["city_id"],
                "source_id": item["source_id"],
                "sampling_rate": args.sampling_rate if args.sampling_count is None else "",
                "sampling_count": args.sampling_count if args.sampling_count is not None else "",
                "valid_pixel_count": int(valid_np.sum()),
                "condition_mode": args.condition_mode,
                "time_sec": elapsed,
                **metrics,
            }
        )
        print(
            f"[{idx + 1}/{len(dataset)}] {frame} "
            f"PSNR={metrics['psnr']:.3f} SSIM={metrics['ssim']:.4f} time={elapsed:.2f}s"
        )

    metric_keys = ["psnr", "ssim", "mse", "nmse", "rmse", "mae"] + [f"{prefix}_{key}" for prefix in ("observed", "unobserved", "measurement") for key in MASKED_METRIC_KEYS]
    summary = summarize_metric_keys(rows, metric_keys)
    summary.update(
        {
            "gain_mode": args.gain_mode,
            "sampling_rate": args.sampling_rate if args.sampling_count is None else None,
            "sampling_count": args.sampling_count,
            "checkpoint": str(args.checkpoint),
            "num_steps": args.num_steps,
            "step_size": args.step_size,
            "dc_iters": args.dc_iters,
            "measurement_noise_std": args.measurement_noise_std,
            "condition_mode": args.condition_mode,
            "third_channel": args.third_channel,
            "samples_per_city": args.samples_per_city,
            "n_frames": len(rows),
        }
    )
    write_metrics_csv(rows, out_root / "metrics.csv")
    write_summary_json(summary, out_root / "summary.json")
    print(f"Saved results to {out_root}")


if __name__ == "__main__":
    main()
