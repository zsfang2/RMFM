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

from rmfm.data import (  # noqa: E402
    RadioMapSeerFlowDataset,
    filter_samples_by_city,
    list_radiomapseer_samples,
    select_samples_per_city,
)
from rmfm.flowdps import RadioMapUNetFlowDPS  # noqa: E402
from rmfm.io import save_mask, save_tensor_image, tensor_to_float01  # noqa: E402
from rmfm.masks import build_measurement, make_exact_ratio_mask, mask_to_tensor, stable_int_seed  # noqa: E402
from rmfm.metrics import compute_metrics, summarize_metrics, write_metrics_csv, write_summary_json  # noqa: E402
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
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

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

    out_root = args.output_dir / args.gain_mode / ratio_tag(args.sampling_rate)
    input_dir = out_root / "input"
    recon_dir = out_root / "recon"
    label_dir = out_root / "label"
    mask_dir = out_root / "masks"
    rows = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        image = item["image"].unsqueeze(0).to(device)
        condition = item["condition"].unsqueeze(0).to(device)
        condition = apply_condition_mode(condition, args.condition_mode)
        mask_seed = stable_int_seed(args.seed, item["gain_path"], f"{args.sampling_rate:.8f}")
        noise_seed = stable_int_seed(args.seed, item["gain_path"], "measurement_noise")
        sample_seed = stable_int_seed(args.seed, item["gain_path"], "flow_sample")

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

        metrics = compute_metrics(tensor_to_float01(recon), tensor_to_float01(image))
        rows.append(
            {
                "frame": frame,
                "mode": item["mode"],
                "city_id": item["city_id"],
                "source_id": item["source_id"],
                "sampling_rate": args.sampling_rate,
                "condition_mode": args.condition_mode,
                "time_sec": elapsed,
                **metrics,
            }
        )
        print(
            f"[{idx + 1}/{len(dataset)}] {frame} "
            f"PSNR={metrics['psnr']:.3f} SSIM={metrics['ssim']:.4f} time={elapsed:.2f}s"
        )

    summary = summarize_metrics(rows)
    summary.update(
        {
            "gain_mode": args.gain_mode,
            "sampling_rate": args.sampling_rate,
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
