#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


METRIC_KEYS = ["psnr", "ssim", "mse", "nmse", "rmse", "mae"]
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


def condition_output_dir(output_dir: Path, condition_mode: str, condition_modes: list[str]) -> Path:
    if condition_mode == "full" and condition_modes == ["full"]:
        return output_dir
    return output_dir / condition_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark sparse recovery at multiple sampling rates.")
    parser.add_argument("--dataset_root", type=Path, default=Path("/home/DataDisk/zsfang/dataset/RadioMapSeer"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("/home/DataDisk/zsfang/rmfm/results/benchmark"))
    parser.add_argument("--gain_modes", nargs="+", default=["DPM"])
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--sampling_rates", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.03, 0.05])
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--data_channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--heatmap_sigma", type=float, default=20.0)
    parser.add_argument("--third_channel", choices=["auto", "ones", "zeros", "cars"], default="auto")
    parser.add_argument("--condition_modes", choices=CONDITION_MODES, nargs="+", default=["full"])
    parser.add_argument("--measurement_noise_std", type=float, default=0.03)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--step_size", type=float, default=50.0)
    parser.add_argument("--dc_iters", type=int, default=3)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_script = Path(__file__).with_name("sample_flowdps.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for condition_mode in args.condition_modes:
        sample_output_dir = condition_output_dir(args.output_dir, condition_mode, args.condition_modes)
        for gain_mode in args.gain_modes:
            for sampling_rate in args.sampling_rates:
                command = [
                    sys.executable,
                    str(sample_script),
                    "--dataset_root",
                    str(args.dataset_root),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--output_dir",
                    str(sample_output_dir),
                    "--gain_mode",
                    gain_mode,
                    "--split",
                    args.split,
                    "--sampling_rate",
                    str(sampling_rate),
                    "--num_samples",
                    str(args.num_samples),
                    "--start_index",
                    str(args.start_index),
                    "--seed",
                    str(args.seed),
                    "--image_size",
                    str(args.image_size),
                    "--data_channels",
                    str(args.data_channels),
                    "--heatmap_sigma",
                    str(args.heatmap_sigma),
                    "--third_channel",
                    args.third_channel,
                    "--condition_mode",
                    condition_mode,
                    "--measurement_noise_std",
                    str(args.measurement_noise_std),
                    "--num_steps",
                    str(args.num_steps),
                    "--step_size",
                    str(args.step_size),
                    "--dc_iters",
                    str(args.dc_iters),
                    "--dtype",
                    args.dtype,
                    "--device",
                    args.device,
                ]
                if args.split_file is not None:
                    command.extend(["--split_file", str(args.split_file)])
                if args.no_progress:
                    command.append("--no_progress")
                print("Running:", " ".join(command))
                subprocess.run(command, check=True)

                summary_path = sample_output_dir / gain_mode / ratio_tag(sampling_rate) / "summary.json"
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                row = {
                    "condition_mode": condition_mode,
                    "gain_mode": gain_mode,
                    "sampling_rate": sampling_rate,
                    "n_frames": summary.get("n_frames", summary.get("n_frames", "")),
                }
                for key in METRIC_KEYS:
                    row[f"{key}_mean"] = summary.get(f"{key}_mean", "")
                    row[f"{key}_std"] = summary.get(f"{key}_std", "")
                summary_rows.append(row)

    summary_csv = args.output_dir / "benchmark_summary.csv"
    fields = ["condition_mode", "gain_mode", "sampling_rate", "n_frames"] + [
        f"{key}_{suffix}" for key in METRIC_KEYS for suffix in ("mean", "std")
    ]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved benchmark summary to {summary_csv}")


if __name__ == "__main__":
    main()
