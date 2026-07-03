#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

import torch

from rmfm.device import add_gpu_argument, device_string


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = (
    Path("/home/DataDisk/zsfang/rmfm/checkpoints/token_unet")
    / "radiomapseer_source_residual_token_unet_flow_dpm"
)
DEFAULT_OUTPUT_DIR = (
    Path("/home/DataDisk/zsfang/rmfm/results/token_unet")
    / "radiomapseer_source_residual_token_unet_flow_dpm_0.00noise"
)
DEFAULT_DATASET_ROOT = Path("/home/DataDisk/zsfang/dataset/RadioMapSeer")
METRIC_KEYS = ("psnr", "ssim", "mse", "nmse", "rmse", "mae")


def ratio_tag(value: float) -> str:
    return f"sr_{value:.4f}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the trained source-residual TokenUNet and save all results to DataDisk."
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split_file", type=Path, default=CHECKPOINT_DIR / "city_splits.json")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sampling_rates",
        type=float,
        nargs="+",
        default=[0.0001, 0.0003, 0.001, 0.003, 0.005, 0.01],
    )
    parser.add_argument(
        "--condition_mode",
        choices=[
            "full",
            "no_building",
            "no_source",
            "building_only",
            "source_only",
            "no_building_source",
            "zero_all",
        ],
        default="full",
    )
    parser.add_argument("--sampler_mode", choices=["no_dc", "dc"], default="dc")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--samples_per_city", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measurement_noise_std", type=float, default=0.00)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--step_size", type=float, default=50.0)
    parser.add_argument("--dc_iters", type=int, default=3)
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="fp32 is the safe default for the source-residual model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    add_gpu_argument(parser)
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_path(path: Path, kind: str) -> None:
    exists = path.is_dir() if kind == "directory" else path.is_file()
    if not exists:
        raise FileNotFoundError(f"Required {kind} not found: {path}")


def resolve_device(args: argparse.Namespace) -> str:
    base_device = args.device
    if base_device is None:
        base_device = "cuda" if torch.cuda.is_available() else "cpu"
    return device_string(base_device, args.gpu)


def summary_path(args: argparse.Namespace, sampling_rate: float) -> Path:
    return (
        args.output_dir
        / f"source_residual_token_unet_{args.sampler_mode}"
        / args.condition_mode
        / "DPM"
        / ratio_tag(sampling_rate)
        / "summary.json"
    )


def build_command(args: argparse.Namespace, sampling_rate: float, device: str) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "sample_token_flowdps.py"),
        "--dataset_root",
        str(args.dataset_root),
        "--checkpoint",
        str(args.checkpoint),
        "--output_dir",
        str(args.output_dir),
        "--gain_mode",
        "DPM",
        "--split",
        "test",
        "--split_file",
        str(args.split_file),
        "--condition_mode",
        args.condition_mode,
        "--sampling_rate",
        str(sampling_rate),
        "--num_samples",
        str(args.num_samples),
        "--samples_per_city",
        str(args.samples_per_city),
        "--start_index",
        str(args.start_index),
        "--seed",
        str(args.seed),
        "--measurement_noise_std",
        str(args.measurement_noise_std),
        "--sampler_mode",
        args.sampler_mode,
        "--num_steps",
        str(args.num_steps),
        "--step_size",
        str(args.step_size),
        "--dc_iters",
        str(args.dc_iters),
        "--dtype",
        args.dtype,
        "--device",
        device,
    ]
    if not args.show_progress:
        command.append("--no_progress")
    return command


def write_combined_summary(args: argparse.Namespace, summaries: list[dict]) -> None:
    fields = ["sampling_rate", "n_frames", "checkpoint", "model_type"]
    fields.extend(f"{key}_{suffix}" for key in METRIC_KEYS for suffix in ("mean", "std"))
    output_path = args.output_dir / "test_summary.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Combined summary: {output_path}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args)
    require_path(args.checkpoint, "file")
    require_path(args.split_file, "file")
    require_path(args.dataset_root, "directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["resolved_device"] = device
    with open(args.output_dir / "test_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=True)

    commands = []
    summaries = []
    for sampling_rate in args.sampling_rates:
        result_path = summary_path(args, sampling_rate)
        command = build_command(args, sampling_rate, device)
        commands.append(shlex.join(command))
        if args.overwrite or not result_path.exists():
            print(f"Testing sampling rate {sampling_rate:g} on {device}")
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        else:
            print(f"Reusing existing result: {result_path}")
        if not result_path.exists():
            raise FileNotFoundError(f"Expected summary not found: {result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["sampling_rate"] = sampling_rate
        summaries.append(summary)

    (args.output_dir / "run_commands.txt").write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )
    write_combined_summary(args, summaries)


if __name__ == "__main__":
    main()
