#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rmfm.device import add_gpu_argument, device_string  # noqa: E402
from rmfm.paths import DEFAULT_DATASET_ROOT, DEFAULT_UNET_RESULT_ROOT  # noqa: E402


METRIC_KEYS = ["psnr", "ssim", "mse", "nmse", "rmse", "mae"]

EXPERIMENT_PRESETS = {
    "sampling_only": {
        "condition_mode": "no_building_source",
        "use_sampling_rates": True,
        "uses_sparse_samples": True,
        "uses_building": False,
        "uses_source": False,
        "uses_third_channel": True,
    },
    "sampling_only_zero": {
        "condition_mode": "zero_all",
        "use_sampling_rates": True,
        "uses_sparse_samples": True,
        "uses_building": False,
        "uses_source": False,
        "uses_third_channel": False,
    },
    "sampling_building": {
        "condition_mode": "no_source",
        "use_sampling_rates": True,
        "uses_sparse_samples": True,
        "uses_building": True,
        "uses_source": False,
        "uses_third_channel": True,
    },
    "sampling_source": {
        "condition_mode": "no_building",
        "use_sampling_rates": True,
        "uses_sparse_samples": True,
        "uses_building": False,
        "uses_source": True,
        "uses_third_channel": True,
    },
    "sampling_building_source": {
        "condition_mode": "full",
        "use_sampling_rates": True,
        "uses_sparse_samples": True,
        "uses_building": True,
        "uses_source": True,
        "uses_third_channel": True,
    },
    "building_source_no_sampling": {
        "condition_mode": "full",
        "use_sampling_rates": False,
        "uses_sparse_samples": False,
        "uses_building": True,
        "uses_source": True,
        "uses_third_channel": True,
    },
}

DEFAULT_EXPERIMENTS = [
    "sampling_only",
    "sampling_building",
    "sampling_source",
    "sampling_building_source",
    "building_source_no_sampling",
]


def ratio_tag(value: float) -> str:
    return f"sr_{value:.4f}".replace(".", "p")


def parse_experiments(names: list[str]) -> list[str]:
    if names == ["all"]:
        return list(EXPERIMENT_PRESETS)
    unknown = [name for name in names if name not in EXPERIMENT_PRESETS]
    if unknown:
        available = ", ".join(["all", *EXPERIMENT_PRESETS])
        raise ValueError(f"Unknown experiments {unknown}. Available: {available}")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark sparse-recovery information ablations for RadioMapSeer."
    )
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_UNET_RESULT_ROOT / "information_ablation")
    parser.add_argument("--gain_modes", nargs="+", default=["DPM"])
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument(
        "--sampling_rates",
        type=float,
        nargs="+",
        default=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.0010],
    )
    parser.add_argument("--sampling_rate_start", type=float, default=None)
    parser.add_argument("--sampling_rate_end", type=float, default=None)
    parser.add_argument("--sampling_rate_step", type=float, default=None)
    parser.add_argument("--no_sampling_rate", type=float, default=0.0)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--samples_per_city",
        type=int,
        default=0,
        help="If > 0, select this many samples from each city after split filtering.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--data_channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--heatmap_sigma", type=float, default=20.0)
    parser.add_argument("--third_channel", choices=["auto", "ones", "zeros", "cars"], default="auto")
    parser.add_argument("--measurement_noise_std", type=float, default=0.03)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--step_size", type=float, default=50.0)
    parser.add_argument("--dc_iters", type=int, default=3)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    add_gpu_argument(parser)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def generate_sampling_rates(start: float, end: float, step: float) -> list[float]:
    start_d = Decimal(str(start))
    end_d = Decimal(str(end))
    step_d = Decimal(str(step))
    if step_d <= 0:
        raise ValueError(f"sampling_rate_step must be > 0, got {step}")
    if start_d > end_d:
        raise ValueError(f"sampling_rate_start must be <= sampling_rate_end, got {start} > {end}")

    values = []
    current = start_d
    while current <= end_d:
        values.append(float(current))
        current += step_d
    return values


def resolve_sampling_rates(args: argparse.Namespace) -> list[float]:
    range_args = [args.sampling_rate_start, args.sampling_rate_end, args.sampling_rate_step]
    if any(value is not None for value in range_args):
        if not all(value is not None for value in range_args):
            raise ValueError(
                "Pass --sampling_rate_start, --sampling_rate_end, and --sampling_rate_step together."
            )
        return generate_sampling_rates(
            args.sampling_rate_start,
            args.sampling_rate_end,
            args.sampling_rate_step,
        )
    return args.sampling_rates


def run_sample(
    args: argparse.Namespace,
    sample_script: Path,
    experiment: str,
    gain_mode: str,
    sampling_rate: float,
    condition_mode: str,
) -> Path:
    experiment_dir = args.output_dir / experiment
    summary_path = experiment_dir / gain_mode / ratio_tag(sampling_rate) / "summary.json"
    if args.skip_existing and summary_path.exists():
        print(f"Reusing {summary_path}")
        return summary_path

    command = [
        sys.executable,
        str(sample_script),
        "--dataset_root",
        str(args.dataset_root),
        "--checkpoint",
        str(args.checkpoint),
        "--output_dir",
        str(experiment_dir),
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
        "--samples_per_city",
        str(args.samples_per_city),
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
        resolved_device,
    ]
    if args.split_file is not None:
        command.extend(["--split_file", str(args.split_file)])
    if args.no_progress:
        command.append("--no_progress")

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary not found: {summary_path}")
    return summary_path


def read_summary_row(
    summary_path: Path,
    experiment: str,
    preset: dict,
    gain_mode: str,
    sampling_rate: float,
) -> dict:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    row = {
        "experiment": experiment,
        "condition_mode": preset["condition_mode"],
        "uses_sparse_samples": int(preset["uses_sparse_samples"]),
        "uses_building": int(preset["uses_building"]),
        "uses_source": int(preset["uses_source"]),
        "uses_third_channel": int(preset["uses_third_channel"]),
        "gain_mode": gain_mode,
        "sampling_rate": sampling_rate,
        "n_frames": summary.get("n_frames", ""),
        "samples_per_city": summary.get("samples_per_city", ""),
        "summary_path": str(summary_path),
    }
    for key in METRIC_KEYS:
        row[f"{key}_mean"] = summary.get(f"{key}_mean", "")
        row[f"{key}_std"] = summary.get(f"{key}_std", "")
    return row


def write_summary(rows: list[dict], path: Path) -> None:
    fields = [
        "experiment",
        "condition_mode",
        "uses_sparse_samples",
        "uses_building",
        "uses_source",
        "uses_third_channel",
        "gain_mode",
        "sampling_rate",
        "n_frames",
        "samples_per_city",
        "summary_path",
    ] + [f"{key}_{suffix}" for key in METRIC_KEYS for suffix in ("mean", "std")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_config(args: argparse.Namespace, experiments: list[str], sampling_rates: list[float]) -> None:
    config = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "gain_modes": args.gain_modes,
        "split": args.split,
        "sampling_rates": sampling_rates,
        "sampling_rate_start": args.sampling_rate_start,
        "sampling_rate_end": args.sampling_rate_end,
        "sampling_rate_step": args.sampling_rate_step,
        "no_sampling_rate": args.no_sampling_rate,
        "experiments": experiments,
        "experiment_presets": {name: EXPERIMENT_PRESETS[name] for name in experiments},
        "num_samples": args.num_samples,
        "start_index": args.start_index,
        "samples_per_city": args.samples_per_city,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "step_size": args.step_size,
        "dc_iters": args.dc_iters,
        "measurement_noise_std": args.measurement_noise_std,
        "third_channel": args.third_channel,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=True)


def main() -> None:
    args = parse_args()
    resolved_device = device_string(args.device, args.gpu)
    experiments = parse_experiments(args.experiments)
    sampling_rates = resolve_sampling_rates(args)
    sample_script = Path(__file__).with_name("sample_flowdps.py")
    rows = []
    write_config(args, experiments, sampling_rates)

    for experiment in experiments:
        preset = EXPERIMENT_PRESETS[experiment]
        rates = sampling_rates if preset["use_sampling_rates"] else [args.no_sampling_rate]
        for gain_mode in args.gain_modes:
            for sampling_rate in rates:
                summary_path = run_sample(
                    args=args,
                    sample_script=sample_script,
                    experiment=experiment,
                    gain_mode=gain_mode,
                    sampling_rate=sampling_rate,
                    condition_mode=preset["condition_mode"],
                )
                rows.append(read_summary_row(summary_path, experiment, preset, gain_mode, sampling_rate))

    summary_csv = args.output_dir / "information_ablation_summary.csv"
    write_summary(rows, summary_csv)
    print(f"Saved information ablation summary to {summary_csv}")


if __name__ == "__main__":
    main()
