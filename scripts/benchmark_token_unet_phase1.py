#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


METRIC_KEYS = ["psnr", "ssim", "mse", "nmse", "rmse", "mae"]

EXPERIMENT_PRESETS = {
    "sampling_only": {
        "condition_mode": "no_building_source",
        "use_sampling_rates": True,
        "uses_sparse_samples": True,
        "uses_building": False,
        "uses_source": False,
    },
    "sampling_building": {
        "condition_mode": "no_source",
        "use_sampling_rates": True,
        "uses_sparse_samples": True,
        "uses_building": True,
        "uses_source": False,
    },
    "sampling_source": {
        "condition_mode": "no_building",
        "use_sampling_rates": True,
        "uses_sparse_samples": True,
        "uses_building": False,
        "uses_source": True,
    },
    "sampling_building_source": {
        "condition_mode": "full",
        "use_sampling_rates": True,
        "uses_sparse_samples": True,
        "uses_building": True,
        "uses_source": True,
    },
    "building_source_no_sampling": {
        "condition_mode": "full",
        "use_sampling_rates": False,
        "uses_sparse_samples": False,
        "uses_building": True,
        "uses_source": True,
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
    parser = argparse.ArgumentParser(description="Benchmark RMFM-TokenUNet-v1 phase-1 validation settings.")
    parser.add_argument("--dataset_root", type=Path, default=Path("/home/DataDisk/zsfang/dataset/RadioMapSeer"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("/home/DataDisk/zsfang/rmfm/results/token_unet_phase1"))
    parser.add_argument("--gain_modes", nargs="+", default=["DPM"])
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--sampling_rates", type=float, nargs="+", default=[0.0001, 0.0003, 0.001, 0.003, 0.01])
    parser.add_argument("--no_sampling_rate", type=float, default=0.0)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--sampler_modes", choices=["no_dc", "dc"], nargs="+", default=["no_dc", "dc"])

    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--samples_per_city", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--data_channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--heatmap_sigma", type=float, default=20.0)
    parser.add_argument("--third_channel", choices=["auto", "ones", "zeros", "cars"], default="auto")
    parser.add_argument("--measurement_noise_std", type=float, default=0.03)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--step_size", type=float, default=50.0)
    parser.add_argument("--dc_iters", type=int, default=3)
    parser.add_argument("--k_max", type=int, default=256)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def run_sample(
    args: argparse.Namespace,
    sample_script: Path,
    sampler_mode: str,
    experiment: str,
    gain_mode: str,
    sampling_rate: float,
    condition_mode: str,
) -> Path:
    summary_path = (
        args.output_dir
        / f"token_unet_{sampler_mode}"
        / condition_mode
        / gain_mode
        / ratio_tag(sampling_rate)
        / "summary.json"
    )
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
        str(args.output_dir),
        "--gain_mode",
        gain_mode,
        "--split",
        args.split,
        "--condition_mode",
        condition_mode,
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
        "--measurement_noise_std",
        str(args.measurement_noise_std),
        "--sampler_mode",
        sampler_mode,
        "--num_steps",
        str(args.num_steps),
        "--step_size",
        str(args.step_size),
        "--dc_iters",
        str(args.dc_iters),
        "--k_max",
        str(args.k_max),
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
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary not found: {summary_path}")
    return summary_path


def read_summary_row(
    summary_path: Path,
    sampler_mode: str,
    experiment: str,
    preset: dict,
    gain_mode: str,
    sampling_rate: float,
) -> dict:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    row = {
        "sampler_mode": sampler_mode,
        "experiment": experiment,
        "condition_mode": preset["condition_mode"],
        "uses_sparse_samples": int(preset["uses_sparse_samples"]),
        "uses_building": int(preset["uses_building"]),
        "uses_source": int(preset["uses_source"]),
        "gain_mode": gain_mode,
        "sampling_rate": sampling_rate,
        "n_frames": summary.get("n_frames", ""),
        "k_max": summary.get("k_max", ""),
        "summary_path": str(summary_path),
    }
    for key in METRIC_KEYS:
        row[f"{key}_mean"] = summary.get(f"{key}_mean", "")
        row[f"{key}_std"] = summary.get(f"{key}_std", "")
    return row


def write_summary(rows: list[dict], path: Path) -> None:
    fields = [
        "sampler_mode",
        "experiment",
        "condition_mode",
        "uses_sparse_samples",
        "uses_building",
        "uses_source",
        "gain_mode",
        "sampling_rate",
        "n_frames",
        "k_max",
        "summary_path",
    ] + [f"{key}_{suffix}" for key in METRIC_KEYS for suffix in ("mean", "std")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_config(args: argparse.Namespace, experiments: list[str]) -> None:
    config = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "gain_modes": args.gain_modes,
        "split": args.split,
        "sampling_rates": args.sampling_rates,
        "no_sampling_rate": args.no_sampling_rate,
        "sampler_modes": args.sampler_modes,
        "experiments": experiments,
        "experiment_presets": {name: EXPERIMENT_PRESETS[name] for name in experiments},
        "num_samples": args.num_samples,
        "samples_per_city": args.samples_per_city,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "step_size": args.step_size,
        "dc_iters": args.dc_iters,
        "k_max": args.k_max,
        "measurement_noise_std": args.measurement_noise_std,
        "third_channel": args.third_channel,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=True)


def main() -> None:
    args = parse_args()
    experiments = parse_experiments(args.experiments)
    sample_script = Path(__file__).with_name("sample_token_flowdps.py")
    rows = []
    write_config(args, experiments)

    for sampler_mode in args.sampler_modes:
        for experiment in experiments:
            preset = EXPERIMENT_PRESETS[experiment]
            rates = args.sampling_rates if preset["use_sampling_rates"] else [args.no_sampling_rate]
            for gain_mode in args.gain_modes:
                for sampling_rate in rates:
                    summary_path = run_sample(
                        args=args,
                        sample_script=sample_script,
                        sampler_mode=sampler_mode,
                        experiment=experiment,
                        gain_mode=gain_mode,
                        sampling_rate=sampling_rate,
                        condition_mode=preset["condition_mode"],
                    )
                    rows.append(read_summary_row(summary_path, sampler_mode, experiment, preset, gain_mode, sampling_rate))

    summary_csv = args.output_dir / "token_unet_phase1_summary.csv"
    write_summary(rows, summary_csv)
    print(f"Saved TokenUNet phase-1 summary to {summary_csv}")


if __name__ == "__main__":
    main()
