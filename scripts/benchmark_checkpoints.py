#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple checkpoints across sampling rates.")
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_glob", type=str, default="checkpoint_step_*.pt")
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=None)
    parser.add_argument("--include_best", action="store_true")
    parser.add_argument("--include_latest", action="store_true")
    parser.add_argument("--include_final", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=Path("/home/DataDisk/zsfang/rmfm/results/checkpoint_sweep"))

    parser.add_argument("--dataset_root", type=Path, default=Path("/home/DataDisk/zsfang/dataset/RadioMapSeer"))
    parser.add_argument("--gain_modes", nargs="+", default=["DPM"])
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--sampling_rates", type=float, nargs="+", default=[0.01, 0.03, 0.05])
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
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_step_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    if path.name == "best.pt":
        return -3
    if path.name == "latest.pt":
        return -2
    if path.name == "final.pt":
        return -1
    return 10**18


def checkpoint_tag(path: Path) -> str:
    return path.stem


def collect_checkpoints(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.checkpoints:
        paths.extend(args.checkpoints)
    else:
        paths.extend(sorted(args.checkpoint_dir.glob(args.checkpoint_glob), key=checkpoint_step))
    for name, enabled in (
        ("best.pt", args.include_best),
        ("latest.pt", args.include_latest),
        ("final.pt", args.include_final),
    ):
        path = args.checkpoint_dir / name
        if enabled and path.exists():
            paths.append(path)
    deduped = []
    seen = set()
    for path in sorted(paths, key=checkpoint_step):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    if not deduped:
        raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}")
    return deduped


def read_summary(summary_csv: Path, checkpoint: Path) -> list[dict]:
    rows = []
    with open(summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["checkpoint"] = checkpoint.name
            row["checkpoint_path"] = str(checkpoint)
            row["checkpoint_step"] = checkpoint_step(checkpoint)
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    benchmark_script = Path(__file__).with_name("benchmark_sampling_rates.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = collect_checkpoints(args)
    all_rows = []

    for idx, checkpoint in enumerate(checkpoints, start=1):
        tag = checkpoint_tag(checkpoint)
        checkpoint_out = args.output_dir / tag
        summary_csv = checkpoint_out / "benchmark_summary.csv"
        if args.skip_existing and summary_csv.exists():
            print(f"[{idx}/{len(checkpoints)}] Reusing existing summary for {checkpoint.name}")
        else:
            command = [
                sys.executable,
                str(benchmark_script),
                "--dataset_root",
                str(args.dataset_root),
                "--checkpoint",
                str(checkpoint),
                "--output_dir",
                str(checkpoint_out),
                "--gain_modes",
                *args.gain_modes,
                "--split",
                args.split,
                "--sampling_rates",
                *[str(x) for x in args.sampling_rates],
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
                "--condition_modes",
                *args.condition_modes,
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
            print(f"[{idx}/{len(checkpoints)}] Running {checkpoint.name}")
            subprocess.run(command, check=True)

        if not summary_csv.exists():
            raise FileNotFoundError(f"Expected summary not found: {summary_csv}")
        all_rows.extend(read_summary(summary_csv, checkpoint))

    out_csv = args.output_dir / "checkpoint_sweep_summary.csv"
    fields = [
        "checkpoint",
        "checkpoint_step",
        "checkpoint_path",
        "condition_mode",
        "gain_mode",
        "sampling_rate",
        "n_frames",
    ] + [
        f"{key}_{suffix}" for key in METRIC_KEYS for suffix in ("mean", "std")
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(f"Saved checkpoint sweep summary to {out_csv}")


if __name__ == "__main__":
    main()
