#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

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
from rmfm.flowdps import parse_dtype  # noqa: E402
from rmfm.io import save_mask, save_tensor_image, tensor_to_float01  # noqa: E402
from rmfm.masks import build_measurement, make_exact_ratio_mask, mask_to_tensor, stable_int_seed  # noqa: E402
from rmfm.metrics import (  # noqa: E402
    MASKED_METRIC_KEYS,
    compute_masked_metrics,
    compute_metrics,
    summarize_metric_keys,
    summarize_metrics,
    write_metrics_csv,
    write_summary_json,
)
from rmfm.modeling_token_unet_flow import load_model_from_checkpoint  # noqa: E402
from rmfm.modeling_source_residual_token_unet_flow import (  # noqa: E402
    load_model_from_checkpoint as load_source_residual_model_from_checkpoint,
)
from rmfm.paths import DEFAULT_DATASET_ROOT, DEFAULT_TOKEN_UNET_RESULT_ROOT  # noqa: E402
from rmfm.token_utils import sparse_tokens_from_mask  # noqa: E402


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


def dense_building_source(condition: torch.Tensor) -> torch.Tensor:
    if condition.shape[1] < 2:
        raise ValueError(f"TokenUNet dense branch expects building/source channels, got {condition.shape[1]}")
    return condition[:, :2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RMFM-TokenUNet-v1 sparse radio-map sampling.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_TOKEN_UNET_RESULT_ROOT / "token_unet_flowdps")
    parser.add_argument("--gain_mode", type=str, default="DPM")
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--data_channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--heatmap_sigma", type=float, default=20.0)
    parser.add_argument("--third_channel", choices=["auto", "ones", "zeros", "cars"], default="auto")
    parser.add_argument("--condition_mode", choices=CONDITION_MODES, default="no_building_source")

    parser.add_argument("--sampling_rate", type=float, default=0.01)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--samples_per_city", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measurement_noise_std", type=float, default=0.03)

    parser.add_argument("--sampler_mode", choices=["no_dc", "dc"], default="dc")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--step_size", type=float, default=50.0)
    parser.add_argument("--dc_iters", type=int, default=3)
    parser.add_argument("--k_max", type=int, default=None)
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


class RadioMapTokenUNetSampler:
    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        dtype: str = "fp16",
        num_train_timesteps: int = 1000,
    ) -> None:
        self.device = device
        self.model_dtype = parse_dtype(dtype, device)
        checkpoint_metadata = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        self.model_type = str(checkpoint_metadata.get("model_type", "token_unet_flow"))
        if self.model_type == "source_residual_token_unet_flow":
            loader = load_source_residual_model_from_checkpoint
            self.output_name = "source_residual_token_unet"
        elif self.model_type == "token_unet_flow":
            loader = load_model_from_checkpoint
            self.output_name = "token_unet"
        else:
            raise ValueError(f"Unsupported checkpoint model_type: {self.model_type}")
        self.model, self.model_config, self.checkpoint = loader(
            checkpoint, device=device, dtype=self.model_dtype
        )
        self.num_train_timesteps = num_train_timesteps
        self.data_channels = int(self.model_config["data_channels"])
        self.k_max = int(self.checkpoint.get("extra", {}).get("k_max", 256))

    @torch.no_grad()
    def predict_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dense_condition: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = (t * float(self.num_train_timesteps)).to(device=x.device)
        pred = self.model(
            x.to(dtype=self.model_dtype),
            timesteps,
            dense_condition=dense_condition.to(device=x.device, dtype=self.model_dtype),
            obs_coords=obs_coords.to(device=x.device, dtype=self.model_dtype),
            obs_values=obs_values.to(device=x.device, dtype=self.model_dtype),
            obs_mask=obs_mask.to(device=x.device),
        )
        return pred.float()

    def data_consistency(
        self,
        x0: torch.Tensor,
        measurement: torch.Tensor,
        mask: torch.Tensor,
        step_size: float,
        num_iters: int,
    ) -> torch.Tensor:
        current = x0.detach().float().requires_grad_(True)
        measurement = measurement.detach().float()
        mask = mask.detach().float()
        if mask.shape[1] == 1 and current.shape[1] > 1:
            mask = mask.repeat(1, current.shape[1], 1, 1)
        if num_iters <= 0 or not torch.any(mask > 0):
            return current.detach()

        for _ in range(num_iters):
            residual = mask * (current - measurement)
            loss = torch.linalg.norm(residual.reshape(residual.shape[0], -1), dim=1).mean()
            grad = torch.autograd.grad(loss, current)[0]
            current = (current - step_size * grad).clamp(-1.0, 1.0).detach().requires_grad_(True)
        return current.detach()

    def sample(
        self,
        measurement: torch.Tensor,
        mask: torch.Tensor,
        dense_condition: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        sampler_mode: str,
        num_steps: int = 20,
        step_size: float = 50.0,
        dc_iters: int = 3,
        generator: torch.Generator | None = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        measurement = measurement.to(self.device).float()
        mask = mask.to(self.device).float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        dense_condition = dense_condition.to(self.device).float()
        obs_coords = obs_coords.to(self.device)
        obs_values = obs_values.to(self.device)
        obs_mask = obs_mask.to(self.device)

        x = torch.randn(
            measurement.shape,
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc=f"TokenUNet-{sampler_mode}")

        for i in iterator:
            t = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = t.expand(x.shape[0])
            v = self.predict_velocity(x, t_batch, dense_condition, obs_coords, obs_values, obs_mask)

            x0_pred = x - t * v
            x1_pred = x + (1.0 - t) * v
            if sampler_mode == "dc":
                x0_dc = self.data_consistency(
                    x0_pred,
                    measurement=measurement,
                    mask=mask,
                    step_size=step_size,
                    num_iters=dc_iters,
                )
                x0_blend = (1.0 - t) * x0_pred + t * x0_dc
            else:
                x0_blend = x0_pred
            x = ((1.0 - t_next) * x0_blend + t_next * x1_pred).clamp(-1.0, 1.0)

        if not torch.isfinite(x).all():
            non_finite = int((~torch.isfinite(x)).sum().item())
            raise FloatingPointError(
                f"Sampler produced {non_finite} non-finite values. "
                "Retry with --dtype fp32; source-residual checkpoints may be unstable in fp16."
            )
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
    sampler = RadioMapTokenUNetSampler(
        checkpoint=args.checkpoint,
        device=device,
        dtype=args.dtype,
    )
    k_max = sampler.k_max if args.k_max is None else args.k_max

    out_root = (
        args.output_dir
        / f"{sampler.output_name}_{args.sampler_mode}"
        / args.condition_mode
        / args.gain_mode
        / ratio_tag(args.sampling_rate)
    )
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
        dense_condition = dense_building_source(condition)

        mask_seed = stable_int_seed(args.seed, item["gain_path"], f"{args.sampling_rate:.8f}")
        noise_seed = stable_int_seed(args.seed, item["gain_path"], "measurement_noise")
        sample_seed = stable_int_seed(args.seed, item["gain_path"], "token_flow_sample")

        mask_np = make_exact_ratio_mask(args.image_size, args.image_size, args.sampling_rate, mask_seed)
        mask = mask_to_tensor(mask_np, device=device)
        measurement = build_measurement(image, mask, args.measurement_noise_std, noise_seed)
        obs_coords, obs_values, obs_mask = sparse_tokens_from_mask(measurement, mask, k_max=k_max)

        generator = torch.Generator(device=device)
        generator.manual_seed(sample_seed)

        t0 = time.perf_counter()
        recon = sampler.sample(
            measurement=measurement,
            mask=mask,
            dense_condition=dense_condition,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            sampler_mode=args.sampler_mode,
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
        metrics = compute_metrics(recon_np, image_np)
        metrics.update(compute_masked_metrics(recon_np, image_np, mask_np, prefix="observed"))
        metrics.update(compute_masked_metrics(recon_np, image_np, 1.0 - mask_np, prefix="unobserved"))
        metrics.update(compute_masked_metrics(recon_np, measurement_np, mask_np, prefix="measurement"))
        rows.append(
            {
                "frame": frame,
                "mode": item["mode"],
                "city_id": item["city_id"],
                "source_id": item["source_id"],
                "sampling_rate": args.sampling_rate,
                "condition_mode": args.condition_mode,
                "sampler_mode": args.sampler_mode,
                "time_sec": elapsed,
                **metrics,
            }
        )
        print(
            f"[{idx + 1}/{len(dataset)}] {frame} "
            f"PSNR={metrics['psnr']:.3f} SSIM={metrics['ssim']:.4f} time={elapsed:.2f}s"
        )

    summary = summarize_metrics(rows)
    regional_keys = [
        f"{prefix}_{key}"
        for prefix in ("observed", "unobserved", "measurement")
        for key in MASKED_METRIC_KEYS
    ]
    summary.update(summarize_metric_keys(rows, regional_keys))
    summary.update(
        {
            "gain_mode": args.gain_mode,
            "sampling_rate": args.sampling_rate,
            "checkpoint": str(args.checkpoint),
            "model_type": sampler.model_type,
            "num_steps": args.num_steps,
            "step_size": args.step_size,
            "dc_iters": args.dc_iters,
            "measurement_noise_std": args.measurement_noise_std,
            "condition_mode": args.condition_mode,
            "sampler_mode": args.sampler_mode,
            "third_channel": args.third_channel,
            "samples_per_city": args.samples_per_city,
            "k_max": k_max,
            "n_frames": len(rows),
        }
    )
    write_metrics_csv(rows, out_root / "metrics.csv")
    write_summary_json(summary, out_root / "summary.json")
    print(f"Saved results to {out_root}")


if __name__ == "__main__":
    main()
