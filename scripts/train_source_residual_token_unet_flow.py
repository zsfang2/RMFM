#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rmfm.device import add_gpu_argument, resolve_torch_device  # noqa: E402
from rmfm.data import (  # noqa: E402
    RadioMapSeerFlowDataset,
    filter_samples_by_city,
    list_radiomapseer_samples,
    split_city_ids,
)
from rmfm.modeling_source_residual_token_unet_flow import (  # noqa: E402
    build_source_residual_token_unet_config,
    create_source_residual_token_unet_flow_model,
    load_checkpoint,
    save_checkpoint,
    sample_dense_features,
)
from rmfm.paths import DEFAULT_DATASET_ROOT, DEFAULT_SOURCE_RESIDUAL_TOKEN_UNET_FLOW_CHECKPOINT_DIR  # noqa: E402
from rmfm.token_utils import sample_sparse_mask, sparse_tokens_from_mask  # noqa: E402


LOG_FIELDS = [
    "wall_time_sec",
    "event",
    "step",
    "train_loss",
    "train_fm_loss",
    "train_prior_loss",
    "train_sing_loss",
    "train_obs_loss",
    "val_loss",
    "val_fm_loss",
    "val_prior_loss",
    "val_sing_loss",
    "val_obs_loss",
    "best_val_loss",
    "best_step",
    "lr",
    "samples_seen",
    "checkpoint",
]


def parse_channel_mults(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def parse_rates(text: str) -> list[float]:
    rates = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not rates:
        raise ValueError("At least one sparse rate is required")
    if any(rate < 0.0 or rate > 1.0 for rate in rates):
        raise ValueError("Sparse rates must be in [0, 1]")
    return rates


def append_logs(output_dir: Path, row: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "train_log.csv"
    jsonl_path = output_dir / "train_log.jsonl"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in LOG_FIELDS})
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def get_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def normalize_config(value: Any) -> Any:
    if isinstance(value, tuple):
        return [normalize_config(x) for x in value]
    if isinstance(value, list):
        return [normalize_config(x) for x in value]
    if isinstance(value, dict):
        return {k: normalize_config(v) for k, v in sorted(value.items())}
    return value


def validate_resume_config(current_config: dict[str, Any], checkpoint_config: dict[str, Any]) -> None:
    if normalize_config(current_config) == normalize_config(checkpoint_config):
        return
    raise ValueError(
        "Checkpoint model_config does not match current v2 architecture arguments. "
        "Use the same image/model/tokenizer/prior arguments, or start a new output directory."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train source-conditioned residual RMFM-TokenUNet-v2."
    )
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_SOURCE_RESIDUAL_TOKEN_UNET_FLOW_CHECKPOINT_DIR,
    )
    parser.add_argument("--gain_modes", nargs="+", default=["DPM"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--data_channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--heatmap_sigma", type=float, default=20.0)
    parser.add_argument("--third_channel", choices=["auto", "ones", "zeros", "cars"], default="auto")

    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--channel_mults", type=str, default="1,2,4,4")
    parser.add_argument("--layers_per_block", type=int, default=2)
    parser.add_argument("--norm_num_groups", type=int, default=32)
    parser.add_argument("--attention_head_dim", type=int, default=8)
    parser.add_argument("--transformer_layers_per_block", type=int, default=1)

    parser.add_argument("--token_dim", type=int, default=256)
    parser.add_argument("--tokenizer_layers", type=int, default=2)
    parser.add_argument("--tokenizer_heads", type=int, default=4)
    parser.add_argument("--fourier_bands", type=int, default=32)
    parser.add_argument("--k_max", type=int, default=256)
    parser.add_argument(
        "--sparse_rates",
        type=str,
        default="0.0,0.0001,0.0003,0.001,0.003,0.005,0.01",
        help="Comma-separated sparse sampling rates randomly used during training.",
    )

    parser.add_argument("--prior_base_channels", type=int, default=32)
    parser.add_argument("--singularity_base_channels", type=int, default=32)
    parser.add_argument("--prior_lowpass_factor", type=int, default=8)
    parser.add_argument("--lambda_prior", type=float, default=0.1)
    parser.add_argument("--lambda_sing", type=float, default=0.01)
    parser.add_argument("--lambda_obs", type=float, default=0.05)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--no_persistent_workers", action="store_true")
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device, e.g. cuda, cuda:0, cuda:1, cpu, or a bare GPU id like 1.",
    )
    add_gpu_argument(parser)
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_val_samples", type=int, default=512)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def make_loader(dataset: RadioMapSeerFlowDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": shuffle,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = not args.no_persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def dense_building_source(condition: torch.Tensor) -> torch.Tensor:
    if condition.shape[1] < 2:
        raise ValueError(f"SourceResidualTokenUNet dense branch expects building/source, got {condition.shape[1]}")
    return condition[:, :2]


def choose_sparse_rates(batch_size: int, rates: list[float], device: torch.device) -> torch.Tensor:
    idx = torch.randint(0, len(rates), (batch_size,), device=device)
    table = torch.tensor(rates, device=device, dtype=torch.float32)
    return table[idx]


def lowpass_target(image: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return image
    height, width = image.shape[-2:]
    low_h = max(1, height // factor)
    low_w = max(1, width // factor)
    low = F.interpolate(image.float(), size=(low_h, low_w), mode="area")
    return F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False).to(dtype=image.dtype)


def laplacian_singularity_target(image: torch.Tensor) -> torch.Tensor:
    base = image[:, :1].float()
    kernel = base.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    lap = F.conv2d(base, kernel, padding=1).abs()
    denom = lap.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    return (lap / denom).clamp(0.0, 1.0).to(dtype=image.dtype)


def observation_consistency_loss(
    x0_hat: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
) -> torch.Tensor:
    pred_values = sample_dense_features(x0_hat[:, :1], obs_coords)
    valid = obs_mask.unsqueeze(-1).to(dtype=pred_values.dtype)
    denom = valid.sum().clamp_min(1.0)
    return (valid * (pred_values - obs_values).abs()).sum() / denom


def flow_matching_losses(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition: torch.Tensor,
    sparse_rates: list[float],
    k_max: int,
    prior_lowpass_factor: int,
    lambda_prior: float,
    lambda_sing: float,
    lambda_obs: float,
) -> dict[str, torch.Tensor]:
    noise = torch.randn_like(image)
    t = torch.rand(image.shape[0], device=image.device, dtype=image.dtype)
    t_view = t.view(-1, 1, 1, 1)
    x_t = (1.0 - t_view) * image + t_view * noise
    target_v = noise - image

    rates = choose_sparse_rates(image.shape[0], sparse_rates, image.device)
    sparse_mask = sample_sparse_mask(image, rates)
    coords, values, obs_mask = sparse_tokens_from_mask(image, sparse_mask, k_max=k_max)
    dense_condition = dense_building_source(condition)

    output = model(
        x_t,
        t * 1000.0,
        dense_condition=dense_condition,
        obs_coords=coords,
        obs_values=values,
        obs_mask=obs_mask,
        return_aux=True,
    )
    pred_v = output["velocity"]
    prior_map = output["prior_map"]
    singularity_map = output["singularity_map"]

    fm_loss = F.mse_loss(pred_v.float(), target_v.float())
    prior_loss = F.mse_loss(prior_map.float(), lowpass_target(image, prior_lowpass_factor).float())
    sing_loss = F.mse_loss(singularity_map.float(), laplacian_singularity_target(image).float())
    x0_hat = x_t - t_view * pred_v
    obs_loss = observation_consistency_loss(x0_hat.float(), coords, values.float(), obs_mask)
    total = fm_loss + lambda_prior * prior_loss + lambda_sing * sing_loss + lambda_obs * obs_loss
    return {
        "loss": total,
        "fm_loss": fm_loss,
        "prior_loss": prior_loss,
        "sing_loss": sing_loss,
        "obs_loss": obs_loss,
    }


@torch.no_grad()
def evaluate_losses(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int,
    channels_last: bool,
    sparse_rates: list[float],
    k_max: int,
    prior_lowpass_factor: int,
    lambda_prior: float,
    lambda_sing: float,
    lambda_obs: float,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "fm_loss": 0.0, "prior_loss": 0.0, "sing_loss": 0.0, "obs_loss": 0.0}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if 0 <= max_batches <= batch_idx:
            break
        image = batch["image"].to(device=device, non_blocking=True)
        condition = batch["condition"].to(device=device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
            condition = condition.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            losses = flow_matching_losses(
                model,
                image,
                condition,
                sparse_rates=sparse_rates,
                k_max=k_max,
                prior_lowpass_factor=prior_lowpass_factor,
                lambda_prior=lambda_prior,
                lambda_sing=lambda_sing,
                lambda_obs=lambda_obs,
            )
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        count += 1
    model.train()
    denom = max(count, 1)
    return {key: value / denom for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    sparse_rates = parse_rates(args.sparse_rates)
    torch.manual_seed(args.seed)
    device = resolve_torch_device(args.device, args.gpu)
    amp_enabled = args.mixed_precision != "no" and device.type == "cuda"
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and args.mixed_precision == "fp16")
    if args.allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    samples = list_radiomapseer_samples(args.dataset_root, args.gain_modes)
    splits = split_city_ids(samples, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    train_samples = filter_samples_by_city(samples, splits["train"], max_samples=args.max_train_samples)
    val_samples = filter_samples_by_city(samples, splits["val"], max_samples=args.max_val_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "city_splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=True)
    with open(args.output_dir / "train_args.json", "w", encoding="utf-8") as f:
        payload = vars(args).copy()
        payload["parsed_sparse_rates"] = sparse_rates
        json.dump(payload, f, indent=2, default=str, ensure_ascii=True)

    train_dataset = RadioMapSeerFlowDataset(
        train_samples,
        image_size=args.image_size,
        data_channels=args.data_channels,
        heatmap_sigma=args.heatmap_sigma,
        third_channel=args.third_channel,
    )
    val_dataset = RadioMapSeerFlowDataset(
        val_samples,
        image_size=args.image_size,
        data_channels=args.data_channels,
        heatmap_sigma=args.heatmap_sigma,
        third_channel=args.third_channel,
    )
    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False)

    model_config = build_source_residual_token_unet_config(
        image_size=args.image_size,
        data_channels=args.data_channels,
        cond_channels=2,
        base_channels=args.base_channels,
        channel_mults=parse_channel_mults(args.channel_mults),
        layers_per_block=args.layers_per_block,
        norm_num_groups=args.norm_num_groups,
        token_dim=args.token_dim,
        tokenizer_layers=args.tokenizer_layers,
        tokenizer_heads=args.tokenizer_heads,
        fourier_bands=args.fourier_bands,
        attention_head_dim=args.attention_head_dim,
        transformer_layers_per_block=args.transformer_layers_per_block,
        prior_base_channels=args.prior_base_channels,
        singularity_base_channels=args.singularity_base_channels,
    )
    model = create_source_residual_token_unet_flow_model(model_config).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    best_val_loss = float("inf")
    best_step = -1
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, map_location="cpu")
        validate_resume_config(model_config, checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        if "scaler_state_dict" in checkpoint and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng_state(checkpoint.get("rng_state"))
        start_step = int(checkpoint.get("step", 0))
        extra = checkpoint.get("extra", {})
        best_val_loss = float(extra.get("best_val_loss", extra.get("val_loss", best_val_loss)))
        best_step = int(extra.get("best_step", start_step if "val_loss" in extra else best_step))

    existing_best_path = args.output_dir / "best.pt"
    if existing_best_path.exists():
        best_checkpoint = load_checkpoint(existing_best_path, map_location="cpu")
        best_extra = best_checkpoint.get("extra", {})
        existing_best = best_extra.get("val_loss", best_extra.get("best_val_loss"))
        if existing_best is not None and float(existing_best) < best_val_loss:
            best_val_loss = float(existing_best)
            best_step = int(best_extra.get("best_step", best_checkpoint.get("step", best_step)))

    train_model = model
    if args.compile:
        train_model = torch.compile(model)

    print(f"Dataset root: {args.dataset_root}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Sparse rates: {sparse_rates} | K max: {args.k_max}")
    print(
        "Loss weights: "
        f"prior={args.lambda_prior} sing={args.lambda_sing} obs={args.lambda_obs} "
        f"| prior lowpass factor={args.prior_lowpass_factor}"
    )
    print(f"Device: {device} | AMP: {args.mixed_precision if amp_enabled else 'no'}")
    print(f"Output: {args.output_dir}")

    train_model.train()
    step = start_step
    running = {"loss": 0.0, "fm_loss": 0.0, "prior_loss": 0.0, "sing_loss": 0.0, "obs_loss": 0.0}
    last_train = {key: float("nan") for key in running}
    run_start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=args.max_steps, initial=step, desc="Training SourceResidualTokenUNet")

    while step < args.max_steps:
        for batch in train_loader:
            image = batch["image"].to(device=device, non_blocking=True)
            condition = batch["condition"].to(device=device, non_blocking=True)
            if args.channels_last:
                image = image.contiguous(memory_format=torch.channels_last)
                condition = condition.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                losses = flow_matching_losses(
                    train_model,
                    image,
                    condition,
                    sparse_rates=sparse_rates,
                    k_max=args.k_max,
                    prior_lowpass_factor=args.prior_lowpass_factor,
                    lambda_prior=args.lambda_prior,
                    lambda_sing=args.lambda_sing,
                    lambda_obs=args.lambda_obs,
                )
                loss_for_backward = losses["loss"] / args.grad_accum_steps

            scaler.scale(loss_for_backward).backward()
            for key in running:
                running[key] += float(losses[key].detach().cpu())

            if (step + 1) % args.grad_accum_steps == 0:
                if args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            pbar.update(1)

            if step % args.log_every == 0:
                last_train = {key: value / max(args.log_every, 1) for key, value in running.items()}
                pbar.set_postfix(loss=f"{last_train['loss']:.5f}", fm=f"{last_train['fm_loss']:.5f}")
                append_logs(
                    args.output_dir,
                    {
                        "wall_time_sec": time.time() - run_start_time,
                        "event": "train",
                        "step": step,
                        "train_loss": last_train["loss"],
                        "train_fm_loss": last_train["fm_loss"],
                        "train_prior_loss": last_train["prior_loss"],
                        "train_sing_loss": last_train["sing_loss"],
                        "train_obs_loss": last_train["obs_loss"],
                        "val_loss": "",
                        "best_val_loss": best_val_loss if best_val_loss != float("inf") else "",
                        "best_step": best_step if best_step >= 0 else "",
                        "lr": current_lr(optimizer),
                        "samples_seen": step * args.batch_size,
                        "checkpoint": "",
                    },
                )
                running = {key: 0.0 for key in running}

            if step % args.val_every == 0:
                val_batches = math.ceil(args.max_val_samples / max(args.batch_size, 1))
                val = evaluate_losses(
                    train_model,
                    val_loader,
                    device,
                    amp_enabled,
                    amp_dtype,
                    val_batches,
                    args.channels_last,
                    sparse_rates=sparse_rates,
                    k_max=args.k_max,
                    prior_lowpass_factor=args.prior_lowpass_factor,
                    lambda_prior=args.lambda_prior,
                    lambda_sing=args.lambda_sing,
                    lambda_obs=args.lambda_obs,
                )
                is_best = val["loss"] < best_val_loss
                if is_best:
                    best_val_loss = val["loss"]
                    best_step = step
                print(
                    f"\n[step {step}] val_loss={val['loss']:.6f} "
                    f"fm={val['fm_loss']:.6f} prior={val['prior_loss']:.6f} "
                    f"sing={val['sing_loss']:.6f} obs={val['obs_loss']:.6f}"
                )
                checkpoint_extra = {
                    "val_loss": val["loss"],
                    "val_fm_loss": val["fm_loss"],
                    "val_prior_loss": val["prior_loss"],
                    "val_sing_loss": val["sing_loss"],
                    "val_obs_loss": val["obs_loss"],
                    "best_val_loss": best_val_loss,
                    "best_step": best_step,
                    "gain_modes": args.gain_modes,
                    "sparse_rates": sparse_rates,
                    "k_max": args.k_max,
                    "prior_lowpass_factor": args.prior_lowpass_factor,
                    "lambda_prior": args.lambda_prior,
                    "lambda_sing": args.lambda_sing,
                    "lambda_obs": args.lambda_obs,
                }
                save_checkpoint(
                    args.output_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    model_config=model_config,
                    step=step,
                    extra=checkpoint_extra,
                    scaler=scaler,
                    rng_state=get_rng_state(),
                )
                if is_best:
                    save_checkpoint(
                        args.output_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        model_config=model_config,
                        step=step,
                        extra=checkpoint_extra,
                        scaler=scaler,
                        rng_state=get_rng_state(),
                    )
                append_logs(
                    args.output_dir,
                    {
                        "wall_time_sec": time.time() - run_start_time,
                        "event": "val",
                        "step": step,
                        "train_loss": last_train["loss"],
                        "train_fm_loss": last_train["fm_loss"],
                        "train_prior_loss": last_train["prior_loss"],
                        "train_sing_loss": last_train["sing_loss"],
                        "train_obs_loss": last_train["obs_loss"],
                        "val_loss": val["loss"],
                        "val_fm_loss": val["fm_loss"],
                        "val_prior_loss": val["prior_loss"],
                        "val_sing_loss": val["sing_loss"],
                        "val_obs_loss": val["obs_loss"],
                        "best_val_loss": best_val_loss,
                        "best_step": best_step,
                        "lr": current_lr(optimizer),
                        "samples_seen": step * args.batch_size,
                        "checkpoint": "best.pt" if is_best else "latest.pt",
                    },
                )

            if step % args.save_every == 0:
                save_checkpoint(
                    args.output_dir / f"checkpoint_step_{step:07d}.pt",
                    model=model,
                    optimizer=optimizer,
                    model_config=model_config,
                    step=step,
                    extra={
                        "best_val_loss": best_val_loss,
                        "best_step": best_step,
                        "gain_modes": args.gain_modes,
                        "sparse_rates": sparse_rates,
                        "k_max": args.k_max,
                        "prior_lowpass_factor": args.prior_lowpass_factor,
                        "lambda_prior": args.lambda_prior,
                        "lambda_sing": args.lambda_sing,
                        "lambda_obs": args.lambda_obs,
                    },
                    scaler=scaler,
                    rng_state=get_rng_state(),
                )

            if step >= args.max_steps:
                break

    final_extra = {
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "gain_modes": args.gain_modes,
        "sparse_rates": sparse_rates,
        "k_max": args.k_max,
        "prior_lowpass_factor": args.prior_lowpass_factor,
        "lambda_prior": args.lambda_prior,
        "lambda_sing": args.lambda_sing,
        "lambda_obs": args.lambda_obs,
    }
    save_checkpoint(
        args.output_dir / "final.pt",
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        step=step,
        extra=final_extra,
        scaler=scaler,
        rng_state=get_rng_state(),
    )
    save_checkpoint(
        args.output_dir / "latest.pt",
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        step=step,
        extra=final_extra,
        scaler=scaler,
        rng_state=get_rng_state(),
    )
    append_logs(
        args.output_dir,
        {
            "wall_time_sec": time.time() - run_start_time,
            "event": "final",
            "step": step,
            "train_loss": last_train["loss"],
            "train_fm_loss": last_train["fm_loss"],
            "train_prior_loss": last_train["prior_loss"],
            "train_sing_loss": last_train["sing_loss"],
            "train_obs_loss": last_train["obs_loss"],
            "best_val_loss": best_val_loss if best_val_loss != float("inf") else "",
            "best_step": best_step if best_step >= 0 else "",
            "lr": current_lr(optimizer),
            "samples_seen": step * args.batch_size,
            "checkpoint": "final.pt",
        },
    )
    pbar.close()
    print(f"Saved final checkpoint to {args.output_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
