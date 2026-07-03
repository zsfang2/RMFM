from __future__ import annotations

import argparse

import torch


def add_gpu_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-gpu",
        "--gpu",
        type=int,
        default=None,
        help=(
            "Use the x-th currently visible CUDA device, e.g. -gpu 1 maps to cuda:1. "
            "This overrides --device when both are provided."
        ),
    )


def device_string(device: str | int = "cuda", gpu: int | None = None) -> str:
    if gpu is not None:
        if gpu < 0:
            raise ValueError(f"GPU id must be non-negative, got {gpu}")
        return f"cuda:{gpu}"

    text = str(device)
    if text.isdigit():
        return f"cuda:{text}"
    return text


def resolve_torch_device(device: str | int = "cuda", gpu: int | None = None) -> torch.device:
    requested = device_string(device, gpu)
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if ":" in requested:
            try:
                gpu_id = int(requested.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device string: {requested}") from exc
            if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                raise ValueError(
                    f"Invalid GPU id {gpu_id}; available ids are 0..{torch.cuda.device_count() - 1}"
                )
    return torch.device(requested)
