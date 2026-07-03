from __future__ import annotations

import torch


def parse_counts(text: str) -> list[int]:
    counts = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not counts:
        raise ValueError("At least one sparse count is required")
    if any(count < 0 for count in counts):
        raise ValueError("Sparse counts must be non-negative")
    return counts


def build_sparse_raster_condition(
    image: torch.Tensor,
    dense_condition: torch.Tensor,
    sparse_counts: list[int],
    fixed_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if image.dim() != 4 or dense_condition.dim() != 4:
        raise ValueError("image and dense_condition must be BCHW tensors")
    if dense_condition.shape[1] < 1:
        raise ValueError("dense_condition must include a building channel")
    if image.shape[-2:] != dense_condition.shape[-2:]:
        raise ValueError("image and dense_condition spatial shapes must match")

    batch, _, height, width = image.shape
    building = dense_condition[:, :1].to(dtype=image.dtype)
    valid = building <= 0.5
    mask = torch.zeros((batch, 1, height, width), device=image.device, dtype=image.dtype)
    flat_mask = mask.view(batch, -1)
    flat_valid = valid.view(batch, -1)

    for batch_idx in range(batch):
        count = fixed_count
        if count is None:
            index = torch.randint(len(sparse_counts), (), device=image.device).item()
            count = sparse_counts[index]
        candidates = torch.nonzero(flat_valid[batch_idx], as_tuple=False).flatten()
        if count > int(candidates.numel()):
            raise ValueError(
                f"Cannot sample k={count} from {int(candidates.numel())} valid pixels"
            )
        if count == 0:
            continue
        selected = candidates[torch.randperm(candidates.numel(), device=image.device)[:count]]
        flat_mask[batch_idx, selected] = 1.0

    sparse_gain = image[:, :1] * mask
    condition = torch.cat([building, sparse_gain, mask], dim=1)
    return condition, mask, valid.to(dtype=image.dtype)
