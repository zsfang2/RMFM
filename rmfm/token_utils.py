from __future__ import annotations

import torch


def sample_sparse_mask(
    x0: torch.Tensor,
    rate: float | torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if x0.dim() != 4:
        raise ValueError(f"x0 must have shape [B, C, H, W], got {tuple(x0.shape)}")
    batch, _, height, width = x0.shape
    total = height * width
    if isinstance(rate, torch.Tensor):
        rates = rate.detach().to(device=x0.device, dtype=torch.float32).flatten()
        if rates.numel() == 1:
            rates = rates.expand(batch)
    else:
        rates = torch.full((batch,), float(rate), device=x0.device)
    if rates.numel() != batch:
        raise ValueError(f"Expected one rate per batch item or scalar rate, got {rates.numel()} for batch {batch}")
    if torch.any((rates < 0.0) | (rates > 1.0)):
        raise ValueError("All rates must be in [0, 1]")

    masks = torch.zeros((batch, 1, height, width), device=x0.device, dtype=torch.bool)
    for item_idx, item_rate in enumerate(rates):
        n_known = int(round(float(item_rate) * total))
        if n_known <= 0:
            continue
        if n_known >= total:
            masks[item_idx, 0] = True
            continue
        idx = torch.randperm(total, device=x0.device, generator=generator)[:n_known]
        masks[item_idx, 0].view(-1)[idx] = True
    return masks


def extract_sparse_tokens(x0: torch.Tensor, sparse_mask: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if x0.dim() != 4:
        raise ValueError(f"x0 must have shape [B, C, H, W], got {tuple(x0.shape)}")
    if sparse_mask.dim() == 3:
        sparse_mask = sparse_mask.unsqueeze(1)
    if sparse_mask.dim() != 4:
        raise ValueError(f"sparse_mask must have shape [B, 1, H, W], got {tuple(sparse_mask.shape)}")

    batch, _, height, width = x0.shape
    value_image = x0[:, :1]
    coords_list: list[torch.Tensor] = []
    values_list: list[torch.Tensor] = []
    for item_idx in range(batch):
        ys, xs = torch.where(sparse_mask[item_idx, 0].bool())
        if ys.numel() == 0:
            coords = x0.new_zeros((0, 2))
            values = x0.new_zeros((0, 1))
        else:
            x_norm = xs.to(dtype=x0.dtype) / max(width - 1, 1)
            y_norm = ys.to(dtype=x0.dtype) / max(height - 1, 1)
            coords = torch.stack([x_norm, y_norm], dim=-1)
            values = value_image[item_idx, 0, ys, xs].unsqueeze(-1)
        coords_list.append(coords)
        values_list.append(values)
    return coords_list, values_list


def pad_or_subsample_tokens(
    coords_list: list[torch.Tensor],
    values_list: list[torch.Tensor],
    k_max: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(coords_list) != len(values_list):
        raise ValueError("coords_list and values_list must have the same length")
    if k_max < 0:
        raise ValueError(f"k_max must be >= 0, got {k_max}")
    if not coords_list:
        raise ValueError("Cannot pad an empty batch")

    batch = len(coords_list)
    device = coords_list[0].device
    dtype = coords_list[0].dtype
    coords = torch.zeros((batch, k_max, 2), device=device, dtype=dtype)
    values = torch.zeros((batch, k_max, 1), device=device, dtype=dtype)
    mask = torch.zeros((batch, k_max), device=device, dtype=torch.bool)

    for item_idx, (item_coords, item_values) in enumerate(zip(coords_list, values_list)):
        count = item_coords.shape[0]
        if count == 0 or k_max == 0:
            continue
        if item_values.shape[0] != count:
            raise ValueError("Each coords/value pair must have the same number of points")
        if count > k_max:
            keep = torch.randperm(count, device=item_coords.device, generator=generator)[:k_max]
            keep = torch.sort(keep).values
            item_coords = item_coords[keep]
            item_values = item_values[keep]
            count = k_max
        coords[item_idx, :count] = item_coords
        values[item_idx, :count] = item_values
        mask[item_idx, :count] = True
    return coords, values, mask


def sparse_tokens_from_mask(
    x0: torch.Tensor,
    sparse_mask: torch.Tensor,
    k_max: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords_list, values_list = extract_sparse_tokens(x0, sparse_mask)
    return pad_or_subsample_tokens(coords_list, values_list, k_max=k_max, generator=generator)
