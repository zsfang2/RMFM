from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .modeling_unet_flow import load_model_from_checkpoint


def parse_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if name == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


class RadioMapUNetFlowDPS:
    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        dtype: str = "fp16",
        num_train_timesteps: int = 1000,
    ) -> None:
        self.device = device
        self.model_dtype = parse_dtype(dtype, device)
        self.model, self.model_config, self.checkpoint = load_model_from_checkpoint(
            checkpoint, device=device, dtype=self.model_dtype
        )
        self.num_train_timesteps = num_train_timesteps
        self.data_channels = int(self.model_config["out_channels"])
        self.cond_channels = int(self.model_config["in_channels"]) - self.data_channels

    def _prepare_condition(self, x: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if self.cond_channels <= 0:
            return x.new_zeros((x.shape[0], 0, x.shape[2], x.shape[3]))
        if condition is None:
            return x.new_zeros((x.shape[0], self.cond_channels, x.shape[2], x.shape[3]))
        condition = condition.to(device=x.device, dtype=x.dtype)
        if condition.dim() == 3:
            condition = condition.unsqueeze(0)
        if condition.shape[-2:] != x.shape[-2:]:
            condition = F.interpolate(condition, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if condition.shape[1] < self.cond_channels:
            pad = x.new_zeros((condition.shape[0], self.cond_channels - condition.shape[1], *condition.shape[-2:]))
            condition = torch.cat([condition, pad], dim=1)
        return condition[:, : self.cond_channels]

    @torch.no_grad()
    def predict_velocity(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        cond = self._prepare_condition(x, condition)
        model_in = torch.cat([x, cond], dim=1).to(dtype=self.model_dtype)
        timesteps = (t * float(self.num_train_timesteps)).to(device=x.device)
        pred = self.model(model_in, timesteps).sample
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
        condition: torch.Tensor | None,
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
        if condition is not None:
            condition = condition.to(self.device).float()
            if condition.dim() == 3:
                condition = condition.unsqueeze(0)

        x = torch.randn(
            measurement.shape,
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="RadioMap-UNet-FlowDPS")

        for i in iterator:
            t = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = t.expand(x.shape[0])
            v = self.predict_velocity(x, t_batch, condition)

            x0_pred = x - t * v
            x1_pred = x + (1.0 - t) * v
            x0_dc = self.data_consistency(
                x0_pred,
                measurement=measurement,
                mask=mask,
                step_size=step_size,
                num_iters=dc_iters,
            )
            x0_blend = (1.0 - t) * x0_pred + t * x0_dc
            x = ((1.0 - t_next) * x0_blend + t_next * x1_pred).clamp(-1.0, 1.0)

        return x.clamp(-1.0, 1.0)
