from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = int(timesteps) + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5).square()
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).float()


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(float(beta_start), float(beta_end), int(timesteps), dtype=torch.float32)


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    output = values.to(timesteps.device).gather(0, timesteps)
    return output.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


@dataclass
class DiffusionSchedule:
    timesteps: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_previous: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor

    @classmethod
    def create(cls, timesteps: int = 1000, schedule: str = "cosine") -> "DiffusionSchedule":
        betas = cosine_beta_schedule(timesteps) if schedule == "cosine" else linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        previous = F.pad(cumulative[:-1], (1, 0), value=1.0)
        return cls(
            timesteps=int(timesteps),
            betas=betas,
            alphas=alphas,
            alphas_cumprod=cumulative,
            alphas_cumprod_previous=previous,
            sqrt_alphas_cumprod=torch.sqrt(cumulative),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - cumulative),
            sqrt_recip_alphas_cumprod=torch.sqrt(1.0 / cumulative),
            sqrt_recipm1_alphas_cumprod=torch.sqrt(1.0 / cumulative - 1.0),
        )

    def to(self, device: torch.device | str) -> "DiffusionSchedule":
        values = {
            name: getattr(self, name).to(device) if isinstance(getattr(self, name), torch.Tensor) else getattr(self, name)
            for name in self.__dataclass_fields__
        }
        return DiffusionSchedule(**values)

    def q_sample(self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean)
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, clean.shape) * clean
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean.shape) * noise
        )

    def predict_start_from_noise(self, noisy: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, timesteps, noisy.shape) * noisy
            - extract(self.sqrt_recipm1_alphas_cumprod, timesteps, noisy.shape) * noise
        )


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.shadow = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            if name not in self.shadow:
                self.shadow[name] = value.detach().clone()
                continue
            if not value.is_floating_point():
                self.shadow[name].copy_(value)
            else:
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow", {})
        if isinstance(shadow, dict):
            self.shadow = {name: value.detach().clone() for name, value in shadow.items()}

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        # Swap EMA weights into the existing module instead of constructing a
        # second CUDA model for every validation and preview pass.
        current = {name: value.detach().clone() for name, value in model.state_dict().items()}
        self.copy_to(model)
        try:
            yield model
        finally:
            model.load_state_dict(current, strict=True)


@torch.no_grad()
def ddim_sample(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, ...],
    *,
    content_proxy: torch.Tensor,
    style_experts: torch.Tensor,
    steps: int = 64,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
    clip_latent: float = 4.0,
    callback: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    device = content_proxy.device
    if initial_noise is None:
        image = torch.randn(shape, device=device, generator=generator)
    else:
        image = initial_noise.to(device)
    total = schedule.timesteps
    count = max(2, min(int(steps), total))
    times = torch.linspace(total - 1, 0, count, device=device).long()
    previous_times = torch.cat([times[1:], torch.tensor([-1], device=device, dtype=torch.long)])

    for index, (time_value, previous_value) in enumerate(zip(times, previous_times)):
        time = torch.full((shape[0],), int(time_value.item()), device=device, dtype=torch.long)
        predicted_noise = model(image, time, content_proxy, style_experts)
        alpha = schedule.alphas_cumprod.to(device)[time_value]
        alpha_previous = (
            schedule.alphas_cumprod.to(device)[previous_value]
            if int(previous_value.item()) >= 0
            else torch.tensor(1.0, device=device)
        )
        clean = (image - torch.sqrt(1.0 - alpha) * predicted_noise) / torch.sqrt(alpha)
        clean = clean.clamp(-float(clip_latent), float(clip_latent))
        sigma = float(eta) * torch.sqrt(
            ((1.0 - alpha_previous) / (1.0 - alpha)).clamp_min(0.0)
            * (1.0 - alpha / alpha_previous).clamp_min(0.0)
        )
        direction = torch.sqrt((1.0 - alpha_previous - sigma.square()).clamp_min(0.0)) * predicted_noise
        noise = torch.randn(image.shape, device=device, generator=generator) if int(previous_value.item()) >= 0 else 0.0
        image = torch.sqrt(alpha_previous) * clean + direction + sigma * noise
        if callback is not None:
            callback(index, image)
    return image


@torch.no_grad()
def classifier_free_guidance_sample(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, ...],
    *,
    content_proxy: torch.Tensor,
    style_experts: torch.Tensor,
    null_style_experts: torch.Tensor | None,
    guidance_scale: float,
    steps: int,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if null_style_experts is None or float(guidance_scale) == 1.0:
        return ddim_sample(
            model,
            schedule,
            shape,
            content_proxy=content_proxy,
            style_experts=style_experts,
            steps=steps,
            eta=eta,
            generator=generator,
            initial_noise=initial_noise,
        )

    device = content_proxy.device
    image = torch.randn(shape, device=device, generator=generator) if initial_noise is None else initial_noise.to(device)
    total = schedule.timesteps
    count = max(2, min(int(steps), total))
    times = torch.linspace(total - 1, 0, count, device=device).long()
    previous_times = torch.cat([times[1:], torch.tensor([-1], device=device, dtype=torch.long)])
    alphas = schedule.alphas_cumprod.to(device)
    for time_value, previous_value in zip(times, previous_times):
        time = torch.full((shape[0],), int(time_value.item()), device=device, dtype=torch.long)
        # Conditional and unconditional predictions share the same image and
        # content features. Running them as one doubled batch removes one full
        # UNet launch per DDIM step.
        doubled = model(
            torch.cat([image, image], dim=0),
            torch.cat([time, time], dim=0),
            torch.cat([content_proxy, content_proxy], dim=0),
            torch.cat([null_style_experts, style_experts], dim=0),
        )
        unconditional, conditional = doubled.chunk(2, dim=0)
        predicted_noise = unconditional + float(guidance_scale) * (conditional - unconditional)
        alpha = alphas[time_value]
        alpha_previous = alphas[previous_value] if int(previous_value.item()) >= 0 else torch.tensor(1.0, device=device)
        clean = (image - torch.sqrt(1.0 - alpha) * predicted_noise) / torch.sqrt(alpha)
        clean = clean.clamp(-4.0, 4.0)
        sigma = float(eta) * torch.sqrt(
            ((1.0 - alpha_previous) / (1.0 - alpha)).clamp_min(0.0)
            * (1.0 - alpha / alpha_previous).clamp_min(0.0)
        )
        direction = torch.sqrt((1.0 - alpha_previous - sigma.square()).clamp_min(0.0)) * predicted_noise
        noise = torch.randn(image.shape, device=device, generator=generator) if int(previous_value.item()) >= 0 else 0.0
        image = torch.sqrt(alpha_previous) * clean + direction + sigma * noise
    return image
