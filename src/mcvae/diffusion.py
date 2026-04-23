from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LatentDiffusionConfig:
    latent_dim: int
    hidden_dim: int = 1024
    time_embed_dim: int = 256
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


def _sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings used by the denoiser network."""
    half: int = dim // 2
    scale: float = math.log(10000) / max(1, half - 1)
    frequencies: torch.Tensor = torch.exp(
        torch.arange(half, device=timesteps.device) * -scale)
    args: torch.Tensor = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding: torch.Tensor = torch.cat(
        [torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    """Gather schedule values at timesteps and reshape for broadcast math."""
    out: torch.Tensor = values.gather(0, timesteps)
    return out.view(timesteps.size(0), *([1] * (len(target_shape) - 1)))


class LatentDenoiser(nn.Module):
    def __init__(self, config: LatentDiffusionConfig) -> None:
        """Build the timestep-conditioned MLP that predicts latent noise."""
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embed_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim +
                      config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.time_embed_dim = config.time_embed_dim

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict Gaussian noise for the noised latent batch `x_t`."""
        t_embedding: torch.Tensor = _sinusoidal_timestep_embedding(
            timesteps, self.time_embed_dim
        )
        t_features: torch.Tensor = self.time_mlp(t_embedding)
        return self.net(torch.cat([x_t, t_features], dim=1))


class LatentDiffusion(nn.Module):
    def __init__(self, config: LatentDiffusionConfig) -> None:
        """Initialize diffusion schedules and the latent denoiser model."""
        super().__init__()
        self.config = config
        self.denoiser = LatentDenoiser(config)

        betas: torch.Tensor = torch.linspace(
            config.beta_start, config.beta_end, config.num_timesteps
        )
        alphas: torch.Tensor = 1.0 - betas
        alphas_cumprod: torch.Tensor = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev: torch.Tensor = torch.cat(
            [torch.ones(1), alphas_cumprod[:-1]])
        posterior_variance: torch.Tensor = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance",
                             posterior_variance.clamp(min=1e-20))

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffuse clean latents to timestep `timesteps`."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t: torch.Tensor = _extract(
            self.sqrt_alphas_cumprod, timesteps, x_start.shape
        )
        sqrt_one_minus_alpha_cumprod_t: torch.Tensor = _extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def predict_noise(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Run the denoiser for the provided latent batch and timesteps."""
        return self.denoiser(x_t, timesteps)

    def loss(self, x_start: torch.Tensor) -> torch.Tensor:
        """Compute the denoising objective on a clean latent batch."""
        batch_size: int = x_start.size(0)
        timesteps: torch.Tensor = torch.randint(
            low=0,
            high=self.config.num_timesteps,
            size=(batch_size,),
            device=x_start.device,
            dtype=torch.long,
        )
        noise: torch.Tensor = torch.randn_like(x_start)
        x_t: torch.Tensor = self.q_sample(x_start, timesteps, noise)
        predicted_noise: torch.Tensor = self.predict_noise(x_t, timesteps)
        return F.mse_loss(predicted_noise, noise)

    def p_sample(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Sample one reverse-diffusion step from `x_t` at `timesteps`."""
        betas_t: torch.Tensor = _extract(self.betas, timesteps, x_t.shape)
        sqrt_one_minus_alpha_cumprod_t: torch.Tensor = _extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape
        )
        sqrt_recip_alphas_t: torch.Tensor = _extract(
            self.sqrt_recip_alphas, timesteps, x_t.shape)
        predicted_noise: torch.Tensor = self.predict_noise(x_t, timesteps)
        model_mean: torch.Tensor = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t
        )
        posterior_variance_t: torch.Tensor = _extract(
            self.posterior_variance, timesteps, x_t.shape)
        noise: torch.Tensor = torch.randn_like(x_t)
        nonzero_mask: torch.Tensor = (timesteps != 0).float().view(
            x_t.size(0), *([1] * (x_t.dim() - 1))
        )
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    def sample(self, count: int, device: torch.device) -> torch.Tensor:
        """Generate `count` latent vectors via iterative reverse diffusion."""
        x_t: torch.Tensor = torch.randn(
            count, self.config.latent_dim, device=device)
        for step in reversed(range(self.config.num_timesteps)):
            timesteps: torch.Tensor = torch.full(
                (count,), step, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, timesteps)
        return x_t
