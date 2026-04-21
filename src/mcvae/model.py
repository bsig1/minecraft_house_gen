from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VAEConfig:
    vocab_size: int
    embedding_dim: int = 32
    latent_dim: int = 256
    base_channels: int = 32


class VoxelVAE(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        c = config.base_channels
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(config.embedding_dim, c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(c),
            nn.SiLU(),
            nn.Conv3d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(c * 2),
            nn.SiLU(),
            nn.Conv3d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(c * 4),
            nn.SiLU(),
        )
        hidden_dim = c * 4 * 4 * 4 * 4
        self.fc_mu = nn.Linear(hidden_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, config.latent_dim)
        self.decoder_input = nn.Linear(config.latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(c * 2),
            nn.SiLU(),
            nn.ConvTranspose3d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(c),
            nn.SiLU(),
            nn.ConvTranspose3d(c, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(c),
            nn.SiLU(),
        )
        self.output_head = nn.Conv3d(c, config.vocab_size, kernel_size=1)

    def encode(self, blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(blocks).permute(0, 4, 1, 2, 3)
        hidden = self.encoder(embedded).flatten(start_dim=1)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        c = self.config.base_channels
        hidden = self.decoder_input(latent).view(latent.size(0), c * 4, 4, 4, 4)
        return self.output_head(self.decoder(hidden))

    def forward(self, blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(blocks)
        latent = self.reparameterize(mu, logvar)
        logits = self.decode(latent)
        return logits, mu, logvar

    def sample(self, count: int, device: torch.device) -> torch.Tensor:
        latent = torch.randn(count, self.config.latent_dim, device=device)
        logits = self.decode(latent)
        return logits.argmax(dim=1)

    def reconstruct(self, blocks: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward(blocks)
        return logits.argmax(dim=1)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def build_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float,
    air_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    class_weights = torch.ones(logits.size(1), device=logits.device, dtype=logits.dtype)
    class_weights[0] = air_loss_weight
    recon = F.cross_entropy(logits, targets, weight=class_weights)
    kld = kl_divergence(mu, logvar)
    loss = recon + beta * kld
    return loss, {
        "loss": float(loss.detach().item()),
        "recon": float(recon.detach().item()),
        "kld": float(kld.detach().item()),
    }
