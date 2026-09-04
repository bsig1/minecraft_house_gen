from __future__ import annotations

import math
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
    pos_embed_dim: int = 0


def _sinusoidal_1d_embedding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """Create sinusoidal embeddings for a single axis."""
    half: int = dim // 2
    scale: float = math.log(10000) / max(1, half - 1)
    frequencies: torch.Tensor = torch.exp(torch.arange(half, device=device) * -scale)
    args: torch.Tensor = torch.arange(length, device=device).float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding: torch.Tensor = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def get_3d_sinusoidal_encoding(depth: int, height: int, width: int, dim: int, device: torch.device) -> torch.Tensor:
    """Generate a 3D sinusoidal positional encoding tensor of shape (dim, depth, height, width)."""
    dim_z = dim // 3
    dim_y = (dim - dim_z) // 2
    dim_x = dim - dim_z - dim_y

    emb_z = _sinusoidal_1d_embedding(depth, dim_z, device).view(depth, 1, 1, dim_z).expand(depth, height, width, dim_z)
    emb_y = _sinusoidal_1d_embedding(height, dim_y, device).view(1, height, 1, dim_y).expand(depth, height, width, dim_y)
    emb_x = _sinusoidal_1d_embedding(width, dim_x, device).view(1, 1, width, dim_x).expand(depth, height, width, dim_x)

    # Concatenate along the channel dimension and permute to (channel, depth, height, width)
    grid = torch.cat([emb_z, emb_y, emb_x], dim=-1)
    return grid.permute(3, 0, 1, 2)


class VoxelVAE(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        """Construct a 3D convolutional VAE for voxel block IDs."""
        super().__init__()
        self.config = config
        c: int = config.base_channels
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.encoder = nn.Sequential(
            nn.Conv3d(config.embedding_dim + config.pos_embed_dim, c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(c),
            nn.SiLU(),
            nn.Conv3d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(c * 2),
            nn.SiLU(),
            nn.Conv3d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(c * 4),
            nn.SiLU(),
        )
        hidden_dim: int = c * 4 * 4 * 4 * 4
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
        self.output_head = nn.Conv3d(c + config.pos_embed_dim, config.vocab_size, kernel_size=1)

    def encode(self, blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode tokenized voxel grids into latent mean and log-variance."""
        embedded: torch.Tensor = self.embedding(blocks).permute(0, 4, 1, 2, 3)
        
        if self.config.pos_embed_dim > 0:
            d, h, w = embedded.shape[2:]
            pos_emb = get_3d_sinusoidal_encoding(d, h, w, self.config.pos_embed_dim, embedded.device)
            # Broadcast positional embedding to match batch size
            pos_emb = pos_emb.unsqueeze(0).expand(embedded.size(0), -1, -1, -1, -1)
            embedded = torch.cat([embedded, pos_emb], dim=1)
            
        hidden: torch.Tensor = self.encoder(embedded).flatten(start_dim=1)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent vectors using the reparameterization trick."""
        std: torch.Tensor = torch.exp(0.5 * logvar)
        eps: torch.Tensor = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into per-voxel class logits."""
        c: int = self.config.base_channels
        hidden: torch.Tensor = self.decoder_input(latent).view(latent.size(0), c * 4, 4, 4, 4)
        decoded: torch.Tensor = self.decoder(hidden)
        
        if self.config.pos_embed_dim > 0:
            d, h, w = decoded.shape[2:]
            pos_emb = get_3d_sinusoidal_encoding(d, h, w, self.config.pos_embed_dim, decoded.device)
            pos_emb = pos_emb.unsqueeze(0).expand(decoded.size(0), -1, -1, -1, -1)
            decoded = torch.cat([decoded, pos_emb], dim=1)
            
        return self.output_head(decoded)

    def forward(self, blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run full VAE forward pass and return logits plus posterior stats."""
        mu: torch.Tensor
        logvar: torch.Tensor
        mu, logvar = self.encode(blocks)
        latent: torch.Tensor = self.reparameterize(mu, logvar)
        logits: torch.Tensor = self.decode(latent)
        return logits, mu, logvar

    def sample(self, count: int, device: torch.device) -> torch.Tensor:
        """Generate voxel samples from random latent vectors."""
        latent: torch.Tensor = torch.randn(count, self.config.latent_dim, device=device)
        logits: torch.Tensor = self.decode(latent)
        return logits.argmax(dim=1)

    def reconstruct(self, blocks: torch.Tensor) -> torch.Tensor:
        """Reconstruct input voxel grids with argmax decoding."""
        logits: torch.Tensor
        logits, _, _ = self.forward(blocks)
        return logits.argmax(dim=1)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Return batch-mean KL divergence between posterior and unit Gaussian."""
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
    """Compute weighted reconstruction plus KL loss for VAE training."""
    class_weights: torch.Tensor = torch.ones(
        logits.size(1), device=logits.device, dtype=logits.dtype
    )
    class_weights[0] = air_loss_weight
    recon: torch.Tensor = F.cross_entropy(logits, targets, weight=class_weights)
    kld: torch.Tensor = kl_divergence(mu, logvar)
    loss: torch.Tensor = recon + beta * kld
    return loss, {
        "loss": float(loss.detach().item()),
        "recon": float(recon.detach().item()),
        "kld": float(kld.detach().item()),
    }
