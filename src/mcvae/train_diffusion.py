from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mcvae.data import MinecraftBuildDataset, split_dataset
from mcvae.diffusion import LatentDiffusion, LatentDiffusionConfig
from mcvae.io import ensure_dir, load_checkpoint, save_checkpoint, save_json
from mcvae.model import VAEConfig, VoxelVAE


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for latent diffusion training."""
    parser = argparse.ArgumentParser(
        description="Train latent diffusion on top of a pretrained Minecraft VAE."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--palette-json", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--time-embed-dim", type=int, default=256)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument(
        "--use-posterior-sample",
        action="store_true",
        help="Train on sampled z from q(z|x) instead of deterministic mu.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for latent diffusion training."""
    return build_parser().parse_args(argv)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: Dataset[dict[str, torch.Tensor | str]],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    """Create a dataloader for train or validation splits."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_vae(checkpoint_path: Path, device: torch.device) -> tuple[VoxelVAE, dict]:
    """Load a pretrained VAE checkpoint and freeze all parameters."""
    checkpoint: dict[str, Any] = load_checkpoint(checkpoint_path, map_location=device)
    config: dict[str, Any] = checkpoint["config"]
    vae: VoxelVAE = VoxelVAE(
        VAEConfig(
            vocab_size=config["palette_size"],
            embedding_dim=config["embedding_dim"],
            latent_dim=config["latent_dim"],
            base_channels=config["base_channels"],
        )
    ).to(device)
    vae.load_state_dict(checkpoint["model_state"])
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False
    return vae, config


def run_epoch(
    diffusion: LatentDiffusion,
    vae: VoxelVAE,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    use_posterior_sample: bool,
) -> dict[str, float]:
    """Run one epoch of latent diffusion optimization or validation."""
    is_train: bool = optimizer is not None
    diffusion.train(is_train)
    totals: dict[str, float] = {"loss": 0.0}
    batches: int = 0

    for batch in loader:
        blocks: torch.Tensor = batch["blocks"].to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            mu: torch.Tensor
            logvar: torch.Tensor
            mu, logvar = vae.encode(blocks)
            latent: torch.Tensor = vae.reparameterize(mu, logvar) if use_posterior_sample else mu

        with torch.set_grad_enabled(is_train):
            autocast_device: str = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
                loss: torch.Tensor = diffusion.loss(latent)

            if is_train:
                assert optimizer is not None
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
                    optimizer.step()

        totals["loss"] += float(loss.detach().item())
        batches += 1

    return {"loss": totals["loss"] / max(1, batches)}


def main(argv: Sequence[str] | None = None) -> None:
    """Train latent diffusion on frozen VAE latents and save checkpoints."""
    args = parse_args(argv)
    seed_everything(args.seed)

    output_dir: Path = ensure_dir(args.output_dir)
    checkpoints_dir: Path = ensure_dir(output_dir / "checkpoints")

    dataset: MinecraftBuildDataset = MinecraftBuildDataset(
        args.data_dir, args.palette_json, limit=args.limit
    )
    train_dataset, val_dataset = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)
    train_loader: DataLoader = make_loader(
        train_dataset, args.batch_size, args.num_workers, shuffle=True
    )
    val_loader: DataLoader = make_loader(
        val_dataset, args.batch_size, args.num_workers, shuffle=False
    )

    device: torch.device = torch.device(args.device)
    vae: VoxelVAE
    vae_config: dict[str, Any]
    vae, vae_config = load_vae(args.vae_checkpoint, device)
    if vae_config["palette_size"] != dataset.vocab_size:
        raise ValueError(
            "VAE palette size does not match current dataset palette size. "
            "Use the same palette used for VAE training."
        )

    diffusion_config: LatentDiffusionConfig = LatentDiffusionConfig(
        latent_dim=vae_config["latent_dim"],
        hidden_dim=args.hidden_dim,
        time_embed_dim=args.time_embed_dim,
        num_timesteps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    diffusion: LatentDiffusion = LatentDiffusion(diffusion_config).to(device)
    optimizer: torch.optim.AdamW = torch.optim.AdamW(
        diffusion.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    use_amp: bool = bool(args.amp and device.type == "cuda")
    scaler: torch.amp.GradScaler | None = torch.amp.GradScaler("cuda") if use_amp else None

    config: dict[str, Any] = {
        "model_type": "latent_diffusion",
        "data_dir": str(args.data_dir.resolve()),
        "palette_json": str(args.palette_json.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "val_ratio": args.val_ratio,
        "num_workers": args.num_workers,
        "limit": args.limit,
        "seed": args.seed,
        "device": str(device),
        "palette_size": dataset.vocab_size,
        "latent_dim": diffusion_config.latent_dim,
        "hidden_dim": diffusion_config.hidden_dim,
        "time_embed_dim": diffusion_config.time_embed_dim,
        "diffusion_steps": diffusion_config.num_timesteps,
        "beta_start": diffusion_config.beta_start,
        "beta_end": diffusion_config.beta_end,
        "use_posterior_sample": args.use_posterior_sample,
    }
    save_json(output_dir / "config.json", config)

    best_val_loss: float = float("inf")
    metrics_path: Path = output_dir / "metrics.jsonl"

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        started: float = time.time()
        train_metrics: dict[str, float] = run_epoch(
            diffusion,
            vae,
            train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=use_amp,
            use_posterior_sample=args.use_posterior_sample,
        )
        val_metrics: dict[str, float] = run_epoch(
            diffusion,
            vae,
            val_loader,
            optimizer=None,
            device=device,
            scaler=None,
            amp_enabled=False,
            use_posterior_sample=args.use_posterior_sample,
        )
        epoch_seconds: float = time.time() - started
        row: dict[str, Any] = {
            "epoch": epoch,
            "seconds": round(epoch_seconds, 2),
            "train": train_metrics,
            "val": val_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"sec={epoch_seconds:.1f}"
        )

        save_checkpoint(
            checkpoints_dir / "last.pt",
            model_state=diffusion.state_dict(),
            optimizer_state=optimizer.state_dict(),
            config=config,
            epoch=epoch,
            best_val_loss=best_val_loss,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                checkpoints_dir / "best.pt",
                model_state=diffusion.state_dict(),
                optimizer_state=optimizer.state_dict(),
                config=config,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )


if __name__ == "__main__":
    main()
