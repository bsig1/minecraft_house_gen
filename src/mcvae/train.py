from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from mcvae.data import MinecraftBuildDataset, split_dataset
from mcvae.io import ensure_dir, save_checkpoint, save_json
from mcvae.model import VAEConfig, VoxelVAE, build_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VAE on Minecraft NPZ voxel builds.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--palette-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--air-loss-weight", type=float, default=0.15)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: VoxelVAE,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    beta: float,
    air_loss_weight: float,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "recon": 0.0, "kld": 0.0}
    batches = 0

    for batch in loader:
        blocks = batch["blocks"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
                logits, mu, logvar = model(blocks)
                loss, stats = build_loss(
                    logits,
                    blocks,
                    mu,
                    logvar,
                    beta=beta,
                    air_loss_weight=air_loss_weight,
                )

            if is_train:
                assert optimizer is not None
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        for key in totals:
            totals[key] += stats[key]
        batches += 1

    return {key: value / max(1, batches) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = ensure_dir(args.output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")

    dataset = MinecraftBuildDataset(args.data_dir, args.palette_json, limit=args.limit)
    train_dataset, val_dataset = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, shuffle=False)

    model_config = VAEConfig(
        vocab_size=dataset.vocab_size,
        embedding_dim=args.embedding_dim,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
    )
    device = torch.device(args.device)
    model = VoxelVAE(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    config = {
        "data_dir": str(args.data_dir.resolve()),
        "palette_json": str(args.palette_json.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "beta": args.beta,
        "air_loss_weight": args.air_loss_weight,
        "embedding_dim": args.embedding_dim,
        "latent_dim": args.latent_dim,
        "base_channels": args.base_channels,
        "val_ratio": args.val_ratio,
        "num_workers": args.num_workers,
        "limit": args.limit,
        "seed": args.seed,
        "device": str(device),
        "palette_size": dataset.vocab_size,
    }
    save_json(output_dir / "config.json", config)

    best_val_loss = float("inf")
    metrics_path = output_dir / "metrics.jsonl"

    for epoch in tqdm(range(1, args.epochs + 1),desc="Epochs"):
        started = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            beta=args.beta,
            air_loss_weight=args.air_loss_weight,
            scaler=scaler,
            amp_enabled=use_amp,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            beta=args.beta,
            air_loss_weight=args.air_loss_weight,
            scaler=None,
            amp_enabled=False,
        )
        epoch_seconds = time.time() - started
        row = {
            "epoch": epoch,
            "seconds": round(epoch_seconds, 2),
            "train": train_metrics,
            "val": val_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"train_recon={train_metrics['recon']:.4f} "
            f"val_recon={val_metrics['recon']:.4f} "
            f"val_kld={val_metrics['kld']:.4f} "
            f"sec={epoch_seconds:.1f}"
        )

        save_checkpoint(
            checkpoints_dir / "last.pt",
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            config=config,
            epoch=epoch,
            best_val_loss=best_val_loss,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                checkpoints_dir / "best.pt",
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                config=config,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )


if __name__ == "__main__":
    main()
