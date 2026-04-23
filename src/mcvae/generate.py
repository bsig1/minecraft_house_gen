from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from mcvae.data import discover_npz_files, load_palette
from mcvae.diffusion import LatentDiffusion, LatentDiffusionConfig
from mcvae.io import ensure_dir, load_checkpoint, write_build_npz, write_structure_nbt
from mcvae.model import VAEConfig, VoxelVAE


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI options controlling output formats and structure metadata."""
    parser.add_argument(
        "--format",
        choices=["npz", "structure", "both"],
        default="npz",
        help="Which output file format to write.",
    )
    parser.add_argument(
        "--structure-author",
        type=str,
        default="mcvae",
        help="Author tag for exported Minecraft structure files.",
    )
    parser.add_argument(
        "--structure-data-version",
        type=int,
        default=None,
        help="Optional DataVersion tag for structure NBT output.",
    )


def add_sample_quality_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI options used to filter and randomize generated samples."""
    parser.add_argument(
        "--min-non-air",
        type=int,
        default=256,
        help="Reject generated samples with fewer than this many non-air blocks.",
    )
    parser.add_argument(
        "--max-sample-attempts",
        type=int,
        default=200,
        help="Maximum latent samples to try while filtering trivial outputs.",
    )
    parser.add_argument(
        "--air-logit-penalty",
        type=float,
        default=0.0,
        help="Subtract this value from the air logit before decoding samples.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature used with --sample-voxels.",
    )
    parser.add_argument(
        "--sample-voxels",
        action="store_true",
        help="Sample each voxel from probabilities instead of taking argmax.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for sample and reconstruct commands."""
    parser = argparse.ArgumentParser(description="Sample or reconstruct Minecraft voxel builds.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="Generate new builds from the latent prior.")
    sample.add_argument("--checkpoint", type=Path, required=True)
    sample.add_argument("--output-dir", type=Path, required=True)
    sample.add_argument("--count", type=int, default=8)
    sample.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    add_output_args(sample)
    add_sample_quality_args(sample)

    sample_diffusion = subparsers.add_parser(
        "sample-diffusion",
        help="Generate new builds with latent diffusion plus a pretrained VAE decoder.",
    )
    sample_diffusion.add_argument("--vae-checkpoint", type=Path, required=True)
    sample_diffusion.add_argument("--diffusion-checkpoint", type=Path, required=True)
    sample_diffusion.add_argument("--output-dir", type=Path, required=True)
    sample_diffusion.add_argument("--count", type=int, default=8)
    sample_diffusion.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    add_output_args(sample_diffusion)
    add_sample_quality_args(sample_diffusion)

    reconstruct = subparsers.add_parser(
        "reconstruct",
        help="Reconstruct one build or a directory of builds with the trained VAE.",
    )
    reconstruct.add_argument("--checkpoint", type=Path, required=True)
    reconstruct.add_argument("--input", type=Path, required=True)
    reconstruct.add_argument("--output-dir", type=Path, required=True)
    reconstruct.add_argument("--limit", type=int, default=None)
    reconstruct.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    add_output_args(reconstruct)

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for sample and reconstruct commands."""
    return build_parser().parse_args(argv)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[VoxelVAE, dict]:
    """Load a trained VAE checkpoint and return model plus saved config."""
    checkpoint: dict[str, Any] = load_checkpoint(checkpoint_path, map_location=device)
    config: dict[str, Any] = checkpoint["config"]
    model: VoxelVAE = VoxelVAE(
        VAEConfig(
            vocab_size=config["palette_size"],
            embedding_dim=config["embedding_dim"],
            latent_dim=config["latent_dim"],
            base_channels=config["base_channels"],
        )
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def load_diffusion_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[LatentDiffusion, dict]:
    """Load a trained latent diffusion checkpoint and return model plus config."""
    checkpoint: dict[str, Any] = load_checkpoint(checkpoint_path, map_location=device)
    config: dict[str, Any] = checkpoint["config"]
    model: LatentDiffusion = LatentDiffusion(
        LatentDiffusionConfig(
            latent_dim=config["latent_dim"],
            hidden_dim=config["hidden_dim"],
            time_embed_dim=config["time_embed_dim"],
            num_timesteps=config["diffusion_steps"],
            beta_start=config["beta_start"],
            beta_end=config["beta_end"],
        )
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def write_outputs(
    output_stem: Path,
    blocks: np.ndarray,
    *,
    args: argparse.Namespace,
    palette: list[str] | None,
    original_shape: tuple[int, int, int],
    axis_order: str,
    source_format: str,
    dataset_path: str,
) -> None:
    """Write one generated build as `.npz`, `.nbt`, or both."""
    if args.format in {"npz", "both"}:
        npz_path: Path = output_stem.with_suffix(".npz")
        write_build_npz(
            npz_path,
            blocks,
            original_shape=original_shape,
            axis_order=axis_order,
            source_format=source_format,
            dataset_path=dataset_path,
        )
        print(f"wrote {npz_path}")

    if args.format in {"structure", "both"}:
        if palette is None:
            raise ValueError("Structure output requires a palette.")
        structure_path: Path = output_stem.with_suffix(".nbt")
        write_structure_nbt(
            structure_path,
            blocks,
            palette=palette,
            original_shape=original_shape,
            axis_order=axis_order,
            author=args.structure_author,
            data_version=args.structure_data_version,
        )
        print(f"wrote {structure_path}")


def decode_latent_samples(
    model: VoxelVAE,
    *,
    batch_size: int,
    device: torch.device,
    air_logit_penalty: float,
    sample_voxels: bool,
    temperature: float,
) -> torch.Tensor:
    """Decode latent vectors into voxel IDs via argmax or multinomial sampling."""
    latent: torch.Tensor = torch.randn(batch_size, model.config.latent_dim, device=device)
    logits: torch.Tensor = model.decode(latent)
    if air_logit_penalty:
        logits[:, 0] = logits[:, 0] - air_logit_penalty
    if sample_voxels:
        if temperature <= 0:
            raise ValueError("--temperature must be greater than 0.")
        probs: torch.Tensor = torch.softmax(logits / temperature, dim=1)
        flat_probs: torch.Tensor = probs.permute(0, 2, 3, 4, 1).reshape(
            -1, model.config.vocab_size
        )
        sampled: torch.Tensor = torch.multinomial(flat_probs, num_samples=1).view(
            batch_size, 32, 32, 32
        )
        return sampled
    return logits.argmax(dim=1)


def generate_filtered_samples(
    model: VoxelVAE,
    *,
    count: int,
    device: torch.device,
    min_non_air: int,
    max_attempts: int,
    air_logit_penalty: float,
    sample_voxels: bool,
    temperature: float,
) -> list[np.ndarray]:
    """Generate samples and keep only those with enough non-air voxels."""
    accepted: list[np.ndarray] = []
    attempts: int = 0
    while len(accepted) < count and attempts < max_attempts:
        batch_size: int = min(count - len(accepted), 8, max_attempts - attempts)
        decoded: torch.Tensor = decode_latent_samples(
            model,
            batch_size=batch_size,
            device=device,
            air_logit_penalty=air_logit_penalty,
            sample_voxels=sample_voxels,
            temperature=temperature,
        )
        attempts += batch_size
        for sample in decoded.cpu().numpy().astype(np.int32):
            non_air: int = int((sample != 0).sum())
            if non_air >= min_non_air:
                accepted.append(sample)
                print(f"accepted sample {len(accepted)}/{count} non_air={non_air}")
            else:
                print(f"rejected trivial sample non_air={non_air}")
            if len(accepted) == count:
                break

    if len(accepted) < count:
        raise RuntimeError(
            f"Only accepted {len(accepted)} of {count} samples after {attempts} attempts. "
            "Try lowering --min-non-air or increasing --air-logit-penalty."
        )
    return accepted


def run_sample(args: argparse.Namespace) -> None:
    """Generate voxel builds from the VAE latent prior."""
    device: torch.device = torch.device(args.device)
    model: VoxelVAE
    config: dict[str, Any]
    model, config = load_model(args.checkpoint, device)
    output_dir: Path = ensure_dir(args.output_dir)
    palette: list[str] | None = (
        load_palette(config["palette_json"]) if args.format in {"structure", "both"} else None
    )
    with torch.no_grad():
        samples: list[np.ndarray] = generate_filtered_samples(
            model,
            count=args.count,
            device=device,
            min_non_air=args.min_non_air,
            max_attempts=args.max_sample_attempts,
            air_logit_penalty=args.air_logit_penalty,
            sample_voxels=args.sample_voxels,
            temperature=args.temperature,
        )
    for index, blocks in enumerate(samples):
        write_outputs(
            output_dir / f"sample_{index:04d}",
            blocks,
            args=args,
            palette=palette,
            original_shape=(32, 32, 32),
            axis_order="YZX",
            source_format="ai_sample",
            dataset_path="generated/sample",
        )


def run_sample_diffusion(args: argparse.Namespace) -> None:
    """Generate voxel builds by sampling latents from the diffusion model."""
    device: torch.device = torch.device(args.device)
    vae: VoxelVAE
    vae_config: dict[str, Any]
    vae, vae_config = load_model(args.vae_checkpoint, device)
    diffusion: LatentDiffusion
    diffusion_config: dict[str, Any]
    diffusion, diffusion_config = load_diffusion_model(args.diffusion_checkpoint, device)

    if int(diffusion_config["latent_dim"]) != int(vae.config.latent_dim):
        raise ValueError(
            "Diffusion latent_dim does not match VAE latent_dim. "
            "Use matching checkpoints."
        )

    output_dir: Path = ensure_dir(args.output_dir)
    palette: list[str] | None = (
        load_palette(vae_config["palette_json"]) if args.format in {"structure", "both"} else None
    )

    accepted: list[np.ndarray] = []
    attempts: int = 0
    max_attempts: int = args.max_sample_attempts
    with torch.no_grad():
        while len(accepted) < args.count and attempts < max_attempts:
            batch_size: int = min(args.count - len(accepted), 8, max_attempts - attempts)
            latent: torch.Tensor = diffusion.sample(batch_size, device=device)
            logits: torch.Tensor = vae.decode(latent)
            if args.air_logit_penalty:
                logits[:, 0] = logits[:, 0] - args.air_logit_penalty
            if args.sample_voxels:
                if args.temperature <= 0:
                    raise ValueError("--temperature must be greater than 0.")
                probs: torch.Tensor = torch.softmax(logits / args.temperature, dim=1)
                flat_probs: torch.Tensor = probs.permute(0, 2, 3, 4, 1).reshape(
                    -1, vae.config.vocab_size
                )
                decoded: torch.Tensor = torch.multinomial(flat_probs, num_samples=1).view(
                    batch_size, 32, 32, 32
                )
            else:
                decoded = logits.argmax(dim=1)

            attempts += batch_size
            for sample in decoded.cpu().numpy().astype(np.int32):
                non_air: int = int((sample != 0).sum())
                if non_air >= args.min_non_air:
                    accepted.append(sample)
                    print(f"accepted sample {len(accepted)}/{args.count} non_air={non_air}")
                else:
                    print(f"rejected trivial sample non_air={non_air}")
                if len(accepted) == args.count:
                    break

    if len(accepted) < args.count:
        raise RuntimeError(
            f"Only accepted {len(accepted)} of {args.count} samples after {attempts} attempts. "
            "Try lowering --min-non-air or increasing --air-logit-penalty."
        )

    for index, blocks in enumerate(accepted):
        write_outputs(
            output_dir / f"sample_{index:04d}",
            blocks,
            args=args,
            palette=palette,
            original_shape=(32, 32, 32),
            axis_order="YZX",
            source_format="ai_sample_diffusion",
            dataset_path="generated/sample_diffusion",
        )


def load_blocks(path: Path) -> np.ndarray:
    """Load just the `blocks` array from a project `.npz` build file."""
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["blocks"], dtype=np.int64)


def load_build(path: Path) -> dict[str, object]:
    """Load a build file and return blocks plus key source metadata."""
    with np.load(path, allow_pickle=False) as data:
        return {
            "blocks": np.asarray(data["blocks"], dtype=np.int64),
            "original_shape": tuple(int(v) for v in data["original_shape"].tolist()),
            "axis_order": str(data["axis_order"].item()),
            "source_format": str(data["source_format"].item()),
            "dataset_path": str(data["dataset_path"].item()),
        }


def iter_inputs(input_path: Path, limit: int | None) -> list[Path]:
    """Return one input file or all discoverable `.npz` files in a directory."""
    if input_path.is_dir():
        return discover_npz_files(input_path, limit=limit)
    return [input_path]


def run_reconstruct(args: argparse.Namespace) -> None:
    """Reconstruct input builds with the VAE and write output artifacts."""
    device: torch.device = torch.device(args.device)
    model: VoxelVAE
    config: dict[str, Any]
    model, config = load_model(args.checkpoint, device)
    output_dir: Path = ensure_dir(args.output_dir)
    input_paths: list[Path] = iter_inputs(args.input, args.limit)
    palette: list[str] | None = (
        load_palette(config["palette_json"]) if args.format in {"structure", "both"} else None
    )

    with torch.no_grad():
        for path in input_paths:
            build: dict[str, object] = load_build(path)
            blocks: object = build["blocks"]
            assert isinstance(blocks, np.ndarray)
            tensor: torch.Tensor = torch.from_numpy(blocks).long().unsqueeze(0).to(device)
            reconstructed: np.ndarray = (
                model.reconstruct(tensor).squeeze(0).cpu().numpy().astype(np.int32)
            )
            write_outputs(
                output_dir / f"{path.stem}_recon",
                reconstructed,
                args=args,
                palette=palette,
                original_shape=tuple(int(v) for v in build["original_shape"]),
                axis_order=str(build["axis_order"]),
                source_format="ai_recon",
                dataset_path=f"reconstructed/{path.name}",
            )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for generation and reconstruction CLI commands."""
    args: argparse.Namespace = parse_args(argv)
    if args.command == "sample":
        run_sample(args)
        return
    if args.command == "sample-diffusion":
        run_sample_diffusion(args)
        return
    if args.command == "reconstruct":
        run_reconstruct(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
