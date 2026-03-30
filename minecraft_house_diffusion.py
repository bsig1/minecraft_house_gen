import argparse
import shutil
import json
import math
import os
import random
import tarfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from npy_to_structure import build_structure_from_npy, load_legacy_palette
from ascii_to_structure import save_structure


DATASET_URL = "https://craftassist.s3-us-west-2.amazonaws.com/pubr/house_data.tar.gz"
NUM_BLOCK_TYPES = 256
MIN_EXPECTED_HOUSE_COUNT = 32
DEFAULT_NBT_PALETTE_JSON = "./legacy_id_palette.json"
DEFAULT_NBT_DATA_VERSION = 4189


# ------------------------------------------------------------
# Dataset download + parsing
# ------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_house_dirs(houses_dir: Path) -> List[Path]:
    if not houses_dir.exists():
        return []
    return sorted(
        house_dir
        for house_dir in houses_dir.iterdir()
        if house_dir.is_dir() and (house_dir / "placed.json").exists()
    )


def count_houses_in_archive(archive_path: Path) -> int:
    with tarfile.open(archive_path, "r:gz") as tar:
        return len(
            {
                Path(member.name).parts[1]
                for member in tar.getmembers()
                if len(Path(member.name).parts) >= 3
                and Path(member.name).parts[0] == "houses"
                and Path(member.name).parts[-1] == "placed.json"
            }
        )


def sanitize_path_part(part: str) -> str:
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in invalid else ch for ch in part).strip()
    cleaned = cleaned.rstrip(".")
    return cleaned or "_"


def extract_archive_safely(archive_path: Path, data_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting dataset", unit="file"):
            posix_parts = Path(member.name).parts
            if not posix_parts:
                continue

            sanitized_parts = [sanitize_path_part(part) for part in posix_parts]
            target_path = data_dir.joinpath(*sanitized_parts)

            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isfile():
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            with extracted, open(target_path, "wb") as f:
                shutil.copyfileobj(extracted, f)


def maybe_download_3dcraft(data_dir: Path) -> None:
    houses_dir = data_dir / "houses"
    current_house_count = len(list_house_dirs(houses_dir))

    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "houses.tar.gz"

    if not archive_path.exists():
        print(f"Downloading 3D-Craft to {archive_path} ...")
        resp = requests.get(DATASET_URL, timeout=120)
        resp.raise_for_status()
        archive_path.write_bytes(resp.content)

    archive_house_count = count_houses_in_archive(archive_path)
    should_extract = (
        not houses_dir.exists()
        or current_house_count == 0
        or (current_house_count < MIN_EXPECTED_HOUSE_COUNT and current_house_count < archive_house_count)
    )
    if not should_extract:
        return

    print(
        f"Extracting {archive_path} ..."
        f" ({current_house_count} houses on disk, {archive_house_count} available in archive)"
    )
    extract_archive_safely(archive_path, data_dir)


def load_splits(data_dir: Path) -> Dict[str, List[str]]:
    splits_path = data_dir / "splits.json"
    if splits_path.exists():
        with open(splits_path, "r") as f:
            return json.load(f)

    house_names = [house_dir.name for house_dir in list_house_dirs(data_dir / "houses")]
    if not house_names:
        raise FileNotFoundError(
            f"No house directories with placed.json were found under {data_dir / 'houses'}."
        )

    if len(house_names) == 1:
        return {"train": house_names, "val": house_names, "test": house_names}

    val_count = max(1, int(round(len(house_names) * 0.1)))
    if val_count >= len(house_names):
        val_count = len(house_names) - 1

    val_names = house_names[:val_count]
    train_names = house_names[val_count:]
    return {"train": train_names, "val": val_names, "test": val_names}


def load_final_house(annotation_path: Path) -> torch.Tensor:
    """
    Returns an [N, 4] int64 tensor with columns:
        [block_type, x, y, z]

    This mirrors the final-built-house logic used by the public VoxelCNN
    training code: it reads the placed.json action sequence, keeps only the
    final block present at each coordinate, and returns the blocks in build
    order of their final surviving placements.
    """
    with open(annotation_path, "r") as f:
        annotation = json.load(f)

    final_house = {}
    types_and_coords = []
    last_timestamp = -1

    for i, item in enumerate(annotation):
        timestamp, annotator_id, coordinate, block_info, action = item
        assert timestamp >= last_timestamp
        last_timestamp = timestamp

        coord = tuple(np.asarray(coordinate, dtype=np.int64).tolist())
        block_type = int(np.asarray(block_info, dtype=np.int64)[0])

        # In the released code, action == "B" removes a block; otherwise it is kept.
        if action == "B":
            final_house.pop(coord, None)
        else:
            final_house[coord] = i

        types_and_coords.append((block_type,) + coord)

    indices = sorted(final_house.values())
    types_and_coords = [types_and_coords[i] for i in indices]
    if not types_and_coords:
        return torch.zeros((0, 4), dtype=torch.int64)
    return torch.tensor(types_and_coords, dtype=torch.int64)


def center_crop_or_pad_house(
    house: torch.Tensor,
    grid_size: int,
    air_id: int,
    unk_original_id: int,
    block_id_map: Dict[int, int],
) -> torch.Tensor:
    """
    Convert a sparse [N,4] house to a dense [S,S,S] token grid.

    Steps:
    1. Recenter coordinates to the house bounding-box center.
    2. Place blocks into a fixed SxSxS voxel cube.
    3. Map raw Minecraft block ids to a reduced vocabulary.
    """
    grid = torch.full((grid_size, grid_size, grid_size), air_id, dtype=torch.long)
    if len(house) == 0:
        return grid

    coords = house[:, 1:].clone()
    raw_types = house[:, 0].clone()

    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values
    center = (mins + maxs).float() / 2.0

    target_center = torch.tensor([(grid_size - 1) / 2.0] * 3)
    shifted = torch.round(coords.float() - center + target_center).long()

    valid = ((shifted >= 0) & (shifted < grid_size)).all(dim=1)
    shifted = shifted[valid]
    raw_types = raw_types[valid]

    for block_type, (x, y, z) in zip(raw_types.tolist(), shifted.tolist()):
        token = block_id_map.get(block_type, unk_original_id)
        grid[x, y, z] = token

    return grid


class CraftHouseDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        subset: str,
        grid_size: int = 16,
        max_houses: int = 0,
        top_k_blocks: int = 24,
        min_blocks_per_house: int = 40,
        vocab_map: Dict[int, int] = None,
        raw_block_keep: List[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.grid_size = grid_size
        self.max_houses = max_houses
        self.top_k_blocks = top_k_blocks
        self.min_blocks_per_house = min_blocks_per_house

        maybe_download_3dcraft(self.data_dir)
        self.splits = load_splits(self.data_dir)
        if subset not in self.splits:
            raise ValueError(f"Unknown subset: {subset}")

        self.houses_sparse = []
        for rel in self.splits[subset]:
            annotation_path = self.data_dir / "houses" / rel / "placed.json"
            if not annotation_path.exists():
                continue
            house = load_final_house(annotation_path)
            if len(house) >= min_blocks_per_house:
                self.houses_sparse.append(house)
            if max_houses and len(self.houses_sparse) >= max_houses:
                break

        if vocab_map is None or raw_block_keep is None:
            counts = Counter()
            for house in self.houses_sparse:
                counts.update(house[:, 0].tolist())
            raw_block_keep = [b for b, _ in counts.most_common(top_k_blocks)]
            self.raw_block_keep = raw_block_keep

            # token 0 = AIR
            # token 1 = MASK
            # token 2 = OTHER/UNK kept for rare blocks
            self.block_id_map = {raw_id: i + 3 for i, raw_id in enumerate(raw_block_keep)}
        else:
            self.raw_block_keep = list(raw_block_keep)
            self.block_id_map = dict(vocab_map)

        self.air_token = 0
        self.mask_token = 1
        self.unk_token = 2
        self.vocab_size = 3 + len(self.raw_block_keep)

        self.houses_dense = [
            center_crop_or_pad_house(
                house,
                grid_size=self.grid_size,
                air_id=self.air_token,
                unk_original_id=self.unk_token,
                block_id_map=self.block_id_map,
            )
            for house in self.houses_sparse
        ]

    def __len__(self) -> int:
        return len(self.houses_dense)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.houses_dense[idx]


# ------------------------------------------------------------
# Discrete diffusion-style denoiser
# ------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class VoxelDenoiser(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 96, time_dim: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.in_proj = nn.Conv3d(hidden, hidden, 3, padding=1)
        self.net = nn.Sequential(
            ResBlock3D(hidden),
            ResBlock3D(hidden),
            ResBlock3D(hidden),
            ResBlock3D(hidden),
        )
        self.out = nn.Conv3d(hidden, vocab_size, 1)

    def forward(self, x_tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_tokens: [B, S, S, S]
        x = self.token_emb(x_tokens)                     # [B, S, S, S, C]
        x = x.permute(0, 4, 1, 2, 3).contiguous()       # [B, C, S, S, S]
        x = self.in_proj(x)

        time_bias = self.time_emb(t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x + time_bias
        x = self.net(x)
        return self.out(x)                              # [B, vocab, S, S, S]


# ------------------------------------------------------------
# Noise process and sampling
# ------------------------------------------------------------

def corrupt_with_mask(
    x0: torch.Tensor,
    t: torch.Tensor,
    num_steps: int,
    mask_token: int,
    air_token: int,
    air_mask_scale: float,
    solid_mask_scale: float,
) -> torch.Tensor:
    """
    Discrete diffusion-like corruption:
    each voxel is replaced by MASK with probability t / T, but non-air voxels
    can be masked more aggressively than air so the denoiser spends more
    capacity learning structure.
    """
    base_p = (t.float() / num_steps).view(-1, 1, 1, 1)
    mask_scale = torch.full_like(x0.float(), solid_mask_scale)
    mask_scale[x0 == air_token] = air_mask_scale
    p = (base_p * mask_scale).clamp_(0.0, 1.0)
    noise = torch.rand_like(x0.float())
    xt = x0.clone()
    xt[noise < p] = mask_token
    return xt


def build_loss_weights(
    vocab_size: int,
    air_token: int,
    unk_token: int,
    air_loss_weight: float,
    unk_loss_weight: float,
    device: str,
) -> torch.Tensor:
    weights = torch.ones(vocab_size, dtype=torch.float32, device=device)
    weights[air_token] = air_loss_weight
    if 0 <= unk_token < vocab_size:
        weights[unk_token] = unk_loss_weight
    return weights


@torch.no_grad()
def sample_structures(
    model: nn.Module,
    num_samples: int,
    grid_size: int,
    vocab_size: int,
    mask_token: int,
    air_token: int,
    num_steps: int,
    device: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    model.eval()
    x = torch.full((num_samples, grid_size, grid_size, grid_size), mask_token, dtype=torch.long, device=device)

    for step in tqdm(range(num_steps, 0, -1), desc="Sampling", leave=False):
        t = torch.full((num_samples,), step, device=device, dtype=torch.long)
        logits = model(x, t) / max(temperature, 1e-6)

        # Keep air somewhat likely; otherwise dense blobs are common early on.
        probs = F.softmax(logits, dim=1)
        sampled = torch.distributions.Categorical(probs=probs.permute(0, 2, 3, 4, 1)).sample()

        # Only reveal a fraction of still-masked voxels at each step.
        reveal_prob = 1.0 / step
        reveal = torch.rand_like(x.float()) < reveal_prob
        can_write = (x == mask_token) & reveal
        x[can_write] = sampled[can_write]

    # Any remaining masked tokens become air.
    x[x == mask_token] = air_token
    return x.cpu()


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def render_ascii_layers(grid: np.ndarray, token_names: Dict[int, str]) -> str:
    lines = []
    sx, sy, sz = grid.shape
    for y in range(sy):
        lines.append(f"Layer y={y}")
        for z in range(sz):
            row = []
            for x in range(sx):
                token = int(grid[x, y, z])
                if token == 0:
                    row.append(".")
                elif token == 1:
                    row.append("?")
                else:
                    row.append(token_names.get(token, "#")[:1])
            lines.append("".join(row))
        lines.append("")
    return "\n".join(lines)


def save_sample_outputs(
    out_dir: Path,
    samples: torch.Tensor,
    token_names: Dict[int, str],
    block_id_map: Dict[int, int] | None = None,
    nbt_palette_json: str | None = None,
    nbt_data_version: int = DEFAULT_NBT_DATA_VERSION,
    nbt_include_air: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_id_to_state = None
    fallback_unknown = None
    if block_id_map is not None and nbt_palette_json:
        palette_path = Path(nbt_palette_json)
        if not palette_path.exists():
            raise FileNotFoundError(f"NBT palette file not found: {palette_path}")
        raw_id_to_state, fallback_unknown = load_legacy_palette(palette_path)
        token_to_raw_id = {int(token): int(raw_id) for raw_id, token in block_id_map.items()}
    else:
        token_to_raw_id = None

    for i, sample in enumerate(samples):
        arr = sample.numpy().astype(np.int16)
        np.save(out_dir / f"sample_{i:03d}.npy", arr)
        with open(out_dir / f"sample_{i:03d}.txt", "w") as f:
            f.write(render_ascii_layers(arr, token_names))
        if token_to_raw_id is not None and raw_id_to_state is not None and fallback_unknown is not None:
            structure = build_structure_from_npy(
                grid=arr,
                token_to_raw_id=token_to_raw_id,
                raw_id_to_state=raw_id_to_state,
                fallback_unknown=fallback_unknown,
                include_air=nbt_include_air,
                data_version=nbt_data_version,
            )
            save_structure(structure, out_dir / f"sample_{i:03d}.nbt")


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    train_ds = CraftHouseDataset(
        data_dir=args.data_dir,
        subset="train",
        grid_size=args.grid_size,
        max_houses=args.max_train_houses,
        top_k_blocks=args.top_k_blocks,
        min_blocks_per_house=args.min_blocks_per_house,
    )
    val_ds = CraftHouseDataset(
        data_dir=args.data_dir,
        subset="val",
        grid_size=args.grid_size,
        max_houses=args.max_val_houses,
        min_blocks_per_house=args.min_blocks_per_house,
        vocab_map=train_ds.block_id_map,
        raw_block_keep=train_ds.raw_block_keep,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = VoxelDenoiser(vocab_size=train_ds.vocab_size, hidden=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_weights = build_loss_weights(
        vocab_size=train_ds.vocab_size,
        air_token=train_ds.air_token,
        unk_token=train_ds.unk_token,
        air_loss_weight=args.air_loss_weight,
        unk_loss_weight=args.unk_loss_weight,
        device=device,
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Loss weights: air={loss_weights[train_ds.air_token].item():.3f}, "
        f"unk={loss_weights[train_ds.unk_token].item():.3f}, "
        "other=1.000"
    )
    print(
        f"Mask scales: air={args.air_mask_scale:.3f}, "
        f"solid={args.solid_mask_scale:.3f}"
    )

    token_names = {0: "air", 1: "mask", 2: "other"}
    for raw_id, tok in train_ds.block_id_map.items():
        token_names[tok] = f"b{raw_id}"

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d} train", leave=False)
        for x0 in train_bar:
            x0 = x0.to(device)
            bsz = x0.shape[0]
            t = torch.randint(1, args.diffusion_steps + 1, (bsz,), device=device)
            xt = corrupt_with_mask(
                x0,
                t,
                args.diffusion_steps,
                train_ds.mask_token,
                train_ds.air_token,
                args.air_mask_scale,
                args.solid_mask_scale,
            )

            logits = model(xt, t)
            loss = F.cross_entropy(logits, x0, weight=loss_weights)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item() * bsz
            train_count += bsz
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(train_count, 1)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch:03d} val", leave=False)
            for x0 in val_bar:
                x0 = x0.to(device)
                bsz = x0.shape[0]
                t = torch.randint(1, args.diffusion_steps + 1, (bsz,), device=device)
                xt = corrupt_with_mask(
                    x0,
                    t,
                    args.diffusion_steps,
                    train_ds.mask_token,
                    train_ds.air_token,
                    args.air_mask_scale,
                    args.solid_mask_scale,
                )
                logits = model(xt, t)
                loss = F.cross_entropy(logits, x0, weight=loss_weights)
                val_loss += loss.item() * bsz
                val_count += bsz
                val_bar.set_postfix(loss=f"{loss.item():.4f}")
        val_loss /= max(val_count, 1)

        print(f"epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        ckpt = {
            "model": model.state_dict(),
            "block_id_map": train_ds.block_id_map,
            "raw_block_keep": train_ds.raw_block_keep,
            "grid_size": args.grid_size,
            "vocab_size": train_ds.vocab_size,
            "args": vars(args),
        }
        torch.save(ckpt, run_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, run_dir / "best.pt")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            samples = sample_structures(
                model=model,
                num_samples=args.num_preview_samples,
                grid_size=args.grid_size,
                vocab_size=train_ds.vocab_size,
                mask_token=train_ds.mask_token,
                air_token=train_ds.air_token,
                num_steps=args.diffusion_steps,
                device=device,
                temperature=args.temperature,
            )
            save_sample_outputs(
                run_dir / f"samples_epoch_{epoch:03d}",
                samples,
                token_names,
                block_id_map=train_ds.block_id_map,
                nbt_palette_json=args.nbt_palette_json,
                nbt_data_version=args.nbt_data_version,
                nbt_include_air=args.nbt_include_air,
            )

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Finished. Best val loss: {best_val:.4f}")
    print(f"Checkpoint saved to: {run_dir / 'best.pt'}")


@torch.no_grad()
def generate(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)

    vocab_size = ckpt["vocab_size"]
    grid_size = ckpt["grid_size"]
    block_id_map = ckpt["block_id_map"]

    model = VoxelDenoiser(vocab_size=vocab_size, hidden=args.hidden_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    token_names = {0: "air", 1: "mask", 2: "other"}
    for raw_id, tok in block_id_map.items():
        token_names[tok] = f"b{raw_id}"

    samples = sample_structures(
        model=model,
        num_samples=args.num_generate,
        grid_size=grid_size,
        vocab_size=vocab_size,
        mask_token=1,
        air_token=0,
        num_steps=args.diffusion_steps,
        device=device,
        temperature=args.temperature,
    )
    out_dir = Path(args.out_dir)
    save_sample_outputs(
        out_dir,
        samples,
        token_names,
        block_id_map=block_id_map,
        nbt_palette_json=args.nbt_palette_json,
        nbt_data_version=args.nbt_data_version,
        nbt_include_air=args.nbt_include_air,
    )
    print(f"Saved {len(samples)} samples to {out_dir}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a discrete diffusion-style voxel generator on 3D-Craft.")
    sub = p.add_subparsers(dest="mode", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--data-dir", type=str, default="./data")
    train_p.add_argument("--run-dir", type=str, default="./runs/minecraft_diffusion")
    train_p.add_argument("--grid-size", type=int, default=16)
    train_p.add_argument("--top-k-blocks", type=int, default=24)
    train_p.add_argument("--min-blocks-per-house", type=int, default=40)
    train_p.add_argument("--max-train-houses", type=int, default=0)
    train_p.add_argument("--max-val-houses", type=int, default=0)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--epochs", type=int, default=20)
    train_p.add_argument("--lr", type=float, default=2e-4)
    train_p.add_argument("--hidden-dim", type=int, default=96)
    train_p.add_argument("--diffusion-steps", type=int, default=16)
    train_p.add_argument("--sample-every", type=int, default=5)
    train_p.add_argument("--num-preview-samples", type=int, default=4)
    train_p.add_argument("--temperature", type=float, default=1.0)
    train_p.add_argument("--num-workers", type=int, default=0)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--cpu", action="store_true")
    train_p.add_argument("--air-loss-weight", type=float, default=0.1)
    train_p.add_argument("--unk-loss-weight", type=float, default=0.5)
    train_p.add_argument("--air-mask-scale", type=float, default=0.35)
    train_p.add_argument("--solid-mask-scale", type=float, default=1.25)
    train_p.add_argument("--nbt-palette-json", type=str, default=DEFAULT_NBT_PALETTE_JSON)
    train_p.add_argument("--nbt-data-version", type=int, default=DEFAULT_NBT_DATA_VERSION)
    train_p.add_argument("--nbt-include-air", action="store_true")

    gen_p = sub.add_parser("generate")
    gen_p.add_argument("--checkpoint", type=str, required=True)
    gen_p.add_argument("--out-dir", type=str, default="./generated")
    gen_p.add_argument("--num-generate", type=int, default=8)
    gen_p.add_argument("--hidden-dim", type=int, default=96)
    gen_p.add_argument("--diffusion-steps", type=int, default=16)
    gen_p.add_argument("--temperature", type=float, default=1.0)
    gen_p.add_argument("--cpu", action="store_true")
    gen_p.add_argument("--nbt-palette-json", type=str, default=DEFAULT_NBT_PALETTE_JSON)
    gen_p.add_argument("--nbt-data-version", type=int, default=DEFAULT_NBT_DATA_VERSION)
    gen_p.add_argument("--nbt-include-air", action="store_true")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "generate":
        generate(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
