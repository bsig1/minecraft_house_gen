from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


@dataclass(frozen=True)
class BuildRecord:
    path: Path
    original_shape: tuple[int, int, int]
    shape: tuple[int, int, int]
    axis_order: str
    source_format: str
    dataset_path: str


def load_palette(palette_path: str | Path) -> list[str]:
    """Load a non-empty block palette from a JSON file."""
    path: Path = Path(palette_path)
    with path.open("r", encoding="utf-8") as handle:
        palette: list[str] = json.load(handle)
    if not isinstance(palette, list) or not palette:
        raise ValueError(f"Palette at {path} must be a non-empty JSON list.")
    return palette


def load_block_mapping(mapping_path: str | Path, original_palette: list[str]) -> tuple[list[str], np.ndarray]:
    """Load a block reduction map and create a new palette and numpy lookup array."""
    with Path(mapping_path).open("r", encoding="utf-8") as handle:
        mapping: dict[str, str] = json.load(handle)

    # Create new palette with mapped names
    new_palette_set = set()
    for block in original_palette:
        new_palette_set.add(mapping.get(block, block))

    new_palette = sorted(list(new_palette_set))

    # Ensure air remains at index 0
    air_block = original_palette[0] if original_palette else "minecraft:air"
    if air_block in new_palette:
        new_palette.remove(air_block)
        new_palette.insert(0, air_block)

    new_palette_index = {name: i for i, name in enumerate(new_palette)}

    # Build index-to-index lookup array
    lookup = np.zeros(len(original_palette), dtype=np.int64)
    for old_idx, old_name in enumerate(original_palette):
        lookup[old_idx] = new_palette_index[mapping.get(old_name, old_name)]

    return new_palette, lookup


def discover_npz_files(data_dir: str | Path, limit: int | None = None) -> list[Path]:
    """Return sorted `.npz` files from `data_dir`, optionally capped by `limit`."""
    files: list[Path] = sorted(Path(data_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}.")
    if limit is not None:
        files = files[:limit]
    return files


class MinecraftBuildDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        data_dir: str | Path,
        palette_path: str | Path,
        limit: int | None = None,
        block_mapping_path: str | Path | None = None,
    ) -> None:
        """Initialize the dataset index and palette metadata."""
        self.data_dir: Path = Path(data_dir)
        self.paths: list[Path] = discover_npz_files(self.data_dir, limit=limit)
        self.palette: list[str] = load_palette(palette_path)
        self.original_vocab_size: int = len(self.palette)
        self.lookup_array: np.ndarray | None = None

        if block_mapping_path is not None:
            self.palette, self.lookup_array = load_block_mapping(block_mapping_path, self.palette)

        self.vocab_size = len(self.palette)

    def __len__(self) -> int:
        """Return the number of indexed build files."""
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        """Load one build and return tensors plus source metadata."""
        path: Path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            blocks: np.ndarray = np.asarray(data["blocks"], dtype=np.int64)
            shape: tuple[int, int, int] = tuple(int(v) for v in data["shape"].tolist())
            original_shape: tuple[int, int, int] = tuple(
                int(v) for v in data["original_shape"].tolist()
            )
            axis_order: str = str(data["axis_order"].item())
            source_format: str = str(data["source_format"].item())
            dataset_path: str = str(data["dataset_path"].item())

        if blocks.shape != (32, 32, 32):
            raise ValueError(f"{path} has shape {blocks.shape}, expected (32, 32, 32).")
        if blocks.min() < 0 or blocks.max() >= self.original_vocab_size:
            raise ValueError(
                f"{path} contains block IDs outside original palette range 0..{self.original_vocab_size - 1}."
            )

        if self.lookup_array is not None:
            blocks = self.lookup_array[blocks]

        return {
            "blocks": torch.from_numpy(blocks).long(),
            "shape": torch.tensor(shape, dtype=torch.long),
            "original_shape": torch.tensor(original_shape, dtype=torch.long),
            "axis_order": axis_order,
            "source_format": source_format,
            "dataset_path": dataset_path,
            "path": str(path),
        }


def split_dataset(
    dataset: Dataset,
    val_ratio: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Split a dataset into train/validation subsets with a fixed RNG seed."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    val_size: int = max(1, int(round(len(dataset) * val_ratio)))
    train_size: int = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split is too large for this dataset.")
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def summarize_dataset(
    records: Iterable[dict[str, torch.Tensor | str]],
    palette: list[str],
) -> dict[str, object]:
    """Compute basic dataset summary metrics from loaded samples."""
    total_samples: int = 0
    total_non_air: int = 0
    for sample in records:
        total_samples += 1
        blocks: torch.Tensor | str = sample["blocks"]
        assert isinstance(blocks, torch.Tensor)
        total_non_air += int((blocks != 0).sum().item())
    return {
        "samples": total_samples,
        "palette_size": len(palette),
        "avg_non_air_blocks": total_non_air / max(1, total_samples),
    }
