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
    path = Path(palette_path)
    with path.open("r", encoding="utf-8") as handle:
        palette = json.load(handle)
    if not isinstance(palette, list) or not palette:
        raise ValueError(f"Palette at {path} must be a non-empty JSON list.")
    return palette


def discover_npz_files(data_dir: str | Path, limit: int | None = None) -> list[Path]:
    files = sorted(Path(data_dir).glob("*.npz"))
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
    ) -> None:
        self.data_dir = Path(data_dir)
        self.paths = discover_npz_files(self.data_dir, limit=limit)
        self.palette = load_palette(palette_path)
        self.vocab_size = len(self.palette)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            blocks = np.asarray(data["blocks"], dtype=np.int64)
            shape = tuple(int(v) for v in data["shape"].tolist())
            original_shape = tuple(int(v) for v in data["original_shape"].tolist())
            axis_order = str(data["axis_order"].item())
            source_format = str(data["source_format"].item())
            dataset_path = str(data["dataset_path"].item())

        if blocks.shape != (32, 32, 32):
            raise ValueError(f"{path} has shape {blocks.shape}, expected (32, 32, 32).")
        if blocks.min() < 0 or blocks.max() >= self.vocab_size:
            raise ValueError(
                f"{path} contains block IDs outside palette range 0..{self.vocab_size - 1}."
            )

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
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    val_size = max(1, int(round(len(dataset) * val_ratio)))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split is too large for this dataset.")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def summarize_dataset(
    records: Iterable[dict[str, torch.Tensor | str]],
    palette: list[str],
) -> dict[str, object]:
    total_samples = 0
    total_non_air = 0
    for sample in records:
        total_samples += 1
        blocks = sample["blocks"]
        assert isinstance(blocks, torch.Tensor)
        total_non_air += int((blocks != 0).sum().item())
    return {
        "samples": total_samples,
        "palette_size": len(palette),
        "avg_non_air_blocks": total_non_air / max(1, total_samples),
    }
