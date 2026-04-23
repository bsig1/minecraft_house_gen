from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    """Create a directory path (including parents) and return it."""
    output: Path = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON object to disk with stable formatting."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_checkpoint(
    path: str | Path,
    *,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    config: dict[str, Any],
    epoch: int,
    best_val_loss: float,
) -> None:
    """Persist a model training checkpoint to a `.pt` file."""
    torch.save(
        {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "config": config,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        },
        Path(path),
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and return a serialized model checkpoint."""
    return torch.load(Path(path), map_location=map_location)


def write_build_npz(
    output_path: str | Path,
    blocks: np.ndarray,
    *,
    original_shape: tuple[int, int, int] | None = None,
    axis_order: str = "YZX",
    source_format: str = "ai",
    dataset_path: str = "generated",
) -> None:
    """Write a generated or reconstructed build in project `.npz` format."""
    blocks = np.asarray(blocks, dtype=np.int32)
    if blocks.shape != (32, 32, 32):
        raise ValueError(f"Expected blocks with shape (32, 32, 32), got {blocks.shape}.")
    shape: np.ndarray = np.asarray(blocks.shape, dtype=np.int32)
    original: np.ndarray = np.asarray(original_shape or tuple(blocks.shape), dtype=np.int32)
    np.savez_compressed(
        Path(output_path),
        blocks=blocks,
        shape=shape,
        original_shape=original,
        axis_order=np.asarray(axis_order),
        source_format=np.asarray(source_format),
        dataset_path=np.asarray(dataset_path),
    )


def _parse_palette_entry(block_name: str) -> tuple[str, dict[str, str]]:
    """Parse `minecraft:block[prop=value]` into name and property map."""
    if "[" not in block_name or not block_name.endswith("]"):
        return block_name, {}
    base_name, raw_props = block_name[:-1].split("[", 1)
    properties: dict[str, str] = {}
    for entry in raw_props.split(","):
        if not entry:
            continue
        key, value = entry.split("=", 1)
        properties[key] = value
    return base_name, properties


def write_structure_nbt(
    output_path: str | Path,
    blocks: np.ndarray,
    *,
    palette: list[str],
    original_shape: tuple[int, int, int] | None = None,
    axis_order: str = "YZX",
    author: str = "mcvae",
    data_version: int | None = None,
) -> None:
    """Write blocks to a Minecraft structure NBT file."""
    try:
        from amulet_nbt import CompoundTag, IntTag, ListTag, NamedTag, StringTag
    except ImportError as exc:
        raise ImportError(
            "Structure export requires amulet_nbt. Install it with `pip install amulet-nbt`."
        ) from exc

    blocks = np.asarray(blocks, dtype=np.int32)
    if blocks.shape != (32, 32, 32):
        raise ValueError(f"Expected blocks with shape (32, 32, 32), got {blocks.shape}.")
    if axis_order != "YZX":
        raise ValueError(f"Unsupported axis order {axis_order!r}. Expected 'YZX'.")

    crop_shape: tuple[int, int, int] = original_shape or tuple(int(v) for v in blocks.shape)
    size_y: int
    size_z: int
    size_x: int
    size_y, size_z, size_x = (int(v) for v in crop_shape)
    cropped: np.ndarray = blocks[:size_y, :size_z, :size_x]

    palette_index: dict[int, int] = {}
    palette_tags: list[Any] = []
    block_tags: list[Any] = []

    for y in range(size_y):
        for z in range(size_z):
            for x in range(size_x):
                block_id: int = int(cropped[y, z, x])
                if block_id < 0 or block_id >= len(palette):
                    raise ValueError(
                        f"Block id {block_id} is outside palette range 0..{len(palette) - 1}."
                    )
                if block_id not in palette_index:
                    block_name: str
                    properties: dict[str, str]
                    block_name, properties = _parse_palette_entry(palette[block_id])
                    palette_entry: dict[str, Any] = {"Name": StringTag(block_name)}
                    if properties:
                        palette_entry["Properties"] = CompoundTag(
                            {key: StringTag(value) for key, value in sorted(properties.items())}
                        )
                    palette_index[block_id] = len(palette_tags)
                    palette_tags.append(CompoundTag(palette_entry))
                block_tags.append(
                    CompoundTag(
                        {
                            "pos": ListTag([IntTag(x), IntTag(y), IntTag(z)]),
                            "state": IntTag(palette_index[block_id]),
                        }
                    )
                )

    root_payload: dict[str, Any] = {
        "size": ListTag([IntTag(size_x), IntTag(size_y), IntTag(size_z)]),
        "palette": ListTag(palette_tags),
        "blocks": ListTag(block_tags),
        "entities": ListTag([]),
        "author": StringTag(author),
    }
    if data_version is not None:
        root_payload["DataVersion"] = IntTag(int(data_version))

    NamedTag(CompoundTag(root_payload)).save_to(str(Path(output_path)), compressed=True)
