"""Utilities for converting generated voxel token grids to Minecraft structure NBT.
"""

import gzip
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ------------------------------------------------------------
# Local NBT + palette helpers
# ------------------------------------------------------------

TAG_END = 0
TAG_INT = 3
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10


@dataclass(frozen=True)
class IntTag:
    value: int


@dataclass(frozen=True)
class StringTag:
    value: str


@dataclass(frozen=True)
class ListTag:
    tag_id: int
    items: List[object]


@dataclass(frozen=True)
class CompoundTag:
    value: Dict[str, object]


BlockState = Tuple[str, Tuple[Tuple[str, str], ...]]


def normalize_block_state(name: str, properties: Optional[Dict[str, str]] = None) -> BlockState:
    props = tuple(sorted((properties or {}).items()))
    return name, props


def block_state_to_nbt(state: BlockState) -> CompoundTag:
    name, properties = state
    value: Dict[str, object] = {"Name": StringTag(name)}
    if properties:
        value["Properties"] = CompoundTag({key: StringTag(val) for key, val in properties})
    return CompoundTag(value)


def block_state_from_json(payload: object) -> BlockState:
    if isinstance(payload, str):
        return normalize_block_state(payload)
    if isinstance(payload, dict):
        if "Name" not in payload:
            raise ValueError("Block-state mapping objects must include 'Name'.")
        name = str(payload["Name"])
        properties = {str(k): str(v) for k, v in payload.get("Properties", {}).items()}
        return normalize_block_state(name, properties)
    raise ValueError(f"Unsupported block-state payload: {payload!r}")


def load_legacy_palette(path: Path) -> Tuple[Dict[int, BlockState], BlockState]:
    payload = json.loads(path.read_text())
    signed_ids = payload.get("signed_ids", {})
    fallback_unknown = payload.get("fallback_unknown")
    if fallback_unknown is None:
        raise ValueError("Palette JSON must include a fallback_unknown block mapping.")

    mapping: Dict[int, BlockState] = {}
    for raw_id_text, state_payload in signed_ids.items():
        mapping[int(raw_id_text)] = block_state_from_json(state_payload)

    return mapping, block_state_from_json(fallback_unknown)


def infer_tag_id(value: object) -> int:
    if isinstance(value, IntTag):
        return TAG_INT
    if isinstance(value, StringTag):
        return TAG_STRING
    if isinstance(value, ListTag):
        return TAG_LIST
    if isinstance(value, CompoundTag):
        return TAG_COMPOUND
    raise TypeError(f"Unsupported NBT value: {type(value)!r}")


def write_string(stream, value: str) -> None:
    encoded = value.encode("utf-8")
    stream.write(struct.pack(">H", len(encoded)))
    stream.write(encoded)


def write_payload(stream, tag_id: int, payload: object) -> None:
    if tag_id == TAG_INT:
        stream.write(struct.pack(">i", payload.value))
        return

    if tag_id == TAG_STRING:
        write_string(stream, payload.value)
        return

    if tag_id == TAG_LIST:
        stream.write(struct.pack(">B", payload.tag_id))
        stream.write(struct.pack(">i", len(payload.items)))
        for item in payload.items:
            write_payload(stream, payload.tag_id, item)
        return

    if tag_id == TAG_COMPOUND:
        for key, value in payload.value.items():
            child_tag_id = infer_tag_id(value)
            write_named_tag(stream, child_tag_id, key, value)
        stream.write(struct.pack(">B", TAG_END))
        return

    raise ValueError(f"Unsupported tag id: {tag_id}")


def write_named_tag(stream, tag_id: int, name: str, payload: object) -> None:
    stream.write(struct.pack(">B", tag_id))
    write_string(stream, name)
    write_payload(stream, tag_id, payload)


def save_structure(root: CompoundTag, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wb") as f:
        write_named_tag(f, TAG_COMPOUND, "", root)


def build_structure_from_npy(
    grid: np.ndarray,
    token_to_raw_id: Dict[int, int],
    raw_id_to_state: Dict[int, BlockState],
    fallback_unknown: BlockState,
    include_air: bool,
    data_version: int,
) -> CompoundTag:
    palette_index: Dict[BlockState, int] = {}
    palette_states: list[BlockState] = []
    blocks: list[CompoundTag] = []

    sx, sy, sz = grid.shape
    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                token = int(grid[x, y, z])
                if token == 0 and not include_air:
                    continue

                if token == 0:
                    state = normalize_block_state("minecraft:air")
                elif token == 2:
                    state = fallback_unknown
                else:
                    raw_id = token_to_raw_id.get(token)
                    if raw_id is None:
                        state = fallback_unknown
                    else:
                        state = raw_id_to_state.get(raw_id, fallback_unknown)

                state_id = palette_index.get(state)
                if state_id is None:
                    state_id = len(palette_states)
                    palette_index[state] = state_id
                    palette_states.append(state)

                blocks.append(
                    CompoundTag(
                        {
                            "state": IntTag(state_id),
                            "pos": ListTag(TAG_INT, [IntTag(x), IntTag(y), IntTag(z)]),
                        }
                    )
                )

    return CompoundTag(
        {
            "DataVersion": IntTag(data_version),
            "size": ListTag(TAG_INT, [IntTag(sx), IntTag(sy), IntTag(sz)]),
            "palette": ListTag(TAG_COMPOUND, [block_state_to_nbt(state) for state in palette_states]),
            "blocks": ListTag(TAG_COMPOUND, blocks),
            "entities": ListTag(TAG_COMPOUND, []),
        }
    )



