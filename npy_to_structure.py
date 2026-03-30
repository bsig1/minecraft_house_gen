import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from ascii_to_structure import CompoundTag, IntTag, ListTag, block_state_to_nbt, normalize_block_state, save_structure


BlockState = Tuple[str, Tuple[Tuple[str, str], ...]]


def load_checkpoint_inverse_map(checkpoint_path: Path) -> Dict[int, int]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return {int(token): int(raw_id) for raw_id, token in ckpt["block_id_map"].items()}


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
                            "pos": ListTag(3, [IntTag(x), IntTag(y), IntTag(z)]),
                        }
                    )
                )

    return CompoundTag(
        {
            "DataVersion": IntTag(data_version),
            "size": ListTag(3, [IntTag(sx), IntTag(sy), IntTag(sz)]),
            "palette": ListTag(10, [block_state_to_nbt(state) for state in palette_states]),
            "blocks": ListTag(10, blocks),
            "entities": ListTag(10, []),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a generated sample .npy into an approximate Java Edition structure .nbt file."
    )
    parser.add_argument("sample_path", type=Path, help="Path to sample_XXX.npy.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Training checkpoint with token mappings.")
    parser.add_argument("--palette-json", type=Path, required=True, help="Legacy ID to modern block-state mapping.")
    parser.add_argument("--output", type=Path, required=True, help="Output .nbt path.")
    parser.add_argument("--include-air", action="store_true", help="Include air blocks in the structure.")
    parser.add_argument("--data-version", type=int, default=4189, help="Minecraft Java DataVersion.")
    args = parser.parse_args()

    grid = np.load(args.sample_path)
    token_to_raw_id = load_checkpoint_inverse_map(args.checkpoint)
    raw_id_to_state, fallback_unknown = load_legacy_palette(args.palette_json)
    structure = build_structure_from_npy(
        grid=grid,
        token_to_raw_id=token_to_raw_id,
        raw_id_to_state=raw_id_to_state,
        fallback_unknown=fallback_unknown,
        include_air=args.include_air,
        data_version=args.data_version,
    )
    save_structure(structure, args.output)
    print(f"Saved approximate structure to {args.output}")


if __name__ == "__main__":
    main()
