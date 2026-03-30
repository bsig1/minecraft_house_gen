import argparse
import gzip
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


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


def parse_ascii_layers(path: Path) -> Tuple[List[List[str]], int, int, int]:
    lines = path.read_text().splitlines()
    layers: Dict[int, List[str]] = {}

    current_y: Optional[int] = None
    current_rows: List[str] = []

    def commit_current() -> None:
        nonlocal current_y, current_rows
        if current_y is None:
            return
        if not current_rows:
            raise ValueError(f"Layer y={current_y} in {path} has no rows.")
        layers[current_y] = current_rows[:]
        current_y = None
        current_rows = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line.strip():
            commit_current()
            continue

        if line.startswith("Layer y="):
            commit_current()
            current_y = int(line.split("=", 1)[1])
            continue

        if current_y is None:
            raise ValueError(f"Encountered voxel row before any layer header in {path}.")
        current_rows.append(line)

    commit_current()

    if not layers:
        raise ValueError(f"No layers were found in {path}.")

    sorted_layers = [layers[y] for y in sorted(layers)]
    height = len(sorted_layers)
    length = len(sorted_layers[0])
    width = len(sorted_layers[0][0])

    for layer_index, rows in enumerate(sorted_layers):
        if len(rows) != length:
            raise ValueError(f"Layer index {layer_index} has {len(rows)} rows, expected {length}.")
        for row_index, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(
                    f"Layer index {layer_index}, row {row_index} has width {len(row)}, expected {width}."
                )

    return sorted_layers, width, height, length


def load_palette(path: Path) -> Dict[str, BlockState]:
    raw_palette = json.loads(path.read_text())
    palette: Dict[str, BlockState] = {}

    for key, value in raw_palette.items():
        if len(key) != 1:
            raise ValueError(f"Palette key {key!r} must be exactly one character.")

        if isinstance(value, str):
            name = value
            properties: Dict[str, str] = {}
        elif isinstance(value, dict):
            if "Name" not in value:
                raise ValueError(f"Palette entry for {key!r} is missing 'Name'.")
            name = str(value["Name"])
            properties = {str(k): str(v) for k, v in value.get("Properties", {}).items()}
        else:
            raise ValueError(f"Palette entry for {key!r} must be a string or object.")

        palette[key] = normalize_block_state(name, properties)

    return palette


def normalize_block_state(name: str, properties: Optional[Dict[str, str]] = None) -> BlockState:
    props = tuple(sorted((properties or {}).items()))
    return name, props


def block_state_to_nbt(state: BlockState) -> CompoundTag:
    name, properties = state
    value: Dict[str, object] = {"Name": StringTag(name)}
    if properties:
        value["Properties"] = CompoundTag({key: StringTag(val) for key, val in properties})
    return CompoundTag(value)


def build_structure(
    layers: List[List[str]],
    palette_map: Dict[str, BlockState],
    include_air: bool,
    data_version: int,
) -> CompoundTag:
    palette_index: Dict[BlockState, int] = {}
    palette_states: List[BlockState] = []
    blocks: List[CompoundTag] = []

    for y, rows in enumerate(layers):
        for z, row in enumerate(rows):
            for x, char in enumerate(row):
                if char not in palette_map:
                    raise ValueError(
                        f"Character {char!r} at x={x}, y={y}, z={z} is missing from the palette mapping."
                    )

                state = palette_map[char]
                if not include_air and state[0] == "minecraft:air":
                    continue

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

    width = len(layers[0][0])
    height = len(layers)
    length = len(layers[0])

    return CompoundTag(
        {
            "DataVersion": IntTag(data_version),
            "size": ListTag(TAG_INT, [IntTag(width), IntTag(height), IntTag(length)]),
            "palette": ListTag(TAG_COMPOUND, [block_state_to_nbt(state) for state in palette_states]),
            "blocks": ListTag(TAG_COMPOUND, blocks),
            "entities": ListTag(TAG_COMPOUND, []),
        }
    )


def write_named_tag(stream, tag_id: int, name: str, payload: object) -> None:
    stream.write(struct.pack(">B", tag_id))
    write_string(stream, name)
    write_payload(stream, tag_id, payload)


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


def save_structure(root: CompoundTag, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wb") as f:
        write_named_tag(f, TAG_COMPOUND, "", root)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a generated ASCII voxel sample into a Java Edition structure .nbt file."
    )
    parser.add_argument("ascii_path", type=Path, help="Path to a sample_XXX.txt file.")
    parser.add_argument(
        "--palette-json",
        type=Path,
        required=True,
        help="JSON file mapping each ASCII character to a Minecraft block state.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./generated/structure_from_ascii.nbt"),
        help="Where to save the compressed structure .nbt file.",
    )
    parser.add_argument(
        "--include-air",
        action="store_true",
        help="Include minecraft:air blocks in the structure so loading can clear space.",
    )
    parser.add_argument(
        "--data-version",
        type=int,
        default=4189,
        help="Minecraft Java DataVersion for the structure file. Default: 4189 (1.21.4).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    layers, width, height, length = parse_ascii_layers(args.ascii_path)
    palette_map = load_palette(args.palette_json)
    structure = build_structure(
        layers=layers,
        palette_map=palette_map,
        include_air=args.include_air,
        data_version=args.data_version,
    )
    save_structure(structure, args.output)

    print(f"Saved structure to {args.output}")
    print(f"Size: {width} x {height} x {length}")
    print(f"Include air: {args.include_air}")


if __name__ == "__main__":
    main()
