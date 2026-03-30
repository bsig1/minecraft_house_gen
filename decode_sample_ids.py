import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


SPECIAL_TOKENS = {
    0: {"label": "air", "raw_id": None},
    1: {"label": "mask", "raw_id": None},
    2: {"label": "unk", "raw_id": None},
}


def build_inverse_map(checkpoint_path: Path) -> dict[int, int]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return {int(token): int(raw_id) for raw_id, token in ckpt["block_id_map"].items()}


def maybe_unsigned(raw_id: int | None, use_unsigned: bool) -> int | None:
    if raw_id is None or not use_unsigned:
        return raw_id
    return raw_id % 256


def decode_sample(sample_path: Path, inverse_map: dict[int, int], use_unsigned: bool) -> dict:
    arr = np.load(sample_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D voxel grid in {sample_path}, got shape {arr.shape}.")

    counts = Counter(int(token) for token in arr.flatten().tolist())
    palette = []
    blocks = []

    seen_tokens = sorted(counts)
    for token in seen_tokens:
        if token in SPECIAL_TOKENS:
            special = SPECIAL_TOKENS[token]
            palette.append(
                {
                    "token": token,
                    "label": special["label"],
                    "raw_id": special["raw_id"],
                }
            )
            continue

        raw_id = inverse_map.get(token)
        palette.append(
            {
                "token": token,
                "label": f"raw_{raw_id}" if raw_id is not None else "unmapped",
                "raw_id": maybe_unsigned(raw_id, use_unsigned),
                "signed_raw_id": raw_id,
            }
        )

    sx, sy, sz = arr.shape
    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                token = int(arr[x, y, z])
                if token == 0:
                    continue

                if token in SPECIAL_TOKENS:
                    raw_id = SPECIAL_TOKENS[token]["raw_id"]
                    label = SPECIAL_TOKENS[token]["label"]
                else:
                    signed_raw_id = inverse_map.get(token)
                    raw_id = maybe_unsigned(signed_raw_id, use_unsigned)
                    label = f"raw_{signed_raw_id}" if signed_raw_id is not None else "unmapped"

                blocks.append(
                    {
                        "pos": [x, y, z],
                        "token": token,
                        "raw_id": raw_id,
                        "label": label,
                    }
                )

    return {
        "shape": list(arr.shape),
        "use_unsigned_ids": use_unsigned,
        "palette": palette,
        "counts": {str(token): counts[token] for token in seen_tokens},
        "blocks": blocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode a generated sample .npy into token IDs and recovered raw block IDs."
    )
    parser.add_argument("sample_path", type=Path, help="Path to a sample_XXX.npy file.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint used to decode token IDs back into raw block IDs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the decoded JSON manifest.",
    )
    parser.add_argument(
        "--signed-ids",
        action="store_true",
        help="Keep raw IDs in their signed form instead of converting negatives to 0-255.",
    )
    args = parser.parse_args()

    inverse_map = build_inverse_map(args.checkpoint)
    decoded = decode_sample(args.sample_path, inverse_map, use_unsigned=not args.signed_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decoded, indent=2))
    print(f"Saved decoded block-id manifest to {args.output}")


if __name__ == "__main__":
    main()
