#!/usr/bin/env python3
"""Build a small fixed RedPajama split without touching the full split.

The output manifests point at completed shards under ``redpajama_1T/v1.0.0``.
This is intended for short routing/imbalance pilot runs, not final accuracy
reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


DATA_PATTERNS = ("*.jsonl", "*.jsonl.zst")
DEFAULT_TRAIN_SUBSET_ORDER = ("c4", "github", "common_crawl")


def complete_files(root: Path) -> list[Path]:
    data_root = root / "v1.0.0"
    files: list[Path] = []
    for pattern in DATA_PATTERNS:
        files.extend(data_root.rglob(pattern))
    return sorted(path for path in files if ".tmp" not in path.name and path.is_file())


def subset_name(root: Path, path: Path) -> str:
    return path.relative_to(root / "v1.0.0").parts[0]


def relative_data_name(root: Path, path: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = (root / "v1.0.0").resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return path.relative_to(root / "v1.0.0").as_posix()


def manifest_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_abs_manifest(paths: list[Path], path: Path) -> None:
    with path.open("w") as f:
        for item in paths:
            f.write(str(item.resolve()) + "\n")


def write_relative_manifest(root: Path, paths: list[Path], path: Path) -> None:
    with path.open("w") as f:
        for item in paths:
            f.write(relative_data_name(root, item) + "\n")


def select_train(files: list[Path], root: Path, max_bytes: int, order: tuple[str, ...]) -> list[Path]:
    by_subset = {name: [] for name in order}
    extra: list[Path] = []
    for path in files:
        by_subset.get(subset_name(root, path), extra).append(path)

    ordered_files: list[Path] = []
    for name in order:
        ordered_files.extend(sorted(by_subset[name], key=lambda p: relative_data_name(root, p)))
    ordered_files.extend(sorted(extra, key=lambda p: relative_data_name(root, p)))

    selected: list[Path] = []
    total = 0
    for path in ordered_files:
        size = path.stat().st_size
        if selected and total + size > max_bytes:
            continue
        selected.append(path)
        total += size
        if total >= max_bytes:
            break
    return selected


def select_validation(files: list[Path], train: list[Path], root: Path, subset: str, count: int) -> list[Path]:
    train_set = set(train)
    candidates = [
        path
        for path in files
        if path not in train_set and subset_name(root, path) == subset
    ]
    if len(candidates) < count:
        candidates = [path for path in files if path not in train_set]
    return sorted(candidates, key=lambda p: (path_size_key(p), relative_data_name(root, p)))[:count]


def path_size_key(path: Path) -> int:
    return path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("redpajama_1T"))
    parser.add_argument("--name", default="mini_10g")
    parser.add_argument("--train-max-gb", type=float, default=10.0)
    parser.add_argument("--validation-subset", default="common_crawl")
    parser.add_argument("--validation-count", type=int, default=1)
    parser.add_argument(
        "--train-subset-order",
        default=",".join(DEFAULT_TRAIN_SUBSET_ORDER),
        help="Comma-separated subset priority for train shard selection.",
    )
    args = parser.parse_args()

    files = complete_files(args.root)
    if not files:
        raise SystemExit(f"No completed RedPajama shards found under {args.root / 'v1.0.0'}")

    max_bytes = int(args.train_max_gb * 1024**3)
    order = tuple(item.strip() for item in args.train_subset_order.split(",") if item.strip())
    train = select_train(files, args.root, max_bytes, order)
    validation = select_validation(files, train, args.root, args.validation_subset, args.validation_count)

    split_root = args.root / "mini_splits" / args.name
    split_root.mkdir(parents=True, exist_ok=True)
    train_manifest = split_root / "train_manifest.txt"
    validation_manifest = split_root / "validation_manifest.txt"
    train_relative_manifest = split_root / "train_shards_relative.txt"
    validation_relative_manifest = split_root / "validation_shards_relative.txt"
    used_relative_manifest = split_root / "used_shards_relative.txt"
    metadata_path = split_root / "metadata.json"

    write_abs_manifest(train, train_manifest)
    write_abs_manifest(validation, validation_manifest)
    write_relative_manifest(args.root, train, train_relative_manifest)
    write_relative_manifest(args.root, validation, validation_relative_manifest)
    write_relative_manifest(args.root, train + validation, used_relative_manifest)

    train_bytes = sum(path.stat().st_size for path in train)
    validation_bytes = sum(path.stat().st_size for path in validation)
    metadata = {
        "created_at_unix": time.time(),
        "root": str(args.root.resolve()),
        "name": args.name,
        "train_max_gb": args.train_max_gb,
        "train_bytes": train_bytes,
        "train_gib": train_bytes / 1024**3,
        "validation_bytes": validation_bytes,
        "validation_gib": validation_bytes / 1024**3,
        "train_shards": len(train),
        "validation_shards": len(validation),
        "train_subsets": sorted({subset_name(args.root, path) for path in train}),
        "validation_subsets": sorted({subset_name(args.root, path) for path in validation}),
        "train_manifest": str(train_manifest.resolve()),
        "validation_manifest": str(validation_manifest.resolve()),
        "train_manifest_sha256": manifest_sha256(train_manifest),
        "validation_manifest_sha256": manifest_sha256(validation_manifest),
        "used_relative_manifest_sha256": manifest_sha256(used_relative_manifest),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"mini split: {split_root}")
    print(f"train shards: {len(train)} ({train_bytes / 1024**3:.2f} GiB)")
    print(f"validation shards: {len(validation)} ({validation_bytes / 1024**3:.2f} GiB)")
    print(f"train manifest: {train_manifest}")
    print(f"validation manifest: {validation_manifest}")
    print(f"metadata: {metadata_path}")
    print("train:")
    for path in train:
        print(f"  {relative_data_name(args.root, path)}")
    print("validation:")
    for path in validation:
        print(f"  {relative_data_name(args.root, path)}")


if __name__ == "__main__":
    main()
