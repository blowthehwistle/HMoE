#!/usr/bin/env python3
"""Build train/validation symlink splits from completed RedPajama shards.

This intentionally ignores ``*.tmp`` files from in-progress downloads. Run this
before starting training, and rerun it later if more downloaded shards should be
included in the next training job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


DATA_PATTERNS = ("*.jsonl", "*.jsonl.zst")


def complete_files(root: Path) -> list[Path]:
    files: list[Path] = []
    data_root = root / "v1.0.0"
    for pattern in DATA_PATTERNS:
        files.extend(data_root.rglob(pattern))
    return sorted(path for path in files if ".tmp" not in path.name and path.is_file())


def relative_link_name(root: Path, path: Path) -> Path:
    return Path(*path.relative_to(root / "v1.0.0").parts)


def relative_data_name(root: Path, path: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = (root / "v1.0.0").resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return path.relative_to(root / "v1.0.0").as_posix()


def recreate_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dst)


def write_manifest(paths: list[Path], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        for path in sorted(paths):
            f.write(str(path.resolve()) + "\n")


def write_relative_manifest(root: Path, paths: list[Path], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        for path in sorted(paths, key=lambda p: relative_data_name(root, p)):
            f.write(relative_data_name(root, path) + "\n")


def read_excluded_relative_paths(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        with path.open() as f:
            for line in f:
                item = line.strip()
                if item and not item.startswith("#"):
                    excluded.add(item)
    return excluded


def manifest_sha256(manifest_path: Path) -> str:
    h = hashlib.sha256()
    with manifest_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("redpajama_1T"))
    parser.add_argument("--validation-count", type=int, default=1)
    parser.add_argument("--prefer-validation-subset", default="c4")
    parser.add_argument(
        "--exclude-relative-manifest",
        action="append",
        default=[],
        type=Path,
        help="Manifest of v1.0.0-relative shard paths to exclude from this split.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Fail instead of overwriting an existing split manifest.",
    )
    args = parser.parse_args()

    files = complete_files(args.root)
    if not files:
        raise SystemExit(f"No completed RedPajama shards found under {args.root / 'v1.0.0'}")

    excluded_relative = read_excluded_relative_paths(args.exclude_relative_manifest)
    if excluded_relative:
        before = len(files)
        files = [p for p in files if relative_data_name(args.root, p) not in excluded_relative]
        print(f"excluded shards: {before - len(files)}")
        if not files:
            raise SystemExit("All completed shards were excluded; no data left for split.")

    preferred = [p for p in files if f"/{args.prefer_validation_subset}/" in p.as_posix()]
    fallback = [p for p in files if p not in preferred]
    validation = (preferred + fallback)[: args.validation_count]
    train = [p for p in files if p not in set(validation)]

    split_root = args.root / "splits"
    train_dir = split_root / "train"
    validation_dir = split_root / "validation"
    train_manifest = split_root / "train_manifest.txt"
    validation_manifest = split_root / "validation_manifest.txt"
    train_relative_manifest = split_root / "train_shards_relative.txt"
    validation_relative_manifest = split_root / "validation_shards_relative.txt"
    used_relative_manifest = split_root / "used_shards_relative.txt"
    metadata_path = split_root / "metadata.json"

    if args.freeze and (train_manifest.exists() or validation_manifest.exists()):
        raise SystemExit(
            f"Split manifests already exist under {split_root}. "
            "Refusing to overwrite because --freeze was set."
        )

    recreate_dir(train_dir)
    recreate_dir(validation_dir)

    train_links: list[Path] = []
    validation_links: list[Path] = []
    for src in train:
        dst = train_dir / relative_link_name(args.root, src)
        link_file(src, dst)
        train_links.append(dst)
    for src in validation:
        dst = validation_dir / relative_link_name(args.root, src)
        link_file(src, dst)
        validation_links.append(dst)

    write_manifest(train_links, train_manifest)
    write_manifest(validation_links, validation_manifest)
    write_relative_manifest(args.root, train, train_relative_manifest)
    write_relative_manifest(args.root, validation, validation_relative_manifest)
    write_relative_manifest(args.root, train + validation, used_relative_manifest)
    metadata = {
        "created_at_unix": time.time(),
        "root": str(args.root.resolve()),
        "excluded_shards": len(excluded_relative),
        "completed_shards": len(files),
        "train_shards": len(train_links),
        "validation_shards": len(validation_links),
        "validation_sources": [str(path.resolve()) for path in validation],
        "train_manifest": str(train_manifest.resolve()),
        "validation_manifest": str(validation_manifest.resolve()),
        "train_relative_manifest": str(train_relative_manifest.resolve()),
        "validation_relative_manifest": str(validation_relative_manifest.resolve()),
        "used_relative_manifest": str(used_relative_manifest.resolve()),
        "train_manifest_sha256": manifest_sha256(train_manifest),
        "validation_manifest_sha256": manifest_sha256(validation_manifest),
        "used_relative_manifest_sha256": manifest_sha256(used_relative_manifest),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"root: {args.root}")
    print(f"completed shards: {len(files)}")
    print(f"train shards: {len(train)}")
    print(f"validation shards: {len(validation)}")
    print(f"train manifest: {train_manifest}")
    print(f"validation manifest: {validation_manifest}")
    print(f"used relative manifest: {used_relative_manifest}")
    print(f"metadata: {metadata_path}")
    for src in validation:
        print(f"validation: {src}")


if __name__ == "__main__":
    main()
