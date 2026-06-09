#!/usr/bin/env python3
"""Check that frozen RedPajama split manifests still point to existing files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> list[Path]:
    with path.open() as f:
        return [Path(line.strip()) for line in f if line.strip() and not line.startswith("#")]


def check_manifest(path: Path) -> tuple[int, list[Path]]:
    files = read_manifest(path)
    missing = [p for p in files if not p.exists()]
    return len(files), missing


def check_relative_manifest(root: Path, path: Path) -> tuple[int, list[Path]]:
    with path.open() as f:
        rels = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    missing = [root / "v1.0.0" / rel for rel in rels if not (root / "v1.0.0" / rel).exists()]
    return len(rels), missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("redpajama_1T"))
    args = parser.parse_args()

    split_root = args.root / "splits"
    metadata_path = split_root / "metadata.json"
    train_manifest = split_root / "train_manifest.txt"
    validation_manifest = split_root / "validation_manifest.txt"
    used_relative_manifest = split_root / "used_shards_relative.txt"

    if not metadata_path.exists():
        raise SystemExit(f"Missing {metadata_path}. Run build_redpajama_splits.py first.")
    metadata = json.loads(metadata_path.read_text())

    for key, manifest in (
        ("train_manifest_sha256", train_manifest),
        ("validation_manifest_sha256", validation_manifest),
    ):
        actual = sha256(manifest)
        expected = metadata.get(key)
        if expected and actual != expected:
            raise SystemExit(f"{manifest} sha256 changed: expected {expected}, got {actual}")

    train_count, train_missing = check_manifest(train_manifest)
    validation_count, validation_missing = check_manifest(validation_manifest)
    used_count = None
    used_missing: list[Path] = []
    if used_relative_manifest.exists():
        used_count, used_missing = check_relative_manifest(args.root, used_relative_manifest)

    if train_missing or validation_missing or used_missing:
        print("Missing files:")
        for path in train_missing + validation_missing + used_missing:
            print(path)
        raise SystemExit(1)

    print(f"split root: {split_root}")
    print(f"train files: {train_count}")
    print(f"validation files: {validation_count}")
    if used_count is not None:
        print(f"used relative files: {used_count}")
    print("split check: OK")


if __name__ == "__main__":
    main()
