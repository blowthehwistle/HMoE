#!/usr/bin/env python3
"""Create a small handoff bundle describing which RedPajama shards were used."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("redpajama_1T"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/handoff/redpajama"))
    args = parser.parse_args()

    split_root = args.root / "splits"
    metadata = split_root / "metadata.json"
    used = split_root / "used_shards_relative.txt"
    train = split_root / "train_shards_relative.txt"
    validation = split_root / "validation_shards_relative.txt"
    for path in (metadata, used, train, validation):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run build_redpajama_splits.py first.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (metadata, used, train, validation):
        shutil.copy2(path, args.output_dir / path.name)

    info = {
        "copied_from": str(split_root.resolve()),
        "usage": (
            "On the next workstation, pass "
            f"--exclude-relative-manifest {args.output_dir / used.name} "
            "to build_redpajama_splits.py to avoid reusing these shards."
        ),
    }
    (args.output_dir / "README.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"handoff bundle: {args.output_dir}")
    print(f"exclude manifest: {args.output_dir / used.name}")


if __name__ == "__main__":
    main()
