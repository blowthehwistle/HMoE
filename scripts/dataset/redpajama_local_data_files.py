#!/usr/bin/env python3
"""Print a comma-separated local RedPajama shard list for TOML/CLI overrides."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("redpajama_1T/v1.0.0"))
    parser.add_argument("--patterns", nargs="+", default=["c4/*.jsonl", "github/*.jsonl", "arxiv/*.jsonl"])
    args = parser.parse_args()

    files: list[Path] = []
    for pattern in args.patterns:
        files.extend(sorted(args.root.glob(pattern)))

    if not files:
        raise SystemExit(f"No files found under {args.root} for patterns {args.patterns}")

    print(",".join(str(path) for path in files))


if __name__ == "__main__":
    main()
