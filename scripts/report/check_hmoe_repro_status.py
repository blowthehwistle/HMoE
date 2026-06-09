#!/usr/bin/env python3
"""Report checkpoint/FLOPs progress for the HMoE-0.4B reproduction run."""

from __future__ import annotations

import argparse
import math
import re
import tomllib
from pathlib import Path


PAPER_ACTIVE_PARAMS = 153_000_000
PAPER_TARGET_FLOPS = 7e19


def read_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def checkpoint_steps(folder: Path) -> list[int]:
    if not folder.is_dir():
        return []
    steps = []
    for child in folder.iterdir():
        match = re.fullmatch(r"step-(\d+)", child.name)
        if match and child.is_dir():
            steps.append(int(match.group(1)))
    return sorted(steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ttp/experiments/hmoe/train_configs/paper_hmoe_0_4b_llama2_repro.toml"),
    )
    parser.add_argument("--active-params", type=int, default=PAPER_ACTIVE_PARAMS)
    parser.add_argument("--target-flops", type=float, default=PAPER_TARGET_FLOPS)
    args = parser.parse_args()

    cfg = read_config(args.config)
    train = cfg["training"]
    checkpoint = cfg["checkpoint"]
    dump_folder = Path(cfg["job"]["dump_folder"])
    ckpt_folder = dump_folder / checkpoint["folder"]

    tokens_per_step = int(train["global_batch_size"]) * int(train["seq_len"])
    flops_per_step = 6 * args.active_params * tokens_per_step
    target_steps = math.ceil(args.target_flops / flops_per_step)
    steps = checkpoint_steps(ckpt_folder)
    latest = steps[-1] if steps else 0
    latest_flops = latest * flops_per_step

    print(f"Config: {args.config}")
    print(f"Checkpoint folder: {ckpt_folder}")
    print(f"Latest checkpoint step: {latest:,}" if latest else "Latest checkpoint step: none")
    print(f"Configured final step: {int(train['steps']):,}")
    print(f"Target step for {args.target_flops:.2e} FLOPs: {target_steps:,}")
    print(f"Estimated FLOPs at latest checkpoint: {latest_flops:.4e}")
    print(f"Progress vs target FLOPs: {100.0 * latest_flops / args.target_flops:.2f}%")
    if steps:
        print("Recent checkpoints: " + ", ".join(f"step-{step}" for step in steps[-5:]))


if __name__ == "__main__":
    main()
