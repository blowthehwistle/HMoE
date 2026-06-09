#!/usr/bin/env python3
"""Compute paper-target FLOPs milestones for the HMoE-0.4B reproduction."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path


PAPER_ACTIVE_PARAMS = 153_000_000
PAPER_TARGET_FLOPS = 7e19


def load_training_config(path: Path) -> dict:
    with path.open("rb") as f:
        config = tomllib.load(f)
    training = config["training"]
    return {
        "steps": int(training["steps"]),
        "global_batch_size": int(training["global_batch_size"]),
        "seq_len": int(training["seq_len"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="ttp/experiments/hmoe/train_configs/paper_hmoe_0_4b_llama2_repro.toml",
        type=Path,
    )
    parser.add_argument("--active-params", type=int, default=PAPER_ACTIVE_PARAMS)
    parser.add_argument("--target-flops", type=float, default=PAPER_TARGET_FLOPS)
    parser.add_argument("--as-json", action="store_true")
    args = parser.parse_args()

    cfg = load_training_config(args.config)
    tokens_per_step = cfg["global_batch_size"] * cfg["seq_len"]
    flops_per_token = 6 * args.active_params
    flops_per_step = flops_per_token * tokens_per_step
    target_steps = math.ceil(args.target_flops / flops_per_step)
    configured_flops = cfg["steps"] * flops_per_step

    report = {
        "config": str(args.config),
        "active_params": args.active_params,
        "target_flops": args.target_flops,
        "global_batch_size": cfg["global_batch_size"],
        "seq_len": cfg["seq_len"],
        "tokens_per_step": tokens_per_step,
        "flops_per_token": flops_per_token,
        "flops_per_step": flops_per_step,
        "target_steps": target_steps,
        "configured_steps": cfg["steps"],
        "configured_flops": configured_flops,
        "configured_flops_delta": configured_flops - args.target_flops,
    }

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"Config: {report['config']}")
    print(f"Active params: {report['active_params']:,}")
    print(f"Tokens / step: {report['tokens_per_step']:,}")
    print(f"FLOPs / token: {report['flops_per_token']:,}")
    print(f"FLOPs / step: {report['flops_per_step']:.4e}")
    print(f"Target FLOPs: {report['target_flops']:.4e}")
    print(f"Target steps: {report['target_steps']:,}")
    print(f"Configured steps: {report['configured_steps']:,}")
    print(f"Configured FLOPs: {report['configured_flops']:.4e}")
    print(f"Delta vs target: {report['configured_flops_delta']:.4e}")


if __name__ == "__main__":
    main()
