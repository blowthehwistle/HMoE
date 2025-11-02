# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import time

from typing import Optional

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type
from torchtitan.components.metrics import build_device_memory_monitor

from ttp.inference.generation import generate
from ttp.inference.build_model import init_config, build_tokenizer_and_model


@record
def test_generate(
    config_path: str,
    checkpoint_path: str,
    prompt: str,
    *,
    temperature: float,
    max_new_tokens: int,
    batch_size: int,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
):
    color = utils.Color

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Load configuration from toml file
    config = init_config(config_path)
    # Load Model
    device_memory_monitor = build_device_memory_monitor()

    tokenizer, model = build_tokenizer_and_model(
            device_memory_monitor,
            config,
            checkpoint_path,
            tp_size=1,
            dp_shard=1,
    )
    
    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )
    device_memory_monitor.reset_peak_stats()

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    if len(args.prompt) == 0:
        logger.warning(
            "The input prompt is empty, model will respond from a empty sequence."
        )

    # Tokenize prompt and repeat batch_size times
    input_ids = (
        (
            torch.tensor(
                tokenizer.encode(prompt, add_bos=True, add_eos=False), dtype=torch.long
            )
            .view(1, -1)
            .repeat(batch_size, 1)
        )
    ).to(device_type)

    device_memory_monitor.reset_peak_stats()

    # Run generation
    t0 = time.monotonic()
    responses = generate(
        model,
        input_ids,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        seed=seed,
        use_kv_caches=True,
    )
    t1 = time.monotonic()
    elapsed_sec = t1 - t0

    # Post process
    B, T = responses.size()  # B: batch_size, T: total seq length
    input_n_tokens = input_ids.size(1)
    generated_n_tokens = T - input_n_tokens  # == max_new_tokens

    if local_rank == 0:
        logger.info(f"Generation completed in {elapsed_sec:.2f} seconds.")

        r, b = color.red, color.blue

        output_data = {
            "metadata": {},
            "responses": [],
        }

        for i, tokens in enumerate(responses):
            inp_tok = tokens[:input_n_tokens].tolist()
            out_tok = tokens[input_n_tokens:].tolist()

            input_text = tokenizer.decode(inp_tok)
            output_text = tokenizer.decode(out_tok)

            _data = {
                "response_idx": i,
                "input_text": input_text,
                "output_text": output_text,
            }
            output_data["responses"].append(_data)

            logger.info(f"{r}\n{input_text}{b}{output_text}\n{color.reset}")

        device_mem_stats = device_memory_monitor.get_peak_stats()
        output_data["metadata"] = {
            "generated_n_tokens": generated_n_tokens,
            "input_n_tokens": input_n_tokens,
            "generation_time_sec": elapsed_sec,
            "tokens_per_sec": (B * T) / elapsed_sec,
            "batch_size": B,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
            "world_size": world_size,
            "torch_version": torch.__version__,
        }

        if args.out:
            print(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation")
    parser.add_argument(
        "--config", type=str, required=True, help="TOML config file path (required)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path to load (required)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Default is 1.0",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max number of tokens to generate. Default is 32",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of samples to run in batch"
    )
    parser.add_argument(
        "--top_k", type=int, help="Prune to select from top_k probabilities. Optional"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic algorithms wherever possible, may be slower",
    )

    parser.add_argument("--prompt", type=str, default="", help="Input prompt")

    parser.add_argument(
        "--out",
        action="store_true",
        default=False,
        help="If specified, prints the report to stdout. Defaults to no output.",
    )

    args = parser.parse_args()

    test_generate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed,
        deterministic=args.deterministic,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
