#!/usr/bin/env python3
"""Export a matched HMoE/MoE TensorBoard comparison to CSV."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_RUNS = {
    "HMoE": Path("outputs/mini_25g/hmoe_0_4b_topk"),
    "MoE": Path("outputs/mini_25g/moe_0_4b_topk"),
}

STEP_TAGS = {
    "train_loss": "loss_metrics/global_avg_loss",
    "lm_loss": "loss/lm_loss",
    "grad_norm": "grad_norm",
    "throughput_tps": "throughput(tps)",
    "step_time_s": "time_metrics/end_to_end(s)",
    "data_loading_s": "time_metrics/data_loading(s)",
    "data_loading_pct": "time_metrics/data_loading(%)",
    "memory_max_active_gib": "memory/max_active(GiB)",
    "memory_max_reserved_gib": "memory/max_reserved(GiB)",
    "estimated_tflops": "estimated_tflops",
    "estimated_mfu_pct": "estimated_mfu(%)",
    "tokens_seen": "n_tokens_seen",
    "validation_loss": "validation_metrics/loss",
    "validation_throughput_tps": "validation_metrics/throughput(tps)",
    "aux_p_penalty_loss": "loss/aux_p_penalty_loss",
    "aux_load_balance_loss": "loss/aux_load_balance_loss",
}


def latest_event_file(run_dir: Path) -> Path:
    files = list((run_dir / "tb").glob("*/events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event file found under {run_dir / 'tb'}")
    return max(files, key=lambda path: path.stat().st_mtime)


def load_scalars(run_dir: Path) -> tuple[Path, dict[str, list]]:
    event_file = latest_event_file(run_dir)
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0}).Reload()
    return event_file, {
        tag: accumulator.Scalars(tag) for tag in accumulator.Tags()["scalars"]
    }


def write_step_metrics(output: Path, runs: dict[str, dict[str, list]]) -> None:
    fieldnames = ["model", "step", *STEP_TAGS]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for model, scalars in runs.items():
            by_step: dict[int, dict[str, float | str | int]] = {}
            for column, tag in STEP_TAGS.items():
                for event in scalars.get(tag, []):
                    by_step.setdefault(event.step, {"model": model, "step": event.step})
                    by_step[event.step][column] = event.value
            writer.writerows(by_step[step] for step in sorted(by_step))


def write_imbalance(output: Path, runs: dict[str, dict[str, list]]) -> None:
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["model", "step", "layer", "relative_imbalance"]
        )
        writer.writeheader()
        for model, scalars in runs.items():
            for tag, events in sorted(scalars.items()):
                if not tag.startswith("moe_imbalance/"):
                    continue
                layer = tag.split("/")[1]
                for event in events:
                    writer.writerow(
                        {
                            "model": model,
                            "step": event.step,
                            "layer": layer,
                            "relative_imbalance": event.value,
                        }
                    )


def value_at(scalars: dict[str, list], tag: str, step: int) -> float:
    return next(event.value for event in scalars[tag] if event.step == step)


def tail_mean(scalars: dict[str, list], tag: str, start_step: int) -> float:
    return statistics.fmean(
        event.value for event in scalars[tag] if event.step >= start_step
    )


def imbalance_summary(scalars: dict[str, list], start_step: int) -> tuple[float, float]:
    values = [
        event.value
        for tag, events in scalars.items()
        if tag.startswith("moe_imbalance/")
        for event in events
        if event.step >= start_step
    ]
    return statistics.fmean(values), max(values)


def write_summary(
    output: Path,
    runs: dict[str, dict[str, list]],
    event_files: dict[str, Path],
    tail_start: int,
) -> None:
    hmoe, moe = runs["HMoE"], runs["MoE"]
    hmoe_imbalance_mean, hmoe_imbalance_max = imbalance_summary(hmoe, tail_start)
    moe_imbalance_mean, moe_imbalance_max = imbalance_summary(moe, tail_start)
    rows = [
        ("validation_loss_step_50", value_at(hmoe, "validation_metrics/loss", 50), value_at(moe, "validation_metrics/loss", 50), "TensorBoard scalar at step 50"),
        ("validation_loss_step_100", value_at(hmoe, "validation_metrics/loss", 100), value_at(moe, "validation_metrics/loss", 100), "TensorBoard scalar at step 100"),
        ("lm_loss_step_100", value_at(hmoe, "loss/lm_loss", 100), value_at(moe, "loss/lm_loss", 100), "TensorBoard scalar at step 100"),
        ("throughput_tps_tail_mean", tail_mean(hmoe, "throughput(tps)", tail_start), tail_mean(moe, "throughput(tps)", tail_start), f"Mean of logged values at steps >= {tail_start}"),
        ("step_time_s_tail_mean", tail_mean(hmoe, "time_metrics/end_to_end(s)", tail_start), tail_mean(moe, "time_metrics/end_to_end(s)", tail_start), f"Mean per-step time at logged steps >= {tail_start}"),
        ("estimated_mfu_pct_tail_mean", tail_mean(hmoe, "estimated_mfu(%)", tail_start), tail_mean(moe, "estimated_mfu(%)", tail_start), f"Mean of logged values at steps >= {tail_start}"),
        ("memory_max_active_gib_tail_mean", tail_mean(hmoe, "memory/max_active(GiB)", tail_start), tail_mean(moe, "memory/max_active(GiB)", tail_start), f"Mean of logged values at steps >= {tail_start}"),
        ("memory_max_reserved_gib_tail_mean", tail_mean(hmoe, "memory/max_reserved(GiB)", tail_start), tail_mean(moe, "memory/max_reserved(GiB)", tail_start), f"Mean of logged values at steps >= {tail_start}"),
        ("relative_imbalance_tail_mean", hmoe_imbalance_mean, moe_imbalance_mean, f"Mean across all MoE layers and logged steps >= {tail_start}"),
        ("relative_imbalance_tail_max", hmoe_imbalance_max, moe_imbalance_max, f"Maximum across all MoE layers and logged steps >= {tail_start}"),
    ]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "hmoe",
                "moe",
                "absolute_delta_hmoe_minus_moe",
                "relative_delta_pct",
                "aggregation",
                "hmoe_event_file",
                "moe_event_file",
            ],
        )
        writer.writeheader()
        for metric, hmoe_value, moe_value, aggregation in rows:
            writer.writerow(
                {
                    "metric": metric,
                    "hmoe": hmoe_value,
                    "moe": moe_value,
                    "absolute_delta_hmoe_minus_moe": hmoe_value - moe_value,
                    "relative_delta_pct": 100 * (hmoe_value / moe_value - 1),
                    "aggregation": aggregation,
                    "hmoe_event_file": event_files["HMoE"],
                    "moe_event_file": event_files["MoE"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmoe-run", type=Path, default=DEFAULT_RUNS["HMoE"])
    parser.add_argument("--moe-run", type=Path, default=DEFAULT_RUNS["MoE"])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/mini_25g/comparison")
    )
    parser.add_argument("--tail-start", type=int, default=50)
    args = parser.parse_args()

    run_dirs = {"HMoE": args.hmoe_run, "MoE": args.moe_run}
    loaded = {model: load_scalars(run_dir) for model, run_dir in run_dirs.items()}
    event_files = {model: result[0] for model, result in loaded.items()}
    runs = {model: result[1] for model, result in loaded.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(args.output_dir / "summary.csv", runs, event_files, args.tail_start)
    write_step_metrics(args.output_dir / "metrics_by_step.csv", runs)
    write_imbalance(args.output_dir / "imbalance_by_layer.csv", runs)

    for filename in ("summary.csv", "metrics_by_step.csv", "imbalance_by_layer.csv"):
        print(args.output_dir / filename)


if __name__ == "__main__":
    main()
