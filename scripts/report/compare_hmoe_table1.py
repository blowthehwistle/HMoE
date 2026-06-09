#!/usr/bin/env python3
"""Compare lm-eval JSON output against HMoE paper Table 1 baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PAPER_BASELINES = {
    "hmoe_0_4b_topk": {
        "label": "HMoE-0.4B Top-K",
        "scores": {
            "piqa": 56.67,
            "hellaswag": 28.26,
            "boolq": 59.80,
            "arc_easy": 31.93,
            "winogrande": 52.49,
            "social_iqa": 32.91,
        },
    },
    "moe_0_4b_topk": {
        "label": "MoE-0.4B Top-K",
        "scores": {
            "piqa": 57.67,
            "hellaswag": 27.81,
            "boolq": 62.13,
            "arc_easy": 29.70,
            "winogrande": 50.59,
            "social_iqa": 32.82,
        },
    },
}


def _extract_acc(metrics: dict[str, Any]) -> float | None:
    for key in ("acc,none", "acc_norm,none", "acc", "acc_norm"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value) * 100.0 if value <= 1.0 else float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lm_eval_json", type=Path)
    parser.add_argument(
        "--baseline",
        choices=sorted(PAPER_BASELINES),
        default="hmoe_0_4b_topk",
        help="Paper Table 1 row to compare against.",
    )
    parser.add_argument("--as-json", action="store_true")
    args = parser.parse_args()

    baseline = PAPER_BASELINES[args.baseline]
    paper_scores_by_task = baseline["scores"]

    with args.lm_eval_json.open() as f:
        payload = json.load(f)

    results = payload.get("results", payload)
    rows = []
    observed_scores = []
    paper_scores = []
    for task, paper_score in paper_scores_by_task.items():
        task_metrics = results.get(task, {})
        observed = _extract_acc(task_metrics) if isinstance(task_metrics, dict) else None
        if observed is not None:
            observed_scores.append(observed)
            paper_scores.append(paper_score)
        rows.append(
            {
                "task": task,
                "paper": paper_score,
                "observed": observed,
                "delta": None if observed is None else observed - paper_score,
            }
        )

    summary = {
        "baseline": args.baseline,
        "baseline_label": baseline["label"],
        "paper_avg": sum(paper_scores_by_task.values()) / len(paper_scores_by_task),
        "observed_avg": None if not observed_scores else sum(observed_scores) / len(observed_scores),
        "rows": rows,
    }
    if summary["observed_avg"] is not None:
        summary["delta_avg"] = summary["observed_avg"] - summary["paper_avg"]
    else:
        summary["delta_avg"] = None

    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(f"| Task | Paper {baseline['label']} | Observed | Delta |")
    print("|---|---:|---:|---:|")
    for row in rows:
        observed = "missing" if row["observed"] is None else f"{row['observed']:.2f}"
        delta = "missing" if row["delta"] is None else f"{row['delta']:+.2f}"
        print(f"| {row['task']} | {row['paper']:.2f} | {observed} | {delta} |")
    observed_avg = "missing" if summary["observed_avg"] is None else f"{summary['observed_avg']:.2f}"
    delta_avg = "missing" if summary["delta_avg"] is None else f"{summary['delta_avg']:+.2f}"
    print(f"| AVG | {summary['paper_avg']:.2f} | {observed_avg} | {delta_avg} |")


if __name__ == "__main__":
    main()
