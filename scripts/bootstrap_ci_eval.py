"""
bootstrap_ci_eval.py

Bootstrap 95% confidence intervals for cascade vs always-all on JBB-Behaviors.
Default: synthetic calibrated scores (~5s). Pass --real for GPU run.

Output: results/bootstrap_ci.csv, results/bootstrap_ci.json
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from cascade import CascadeAllocator, AlwaysAllAllocator, SingleArmAllocator
from load_data import get_benchmark_data
from metrics_utils import bootstrap_ci
from bandit_benchmark import build_synthetic_detectors, build_real_detectors, _run_cascade_synthetic

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _run_synthetic(pg, sg, wg, cj, prompts, labels, policy: str):
    preds = []
    if policy == "cascade":
        for p, l in zip(prompts, labels):
            d = _run_cascade_synthetic(pg, sg, wg, cj, p, l, 0.75, 0.35, 0.5)
            preds.append(1 if d.is_harmful else 0)
    elif policy == "always_all":
        for p, l in zip(prompts, labels):
            scores = [a.score_with_label(p, l).score for a in (pg, sg, wg, cj)]
            preds.append(1 if sum(scores) / 4 >= 0.5 else 0)
    else:
        raise ValueError(policy)
    return preds


def _run_real(portfolio, prompts, policy: str):
    if policy == "cascade":
        alloc = CascadeAllocator(
            portfolio["prompt_guard"], portfolio["shieldgemma"],
            portfolio["wildguard"], portfolio["claude_judge"],
            confident_low=-1.0,
        )
    else:
        alloc = AlwaysAllAllocator(
            portfolio["prompt_guard"], portfolio["shieldgemma"],
            portfolio["wildguard"], portfolio["claude_judge"],
        )
    return [1 if alloc.evaluate(p).is_harmful else 0 for p in prompts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()

    prompts, labels = get_benchmark_data()
    mode = "real" if args.real else "synthetic"

    rows = []
    for policy in ("cascade", "always_all"):
        if args.real:
            from detectors import build_portfolio
            portfolio = build_portfolio(use_real_claude=True)
            preds = _run_real(portfolio, prompts, policy)
        else:
            pg, sg, wg, cj = build_synthetic_detectors()
            preds = _run_synthetic(pg, sg, wg, cj, prompts, labels, policy)

        for metric in ("recall", "precision", "fpr"):
            point, lo, hi = bootstrap_ci(preds, labels, metric=metric)
            rows.append({
                "policy": policy,
                "metric": metric,
                "point": round(point, 4),
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
                "mode": mode,
            })
            print(f"{policy:12} {metric:10} {point:.3f}  [{lo:.3f}, {hi:.3f}]")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, f"bootstrap_ci_{mode}.csv")
    df.to_csv(csv_path, index=False)
    json_path = os.path.join(RESULTS_DIR, f"bootstrap_ci_{mode}.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
