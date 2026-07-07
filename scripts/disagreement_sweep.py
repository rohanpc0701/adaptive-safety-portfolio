"""
disagreement_sweep.py

Sweep cascade disagreement_threshold to map the precision / recall / judge-utilization
tradeoff. Default: calibrated synthetic scores (~10s, no GPU). Pass --real for GPU run.

Output: results/disagreement_sweep.csv, results/disagreement_sweep.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from cascade import CascadeAllocator
from load_data import get_benchmark_data
from metrics_utils import metrics_from_decisions
from bandit_benchmark import build_synthetic_detectors, build_real_detectors, _run_cascade_synthetic

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


def _evaluate_synthetic_cascade(pg, sg, wg, cj, prompts, labels, disagreement_threshold):
    decisions = []
    for prompt, label in zip(prompts, labels):
        d = _run_cascade_synthetic(
            pg, sg, wg, cj, prompt, label,
            confident_high=0.75,
            disagreement_threshold=disagreement_threshold,
            decision_threshold=0.5,
        )
        decisions.append(d)
    return decisions


def _evaluate_real_cascade(pg, sg, wg, cj, prompts, disagreement_threshold):
    alloc = CascadeAllocator(
        pg, sg, wg, cj,
        confident_low=-1.0,
        confident_high=0.75,
        disagreement_threshold=disagreement_threshold,
    )
    return [alloc.evaluate(p) for p in prompts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Real models (GPU + API key)")
    args = parser.parse_args()

    prompts, labels = get_benchmark_data()
    print(f"Loaded {len(prompts)} prompts\n")

    if args.real:
        pg, sg, wg, cj = build_real_detectors()
        eval_fn = lambda dt: _evaluate_real_cascade(pg, sg, wg, cj, prompts, dt)
        mode = "real"
    else:
        pg, sg, wg, cj = build_synthetic_detectors()
        eval_fn = lambda dt: _evaluate_synthetic_cascade(pg, sg, wg, cj, prompts, labels, dt)
        mode = "synthetic"

    rows = []
    for dt in THRESHOLDS:
        decisions = eval_fn(dt)
        m = metrics_from_decisions(decisions, labels)
        rows.append({
            "disagreement_threshold": dt,
            "recall": round(m["recall"], 4),
            "precision": round(m["precision"], 4),
            "fpr": round(m["fpr"], 4),
            "avg_cost": round(m["avg_cost"], 2),
            "avg_latency_ms": round(m["avg_latency_ms"], 1),
            "pct_judge": round(m["pct_judge"], 4),
        })
        print(
            f"dt={dt:.2f}  recall={m['recall']:.3f}  precision={m['precision']:.3f}  "
            f"judge={m['pct_judge']:.1%}  cost={m['avg_cost']:.1f}"
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, f"disagreement_sweep_{mode}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax1, ax2, ax3 = axes
    x = df["disagreement_threshold"]

    ax1.plot(x, df["recall"], "o-", color="#2ca02c", lw=2, label="Recall")
    ax1.plot(x, df["precision"], "s-", color="#1f77b4", lw=2, label="Precision")
    ax1.set_xlabel("Disagreement threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Precision–recall vs disagreement threshold")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2.plot(x, df["pct_judge"] * 100, "o-", color="#9467bd", lw=2)
    ax2.set_xlabel("Disagreement threshold")
    ax2.set_ylabel("% reaching judge arm")
    ax2.set_title("Judge utilization")
    ax2.grid(alpha=0.3)

    ax3.plot(x, df["avg_cost"], "o-", color="#d62728", lw=2)
    ax3.set_xlabel("Disagreement threshold")
    ax3.set_ylabel("Avg cost (proxy units)")
    ax3.set_title("Compute cost")
    ax3.grid(alpha=0.3)

    fig.suptitle(f"Cascade disagreement threshold sweep ({mode})", fontweight="bold")
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, f"disagreement_sweep_{mode}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
