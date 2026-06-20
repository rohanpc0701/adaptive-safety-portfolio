"""
benchmark.py

Runs every allocator (single arms, always-all baseline, cascade) against the
benchmark dataset and produces:
  1. A results table (precision, recall, FPR, avg cost, avg latency per allocator)
  2. The money chart: recall vs. average compute cost, one point per allocator,
     showing the cascade getting close to "always-all" recall at a fraction
     of the cost.

Run: python3 benchmark.py
Output: results/results_table.csv, results/recall_vs_cost.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import matplotlib.pyplot as plt

from detectors import build_portfolio
from cascade import CascadeAllocator, AlwaysAllAllocator, SingleArmAllocator
from load_data import get_benchmark_data


def compute_metrics(predictions, labels, costs, latencies):
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    avg_cost = sum(costs) / len(costs) if costs else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "accuracy": accuracy,
        "avg_cost": avg_cost,
        "avg_latency_ms": avg_latency,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def run_allocator(allocator, prompts, labels, name):
    predictions = []
    costs = []
    latencies = []
    decisions = []

    for prompt in prompts:
        decision = allocator.evaluate(prompt)
        predictions.append(1 if decision.is_harmful else 0)
        costs.append(decision.total_cost)
        latencies.append(decision.total_latency_ms)
        decisions.append(decision)

    metrics = compute_metrics(predictions, labels, costs, latencies)
    metrics["name"] = name
    return metrics, decisions


def print_escalation_breakdown(decisions, prompts):
    """Shows how often the cascade stops early vs. escalates - the actual
    mechanism story, not just the aggregate recall/cost numbers."""
    from collections import Counter
    n_arms_counter = Counter(len(d.arms_called) for d in decisions)
    total = len(decisions)
    print("\nCascade escalation breakdown:")
    for n_arms in sorted(n_arms_counter):
        count = n_arms_counter[n_arms]
        pct = 100 * count / total
        print(f"  Stopped after {n_arms} arm(s): {count}/{total} inputs ({pct:.0f}%)")


def main():
    print("Loading benchmark data...")
    prompts, labels = get_benchmark_data()
    print(f"Loaded {len(prompts)} prompts ({sum(labels)} harmful, {len(labels) - sum(labels)} benign)\n")

    print("Building portfolio (MOCK_MODE - swap to real models on your GPU machine)...")
    portfolio = build_portfolio(use_real_claude=True)

    allocators = {
        "prompt_guard_only": SingleArmAllocator(portfolio["prompt_guard"]),
        "shieldgemma_only": SingleArmAllocator(portfolio["shieldgemma"]),
        "wildguard_only": SingleArmAllocator(portfolio["wildguard"]),
        "claude_judge_only": SingleArmAllocator(portfolio["claude_judge"]),
        "always_all": AlwaysAllAllocator(
            portfolio["prompt_guard"], portfolio["shieldgemma"],
            portfolio["wildguard"], portfolio["claude_judge"],
        ),
        "cascade": CascadeAllocator(
            portfolio["prompt_guard"], portfolio["shieldgemma"],
            portfolio["wildguard"], portfolio["claude_judge"],
        ),
    }

    all_metrics = []
    all_decisions = {}
    for name, allocator in allocators.items():
        print(f"Running {name}...")
        metrics, decisions = run_allocator(allocator, prompts, labels, name)
        all_metrics.append(metrics)
        all_decisions[name] = decisions

    df = pd.DataFrame(all_metrics)
    df = df[["name", "recall", "precision", "fpr", "accuracy", "avg_cost", "avg_latency_ms", "tp", "fp", "tn", "fn"]]
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    out_csv = os.path.join(os.path.dirname(__file__), "..", "results", "results_table.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults table saved to {out_csv}\n")
    print(df.to_string(index=False))

    print_escalation_breakdown(all_decisions["cascade"], prompts)

    # --- The money chart: recall vs. avg cost ---
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "prompt_guard_only": "#888888",
        "shieldgemma_only": "#888888",
        "wildguard_only": "#888888",
        "claude_judge_only": "#888888",
        "always_all": "#d62728",
        "cascade": "#2ca02c",
    }
    markers = {
        "prompt_guard_only": "o", "shieldgemma_only": "s", "wildguard_only": "^",
        "claude_judge_only": "D", "always_all": "*", "cascade": "*",
    }

    # Manual label offsets to avoid overlap (prompt_guard_only and cascade
    # land at nearly the same x position, so they need different offsets).
    label_offsets = {
        "prompt_guard_only": (10, -18),
        "shieldgemma_only": (10, -5),
        "wildguard_only": (10, 8),
        "claude_judge_only": (10, 8),
        "always_all": (-90, 8),
        "cascade": (10, 14),
    }

    for _, row in df.iterrows():
        name = row["name"]
        ax.scatter(
            row["avg_cost"], row["recall"],
            s=320 if name in ("cascade", "always_all") else 150,
            c=colors.get(name, "#1f77b4"),
            marker=markers.get(name, "o"),
            edgecolors="black",
            linewidths=1,
            label=name,
            zorder=3 if name in ("cascade", "always_all") else 2,
        )
        offset = label_offsets.get(name, (8, 5))
        ax.annotate(name, (row["avg_cost"], row["recall"]),
                    textcoords="offset points", xytext=offset, fontsize=9)

    # Explicit callout: the actual point of the chart.
    cascade_row = df[df["name"] == "cascade"].iloc[0]
    always_row = df[df["name"] == "always_all"].iloc[0]
    if always_row["avg_cost"] > 0:
        reduction = always_row["avg_cost"] / cascade_row["avg_cost"]
        ax.annotate(
            f"Cascade matches always-all recall\nat {reduction:.1f}x less compute",
            xy=(cascade_row["avg_cost"], cascade_row["recall"]),
            xytext=(cascade_row["avg_cost"] + 14, cascade_row["recall"] - 0.18),
            fontsize=10, fontweight="bold", color="#1a6e1a",
            arrowprops=dict(arrowstyle="->", color="#1a6e1a", lw=1.5),
        )

    ax.set_xlabel("Average compute cost per input (relative units)")
    ax.set_ylabel("Recall (fraction of harmful prompts caught)")
    ax.set_title("Recall vs. Compute Cost: Cascade vs. Single Arms vs. Always-All")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(__file__), "..", "results", "recall_vs_cost.png")
    plt.savefig(out_png, dpi=150)
    print(f"\nChart saved to {out_png}")

    return df, all_decisions


if __name__ == "__main__":
    main()
