"""
precision_fix_eval.py

Evaluates the low-confidence escalation fix to the cascade's precision gap.

PROBLEM DIAGNOSED IN FINDINGS:
  The fixed cascade precision (0.748) is worse than always-all (0.830) because
  Claude's better precision (0.839) only gets applied when mid-tier arms
  *disagree*. When ShieldGemma and WildGuard agree on a false positive
  (both score a safe prompt at 0.6) — a common case since both have high FPR
  individually — nothing corrects them. Only 22% of inputs reach Claude.

FIX:
  CascadeAllocator now accepts low_conf_low / low_conf_high params. When mid-tier
  arms agree (disagreement < threshold) BUT their average score falls in the
  low-confidence window [low_conf_low, low_conf_high], escalate to Claude anyway.
  Claude's precision overrides the uncertain agreement.

THIS SCRIPT:
  Sweeps the low-confidence window width on synthetic data, showing the
  precision/recall/cost tradeoff. Identifies the window that best improves
  precision without sacrificing recall. Two key comparisons:
    1. Original cascade (no fix, window disabled) vs. fixed cascade
    2. Always-all (upper precision bound) vs. fixed cascade
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from cascade import CascadeAllocator, CascadeDecision
from bandit_benchmark import build_synthetic_detectors, _run_cascade_synthetic
from load_data import get_benchmark_data

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Reference numbers from the real benchmark run (results_table.csv)
REAL_BASELINE = {
    "cascade_original": {"precision": 0.748, "recall": 0.95, "fpr": 0.32, "avg_cost": 31.1},
    "always_all":       {"precision": 0.830, "recall": 0.93, "fpr": 0.19, "avg_cost": 49.0},
}


def run_cascade_with_low_conf(pg, sg, wg, cj, prompts, labels,
                               low_conf_low, low_conf_high,
                               confident_high=0.75, disagreement_threshold=0.35,
                               decision_threshold=0.5):
    always_all_cost = pg.cost + sg.cost + wg.cost + cj.cost
    decisions = []
    for prompt, label in zip(prompts, labels):
        # Run stage 1
        pg_r = pg.score_with_label(prompt, label)
        arms_called = [pg.name]
        total_cost = pg_r.cost
        total_latency = pg_r.latency_ms
        trace = [{"arm": pg.name, "score": pg_r.score}]

        if pg_r.score >= confident_high:
            decisions.append(CascadeDecision(
                final_score=pg_r.score, is_harmful=True,
                arms_called=arms_called, total_cost=total_cost,
                total_latency_ms=total_latency, trace=trace))
            continue

        # Stage 2
        sg_r = sg.score_with_label(prompt, label)
        wg_r = wg.score_with_label(prompt, label)
        arms_called.extend([sg.name, wg.name])
        total_cost += sg_r.cost + wg_r.cost
        total_latency += sg_r.latency_ms + wg_r.latency_ms
        trace.append({"arm": sg.name, "score": sg_r.score})
        trace.append({"arm": wg.name, "score": wg_r.score})

        mid_avg = (sg_r.score + wg_r.score) / 2.0
        disagreement = abs(sg_r.score - wg_r.score)
        low_conf = low_conf_low <= mid_avg <= low_conf_high

        if disagreement < disagreement_threshold and not low_conf:
            decisions.append(CascadeDecision(
                final_score=mid_avg,
                is_harmful=mid_avg >= decision_threshold,
                arms_called=arms_called, total_cost=total_cost,
                total_latency_ms=total_latency, trace=trace))
            continue

        # Stage 3
        cj_r = cj.score_with_label(prompt, label)
        arms_called.append(cj.name)
        total_cost += cj_r.cost
        total_latency += cj_r.latency_ms
        trace.append({"arm": cj.name, "score": cj_r.score})

        final_score = (mid_avg + cj_r.score) / 2.0
        decisions.append(CascadeDecision(
            final_score=final_score,
            is_harmful=final_score >= decision_threshold,
            arms_called=arms_called, total_cost=total_cost,
            total_latency_ms=total_latency, trace=trace))

    return decisions


def metrics(decisions, labels):
    tp = sum(d.is_harmful and l == 1 for d, l in zip(decisions, labels))
    tn = sum(not d.is_harmful and l == 0 for d, l in zip(decisions, labels))
    fp = sum(d.is_harmful and l == 0 for d, l in zip(decisions, labels))
    fn = sum(not d.is_harmful and l == 1 for d, l in zip(decisions, labels))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    avg_cost = np.mean([d.total_cost for d in decisions])
    pct_claude = sum(1 for d in decisions if "claude_judge" in d.arms_called) / len(decisions)
    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "fpr": round(fpr, 3),
        "avg_cost": round(float(avg_cost), 2),
        "pct_claude": round(pct_claude, 3),
    }


def main():
    print("Loading data...")
    prompts, labels = get_benchmark_data(seed=42)
    print(f"  {len(prompts)} prompts")

    print("Building synthetic detectors...")
    pg, sg, wg, cj = build_synthetic_detectors()

    # Sweep low_conf window widths (symmetric around 0.5)
    # Width 0.0 = original cascade (no fix), 0.6 = widest window [0.2, 0.8]
    widths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    rows = []

    print("\nSweeping low-confidence window width...")
    print(f"{'Window':<18} {'Recall':>8} {'Precision':>10} {'FPR':>6} "
          f"{'AvgCost':>8} {'%Claude':>8}")
    print("-" * 62)

    for w in widths:
        low_conf_low = 0.5 - w / 2
        low_conf_high = 0.5 + w / 2
        label = f"[{low_conf_low:.2f},{low_conf_high:.2f}]" if w > 0 else "disabled"

        decisions = run_cascade_with_low_conf(
            pg, sg, wg, cj, prompts, labels,
            low_conf_low=low_conf_low if w > 0 else 1.1,  # 1.1 = unreachable = disabled
            low_conf_high=low_conf_high if w > 0 else -0.1,
        )
        m = metrics(decisions, labels)
        rows.append({"window": label, "width": w, **m})
        print(f"{label:<18} {m['recall']:>8.3f} {m['precision']:>10.3f} "
              f"{m['fpr']:>6.3f} {m['avg_cost']:>8.2f} {m['pct_claude']:>8.1%}")

    df = pd.DataFrame(rows)

    print("\nReference (real model benchmark):")
    print(f"  original cascade  recall=0.950 precision=0.748 fpr=0.32 avg_cost=31.10")
    print(f"  always-all        recall=0.930 precision=0.830 fpr=0.19 avg_cost=49.00")

    # Find best tradeoff: max precision gain with recall >= 0.93
    viable = df[df["recall"] >= 0.93]
    if not viable.empty:
        best = viable.loc[viable["precision"].idxmax()]
        print(f"\nBest window (recall≥0.93): {best['window']}  "
              f"precision={best['precision']:.3f}  cost={best['avg_cost']:.2f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, "precision_fix_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    widths_plot = df["width"].values
    ax = axes[0]
    ax.plot(widths_plot, df["precision"], "o-", color="#2ca02c", lw=2, label="Precision")
    ax.plot(widths_plot, df["recall"], "s--", color="#1f77b4", lw=2, label="Recall")
    ax.axhline(REAL_BASELINE["always_all"]["precision"], ls=":", color="#2ca02c",
               alpha=0.6, label=f"always-all precision ({REAL_BASELINE['always_all']['precision']})")
    ax.axhline(REAL_BASELINE["always_all"]["recall"], ls=":", color="#1f77b4",
               alpha=0.6, label=f"always-all recall ({REAL_BASELINE['always_all']['recall']})")
    ax.set_xlabel("Low-confidence window width\n(0 = original, disabled)")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall\nvs Low-Conf Window Width")
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.1)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(widths_plot, df["avg_cost"], "o-", color="#d62728", lw=2)
    ax2.axhline(REAL_BASELINE["always_all"]["avg_cost"], ls=":", color="#d62728",
                alpha=0.6, label=f"always-all cost ({REAL_BASELINE['always_all']['avg_cost']})")
    ax2.set_xlabel("Low-confidence window width")
    ax2.set_ylabel("Avg cost per input")
    ax2.set_title("Compute Cost\nvs Low-Conf Window Width")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    ax3.plot(widths_plot, df["pct_claude"] * 100, "o-", color="#9467bd", lw=2)
    ax3.set_xlabel("Low-confidence window width")
    ax3.set_ylabel("% inputs reaching Claude")
    ax3.set_title("Claude Utilisation\n(was 22% in original cascade)")
    ax3.axhline(22, ls=":", color="#9467bd", alpha=0.6, label="original 22%")
    ax3.axhline(100, ls=":", color="gray", alpha=0.3, label="always-all 100%")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    fig.suptitle("Low-Confidence Escalation Fix: Precision/Recall/Cost Tradeoff\n"
                 "(synthetic calibrated scores — real-model numbers will differ)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "precision_fix_sweep.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
