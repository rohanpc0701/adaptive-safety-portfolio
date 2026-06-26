"""
bandit_benchmark.py

Simulates online learning: presents JBB-Behaviors prompts one at a time,
updates the EXP3 bandit after each true label is revealed, and tracks how
metrics evolve as the bandit learns.

Compares:
  - Fixed-threshold cascade (CascadeAllocator, thresholds from real benchmark)
  - EXP3-adaptive cascade (EXP3CascadeAllocator, learns online from same stream)

Two modes:
  --synthetic  Calibrated synthetic scores (default; fast, no GPU). Runs in ~5s.
               Per-prompt scores sampled from Beta distributions fitted to
               the real benchmark recall/FPR per arm. Bandit learning curves
               are realistic; exact numbers will differ from real-model run.
  (no flag)    Real model weights. Needs GPU; ~hours on JBB-200.

Output:
  results/bandit_learning_curve.png  — rolling recall/precision/cost vs time
  results/bandit_vs_fixed.csv        — final metrics comparison table
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

import detectors as det_module
from detectors import (
    PromptGuardDetector, ShieldGemmaDetector,
    WildGuardDetector, ClaudeJudgeDetector,
    DetectorResult,
)
from cascade import CascadeAllocator, CascadeDecision
from bandit_cascade import EXP3CascadeAllocator, STAGE1_THRESHOLDS
from load_data import get_benchmark_data

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ---------------------------------------------------------------------------
# Synthetic scoring calibrated to real benchmark results
# (from results/results_table.csv).
# For each arm, per-prompt scores are sampled from a Beta distribution whose
# mean is set to the arm's real recall (for harmful prompts) or FPR (for
# benign prompts), with spread controlled by `concentration`. This produces
# realistic score distributions without needing GPU.
# ---------------------------------------------------------------------------
_ARM_PARAMS = {
    # arm_name: (recall_mean, fpr_mean, cost, latency_ms_mean)
    "prompt_guard_2_86m": (0.31, 0.17, 1.0, 81.0),
    "shieldgemma_2b":     (0.96, 0.33, 8.0, 19.0),
    "wildguard_7b":       (0.98, 0.40, 25.0, 457.0),
    "claude_judge":       (0.94, 0.18, 15.0, 1530.0),
}
_BETA_CONCENTRATION = 4.0  # higher = tighter scores around the mean


def _beta_params(mean: float, concentration: float):
    """Convert mean + concentration to Beta(alpha, beta) params."""
    mean = np.clip(mean, 0.01, 0.99)
    alpha = mean * concentration
    beta = (1 - mean) * concentration
    return alpha, beta


class SyntheticDetector:
    """
    Drops in as a BaseDetector replacement for synthetic (no-GPU) runs.
    Scores are sampled per-prompt from a Beta distribution calibrated to
    the real benchmark recall and FPR for this arm.
    """
    def __init__(self, name, recall_mean, fpr_mean, cost, latency_mean):
        self.name = name
        self.cost = cost
        self._recall_mean = recall_mean
        self._fpr_mean = fpr_mean
        self._latency_mean = latency_mean
        self._rng = np.random.default_rng()

    def score(self, prompt: str, label: int = None) -> DetectorResult:
        raise RuntimeError("Use score_with_label() in synthetic mode")

    def score_with_label(self, prompt: str, label: int) -> DetectorResult:
        mean = self._recall_mean if label == 1 else self._fpr_mean
        alpha, beta = _beta_params(mean, _BETA_CONCENTRATION)
        score = float(self._rng.beta(alpha, beta))
        latency = float(self._rng.normal(self._latency_mean, self._latency_mean * 0.1))
        return DetectorResult(score=score, cost=self.cost, latency_ms=max(1.0, latency))


def build_synthetic_detectors():
    detectors = {}
    for name, (recall, fpr, cost, lat) in _ARM_PARAMS.items():
        detectors[name] = SyntheticDetector(name, recall, fpr, cost, lat)
    return (
        detectors["prompt_guard_2_86m"],
        detectors["shieldgemma_2b"],
        detectors["wildguard_7b"],
        detectors["claude_judge"],
    )


def build_real_detectors():
    det_module.MOCK_MODE = False
    return (
        PromptGuardDetector(),
        ShieldGemmaDetector(),
        WildGuardDetector(),
        ClaudeJudgeDetector(),
    )


# ---------------------------------------------------------------------------
# Cascade runners (synthetic vs real need different call sites)
# ---------------------------------------------------------------------------

def _run_cascade_synthetic(pg, sg, wg, cj, prompt, label,
                            confident_high, disagreement_threshold=0.35,
                            decision_threshold=0.5):
    arms_called, trace, total_cost, total_latency = [], [], 0.0, 0.0

    pg_r = pg.score_with_label(prompt, label)
    arms_called.append(pg.name)
    total_cost += pg_r.cost
    total_latency += pg_r.latency_ms
    trace.append({"arm": pg.name, "score": pg_r.score})

    if pg_r.score >= confident_high:
        return CascadeDecision(final_score=pg_r.score, is_harmful=True,
                               arms_called=arms_called, total_cost=total_cost,
                               total_latency_ms=total_latency, trace=trace)

    sg_r = sg.score_with_label(prompt, label)
    wg_r = wg.score_with_label(prompt, label)
    arms_called.extend([sg.name, wg.name])
    total_cost += sg_r.cost + wg_r.cost
    total_latency += sg_r.latency_ms + wg_r.latency_ms
    trace.append({"arm": sg.name, "score": sg_r.score})
    trace.append({"arm": wg.name, "score": wg_r.score})

    mid_avg = (sg_r.score + wg_r.score) / 2.0
    disagreement = abs(sg_r.score - wg_r.score)

    if disagreement < disagreement_threshold:
        return CascadeDecision(final_score=mid_avg,
                               is_harmful=mid_avg >= decision_threshold,
                               arms_called=arms_called, total_cost=total_cost,
                               total_latency_ms=total_latency, trace=trace)

    cj_r = cj.score_with_label(prompt, label)
    arms_called.append(cj.name)
    total_cost += cj_r.cost
    total_latency += cj_r.latency_ms
    trace.append({"arm": cj.name, "score": cj_r.score})

    final_score = (mid_avg + cj_r.score) / 2.0
    return CascadeDecision(final_score=final_score,
                           is_harmful=final_score >= decision_threshold,
                           arms_called=arms_called, total_cost=total_cost,
                           total_latency_ms=total_latency, trace=trace)


def run_fixed_synthetic(pg, sg, wg, cj, prompts, labels, confident_high=0.75):
    return [
        _run_cascade_synthetic(pg, sg, wg, cj, p, l, confident_high)
        for p, l in zip(prompts, labels)
    ]


def run_bandit_synthetic(pg, sg, wg, cj, prompts, labels, gamma=0.1, seed=42):
    rng = np.random.default_rng(seed)
    K = len(STAGE1_THRESHOLDS)
    weights = np.ones(K, dtype=float)
    always_all_cost = pg.cost + sg.cost + wg.cost + cj.cost

    decisions = []
    weight_history = []

    def arm_probs(w):
        return (1 - gamma) * w / w.sum() + gamma / K

    for prompt, label in zip(prompts, labels):
        probs = arm_probs(weights)
        arm_idx = int(rng.choice(K, p=probs))
        confident_high = STAGE1_THRESHOLDS[arm_idx]

        decision = _run_cascade_synthetic(
            pg, sg, wg, cj, prompt, label, confident_high
        )
        decisions.append(decision)

        correct = decision.is_harmful == bool(label)
        cost_fraction = decision.total_cost / always_all_cost
        reward = (0.5 + 0.5 * (1.0 - cost_fraction)) if correct else 0.0

        estimated = reward / probs[arm_idx]
        weights[arm_idx] *= np.exp(gamma * estimated / K)
        weights /= weights.sum()
        weights *= K
        weight_history.append(weights.copy())

    return decisions, weights, weight_history


def run_fixed_real(pg, sg, wg, cj, prompts, labels):
    alloc = CascadeAllocator(pg, sg, wg, cj, confident_low=-1.0,
                             confident_high=0.75, disagreement_threshold=0.35,
                             decision_threshold=0.5)
    return [alloc.evaluate(p) for p in prompts]


def run_bandit_real(pg, sg, wg, cj, prompts, labels, gamma=0.1, seed=42):
    np.random.seed(seed)
    alloc = EXP3CascadeAllocator(pg, sg, wg, cj, gamma=gamma)
    decisions = []
    for prompt, label in zip(prompts, labels):
        d = alloc.evaluate(prompt)
        alloc.update(d, bool(label))
        decisions.append(d)
    return decisions, alloc.weights, [h["weights"] for h in alloc.history]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def rolling_metrics(decisions, labels, window=20):
    n = len(decisions)
    recalls, precisions, costs = [], [], []
    always_all_cost = 49.0

    for i in range(n):
        start = max(0, i - window + 1)
        chunk_d = decisions[start: i + 1]
        chunk_l = labels[start: i + 1]
        tp = sum(d.is_harmful and l == 1 for d, l in zip(chunk_d, chunk_l))
        fn = sum(not d.is_harmful and l == 1 for d, l in zip(chunk_d, chunk_l))
        fp = sum(d.is_harmful and l == 0 for d, l in zip(chunk_d, chunk_l))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        avg_cf = np.mean([d.total_cost / always_all_cost for d in chunk_d])
        recalls.append(recall)
        precisions.append(precision)
        costs.append(avg_cf)

    return recalls, precisions, costs


def final_metrics(decisions, labels):
    tp = sum(d.is_harmful and l == 1 for d, l in zip(decisions, labels))
    tn = sum(not d.is_harmful and l == 0 for d, l in zip(decisions, labels))
    fp = sum(d.is_harmful and l == 0 for d, l in zip(decisions, labels))
    fn = sum(not d.is_harmful and l == 1 for d, l in zip(decisions, labels))
    n = len(decisions)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    avg_cost = np.mean([d.total_cost for d in decisions])
    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "fpr": round(fpr, 3),
        "accuracy": round((tp + tn) / n, 3),
        "avg_cost": round(float(avg_cost), 2),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(fixed_d, bandit_d, weight_history, labels, out_path, synthetic):
    n = len(labels)
    window = max(10, n // 10)
    fr, fpr_curve, fc = rolling_metrics(fixed_d, labels, window)
    br, bpr_curve, bc = rolling_metrics(bandit_d, labels, window)

    wh = np.array(weight_history)
    thresh_labels = [f"{t:.2f}" for t in STAGE1_THRESHOLDS]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(fr, label="Fixed cascade", color="#d62728", lw=1.5)
    ax1.plot(br, label="EXP3 bandit", color="#2ca02c", lw=1.5)
    ax1.set_title(f"Rolling Recall (window={window})")
    ax1.set_xlabel("Prompt index")
    ax1.set_ylabel("Recall")
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fpr_curve, label="Fixed cascade", color="#d62728", lw=1.5)
    ax2.plot(bpr_curve, label="EXP3 bandit", color="#2ca02c", lw=1.5)
    ax2.set_title(f"Rolling Precision (window={window})")
    ax2.set_xlabel("Prompt index")
    ax2.set_ylabel("Precision")
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(fc, label="Fixed cascade", color="#d62728", lw=1.5)
    ax3.plot(bc, label="EXP3 bandit", color="#2ca02c", lw=1.5)
    ax3.set_title(f"Rolling Cost Fraction vs Always-All (window={window})")
    ax3.set_xlabel("Prompt index")
    ax3.set_ylabel("Avg cost / always-all cost")
    ax3.set_ylim(-0.05, 1.1)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    if wh.shape[0] > 0:
        for i, tl in enumerate(thresh_labels):
            ax4.plot(wh[:, i], label=f"t={tl}", lw=1.2)
        ax4.set_title("EXP3 Arm Weights Over Time\n(converges = bandit found best threshold)")
        ax4.set_xlabel("Prompt index")
        ax4.set_ylabel("Weight (higher = preferred)")
        ax4.legend(fontsize=8, ncol=2)
        ax4.grid(alpha=0.3)

    mode_tag = "(calibrated synthetic scores)" if synthetic else "(real models)"
    fig.suptitle(
        f"EXP3 Bandit vs Fixed-Threshold Cascade — JBB-Behaviors {mode_tag}",
        fontsize=13, fontweight="bold",
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", default=False,
                        help="Use calibrated synthetic scores (no GPU). "
                             "Distributions match real benchmark recall/FPR per arm.")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="EXP3 exploration rate (default 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.synthetic:
        print("SYNTHETIC MODE — scores sampled from Beta distributions fitted to "
              "real benchmark recall/FPR. No GPU needed. Results are illustrative.")
    else:
        print("REAL MODEL MODE — loading actual model weights. Needs GPU. "
              "This will take significant time.")

    print("Loading data...")
    prompts, labels = get_benchmark_data(seed=args.seed)
    print(f"  {len(prompts)} prompts ({sum(labels)} harmful, "
          f"{len(labels)-sum(labels)} benign)")

    if args.synthetic:
        pg, sg, wg, cj = build_synthetic_detectors()
        fixed_d = run_fixed_synthetic(pg, sg, wg, cj, prompts, labels)
        bandit_d, final_weights, wh = run_bandit_synthetic(
            pg, sg, wg, cj, prompts, labels, gamma=args.gamma, seed=args.seed
        )
    else:
        pg, sg, wg, cj = build_real_detectors()
        fixed_d = run_fixed_real(pg, sg, wg, cj, prompts, labels)
        bandit_d, final_weights, wh = run_bandit_real(
            pg, sg, wg, cj, prompts, labels, gamma=args.gamma, seed=args.seed
        )

    fixed_m = final_metrics(fixed_d, labels)
    bandit_m = final_metrics(bandit_d, labels)

    print("\n=== Final metrics ===")
    print(f"{'Metric':<15} {'Fixed':>10} {'EXP3 Bandit':>12}")
    print("-" * 40)
    for k in ["recall", "precision", "fpr", "accuracy", "avg_cost"]:
        print(f"{k:<15} {fixed_m[k]:>10} {bandit_m[k]:>12}")

    dom = STAGE1_THRESHOLDS[int(np.argmax(final_weights))]
    print(f"\nBandit converged to dominant threshold: {dom:.2f}")
    print("Final arm weights: " +
          ", ".join(f"{t:.2f}→{w:.2f}"
                    for t, w in zip(STAGE1_THRESHOLDS, final_weights)))

    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, "bandit_vs_fixed.csv")
    pd.DataFrame({
        "allocator": ["fixed_cascade", "exp3_bandit"],
        **{k: [fixed_m[k], bandit_m[k]] for k in fixed_m},
    }).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    png_path = os.path.join(RESULTS_DIR, "bandit_learning_curve.png")
    plot_results(fixed_d, bandit_d, wh, labels, png_path, synthetic=args.synthetic)

    if args.synthetic:
        print("\nNOTE: run without --synthetic on a GPU machine to get real-model "
              "numbers. The mechanism and learning curve shape are valid; "
              "exact metrics will differ.")


if __name__ == "__main__":
    main()
