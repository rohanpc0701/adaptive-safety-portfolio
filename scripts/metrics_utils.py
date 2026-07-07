"""Shared evaluation metrics and bootstrap confidence intervals."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def compute_metrics(
    predictions: Sequence[int],
    labels: Sequence[int],
    costs: Sequence[float] | None = None,
    latencies: Sequence[float] | None = None,
) -> dict:
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

    judge_name = "claude_judge"
    pct_judge = 0.0
    if costs is not None:
        # Caller may pass arms_called counts via a separate path; default 0.
        pass

    return {
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "accuracy": accuracy,
        "avg_cost": avg_cost,
        "avg_latency_ms": avg_latency,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "pct_judge": pct_judge,
    }


def metrics_from_decisions(decisions, labels: Sequence[int]) -> dict:
    predictions = [1 if d.is_harmful else 0 for d in decisions]
    costs = [d.total_cost for d in decisions]
    latencies = [d.total_latency_ms for d in decisions]
    m = compute_metrics(predictions, labels, costs, latencies)
    m["pct_judge"] = sum(
        1 for d in decisions if any("claude" in a for a in d.arms_called)
    ) / len(decisions)
    return m


def bootstrap_ci(
    predictions: Sequence[int],
    labels: Sequence[int],
    metric: str = "recall",
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap CI for recall, precision, or fpr.
    Returns (point_estimate, lower, upper).
    """
    preds = np.asarray(predictions, dtype=int)
    labs = np.asarray(labels, dtype=int)
    n = len(labs)
    if n == 0:
        return 0.0, 0.0, 0.0

    def _metric(p, l):
        m = compute_metrics(p.tolist(), l.tolist())
        return m[metric]

    point = _metric(preds, labs)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples.append(_metric(preds[idx], labs[idx]))

    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)
