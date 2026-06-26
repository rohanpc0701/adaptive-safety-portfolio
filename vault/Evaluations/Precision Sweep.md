# Precision Sweep

**Script:** `scripts/precision_fix_eval.py`

Sweeps `low_conf_low / low_conf_high` window widths on [[JBB Benchmark]] to measure the precision tradeoff from the [[Precision Fix]].

---

## Real-model result

Window never fires across 7 widths (0.0 to 0.6). Cost and Claude utilization constant at 31.12 / 21.5%.

→ [[F6 Bimodal Scores]]

## Synthetic result (earlier run, calibrated detectors)

| Window | Precision | Claude % | Avg cost |
|---|---|---|---|
| disabled | 0.877 | 16.5% | 35.2 |
| [0.35, 0.65] | 0.935 | 29.5% | 38.3 |
| [0.25, 0.75] | 0.962 | 41.0% | 39.0 |

Synthetic detectors use Beta distributions calibrated to real recall/FPR but with smoother distributions — more mass in the uncertain [0.35, 0.65] range. Real models are bimodal; synthetic are not.

## What the discrepancy reveals

Synthetic calibration preserves marginal recall/FPR but not the *shape* of score distributions. Bimodality is a property that synthetic calibration doesn't capture. → [[F6 Bimodal Scores]]

## Related

- [[Precision Fix]] — the mechanism being evaluated
- [[F6 Bimodal Scores]] — why the window doesn't fire on real models
