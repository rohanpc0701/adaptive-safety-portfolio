# F6: Mid-Tier Score Distributions Are Bimodal

> Low-confidence escalation window never fires on real models — scores are bimodal, not uncertain.

---

## What we expected

Synthetic calibrated detectors (Beta distributions) produce smooth score distributions with meaningful probability mass in the uncertain [0.35, 0.65] range when mid-tier arms agree. The [[Precision Fix]] would trigger there and send ambiguous inputs to Claude.

## What actually happened

With real models, `avg_cost = 31.12` and `pct_claude = 21.5%` are **identical** across all 7 window widths ([0.45,0.55] through [0.20,0.80]). The window never fires.

## Why

ShieldGemma and WildGuard score JBB prompts bimodally:
- Harmful prompts → both score **0.80–0.95**
- Benign prompts → both score **0.05–0.25**
- When they agree (disagreement < 0.35), their average is almost never in [0.35, 0.65]

The uncertain middle region has essentially no real-model probability mass.

## Implication for the precision gap

False positives happen when both arms simultaneously score a benign JBB prompt at ~0.80-0.90 — *high confidence agreement on a false positive*. A low-confidence window can't catch that. The real precision fix requires:
1. **Tighter disagreement threshold** — escalate to Claude even when arms agree (higher Claude utilization cost), or
2. **Better mid-tier arms** — lower individual FPR so simultaneous high-confidence false positives are rarer

## Broader implication

Synthetic calibration that preserves marginal recall/FPR does *not* preserve score distribution shape. Bimodality is lost. Sweep results on synthetic data are misleading for mechanism-dependent interventions.

## Related

- [[Precision Sweep]] — the sweep that revealed this
- [[Precision Fix]] — the mechanism that doesn't work
- [[F1 Compute Tradeoff]] — the precision gap being addressed
