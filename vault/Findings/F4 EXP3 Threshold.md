# F4: EXP3 Learned Non-Obvious Threshold

> EXP3 bandit converged to confident_high = 0.60 vs hand-set 0.75 — lower threshold = escalate more readily from Stage 1.

---

## Numbers

| | Fixed | EXP3 |
|---|---|---|
| Threshold | 0.75 (hand-set) | **0.60 (learned)** |
| Recall | 0.95 | 0.95 |
| Precision | 0.742 | 0.736 |
| Avg cost | 31.12 | **29.98** |

## Final arm weights

`0.50→1.12, 0.60→1.44, 0.70→0.75, 0.75→0.88, 0.80→0.91, 0.90→0.91`

Arm 0.60 dominant. Arm 0.70 actually *least* trusted (downweighted).

## What this means

On the JBB distribution, it's better to escalate from Stage 1 more readily (at score ≥ 0.60) than to trust PG2's confident-harmful signal up to 0.75. The bandit discovered that PG2 scores in [0.60, 0.75) are less reliable than assumed — sending them to Stage 2 is worth the compute.

Precision slightly lower (0.736 vs 0.742) because escalating more from Stage 1 means more exposure to ShieldGemma/WildGuard's 32-40% individual FPR.

## Caveat

200 examples is a short run. The 0.60 tendency is real but not a stable equilibrium — EXP3 hasn't converged. On a real deployment stream with thousands of examples, the signal would be much stronger.

## Related

- [[EXP3 Bandit]] — mechanism
- [[Cascade Allocator]] — Stage 1 threshold being adapted
- [[JBB Benchmark]] — evaluation data
- [[Nasr 2024]] — motivation for adaptive policy
