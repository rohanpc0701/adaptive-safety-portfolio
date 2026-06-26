# F1: Compute Tradeoff

> Cascade achieves higher recall than always-all at lower compute — with a real precision cost.

---

## Numbers

| | Cascade | Always-all |
|---|---|---|
| Recall | **0.95** | 0.93 |
| Precision | 0.742 | **0.823** |
| FPR | 0.33 | 0.20 |
| Avg cost (proxy) | **31.1** | 49.0 |
| Latency (ms) | **1291** | 2583 |

- **1.6×** less compute (proxy cost)
- **2.0×** less wall-clock latency
- Precision gap: −0.081

## Why the precision gap exists

Claude's superior precision (0.862 solo) only applies to the 22% of inputs that reach Stage 3 (disagreement between ShieldGemma and WildGuard). When mid-tier arms agree on a false positive, Claude is never called. → [[Precision Fix]] (proposed solution that doesn't work on real models → [[F6 Bimodal Scores]])

## Honest framing

The precision gap is real and worth reporting. "Always-all" includes Claude on every input; cascade only uses Claude when needed. The tradeoff is compute vs precision, not compute vs recall.

## Related

- [[Cascade Allocator]] — the mechanism
- [[JBB Benchmark]] — evaluation
- [[F6 Bimodal Scores]] — why precision fix doesn't work
- [[Hua 2024]] — prior work on cost-constrained monitor combination
