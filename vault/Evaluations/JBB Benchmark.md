# JBB Benchmark

**Dataset:** `JailbreakBench/JBB-Behaviors` (HuggingFace)
**Script:** `scripts/benchmark.py`
**200 prompts:** 100 harmful + 100 adversarially-styled benign

Primary evaluation for recall / precision / cost / latency.

---

## Results (real models, A100 40GB)

| Allocator | Recall | Precision | FPR | Avg cost | Latency (ms) |
|---|---|---|---|---|---|
| PG2 only | 0.31 | 0.646 | 0.17 | 1.0 | 94 |
| ShieldGemma only | 0.95 | 0.748 | 0.32 | 8.0 | 50 |
| WildGuard only | 0.98 | 0.710 | 0.40 | 25.0 | 1085 |
| Claude only | 0.94 | **0.862** | 0.15 | 15.0 | 1534 |
| Always-all | 0.93 | 0.823 | 0.20 | 49.0 | 2583 |
| **Cascade** | **0.95** | 0.742 | 0.33 | **31.1** | **1291** |

## Escalation breakdown

- 18% stop at Stage 1
- 60% stop at Stage 2
- 22% reach Claude

## JBB benign prompts caveat

Benign prompts are *adversarially styled* — intentionally contain jailbreak framing patterns. This explains why cascade FPR (0.33) is much higher on JBB benign than on genuinely safe prompts. See [[XSTest]] for comparison.

## Related

- [[Cascade Allocator]] — system under evaluation
- [[F1 Compute Tradeoff]] — headline finding
- [[F2 PG2 Blindspot]] — PG2 0.31 recall explained
- [[F3 XSTest Gap]] — JBB benign vs XSTest FPR comparison
