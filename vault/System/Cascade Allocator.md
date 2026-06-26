# Cascade Allocator

**File:** `scripts/cascade.py`

Core mechanism. Runs [[Detector Arms]] in order, stops early when confident.

---

## Stages

```
Prompt
  │
  ▼
[Stage 1] Prompt Guard 2 (86M)
  ├─ score ≥ 0.75 → HARMFUL, stop
  └─ uncertain → escalate
       │
       ▼
[Stage 2] ShieldGemma 2B + WildGuard 7B (parallel)
  ├─ agree with confidence + outside low-conf window → stop, trust average
  └─ disagree OR low-confidence agree → escalate
       │
       ▼
[Stage 3] Claude Sonnet 4.6 (judge)
  └─ final score = avg(mid-tier avg, Claude score)
```

## Key thresholds

| Param | Default | Meaning |
|---|---|---|
| `confident_high` | 0.75 | PG2 score above which = flag harmful, stop |
| `confident_low` | -1.0 | (unreachable) PG2 score below which = benign, stop |
| `disagreement_threshold` | 0.35 | \|SG - WG\| above which = escalate to Claude |
| `low_conf_low / high` | 0.35 / 0.65 | mid-tier avg in this range = escalate to Claude |

`confident_low = -1.0` (unreachable) because real PG2 scores on harmful prompts with no jailbreak phrasing fall in the same near-zero band as genuine benign prompts. → [[F2 PG2 Blindspot]]

## Escalation breakdown (JBB-Behaviors, real models)

- 18% stop at Stage 1 (PG2 confident harmful)
- 60% stop at Stage 2 (mid-tier agree with confidence)
- 22% reach Stage 3 (Claude judge)

## Related

- [[EXP3 Bandit]] — adaptive version of Stage 1 threshold
- [[Precision Fix]] — proposed extension to Stage 2 exit condition
- [[JBB Benchmark]] — primary evaluation
- [[F1 Compute Tradeoff]] — the main result
