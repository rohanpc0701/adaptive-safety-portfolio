# Precision Fix

**File:** `scripts/precision_fix_eval.py`

Proposed extension to Stage 2 exit condition in [[Cascade Allocator]] to close the precision gap.

---

## Hypothesis

Precision gap (cascade 0.742 vs always-all 0.823) caused by mid-tier arms agreeing on false positives at *low confidence* (both score ~0.5) — Claude never called to override.

**Fix:** add `low_conf_low=0.35 / low_conf_high=0.65`. If mid-tier agree but average falls in this window, escalate to Claude anyway.

## Real-model result

Window never fires. Cost and Claude utilization identical across all widths:

| Window | Precision | % Claude | Avg cost |
|---|---|---|---|
| disabled | 0.748 | 21.5% | 31.12 |
| [0.45, 0.55] | 0.742 | 21.5% | 31.12 |
| [0.35, 0.65] | 0.754 | 21.5% | 31.12 |
| [0.25, 0.75] | 0.760 | 21.5% | 31.12 |
| [0.20, 0.80] | 0.748 | 21.5% | 31.12 |

→ [[F6 Bimodal Scores]]

## What this means

Hypothesis was wrong. False positives happen at *high confidence*, not low confidence. Both ShieldGemma and WildGuard score adversarially-styled benign prompts at 0.80–0.95 simultaneously. Real fix: tighter `disagreement_threshold` (escalate more to Claude) or lower FPR mid-tier arms.

## Related

- [[Cascade Allocator]] — Stage 2 exit condition
- [[F6 Bimodal Scores]] — mechanism explaining why fix doesn't fire
- [[F1 Compute Tradeoff]] — precision gap being addressed
