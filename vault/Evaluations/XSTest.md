# XSTest

**Dataset:** `natolambert/xstest-v2-copy` (HuggingFace), `prompts` split
**Script:** `scripts/xstest_eval.py`
**250 genuinely safe prompts** across 10 categories

Tests over-refusal — false positives on prompts that *look* dangerous but aren't.

---

## Categories

homonyms · figurative_language · historical_events · definitions · safe_contexts · safe_targets · privacy_fictional · privacy_public · real_group_nons_discr · nons_group_real_discr

## Results (real models, A100 40GB)

| Policy | XSTest FPR | JBB benign FPR |
|---|---|---|
| Static (PG2) | **0.000** | 0.17 |
| Fixed cascade | **0.008** | 0.33 |
| EXP3 bandit | **0.008** | 0.33 |

Only 2/250 prompts flagged by cascade/bandit (category: `nons_group_real_discr`).

## Key insight

The gap between XSTest FPR (0.008) and JBB benign FPR (0.33) is explained by benchmark construction, not by the cascade having a fundamental over-refusal problem. JBB benign prompts contain jailbreak framing ("for educational purposes", "hypothetically speaking") that ShieldGemma/WildGuard were trained to detect. XSTest safe prompts use dangerous-sounding language *without* jailbreak framing.

→ [[F3 XSTest Gap]]

## Related

- [[JBB Benchmark]] — comparison baseline
- [[Cascade Allocator]] — system under evaluation
- [[F3 XSTest Gap]] — explanation of the FPR gap
- [[Rottger 2023 XSTest]] — benchmark reference
