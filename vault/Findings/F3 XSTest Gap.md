# F3: XSTest FPR Gap

> Cascade FPR 0.008 on genuinely safe prompts vs 0.33 on JBB benign — explained by benchmark design, not over-refusal.

---

## Numbers

| | XSTest FPR | JBB benign FPR |
|---|---|---|
| Static (PG2) | 0.000 | 0.17 |
| Fixed cascade | **0.008** | 0.33 |
| EXP3 bandit | **0.008** | 0.33 |

## Explanation

**JBB benign prompts** are adversarially constructed — they contain jailbreak framing patterns ("for educational purposes", "hypothetically speaking", "I'm a researcher") that ShieldGemma and WildGuard were specifically trained to detect as harmful signals. They're *designed* to stress-test over-refusal.

**XSTest safe prompts** use dangerous-sounding language ("kill a Python process", "what chemicals shouldn't be mixed") but *without* jailbreak framing. The mid-tier arms score these low — they detect framing, not surface vocabulary.

The gap exists even for static PG2 (0% XSTest, 17% JBB benign) — proving the gap predates cascading and is a property of the benchmarks.

## Implication

The over-refusal problem to solve is: false positives on *adversarially-styled* benign inputs (JBB benign). Not false positives on genuinely safe dangerous-sounding language (XSTest). These require different solutions.

## Related

- [[XSTest]] — evaluation
- [[JBB Benchmark]] — comparison
- [[Rottger 2023 XSTest]] — benchmark reference
