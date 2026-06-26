# Adaptive Safety Portfolio

Inference-time compute allocation for LLM jailbreak detection.

---

## System

- [[Cascade Allocator]] — core mechanism, 3-stage escalation
- [[EXP3 Bandit]] — online adaptive threshold learning
- [[Detector Arms]] — PG2, ShieldGemma, WildGuard, Claude judge
- [[Precision Fix]] — low-confidence escalation (proposed + tested)

## Evaluations

- [[JBB Benchmark]] — primary recall/precision/cost result
- [[XSTest]] — over-refusal on genuine safe prompts
- [[PAIR Attacker]] — adaptive adversarial robustness
- [[Precision Sweep]] — real-model sweep of low-conf window widths

## Findings

- [[F1 Compute Tradeoff]] — 1.6× less cost, higher recall, real precision gap
- [[F2 PG2 Blindspot]] — floor arm misses 69/100 harmful prompts
- [[F3 XSTest Gap]] — FPR gap explained by benchmark design not over-refusal
- [[F4 EXP3 Threshold]] — bandit learned 0.60 vs hand-set 0.75
- [[F5 PAIR Bandit Harder]] — bandit 60% harder to escape than fixed cascade
- [[F6 Bimodal Scores]] — precision fix never fires; scores are bimodal

## References

- [[Hua 2024]] — cost-constrained monitors (closest prior work)
- [[Nasr 2024]] — attacker moves second; fixed policy = fixed target
- [[Chao 2023 PAIR]] — PAIR methodology
- [[Rottger 2023 XSTest]] — XSTest benchmark
- [[JailbreakBench]] — JBB-Behaviors dataset
