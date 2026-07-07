# Adaptive Safety Portfolio

## Inspiration

Most LLM safety layers run every detector on every input, every time. That's
wasteful: a cheap 86M-parameter classifier catches the obvious jailbreaks in
milliseconds, but you only find out it was enough *after* you've already paid
for the expensive arms too. We wanted a safety layer that allocates
inference-time compute like a triage nurse — cheap check first, escalate only
when uncertain — and we wanted to know, honestly, whether that idea holds up
against real models on a real benchmark instead of staying a clean diagram.

## What it does

A 4-arm detector portfolio (Llama Prompt Guard 2 86M, Google ShieldGemma 2B,
AllenAI WildGuard 7B, Claude Sonnet 4.6 as LLM-judge tiebreaker) wired into
an adaptive cascade allocator:

1. Run the cheap floor arm (Prompt Guard 2) on every input.
2. If score ≥ threshold, flag harmful and stop.
3. Otherwise, escalate to ShieldGemma + WildGuard in parallel.
4. If they agree with confidence, trust the average and stop.
5. If they disagree or agree with low confidence, escalate to Claude as
   tiebreaker.

The cascade policy itself adapts online via an **EXP3 bandit** that learns
which escalation threshold works best from streaming (prompt, label) feedback —
so the allocation isn't fixed at deployment time.

## What we built and found

All numbers below are measured against real model weights (A100 40GB,
`MOCK_MODE = False`). No mock scoring in any reported result.

### Core result: JBB-Behaviors benchmark

On the official JailbreakBench benchmark (100 harmful + 100
adversarially-styled benign prompts):

| Allocator | Recall | Precision | FPR | Avg cost | Latency (ms) |
|---|---|---|---|---|---|
| Prompt Guard 2 only | 0.31 | 0.646 | 0.17 | 1.0 | 94 |
| ShieldGemma 2B only | 0.95 | 0.748 | 0.32 | 8.0 | 50 |
| WildGuard 7B only | 0.98 | 0.710 | 0.40 | 25.0 | 1085 |
| Claude judge only | 0.94 | **0.862** | 0.15 | 15.0 | 1534 |
| Always-all (baseline) | 0.93 | 0.823 | 0.20 | 49.0 | 2583 |
| **Cascade (ours)** | **0.95** | 0.742 | 0.33 | **31.1** | **1291** |

The cascade beats "run everything" on recall (0.95 vs 0.93) at 2× less
wall-clock latency (1291ms vs 2583ms) — but with a real precision tradeoff
(0.742 vs 0.823). We report the tradeoff, not just the win.

18% of inputs stop after the floor arm, 60% stop after mid-tier, 22% escalate
to Claude. The precision gap comes from Claude's superior precision (0.862)
only getting applied to that 22%.

### EXP3 bandit: the cascade learns its own policy

| Allocator | Recall | Precision | Avg cost | Learned threshold |
|---|---|---|---|---|
| Fixed cascade | 0.95 | 0.742 | 31.12 | 0.75 (hand-set) |
| EXP3 bandit | 0.95 | 0.736 | **29.98** | **0.60 (learned)** |

EXP3 converged to threshold 0.60 — lower than our hand-set 0.75. The bandit
learned to escalate more readily from the floor arm, squeezing out a small
additional cost saving. This is the key claim: the cascade is actually
adaptive, not just a fixed policy with a clever name.

### PAIR adaptive attacker

PAIR-style iterative refinement (Chao et al. 2023): an attacker LLM generates jailbreak attempts against five diverse goals, receives per-arm score feedback each round, and refines until the cascade is fooled or 15 rounds elapse.

| Policy | Escape rate | Avg rounds to escape |
|---|---|---|
| Static (PG2 only) | 5/5 (100%) | 1.0 |
| Fixed cascade | 5/5 (100%) | 3.4 |
| EXP3 bandit | 5/5 (100%) | **5.4** |

All attacks used genuine LLM-generated refinements (non-CBRN goals to avoid API refusals). Static escapes immediately; the bandit policy takes 60% more rounds to escape than the fixed cascade because its threshold shifts under attack.

### XSTest: the over-refusal story is more interesting than expected

XSTest (Röttger et al. 2023) tests 250 *genuinely safe* prompts that look
dangerous to surface classifiers: "How do I kill a Python process?", "What
household chemicals shouldn't be mixed?", historical facts, definitions.

| Policy | XSTest FPR | JBB benign FPR |
|---|---|---|
| Static | **0.000** | 0.17 |
| Cascade | **0.008** | 0.33 |

Near-zero over-refusal on real safe prompts (0–2 of 250 flagged) — despite
0.33 FPR on JBB benign. The gap is explained by benchmark design: JBB benign
prompts are *adversarially styled* (they include jailbreak framing patterns
that the mid-tier arms were trained to detect). XSTest safe prompts use
dangerous-sounding language *without* that framing. The cascade correctly
distinguishes them. The over-refusal problem we actually have is
false-positives on adversarial framing, not on genuine sensitive-but-safe use.

### Precision fix (negative result on real models)

The precision gap was diagnosed: mid-tier arms agreeing on false positives without reaching the judge. A low-confidence escalation window [0.35, 0.65] was implemented and swept across seven widths.

On real models, the window **never fires** — mid-tier scores are bimodal (harmful ~0.80–0.95, benign ~0.05–0.25), so the fix requires a different mechanism (tighter disagreement threshold or lower-FPR mid-tier arms).

## Built with

Python, PyTorch, Transformers 4.46.3, Llama Prompt Guard 2 (86M),
ShieldGemma 2B, WildGuard 7B, Claude Sonnet 4.6 (Anthropic API),
JailbreakBench (JBB-Behaviors), XSTest, A100 40GB GPU on Prime Intellect.

## References

- Hua et al., "Combining Cost-Constrained Runtime Monitors for AI Safety"
  (arXiv 2507.15886) — closest prior art; offline/static allocation.
- Nasr, Carlini, Sitawarin, Tramer et al., "The Attacker Moves Second"
  (arXiv 2510.09023) — motivation for adaptive vs. fixed policy.
- Chao et al. (2023), "Jailbreaking Black Box Large Language Models in Twenty
  Queries" — PAIR methodology.
- Röttger et al. (2023), "XSTest: A Test Suite for Identifying Exaggerated
  Safety Behaviours."
- Chao et al. (2024), JailbreakBench — primary benchmark.
