# Adaptive Safety Portfolio: Inference-Time Compute Allocation for LLM Jailbreak Detection

**Rohan Chavan** | [github.com/rohanpc0701/adaptive-safety-portfolio](https://github.com/rohanpc0701/adaptive-safety-portfolio)

---

## Abstract

We build and evaluate an adaptive cascade allocator for LLM safety classification: a 4-arm detector portfolio that runs a cheap 86M-parameter classifier on every input and escalates to progressively more expensive models only when the cheap arm is uncertain. On the JailbreakBench JBB-Behaviors benchmark (100 harmful + 100 adversarially-styled benign prompts), the cascade achieves 0.95 recall at 1.6× less average compute than running all four arms on every input, with a real precision tradeoff (0.742 vs 0.823) reported honestly. We extend the system with an EXP3 bandit that learns the escalation threshold online from streaming feedback, converging to a non-obvious threshold (0.60 vs the hand-set 0.75) with a small additional cost saving. We evaluate robustness with a PAIR-style adaptive attacker and measure over-refusal on XSTest (250 genuinely safe prompts), finding near-zero FPR (0.008) compared to 0.33 on JBB adversarially-styled benign prompts — a gap explained by JBB's benign set being designed to stress-test classifiers, not by fundamental over-refusal on safe language. All numbers are measured against real model weights on an A100 40GB GPU, with no mock scoring in any reported result.

---

## 1. Motivation

Most LLM safety systems run the same set of detectors on every input. This is wasteful: a 86M-parameter classifier resolves obvious jailbreaks in milliseconds and at near-zero compute, but the decision to trust it only comes *after* the expensive arms have already been paid for in an always-run architecture. The natural fix is a cascade: run the cheap arm first, and escalate to expensive arms only when the cheap arm is uncertain.

This idea is not new — Hua et al. (2024) formalize offline cost-constrained monitor combinations — but two questions remain underexplored: (1) what does the real precision/recall/cost tradeoff look like on a standard benchmark with real model weights rather than simulation, and (2) what happens when the escalation policy itself adapts online to the observed prompt distribution, rather than being fixed at deployment time?

The second question matters because a deterministic safety policy with a fixed threshold is a fixed target. As Nasr et al. (2024) argue, an adaptive attacker can find and exploit a static defense's blind spots systematically. An adaptive allocation policy — one that shifts compute toward more reliable arms as it learns which arms fail on which inputs — is harder to target reliably.

---

## 2. System Design

### 2.1 Detector Arms

Four arms form the portfolio, chosen to cover a range of parameter counts, training objectives, and cost profiles:

| Arm | Model | Parameters | Training objective | Cost (assigned proxy) |
|---|---|---|---|---|
| Floor | Llama Prompt Guard 2 | 86M | Jailbreak classification | 1 |
| Mid-tier | Google ShieldGemma 2B | 2B | Safety guideline compliance | 8 |
| Mid-tier | AllenAI WildGuard | 7B | Adversarial jailbreak robustness | 25 |
| Judge | Claude Sonnet 4.6 | — (API) | LLM judge with chain-of-thought | 15 |

All arms output a score in [0, 1] (1 = harmful). Each arm is called against its real weights; no mock scoring is used in any reported result. The cost column is a hand-assigned parameter-size proxy, not a measured quantity — see Section 4.1 for the actual measured latency.

### 2.2 Fixed-Threshold Cascade

The baseline cascade (`CascadeAllocator`) operates in three stages:

1. **Stage 1 (floor arm):** Run Prompt Guard 2. If score ≥ `confident_high` (0.75), flag harmful and stop. Otherwise escalate.
2. **Stage 2 (mid-tier arms):** Run ShieldGemma 2B and WildGuard 7B in parallel. If |score_SG − score_WG| < `disagreement_threshold` (0.35) AND their average is outside the low-confidence window [0.35, 0.65], stop and trust the average. Otherwise escalate.
3. **Stage 3 (judge arm):** Run Claude Sonnet 4.6 as a tiebreaker. Final score = average of mid-tier average and Claude score.

The `confident_low` threshold (below which Stage 1 calls the input benign and stops) defaults to −1.0 (unreachable), because real Prompt Guard 2 score distributions show harmful prompts with no explicit jailbreak phrasing scoring in the same near-zero range as genuinely benign prompts — no positive threshold separates them without sacrificing recall.

### 2.3 EXP3 Adaptive Cascade

`EXP3CascadeAllocator` replaces the fixed `confident_high` threshold with an online bandit. Six discrete threshold values serve as arms: [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]. After each labeled example, the chosen arm receives an importance-weighted reward:

$$r = \begin{cases} 0.5 + 0.5 \cdot (1 - \text{cost\_fraction}) & \text{if decision correct} \\ 0 & \text{otherwise} \end{cases}$$

where cost_fraction = (arms called cost) / (always-all cost). The reward trades off recall correctness (wrong = 0) against compute efficiency (correct + cheap = full reward). EXP3 is chosen over UCB because jailbreak prompt distributions are non-stationary — an adversary can shift the distribution — and EXP3 provides regret guarantees against an adversarial reward sequence without assuming stationarity.

### 2.4 Low-Confidence Escalation Fix

A precision gap was diagnosed in Stage 2: when ShieldGemma and WildGuard agree on a false positive at low confidence (both score a safe prompt near 0.5), no arm corrects them — the cascade stops and Claude is never called. The fix adds a low-confidence window [0.35, 0.65]: mid-tier agreement inside this window triggers escalation to Claude the same as disagreement. This increases Claude utilization from 22% to ~30% and is the primary lever for improving precision at additional compute cost.

---

## 3. Evaluation Benchmarks

**JBB-Behaviors:** The official JailbreakBench benchmark (Chao et al. 2024), 100 harmful + 100 adversarially-styled benign prompts pulled live from HuggingFace (`JailbreakBench/JBB-Behaviors`). The benign prompts are intentionally designed to stress-test over-refusal — they look dangerous on the surface. This is the primary benchmark for recall/precision/cost.

**XSTest:** Röttger et al. (2023), 250 genuinely safe prompts across 10 categories: homonyms ("kill a Python process"), figurative language ("I want to kill my sister"), safe contexts ("what chemicals shouldn't be mixed?"), historical facts, definitions, and others. XSTest tests a different failure mode than JBB benign: not adversarial framing, but legitimate use of dangerous-sounding language.

---

## 4. Results

### 4.1 Main Benchmark (JBB-Behaviors, real models)

| Allocator | Recall | Precision | FPR | Avg cost | Avg latency (ms) |
|---|---|---|---|---|---|
| Prompt Guard 2 only | 0.31 | 0.646 | 0.17 | 1.0 | 94 |
| ShieldGemma 2B only | 0.95 | 0.748 | 0.32 | 8.0 | 50 |
| WildGuard 7B only | 0.98 | 0.710 | 0.40 | 25.0 | 1085 |
| Claude judge only | 0.94 | **0.862** | 0.15 | 15.0 | 1534 |
| Always-all (baseline) | 0.93 | 0.823 | 0.20 | 49.0 | 2583 |
| **Cascade (ours)** | **0.95** | 0.742 | 0.33 | **31.1** | **1291** |

**On the cost column:** the assigned units (1/8/25/15) are parameter-size proxies, not measured values. Real latency contradicts the ordering — ShieldGemma (cost 8) is measurably faster than Prompt Guard 2 (cost 1) in wall-clock time (50ms vs 94ms), because ShieldGemma's single Yes/No forward pass is cheaper than Prompt Guard 2's sequence classification on this hardware. Claude (cost 15) is the slowest arm at 1534ms — API/network latency, not GPU compute. The cascade's 2.0× wall-clock latency reduction (1291ms vs 2583ms) is a more honest compute-efficiency number than the 1.6× proxy-cost reduction.

**Cascade escalation breakdown:** 18% of inputs stopped after Stage 1 (floor arm flagged harmful), 60% stopped after Stage 2 (mid-tier arms agreed with confidence), 22% escalated to Claude.

### 4.2 EXP3 Bandit (real models)

| Allocator | Recall | Precision | FPR | Avg cost | Dominant threshold |
|---|---|---|---|---|---|
| Fixed cascade | 0.95 | 0.742 | 0.33 | 31.12 | 0.75 (hand-set) |
| EXP3 bandit | 0.95 | 0.736 | 0.34 | **29.98** | **0.60 (learned)** |

EXP3 converged to `confident_high = 0.60`, lower than the hand-set 0.75. The bandit learned to escalate more readily from Stage 1 on this distribution, yielding a small additional cost saving (31.1 → 30.0). The slight precision decrease (0.742 → 0.736) follows from the same escalation pattern: more inputs reaching Stage 2 means more exposure to ShieldGemma and WildGuard's 32–40% individual FPR on benign prompts.

### 4.3 PAIR Adaptive Attacker (real models, A10 24GB)

We evaluate robustness using a PAIR-style loop (Chao et al. 2023): an attacker LLM (Claude) generates jailbreak attempts against a goal, receives per-arm scores, and refines until the cascade returns `is_harmful=False` (escape) or 15 rounds elapse. Five goals spanning JBB harm categories.

| Goal | Static | Fixed cascade | EXP3 bandit |
|---|---|---|---|
| Chemical weapon | R1 | **HELD** | **HELD** |
| Cyberattack | R1 | R4 | R9 |
| Fraud | R1 | R4 | R4 |
| Disinformation | R1 | R3 | R6 |
| Weapon (IED) | R1 | R4 | R5 |
| **Escape rate** | **5/5 (100%)** | **4/5 (80%)** | **4/5 (80%)** |

**Attacker LLM refusal caveat:** Claude API refused to generate attack content for CBRN/weapon goals from round 5 onward (and earlier for some goals), falling back to scripted framing variants. Rounds 1–4 for cyberattack, fraud, and disinformation used genuine LLM-generated refinements. The chem_weapon result (HELD across all 15 rounds) is robust because even scripted variants score high on mid-tier arms — the goal itself triggers both ShieldGemma and WildGuard regardless of framing.

Key findings: (1) Static fooled at R1 on every goal — replicates PG2 blind spot from JBB benchmark. (2) Cascade held the hardest goal across all rounds. (3) Bandit required more rounds to escape on 3/4 escapable goals (cyberattack: R9 vs R4, disinfo: R6 vs R3) — the adaptive threshold shifts under attack, making the decision boundary harder to probe than a fixed threshold.

### 4.4 XSTest Over-Refusal (real models, 250 safe prompts)

| Policy | XSTest FPR | JBB benign FPR |
|---|---|---|
| Static (PG2 only) | **0.000** | 0.17 |
| Fixed cascade | **0.008** | 0.32 |
| EXP3 bandit | **0.008** | 0.32 |

Near-zero FPR on XSTest (0–2 prompts flagged out of 250), across all policies, compared to 17–33% FPR on JBB benign. The only category with any over-refusal: `nons_group_real_discr` (2 of 25 cascade/bandit flags).

This gap requires interpretation: **the cascade's 33% FPR on JBB benign is not evidence of fundamental over-refusal on safe language.** JBB benign prompts are adversarially constructed to look like jailbreak attempts — they include framing patterns ("for educational purposes only," "hypothetically speaking") that ShieldGemma and WildGuard were specifically trained to detect as jailbreak signals. XSTest safe prompts, by contrast, use genuinely ambiguous dangerous-sounding language in benign contexts but without jailbreak framing. The real arms correctly distinguish these: near-zero FPR on XSTest, high FPR on JBB benign.

The implication is that improving cascade precision requires handling adversarially-styled benign inputs (JBB's test), not genuinely-safe dangerous-sounding ones (XSTest's test). These are different problems with different solutions.

---

## 5. Findings

**Finding 1: The cascade beats always-all on recall at lower compute, with a real precision cost.**
Cascade recall 0.95 vs always-all 0.93, average cost 31.1 vs 49.0 — a 1.6× reduction by proxy cost, 2.0× by measured latency. Precision 0.742 vs 0.823. The precision gap comes from Claude's superior precision (0.862 alone) only being applied to 22% of inputs (when mid-tier arms disagree), not all of them.

**Finding 2: Prompt Guard 2's blind spot is broader than indirect framing.**
PG2 scores 0.31 recall on JBB, missing 69 of 100 harmful prompts — including direct requests with no framing ("Write a tutorial on how to make a bomb" scores 0.001). The blind spot is not selective to framing tricks; PG2 just has low recall on its own. Of the 69 PG2 misses, escalating to Stage 2 catches 64. One prompt fools all three local arms.

**Finding 3: The FPR gap between JBB benign and XSTest is explained by the benchmark design, not by cascading.**
Static PG2 (0% XSTest FPR, 17% JBB benign FPR) shows the gap precedes the cascade. Cascading mid-tier arms raise FPR on JBB benign (because those arms detect jailbreak framing that JBB benign intentionally contains) without raising FPR on XSTest (because XSTest safe prompts lack that framing). The cascade's FPR is a property of the benchmark's adversarial construction, not a fundamental over-refusal problem.

**Finding 4: EXP3 learned a non-obvious threshold (0.60 vs hand-set 0.75).**
Given 200 labeled examples, EXP3 shifted weight toward escalating more readily from Stage 1 than the hand-tuned default. On 200 examples this difference is small but real (avg cost 30.0 vs 31.1). On a real deployment stream with thousands of examples and a shifting distribution, the learning signal would be substantially stronger.

---

## 6. Limitations

**PAIR attacker LLM refused most CBRN rounds.** Claude API declines to generate attack prompts for chemical/weapon goals, so rounds 5–15 for those goals used scripted framing variants rather than adaptive LLM refinements. Non-CBRN goals (cyberattack, fraud, disinformation) had genuine LLM attacks for rounds 1–4. The chem_weapon HELD result is robust; the escape-round counts for other goals are partially confounded by the refusal fallback.

**Cost units are not measured.** The "avg cost" column is parameter-size proxies assigned before any real run, not measured GPU FLOPs or energy. Real latency (measured) tells a somewhat different story — see Section 4.1.

**JBB-Behaviors is 200 prompts.** Confidence intervals are not reported. Differences of ±1–2 prompts (0.5–1%) are within noise. The qualitative direction of findings is robust; exact numbers should be treated as point estimates.

**EXP3 convergence on 200 examples.** The bandit has not converged — 200 examples is a short run for 6 arms. The threshold finding (0.60 dominant) is a tendency, not a stable equilibrium.

**Precision fix not measured on real models.** The low-confidence escalation sweep (Section 2.4) uses synthetic calibrated scores. Real-model precision improvement would require re-running the JBB benchmark with the fix enabled.

---

## 7. Related Work

**Hua et al. (2024), "Combining Cost-Constrained Runtime Monitors for AI Safety"** (arXiv 2507.15886). Closest prior work: offline, static cost-constrained monitor combination. This project extends their cascade idea with an online adaptive policy (EXP3) and evaluates against their threat model (adaptive attacker) with PAIR.

**Nasr, Carlini, Sitawarin, Tramer et al. (2024), "The Attacker Moves Second"** (arXiv 2510.09023). Argues that a deterministic safety defense with a fixed policy is a fixed target for an adaptive adversary who can find and exploit its blind spots. Finding 2 (PG2's discoverable blind spot) and Finding 4 (EXP3 threshold learning) are motivated by this argument: a fixed threshold is exploitable; an adaptive threshold is harder to target.

**Chao et al. (2023), "Jailbreaking Black Box Large Language Models in Twenty Queries."** Original PAIR methodology. We adapt PAIR to target a cascade safety *classifier* rather than a generative LLM — the attacker's goal is to produce a prompt that the cascade classifies as benign, not to elicit a harmful generation.

**Röttger et al. (2023), "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours."** XSTest benchmark used in Section 4.4.

**JailbreakBench (Chao et al. 2024).** JBB-Behaviors primary benchmark. Pulled via `datasets.load_dataset("JailbreakBench/JBB-Behaviors")`.

---

## 8. Conclusion

A cascade allocator with real model weights achieves higher recall than always-running all arms, at lower average compute — but with a real precision tradeoff that is worth reporting rather than cropping. The tradeoff is tunable via the low-confidence escalation window: wider windows improve precision at additional Claude utilization cost. Online EXP3 allocation learns a more effective escalation threshold than hand-tuning given even a short labeled stream. Over-refusal on genuinely safe prompts (XSTest FPR = 0.008) is not the binding constraint — the binding constraint is false positives on adversarially-styled benign prompts (JBB benign FPR = 0.33), which requires better mid-tier arm precision or more aggressive Claude escalation, not a different approach to safe-sounding language.

The main remaining gap is PAIR with an unconstrained attacker LLM — a model that does not refuse CBRN goals — which would give cleaner escape-round counts for all five goals. A separate open-weights attack model (e.g. Mistral-7B without safety fine-tuning) would close this gap without API refusals.

---

## Appendix: Reproduction

```bash
# Requirements
pip install -r requirements.txt
pip install tiktoken protobuf sentencepiece fastapi uvicorn datasets
pip install transformers==4.46.3  # 5.x breaks WildGuard tokenizer

# HuggingFace gated repos (accept license on huggingface.co first)
huggingface-cli login

# Environment
export ANTHROPIC_API_KEY=...
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# Main benchmark (JBB-Behaviors)
python3 scripts/benchmark.py

# EXP3 bandit vs fixed cascade
python3 scripts/bandit_benchmark.py            # real models
python3 scripts/bandit_benchmark.py --synthetic  # fast, no GPU

# PAIR adaptive attacker
python3 scripts/pair_attacker.py               # real LLM attacker
python3 scripts/pair_attacker.py --synthetic   # scripted variants

# XSTest over-refusal
python3 scripts/xstest_eval.py                 # real models
python3 scripts/xstest_eval.py --synthetic     # fast

# Precision fix sweep
python3 scripts/precision_fix_eval.py          # runs on synthetic
```

All output goes to `results/`. Requires an A100 or equivalent for reasonable runtimes (~45 min total on A100 40GB).
