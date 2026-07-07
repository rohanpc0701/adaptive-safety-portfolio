# Adaptive Safety Portfolio

**Inference-time compute allocation for LLM jailbreak detection**

[Technical write-up](writeup.md) · [Blog draft](blog_post.md) · [GitHub](https://github.com/rohanpc0701/adaptive-safety-portfolio)

**Rohan Chavan** · Inference-Time Compute Hackathon 2026

Most LLM safety layers run every detector on every input. This project allocates compute like triage: a cheap classifier runs first; expensive models escalate only when the cheap arm is uncertain. Evaluated end-to-end on real model weights against JailbreakBench, XSTest, and a PAIR-style adaptive attacker.

---

## Highlights

| | Cascade | Always-all baseline |
|---|---|---|
| **Recall** | **0.95** | 0.93 |
| **Precision** | 0.74 | **0.82** |
| **Avg latency** | **1,291 ms** | 2,583 ms |

- **2× lower latency** than running all four detectors on every prompt
- **EXP3 bandit** learns escalation thresholds online (converged to 0.60 vs hand-set 0.75)
- **PAIR red-team:** adaptive policy takes **60% more rounds** to escape than a fixed cascade
- **XSTest FPR 0.008** on genuinely safe prompts (vs 0.33 on adversarially-styled JBB benign)
- Interactive **live demo** with FastAPI backend and arm-by-arm decision trace

All benchmark numbers below use real model weights (`MOCK_MODE = False` in `detectors.py`). Cost units in tables are hand-assigned size proxies; latency is measured wall-clock time.

---

## How it works

```
   Prompt
     │
     ▼
 ┌─────────────────────┐
 │ Prompt Guard 2 (86M) │  floor arm — runs on every input
 └─────────┬────────────┘
           │ confident harmful? ──► STOP
           │ uncertain
           ▼
 ┌─────────────────────┬─────────────────────┐
 │   ShieldGemma 2B     │    WildGuard 7B     │  mid-tier (parallel)
 └─────────┬───────────┴─────────┬───────────┘
           │ agree with confidence? ──► STOP
           │ disagree
           ▼
 ┌─────────────────────┐
 │   LLM judge          │  tiebreaker (Anthropic API)
 └─────────────────────┘
```

**Detector arms:** Llama Prompt Guard 2 (86M), Google ShieldGemma 2B, AllenAI WildGuard 7B, Claude Sonnet 4.6 as judge.

**Escalation breakdown (JBB-Behaviors):** 18% stop at stage 1 · 60% at stage 2 · 22% reach the judge.

An **EXP3 bandit** variant (`bandit_cascade.py`) treats the stage-1 threshold as a learnable arm, updating from streaming labeled feedback — appropriate when prompt distributions shift under attack.

---

## Results

### JailbreakBench (JBB-Behaviors)

200 prompts (100 harmful + 100 adversarially-styled benign), loaded from `JailbreakBench/JBB-Behaviors`.

| Allocator | Recall | Precision | FPR | Avg cost† | Latency (ms) |
|---|---|---|---|---|---|
| Prompt Guard 2 only | 0.31 | 0.65 | 0.17 | 1.0 | 94 |
| ShieldGemma 2B only | 0.95 | 0.75 | 0.32 | 8.0 | 50 |
| WildGuard 7B only | 0.98 | 0.71 | 0.40 | 25.0 | 1,085 |
| LLM judge only | 0.94 | **0.86** | 0.15 | 15.0 | 1,534 |
| Always-all | 0.93 | 0.82 | 0.20 | 49.0 | 2,583 |
| **Cascade** | **0.95** | 0.74 | 0.33 | **31.1** | **1,291** |

†Cost units are relative proxies (parameter count / API tier), not measured FLOPs. See latency column for measured runtime.

![Recall vs compute cost](results/recall_vs_cost.png)

### EXP3 bandit vs fixed cascade

| Allocator | Recall | Precision | Avg cost | Threshold |
|---|---|---|---|---|
| Fixed cascade | 0.95 | 0.74 | 31.1 | 0.75 (hand-set) |
| EXP3 bandit | 0.95 | 0.74 | **30.0** | **0.60 (learned)** |

### PAIR adaptive attacker

PAIR-style loop (Chao et al. 2023): attacker LLM iteratively refines jailbreaks using per-arm score feedback. Five non-CBRN goals, up to 15 rounds.

| Goal | Static | Fixed cascade | EXP3 bandit |
|---|---|---|---|
| Harassment | R1 | R3 | R6 |
| Cyberattack | R1 | R5 | R5 |
| Fraud | R1 | R2 | R6 |
| Disinformation | R1 | R4 | **R8** |
| Manipulation | R1 | R3 | R2 |
| **Avg rounds to escape** | **1.0** | **3.4** | **5.4** |

Static (PG2 only) escapes immediately — confirming the floor arm's recall blind spot. The bandit policy is harder to probe because its threshold shifts under attack.

### XSTest over-refusal

250 genuinely safe prompts (Röttger et al. 2023) — homonyms, figurative language, safe contexts for dangerous-sounding terms.

| Policy | XSTest FPR | JBB benign FPR |
|---|---|---|
| Static (PG2 only) | **0.000** | 0.17 |
| Cascade | **0.008** | 0.33 |
| EXP3 bandit | **0.008** | 0.32 |

Near-zero FPR on real safe language; higher FPR on JBB benign reflects adversarial framing in that benchmark, not fundamental over-refusal on sensitive-but-safe prompts.

---

## Findings

1. **Real tradeoff:** cascade wins on recall and compute; precision drops because the judge only runs on 22% of inputs (mid-tier disagreement).
2. **PG2 blind spot:** misses 69/100 harmful JBB prompts, including direct requests with no jailbreak phrasing. Mid-tier arms recover 64/69.
3. **Bimodal scores:** a proposed low-confidence escalation fix never fires on real models — mid-tier scores cluster at high or low, not in the uncertain middle.
4. **Adaptive > fixed under attack:** EXP3 bandit requires more PAIR rounds to escape than a fixed-threshold cascade.
5. **Benchmark design matters:** XSTest vs JBB benign FPR gap is explained by adversarial styling, not cascade architecture.

---

## Live demo

`demo/index.html` connects to a FastAPI backend wrapping the real cascade.

```bash
# GPU machine
cd scripts && uvicorn demo_server:app --host 0.0.0.0 --port 8000

# Local machine (SSH tunnel)
ssh -L 8765:localhost:8000 user@gpu-host
open demo/index.html
```

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `POST /evaluate` | `{prompt, policy?: "static"\|"cascade"}` → decision |
| `POST /evaluate_trace` | Same + `cost_pct_of_always_all` and per-arm trace |

---

## Reproduce

```bash
pip install -r requirements.txt
pip install fastapi uvicorn sentencepiece tiktoken protobuf datasets
pip install transformers==4.46.3   # 5.x breaks WildGuard tokenizer

huggingface-cli login              # gated models: PG2, ShieldGemma, WildGuard
export ANTHROPIC_API_KEY=...
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

python3 scripts/benchmark.py              # main JBB benchmark
python3 scripts/bandit_benchmark.py       # EXP3 vs fixed
python3 scripts/pair_attacker.py          # PAIR red-team
python3 scripts/xstest_eval.py            # over-refusal audit
python3 scripts/precision_fix_eval.py     # precision sweep
```

Synthetic modes (`--synthetic`) are available for fast local runs without a GPU. Full real-model reproduction needs an A100-class GPU (~45 min total).

---

## Project structure

```
scripts/
  detectors.py           # 4-arm portfolio + unified scoring interface
  cascade.py             # fixed-threshold cascade allocator
  bandit_cascade.py      # EXP3 adaptive threshold learning
  benchmark.py           # JBB evaluation + recall/cost chart
  pair_attacker.py       # PAIR-style adaptive red-team
  xstest_eval.py         # over-refusal audit
  precision_fix_eval.py  # low-confidence escalation sweep
  demo_server.py         # FastAPI backend for live demo
demo/
  index.html             # 3-tab interactive dashboard
results/                 # benchmark CSVs and plots
writeup.md               # full technical report
vault/                   # Obsidian knowledge graph (findings, system design)
```

---

## References

- Hua et al., [Combining Cost-Constrained Runtime Monitors for AI Safety](https://arxiv.org/abs/2507.15886) (arXiv 2507.15886)
- Nasr et al., [The Attacker Moves Second](https://arxiv.org/abs/2510.09023) (arXiv 2510.09023)
- Chao et al., [Jailbreaking Black Box LLMs in Twenty Queries](https://arxiv.org/abs/2310.04451) (PAIR)
- Röttger et al., [XSTest](https://arxiv.org/abs/2308.01263)
- [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) (JBB-Behaviors)

---

## About

Built by **Rohan Chavan**. Prior work: 1st place, Amazon Trusted AI Challenge — adversarial jailbreak detection.
