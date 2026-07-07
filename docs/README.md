# Documentation index

## Reading order

| Order | Document | Audience |
|-------|----------|----------|
| 1 | [README](../README.md) | Everyone — overview, results, quick start |
| 2 | [Technical write-up](../writeup.md) | Researchers — full methodology and limitations |
| 3 | [Blog post draft](../blog_post.md) | General technical audience — narrative + findings |
| 4 | [Devpost submission](../devpost_description.md) | Hackathon judges |
| 5 | [Obsidian vault](../vault/Home.md) | Deep dive — linked findings and system design |

## Deployment

- [Deploy guide](../deploy/README.md) — Hugging Face Spaces, self-hosted GPU, Modal

## Scripts reference

| Script | Purpose |
|--------|---------|
| `scripts/benchmark.py` | Main JBB-Behaviors evaluation |
| `scripts/disagreement_sweep.py` | Precision–recall vs disagreement threshold |
| `scripts/bootstrap_ci_eval.py` | Bootstrap confidence intervals |
| `scripts/bandit_benchmark.py` | EXP3 vs fixed cascade |
| `scripts/pair_attacker.py` | PAIR red-team (`--attacker claude\|hf\|synthetic`) |
| `scripts/xstest_eval.py` | Over-refusal audit |
| `scripts/precision_fix_eval.py` | Low-confidence escalation sweep |

## Results (canonical)

Real-model runs are the source of truth in `results/`:

- `results_table.csv` — main benchmark
- `pair_attack_results.csv` — PAIR v2 (non-CBRN, LLM attacker)
- `precision_fix_sweep_real.csv` — real-model precision sweep
- `disagreement_sweep_synthetic.csv` — threshold Pareto (re-run with `--real` on GPU)

Archived older runs live in `results/archive/`.
