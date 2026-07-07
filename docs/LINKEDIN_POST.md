# LinkedIn post (copy-paste ready)

**What if your LLM safety layer spent compute like a triage nurse?**

Most systems run every detector on every input. I built an adaptive cascade that runs a cheap 86M classifier first and escalates to heavier models only when uncertain.

**Results on JailbreakBench (real models, A100):**
→ 95% recall vs 93% for always-run-all
→ 2× lower latency (1.3s vs 2.6s per prompt)
→ Honest tradeoff: precision 0.74 vs 0.82

**What surprised me:**
1. Prompt Guard 2 misses direct harmful requests — not just clever framing ("make a bomb" scores 0.001)
2. Near-zero over-refusal on XSTest safe prompts (0.8% FPR) vs 33% on adversarially-styled JBB benign — benchmark design matters
3. EXP3 bandit learned threshold 0.60 vs my hand-tuned 0.75
4. Adaptive policy takes 60% more PAIR attack rounds to escape than fixed cascade

Open source, fully reproducible, interactive demo included.

🔗 https://github.com/rohanpc0701/adaptive-safety-portfolio

#MachineLearning #AISafety #LLM #MLOps #Python

---

**Optional first comment:** Live demo deploy guide in `deploy/README.md` — Hugging Face Spaces with CPU synthetic mode or GPU real models.
