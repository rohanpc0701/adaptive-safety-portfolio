# Chao et al. 2023 — PAIR

**"Jailbreaking Black Box Large Language Models in Twenty Queries"**
Chao, Robey, Dobriban, Hassani, Pappas, Wong

---

## Core idea

PAIR: Prompt Automatic Iterative Refinement. Attacker LLM iteratively generates jailbreak attempts, receives feedback (did target comply?), refines. Shows that ~20 queries is often enough to jailbreak SOTA LLMs.

## Adaptation in this project

Original PAIR targets a *generative* LLM — attacker wins when the model produces harmful content.

This project adapts PAIR to target a *cascade safety classifier* — attacker wins when the cascade returns `is_harmful=False`. Feedback is richer: attacker sees per-arm scores, not just binary caught/not.

## Related

- [[PAIR Attacker]] — our adaptation
- [[F5 PAIR Bandit Harder]] — result
