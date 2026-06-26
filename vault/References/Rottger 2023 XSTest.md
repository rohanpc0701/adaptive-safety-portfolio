# Röttger et al. 2023 — XSTest

**"XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models"**

---

## Core idea

250 safe prompts designed to trigger over-refusal in safety-tuned LLMs. Categories: homonyms, figurative language, historical events, safe contexts, etc. Tests whether models refuse benign requests because they *sound* dangerous.

## Relevance

Used in this project to distinguish two types of FPR:
- FPR on adversarially-styled benign prompts (JBB benign) → 0.33
- FPR on genuinely safe prompts (XSTest) → 0.008

The gap between these reveals that the cascade's over-refusal is a property of JBB's benchmark construction, not a fundamental problem with safe-sounding language. → [[F3 XSTest Gap]]

## Related

- [[XSTest]] — evaluation
- [[F3 XSTest Gap]] — key finding
- [[JBB Benchmark]] — comparison
