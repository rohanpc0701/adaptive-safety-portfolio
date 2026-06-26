# JailbreakBench

**JBB-Behaviors dataset**
Chao et al. 2024
HuggingFace: `JailbreakBench/JBB-Behaviors`

---

## Dataset

200 prompts: 100 harmful + 100 adversarially-styled benign.

Benign prompts are intentionally constructed to stress-test classifiers — they look like jailbreaks but ask for legitimate things. This is by design, not a flaw.

## Usage in this project

Primary benchmark for [[JBB Benchmark]]. Loaded via:
```python
datasets.load_dataset("JailbreakBench/JBB-Behaviors")
```

## Related

- [[JBB Benchmark]] — evaluation
- [[F3 XSTest Gap]] — why JBB benign FPR is high (adversarial construction)
