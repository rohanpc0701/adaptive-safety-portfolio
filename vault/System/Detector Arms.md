# Detector Arms

Four arms in the portfolio. All run against real weights — `MOCK_MODE = False`.

---

| Arm | Model | Params | Cost proxy | Latency (measured) | Individual recall | Individual precision |
|---|---|---|---|---|---|---|
| Floor | Llama Prompt Guard 2 | 86M | 1 | 94ms | 0.31 | 0.646 |
| Mid-tier | Google ShieldGemma 2B | 2B | 8 | 50ms | 0.95 | 0.748 |
| Mid-tier | AllenAI WildGuard 7B | 7B | 25 | 1085ms | 0.98 | 0.710 |
| Judge | Claude Sonnet 4.6 | — (API) | 15 | 1534ms | 0.94 | 0.862 |

## Cost column caveat

Units are hand-assigned parameter-size proxies, not measured FLOPs or energy. ShieldGemma (cost 8) is measurably *faster* than PG2 (cost 1) — 50ms vs 94ms — because single Yes/No forward pass vs sequence classification. Claude (cost 15) is slowest at 1534ms due to API/network latency.

## HuggingFace model IDs

- PG2: `meta-llama/Llama-Prompt-Guard-2-86M`
- ShieldGemma: `google/shieldgemma-2b`
- WildGuard: `allenai/wildguard`
- Claude: `claude-sonnet-4-6` (Anthropic API)

## Critical dep

`transformers==4.46.3` — 5.x breaks WildGuard sentencepiece tokenizer (misdetected as tiktoken BPE).

## Related

- [[Cascade Allocator]] — how arms are called
- [[EXP3 Bandit]] — adapts PG2's Stage 1 threshold
- [[F2 PG2 Blindspot]] — PG2 recall = 0.31; why it's the floor not the only arm
