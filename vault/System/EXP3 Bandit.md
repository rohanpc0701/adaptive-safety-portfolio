# EXP3 Bandit

**File:** `scripts/bandit_cascade.py`

Replaces the fixed `confident_high` threshold in Stage 1 of [[Cascade Allocator]] with an online bandit that learns which threshold performs best from streaming (prompt, label) feedback.

---

## Why EXP3 not UCB

Jailbreak prompt distributions are non-stationary — an adversary can shift them. EXP3 provides regret guarantees against an *adversarial* reward sequence without assuming stationarity. UCB assumes i.i.d. rewards. → [[Nasr 2024]]

## Arms (threshold values)

`[0.50, 0.60, 0.70, 0.75, 0.80, 0.90]` — 6 discrete values for `confident_high`.

## Reward function

$$r = \begin{cases} 0.5 + 0.5 \cdot (1 - \text{cost\_fraction}) & \text{correct decision} \\ 0 & \text{wrong decision} \end{cases}$$

Trades off correctness against compute efficiency. Cheap + correct = full reward.

## Update rule

```
weights[arm] *= exp(gamma * (reward / prob) / K)
renormalize
```

`gamma = 0.1`, importance-weighted to correct for exploration bias.

## Real-model result

| | Fixed cascade | EXP3 bandit |
|---|---|---|
| Recall | 0.95 | 0.95 |
| Precision | 0.742 | 0.736 |
| Avg cost | 31.12 | **29.98** |
| Dominant threshold | 0.75 (fixed) | **0.60 (learned)** |

→ [[F4 EXP3 Threshold]]

## Related

- [[Cascade Allocator]] — parent system
- [[JBB Benchmark]] — evaluation dataset
- [[PAIR Attacker]] — bandit harder to evade than fixed (→ [[F5 PAIR Bandit Harder]])
- [[Nasr 2024]] — motivation for adaptive vs fixed policy
