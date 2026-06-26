"""
bandit_cascade.py

EXP3-adaptive cascade allocator: learns the escalation threshold at stage 1
online, from streaming feedback, instead of fixing it statically.

WHY EXP3:
  Jailbreak detection is an adversarial setting — the prompt distribution isn't
  stationary, attackers adapt. EXP3 (Exponential-weight algorithm for
  Exploration and Exploitation) makes no assumptions about the reward
  distribution and is optimal against an adversary, unlike UCB (which assumes
  stochastic rewards).

WHAT THE BANDIT DECIDES:
  After running PromptGuard 2, the cascade must choose: stop here (call it
  harmful) or escalate to the mid-tier arms. This is controlled by
  `confident_high` — the score threshold above which PG2 is trusted.

  In the fixed cascade, this threshold never changes. Here, EXP3 treats each
  of K discrete threshold values as an arm and learns, from streaming
  (prompt, label) pairs, which threshold best balances recall against compute
  savings. Lower threshold = stop earlier = cheaper but more false negatives.
  Higher threshold = escalate more = better recall but less compute saving.

ARMS (discrete confident_high values):
  [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]

REWARD (per decision, after true label revealed):
  - Correct AND stopped early (saved compute):  +1.0
  - Correct AND escalated (less savings):       +0.5 + 0.5*(1 - cost_fraction)
  - Wrong decision (false positive or false negative):  0.0
  Recall errors (false negatives on harmful) are treated identically to
  false positives here, but the reward implicitly penalises them via 0 reward
  — a more refined version could weight FN > FP if the application demands it.

ONLINE vs BATCH:
  .evaluate(prompt) picks an arm stochastically, runs the cascade, returns a
  CascadeDecision. Call .update(decision, true_label) after each example to
  update the EXP3 weights. In a real deployment, labels may be delayed or
  absent — the design intentionally separates evaluate/update to support that.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from cascade import CascadeDecision
from detectors import BaseDetector


STAGE1_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]


class EXP3CascadeAllocator:
    """
    Online-adaptive cascade using EXP3 to learn the stage-1 escalation
    threshold. Everything downstream of stage 1 (mid-tier disagree → Claude)
    is fixed, identical to CascadeAllocator.
    """

    def __init__(
        self,
        prompt_guard: BaseDetector,
        shieldgemma: BaseDetector,
        wildguard: BaseDetector,
        claude_judge: BaseDetector,
        gamma: float = 0.1,
        decision_threshold: float = 0.5,
        disagreement_threshold: float = 0.35,
    ):
        self.prompt_guard = prompt_guard
        self.shieldgemma = shieldgemma
        self.wildguard = wildguard
        self.claude_judge = claude_judge
        self.decision_threshold = decision_threshold
        self.disagreement_threshold = disagreement_threshold

        self.gamma = gamma  # exploration rate: fraction of weight on uniform draw
        K = len(STAGE1_THRESHOLDS)
        self.weights = np.ones(K, dtype=float)

        # for compute-saving fraction in reward
        self._always_all_cost = (
            prompt_guard.cost + shieldgemma.cost + wildguard.cost + claude_judge.cost
        )

        # state set by evaluate(), consumed by update()
        self._pending_arm: Optional[int] = None
        self._pending_prob: Optional[float] = None

        # history for analysis / plotting
        self.history: List[dict] = []

    def _arm_probs(self) -> np.ndarray:
        K = len(STAGE1_THRESHOLDS)
        probs = (1 - self.gamma) * self.weights / self.weights.sum() + self.gamma / K
        return probs

    def evaluate(self, prompt: str) -> CascadeDecision:
        probs = self._arm_probs()
        arm_idx = int(np.random.choice(len(STAGE1_THRESHOLDS), p=probs))
        confident_high = STAGE1_THRESHOLDS[arm_idx]

        self._pending_arm = arm_idx
        self._pending_prob = float(probs[arm_idx])

        arms_called, trace, total_cost, total_latency = [], [], 0.0, 0.0

        pg = self.prompt_guard.score(prompt)
        arms_called.append(self.prompt_guard.name)
        total_cost += pg.cost
        total_latency += pg.latency_ms
        trace.append({"arm": self.prompt_guard.name, "score": pg.score,
                       "threshold_used": confident_high})

        if pg.score >= confident_high:
            return CascadeDecision(
                final_score=pg.score, is_harmful=True,
                arms_called=arms_called, total_cost=total_cost,
                total_latency_ms=total_latency, trace=trace,
            )

        sg = self.shieldgemma.score(prompt)
        wg = self.wildguard.score(prompt)
        arms_called.extend([self.shieldgemma.name, self.wildguard.name])
        total_cost += sg.cost + wg.cost
        total_latency += sg.latency_ms + wg.latency_ms
        trace.append({"arm": self.shieldgemma.name, "score": sg.score})
        trace.append({"arm": self.wildguard.name, "score": wg.score})

        mid_avg = (sg.score + wg.score) / 2.0
        disagreement = abs(sg.score - wg.score)

        if disagreement < self.disagreement_threshold:
            final_score = mid_avg
            return CascadeDecision(
                final_score=final_score,
                is_harmful=final_score >= self.decision_threshold,
                arms_called=arms_called, total_cost=total_cost,
                total_latency_ms=total_latency, trace=trace,
            )

        cj = self.claude_judge.score(prompt)
        arms_called.append(self.claude_judge.name)
        total_cost += cj.cost
        total_latency += cj.latency_ms
        trace.append({"arm": self.claude_judge.name, "score": cj.score})

        final_score = (mid_avg + cj.score) / 2.0
        return CascadeDecision(
            final_score=final_score,
            is_harmful=final_score >= self.decision_threshold,
            arms_called=arms_called, total_cost=total_cost,
            total_latency_ms=total_latency, trace=trace,
        )

    def update(self, decision: CascadeDecision, true_label: bool) -> None:
        """
        EXP3 importance-weighted update. Call immediately after evaluate()
        once the true label is known.
        """
        if self._pending_arm is None:
            raise RuntimeError("update() called without a preceding evaluate()")

        arm = self._pending_arm
        prob = self._pending_prob
        correct = decision.is_harmful == true_label
        cost_fraction = decision.total_cost / self._always_all_cost

        if correct:
            # reward in (0.5, 1.0]: full reward if stopped early + correct
            reward = 0.5 + 0.5 * (1.0 - cost_fraction)
        else:
            reward = 0.0  # wrong is wrong regardless of compute saved

        # EXP3 importance-weighted gradient step
        K = len(STAGE1_THRESHOLDS)
        estimated = reward / prob
        self.weights[arm] *= np.exp(self.gamma * estimated / K)

        # re-normalise to the same scale to prevent overflow
        self.weights /= self.weights.sum()
        self.weights *= K

        self.history.append({
            "arm_idx": arm,
            "threshold": STAGE1_THRESHOLDS[arm],
            "correct": correct,
            "reward": reward,
            "cost_fraction": cost_fraction,
            "weights": self.weights.copy(),
            "probs": self._arm_probs().copy(),
        })

        self._pending_arm = None
        self._pending_prob = None

    def dominant_threshold(self) -> float:
        """Current highest-weight arm's threshold — what the bandit has learned."""
        return STAGE1_THRESHOLDS[int(np.argmax(self.weights))]
