"""
cascade.py

The cascade allocator: decides which detector arms to run, per input, and
in what order, instead of running every arm on every input.

This is the core mechanism of the project. The story:
  - Run the cheapest arm first.
  - If it's confident (very low or very high score), stop there.
  - If it's uncertain, escalate to the next arm(s).
  - If two mid-tier arms disagree, escalate to the most expensive arm (Claude)
    as a tiebreaker.

This gets most of the recall of "always run everything" at a fraction of the
average compute cost, which is the chart that proves the idea works.
"""

from dataclasses import dataclass, field
from typing import List
from detectors import BaseDetector, DetectorResult


@dataclass
class CascadeDecision:
    final_score: float
    is_harmful: bool
    arms_called: List[str] = field(default_factory=list)
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    trace: List[dict] = field(default_factory=list)  # for the demo UI


class CascadeAllocator:
    """
    Fixed-threshold cascade. This is intentionally simple (no bandit/online
    learning) so it's finishable and verifiable in a few hours. The bandit
    version is the documented "natural next step", not something we claim
    to have built.
    """

    def __init__(
        self,
        prompt_guard: BaseDetector,
        shieldgemma: BaseDetector,
        wildguard: BaseDetector,
        claude_judge: BaseDetector,
        confident_low: float = -1.0,
        confident_high: float = 0.75,
        disagreement_threshold: float = 0.35,
        decision_threshold: float = 0.5,
    ):
        self.prompt_guard = prompt_guard
        self.shieldgemma = shieldgemma
        self.wildguard = wildguard
        self.claude_judge = claude_judge

        # Tunable thresholds — these define the whole policy.
        # confident_low defaults to -1.0 (unreachable) on purpose: on real
        # PromptGuard scores, several indirectly-framed harmful prompts
        # (fictional/hypothetical/translation-wrapped) score in the same
        # near-zero band as true benign prompts, so no positive threshold
        # can separate them without losing recall.
        self.confident_low = confident_low      # below this from the floor arm -> stop, call it benign
        self.confident_high = confident_high    # above this from the floor arm -> stop, call it harmful
        self.disagreement_threshold = disagreement_threshold  # how far apart mid-tier scores must be to escalate
        self.decision_threshold = decision_threshold  # final score -> binary decision

    def evaluate(self, prompt: str) -> CascadeDecision:
        arms_called = []
        trace = []
        total_cost = 0.0
        total_latency = 0.0

        # Stage 1: cheap floor arm, always runs.
        pg_result = self.prompt_guard.score(prompt)
        arms_called.append(self.prompt_guard.name)
        total_cost += pg_result.cost
        total_latency += pg_result.latency_ms
        trace.append({"arm": self.prompt_guard.name, "score": pg_result.score})

        if pg_result.score <= self.confident_low:
            return CascadeDecision(
                final_score=pg_result.score,
                is_harmful=False,
                arms_called=arms_called,
                total_cost=total_cost,
                total_latency_ms=total_latency,
                trace=trace,
            )
        if pg_result.score >= self.confident_high:
            return CascadeDecision(
                final_score=pg_result.score,
                is_harmful=True,
                arms_called=arms_called,
                total_cost=total_cost,
                total_latency_ms=total_latency,
                trace=trace,
            )

        # Stage 2: uncertain -> escalate to both mid-tier specialists.
        sg_result = self.shieldgemma.score(prompt)
        wg_result = self.wildguard.score(prompt)
        arms_called.extend([self.shieldgemma.name, self.wildguard.name])
        total_cost += sg_result.cost + wg_result.cost
        total_latency += sg_result.latency_ms + wg_result.latency_ms
        trace.append({"arm": self.shieldgemma.name, "score": sg_result.score})
        trace.append({"arm": self.wildguard.name, "score": wg_result.score})

        mid_tier_avg = (sg_result.score + wg_result.score) / 2.0
        disagreement = abs(sg_result.score - wg_result.score)

        if disagreement < self.disagreement_threshold:
            # They agree, trust the average, stop here.
            final_score = mid_tier_avg
            return CascadeDecision(
                final_score=final_score,
                is_harmful=final_score >= self.decision_threshold,
                arms_called=arms_called,
                total_cost=total_cost,
                total_latency_ms=total_latency,
                trace=trace,
            )

        # Stage 3: mid-tier arms disagree -> escalate to Claude as tiebreaker.
        cj_result = self.claude_judge.score(prompt)
        arms_called.append(self.claude_judge.name)
        total_cost += cj_result.cost
        total_latency += cj_result.latency_ms
        trace.append({"arm": self.claude_judge.name, "score": cj_result.score})

        final_score = (mid_tier_avg + cj_result.score) / 2.0
        return CascadeDecision(
            final_score=final_score,
            is_harmful=final_score >= self.decision_threshold,
            arms_called=arms_called,
            total_cost=total_cost,
            total_latency_ms=total_latency,
            trace=trace,
        )


class AlwaysAllAllocator:
    """Baseline: run every arm on every input, average the scores."""

    def __init__(self, prompt_guard, shieldgemma, wildguard, claude_judge, decision_threshold: float = 0.5):
        self.arms = [prompt_guard, shieldgemma, wildguard, claude_judge]
        self.decision_threshold = decision_threshold

    def evaluate(self, prompt: str) -> CascadeDecision:
        results = [arm.score(prompt) for arm in self.arms]
        final_score = sum(r.score for r in results) / len(results)
        total_cost = sum(r.cost for r in results)
        total_latency = sum(r.latency_ms for r in results)
        return CascadeDecision(
            final_score=final_score,
            is_harmful=final_score >= self.decision_threshold,
            arms_called=[arm.name for arm in self.arms],
            total_cost=total_cost,
            total_latency_ms=total_latency,
            trace=[{"arm": arm.name, "score": r.score} for arm, r in zip(self.arms, results)],
        )


class SingleArmAllocator:
    """Baseline: run exactly one arm, for comparison."""

    def __init__(self, arm: BaseDetector, decision_threshold: float = 0.5):
        self.arm = arm
        self.decision_threshold = decision_threshold

    def evaluate(self, prompt: str) -> CascadeDecision:
        result = self.arm.score(prompt)
        return CascadeDecision(
            final_score=result.score,
            is_harmful=result.score >= self.decision_threshold,
            arms_called=[self.arm.name],
            total_cost=result.cost,
            total_latency_ms=result.latency_ms,
            trace=[{"arm": self.arm.name, "score": result.score}],
        )
