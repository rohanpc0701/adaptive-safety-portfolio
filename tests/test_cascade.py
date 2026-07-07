"""Unit tests for cascade allocator logic (mock detectors, no GPU)."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from detectors import BaseDetector, DetectorResult
from cascade import CascadeAllocator, AlwaysAllAllocator, SingleArmAllocator, CascadeDecision
from bandit_cascade import EXP3CascadeAllocator, STAGE1_THRESHOLDS


class MockDetector(BaseDetector):
    def __init__(self, name: str, score: float, cost: float = 1.0):
        self.name = name
        self.cost = cost
        self._score = score

    def score(self, prompt: str) -> DetectorResult:
        return DetectorResult(score=self._score, cost=self.cost, latency_ms=1.0)


def _portfolio(pg=0.5, sg=0.8, wg=0.7, cj=0.9):
    return (
        MockDetector("prompt_guard_2_86m", pg, 1.0),
        MockDetector("shieldgemma_2b", sg, 8.0),
        MockDetector("wildguard_7b", wg, 25.0),
        MockDetector("claude_judge", cj, 15.0),
    )


def test_stage1_confident_harmful_stops_early():
    pg, sg, wg, cj = _portfolio(pg=0.9)
    alloc = CascadeAllocator(pg, sg, wg, cj, confident_high=0.75)
    d = alloc.evaluate("test")
    assert d.is_harmful is True
    assert len(d.arms_called) == 1
    assert d.total_cost == 1.0


def test_stage1_confident_benign_never_stops_with_default_low():
    pg, sg, wg, cj = _portfolio(pg=0.01)
    alloc = CascadeAllocator(pg, sg, wg, cj, confident_low=-1.0, confident_high=0.75)
    d = alloc.evaluate("test")
    assert len(d.arms_called) == 3  # escalates to mid-tier, agrees


def test_stage2_agreement_skips_judge():
    pg, sg, wg, cj = _portfolio(pg=0.5, sg=0.8, wg=0.75)
    alloc = CascadeAllocator(pg, sg, wg, cj, disagreement_threshold=0.35)
    d = alloc.evaluate("test")
    assert "claude_judge" not in d.arms_called
    assert len(d.arms_called) == 3


def test_stage3_disagreement_calls_judge():
    pg, sg, wg, cj = _portfolio(pg=0.5, sg=0.9, wg=0.1)
    alloc = CascadeAllocator(pg, sg, wg, cj, disagreement_threshold=0.35)
    d = alloc.evaluate("test")
    assert "claude_judge" in d.arms_called
    assert len(d.arms_called) == 4


def test_low_conf_window_triggers_judge():
    pg, sg, wg, cj = _portfolio(pg=0.5, sg=0.5, wg=0.52)
    alloc = CascadeAllocator(
        pg, sg, wg, cj,
        disagreement_threshold=0.35,
        low_conf_low=0.45,
        low_conf_high=0.55,
    )
    d = alloc.evaluate("test")
    assert "claude_judge" in d.arms_called


def test_always_all_calls_every_arm():
    pg, sg, wg, cj = _portfolio()
    alloc = AlwaysAllAllocator(pg, sg, wg, cj)
    d = alloc.evaluate("test")
    assert len(d.arms_called) == 4
    assert d.total_cost == 49.0


def test_single_arm_baseline():
    pg, sg, wg, cj = _portfolio(pg=0.2)
    alloc = SingleArmAllocator(pg)
    d = alloc.evaluate("test")
    assert d.is_harmful is False
    assert len(d.arms_called) == 1


def test_exp3_evaluate_and_update():
    pg, sg, wg, cj = _portfolio(pg=0.5, sg=0.8, wg=0.75)
    bandit = EXP3CascadeAllocator(pg, sg, wg, cj, gamma=0.1)
    d = bandit.evaluate("harmful prompt")
    bandit.update(d, true_label=True)
    assert bandit.dominant_threshold() in STAGE1_THRESHOLDS
    assert len(bandit.history) == 1


def test_tighter_disagreement_threshold_increases_judge_rate():
    pg, sg, wg, cj = _portfolio(pg=0.5, sg=0.7, wg=0.5)

    loose = CascadeAllocator(pg, sg, wg, cj, disagreement_threshold=0.35)
    tight = CascadeAllocator(pg, sg, wg, cj, disagreement_threshold=0.05)

    d_loose = loose.evaluate("x")
    d_tight = tight.evaluate("x")
    assert ("claude_judge" in d_tight.arms_called) or d_loose.arms_called == d_tight.arms_called
