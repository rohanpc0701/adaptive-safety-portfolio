"""
One-off: re-test CascadeAllocator with confident_low disabled (effectively
-1.0, unreachable) so the floor arm's "confidently benign" shortcut never
fires. Real PromptGuard scores show this shortcut is the cause of cascade's
real-world recall gap (see inspect_score_distribution.py finding: several
indirectly-framed harmful prompts score in the same near-zero band as true
benign prompts, so no positive threshold can separate them).

This quantifies the cost/recall tradeoff of disabling that shortcut,
instead of just reporting the broken number.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from detectors import build_portfolio
from cascade import CascadeAllocator
from load_data import get_benchmark_data
from benchmark import compute_metrics, run_allocator, print_escalation_breakdown

prompts, labels = get_benchmark_data()
portfolio = build_portfolio(use_real_claude=True)

cascade_retuned = CascadeAllocator(
    portfolio["prompt_guard"], portfolio["shieldgemma"],
    portfolio["wildguard"], portfolio["claude_judge"],
    confident_low=0.0002,  # only literal float-zero counts as "confidently benign"
    confident_high=0.75,
    disagreement_threshold=0.35,
)

metrics, decisions = run_allocator(cascade_retuned, prompts, labels, "cascade_no_floor_shortcut")
print(metrics)
print_escalation_breakdown(decisions, prompts)
