"""
One-off diagnostic: print PromptGuard's real score distribution, split by
true label, so cascade.py's confident_low/confident_high thresholds can be
retuned against real model behavior instead of mock-tuned guesses.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from detectors import PromptGuardDetector
from load_data import get_benchmark_data

prompts, labels = get_benchmark_data()
pg = PromptGuardDetector()

harmful_scores = []
benign_scores = []
for p, l in zip(prompts, labels):
    r = pg.score(p)
    if l == 1:
        harmful_scores.append(r.score)
    else:
        benign_scores.append(r.score)

print("PromptGuard scores on HARMFUL prompts (sorted):")
print([round(s, 3) for s in sorted(harmful_scores)])
print(f"\nPromptGuard scores on BENIGN prompts (sorted):")
print([round(s, 3) for s in sorted(benign_scores)])

mid_harmful = [s for s in harmful_scores if 0.05 < s < 0.95]
mid_benign = [s for s in benign_scores if 0.05 < s < 0.95]
print(f"\nHarmful scores NOT near 0 or 1 (would land in 'uncertain' band under wide thresholds): {len(mid_harmful)}/{len(harmful_scores)}")
print(f"Benign scores NOT near 0 or 1: {len(mid_benign)}/{len(benign_scores)}")
