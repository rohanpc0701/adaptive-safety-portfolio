"""
demo_server.py

Backend for demo/index.html: loads the real 4-arm portfolio once at
startup and exposes:
  GET  /health          - liveness check
  POST /evaluate         - {prompt, policy: "static"|"cascade"} -> decision
                            (policy defaults to "cascade" for backward
                            compatibility with the single-page demo, which
                            only ever sends {prompt})
  POST /evaluate_trace    - {prompt} -> same shape as /evaluate's cascade
                             path, with an added cost_pct_of_always_all field

All three run the real CascadeAllocator / SingleArmAllocator
(scripts/cascade.py, validated thresholds) against real model weights -
nothing here is mocked.

Run: python3 scripts/demo_server.py
Then tunnel port 8000 to your local machine and open demo/index.html.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from detectors import build_portfolio
from cascade import CascadeAllocator, SingleArmAllocator

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading real detector portfolio (this happens once at startup)...")
portfolio = build_portfolio(use_real_claude=True)
cascade = CascadeAllocator(
    portfolio["prompt_guard"], portfolio["shieldgemma"],
    portfolio["wildguard"], portfolio["claude_judge"],
)
static_policy = SingleArmAllocator(portfolio["prompt_guard"])
print("Portfolio loaded. Ready.")

ARM_LABELS = {
    "prompt_guard_2_86m": "Prompt Guard 2 (86M)",
    "shieldgemma_2b": "ShieldGemma 2B",
    "wildguard_7b": "WildGuard 7B",
    "claude_judge": "Claude (judge)",
}

ALWAYS_ALL_COST = 1.0 + 8.0 + 25.0 + 15.0


class EvaluateRequest(BaseModel):
    prompt: str
    policy: str = "cascade"


def _decision_to_dict(decision):
    trace = [
        {"name": ARM_LABELS.get(t["arm"], t["arm"]), "score": t["score"]}
        for t in decision.trace
    ]
    return {
        "trace": trace,
        "final_score": decision.final_score,
        "is_harmful": decision.is_harmful,
        "arms_called": len(decision.arms_called),
        "total_cost": decision.total_cost,
        "always_all_cost": ALWAYS_ALL_COST,
        "cost_pct_of_always_all": decision.total_cost / ALWAYS_ALL_COST,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    allocator = static_policy if req.policy == "static" else cascade
    decision = allocator.evaluate(req.prompt)
    return _decision_to_dict(decision)


@app.post("/evaluate_trace")
def evaluate_trace(req: EvaluateRequest):
    decision = cascade.evaluate(req.prompt)
    return _decision_to_dict(decision)
