"""
Public demo server for Hugging Face Spaces and local hosting.

DEMO_MODE=synthetic (default): calibrated Beta-score detectors, runs on CPU.
DEMO_MODE=real: full model weights (GPU + ANTHROPIC_API_KEY required).

Serves demo/index.html and FastAPI /evaluate endpoints on port 7860.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DEMO_MODE = os.environ.get("DEMO_MODE", "synthetic").lower()

if DEMO_MODE == "synthetic":
    from bandit_benchmark import build_synthetic_detectors
    from cascade import CascadeAllocator, SingleArmAllocator, CascadeDecision
    from bandit_benchmark import _run_cascade_synthetic

    pg, sg, wg, cj = build_synthetic_detectors()

    class _SyntheticCascade:
        def evaluate(self, prompt: str) -> CascadeDecision:
            return _run_cascade_synthetic(pg, sg, wg, cj, prompt, 1, 0.75, 0.35, 0.5)

    class _SyntheticStatic:
        def evaluate(self, prompt: str) -> CascadeDecision:
            r = pg.score_with_label(prompt, 1)
            return CascadeDecision(
                final_score=r.score, is_harmful=r.score >= 0.5,
                arms_called=[pg.name], total_cost=r.cost,
                total_latency_ms=r.latency_ms,
                trace=[{"arm": pg.name, "score": r.score}],
            )

    cascade = _SyntheticCascade()
    static_policy = _SyntheticStatic()
    print("Demo loaded in SYNTHETIC mode (CPU, calibrated scores).")
else:
    from detectors import build_portfolio
    from cascade import CascadeAllocator, SingleArmAllocator

    portfolio = build_portfolio(use_real_claude=True)
    cascade = CascadeAllocator(
        portfolio["prompt_guard"], portfolio["shieldgemma"],
        portfolio["wildguard"], portfolio["claude_judge"],
    )
    static_policy = SingleArmAllocator(portfolio["prompt_guard"])
    print("Demo loaded in REAL mode (GPU models).")

ARM_LABELS = {
    "prompt_guard_2_86m": "Prompt Guard 2 (86M)",
    "shieldgemma_2b": "ShieldGemma 2B",
    "wildguard_7b": "WildGuard 7B",
    "claude_judge": "Claude (judge)",
}
ALWAYS_ALL_COST = 49.0

app = FastAPI(title="Adaptive Safety Portfolio")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEMO_DIR = ROOT / "demo"


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
        "demo_mode": DEMO_MODE,
    }


@app.get("/health")
def health():
    return {"status": "ok", "demo_mode": DEMO_MODE}


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    alloc = static_policy if req.policy == "static" else cascade
    return _decision_to_dict(alloc.evaluate(req.prompt))


@app.post("/evaluate_trace")
def evaluate_trace(req: EvaluateRequest):
    return _decision_to_dict(cascade.evaluate(req.prompt))


@app.get("/")
def index():
    return FileResponse(DEMO_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
