# Claude Code Handoff — Adaptive Safety Portfolio

Paste this entire prompt into Claude Code to get it running locally.

---

## Prompt to paste into Claude Code

```
I have a hackathon demo I need to run locally. Here's the full context:

## What it is
A three-tab dashboard called "Adaptive Safety Portfolio" that demos an adaptive 
safety cascade: routes prompts through up to 4 jailbreak detectors sequentially, 
escalating only when uncertain, saving compute while maintaining recall.

## Frontend
The frontend is already built — it's a single HTML file at `demo/index.html`. 
Open it directly in a browser (no build step needed). It talks to a local 
FastAPI backend.

## Backend API contract
The frontend expects these endpoints at http://localhost:8765 (or editable in-UI):

  GET  /health
  → 200 OK (just for connection status check)

  POST /evaluate
  Content-Type: application/json
  Body: { "prompt": "<string>", "policy": "static" | "cascade" }
  Response: {
    "final_score": 0.94,
    "is_harmful": true,
    "arms_called": ["Llama Prompt Guard 2 (86M)", "ShieldGemma 2B", "WildGuard 7B"]
  }

  POST /evaluate_trace
  Content-Type: application/json
  Body: { "prompt": "<string>" }
  Response: {
    "trace": [
      { "arm": "Llama Prompt Guard 2 (86M)", "score": 0.43 },
      { "arm": "ShieldGemma 2B",             "score": 0.87 },
      { "arm": "WildGuard 7B",               "score": 0.79 }
    ],
    "is_harmful": true,
    "arms_called": ["Llama Prompt Guard 2 (86M)", "ShieldGemma 2B", "WildGuard 7B"],
    "cost_pct_of_always_all": 0.63
  }
  CORS must be open: allow_origins=["*"]

## The four arms (in order)
1. Llama Prompt Guard 2 (86M)  — cheap floor, runs first
2. ShieldGemma 2B              — precision specialist, runs as pair with #3
3. WildGuard 7B                — adversarial generalist, runs as pair with #2
4. Claude (judge)              — tiebreaker, only called if #2 and #3 disagree

## Cascade logic
- Score thresholds: ≤ 0.15 = benign, ≥ 0.75 = harmful, between = uncertain
- Static policy: run arm 1 only, use its score as final
- Cascade policy:
  - Arm 1 confident (≤0.15 or ≥0.75) → stop, 1 arm called
  - Arm 1 uncertain → escalate to arms 2+3 simultaneously
  - Arms 2+3 agree (both sides of 0.5) → stop, 3 arms called
  - Arms 2+3 disagree → escalate to arm 4 (Claude), 4 arms called
  - Note: 2 arms is never the final count — mid-tier always runs as a pair

## What I need from you
Look at the existing `demo_server.py` (if it exists) and either:
1. Wire up the three endpoints above to the actual model weights, OR
2. If the models aren't loaded yet, create a working FastAPI server that 
   implements the cascade logic correctly — even with mock scores for now — 
   so the frontend UI is fully testable.

Make sure CORS is configured open. The frontend runs at file:// or localhost 
on a different port.

Also check `demo/index.html` is openable directly in a browser and the 
backend URL input (default http://localhost:8765) is pointed correctly.

Full design spec and component breakdown: `design_handoff_adaptive_safety_portfolio/README.md`
```

---

## What Claude Code will do

1. Inspect `demo_server.py` (your existing FastAPI server)
2. Add or wire the three endpoints (`/health`, `/evaluate`, `/evaluate_trace`) 
   to the actual model weights running on the pod
3. Confirm CORS is open
4. Verify `demo/index.html` opens in a browser and connects to the backend

## SSH tunnel reminder
The frontend hardcodes `http://localhost:8765`. You need the tunnel running:
```
ssh -L 8765:localhost:8000 ubuntu@<pod-ip>
```
Or change the URL in the top bar of the UI at runtime.
