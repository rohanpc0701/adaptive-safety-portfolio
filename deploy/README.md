# Deployment

## Hugging Face Spaces (recommended for public demo)

1. Create a new **Docker** Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Clone this repo (or copy `deploy/huggingface/` contents into the Space repo).
3. Set Space secrets:
   - `ANTHROPIC_API_KEY` (optional, for judge arm in real mode)
   - `HF_TOKEN` (for gated models in real mode)
4. Set environment variables:
   - `DEMO_MODE=synthetic` — CPU-friendly calibrated demo (default)
   - `DEMO_MODE=real` — full GPU models (needs A10G+ GPU hardware)
5. Build uses `deploy/huggingface/Dockerfile`; serves on port **7860**.

```bash
# Local smoke test (synthetic, no GPU)
cd safety-portfolio
pip install -r requirements.txt
DEMO_MODE=synthetic python deploy/huggingface/app.py
# Open http://localhost:7860
```

## Self-hosted (GPU machine)

```bash
cd scripts && uvicorn demo_server:app --host 0.0.0.0 --port 8000
ssh -L 8765:localhost:8000 user@gpu-host
open demo/index.html
```

## Modal / other hosts

Wrap `scripts/demo_server.py` or `deploy/huggingface/app.py` in your platform's container entrypoint. The FastAPI app loads models once at startup.
