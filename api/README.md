# `api/` — FastAPI service

> ⚠️ Educational prototype — NOT a medical device.

HTTP service for the Multimodal Biosignal Classifier.

## Status
Phase 0 — infrastructure only: `GET /health` and `GET /` (service info + the
educational disclaimer). `POST /predict` and the Claude-generated report endpoint
arrive in Phases 1–3.

## Install & run
```bash
pip install -e ".[dev]"
uvicorn biosignal_api.main:app --reload
# http://127.0.0.1:8000  ->  /  ·  /health  ·  /docs
pytest
```

## Endpoints (Phase 0)
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness probe |
| GET | `/` | service info — **always carries the disclaimer** |

The Claude API explanation layer (`explain.py`) reads `ANTHROPIC_API_KEY` /
`ANTHROPIC_MODEL` from `.env` (see `.env.example`); implemented in Phase 3.
