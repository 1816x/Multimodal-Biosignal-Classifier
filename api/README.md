# `api/` — FastAPI service

> ⚠️ Educational prototype — NOT a medical device.

HTTP service for the Multimodal Biosignal Classifier.

## Status
**Phase 3** — `GET /health`, `GET /` (service info + disclaimer), `POST /predict`
(multimodal ECG + PPG + accelerometer activity classification; the model serves whatever
modalities its checkpoint was trained on), and `POST /report` (the same prediction plus a
Claude-generated natural-language report, always disclaimer-prefixed).

## Install & run
```bash
pip install -e ".[dev,predict,explain]"   # predict adds torch/numpy/scipy; explain adds the Claude SDK
pip install -e ../model                    # sibling package the model is loaded from
uvicorn biosignal_api.main:app --reload
# http://127.0.0.1:8000  ->  /  ·  /health  ·  /predict  ·  /report  ·  /docs
pytest
```
`POST /predict` loads `MODEL_CHECKPOINT` (default `model/checkpoints/multimodal_phase2.pt`,
produced by `python -m biosignal_model.train`). torch is imported lazily, so the
service still starts without the `predict` extra — the endpoint just returns **503**
until the model is available. `POST /report` additionally needs the `explain` extra and an
`ANTHROPIC_API_KEY`; without either it returns **503**.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness probe |
| GET | `/` | service info — **always carries the disclaimer** |
| POST | `/predict` | one window per modality (ECG/PPG samples, ACC `[x,y,z]`) → activity + confidence + segment (**disclaimer always attached**); 422 if a required modality is missing; 503 if the model isn't trained/installed |
| POST | `/report` | same input as `/predict` → prediction + a Claude-generated natural-language report (**disclaimer-prefixed**); 422 if a required modality is missing; 503 if the model or the report layer (`anthropic` / `ANTHROPIC_API_KEY`) is unavailable; 502 if the Claude API call fails |

The Claude API explanation layer (`explain.py`) reads `ANTHROPIC_API_KEY` /
`ANTHROPIC_MODEL` from `.env` (see `.env.example`). The disclaimer is prepended
deterministically to every report, and all network access is confined to a single
`_complete` seam so the layer is tested without a network connection.
