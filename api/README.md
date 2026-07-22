# `api/` — FastAPI service

> ⚠️ Educational prototype — NOT a medical device.

HTTP service for the Multimodal Biosignal Classifier.

## Status
**Phase 2** — `GET /health`, `GET /` (service info + disclaimer), and `POST /predict`
(multimodal ECG + PPG + accelerometer activity classification; the model serves whatever
modalities its checkpoint was trained on). The Claude-generated report endpoint arrives
in Phase 3.

## Install & run
```bash
pip install -e ".[dev,predict]"        # predict extra adds torch/numpy/scipy
pip install -e ../model                # sibling package the model is loaded from
uvicorn biosignal_api.main:app --reload
# http://127.0.0.1:8000  ->  /  ·  /health  ·  /predict  ·  /docs
pytest
```
`POST /predict` loads `MODEL_CHECKPOINT` (default `model/checkpoints/multimodal_phase2.pt`,
produced by `python -m biosignal_model.train`). torch is imported lazily, so the
service still starts without the `predict` extra — the endpoint just returns **503**
until the model is available.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness probe |
| GET | `/` | service info — **always carries the disclaimer** |
| POST | `/predict` | one window per modality (ECG/PPG samples, ACC `[x,y,z]`) → activity + confidence + segment (**disclaimer always attached**); 422 if a required modality is missing; 503 if the model isn't trained/installed |

The Claude API explanation layer (`explain.py`) reads `ANTHROPIC_API_KEY` /
`ANTHROPIC_MODEL` from `.env` (see `.env.example`); implemented in Phase 3.
