# Architecture

> ⚠️ Educational prototype — NOT a medical device. See the repository README.

## Dataflow

```
raw signals (ECG + PPG + accelerometer, PPG-DaLiA)
      |
      v
[model/ · preprocessing]  resample to a common rate -> window -> normalize
      |
      v
[model/ · PyTorch]  modality-configurable 1-D CNN encoders -> late fusion -> classifier
      |
      v
class + confidence + relevant segment
      |
      +-----------------------------+
      v                             v
[api/ · FastAPI]             [api/ · explain.py]
serves prediction     -->    Claude API -> clinical-style report
      |                       (+ educational disclaimer)
      v
[dashboard/ · Next.js]  signal plot + prediction + report
```

## Components

- **`model/`** (Python) — dataset loaders, preprocessing, and the PyTorch model.
  Modality-configurable via `biosignal_model.config.ModelConfig`: the same code path
  runs ECG-only (Phase 1) or ECG + PPG + accelerometer (Phase 2).
- **`api/`** (Python, FastAPI) — the service. Phase 0 exposes `GET /health` and
  `GET /` (info + disclaimer); `POST /predict` and the report endpoint arrive with
  Phases 1–3. Pydantic v2 schemas live in `schemas.py`.
- **`dashboard/`** (TypeScript, Next.js) — visualization, built in Phase 4.

## Modality configuration

`ModelConfig.modalities` selects which encoders are active. Native PPG-DaLiA rates
(ECG 700 Hz, PPG 64 Hz, ACC 32 Hz) are resampled to `ModelConfig.target_hz` so the
per-modality encoders share a time base before fusion.

## Why this shape

The point of the project is to be a *product*, not a notebook. Keeping the model,
the service, and the UI as separate installable pieces — with the LLM explanation as
its own module — mirrors how such a system would actually be deployed, and lets each
phase land independently. See `design-decisions.md` for the reasoning.
