# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/).

> ⚠️ Educational prototype — NOT a medical device.

## [0.1.0] — 2026-07-23

First tagged release: an end-to-end, **multimodal** biosignal activity classifier
delivered as a product slice (model → API → LLM report → dashboard), with honest,
subject-wise metrics and the educational disclaimer travelling with every output.

### Added
- **Phase 0 — Scaffolding.** Repo structure, disclaimer, dataset selection
  (PPG-DaLiA, CC BY 4.0), design docs, and the FastAPI skeleton (`GET /health`, `GET /`).
- **Phase 1 — ECG-only classifier.** PPG-DaLiA loader, a 1-D CNN, a subject-wise
  training CLI with honest metrics, and `POST /predict`.
- **Phase 2 — Multimodal fusion.** ECG + PPG + 3-axis accelerometer: cross-device
  resampling to a shared 64 Hz base, per-modality normalization, one encoder per
  modality with late fusion. Test accuracy **0.371 (ECG-only) → 0.650 (multimodal)**
  on the same held-out subjects.
- **Phase 3 — Claude report layer.** `POST /report` turns the numeric prediction into
  a plain-language, disclaimer-prefixed report via the Claude API.
- **Phase 4 — Next.js dashboard.** Signal plots (ECG/PPG/ACC) with the relevant
  segment shaded, prediction + report panels, an always-on disclaimer, and a
  backend-less demo mode (badged synthetic/canned data). Talks to the API through a
  server-side proxy (no CORS).
- **Phase 5 — Release.** GitHub Actions CI (Python light/full tiers + dashboard),
  Docker Compose deployment (api + dashboard), a [`MODEL_CARD.md`](MODEL_CARD.md), and
  an in-app "About this model" panel surfacing the honest metrics.

### Fixed
- Corrected the Phase 2 *validation* Weighted-F1 in the README (`0.744` → `0.705`) to
  match the canonical `model/metrics/phase2_multimodal.json`.

[0.1.0]: https://github.com/1816x/multimodal-biosignal-classifier/releases/tag/v0.1.0
