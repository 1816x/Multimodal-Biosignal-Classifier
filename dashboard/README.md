# `dashboard/` — Next.js visualization (Phase 4)

> ⚠️ Educational prototype — NOT a medical device.

**Placeholder.** The dashboard is intentionally **not implemented yet** — per the
project roadmap it is built in **Phase 4**, after the model (Phases 1–2) and the
Claude explanation layer (Phase 3) exist to feed it.

## Planned scope (Phase 4)
- Plot the input signals (ECG / PPG / accelerometer windows).
- Show the model's prediction (activity class + confidence) and the relevant signal
  segment.
- Render the Claude-generated natural-language report — **with the educational,
  not-a-medical-tool disclaimer shown prominently in the UI.**

## Planned stack
Next.js 16 / React 19 (TypeScript), talking to the FastAPI service in `api/`.
Requires Node ≥ 20.
