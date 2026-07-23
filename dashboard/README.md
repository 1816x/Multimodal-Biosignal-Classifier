# `dashboard/` — Next.js visualization (Phase 4)

> ⚠️ Educational prototype — NOT a medical device.

The dashboard for the Multimodal Biosignal Classifier: it plots an ECG + PPG +
accelerometer window, shows the model's activity **prediction** (class +
confidence) with the **relevant segment** shaded across all three signals, and
renders the **Claude-generated report** — with the educational, not-a-medical-tool
disclaimer shown prominently throughout.

**Stack:** Next.js 16 / React 19 (TypeScript, App Router), Tailwind CSS, Recharts.
Requires **Node ≥ 20**.

## Quickstart

```bash
cd dashboard
npm install
npm run dev          # http://localhost:3000
```

The dashboard works **with or without a backend**:

- **No backend (fresh clone).** It ships a few **synthetic** sample windows and
  **canned** demo results, so `npm run dev` shows real signal plots plus an
  example prediction and report immediately. Everything synthetic or canned is
  clearly **badged as demo** — it is never presented as a live model output.
- **With the API.** Point it at a running FastAPI service (see below) and the
  same UI shows **live** predictions and reports.

## Connecting to the API

The browser only ever talks to this app; requests to `/api/health`, `/api/predict`
and `/api/report` are **proxied server-side** to the FastAPI service. This avoids
CORS and keeps the upstream URL off the client. Configure it with a server-only
env var:

```bash
cp .env.local.example .env.local
# API_BASE_URL=http://127.0.0.1:8000   (default when unset)
```

Run the API (from the repo root) so the dashboard can reach it:

```bash
pip install -e "api[dev,predict,explain]" && pip install -e model
python -m biosignal_model.train           # produces the checkpoint /predict needs
uvicorn biosignal_api.main:app            # http://127.0.0.1:8000
# set ANTHROPIC_API_KEY in the repo-root .env for live /report
```

Without a trained checkpoint the API returns **503** for `/predict` and `/report`;
the dashboard detects this and falls back to the badged demo result, telling you
why.

## Input

- **Sample dropdown** (default) — the bundled synthetic windows.
- **Upload** a JSON window (`{ "ecg": […512…], "ppg": […512…], "acc": [[x,y,z]…512…] }`)
  or **paste** the same JSON. Windows are validated to the canonical 512-sample
  (8 s @ 64 Hz) grid before use.

Two actions mirror the API: **Run prediction** (`/predict`, fast) and **Explain
with Claude** (`/report`, an explicit opt-in LLM call).

## Layout

```
app/
  layout.tsx            global disclaimer banner + shell
  page.tsx              -> DashboardClient
  api/{health,predict,report}/route.ts   server-side proxy to FastAPI
components/             DisclaimerBanner, StatusBanner, DemoBadge, InputControls,
                       SignalChart, SignalPanel, PredictionCard, ReportPanel, DashboardClient
lib/
  types.ts             TS mirrors of the API schemas
  constants.ts         DISCLAIMER, WINDOW_SAMPLES (512), activities (mirrors the backend)
  api.ts               browser client -> /api/*
  proxy.ts             server proxy helper
  validateWindow.ts    pure validator for uploaded/pasted windows
  fixtures/            synthetic sample windows + canned demo results
```

## Scripts

```bash
npm run dev         # dev server
npm run build       # production build
npm run typecheck   # tsc --noEmit
npm run lint        # eslint
npm test            # vitest (unit + component tests)
```
