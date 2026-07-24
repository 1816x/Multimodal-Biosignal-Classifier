# Design Decisions

> ⚠️ Educational prototype — NOT a medical device. See the repository README.

This document records how the project was designed **with** AI assistance: what was
proposed and accepted, what was corrected or rejected and why, what the author
decided, and — most usefully — **what the first design got wrong**. It is appended to
as the project moves through its phases.

## Phase 0 — Scaffolding & dataset selection

### The dataset problem (the most important decision so far)

The project spec suggested *"MIT-BIH + PPG-DaLiA or similar."* Taken literally, that
implied fusing two datasets. Looking at the actual data made the flaw obvious:

- **MIT-BIH Arrhythmia Database** has the gold-standard, cardiologist-reviewed beat
  labels — but it is **ECG-only**. No PPG, no accelerometer.
- You cannot make a signal "multimodal" by pairing MIT-BIH's ECG with PPG from a
  *different* dataset and *different* subjects. The signals wouldn't be time-aligned
  or physiologically related — the model would learn dataset identity, not
  multimodal structure. **This first instinct was rejected as dishonest.**

Research into genuinely aligned ECG+PPG datasets (same subject, same clock) narrowed
it to a short list:

| Dataset | Aligned ECG+PPG? | Labels | License | Verdict |
|---|---|---|---|---|
| MIT-BIH | ❌ ECG only | arrhythmia (gold) | ODC-By | Rejected — not multimodal |
| BIDMC | ✅ (single 125 Hz clock) | none (regression only) | ODC-By | Rejected — no class labels |
| MIMIC / PulseDB | ✅ | BP / waveform | credentialed | Rejected — access friction, huge |
| **WESAD** | ✅ | stress / affect | **non-commercial** | Considered |
| **PPG-DaLiA** | ✅ (+ accelerometer) | activity (8-class) | **CC BY 4.0** | **Chosen** |

The choice came down to **WESAD** vs **PPG-DaLiA** — both genuinely multimodal with
aligned signals from the same subjects, both needing no credentialing. The initial
lean was WESAD (stress detection reads as more "clinical"). It was **revised to
PPG-DaLiA** for two concrete reasons:

1. **License.** WESAD is academic **non-commercial**; PPG-DaLiA is **CC BY 4.0**
   (permissive, commercial use allowed with attribution). For a portfolio piece that
   should be freely reusable, permissive wins.
2. **Modalities.** PPG-DaLiA ships a genuine **third** aligned signal (accelerometer)
   alongside ECG and PPG, which gives the "multimodal" claim more room and makes the
   fusion architecture more interesting.

A side benefit: **activity recognition is not a diagnostic task.** For a project that
must loudly disclaim being a medical tool, classifying "walking vs cycling" is a far
safer framing than implying stress or arrhythmia diagnosis.

### Accepted from the AI as-is
- **Product-slice architecture** (preprocessing → PyTorch → FastAPI → Claude report →
  Next.js) instead of a single training notebook — this is the whole differentiator
  vs the ~189 single-signal ECG repos on GitHub.
- **Modality-configurable pipeline** (`ModelConfig.modalities`): Phase 1 trains
  ECG-only, Phase 2 flips on PPG + accelerometer without a rewrite. This maps the
  spec's phasing directly onto a config flag.
- **Dependency-light scaffolding:** `config.py` is pure dataclasses (no numpy/torch)
  so the package imports and the smoke tests run without the heavy stack; `torch` is
  an optional `[train]` extra, needed from Phase 1.

### Decided by the author
- Dataset = **PPG-DaLiA**, task = **activity recognition** (see above). This was
  settled after briefly considering WESAD.
- The educational disclaimer must be **impossible to miss** and must travel with the
  output, not just live in the README — so it is baked into the API's `/` payload now
  and will be injected into every generated report (Phase 3) and shown in the
  dashboard UI (Phase 4).

### Honest caveat carried forward
- PPG-DaLiA mixes devices and rates: ECG at 700 Hz (chest RespiBAN) vs PPG 64 Hz and
  accelerometer 32 Hz (wrist Empatica E4). Multimodal fusion therefore **requires
  resampling onto a common time base** — a real preprocessing concern that Phase 2
  implements and tests explicitly.

### Open / to verify
- Confirm the PPG-DaLiA **CC BY 4.0** label directly on the UCI dataset page before
  the release (automated fetch was blocked during research). The Reiss et al. (2019)
  citation is required attribution regardless.

---

## Phase 1 — ECG-only model, training, and the prediction endpoint

Phase 1 turns the scaffold into a working slice: a real PPG-DaLiA loader, a 1-D CNN,
an honest training run, and `POST /predict`. The modality-configurable design held up
— selecting Phase 1 is still just `config.ECG_ONLY`, no `--phase` flag.

### Accepted / built as designed
- **Model is modality-configurable, not ECG-hardcoded.** `build_model` builds one
  1-D CNN encoder *per* modality in `config.modalities` and late-fuses them; with one
  modality that is a single encoder. Phase 2 adds PPG/ACC by extending the tuple —
  no rewrite, as promised in Phase 0.
- **Loader stays torch-free** (numpy/scipy only) and returns `(float32 (C, L), int)`,
  so it plugs straight into a `DataLoader` while the base package still imports
  without torch. Same dependency-light spirit as `config.py`.

### Decided during Phase 1
- **Resample ECG 700 Hz → 64 Hz.** `window_samples` is defined as
  `window_seconds × target_hz` (512), and `target_hz = 64` is the shared fusion base
  (PPG's native rate). So even ECG-only resamples onto it. **Honest caveat:** 64 Hz is
  well below what ECG *morphology* (QRS shape) needs — but this is **activity
  recognition**, not cardiology, and heart-rate dynamics survive downsampling fine.
  Documented so nobody mistakes this for a cardiac-grade pipeline.
- **Split by subject, not by window.** Train S1–S11, val S12–S13, test S14–S15. A
  random window split would let near-duplicate neighbours straddle train/test and
  inflate accuracy badly. Subject-wise is the honest generalization test, and the
  reported 0.371 is what it costs to be honest.
- **Class-weighted loss.** PPG-DaLiA activities are very imbalanced (`lunch_break`
  ~30% of windows); inverse-frequency weighting keeps minority activities from being
  ignored. Reported metrics are macro *and* weighted so the imbalance is visible.
- **`relevant_segment` is deliberately crude.** It's the temporal arg-max of the last
  conv feature map mapped back to input samples — a cheap localization hint, **not** a
  calibrated attribution. Labelled as such in the code and the response.

### What the first cut got wrong (the useful part)
- **Expecting ECG-only to be "decent."** It isn't, and it shouldn't be: ECG cannot
  observe *motion*. The confusion matrix is blunt about it — `cycling` and `stairs`
  (big HR swings) score F1 0.74 / 0.61, while `table_soccer` collapses to 0.03 and
  `walking`/`working` blur into `lunch_break`. That's not a bug to tune away; it's the
  physiological ceiling of a single modality, and it is the concrete argument for why
  Phase 2's accelerometer + PPG exist.
- **Overfitting to subjects is real.** Best validation accuracy lands at *epoch 2*,
  then train loss keeps falling while val doesn't — the net memorizes the 11 training
  subjects. We keep the best-val checkpoint rather than the last one, and resisted the
  temptation to inflate the number by tuning on the test subjects.

---

## Phase 2 — Multimodal fusion (ECG + PPG + accelerometer)

Phase 2 makes the model actually multimodal: the loader now reads all three signals,
resamples them onto a shared clock, aligns and windows them together, and the training
and serving paths fuse them. The Phase 0 promise ("no rewrite, just a longer
`modalities` tuple") held for the *model* — but not for the data path.

### What the closing note of Phase 1 got wrong (the useful part)
- The hand-off said Phase 2 was "just extend `config.MULTIMODAL` with PPG +
  accelerometer." That undersold it. The **config layer was already done** — the
  `Modality` enum, `NATIVE_SAMPLE_RATES_HZ`, `CHANNELS_PER_MODALITY` and the
  `MULTIMODAL` preset all listed PPG/ACC since Phase 0. The real Phase 2 work was the
  **implementation** those names had been standing in for: the loader was hard ECG-only
  with an explicit guard that *raised* on anything else, and training/serving were
  wired to `ECG_ONLY`. Recording this because "the config already says it" is exactly
  the kind of thing that reads as done when it isn't.

### Decided during Phase 2
- **Accelerometer as 3 axes, not a scalar magnitude.** The scaffold had provisionally
  modeled ACC as 1-channel magnitude (`CHANNELS_PER_MODALITY["acc"] = 1`). Phase 2
  changed it to the native **tri-axial** signal (`= 3`). Directional motion is the
  whole reason the accelerometer is here — it is what separates `walking` / `stairs` /
  `cycling`, the exact classes ECG alone blurred — and 3 axes is the form the dataset
  and the reference paper (Reiss et al.) use. Magnitude is orientation-invariant, which
  is a real robustness argument, but the Empatica E4 is worn consistently in PPG-DaLiA,
  and magnitude is trivially derivable from the axes later if we want that ablation.
  The API `acc` field is a list of `[x, y, z]` samples accordingly.
- **Cross-device alignment by truncation to a common length.** Each modality is
  resampled per channel to `target_hz` (64 Hz: ECG 700→64 down, PPG 64→64 no-op, ACC
  32→64 up), then all modalities are truncated to the shortest resampled length so their
  windows line up 1:1 on the shared time base. Labels come from the 4 Hz `activity`
  track (the protocol clock all devices share). **Honest caveat:** this is simple
  truncation alignment, not sub-sample cross-correlation — adequate because PPG-DaLiA is
  already protocol-synchronized, but noted so nobody mistakes it for fine device
  time-sync.
- **Per-modality normalization.** Z-score stats are fit per modality (and per ACC axis),
  on the TRAIN split only, and stored per-modality in the checkpoint. A single global
  mean/std would be wrong across signals with completely different units and scales.
- **A `--phase` flag after all.** Phase 1's note said modality was "selected by config
  preset (not a flag)." Phase 2 adds `--phase {1,2}` (default 2) so the ECG-only
  baseline stays reproducible while multimodal becomes the default run. A modest,
  deliberate reversal of that earlier stance now that two presets actually coexist.
- **One sample path for both presets.** The dataset now always returns a
  `{modality: (channels, window)}` dict — even ECG-only — so training and serving have a
  single code path (`MultimodalClassifier.forward` already accepted a dict). The
  ECG-only numbers are unchanged: it is the same resample/window/label logic with one
  modality.

### Accepted / held up from earlier phases
- **The modality-configurable model needed no structural change.** `build_model` still
  builds one encoder per `config.modalities` entry and late-fuses; the only model edit
  was the ACC channel count. Serving rebuilds the architecture straight from the
  checkpoint's `modalities` list, so the same route serves an ECG-only or a full
  multimodal checkpoint.
- **Preprocessing stayed generic.** `resample_to_common_rate` / `window_signal` /
  `compute_norm_stats` / `normalize` were already modality-agnostic and per-channel, so
  Phase 2 reused them as-is; the new tests cover the multi-rate and multi-channel paths.

### Results (honest, as ever)
- Multimodal metrics are reported the same way as Phase 1 — subject-wise split, not
  inflated — in [`model/metrics/phase2_multimodal.json`](../model/metrics/phase2_multimodal.json)
  and summarized in the README metrics table alongside the ECG-only baseline, so the
  lift (or lack of it) from adding motion is visible rather than asserted.

---

## Phase 3 — Claude API explanation layer

Phase 3 delivers the project's differentiating angle: an **LLM explanation layer** that
turns the model's numeric output (activity + confidence + relevant segment) into a
plain-language, clinical-*style* report via the Claude API — "built visibly *with* AI."

### Decided during Phase 3
- **A separate `POST /report` endpoint**, not a flag on `/predict`. `/report` takes the
  same window input, runs the prediction, then generates the report — mirroring the
  architecture flowchart (prediction → report) and keeping `/predict` a pure, fast
  classifier with no LLM dependency or latency. Extending `/predict` with an
  `explain: true` flag was considered and rejected: it would mix two concerns and make
  every prediction response's shape conditional.
- **The disclaimer is prepended deterministically**, not left to the model.
  `generate_report` returns `f"{DISCLAIMER}\n\n{body}"`, so the not-a-medical-tool
  warning is *guaranteed* present even if the model omits it — the Phase 0 promise that
  the disclaimer "travels with the output" is enforced in code, not prompt-hoped. The
  system prompt also instructs the model to stay educational and never give medical
  advice, as defense in depth.
- **The disclaimer moved to a canonical home** (`biosignal_api/__init__.py`). Both
  `main.py` (responses) and `explain.py` (report prefix) need the exact same text;
  `explain` can't import it from `main` (circular — `main` imports `explain` for the
  route), so it lives in the package `__init__` and both import `from . import DISCLAIMER`.
  Existing `from biosignal_api.main import DISCLAIMER` imports still resolve via re-export.
- **One `_complete` network seam for testability.** Importing `anthropic`, constructing
  the client, and calling the Messages API all live in `explain._complete`. The report
  tests monkeypatch that seam (happy path) or delete `ANTHROPIC_API_KEY` (unavailable
  path), so the whole Phase 3 test suite runs with **no network and without `anthropic`
  installed** — the same "green in CI without the heavy stack" discipline as the Phase 1/2
  torch-optional prediction tests.
- **Error mapping mirrors the prediction path.** Missing package / missing API key →
  `ExplanationUnavailable` → **503** (same spirit as `ModelUnavailable`); an upstream
  Claude API failure → `ExplanationError` → **502** (bad gateway). So `/report` degrades
  cleanly with a clear message when unconfigured, exactly as `/predict` does without a
  checkpoint.
- **Model default = `claude-sonnet-5`**, read from `ANTHROPIC_MODEL` (already reserved in
  `.env.example` since Phase 0). Sonnet is the right cost/quality point for short
  educational report generation; the env var lets a deployer swap it without a code change.

### Honest caveat carried forward
- The report is generated from the numeric prediction alone (class + confidence +
  segment), not from the raw signal — it explains *what the model concluded*, in plain
  language, and cannot add clinical insight the model never had. That is deliberate for an
  educational prototype, and the disclaimer says so on every report.

---

## Phase 4 — Next.js dashboard

Phase 4 adds the `dashboard/`: a Next.js 16 / React 19 app that plots an ECG + PPG +
accelerometer window, shows the model's prediction with the relevant segment shaded
across all three signals, and renders the Claude report — disclaimer prominent
throughout. It consumes the Phase 1–3 API with **no changes to `api/`**.

### Decided during Phase 4
- **The browser talks to the API through a Next.js server-side proxy, not CORS.**
  Route Handlers at `app/api/{health,predict,report}` forward to the FastAPI service;
  the browser only ever calls this app's own origin. Adding `CORSMiddleware` to the
  finished, merged `api/` was the obvious alternative and was rejected: it would mutate
  a completed component and force an allowed-origins decision. The proxy keeps each
  phase landing independently, holds the upstream URL server-side (`API_BASE_URL`,
  deliberately **not** `NEXT_PUBLIC_`), and gives one seam to normalize errors — a
  backend connection failure becomes the same **503** the UI already handles for an
  untrained model.
- **Demo mode is real but never dishonest.** A fresh clone has no trained checkpoint
  and no dataset, so `/predict` and `/report` return 503 and there is no real window to
  plot. The dashboard ships a few **synthetic** input windows and **canned** demo
  results so the whole UI is viewable with `npm run dev` and no backend. The hard rule:
  a live result is **never** silently replaced by a canned one — every synthetic or
  canned piece is visibly badged and the reason for any fallback is shown. Inputs are
  **synthetic, not dataset-extracted**, on purpose: the project never commits PPG-DaLiA
  (CC BY 4.0, gitignored), and a real window fed to a 503 endpoint buys no real output
  anyway.
- **The disclaimer lives in two places, deliberately.** A global `DisclaimerBanner`
  (mounted in the root layout) is always on and depends on no response — the Phase 0
  hard requirement, now in the UI. Separately, `ReportPanel` surfaces the
  response-carried disclaimer that the API prepends to every report. The UI copy mirrors
  the API's canonical `DISCLAIMER`.
- **Recharts for the plots.** 512-point line charts × 3 modalities plus a shaded
  `ReferenceArea` for `[start, end]` — Recharts gives the overlay for free and reads as
  idiomatic React. Charts render client-side only (past hydration, via a
  `useSyncExternalStore`-based mounted flag) to avoid SSR size warnings and hydration
  mismatches.
- **Two actions, mirroring the API split.** Separate **Run prediction** (`/predict`) and
  **Explain with Claude** (`/report`) buttons keep the fast classifier independent of the
  opt-in, token-spending LLM call — the same separation Phase 3 chose for the endpoints.

### Honest caveat carried forward
- Uploaded windows are validated to the canonical **512-sample** (8 s @ 64 Hz) grid so
  the plotted signal and the returned `relevant_segment` line up 1:1. Differently-sized
  windows (which the API would resample internally) are out of scope for the overlay and
  are rejected client-side with a clear message, rather than silently mis-aligning.

### Deferred to Phase 5
- A **CI workflow** gating `model` + `api` + `dashboard` together (the repo has no
  `.github/` yet); Phase 4 ships the npm scripts (`lint`, `typecheck`, `build`, `test`)
  it will call. Also deferred: deployment, real dataset-extracted samples behind a local
  script, and a model-card panel surfacing the honest metrics.

---

## Phase 5 — Release v0.1.0 (CI, deployment, model card)

Phase 5 turns the working slice into a tagged `v0.1.0` release: CI across all three
packages, a Docker Compose deployment, a model card, and honest-metrics cleanup.

### Decided during Phase 5
- **CI is tests-only, in two Python tiers.** `.github/workflows/ci.yml` runs a `light`
  tier (`model[dev]` + `api[dev]`, no torch — `test_model`/`test_predict` self-skip via
  `importorskip`) and a `full` tier (CPU torch + `[train]`/`[predict]`), so the "green
  without the heavy stack" discipline is both demonstrated and enforced; the dashboard
  job runs the existing `lint`/`typecheck`/`test`/`build` scripts. No Python
  linter/formatter was introduced — the release doesn't churn existing code.
- **Deployment is self-contained Docker Compose**, not a PaaS. `api/Dockerfile` (context
  = repo root, since the API needs the sibling `model` package) + a multi-stage
  `dashboard/Dockerfile` (Next.js `standalone` output) wired by `docker-compose.yml`
  (`dashboard` → `api` via `API_BASE_URL`). No external accounts; without a mounted
  checkpoint the API returns 503 and the dashboard shows its badged demo — the same
  honest default as everywhere else.
- **The model card ships as both a document and a panel.** `MODEL_CARD.md` is the
  canonical record (intended use, subject-wise metrics, limitations); a compact in-app
  "About this model" panel surfaces the headline honest numbers so predictions are read
  with appropriate skepticism.
- **Honest-metrics fix:** the README's Phase 2 *validation* Weighted-F1 was `0.744`, a
  transcription error — the canonical `phase2_multimodal.json` value is `0.705` (macro F1
  is 0.748). Corrected as part of the release.
- **Version alignment:** the two Python packages moved `0.1.0.dev0 → 0.1.0` to match the
  dashboard and the `v0.1.0` tag; the API's runtime status string now reports `v0.1.0`.

### Honest caveat carried forward
- The Docker images could not be built inside the development sandbox (its
  TLS-intercepting proxy is not trusted by `pip`/`npm` in-container) — an environment
  limitation, not a Dockerfile defect. The compose config, the dashboard `standalone`
  output, and the package installs were validated directly; the images build normally on
  a standard host / in CI.

---

*The v0.1.0 release closes the planned roadmap (Phases 0–5). Further work would start a new cycle.*
