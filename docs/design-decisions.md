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

*Phases 3–5 will append their own decisions below as they are built.*
