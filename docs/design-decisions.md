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

*Phases 1–5 will append their own decisions below as they are built.*
