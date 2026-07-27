# Model Card — Multimodal Biosignal Classifier

> ⚠️ **EDUCATIONAL PROTOTYPE — NOT an approved medical or diagnostic tool.**
> This service must not be used for clinical decisions. Model outputs are
> illustrative only. The model has **not** been clinically validated.

A modality-configurable 1‑D CNN that classifies short windows of wearable
biosignals (**ECG + PPG + accelerometer**) into one of **8 everyday activities**.
Built as a learning / portfolio project; see [`README.md`](README.md) and
[`docs/design-decisions.md`](docs/design-decisions.md) for the full story.

- **Version:** 0.1.0
- **Task:** Human activity recognition (8‑class), **not** a diagnostic task
- **Framework:** PyTorch
- **License (code):** MIT · **License (dataset):** CC BY 4.0

## Model details

- **Architecture:** one 1‑D CNN encoder **per modality**, late‑fused, then a linear
  classifier head. The same code path runs ECG‑only (Phase 1) or the full
  ECG + PPG + accelerometer set (Phase 2), selected by `ModelConfig.modalities`.
- **Parameters:** **53,256** (ECG‑only) · **157,896** (multimodal).
- **Input:** one 8‑second window per modality, resampled to a shared **64 Hz**
  grid → **512** samples (ECG, PPG; accelerometer as 512 `[x, y, z]` triples).
- **Output:** predicted activity, **temperature‑calibrated** confidence (softmax of
  `logits / T`, with T fit on validation), and a crude `relevant_segment` (temporal
  arg‑max of the last conv feature map — a localization *hint*, **not** a calibrated
  attribution).
- **Regularization (v0.2):** train‑time data augmentation (jitter / scaling /
  time‑shift / magnitude‑warp), encoder + head dropout, and a cosine LR schedule with
  early‑stopping.
- **Classes:** `sitting`, `stairs`, `table_soccer`, `cycling`, `driving`,
  `lunch_break`, `walking`, `working`.

## Intended use

- **In scope:** education and portfolio demonstration — showing an honest,
  end‑to‑end multimodal ML slice (preprocessing → model → API → LLM report → UI).
- **Out of scope (do not use for):** any medical, diagnostic, clinical, or
  health‑monitoring purpose; any real decision about a person. Activity
  recognition was deliberately chosen over stress/arrhythmia framings precisely
  because it is **not** a diagnostic task.

## Training data

- **Dataset:** [PPG‑DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) —
  UCI Machine Learning Repository, dataset #495. **CC BY 4.0** (attribution
  required). 15 subjects, ~36 h; modalities time‑aligned from the same subjects:
  ECG (chest, 700 Hz), PPG/BVP (wrist, 64 Hz), 3‑axis accelerometer (wrist, 32 Hz).
- **Citation:** A. Reiss, I. Indlekofer, P. Schmidt, K. Van Laerhoven, *"Deep PPG:
  Large‑scale Heart Rate Estimation with Convolutional Neural Networks"*,
  **Sensors**, 2019.
- **The dataset is downloaded, never committed** (gitignored). The committed
  honest record of results is `model/metrics/*.json`.
- **Split — by subject, not by window:** train **S1–S11**, validation **S12–S13**,
  test **S14–S15**. No window from a test subject is ever seen in training. A
  random window split would let near‑duplicate neighbours straddle train/test and
  inflate accuracy; subject‑wise is the honest generalization test.

## Training configuration

| | |
|---|---|
| Epochs | 25 max · cosine LR schedule · early‑stopping (patience 6) · best‑val checkpoint kept |
| Batch size | 128 |
| Learning rate | 1e‑3 (Adam), weight decay 1e‑4 |
| Window / stride | 8 s / 2 s |
| Sampling rate | 64 Hz (shared fusion base) |
| Loss | cross‑entropy, **class‑weighted** (inverse frequency) |
| Augmentation (train only) | jitter · scaling · time‑shift · magnitude‑warp |
| Dropout | 0.3 (head) · 0.1 (encoders) |
| Calibration | temperature scaling (T fit on validation) |
| Seed | 42 |

Class weighting counters strong activity imbalance (`lunch_break` alone is ~30% of
windows). Chance for 8 balanced classes is **12.5%**.

## Evaluation (honest, subject‑wise, not inflated)

Reported on the **held‑out test subjects (S14–S15)**, with validation (S12–S13)
alongside. Full numbers, per‑class breakdowns, and confusion matrices live in
[`model/metrics/phase1_ecg.json`](model/metrics/phase1_ecg.json) and
[`model/metrics/phase2_multimodal.json`](model/metrics/phase2_multimodal.json).

| Model | Split | Accuracy | Macro F1 | Weighted F1 | n |
|---|---|---|---|---|---|
| ECG‑only (Phase 1) | Validation (S12–S13) | 0.463 | 0.498 | 0.433 | 6,334 |
| ECG‑only (Phase 1) | **Test (S14–S15)** | **0.609** | **0.645** | 0.620 | 6,134 |
| Multimodal (Phase 2) | Validation (S12–S13) | 0.768 | 0.796 | 0.760 | 6,334 |
| **Multimodal (Phase 2)** | **Test (S14–S15)** | **0.776** | **0.814** | 0.761 | 6,134 |

**Per‑class test F1 (multimodal vs ECG‑only):** `cycling` 1.00 (0.89), `driving`
**0.93 (0.68)**, `table_soccer` **0.89 (0.20)**, `sitting` 0.85 (0.93), `stairs` 0.84
(0.74), `walking` **0.81 (0.66)**, `lunch_break` 0.71 (0.43), `working` 0.47 (0.61).

**Read (v0.2):** ECG‑only is ~5× chance; adding the wrist **accelerometer + PPG** plus
the v0.2 regularization lifts test accuracy to **0.776** on the *same* held‑out subjects.
The motion classes jump (`walking` 0.36→0.81, `table_soccer` 0.20→0.89, `driving`
0.68→0.93) and the overfitting gap closes — best validation now lands at **epoch ~10**
(was epoch 2) and **test ≈ validation**. Still **not** a solved task — see limitations.

## Limitations

- **Weakest class:** `working` is now the weakest (test F1 0.47, recall 0.32) — the
  model confuses it with `lunch_break` (both sedentary desk activities), and ECG‑only
  actually reads `working`/`sitting` slightly better than the multimodal model.
- **ECG‑only ceiling:** ECG cannot observe motion, so `table_soccer` (F1 0.20) and other
  motion‑defined classes stay capped without the accelerometer.
- **Small held‑out set:** validation and test are 2 subjects each, so the numbers are
  noisy — the val/test ordering can even flip (here test > val). Treat single figures
  with appropriate error bars.
- **64 Hz resampling:** well below what ECG *morphology* (QRS shape) needs — fine
  for activity/heart‑rate dynamics, but **not** a cardiac‑grade pipeline.
- **Truncation alignment:** modalities are aligned by truncation to a common
  length, not sub‑sample cross‑correlation (adequate because PPG‑DaLiA is already
  protocol‑synchronized).
- **`relevant_segment` is not calibrated** — a cheap localization hint only.
- **The report explains the prediction, not the raw signal:** the Claude report is
  generated from the numeric output (class + confidence + segment); it cannot add
  clinical insight the model never had.
- **All 15 subjects (v0.2):** the model now trains on all 15 subjects (S1–S11 incl. S6
  / S12–S13 / S14–S15); the earlier S6‑omitted caveat no longer applies.

## Ethical considerations

This is an educational prototype and **not** a medical device. It has not been
clinically validated and must never inform a health decision. The educational
disclaimer is baked into the API responses, prepended to every generated report,
and shown prominently in the dashboard UI — it travels with the output, by design.
The dataset is used under CC BY 4.0 with attribution above.
