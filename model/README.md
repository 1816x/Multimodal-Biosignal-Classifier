# `model/` — preprocessing + PyTorch model

> ⚠️ Educational prototype — NOT a medical device.

Signal preprocessing and the modality-configurable PyTorch classifier for the
Multimodal Biosignal Classifier (dataset: PPG-DaLiA — ECG + PPG + accelerometer).

## Status
**Phase 1 — ECG-only, implemented.** The loader, preprocessing, 1-D CNN, and training
CLI are real and produce an honest checkpoint + metrics. PPG/accelerometer fusion is
Phase 2 (the architecture already supports it — just a longer `config.modalities`).

## Install & train
```bash
pip install -e ".[train,dev]"          # numpy/pandas/scipy + PyTorch + pytest
python scripts/download_data.py        # PPG-DaLiA (~2.6 GB, CC BY 4.0) -> ./data
python -m biosignal_model.train        # all 15 subjects, ~4.5 min on CPU (--smoke to spot-check)
pytest
```
Training writes `checkpoints/ecg_phase1.pt` (gitignored) and `metrics/phase1_ecg.json`.
Test accuracy is **0.371** (8-class, subject-wise split) — see the root README's
*Model metrics* for the honest read.

## Layout
```
src/biosignal_model/
  config.py          modality/task config + TrainConfig (real)
  datasets/          PPG-DaLiA loader — ECG windows + activity labels
  preprocessing.py   resample / window / normalize
  model.py           modality-configurable 1-D CNN encoders + late-fusion head
  train.py           training CLI (subject-wise split, honest metrics)
scripts/download_data.py   PPG-DaLiA fetch + extract
```
