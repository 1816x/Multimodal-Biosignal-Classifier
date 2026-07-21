# `model/` — preprocessing + PyTorch model

> ⚠️ Educational prototype — NOT a medical device.

Signal preprocessing and the modality-configurable PyTorch classifier for the
Multimodal Biosignal Classifier (dataset: PPG-DaLiA — ECG + PPG + accelerometer).

## Status
Phase 0 — package scaffolding. `config.py` is real (the modality/task config used by
the tests); `datasets/`, `preprocessing.py`, `model.py`, and `train.py` are
documented **stubs** that land in Phases 1–2 (see the root README roadmap).

## Install
```bash
pip install -e ".[dev]"      # base (numpy/pandas/scipy) + pytest
pip install -e ".[train]"    # adds PyTorch — needed from Phase 1
pytest
```

## Layout
```
src/biosignal_model/
  config.py          modality + task config (real)
  datasets/          PPG-DaLiA loader (stub)
  preprocessing.py   resample / window / normalize (stub)
  model.py           1-D CNN encoders + fusion head (stub)
  train.py           training entrypoint (stub)
scripts/download_data.py   PPG-DaLiA fetch (documented; wired up Phase 1)
```
