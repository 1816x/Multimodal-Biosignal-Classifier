# `model/` — preprocessing + PyTorch model

> ⚠️ Educational prototype — NOT a medical device.

Signal preprocessing and the modality-configurable PyTorch classifier for the
Multimodal Biosignal Classifier (dataset: PPG-DaLiA — ECG + PPG + accelerometer).

## Status
**Phase 2 — multimodal (ECG + PPG + accelerometer), implemented.** The loader,
cross-device resampling + window alignment, per-modality normalization, the 1-D CNN with
one encoder per modality + late fusion, and the training CLI are real and produce an
honest checkpoint + metrics. `--phase 1` reproduces the ECG-only baseline unchanged.

## Install & train
```bash
pip install -e ".[train,dev]"          # numpy/pandas/scipy + PyTorch + pytest
python scripts/download_data.py        # PPG-DaLiA (~2.6 GB, CC BY 4.0) -> ./data
python -m biosignal_model.train        # phase 2 multimodal, all 15 subjects (--phase 1 = ECG baseline, --smoke to spot-check)
pytest
```
Training writes `checkpoints/multimodal_phase2.pt` (gitignored) and
`metrics/phase2_multimodal.json` (`--phase 1` writes the ECG-only baseline). See the
root README's *Model metrics* for the honest read of both.

## Layout
```
src/biosignal_model/
  config.py          modality/task config + TrainConfig (real)
  datasets/          PPG-DaLiA loader — per-modality aligned windows + activity labels
  preprocessing.py   resample / window / normalize
  model.py           modality-configurable 1-D CNN encoders + late-fusion head
  train.py           training CLI (subject-wise split, honest metrics)
scripts/download_data.py   PPG-DaLiA fetch + extract
```
