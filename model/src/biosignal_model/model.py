"""PyTorch model for the multimodal biosignal classifier.

A modality-configurable architecture: one 1-D CNN encoder per active signal
(ECG / PPG / accelerometer), late-fused into a small classifier head. Phase 1
trains the ECG-only encoder; Phase 2 adds the PPG/ACC encoders + fusion — the same
code path, just a longer ``config.modalities`` tuple.

``torch`` is an optional (heavy) dependency imported lazily inside the builder so
this module can be inspected without it. Install with: ``pip install -e "model[train]"``.

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

# Input channels per modality: ECG 1, PPG 1 (BVP), ACC 3 (tri-axial x/y/z). Phase 2
# feeds the accelerometer as 3 axes (not a scalar magnitude) so the model keeps the
# directional motion cues that separate walking / stairs / cycling.
CHANNELS_PER_MODALITY: dict[str, int] = {"ecg": 1, "ppg": 1, "acc": 3}

_CLASSES = None  # cache so the lazily-built nn.Module subclasses are defined once


def _torch_classes():
    """Define (once) and return the torch model classes. Keeps torch import lazy."""
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES

    import torch
    from torch import nn

    class ModalityEncoder(nn.Module):
        """Small 1-D CNN: 3× (Conv → BN → ReLU → MaxPool). Returns a feature map."""

        def __init__(self, in_channels: int, widths=(32, 64, 128), kernels=(7, 5, 3)):
            super().__init__()
            layers = []
            prev = in_channels
            for w, k in zip(widths, kernels):
                layers += [
                    nn.Conv1d(prev, w, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(w),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                ]
                prev = w
            self.net = nn.Sequential(*layers)
            self.out_channels = prev

        def forward(self, x):  # x: (B, C, L) -> (B, out_channels, L//8)
            return self.net(x)

    class MultimodalClassifier(nn.Module):
        """Per-modality encoders → global average pool → concat (late fusion) → head.

        ``forward`` accepts either a single tensor (the one active modality, Phase 1)
        or a ``{modality_value: tensor}`` dict (Phase 2). The last feature map per
        modality is retained for :meth:`relevant_segment`.
        """

        def __init__(self, modality_keys: list[str], num_classes: int, dropout: float = 0.3):
            super().__init__()
            self.modality_keys = list(modality_keys)
            self.encoders = nn.ModuleDict(
                {m: ModalityEncoder(CHANNELS_PER_MODALITY[m]) for m in self.modality_keys}
            )
            feat_dim = sum(self.encoders[m].out_channels for m in self.modality_keys)
            self.head = nn.Sequential(
                nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(128, num_classes),
            )
            self._last_maps: dict[str, "torch.Tensor"] = {}

        def _as_dict(self, x):
            if isinstance(x, dict):
                return x
            if len(self.modality_keys) != 1:
                raise ValueError("pass a {modality: tensor} dict when >1 modality is active")
            return {self.modality_keys[0]: x}

        def forward(self, x):
            inputs = self._as_dict(x)
            feats, self._last_maps = [], {}
            for m in self.modality_keys:
                fmap = self.encoders[m](inputs[m])      # (B, hidden, L')
                self._last_maps[m] = fmap
                feats.append(fmap.mean(dim=2))          # global avg pool -> (B, hidden)
            return self.head(torch.cat(feats, dim=1))

        @torch.no_grad()
        def relevant_segment(self, x, window_samples: int) -> tuple[int, int]:
            """Coarse [start, end] input-sample span the model attends to (single sample).

            Honest and deliberately simple: the temporal arg-max of the last conv
            feature map (mean |activation| over channels), mapped back to the input
            grid. Not a calibrated attribution — a Phase 1 localization hint only.
            """
            was_training = self.training
            self.eval()
            self.forward(x)
            fmap = next(iter(self._last_maps.values()))     # (B, hidden, L')
            saliency = fmap.abs().mean(dim=1)               # (B, L')
            pos = int(saliency.argmax(dim=1)[0].item())
            length = fmap.shape[2]
            if was_training:
                self.train()
            start = pos * window_samples // length
            end = min(window_samples, (pos + 1) * window_samples // length)
            return start, end

    _CLASSES = (ModalityEncoder, MultimodalClassifier)
    return _CLASSES


def build_model(config):
    """Construct the multimodal classifier for the given ``ModelConfig``.

    Builds one 1-D CNN encoder per modality in ``config.modalities`` and a
    late-fusion classifier head over ``config.num_classes``. For Phase 1
    (``config.ECG_ONLY``) that is a single ECG encoder.
    """
    _, MultimodalClassifier = _torch_classes()
    modality_keys = [m.value for m in config.modalities]
    return MultimodalClassifier(modality_keys, config.num_classes)
