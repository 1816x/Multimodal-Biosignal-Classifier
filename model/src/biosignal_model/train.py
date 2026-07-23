"""Training entrypoint for the multimodal biosignal classifier.

Trains on PPG-DaLiA (subject-wise split) and reports honest metrics (accuracy,
per-class precision/recall/F1, confusion matrix). ``--phase`` selects the modality
preset: phase 1 is ECG-only, phase 2 is the full multimodal model (ECG + PPG +
accelerometer). Metrics are documented, not asserted.

Usage:
    python -m biosignal_model.train                  # phase 2 (multimodal), all 15 subjects
    python -m biosignal_model.train --phase 1        # reproduce the ECG-only baseline
    python -m biosignal_model.train --smoke          # fast end-to-end check (few subjects)
    python -m biosignal_model.train --epochs 20 --lr 5e-4

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from . import config as cfg
from .datasets.ppg_dalia import PPGDaLiADataset
from .model import build_model


# --------------------------------------------------------------------------- utils
def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_dataset(data_dir, subjects, model_cfg, stride_seconds):
    return PPGDaLiADataset(data_dir, subjects, model_cfg, stride_seconds=stride_seconds)


def build_splits(data_dir, model_cfg, train_cfg):
    """Load train/val/test (subject-wise) and z-score all three with train-fit stats.

    Normalization stats are per-modality (``{modality: (mean, std)}``), fit on the
    TRAIN split only so no test/val statistics leak into training.
    """
    print(f"Loading train subjects {list(train_cfg.train_subjects)} …", flush=True)
    train = _build_dataset(data_dir, train_cfg.train_subjects, model_cfg, train_cfg.stride_seconds)
    print(f"Loading val subjects {list(train_cfg.val_subjects)} …", flush=True)
    val = _build_dataset(data_dir, train_cfg.val_subjects, model_cfg, train_cfg.stride_seconds)
    print(f"Loading test subjects {list(train_cfg.test_subjects)} …", flush=True)
    test = _build_dataset(data_dir, train_cfg.test_subjects, model_cfg, train_cfg.stride_seconds)

    norm_stats = train.fit_norm_stats()  # fit on TRAIN only — no leakage
    for ds in (train, val, test):
        ds.apply_norm(norm_stats)
    print(f"windows — train {len(train)}, val {len(val)}, test {len(test)}", flush=True)
    return train, val, test, norm_stats


def class_weights(labels, num_classes):
    """Inverse-frequency weights so the imbalanced minority activities still count."""
    import torch

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


# ------------------------------------------------------------------------- metrics
def confusion_matrix(targets, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (targets, preds), 1)
    return cm


def metrics_report(targets, preds, class_names):
    """Accuracy + per-class precision/recall/F1/support + macro/weighted averages."""
    c = len(class_names)
    cm = confusion_matrix(targets, preds, c)
    support = cm.sum(axis=1)
    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total else 0.0

    per_class = {}
    precisions, recalls, f1s = [], [], []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[name] = {"precision": float(p), "recall": float(r), "f1": float(f1),
                           "support": int(support[i])}
        precisions.append(p); recalls.append(r); f1s.append(f1)

    w = support / total if total else np.zeros(c)
    return {
        "accuracy": accuracy,
        "macro_avg": {"precision": float(np.mean(precisions)), "recall": float(np.mean(recalls)),
                      "f1": float(np.mean(f1s))},
        "weighted_avg": {"precision": float(np.sum(np.array(precisions) * w)),
                         "recall": float(np.sum(np.array(recalls) * w)),
                         "f1": float(np.sum(np.array(f1s) * w))},
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(total),
    }


def print_report(title, report, class_names):
    print(f"\n=== {title} ===")
    print(f"accuracy: {report['accuracy']:.4f}   (n={report['n_samples']})")
    print(f"macro    P/R/F1: {report['macro_avg']['precision']:.3f} / "
          f"{report['macro_avg']['recall']:.3f} / {report['macro_avg']['f1']:.3f}")
    print(f"weighted P/R/F1: {report['weighted_avg']['precision']:.3f} / "
          f"{report['weighted_avg']['recall']:.3f} / {report['weighted_avg']['f1']:.3f}")
    print(f"{'class':<14}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}")
    for name in class_names:
        m = report["per_class"][name]
        print(f"{name:<14}{m['precision']:>7.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}{m['support']:>9}")
    print("confusion matrix (rows=true, cols=pred):")
    header = "".join(f"{n[:6]:>7}" for n in class_names)
    print(f"{'':<10}{header}")
    for i, name in enumerate(class_names):
        row = "".join(f"{v:>7}" for v in report["confusion_matrix"][i])
        print(f"{name[:10]:<10}{row}")


# ------------------------------------------------------------------------- training
def _loader(dataset, batch_size, shuffle):
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate(model, loader):
    import torch

    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            all_p.append(logits.argmax(dim=1).cpu().numpy())
            all_t.append(y.cpu().numpy())
    return np.concatenate(all_t), np.concatenate(all_p)


def train_model(model, train_ds, val_ds, train_cfg, class_names):
    import torch
    from torch import nn

    train_loader = _loader(train_ds, train_cfg.batch_size, shuffle=True)
    val_loader = _loader(val_ds, train_cfg.batch_size, shuffle=False)
    weights = class_weights(train_ds.labels, len(class_names))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate,
                                 weight_decay=train_cfg.weight_decay)

    best_val, best_state = -1.0, None
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running, n = 0.0, 0
        t0 = time.time()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            bs = y.shape[0]  # x is a {modality: tensor} dict, so count from the labels
            running += loss.item() * bs
            n += bs
        targets, preds = evaluate(model, val_loader)
        val_acc = float((targets == preds).mean())
        print(f"epoch {epoch:2d}/{train_cfg.epochs}  train_loss={running/n:.4f}  "
              f"val_acc={val_acc:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"best val_acc={best_val:.4f}", flush=True)
    return model, best_val


# ----------------------------------------------------------------------------- CLI
def _subject_list(s: str) -> tuple[int, ...]:
    """Parse a comma-separated subject list like ``1,2,3`` into ``(1, 2, 3)``."""
    return tuple(int(x) for x in s.split(",") if x.strip())


def _apply_smoke(train_cfg):
    """Fast end-to-end validation: few subjects, few epochs (not a real result)."""
    from dataclasses import replace

    return replace(train_cfg, train_subjects=(1, 2), val_subjects=(3,), test_subjects=(4,), epochs=2)


def main() -> None:
    defaults = cfg.PHASE1_TRAIN  # phase 1 and 2 share hyperparameters; only artifacts differ
    parser = argparse.ArgumentParser(description="Train the PPG-DaLiA activity classifier.")
    parser.add_argument("--phase", type=int, default=2, choices=(1, 2),
                        help="1 = ECG-only baseline; 2 = multimodal ECG+PPG+ACC (default)")
    parser.add_argument("--data-dir", default="data", help="DATA_DIR with PPG_FieldStudy/ (default: data)")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--stride-seconds", type=float, default=defaults.stride_seconds)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--out", default=None, help="checkpoint output path (default: per-phase preset)")
    parser.add_argument("--metrics", default=None, help="metrics JSON output path (default: per-phase preset)")
    parser.add_argument("--train-subjects", type=_subject_list, default=None,
                        help="override subject-wise split, e.g. 1,2,3 (default: preset)")
    parser.add_argument("--val-subjects", type=_subject_list, default=None, help="override val subjects")
    parser.add_argument("--test-subjects", type=_subject_list, default=None, help="override test subjects")
    parser.add_argument("--smoke", action="store_true", help="fast pipeline check on a few subjects")
    args = parser.parse_args()

    # Select the modality preset and matching artifact paths from --phase.
    base, model_cfg = (cfg.PHASE2_TRAIN, cfg.MULTIMODAL) if args.phase == 2 else (cfg.PHASE1_TRAIN, cfg.ECG_ONLY)

    from dataclasses import replace
    train_cfg = replace(
        base, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        weight_decay=args.weight_decay, stride_seconds=args.stride_seconds, seed=args.seed,
        checkpoint_path=args.out or base.checkpoint_path, metrics_path=args.metrics or base.metrics_path,
        train_subjects=args.train_subjects or base.train_subjects,
        val_subjects=args.val_subjects or base.val_subjects,
        test_subjects=args.test_subjects or base.test_subjects,
    )
    if args.smoke:
        train_cfg = _apply_smoke(train_cfg)
        print("[smoke] tiny subset / 2 epochs — NOT a reportable result")

    modality_str = "+".join(m.value for m in model_cfg.modalities)
    print(f"Phase {args.phase} — modalities: {modality_str}")
    set_seed(train_cfg.seed)

    import torch
    torch.manual_seed(train_cfg.seed)

    t_start = time.time()
    train_ds, val_ds, test_ds, norm_stats = build_splits(args.data_dir, model_cfg, train_cfg)
    print("train class distribution:", train_ds.class_distribution())

    model = build_model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    model, best_val = train_model(model, train_ds, val_ds, train_cfg, model_cfg.class_names)

    # honest evaluation on the held-out (subject-disjoint) test set
    targets, preds = evaluate(model, _loader(test_ds, train_cfg.batch_size, shuffle=False))
    test_report = metrics_report(targets, preds, list(model_cfg.class_names))
    val_t, val_p = evaluate(model, _loader(val_ds, train_cfg.batch_size, shuffle=False))
    val_report = metrics_report(val_t, val_p, list(model_cfg.class_names))
    print_report("VALIDATION (subjects %s)" % list(train_cfg.val_subjects), val_report, model_cfg.class_names)
    print_report("TEST (held-out subjects %s)" % list(train_cfg.test_subjects), test_report, model_cfg.class_names)

    # ---- persist checkpoint (gitignored) ----
    ckpt_path = Path(train_cfg.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": list(model_cfg.class_names),
            "modalities": [m.value for m in model_cfg.modalities],
            "window_samples": model_cfg.window_samples,
            "target_hz": model_cfg.target_hz,
            # per-modality z-score stats: {modality: {"mean": (C,1), "std": (C,1)}}
            "norm_stats": {k: {"mean": mn, "std": sd} for k, (mn, sd) in norm_stats.items()},
            "test_metrics": test_report,
        },
        ckpt_path,
    )
    print(f"\nsaved checkpoint -> {ckpt_path}")

    # ---- persist metrics JSON (committed as the honest Phase 1 record) ----
    metrics_path = Path(train_cfg.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": args.phase,
        "task": f"PPG-DaLiA activity recognition ({modality_str})",
        "note": "Educational prototype — NOT a medical device. Subject-wise split; metrics not inflated.",
        "modalities": [m.value for m in model_cfg.modalities],
        "target_hz": model_cfg.target_hz,
        "window_seconds": model_cfg.window_seconds,
        "split": {"train": list(train_cfg.train_subjects), "val": list(train_cfg.val_subjects),
                  "test": list(train_cfg.test_subjects)},
        "hyperparams": {"epochs": train_cfg.epochs, "batch_size": train_cfg.batch_size,
                        "lr": train_cfg.learning_rate, "weight_decay": train_cfg.weight_decay,
                        "stride_seconds": train_cfg.stride_seconds, "seed": train_cfg.seed,
                        "class_weighted_loss": True},
        "model_params": int(n_params),
        "best_val_accuracy": float(best_val),
        "validation": val_report,
        "test": test_report,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"saved metrics   -> {metrics_path}")
    print(f"\nTOTAL time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
