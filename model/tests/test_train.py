"""Tests for train.py helpers (no torch needed — the heavy stack is imported lazily)."""
import numpy as np
import pytest

from biosignal_model.train import _subject_list, fit_temperature


def test_subject_list_parses_csv():
    assert _subject_list("1,2,3") == (1, 2, 3)
    assert _subject_list("14, 15") == (14, 15)
    assert _subject_list("7") == (7,)


def test_subject_list_ignores_blanks():
    assert _subject_list("1, ,2,") == (1, 2)


def test_subject_list_rejects_nonint():
    with pytest.raises(ValueError):
        _subject_list("1,x,3")


# ---- v0.2 temperature calibration (torch-free numpy/scipy) ----
def test_fit_temperature_returns_a_sane_positive_scalar():
    rng = np.random.default_rng(0)
    n, c = 500, 8
    targets = rng.integers(0, c, size=n)
    logits = rng.normal(0, 1, size=(n, c))
    logits[np.arange(n), targets] += 1.0  # roughly calibrated
    t = fit_temperature(logits, targets)
    assert 0.3 < t < 3.0


def test_fit_temperature_softens_overconfident_logits():
    rng = np.random.default_rng(1)
    n, c = 500, 8
    targets = rng.integers(0, c, size=n)
    base = rng.normal(0, 1, size=(n, c))
    base[np.arange(n), targets] += 1.0
    # scaling logits up makes the model over-confident -> optimal T grows
    t_over = fit_temperature(base * 5.0, targets)
    assert t_over > fit_temperature(base, targets)
    assert t_over > 1.0
