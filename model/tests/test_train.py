"""Tests for train.py CLI helpers (no torch needed — the heavy stack is imported lazily)."""
import pytest

from biosignal_model.train import _subject_list


def test_subject_list_parses_csv():
    assert _subject_list("1,2,3") == (1, 2, 3)
    assert _subject_list("14, 15") == (14, 15)
    assert _subject_list("7") == (7,)


def test_subject_list_ignores_blanks():
    assert _subject_list("1, ,2,") == (1, 2)


def test_subject_list_rejects_nonint():
    with pytest.raises(ValueError):
        _subject_list("1,x,3")
