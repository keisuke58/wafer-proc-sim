"""Tests for vision/neu_real_classifier.py (real NEU-CLS images).

The NEU-CLS data lives under data/external/ (gitignored, ~27 MB). When it is
absent these tests skip rather than fail, so CI without the dataset stays green.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.neu_real_classifier import (
    CLASSES,
    DEFAULT_ROOT,
    evaluate,
    _resize,
)

def _detect_data() -> bool:
    """True only if the adapter discovers all 6 NEU categories on disk."""
    if not os.path.isdir(DEFAULT_ROOT):
        return False
    try:
        from vision.neu_steel_adapter import NeuSteelAdapter
        return set(CLASSES).issubset(set(NeuSteelAdapter(DEFAULT_ROOT)
                                         .available_categories))
    except Exception:
        return False


_HAS_DATA = _detect_data()
requires_data = pytest.mark.skipif(not _HAS_DATA,
                                   reason="NEU-CLS not downloaded (data/external)")


# ── helpers that need no data ─────────────────────────────────────────────────

class TestResize:
    def test_resize_shape(self):
        img = np.arange(200 * 200, dtype=np.float32).reshape(200, 200)
        out = _resize(img, 64)
        assert out.shape == (64, 64)
        assert out.dtype == img.dtype

    def test_resize_preserves_range(self):
        img = np.random.default_rng(0).uniform(0, 255, (200, 180)).astype(np.float32)
        out = _resize(img, 96)
        assert out.min() >= img.min() and out.max() <= img.max()


class TestEvaluate:
    def test_perfect(self):
        names = ["a", "b", "c"]
        y = np.array([0, 1, 2, 0, 1, 2])
        m = evaluate(y, y, names, "perfect")
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["macro_f1"] == pytest.approx(1.0)
        assert m["balanced_acc"] == pytest.approx(1.0)
        assert np.trace(np.array(m["confusion"])) == 6

    def test_metric_keys(self):
        names = ["a", "b"]
        y = np.array([0, 1, 0, 1])
        m = evaluate(y, y, names, "m")
        for k in ("accuracy", "macro_f1", "balanced_acc",
                  "per_class_f1", "confusion"):
            assert k in m
        assert set(m["per_class_f1"]) == set(names)


# ── data-dependent ────────────────────────────────────────────────────────────

@requires_data
class TestRealData:
    def test_load_shapes_and_balance(self):
        from vision.neu_real_classifier import load_neu_dataset
        X, y, names = load_neu_dataset(size=64, max_per_class=20)
        assert X.shape == (6 * 20, 64, 64)
        assert X.dtype == np.float32
        assert names == CLASSES
        for c in range(6):
            assert int((y == c).sum()) == 20
        assert 0.0 <= X.min() and X.max() <= 255.0

    @pytest.mark.slow
    def test_cnn_beats_chance_on_real_images(self):
        pytest.importorskip("torch")
        from sklearn.model_selection import train_test_split

        from vision.neu_real_classifier import load_neu_dataset, train_cnn

        X, y, names = load_neu_dataset(size=64, max_per_class=120)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=0)
        m, history = train_cnn(X_tr, y_tr, X_te, y_te, names, epochs=12, seed=0)
        assert history is not None
        # 6-class chance = 0.167; a working CNN must clear 0.55
        assert m["accuracy"] > 0.55, f"real-image CNN acc {m['accuracy']:.2f} too low"
        assert m["macro_f1"] > 0.55
