"""Tests for vision/intelligent_inspection.py (metrology→vision unification)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.intelligent_inspection import (
    GRADE_A_MAX,
    GRADE_B_MAX,
    _chip_to_image_params,
    build_dataset,
    metrology_grade,
)


class TestMetrologyGrade:
    def test_thresholds(self):
        assert metrology_grade(0.3) == 0          # A: < 1 µm
        assert metrology_grade(GRADE_A_MAX) == 1  # boundary → B
        assert metrology_grade(3.0) == 1          # B
        assert metrology_grade(GRADE_B_MAX) == 2  # boundary → C
        assert metrology_grade(8.0) == 2          # C reject

    def test_monotonic(self):
        chips = [0.1, 0.9, 1.1, 4.9, 5.1, 12.0]
        grades = [metrology_grade(c) for c in chips]
        assert grades == sorted(grades)  # non-decreasing in chipping


class TestImageParamMapping:
    def test_severity_increases_with_chipping(self):
        s_lo, _ = _chip_to_image_params(0.3)
        s_hi, _ = _chip_to_image_params(8.0)
        assert s_hi > s_lo
        assert 0.0 <= s_lo <= s_hi <= 0.92

    def test_extent_clamped(self):
        for chip in (0.0, 0.5, 50.0):
            _, ext = _chip_to_image_params(chip)
            assert 3.0 <= ext <= 14.0


class TestDataset:
    def test_shapes_and_grade_span(self):
        X, y, chips, procs = build_dataset(n_per_process=15, H=64, W=64, seed=0)
        n = 15 * 6
        assert X.shape == (n, 64, 64)
        assert y.shape == (n,) and chips.shape == (n,) and procs.shape == (n,)
        assert set(np.unique(procs)) == set(range(6))
        # processes span clean→heavy, so more than one grade must appear
        assert len(set(np.unique(y))) >= 2

    def test_grade_matches_chipping(self):
        _, y, chips, _ = build_dataset(n_per_process=20, H=64, W=64, seed=1)
        for g, c in zip(y, chips):
            assert g == metrology_grade(c)


class TestVisionAgreement:
    @pytest.mark.slow
    def test_cnn_agrees_with_metrology(self):
        pytest.importorskip("torch")
        from sklearn.model_selection import train_test_split

        from vision.intelligent_inspection import train_vision_grader

        X, y, _, _ = build_dataset(n_per_process=100, seed=0)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=0)
        _, m, hist = train_vision_grader(X_tr, y_tr, X_te, y_te,
                                         epochs=15, seed=0)
        assert hist is not None
        assert m["accuracy"] > 0.70, f"agreement acc {m['accuracy']:.2f} too low"
        # safety-critical: reject grade must be caught from the image
        assert m["detection_recall"] > 0.80, \
            f"reject recall {m['detection_recall']:.2f} too low"
