"""Tests for vision/kerf_quality_classifier.py (small data, fast)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.kerf_quality_classifier import (
    FEATURE_NAMES,
    GRADE_SPECS,
    evaluate_multiclass,
    extract_feature_matrix,
    extract_features,
    make_grade_dataset,
)


# ── dataset generation ────────────────────────────────────────────────────────

class TestGradeDataset:
    def test_shapes_and_dtypes(self):
        X, y = make_grade_dataset(n_per_class=5, H=64, W=64, seed=0)
        assert X.shape == (15, 64, 64)
        assert X.dtype == np.float32
        assert y.shape == (15,)
        assert y.dtype == np.int64

    def test_labels_balanced_and_valid(self):
        X, y = make_grade_dataset(n_per_class=7, H=64, W=64, seed=1)
        assert set(np.unique(y)) == {0, 1, 2}
        for grade in (0, 1, 2):
            assert int((y == grade).sum()) == 7

    def test_pixel_range(self):
        X, _ = make_grade_dataset(n_per_class=4, H=64, W=64, seed=2)
        assert X.min() >= 0.0
        assert X.max() <= 255.0

    def test_reproducible_with_seed(self):
        X1, y1 = make_grade_dataset(n_per_class=3, H=32, W=32, seed=9)
        X2, y2 = make_grade_dataset(n_per_class=3, H=32, W=32, seed=9)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_grade_specs_severity_ordering(self):
        # Grade A < B < C severity ranges must not overlap
        assert GRADE_SPECS[0]["severity"][1] < GRADE_SPECS[1]["severity"][0]
        assert GRADE_SPECS[1]["severity"][1] < GRADE_SPECS[2]["severity"][0]

    def test_grade_c_darker_edges_than_a(self):
        """Heavy chipping (C) should produce more dark edge pixels than A."""
        X, y = make_grade_dataset(n_per_class=10, H=64, W=64, seed=3)
        F = extract_feature_matrix(X)
        dark_frac = F[:, FEATURE_NAMES.index("edge_dark_frac")]
        assert dark_frac[y == 2].mean() > dark_frac[y == 0].mean()


# ── features ──────────────────────────────────────────────────────────────────

class TestFeatures:
    def test_feature_vector_shape_and_finite(self):
        X, _ = make_grade_dataset(n_per_class=2, H=64, W=64, seed=4)
        f = extract_features(X[0])
        assert f.shape == (len(FEATURE_NAMES),)
        assert np.all(np.isfinite(f))

    def test_feature_matrix_shape(self):
        X, _ = make_grade_dataset(n_per_class=3, H=64, W=64, seed=5)
        F = extract_feature_matrix(X)
        assert F.shape == (9, len(FEATURE_NAMES))
        assert np.all(np.isfinite(F))


# ── baseline beats chance level ───────────────────────────────────────────────

class TestBaselinePerformance:
    def test_random_forest_beats_chance(self):
        """3-class chance = 33%; baseline must exceed 70% on small data."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X, y = make_grade_dataset(n_per_class=40, H=64, W=64, seed=0)
        F = extract_feature_matrix(X)
        F_tr, F_te, y_tr, y_te = train_test_split(
            F, y, test_size=0.3, stratify=y, random_state=0)
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(F_tr, y_tr)
        acc = float((rf.predict(F_te) == y_te).mean())
        assert acc > 0.70, f"RF accuracy {acc:.2f} should be well above chance (0.33)"


# ── multi-metric evaluation ───────────────────────────────────────────────────

class TestEvaluation:
    def test_perfect_prediction(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        m = evaluate_multiclass(y, y, "perfect")
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["macro_f1"] == pytest.approx(1.0)
        assert m["defect_f1"] == pytest.approx(1.0)
        assert m["detection_recall"] == pytest.approx(1.0)
        assert m["false_positive_rate"] == pytest.approx(0.0)

    def test_required_metric_keys(self):
        """Multi-metric policy: macro-F1 alone is not enough."""
        y = np.array([0, 1, 2, 0, 1, 2])
        m = evaluate_multiclass(y, y, "m")
        for key in ("accuracy", "macro_f1", "f1_grade_a", "f1_grade_b",
                    "f1_grade_c", "defect_f1", "detection_recall",
                    "false_positive_rate"):
            assert key in m

    def test_all_predicted_defect_has_high_fpr(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.full(6, 2)
        m = evaluate_multiclass(y_true, y_pred, "all_c")
        assert m["detection_recall"] == pytest.approx(1.0)
        assert m["false_positive_rate"] == pytest.approx(1.0)
        assert m["macro_f1"] < 0.5
