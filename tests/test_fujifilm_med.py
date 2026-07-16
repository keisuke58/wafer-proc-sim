"""Tests for fujifilm_med/lesion_cad.py — carrying the wafer-inspection
detectors + FROC evaluation over to synthetic medical lesion detection."""
import numpy as np
import pytest

lc = pytest.importorskip("fujifilm_med.lesion_cad")


class TestImage:
    def test_image_has_lesions_and_mask(self):
        img, mask, cents = lc.make_image(100)
        assert img.shape == (lc.H, lc.W)
        assert mask.any() and len(cents) >= 2
        assert 0.0 <= img.min() and img.max() <= 1.3


class TestAuroc:
    def test_auroc_perfect_separation(self):
        lab = np.zeros((8, 8), bool); lab[0, 0] = True
        sc = lab.astype(float)                    # perfectly aligned score
        assert lc.auroc(sc, lab) == pytest.approx(1.0)


class TestDetectors:
    def test_matched_filter_best_pixel_auroc(self):
        res = lc.run()
        a = res["pixel_auroc"]
        assert a["LoG matched filter"] > a["adaptive multi-scale z"]
        assert a["LoG matched filter"] > a["fixed |I-mean|"]

    def test_matched_filter_best_froc(self):
        res = lc.run()
        s = res["sensitivity_at_1fp_per_image"]
        # the scale-matched filter is the right medical detector; it beats the
        # wafer anomaly detector transferred as-is on the clinical FROC metric
        assert s["LoG matched filter"] > 0.5
        assert s["LoG matched filter"] > s["adaptive multi-scale z"]

    def test_wafer_detector_froc_collapses(self):
        # honest transfer result: good pixel AUROC but poor FROC (fires on
        # tissue texture with no clean reference die)
        res = lc.run()
        assert res["pixel_auroc"]["adaptive multi-scale z"] > 0.85
        assert res["sensitivity_at_1fp_per_image"]["adaptive multi-scale z"] \
            < 0.3
