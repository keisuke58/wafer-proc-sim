"""Tests for keyence/im_dimension — one-shot image dimension measurement."""
import numpy as np
import pytest

im = pytest.importorskip("keyence.im_dimension")


def _dims(pose, seed):
    rng = np.random.default_rng(seed)
    return im.measure(im.render(pose, rng))


def test_one_shot_accuracy_within_drawing_tolerance():
    d = _dims((0.8, -0.5, np.deg2rad(3.0)), seed=0)
    assert abs(d["dia_A_mm"] - 6.0) * 1000 < 3.0
    assert abs(d["dia_B_mm"] - 6.0) * 1000 < 3.0
    assert abs(d["pitch_mm"] - 12.0) * 1000 < 5.0
    assert abs(d["width_mm"] - 14.0) * 1000 < 10.0


def test_measurement_is_placement_invariant():
    """The whole product concept: wildly different placements must give the
    same dimensions (measured in the recovered part frame)."""
    a = _dims((-2.5, 1.8, np.deg2rad(-4.5)), seed=1)
    b = _dims((2.8, -1.6, np.deg2rad(4.5)), seed=2)
    assert abs(a["dia_A_mm"] - b["dia_A_mm"]) * 1000 < 2.0
    assert abs(a["pitch_mm"] - b["pitch_mm"]) * 1000 < 3.0
    assert abs(a["width_mm"] - b["width_mm"]) * 1000 < 8.0


def test_subpixel_repeatability_beats_caliper():
    rep = im.repeatability(n=8, seed=5)
    cal = im.caliper_study(seed=6)
    sig_dia_um = rep[:, 0].std(ddof=1) * 1000
    assert sig_dia_um < 2.0                      # sub-pixel: ~1/30 px
    assert cal["dia_A_mm"].std(ddof=1) * 1000 > 5 * sig_dia_um


def test_go_ng_flags_out_of_tolerance():
    d = _dims((0.0, 0.0, 0.0), seed=3)
    gn = im.go_ng(d)
    assert all(v["go"] for v in gn.values())
    d_bad = dict(d)
    d_bad["dia_A_mm"] = 6.05                     # 50 µm oversize
    assert not im.go_ng(d_bad)["dia_A_mm"]["go"]
