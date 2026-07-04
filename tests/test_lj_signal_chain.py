"""Tests for keyence/lj_signal_chain — triangulation signal chain."""
import numpy as np
import pytest

lj = pytest.importorskip("keyence.lj_signal_chain")


def test_clean_extraction_subpixel():
    """On a clean bright column every sub-pixel method beats 1 px (4 µm)."""
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(50):
        z = rng.uniform(-100, 100)
        col = lj.render_column(z, rng=rng)
        errs.append(lj.extract_robust(col) - z)
    assert np.sqrt(np.mean(np.square(errs))) < 4.0


def test_robust_rejects_multipath():
    """With a strong multipath lobe the robust extractor must stay locked on
    the true (shortest-path) lobe while the plain centroid is dragged off."""
    rng = np.random.default_rng(1)
    e_rob, e_cen = [], []
    for _ in range(80):
        z = rng.uniform(-100, 100)
        col = lj.render_column(z, reflectance=0.15, multipath=0.45, rng=rng)
        e_rob.append(lj.extract_robust(col) - z)
        e_cen.append(lj.extract_centroid(col) - z)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    assert rms(e_rob) < 3.0
    assert rms(e_rob) < 0.5 * rms(e_cen)


def test_dimensions_within_tolerance():
    """The reference-part scan must yield GO on all toleranced dimensions."""
    x, z_true, z_meas = lj.scan_profile(rng=np.random.default_rng(2))
    meas = lj.measure_dimensions(x, z_meas)
    for key in ("step_height_um", "groove_width_mm", "flatness_um"):
        assert meas[key]["ok"], f"{key} out of tolerance: {meas[key]}"
