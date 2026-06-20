"""
Tests for process-embedded SHM dicing monitor (insitu/).

Claims under test: deeper-damage cuts radiate more AE energy; the self-baselining
Mahalanobis detector alarms on the blade's catastrophic chip with positive lead
time while staying quiet on the dry low-damage routes; features are finite.
"""
from __future__ import annotations

import numpy as np
import pytest

from quantum.coherence import CANONICAL_SIGNATURES, ProcessSignature
from insitu import (
    simulate_ae, window_features, MahaDetector, monitor, monitor_all, FEATURE_NAMES,
)


def test_ae_shapes_and_finite():
    tr = simulate_ae(CANONICAL_SIGNATURES["blade"], seed=1)
    assert tr.x.shape == tr.t.shape and np.all(np.isfinite(tr.x))
    feats, times = window_features(tr)
    assert feats.shape[1] == len(FEATURE_NAMES)
    assert feats.shape[0] == times.shape[0] and np.all(np.isfinite(feats))


def test_deeper_damage_more_ae_energy():
    shallow = monitor(ProcessSignature("x", ssd_um=0.5, debris_um=0.0, wet=False), seed=2)
    deep = monitor(ProcessSignature("x", ssd_um=10.0, debris_um=12.0, wet=True), seed=2)
    assert deep.cum_energy[-1] > shallow.cum_energy[-1]


def test_blade_has_catastrophe_dry_routes_do_not():
    res = monitor_all(seed=3)
    assert res["blade"].catastrophe_s is not None
    assert res["stealth"].catastrophe_s is None
    assert res["plasma"].catastrophe_s is None


def test_blade_detected_with_positive_lead_time():
    r = monitor(CANONICAL_SIGNATURES["blade"], seed=4)
    assert r.lead_time_s is not None and r.lead_time_s > 0.0
    # the catastrophic crack window should be caught
    assert r.detection_rate == r.detection_rate  # not NaN
    assert r.detection_rate > 0.0


def test_dry_route_low_false_alarms():
    r = monitor(CANONICAL_SIGNATURES["stealth"], seed=5)
    # stealth stays near baseline → few false alarms (allow a small rate)
    assert (r.false_pos_rate != r.false_pos_rate) or (r.false_pos_rate < 0.2)


def test_blade_alarms_more_than_stealth():
    res = monitor_all(seed=6)
    assert res["blade"].alarms.sum() >= res["stealth"].alarms.sum()


def test_maha_detector_baseline_quiet():
    tr = simulate_ae(CANONICAL_SIGNATURES["stealth"], seed=7)
    feats, _ = window_features(tr)
    det = MahaDetector().fit(feats)
    d2 = det.distance2(feats[: det.n_baseline])
    # baseline windows should sit below the χ² threshold on average
    assert np.mean(d2) < det.thresh_


def test_reproducible_with_seed():
    a = simulate_ae(CANONICAL_SIGNATURES["blade"], seed=42)
    b = simulate_ae(CANONICAL_SIGNATURES["blade"], seed=42)
    assert np.allclose(a.x, b.x)
