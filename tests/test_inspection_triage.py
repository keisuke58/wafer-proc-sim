"""Tests for vision/inspection_triage.py (image triage + metrology fusion)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.inspection_triage import (
    CRACK_REJECT_UM,
    PROCESSES,
    sample_crack_um,
    simulate_triage,
)


class TestCrackSampling:
    def test_shape_and_nonneg(self):
        procs = np.array([0, 1, 2, 3, 4, 5] * 4)
        cracks = sample_crack_um(procs, seed=0)
        assert cracks.shape == procs.shape
        assert (cracks >= 0).all()

    def test_blade_cracks_exceed_stealth(self):
        blade = PROCESSES.index("blade")
        stealth = PROCESSES.index("stealth")
        cb = sample_crack_um(np.full(300, blade), seed=0)
        cs = sample_crack_um(np.full(300, stealth), seed=0)
        # blade is a crack-driven reject source; stealth is essentially clean
        assert cb.mean() > CRACK_REJECT_UM > cs.mean()


class TestSimulateTriage:
    def _setup(self):
        # 10 dies: indices 0-3 true reject. CNN flags reject (pred==2) only for 0,1.
        cnn_pred = np.array([2, 2, 0, 1, 0, 0, 0, 0, 0, 0])
        cnn_conf = np.array([.9, .6, .9, .6, .9, .9, .9, .9, .9, .9])
        true_rej = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        return cnn_pred, cnn_conf, true_rej

    def test_never_route_is_cnn_only(self):
        pred, conf, tr = self._setup()
        m = simulate_triage(pred, conf, tr, tau=0.0)
        assert m["fast_fraction"] == pytest.approx(1.0)
        # CNN flags only dies 0,1 of the 4 true rejects → recall 0.5
        assert m["reject_recall"] == pytest.approx(0.5)
        assert m["escape_rate"] == pytest.approx(0.5)

    def test_route_all_is_perfect(self):
        pred, conf, tr = self._setup()
        m = simulate_triage(pred, conf, tr, tau=1.01)
        assert m["fast_fraction"] == pytest.approx(0.0)
        assert m["reject_recall"] == pytest.approx(1.0)
        assert m["escape_rate"] == pytest.approx(0.0)

    def test_routing_recovers_low_confidence_rejects(self):
        pred, conf, tr = self._setup()
        # τ=0.7 routes dies with conf<0.7 (indices 1 and 3) to metrology.
        # die 3 is a true reject CNN missed → recovered; die 1 already flagged.
        m = simulate_triage(pred, conf, tr, tau=0.7)
        assert m["routed_fraction"] == pytest.approx(0.2)
        # recovered die 3 → now 3 of 4 caught (die 2 still missed, conf high)
        assert m["reject_recall"] == pytest.approx(0.75)

    def test_higher_tau_routes_more(self):
        pred, conf, tr = self._setup()
        routed = [simulate_triage(pred, conf, tr, t)["routed_fraction"]
                  for t in (0.0, 0.7, 0.95, 1.01)]
        assert routed == sorted(routed)  # non-decreasing


class TestFullRun:
    @pytest.mark.slow
    def test_fusion_beats_cnn_only_on_escapes(self):
        pytest.importorskip("torch")
        from vision.inspection_triage import run_demo
        stats = run_demo(n_per_process=80, epochs=12, seed=0,
                         out_path=os.path.join(
                             os.path.dirname(__file__), "..", "results",
                             "_test_triage.png"))
        # fusion operating point must not be worse than CNN-only on escapes,
        # while still keeping a meaningful fast fraction
        assert stats["operating_point"]["escape_rate"] <= \
            stats["cnn_only"]["escape_rate"] + 1e-9
        assert stats["operating_point"]["fast_fraction"] > 0.3
