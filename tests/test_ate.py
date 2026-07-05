"""Tests for the Advantest pair: jitter/bathtub + shmoo/March."""
import numpy as np
import pytest

jb = pytest.importorskip("ate.jitter_bathtub")
sm = pytest.importorskip("ate.shmoo_march")


class TestJitter:
    def test_dual_dirac_recovers_injected_rj(self):
        """The closed loop: the Q-scale fit must recover the injected RJ
        sigma within the known dual-Dirac bias (~20 %)."""
        res, _ = jb.analyze(seed=1)
        assert res["rj_error_pct"] < 20.0
        assert 0 < res["eye_open_1e12_ps"] < jb.UI_PS

    def test_prbs7_properties(self):
        b = jb.prbs7(127 * 4)
        # period 127, balanced within 1
        assert np.array_equal(b[:127], b[127:254])
        assert abs(int(b[:127].sum()) - 64) <= 1


class TestShmooMarch:
    def test_fmax_monotonic_in_vdd(self):
        v = np.linspace(0.6, 1.2, 30)
        f = sm.fmax_ghz(v)
        assert np.all(np.diff(f) > 0)

    def test_vmin_consistent_with_fmax(self):
        vmin = sm.vmin_at(1.0)
        assert sm.fmax_ghz(vmin) >= 1.0 > sm.fmax_ghz(vmin - 0.02)

    def test_march_c_catches_all_classes_naive_misses_coupling(self):
        cov = sm.coverage(n_mem=32, n_trials=25, seed=3)
        for k, v in cov.items():
            assert v["march_c"] == 1.0, k
        assert cov["SAF0"]["naive_4n"] == 1.0
        assert cov["CFid"]["naive_4n"] < 0.2
        assert cov["CFin"]["naive_4n"] < 0.8
