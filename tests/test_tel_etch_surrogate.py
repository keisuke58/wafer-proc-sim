"""Tests for telproc/etch_surrogate.py — feature-scale etch profile
simulation and the GP surrogate that replaces it."""
import numpy as np
import pytest

es = pytest.importorskip("telproc.etch_surrogate")


BASE = (0.5, 0.8, 1.1, 60.0, 200.0)   # passiv, ion_frac, flux, mask_w, time


class TestEtchPhysics:
    def test_rie_lag_bottom_narrower_than_top(self):
        r = es.simulate_profile(BASE)
        assert r["cd_bot_nm"] < r["cd_top_nm"]      # ARDE / RIE lag

    def test_deeper_with_more_time(self):
        short = es.simulate_profile((0.5, 0.8, 1.1, 60.0, 100.0))
        long = es.simulate_profile((0.5, 0.8, 1.1, 60.0, 240.0))
        assert long["depth_nm"] > short["depth_nm"]

    def test_passivation_suppresses_bow(self):
        bows = [es.simulate_profile((p,) + BASE[1:])["bow_nm"]
                for p in (0.1, 0.5, 0.8)]
        assert bows[0] > bows[1] > bows[2]          # more passiv -> less bow

    def test_passivation_straightens_sidewall(self):
        a = [es.simulate_profile((p,) + BASE[1:])["sidewall_deg"]
             for p in (0.1, 0.8)]
        assert a[1] > a[0]                           # closer to vertical

    def test_profile_arrays_valid(self):
        r = es.simulate_profile(BASE)
        assert r["hw"].min() > 0
        assert np.all(np.diff(r["z"]) > 0)


class TestSurrogate:
    def test_surrogate_accurate_and_fast(self):
        res = es.run()
        # fidelity: the learned surrogate reproduces the physics
        assert res["surrogate_r2"]["cd_bot_nm"] > 0.9
        assert res["surrogate_r2"]["depth_nm"] > 0.9
        # payoff: it is far cheaper to evaluate than the simulation
        assert res["speedup"]["factor"] > 10

    def test_inverse_design_hits_target(self):
        res = es.run()
        assert res["inverse_design"]["cd_err_pct"] < 5.0
        assert res["inverse_design"]["depth_err_pct"] < 5.0
