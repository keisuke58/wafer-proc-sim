"""Tests for disco/plasma_dicing.py (Bosch plasma dicing vs blade) and
disco/blade_rul.py (blade predictive-maintenance digital twin)."""
import numpy as np
import pytest

pd = pytest.importorskip("disco.plasma_dicing")
br = pytest.importorskip("disco.blade_rul")


class TestPlasmaDicing:
    def test_passivation_reduces_scallops(self):
        amps = [pd.bosch_sidewall(80.0, p)[2] for p in (0.2, 0.5, 0.8)]
        assert amps[0] > amps[1] > amps[2]

    def test_passivation_straightens_sidewall(self):
        a = [pd.bosch_sidewall(80.0, p)[3] for p in (0.2, 0.8)]
        assert a[1] >= a[0]                      # closer to 90 deg

    def test_plasma_chipping_far_below_blade(self):
        assert pd.chipping_plasma() < pd.chipping_blade(thickness_um=80.0) / 10

    def test_narrow_kerf_yields_more_dies(self):
        res = pd.run()
        ke = res["kerf_economics"]
        assert ke["dies_plasma"] > ke["dies_blade"]
        assert ke["gain_pct"] > 0


class TestBladeRul:
    def test_chipping_monotone_and_anchored(self):
        assert abs(float(br.chipping_vs_cuts(0)) - br.CHIP0_UM) < 1e-6
        assert abs(float(br.chipping_vs_cuts(br.LIFE0))
                   - br.CHIP_SPEC_UM) < 1e-6
        c = br.chipping_vs_cuts(np.array([1000, 3000, 5000]))
        assert c[0] < c[1] < c[2]

    def test_eol_near_nominal_life(self):
        assert abs(br.eol_cuts() - br.LIFE0) <= 1

    def test_rul_model_accurate(self):
        res = br.run()
        assert res["rul_model"]["test_R2"] > 0.8

    def test_replacement_has_interior_optimum(self):
        res = br.run()
        e = res["replacement_economics"]
        # optimal beats both running to failure and replacing too early
        assert e["optimal_cost_per_good_die"] < e["run_to_failure_cost"]
        assert e["optimal_cost_per_good_die"] <= e["replace_too_early_cost"]
        assert 0 < e["optimal_replace_at_cuts"] < br.LIFE0
