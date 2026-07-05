"""Tests for optimization/blade_dressing — dressing policy optimization."""
import numpy as np
import pytest

bd = pytest.importorskip("optimization.blade_dressing")


def test_dress_pass_resets_blade_state():
    """After every dress pass the recorded state is back to fresh."""
    hard = bd.make_hardness(1200)
    r = bd.simulate(("fixed", 100), hard)
    assert len(r.dress_at) >= 10
    assert np.allclose(r.sharp[r.dress_at], bd.S_FRESH)
    assert np.allclose(r.load[r.dress_at], bd.L_FRESH)


def test_no_dressing_runs_into_force_runaway():
    """Without dressing the blade glazes until the bond self-sharpens, and
    the high-force excursions produce chipping violations."""
    hard = bd.make_hardness(4000)
    r = bd.simulate(("none",), hard)
    assert len(r.ss_at) > 0                      # self-sharpening happened
    assert r.force.max() > bd.F_CRIT * bd.F0     # force crossed the trigger
    assert r.violations > 0                      # and yield paid for it


def test_dressing_suppresses_violations_and_cost():
    """A sane dressing policy must beat no-dressing on total cost and keep
    chipping essentially inside spec."""
    hard = bd.make_hardness(4000)
    none = bd.simulate(("none",), hard).cost_per_1000()
    cbm = bd.simulate(("cbm", 1.4), hard).cost_per_1000()
    assert cbm["total"] < 0.7 * none["total"]
    assert cbm["violations_per_1000"] < 1.0


def test_cbm_competitive_with_best_fixed_and_robust_to_mix_shift():
    """CBM is at least as good as the oracle-tuned fixed interval on the
    tuning mix, and stays ahead when the product mix shifts."""
    hard = bd.make_hardness(6000)
    sw = bd.sweep(hard)
    best_fixed = sw["fixed"][sw["best_fixed_N"]]["total"]
    best_cbm = sw["cbm"][sw["best_cbm_theta"]]["total"]
    assert best_cbm <= best_fixed * 1.02
    assert sw["shifted"]["cbm"]["total"] <= sw["shifted"]["fixed"]["total"]


def test_exposure_grows_with_dressing_frequency():
    """More dress passes must consume more blade exposure."""
    hard = bd.make_hardness(2000)
    lazy = bd.simulate(("fixed", 400), hard)
    eager = bd.simulate(("fixed", 50), hard)
    assert eager.exposure_used_um > lazy.exposure_used_um
