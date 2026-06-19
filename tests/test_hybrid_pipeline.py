"""
End-to-end tests for the hybrid orchestrator + multi-objective optimizer.
"""
from __future__ import annotations

import math

import pytest

from sic.hybrid import Recipe, HybridDicer, ROUTES
from sic.hybrid.optimize import optimize_route, compare_routes, knee_point


def test_all_routes_run_end_to_end():
    dicer = HybridDicer()
    results = dicer.run_all_routes(Recipe())
    assert set(results) == set(ROUTES)
    for name, res in results.items():
        assert res.throughput_wph > 0.0
        assert math.isfinite(res.feasibility.die_strength_MPa)
        assert res.stage_a.stage == "A" and res.stage_b.stage == "B"
        obj = res.objectives
        assert obj["die_strength_MPa"] > 0 and obj["throughput_wph"] > 0


def test_series_throughput_below_each_stage():
    res = HybridDicer().run(Recipe(route="laser+blade"))
    # series (per-wafer times add) must be slower than either stage alone
    assert res.throughput_wph <= res.stage_a.throughput_wph + 1e-6
    assert res.throughput_wph <= res.stage_b.throughput_wph + 1e-6


def test_unknown_route_raises():
    with pytest.raises(ValueError):
        HybridDicer().run(Recipe(route="laser+sawmagic"))


def test_optimize_route_finds_feasible_and_pareto():
    s = optimize_route("laser+stealth", Recipe(), n_samples=60, seed=1)
    assert s.n_feasible >= 1, "expected at least one feasible stealth recipe"
    assert len(s.pareto) >= 1
    # Pareto front is non-dominated: no point beats another on BOTH objectives
    for a in s.pareto:
        for b in s.pareto:
            if a is b:
                continue
            better_both = (b.die_strength_MPa > a.die_strength_MPa and
                           b.throughput_wph > a.throughput_wph)
            assert not better_both


def test_knee_point_on_pareto():
    s = optimize_route("laser+stealth", Recipe(), n_samples=60, seed=2)
    k = knee_point(s)
    assert k is not None
    assert k in s.pareto


def test_compare_routes_covers_all():
    searches = compare_routes(Recipe(), n_samples=40, seed=3)
    assert set(searches) == set(ROUTES)
    assert sum(s.n_feasible for s in searches.values()) >= 1
