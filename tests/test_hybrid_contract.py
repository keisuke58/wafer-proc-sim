"""
Contract + smoke tests for the hybrid 2nm dicing stages.

Every DicingStage must return a StageOutcome with finite, non-negative KPIs,
regardless of which solver it wraps. Stage B plasma additionally depends on the
upstream groove being passed in.
"""
from __future__ import annotations

import math

import pytest

from sic.hybrid import (
    Recipe, StageOutcome, LaserGrooveStage, StealthStage, BladeStage, PlasmaStage,
)

ALL_STAGES = [LaserGrooveStage, StealthStage, BladeStage, PlasmaStage]


def _finite_nonneg(o: StageOutcome) -> None:
    for k in ("kerf_um", "chipping_um", "haz_um", "subsurface_crack_um",
              "residual_stress_MPa", "lowk_cleared_um", "throughput_wph"):
        v = getattr(o, k)
        assert isinstance(v, float), f"{k} not float: {v!r}"
        assert math.isfinite(v), f"{k} not finite: {v}"
        assert v >= 0.0, f"{k} negative: {v}"


@pytest.mark.parametrize("StageCls", ALL_STAGES)
def test_stage_returns_valid_outcome(StageCls):
    recipe = Recipe()
    upstream = LaserGrooveStage().simulate(recipe)   # plasma needs a real groove
    out = StageCls().simulate(recipe, upstream=upstream)
    assert isinstance(out, StageOutcome)
    assert out.stage in ("A", "B")
    assert out.method
    _finite_nonneg(out)


def test_groove_clears_some_lowk():
    out = LaserGrooveStage().simulate(Recipe())
    assert out.lowk_cleared_um > 0.0
    assert out.stage == "A"


def test_stealth_low_residual_vs_blade():
    recipe = Recipe()
    groove = LaserGrooveStage().simulate(recipe)
    stealth = StealthStage().simulate(recipe, groove)
    blade = BladeStage().simulate(recipe, groove)
    # Non-contact stealth should leave less residual stress than mechanical blade.
    assert stealth.residual_stress_MPa <= blade.residual_stress_MPa


def test_plasma_blocked_without_groove():
    recipe = Recipe()
    # No upstream → street not opened → plasma reports wall-width "chipping".
    blocked = PlasmaStage().simulate(recipe, upstream=None)
    assert blocked.extra["street_open"] is False
    opened = PlasmaStage().simulate(recipe, LaserGrooveStage().simulate(recipe))
    assert opened.extra["street_open"] is True
    assert opened.chipping_um < blocked.chipping_um


def test_recipe_copy_overrides():
    r = Recipe()
    r2 = r.copy(route="laser+blade", street_width_um=40.0)
    assert r2.route == "laser+blade" and r2.street_width_um == 40.0
    assert r.route == "laser+stealth"  # original untouched
