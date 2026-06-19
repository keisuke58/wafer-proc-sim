"""
Tests for the 2nm feasibility constraints + die-strength model.
"""
from __future__ import annotations

import pytest

from sic.hybrid import Recipe, StageOutcome, Constraints2nm


def _groove(lowk_cleared=5.0, haz=1.0, chip=0.5):
    return StageOutcome(stage="A", method="laser_groove", kerf_um=8.0,
                        chipping_um=chip, haz_um=haz, subsurface_crack_um=0.5,
                        residual_stress_MPa=50.0, lowk_cleared_um=lowk_cleared,
                        throughput_wph=30.0)


def _singulation(chip=0.5, crack=0.2, haz=0.0, method="stealth", street_open=True):
    return StageOutcome(stage="B", method=method, kerf_um=2.0, chipping_um=chip,
                        haz_um=haz, subsurface_crack_um=crack,
                        residual_stress_MPa=10.0, throughput_wph=25.0,
                        extra={"street_open": street_open, "ar_feasible": True})


def test_die_strength_decreases_with_crack():
    c = Constraints2nm()
    # In the strength-reducing regime (cracks larger than the intrinsic flaw),
    # deeper cracks give lower strength. Sub-µm cracks are pristine-limited (capped).
    shallow = c.die_strength_MPa("SiC", 20.0)
    deep = c.die_strength_MPa("SiC", 80.0)
    assert shallow > deep > 0.0
    # And the pristine cap holds for tiny cracks.
    assert c.die_strength_MPa("SiC", 0.1) >= c.die_strength_MPa("SiC", 20.0)


def test_feasible_clean_route():
    c = Constraints2nm()
    rep = c.evaluate(Recipe(), _groove(), _singulation())
    assert rep.feasible, rep.violations
    assert rep.penalty == 0.0
    assert bool(rep) is True


def test_lowk_not_cleared_is_infeasible():
    c = Constraints2nm()
    rep = c.evaluate(Recipe(lowk_stack_um=3.0), _groove(lowk_cleared=1.0), _singulation())
    assert not rep.feasible
    assert any("low-k" in v for v in rep.violations)
    assert rep.penalty > 0.0


def test_excess_haz_flags_delamination():
    c = Constraints2nm(lowk_haz_max_um=2.0)
    rep = c.evaluate(Recipe(), _groove(haz=6.0), _singulation())
    assert not rep.feasible
    assert any("HAZ" in v for v in rep.violations)


def test_excess_chipping_flagged():
    c = Constraints2nm()
    rep = c.evaluate(Recipe(street_width_um=25.0), _groove(chip=0.5),
                     _singulation(chip=20.0))
    assert not rep.feasible
    assert any("chipping" in v for v in rep.violations)


def test_plasma_blocked_street_infeasible():
    c = Constraints2nm()
    rep = c.evaluate(Recipe(), _groove(),
                     _singulation(method="plasma", street_open=False))
    assert not rep.feasible
    assert any("plasma street blocked" in v for v in rep.violations)
