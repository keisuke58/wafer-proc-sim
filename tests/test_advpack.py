"""
Tests for the connected advanced-packaging line (thin→warp→bond→singulate).
"""
from __future__ import annotations

import math

import pytest

from advpack import StackConfig, PackagingLine, stoney_bow_um
from advpack.line import STAGES


def test_line_runs_all_stages():
    r = PackagingLine().run(StackConfig())
    assert [s.stage for s in r.states] == ["thin_grind", "warpage", "bond", "singulate"]
    f = r.final
    for v in (f.bow_um, f.overlay_error_nm, f.die_strength_MPa):
        assert math.isfinite(v) and v >= 0


def test_warp_grows_as_wafer_thins():
    # Stoney bow ~ 1/t² : thinner wafer must warp more (before the cap).
    L = PackagingLine()
    thick = L.run(StackConfig(target_thickness_um=700.0)).states[1].bow_um
    thin = L.run(StackConfig(target_thickness_um=200.0)).states[1].bow_um
    assert thin > thick


def test_stoney_quadratic_scaling():
    b1 = stoney_bow_um(50.0, 2.0, 400.0, 300.0, "Si")
    b2 = stoney_bow_um(50.0, 2.0, 200.0, 300.0, "Si")
    # halving thickness → ~4× bow
    assert b2 / b1 == pytest.approx(4.0, rel=0.05)


def test_thick_yields_thin_fails():
    L = PackagingLine()
    thick = L.run(StackConfig(target_thickness_um=775.0, dice_method="laser"))
    thin = L.run(StackConfig(target_thickness_um=50.0, dice_method="laser"))
    assert thick.yielded
    assert not thin.yielded
    assert any("overlay" in v or "void" in v for v in thin.violations)


def test_blade_more_delamination_than_laser():
    blade = PackagingLine().run(StackConfig(target_thickness_um=775.0, dice_method="blade"))
    laser = PackagingLine().run(StackConfig(target_thickness_um=775.0, dice_method="laser"))
    assert blade.final.delam_risk >= laser.final.delam_risk


def test_more_stack_layers_raise_residual():
    s2 = PackagingLine().run(StackConfig(n_stack=2)).final.residual_stress_MPa
    s8 = PackagingLine().run(StackConfig(n_stack=8)).final.residual_stress_MPa
    assert s8 > s2


def test_config_copy():
    c = StackConfig().copy(material="SiC", target_thickness_um=40.0)
    assert c.material == "SiC" and c.target_thickness_um == 40.0
