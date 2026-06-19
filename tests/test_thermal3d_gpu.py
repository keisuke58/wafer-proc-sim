"""
Tests for the GPU 3D dicing thermal solver (runs on CPU in CI).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fem.dicing_thermal_3d_gpu import (
    Sim3DConfig, DicingThermal3D, simulate, get_device,
)


def _small(**kw):
    # tiny, fast domain for tests
    base = dict(lx_um=60.0, ly_um=60.0, lz_um=60.0, dx_um=5.0,
                max_steps=1500, speed_mm_s=120.0)
    base.update(kw)
    return Sim3DConfig(**base)


def test_cfl_stability_no_blowup():
    # explicit FD must stay bounded by the ablation cap (no NaN/Inf)
    r = simulate(_small(process="laser", power_W=25.0), keep_field=True)
    T = r.field
    assert np.isfinite(T).all()
    assert T.max() <= Sim3DConfig().t_ablation_C + 1.0
    assert T.min() >= 0.0


def test_no_source_stays_ambient():
    cfg = _small(process="laser", power_W=0.0)
    sim = DicingThermal3D(cfg)
    r = sim.run(keep_field=True)
    assert abs(r.field.max() - cfg.t_amb_C) < 1e-3
    assert r.peak_T_C == pytest.approx(cfg.t_amb_C, abs=1e-2)


def test_groove_deepens_with_power():
    lo = simulate(_small(process="laser", power_W=12.0))
    hi = simulate(_small(process="laser", power_W=40.0))
    # more power → at least as much ablation depth and volume
    assert hi.groove_depth_um >= lo.groove_depth_um
    assert hi.ablated_volume_um3 >= lo.ablated_volume_um3


def test_laser_hotter_than_blade_for_silicon():
    # Si conducts blade friction away → laser drives more thermal damage
    laser = simulate(_small(process="laser", power_W=25.0))
    blade = simulate(_small(process="blade", power_W=25.0))
    assert laser.haz_volume_um3 >= blade.haz_volume_um3


def test_result_kpis_finite_and_nonneg():
    r = simulate(_small(process="blade", power_W=30.0))
    for k, v in r.kpis().items():
        if isinstance(v, (int, float)):
            assert np.isfinite(v) and v >= 0, f"{k}={v}"


def test_device_selection():
    assert get_device("cpu").type == "cpu"
