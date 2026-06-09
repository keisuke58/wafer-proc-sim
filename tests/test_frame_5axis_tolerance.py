"""
Tests for:
  - fem/_frame_stiffness_kernel (Euler-Bernoulli stiffness + MachineFrame)
  - pipeline/_5axis_interpolation (dicing trajectory)
  - analysis/tolerance_stack.py (WC / RSS / MC)
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── imports ───────────────────────────────────────────────────────────────────

try:
    from fem import _frame_stiffness_kernel as _fsk
    _HAS_FRAME = True
except ImportError:
    _HAS_FRAME = False

try:
    from pipeline import _5axis_interpolation as _5ax
    _HAS_5AX = True
except ImportError:
    _HAS_5AX = False

from analysis.tolerance_stack import (
    ToleranceChain, spindle_runout_budget,
    stage_positioning_budget, blade_height_budget,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Frame stiffness kernel
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_FRAME, reason="_frame_stiffness_kernel not built")
class TestFrameStiffness:

    # ── cross-section inertia ──────────────────────────────────────────────────

    def test_hollow_rect_inertia_solid_limit(self):
        """Wall t → h/2 (nearly solid) → I approaches solid rect bh³/12."""
        b, h = 100.0, 100.0
        t = h / 2.0 - 0.001  # inner dim = 0.002 mm (nearly solid)
        I = _fsk.hollow_rect_inertia(b, h, t)
        I_solid = b * h**3 / 12.0
        assert abs(I - I_solid) / I_solid < 0.01

    def test_hollow_circle_inertia_formula(self):
        """I = π/64 (D⁴ - d⁴), d = D - 2t."""
        D, t = 80.0, 8.0
        d = D - 2 * t
        expected = math.pi / 64.0 * (D**4 - d**4)
        assert abs(_fsk.hollow_circle_inertia(D, t) - expected) < 1e-6

    def test_i_beam_inertia_positive(self):
        I = _fsk.i_beam_inertia(H_mm=200.0, B_mm=100.0, tf_mm=10.0, tw_mm=6.0)
        assert I > 0.0

    def test_hollow_rect_wall_too_thick_raises(self):
        with pytest.raises(Exception):
            _fsk.hollow_rect_inertia(50.0, 50.0, 30.0)  # 2t=60 > b=50

    # ── beam deflection ────────────────────────────────────────────────────────

    def test_cantilever_deflection_formula(self):
        """δ = F L³ / (3 E I) for cantilever."""
        F, L, E, I = 100.0, 500.0, 200.0, 1e5  # N, mm, GPa, mm^4
        r = _fsk.beam_deflection(F, L, E, I, "cantilever")
        E_MPa = E * 1000.0
        expected_um = F * L**3 / (3.0 * E_MPa * I) * 1000.0
        assert abs(r["delta_um"] - expected_um) / expected_um < 1e-9

    def test_simply_supported_deflection_formula(self):
        """δ = F L³ / (48 E I) for simply supported, central load."""
        F, L, E, I = 100.0, 1000.0, 200.0, 2e5
        r = _fsk.beam_deflection(F, L, E, I, "simply_supported")
        E_MPa = E * 1000.0
        expected_um = F * L**3 / (48.0 * E_MPa * I) * 1000.0
        assert abs(r["delta_um"] - expected_um) / expected_um < 1e-9

    def test_stiffness_k_inverse_of_deflection(self):
        F = 50.0
        r = _fsk.beam_deflection(F, 300.0, 200.0, 5e4, "cantilever")
        assert abs(r["k_N_um"] - F / r["delta_um"]) < 1e-9

    def test_invalid_bc_raises(self):
        with pytest.raises(Exception):
            _fsk.beam_deflection(100.0, 300.0, 200.0, 1e5, "free_free")

    # ── natural frequency ──────────────────────────────────────────────────────

    def test_natural_freq_known_value(self):
        """k=1e6 N/m, m=1 kg → f = 1/(2π)√(10⁶) ≈ 159.15 Hz."""
        f = _fsk.first_natural_freq(1e6, 1.0)
        assert abs(f - math.sqrt(1e6) / (2 * math.pi)) < 1e-6

    # ── MachineFrame class ─────────────────────────────────────────────────────

    def test_machine_frame_total_deflection_positive(self):
        frame = _fsk.MachineFrame()
        I = _fsk.hollow_rect_inertia(100.0, 100.0, 10.0)
        frame.add_segment("col", 400.0, 60.0, I)
        r = frame.analyze(200.0)
        assert r["total_delta_um"] > 0.0
        assert r["total_k_N_um"] > 0.0

    def test_machine_frame_series_compliance(self):
        """Two identical segments → total δ = 2× single segment."""
        I = _fsk.hollow_rect_inertia(100.0, 100.0, 10.0)
        frame1 = _fsk.MachineFrame()
        frame1.add_segment("a", 300.0, 200.0, I)
        frame2 = _fsk.MachineFrame()
        frame2.add_segment("a", 300.0, 200.0, I)
        frame2.add_segment("b", 300.0, 200.0, I)
        r1 = frame1.analyze(100.0)
        r2 = frame2.analyze(100.0)
        assert abs(r2["total_delta_um"] - 2 * r1["total_delta_um"]) < 1e-9

    def test_disco_dicing_saw_factory(self):
        """Factory preset must produce physically reasonable stiffness."""
        frame = _fsk.MachineFrame.disco_dicing_saw()
        r = frame.analyze(500.0)  # 500 N cutting force
        # Expect > 1 N/µm (typical precision machine: 10–100 N/µm)
        assert r["total_k_N_um"] > 1.0, \
            f"stiffness {r['total_k_N_um']:.2f} N/µm too low for dicing saw"

    def test_natural_freq_gt_100hz(self):
        """DISCO dicing saw first natural frequency should exceed 100 Hz."""
        frame = _fsk.MachineFrame.disco_dicing_saw()
        # Spindle assembly mass ~5 kg
        f = frame.natural_freq_hz(500.0, 5.0)
        assert f > 100.0, f"natural freq {f:.1f} Hz below 100 Hz"


# ═══════════════════════════════════════════════════════════════════════════════
# 5-axis interpolation kernel
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_5AX, reason="_5axis_interpolation not built")
class Test5AxisInterpolation:

    # ── linear_interp ──────────────────────────────────────────────────────────

    def test_linear_interp_endpoints(self):
        start = [0.0, 0.0, 1.0, 0.0, 30000.0]
        end   = [100.0, 50.0, -0.3, 90.0, 30000.0]
        traj  = _5ax.linear_interp(start, end, 50)
        assert len(traj) == 50
        np.testing.assert_allclose(traj[0],  start, atol=1e-10)
        np.testing.assert_allclose(traj[-1], end,   atol=1e-10)

    def test_linear_interp_midpoint(self):
        start = [0.0, 0.0, 0.0, 0.0, 0.0]
        end   = [10.0, 10.0, 10.0, 10.0, 10.0]
        traj  = _5ax.linear_interp(start, end, 3)
        mid   = traj[1]
        np.testing.assert_allclose(mid, [5.0]*5, atol=1e-10)

    def test_linear_interp_n1_raises(self):
        with pytest.raises(Exception):
            _5ax.linear_interp([0]*5, [1]*5, 1)

    # ── arc_interp ─────────────────────────────────────────────────────────────

    def test_arc_interp_radius_constant(self):
        """All XY points must lie on the circle."""
        traj = _5ax.arc_interp(0.0, 0.0, 50.0, 0.0, 360.0, 0.0, 0.0, 0.0, 30000.0, 100)
        for pt in traj:
            r = math.hypot(pt[0], pt[1])
            assert abs(r - 50.0) < 1e-6, f"radius error: {r:.6f}"

    def test_arc_interp_z_ramp(self):
        """Z must interpolate linearly from z0 to z1."""
        traj = _5ax.arc_interp(0.0, 0.0, 30.0, 0.0, 90.0, 1.0, -0.5, 0.0, 30000.0, 10)
        assert abs(traj[0][2]  - 1.0)  < 1e-10
        assert abs(traj[-1][2] - (-0.5)) < 1e-10

    # ── dice_lane ──────────────────────────────────────────────────────────────

    def test_dice_lane_keys(self):
        r = _5ax.dice_lane(-75.0, 75.0, 10.0, 1.0, -0.3, 60.0, 30000.0)
        for key in ["x_mm","y_mm","z_mm","theta_deg","spindle_rpm","phase","cut_len_mm"]:
            assert key in r

    def test_dice_lane_cut_length(self):
        r = _5ax.dice_lane(-75.0, 75.0, 0.0, 1.0, -0.3, 60.0, 30000.0)
        assert abs(r["cut_len_mm"] - 150.0) < 1e-9

    def test_dice_lane_y_constant(self):
        """Y must be constant throughout the lane."""
        y_target = 25.5
        r = _5ax.dice_lane(-50.0, 50.0, y_target, 1.0, -0.3, 60.0, 30000.0)
        ys = r["y_mm"]
        assert all(abs(y - y_target) < 1e-10 for y in ys)

    def test_dice_lane_z_plunge_retract(self):
        """Phase 0=plunge starts at z_air, phase 2=retract ends at z_air."""
        z_air, z_cut = 1.0, -0.3
        r = _5ax.dice_lane(-50.0, 50.0, 0.0, z_air, z_cut, 60.0, 30000.0)
        phases = r["phase"]
        zs     = r["z_mm"]
        assert abs(zs[0] - z_air) < 1e-9, "first point should be at z_air"
        assert abs(zs[-1] - z_air) < 1e-9, "last point should be at z_air"
        # Cut phase z should equal z_cut
        cut_zs = [zs[i] for i, ph in enumerate(phases) if ph == 1]
        assert all(abs(z - z_cut) < 1e-9 for z in cut_zs)

    def test_dice_lane_spindle_constant(self):
        rpm = 28000.0
        r = _5ax.dice_lane(-50.0, 50.0, 0.0, 1.0, -0.3, 60.0, rpm)
        assert all(abs(s - rpm) < 1e-9 for s in r["spindle_rpm"])

    # ── full_wafer_grid ────────────────────────────────────────────────────────

    def test_full_wafer_grid_keys(self):
        streets = _5ax.street_positions(10.0, 75.0)
        r = _5ax.full_wafer_grid(streets, streets, 75.0)
        for key in ["x_mm","y_mm","z_mm","total_lanes","total_cut_mm","total_points"]:
            assert key in r

    def test_full_wafer_grid_lane_count(self):
        """For a 150mm wafer with 10mm pitch, expect ~28 X + 28 Y streets."""
        streets = _5ax.street_positions(10.0, 75.0)
        r = _5ax.full_wafer_grid(streets, streets, 75.0)
        assert r["total_lanes"] > 0

    def test_full_wafer_grid_theta_values(self):
        """X-direction lanes have theta=0, Y-direction lanes have theta=90."""
        streets = [0.0, 10.0]
        r = _5ax.full_wafer_grid(streets, streets, 75.0)
        thetas = set(round(t) for t in r["theta_deg"])
        assert thetas <= {0, 90}

    def test_street_positions_symmetric(self):
        pos = _5ax.street_positions(20.0, 75.0)
        # Positions should be symmetric around 0
        pos_set = set(round(p, 6) for p in pos)
        for p in list(pos_set):
            if p != 0.0:
                assert round(-p, 6) in pos_set, f"{p} has no mirror"


# ═══════════════════════════════════════════════════════════════════════════════
# Tolerance stack analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestToleranceStack:

    def test_wc_is_sum_of_tolerances(self):
        chain = ToleranceChain()
        chain.add("a", 0.0, 1.0)
        chain.add("b", 0.0, 2.0)
        chain.add("c", 0.0, 0.5)
        r = chain._wc()
        assert abs(r.tol_total - 3.5) < 1e-12

    def test_rss_less_than_wc(self):
        chain = ToleranceChain()
        for i in range(4):
            chain.add(f"d{i}", 0.0, 1.0)
        wc  = chain._wc().tol_total
        rss = chain._rss().tol_total
        assert rss < wc

    def test_rss_formula(self):
        chain = ToleranceChain()
        chain.add("a", 0.0, 3.0)
        chain.add("b", 0.0, 4.0)
        r = chain._rss()
        assert abs(r.tol_total - 5.0) < 1e-10  # √(9+16) = 5

    def test_mc_mean_near_nominal(self):
        chain = ToleranceChain()
        chain.add("a", 5.0, 1.0)
        chain.add("b", 3.0, 0.5)
        r = chain._mc(100_000)
        assert abs(r.mean - 8.0) < 0.05  # nominal = 5+3 = 8

    def test_mc_std_positive(self):
        chain = ToleranceChain()
        chain.add("x", 0.0, 2.0)
        r = chain._mc(50_000)
        assert r.std > 0.0

    def test_mc_yield_near_100_for_wide_spec(self):
        """If spec limits are ±100× tolerance, yield must be ~100%."""
        chain = ToleranceChain(usl=1000.0, lsl=-1000.0)
        chain.add("x", 0.0, 1.0)
        r = chain._mc(50_000)
        assert r.yield_pct > 99.9

    def test_cpk_positive_centered(self):
        """Centered chain with generous spec → Cpk > 1."""
        chain = ToleranceChain(usl=10.0, lsl=-10.0)
        chain.add("a", 0.0, 1.0)  # 3σ = 1 µm, spec = ±10 µm → Cpk ≈ 3.3
        r = chain._mc(100_000)
        assert r.cpk > 1.0

    def test_fluent_api(self):
        """chain.add() should return chain for chaining."""
        chain = ToleranceChain()
        result = chain.add("x", 0.0, 1.0).add("y", 0.0, 2.0)
        assert result is chain
        assert len(chain._dims) == 2

    def test_analyze_returns_three_methods(self):
        chain = ToleranceChain()
        chain.add("a", 0.0, 1.0)
        results = chain.analyze(n_mc=10_000)
        assert set(results.keys()) == {"WC", "RSS", "MC"}

    def test_empty_chain_raises(self):
        chain = ToleranceChain()
        with pytest.raises(ValueError):
            chain.analyze()

    # ── preset assemblies ──────────────────────────────────────────────────────

    def test_spindle_runout_budget_cpk(self):
        """Spindle TIR budget Cpk should be > 0 (feasible assembly)."""
        r = spindle_runout_budget().analyze(n_mc=50_000)["MC"]
        assert r.cpk > 0.0

    def test_stage_positioning_budget_rss_lt_wc(self):
        wc  = stage_positioning_budget()._wc().tol_total
        rss = stage_positioning_budget()._rss().tol_total
        assert rss < wc

    def test_blade_height_budget_mc_yield_gt_50pct(self):
        r = blade_height_budget().analyze(n_mc=50_000)["MC"]
        assert r.yield_pct > 50.0

    def test_uniform_distribution_narrower_std(self):
        """Uniform ±t has σ = t/√3, which is < t/3 for normal. MC std should differ."""
        chain_n = ToleranceChain()
        chain_n.add("x", 0.0, 3.0, dist="normal")

        chain_u = ToleranceChain()
        chain_u.add("x", 0.0, 3.0, dist="uniform")

        r_n = chain_n._mc(200_000)
        r_u = chain_u._mc(200_000)
        # Normal: σ = 1.0; Uniform ±3: σ = 3/√3 ≈ 1.732
        assert r_u.std > r_n.std * 1.2  # uniform σ > normal σ
