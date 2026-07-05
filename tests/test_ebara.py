"""Tests for ebara/ — dry-pump pumpdown physics + pattern-density CMP."""
import numpy as np
import pytest

vp = pytest.importorskip("ebara.vacuum_pump")
cp = pytest.importorskip("ebara.cmp_pattern")


class TestPumpdown:
    def test_matches_analytic_exponential(self):
        """Constant S, no outgassing: t(p0->p) = V/S ln(p0/p)."""
        t, p = vp.pumpdown(with_outgassing=False, p_ult0=0.0,
                           t_end_s=600.0)
        t10 = vp.time_to(t, p, 10.0)
        t_ana = vp.V_CHAMBER / vp.S0 * np.log(vp.P_ATM / 10.0)
        assert abs(t10 - t_ana) / t_ana < 0.02

    def test_ultimate_is_outgassing_floor_over_speed(self):
        """24 h equilibrium ~= p_ult0 + Q(24h)/S0."""
        t, p = vp.pumpdown()
        expect = vp.ultimate_pressure()
        assert abs(p[-1] - expect) / expect < 0.10

    def test_dry_vent_beats_bigger_pump(self):
        """Time to the 1 Pa crossover: pump x2 buys ~2x, dry-N2 vent far
        more (until the volume-gas limit takes over)."""
        t, p_wet = vp.pumpdown()
        _, p_2x = vp.pumpdown(s0=2 * vp.S0)
        _, p_dry = vp.pumpdown(dry_vent=True)
        t_wet = vp.time_to(t, p_wet, vp.P_CROSS)
        t_2x = vp.time_to(t, p_2x, vp.P_CROSS)
        t_dry = vp.time_to(t, p_dry, vp.P_CROSS)
        assert 1.5 < t_wet / t_2x < 2.5
        assert t_wet / t_dry > 4.0

    def test_speed_curve_collapses_at_ultimate(self):
        assert vp.pump_speed(vp.P_ULT0) == 0.0
        assert vp.pump_speed(1e4) > 0.99 * vp.S0


class TestCmpPattern:
    def test_uniform_density_polishes_uniformly(self):
        _, rho = cp.layout_density(blocks=[0.5] * 5)
        reff = cp.effective_density(rho)
        rem = cp.polish(reff, 3.0)
        assert np.ptp(rem) < 1e-9

    def test_clear_time_proportional_to_density(self):
        _, r1 = cp.layout_density(blocks=[0.3] * 5)
        _, r2 = cp.layout_density(blocks=[0.6] * 5)
        t1 = cp.clear_times(cp.effective_density(r1)).max() - \
            cp.H_OVER / cp.R_BLANKET
        t2 = cp.clear_times(cp.effective_density(r2)).max() - \
            cp.H_OVER / cp.R_BLANKET
        assert abs(t2 / t1 - 2.0) < 0.02

    def test_effective_density_is_bounded_and_smoother(self):
        _, rho = cp.layout_density()
        reff = cp.effective_density(rho)
        assert reff.min() >= rho.min() - 1e-9
        assert reff.max() <= rho.max() + 1e-9
        assert reff.std() < rho.std()

    def test_blanket_rate_after_landing(self):
        """Once the step is closed, removal advances at the blanket rate."""
        _, rho = cp.layout_density(blocks=[0.5] * 5)
        reff = cp.effective_density(rho)
        t_land = float(cp.clear_times(reff).min()) - \
            cp.H_OVER / cp.R_BLANKET
        d = cp.polish(reff, t_land + 1.0) - cp.polish(reff, t_land)
        assert np.allclose(d, cp.R_BLANKET, rtol=1e-6)

    def test_longer_planarization_length_smooths_more(self):
        res = cp.run()
        s = res["range_vs_Lpl_nm"]
        assert s["L_pl_0.5mm"] > s["L_pl_1.5mm"] > s["L_pl_3.0mm"]
