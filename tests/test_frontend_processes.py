"""Tests for amat/ (implant+anneal, PVD), kokusai/ (oxidation, batch
furnace) and sumco/ (CZ growth) — the front-end process fill-in."""
import numpy as np
import pytest

ia = pytest.importorskip("amat.implant_anneal")
pv = pytest.importorskip("amat.pvd_sputter")
ox = pytest.importorskip("kokusai.oxidation")
bf = pytest.importorskip("kokusai.batch_furnace")
cz = pytest.importorskip("sumco.cz_growth")


class TestImplantAnneal:
    def test_profile_integral_returns_dose(self):
        x, n = ia.implant_profile("B", 30.0, 1e15)
        dose = np.sum(n) * (x[1] - x[0]) * 1e-21 * 1e14
        assert abs(dose - 1e15) / 1e15 < 0.02      # surface loss only

    def test_fd_diffusion_matches_analytic_gaussian(self):
        x, n0 = ia.implant_profile("B", 30.0, 1e15)
        n_fd = ia.anneal_fd(x, n0, ia.diffusivity(1050.0), 10.0)
        ana = ia.gaussian_after_anneal("B", 30.0, 1e15, x, 1050.0, 10.0)
        core = n_fd > n_fd.max() * 1e-3
        err = np.max(np.abs(n_fd[core] - ana[core])) / ana[core].max()
        assert err < 0.05                          # channel tail excluded

    def test_heavier_ions_stop_shallower(self):
        rp_b = ia.rp_drp("B", 50.0)[0]
        rp_p = ia.rp_drp("P", 50.0)[0]
        rp_as = ia.rp_drp("As", 50.0)[0]
        assert rp_b > rp_p > rp_as

    def test_furnace_drives_junction_deeper_than_rta(self):
        x, n0 = ia.implant_profile("B", 30.0, 1e15)
        xj_rta = ia.junction_depth(
            x, ia.anneal_fd(x, n0, ia.diffusivity(1050.0), 10.0))
        xj_fur = ia.junction_depth(
            x, ia.anneal_fd(x, n0, ia.diffusivity(1000.0), 1800.0))
        assert xj_fur > 1.5 * xj_rta


class TestPvdSputter:
    def test_flat_surface_gets_unit_flux(self):
        x, y = pv.make_trench(0.5)
        f = pv.flux(x, y, 1.0)
        flat = f[x < -1.05 * pv.W_TRENCH]
        assert abs(flat.mean() - 1.0) < 0.01

    def test_coverage_decreases_with_aspect_ratio(self):
        c = [pv.deposit(ar, 1.0)["coverage"] for ar in (0.5, 1.0, 2.0, 3.0)]
        assert all(c[i] > c[i + 1] for i in range(len(c) - 1))

    def test_collimation_improves_coverage(self):
        for ar in (1.0, 3.0):
            assert pv.deposit(ar, 20.0)["coverage"] \
                > pv.deposit(ar, 1.0)["coverage"]


class TestOxidation:
    def test_ode_matches_analytic(self):
        xa = float(ox.x_analytic(1.0, ox.DRY))
        xn = ox.x_ode(1.0, ox.DRY, n=50000)
        assert abs(xn - xa) / xa < 0.001

    def test_linear_and_parabolic_asymptotes(self):
        p = {**ox.DRY, "tau_h": 0.0}
        t = 0.002
        assert abs(float(ox.x_analytic(t, p))
                   - ox.DRY["B_um2h"] / ox.DRY["A_um"] * t) \
            / (ox.DRY["B_um2h"] / ox.DRY["A_um"] * t) < 0.02
        t = 100.0
        assert abs(float(ox.x_analytic(t, ox.WET))
                   - np.sqrt(ox.WET["B_um2h"] * t)) \
            / np.sqrt(ox.WET["B_um2h"] * t) < 0.02

    def test_wet_much_faster_than_dry(self):
        t_wet = ox.time_for(0.5, ox.WET)
        t_dry = ox.time_for(0.5, ox.DRY)
        assert t_dry > 10 * t_wet


class TestBatchFurnace:
    def test_mass_balance(self):
        res = bf.run()
        assert res["mass_balance_err"] < 1e-9

    def test_zero_depletion_is_uniform(self):
        rate, _ = bf.profile(depletion=1e-12)
        assert bf.uniformity(rate) < 1e-6

    def test_ramp_flattens_depletion(self):
        rate0, _ = bf.profile()
        rate1, _ = bf.profile(bf.solve_ramp())
        assert bf.uniformity(rate0) > 10.0
        assert bf.uniformity(rate1) < 0.1


class TestCzGrowth:
    def test_vmax_scales_as_inverse_sqrt_radius(self):
        assert abs(cz.vmax_exponent() + 0.5) < 0.01

    def test_voronkov_classification(self):
        g = 40.0                                   # K/cm
        v_neutral = cz.XI_CRIT * g * 600.0         # mm/h
        assert cz.defect_mode(v_neutral, g) == 0
        assert cz.defect_mode(v_neutral * 1.3, g) == 1     # vacancy
        assert cz.defect_mode(v_neutral * 0.7, g) == -1    # interstitial

    def test_perfect_window_shrinks_with_diameter(self):
        w150 = np.subtract(*cz.perfect_window(0.075)[::-1])
        w300 = np.subtract(*cz.perfect_window(0.150)[::-1])
        assert 0 < w300 < w150
