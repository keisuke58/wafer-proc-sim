"""Tests for canon/ — KrF projection imaging + nanoimprint fluid physics."""
import numpy as np
import pytest

li = pytest.importorskip("canon.litho_imaging")
ni = pytest.importorskip("canon.nanoimprint")


class TestLithoImaging:
    def test_relaxed_feature_prints_mask_cd(self):
        """Far from the resolution limit the printed CD tracks the mask CD
        (threshold calibrated at nominal, then checked at a 500 nm pitch)."""
        pitch = 1000.0
        th = li.calibrate_threshold(pitch)
        x, I = li.aerial_image(pitch, 0.0)
        cd = li.measure_cd(x, I, th)
        assert abs(cd - pitch / 2) / (pitch / 2) < 0.02

    def test_bossung_symmetric_in_focus(self):
        """No aberrations => CD(+z) == CD(-z)."""
        pitch = 2 * li.HP_NOM
        th = li.calibrate_threshold(pitch)
        for z in (100.0, 200.0):
            xp, Ip = li.aerial_image(pitch, +z)
            xm, Im = li.aerial_image(pitch, -z)
            cdp = li.measure_cd(xp, Ip, th)
            cdm = li.measure_cd(xm, Im, th)
            assert abs(cdp - cdm) < 0.5

    def test_coherent_cutoff_kills_contrast(self):
        """Pitch below λ/(NA(1+σ)) passes no ±1 orders: contrast ~ 0."""
        p_cut = li.WAVELEN / (li.NA * (1 + 0.7))
        assert li.image_contrast(0.95 * p_cut) < 0.02
        assert li.image_contrast(1.4 * p_cut) > 0.1

    def test_offaxis_beats_conventional_dof(self):
        """At k1=0.35 the matched dipole must widen the process window."""
        pitch = 2 * li.HP_NOM
        dof_c = li.process_window(pitch, "conv")
        dof_d = li.process_window(pitch, "dipole_x")
        assert dof_d > 2.0 * dof_c

    def test_dipole_contrast_highest(self):
        pitch = 2 * li.HP_NOM
        c = {k: li.image_contrast(pitch, k)
             for k in ("conv", "annular", "dipole_x")}
        assert c["dipole_x"] > c["annular"] > c["conv"]


class TestNanoimprint:
    def test_ode_matches_analytic_squeeze_flow(self):
        """RK2 integration vs 1/h = 1/h0 + kt closed form."""
        t, h = ni.spread_ode(200.0, t_end_s=4.0, n=16000)
        ana = ni.spread_analytic(200.0, t)
        assert np.max(np.abs(h - ana) / ana) < 0.001

    def test_fill_time_scales_as_pitch_squared(self):
        r = ni.t_spread(200.0) / ni.t_spread(100.0)
        assert abs(r - 4.0) < 0.01

    def test_film_thins_monotonically(self):
        _, h = ni.spread_ode(150.0, t_end_s=2.0)
        assert np.all(np.diff(h) < 0)

    def test_levers_rank_throughput(self):
        """air -> He ambient -> He + fine drops must be strictly better."""
        w_air, _ = ni.wph_at_spec(200.0, ni.TAU_GAS_AIR_S)
        w_he, _ = ni.wph_at_spec(200.0, ni.TAU_GAS_HE_S)
        w_fine, _ = ni.wph_at_spec(100.0, ni.TAU_GAS_HE_S)
        assert w_fine > w_he > w_air
        assert w_fine / w_air > 2.0
