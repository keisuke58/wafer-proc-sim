"""Tests for the Sony-facing trio: CMOS signal chain, STOP, hybrid AF."""
import numpy as np
import pytest

cmos = pytest.importorskip("optomech.sensor.cmos_signal_chain")
stop = pytest.importorskip("optomech.optics.stop_analysis")
af = pytest.importorskip("optomech.control.af_hybrid")


# ── CMOS signal chain ───────────────────────────────────────────────────────
class TestCmos:
    def test_ptc_recovers_conversion_gain(self):
        """The PTC shot-noise slope must recover the ground-truth system
        gain the simulator was built with (the closed-loop check)."""
        mean_dn, var_t, _ = cmos.photon_transfer_curve()
        k = cmos.estimate_conversion_gain(mean_dn, var_t)
        assert abs(k - cmos.CONV_GAIN_DN_E) / cmos.CONV_GAIN_DN_E < 0.05

    def test_shot_noise_region_slope(self):
        """var vs mean is linear (slope 1 in log-log) where shot dominates."""
        mean_dn, var_t, _ = cmos.photon_transfer_curve()
        read_var = var_t[np.argmin(mean_dn)]
        m = (mean_dn > 30 * read_var) & \
            (mean_dn < 0.4 * cmos.FULL_WELL_E * cmos.CONV_GAIN_DN_E)
        slope = np.polyfit(np.log(mean_dn[m]), np.log(var_t[m]), 1)[0]
        assert 0.9 < slope < 1.1

    def test_hdr_extends_dynamic_range_by_ratio(self):
        _, _, _, _, dr1, dr2 = cmos.hdr_merge_snr(ratio=16.0)
        assert abs((dr2 - dr1) - 20.0 * np.log10(16.0)) < 0.5


# ── STOP chain ──────────────────────────────────────────────────────────────
class TestStop:
    def test_fe_growth_linear_and_matches_reference(self):
        """Al barrel FE sensitivity must match the committed axisymmetric FE
        result (0.968 µm/°C) and be linear in ΔT."""
        E, alpha = stop.MATERIALS["Al 6061"]
        slope, r2, _, _ = stop.fe_sensitivity_um_per_C(E, alpha)
        assert abs(slope - 0.968) < 0.02
        assert r2 > 0.9999

    def test_lower_cte_gives_wider_temperature_window(self):
        _, res = stop.run_stop()
        w = {name: r["window_C"] for name, r in res.items()}
        assert w["Invar 36"] > w["Ti-6Al-4V"] > w["Al 6061"]

    def test_design_meets_spec_at_reference_temperature(self):
        _, res = stop.run_stop()
        for r in res.values():
            assert r["mtf60_at_ref"] >= stop.mw.MTF_SPEC


# ── hybrid AF ───────────────────────────────────────────────────────────────
class TestAfHybrid:
    def test_pdaf_estimate_accurate_in_bright_light(self):
        rng = np.random.default_rng(2)
        errs = []
        for dz in np.linspace(-300, 300, 9):
            est, conf = af.pdaf_measure(af._scene(rng), dz, 2000.0, rng)
            errs.append(est - dz)
            assert conf > 0.8
        assert np.sqrt(np.mean(np.square(errs))) < 10.0

    def test_hybrid_beats_contrast_sweep_in_bright_light(self):
        rng = np.random.default_rng(3)
        t_c, t_h = [], []
        for k in range(5):
            scene = af._scene(rng)
            dz0 = (120.0 + 80.0 * k) * (-1 if k % 2 else 1)
            tc, ec, _ = af.contrast_af(scene, dz0, 2000.0, rng)
            th, eh, _ = af.hybrid_af(scene, dz0, 2000.0, rng)
            t_c.append(tc)
            t_h.append(th)
            assert abs(ec) <= af.DOF_UM
            assert abs(eh) <= af.DOF_UM
        assert np.mean(t_h) < 0.4 * np.mean(t_c)
