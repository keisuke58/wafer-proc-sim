"""Tests for cleaning/spin_clean — single-wafer wet cleaning physics."""
import numpy as np
import pytest

sc = pytest.importorskip("cleaning.spin_clean")


def test_force_scaling_laws():
    """Adhesion ∝ d, drag ∝ d² — the reason small particles are hard."""
    f1 = sc.adhesion_force(30e-9)
    f2 = sc.adhesion_force(60e-9)
    assert np.isclose(f2 / f1, 2.0, rtol=1e-9)
    tau = sc.wall_shear_spin()
    d1 = sc.drag_force(30e-9, tau)
    d2 = sc.drag_force(60e-9, tau)
    assert np.isclose(d2 / d1, 4.0, rtol=1e-9)


def test_pre_monotonic_and_megasonic_helps_small_particles():
    tau = sc.wall_shear_spin()
    spin = sc.removal_efficiency([30.0, 60.0, 100.0, 150.0], tau)
    assert np.all(np.diff(spin) > 0)
    mega30 = sc.removal_efficiency([30.0], tau, megasonic=True)[0]
    assert mega30 > spin[0] + 0.5          # transformative, not incremental


def test_full_undercut_releases_particle():
    tau = sc.wall_shear_spin()
    pre = sc.removal_efficiency([20.0], tau,
                                etch_nm=sc.A_CONTACT * 1e9 + 0.1)[0]
    assert pre == 1.0


def test_collapse_ar_ordering_and_quarter_power_law():
    ar_w = sc.max_aspect_ratio(sc.SIGMA_WATER)
    ar_i = sc.max_aspect_ratio(sc.SIGMA_IPA)
    ar_c = sc.max_aspect_ratio(sc.SIGMA_SCCO2)
    assert ar_c > ar_i > ar_w
    # AR_max ∝ σ^(-1/4)
    assert np.isclose(ar_i / ar_w,
                      (sc.SIGMA_WATER / sc.SIGMA_IPA) ** 0.25, rtol=1e-6)


def test_film_thinning_matches_analytic():
    t, ha, hn = sc.film_thinning()
    m = ha > 1e-3
    assert np.max(np.abs(hn[m] - ha[m]) / ha[m]) < 0.02


def test_kerf_debris_removed_by_spin_alone():
    """The DISCO cross-over claim: micron-scale kerf debris needs no
    megasonic — spin shear alone removes it completely."""
    tau = sc.wall_shear_spin()
    assert sc.removal_efficiency([500.0], tau)[0] > 0.99
    assert sc.removal_efficiency([1000.0], tau)[0] > 0.99


class TestSprayDry:
    def test_damage_ceiling_drops_with_half_pitch(self):
        import cleaning.spray_dry as sd
        assert sd.v_damage(64.0) > sd.v_damage(32.0) > sd.v_damage(16.0)

    def test_window_closes_at_fine_node_for_small_particles(self):
        """The headline: 30nm-particle removal is spray-safe at hp32 but
        NOT at hp16 — damage-free spray runs out of window."""
        import cleaning.spray_dry as sd
        v30 = sd.v_removal(30.0)
        assert v30 < sd.v_damage(32.0)
        assert v30 > sd.v_damage(16.0)

    def test_sublayer_shielding_raises_removal_velocity(self):
        import cleaning.spray_dry as sd
        assert sd.v_removal(30.0) > sd.v_removal(100.0) * 5

    def test_ipa_supply_kills_watermarks(self):
        import cleaning.spray_dry as sd
        wm0 = sd.watermark_density(0.0)
        wm1 = sd.watermark_density(1.0)
        assert wm1 < wm0 / 100.0
