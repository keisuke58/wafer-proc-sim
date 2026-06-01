"""
Physics model unit tests — wafer-proc-sim.

Tests verify that physical models produce values consistent with
published literature references. Tolerances reflect measurement
uncertainty in the original experimental data.
"""
import math
import pytest
import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# TEL Process Model: µ_inv calibration
# ════════════════════════════════════════════════════════════════════════════

class TestChannelMobility:
    """
    Matthiessen's rule µ_inv model calibrated to:
    Chung et al. (2001) IEEE Electron Device Lett. — NO anneal 1175°C
    """
    from fem.tel_process_model import channel_mobility_inv, ald_film

    def test_standard_dit_matches_literature(self):
        """Dit = 1e12 cm⁻²eV⁻¹ → µ_inv ≈ 35 cm²/Vs (Kimoto & Cooper 2014)."""
        from fem.tel_process_model import channel_mobility_inv, ald_film
        ald = ald_film(50, 250, "HfO2")
        mu = channel_mobility_inv(1e12, ald["Cox_fF_um2"])
        assert 28 <= mu <= 42, f"Expected 28–42 cm²/Vs, got {mu:.1f}"

    def test_no_anneal_high_dit(self):
        """Dit = 1e13 → µ_inv < 10 cm²/Vs (untreated SiC/SiO₂)."""
        from fem.tel_process_model import channel_mobility_inv, ald_film
        ald = ald_film(50, 250, "HfO2")
        mu = channel_mobility_inv(1e13, ald["Cox_fF_um2"])
        assert mu < 10, f"High-Dit mobility should be < 10 cm²/Vs, got {mu:.1f}"

    def test_no_anneal_low_dit(self):
        """Dit = 1e11 → µ_inv > 55 cm²/Vs (NO anneal treatment)."""
        from fem.tel_process_model import channel_mobility_inv, ald_film
        ald = ald_film(50, 250, "HfO2")
        mu = channel_mobility_inv(1e11, ald["Cox_fF_um2"])
        assert mu > 55, f"Low-Dit mobility should be > 55 cm²/Vs, got {mu:.1f}"

    def test_mobility_monotone_decreasing_with_dit(self):
        """µ_inv must decrease monotonically as Dit increases."""
        from fem.tel_process_model import channel_mobility_inv, ald_film
        ald = ald_film(50, 250, "HfO2")
        dits = [1e10, 1e11, 1e12, 1e13]
        mus = [channel_mobility_inv(d, ald["Cox_fF_um2"]) for d in dits]
        for i in range(len(mus) - 1):
            assert mus[i] > mus[i+1], "µ_inv should decrease with increasing Dit"

    def test_backward_compat_wrapper(self):
        """channel_mobility() wrapper returns same result as channel_mobility_inv()."""
        from fem.tel_process_model import channel_mobility, channel_mobility_inv, ald_film
        ald = ald_film(50, 250, "HfO2")
        assert abs(channel_mobility(1e12, ald["Cox_fF_um2"]) -
                   channel_mobility_inv(1e12, ald["Cox_fF_um2"])) < 0.01


# ════════════════════════════════════════════════════════════════════════════
# Backend Model: Wire Bonding / IMC / Package
# ════════════════════════════════════════════════════════════════════════════

class TestIMCGrowth:
    """
    Arrhenius IMC growth calibrated to:
    Breach et al. (2004) Microelectron. Reliab. 44:973
    """

    def test_au_125c_1000h_matches_breach(self):
        """Au-Al IMC @ 125°C / 1000h → d ≈ 0.29 µm (Breach 2004, ±15%)."""
        from fem.backend_model import imc_thickness
        d = imc_thickness("Au", T_C=125, t_h=1000)
        assert 0.24 <= d <= 0.34, f"Expected 0.24–0.34 µm, got {d:.3f}"

    def test_higher_temperature_grows_faster(self):
        """IMC grows faster at 175°C than 125°C (Arrhenius)."""
        from fem.backend_model import imc_thickness
        d_125 = imc_thickness("Au", T_C=125, t_h=1000)
        d_175 = imc_thickness("Au", T_C=175, t_h=1000)
        assert d_175 > d_125, "Higher temperature should give thicker IMC"

    def test_imc_square_root_time(self):
        """IMC thickness ∝ √t (diffusion-limited growth)."""
        from fem.backend_model import imc_thickness
        d_1000 = imc_thickness("Au", T_C=125, t_h=1000)
        d_4000 = imc_thickness("Au", T_C=125, t_h=4000)
        ratio = d_4000 / d_1000
        assert 1.8 <= ratio <= 2.2, f"√4 ratio should be ~2.0, got {ratio:.2f}"


class TestHeelCrackFatigue:
    """Coffin-Manson calibration: Au @ ΔT=100K → ~50k cycles."""

    def test_au_100k_fatigue_life(self):
        """Au wire @ ΔT=100K should give 30k–100k cycles."""
        from fem.backend_model import heel_crack_life
        N = heel_crack_life("Au", delta_T=100)
        assert 30000 <= N <= 100000, f"Expected 30k–100k, got {N:.0f}"

    def test_cu_stronger_than_au(self):
        """Cu wire should have higher fatigue life than Au at same ΔT."""
        from fem.backend_model import heel_crack_life
        N_au = heel_crack_life("Au", delta_T=100)
        N_cu = heel_crack_life("Cu", delta_T=100)
        assert N_cu > N_au, "Cu should outlast Au in fatigue"

    def test_higher_dt_reduces_life(self):
        """Higher ΔT should reduce fatigue life."""
        from fem.backend_model import heel_crack_life
        N_50  = heel_crack_life("Au", delta_T=50)
        N_150 = heel_crack_life("Au", delta_T=150)
        assert N_50 > N_150, "Higher ΔT should reduce fatigue life"


class TestTimoshenkoWarpage:
    """Timoshenko bimetallic beam warpage."""

    def test_zero_dt_zero_warpage(self):
        """ΔT=0 → warpage = 0."""
        from fem.backend_model import timoshenko_warpage
        result = timoshenko_warpage("4H-SiC", "AlN_sub", delta_T=0)
        assert result["warpage_um"] == pytest.approx(0, abs=0.1)

    def test_warpage_linear_with_dt(self):
        """Warpage should scale approximately linearly with ΔT."""
        from fem.backend_model import timoshenko_warpage
        w50  = timoshenko_warpage("4H-SiC", "Cu_DBC", delta_T=50)["warpage_um"]
        w100 = timoshenko_warpage("4H-SiC", "Cu_DBC", delta_T=100)["warpage_um"]
        ratio = w100 / w50
        assert 1.8 <= ratio <= 2.2, f"Warpage should ~double with 2×ΔT, got {ratio:.2f}"

    def test_high_cte_mismatch_more_warpage(self):
        """Cu DBC (Δα=13ppm/K) should warp more than AlN (Δα=0.5ppm/K)."""
        from fem.backend_model import timoshenko_warpage
        w_aln = timoshenko_warpage("4H-SiC", "AlN_sub", delta_T=100)["warpage_um"]
        w_cu  = timoshenko_warpage("4H-SiC", "Cu_DBC",  delta_T=100)["warpage_um"]
        assert w_cu > w_aln, "Cu DBC (large CTE mismatch) should warp more"


# ════════════════════════════════════════════════════════════════════════════
# Crystal Anisotropy
# ════════════════════════════════════════════════════════════════════════════

class TestCrystalAnisotropy:
    """
    4H-SiC crystal anisotropy calibrated to:
    Optics & Laser Technology (2024) — {10-10} roughness 20% lower than {11-20}
    """

    def test_m_plane_best_quality(self):
        """m-plane (θ=0°) should have best quality (factor=1.0)."""
        from fem.crystal_anisotropy import chipping_factor_vs_angle
        assert chipping_factor_vs_angle(0) == pytest.approx(1.0, abs=0.01)

    def test_a_plane_28pct_worse(self):
        """a-plane (θ=30°) chipping factor ≈ 1.28 (Optics & Laser Tech 2024)."""
        from fem.crystal_anisotropy import chipping_factor_vs_angle
        factor = chipping_factor_vs_angle(30)
        assert 1.20 <= factor <= 1.36, f"Expected 1.20–1.36, got {factor:.3f}"

    def test_roughness_factor_a_plane(self):
        """a-plane roughness factor ≈ 1.20 (paper value)."""
        from fem.crystal_anisotropy import roughness_factor_vs_angle
        r = roughness_factor_vs_angle(30)
        assert 1.15 <= r <= 1.25, f"Expected 1.15–1.25, got {r:.3f}"

    def test_60deg_periodicity(self):
        """4H-SiC has 60° crystal symmetry — K_Ic(0°) ≈ K_Ic(60°)."""
        from fem.crystal_anisotropy import kic_vs_angle
        k0  = kic_vs_angle(0)
        k60 = kic_vs_angle(60)
        assert abs(k0 - k60) < 0.01, f"60° periodicity violated: {k0:.3f} vs {k60:.3f}"

    def test_correction_monotone_from_0_to_30(self):
        """Chipping factor increases from θ=0° to θ=30°."""
        from fem.crystal_anisotropy import chipping_factor_vs_angle
        factors = [chipping_factor_vs_angle(t) for t in range(0, 31, 5)]
        assert factors == sorted(factors), "Factor should increase monotonically 0→30°"


# ════════════════════════════════════════════════════════════════════════════
# TEL Cleaning Model
# ════════════════════════════════════════════════════════════════════════════

class TestCleaningModel:
    """TEL CELLESTA cleaning sequence physics."""

    def test_pregate_sic_carbon_removal(self):
        """Pre-Gate SiC sequence (Piranha+HF+SC2+O₃+HF) should remove >99% carbon."""
        from fem.tel_cleaning_model import run_sequence
        r = run_sequence("pregate_sic", "blade")
        assert r["reduction"]["carbon"] > 99.0, \
            f"SiC Pre-Gate should remove >99% carbon, got {r['reduction']['carbon']:.1f}%"

    def test_metal_removal_high(self):
        """All sequences should achieve >99% metal removal."""
        from fem.tel_cleaning_model import run_sequence
        for seq in ["post_cmp", "pregate_sic"]:
            r = run_sequence(seq, "blade")
            assert r["reduction"]["metal"] > 99.0, \
                f"{seq}: metal removal should be >99%, got {r['reduction']['metal']:.1f}%"

    def test_sic_pregate_better_than_si_rca(self):
        """SiC-optimised sequence should reduce Dit more than Si RCA."""
        from fem.tel_cleaning_model import run_sequence, SurfaceState
        def get_dit(seq):
            state = SurfaceState("blade")
            for step in __import__("fem.tel_cleaning_model",
                                   fromlist=["SEQUENCES"]).SEQUENCES[seq]["steps"]:
                state.apply_step(step)
            return state.dit_contribution()
        dit_sic = get_dit("pregate_sic")
        dit_si  = get_dit("pregate_si")
        assert dit_sic < dit_si, "SiC-optimised cleaning should give lower Dit than Si RCA"


# ════════════════════════════════════════════════════════════════════════════
# CZ Growth (Si wafer)
# ════════════════════════════════════════════════════════════════════════════

class TestSiCZGrowth:
    """Voronkov V/G law for Si CZ crystal growth."""

    def test_critical_vg_positive(self):
        """Critical V/G ratio must be positive."""
        from fem.shinetsu_si_wafer_model import cz_critical_vg
        assert cz_critical_vg() > 0

    def test_perfect_zone_exists(self):
        """There should be a pull-rate window giving perfect crystal (V/G near critical)."""
        from fem.shinetsu_si_wafer_model import defect_type
        # Pull rate ~0.4 mm/min should be in or near perfect zone at G=30K/cm
        result = defect_type(0.4, 30)
        # Either perfect or I-rich (but not V-rich with high COP)
        assert result["COP_cm3"] < 1e5, "Low pull rate should give low COP"

    def test_high_pull_rate_gives_cop(self):
        """High pull rate → V-rich → COP dominant."""
        from fem.shinetsu_si_wafer_model import defect_type
        result = defect_type(2.5, 30)
        assert "V-rich" in result["defect_type"], \
            f"High pull rate should be V-rich, got: {result['defect_type']}"

    def test_cmp_roughness_device_grade(self):
        """CMP with fine slurry and moderate MRR should give device-grade Ra."""
        from fem.shinetsu_si_wafer_model import si_cmp_roughness
        result = si_cmp_roughness(MRR_nm_min=30, pressure_kPa=20, slurry_size_nm=30)
        assert result["Ra_nm"] < 0.1, \
            f"Device grade requires Ra < 0.1nm, got {result['Ra_nm']:.4f}"
        assert result["quality"] == "device grade"


# ════════════════════════════════════════════════════════════════════════════
# Die Strength Model: Griffith + Weibull pipeline
# ════════════════════════════════════════════════════════════════════════════

class TestDieStrength:
    """
    Griffith fracture strength + Weibull failure probability pipeline.
    """

    def test_smaller_chipping_gives_higher_strength(self):
        """Smaller chipping → larger σ_f (Griffith: σ ∝ a^-0.5)."""
        from fem.die_strength_model import fracture_strength
        s2  = fracture_strength(2,  "4H-SiC")
        s10 = fracture_strength(10, "4H-SiC")
        assert s2 > s10, "Smaller chipping should give higher fracture strength"

    def test_m_face_stronger_than_a_face(self):
        """m-face (θ=0°) should be stronger than a-face (θ=30°)."""
        from fem.die_strength_model import fracture_strength
        s_m = fracture_strength(5, "4H-SiC", theta_deg=0)
        s_a = fracture_strength(5, "4H-SiC", theta_deg=30)
        assert s_m > s_a, "m-face should be stronger than a-face"

    def test_failure_rate_zero_for_tiny_chipping(self):
        """Near-zero chipping → near-zero failure rate."""
        from fem.die_strength_model import assembly_failure_rate
        r = assembly_failure_rate(0.5, "4H-SiC", 0, sigma_assembly_MPa=80)
        assert r["P_failure_ppm"] < 1.0, \
            f"Tiny chipping should give < 1 PPM failure, got {r['P_failure_ppm']}"

    def test_large_chipping_high_failure(self):
        """Large chipping + a-face → elevated failure rate."""
        from fem.die_strength_model import assembly_failure_rate
        r = assembly_failure_rate(15, "4H-SiC", 30, sigma_assembly_MPa=80)
        assert r["P_failure_ppm"] > 10, \
            f"Large a-face chipping should give > 10 PPM, got {r['P_failure_ppm']}"

    def test_sic_stronger_than_gan(self):
        """SiC (K_Ic=2.8 MPa√m) should give higher σ_f than GaN (K_Ic=0.9)."""
        from fem.die_strength_model import fracture_strength
        s_sic = fracture_strength(5, "4H-SiC")
        s_gan = fracture_strength(5, "GaN")
        assert s_sic > s_gan, "SiC should be stronger than GaN at same chipping"

    def test_safety_margin_definition(self):
        """Safety margin = σ_f / σ_assembly."""
        from fem.die_strength_model import assembly_failure_rate
        r = assembly_failure_rate(2, "4H-SiC", 0, sigma_assembly_MPa=80)
        expected = r["fracture_str_MPa"] / 80
        assert abs(r["safety_margin"] - expected) < 0.1


# ════════════════════════════════════════════════════════════════════════════
# Laser Groove: Bessel beam vs Gaussian
# ════════════════════════════════════════════════════════════════════════════

class TestBesselBeam:
    """Bessel beam stealth dicing should outperform Gaussian."""

    def test_bessel_less_chipping_than_gaussian(self):
        """Bessel beam should produce less surface chipping than Gaussian (IEEE 2024)."""
        from fem.laser_groove_thermal_2d import bessel_beam_stealth, stealth_dicing
        p = {"laser_power_W": 0.5, "scan_speed_mm_s": 300,
             "pulse_freq_kHz": 100, "focal_depth_um": 100,
             "numerical_aperture": 0.65, "n_layers": 3, "pulse_regime": "fs"}
        gauss  = stealth_dicing(p)
        bessel = bessel_beam_stealth(p)
        if gauss["surface_chipping_um"] > 0:
            assert bessel["surface_chipping_um"] < gauss["surface_chipping_um"], \
                "Bessel should produce less chipping than Gaussian"

    def test_bessel_chip_reduction_reported(self):
        """chip_reduction_vs_gaussian_pct should be positive (IEEE 2024: ~40%)."""
        from fem.laser_groove_thermal_2d import bessel_beam_stealth
        p = {"laser_power_W": 0.5, "scan_speed_mm_s": 300,
             "pulse_freq_kHz": 100, "focal_depth_um": 100,
             "numerical_aperture": 0.65, "n_layers": 3, "pulse_regime": "fs"}
        result = bessel_beam_stealth(p)
        assert result["chip_reduction_vs_gaussian_pct"] >= 0, \
            "Bessel should not increase chipping vs Gaussian"
