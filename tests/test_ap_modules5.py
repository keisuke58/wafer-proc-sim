"""Tests for AP試験 fifth wave:
  materials/gan_dicing_kernel.cpp   → GaN基板ダイシング (エピ反り, 方法比較)
  materials/ga2o3_kernel.cpp        → β-Ga₂O₃ (劈開リスク, SiC比較, 材料ベンチマーク)
  materials/inp_dicing_kernel.cpp   → InPフォトニクス (レーザーHAZ, 歩留, IOWN)
"""
from __future__ import annotations
import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ══════════════════════════════════════════════════════════════════════════════
# GaN dicing
# ══════════════════════════════════════════════════════════════════════════════

class TestGaNDicing:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("materials._gan_dicing_kernel",
                             reason="gan_dicing_kernel .so not built")
        from materials._gan_dicing_kernel import (
            gan_substrate_properties, bow_from_epitaxy,
            dicing_method_comparison, gan_market_context,
        )
        self.gan_substrate_properties  = gan_substrate_properties
        self.bow_from_epitaxy          = bow_from_epitaxy
        self.dicing_method_comparison  = dicing_method_comparison
        self.gan_market_context        = gan_market_context

    # --- substrate properties -------------------------------------------------
    def test_substrate_props_keys(self):
        r = self.gan_substrate_properties("GaN-on-Si")
        for k in ("E_substrate_GPa", "K_IC_MPa_sqrtm", "H_GPa",
                  "delta_alpha_ppm", "recommended_method",
                  "blade_score", "stealth_score", "laser_score"):
            assert k in r, f"missing key: {k}"

    def test_substrate_sic_less_cte_mismatch_than_si(self):
        r_si  = self.gan_substrate_properties("GaN-on-Si")
        r_sic = self.gan_substrate_properties("GaN-on-SiC")
        assert r_sic["delta_alpha_ppm"] < r_si["delta_alpha_ppm"], \
            "GaN-on-SiC should have smaller CTE mismatch (SiC CTE closer to GaN)"

    def test_substrate_recommended_method_not_empty(self):
        for sub in ("GaN-on-Si", "GaN-on-SiC", "Freestanding"):
            r = self.gan_substrate_properties(sub)
            assert len(r["recommended_method"]) > 0

    def test_stealth_score_highest_on_si(self):
        r = self.gan_substrate_properties("GaN-on-Si")
        assert r["stealth_score"] > r["blade_score"], \
            "Stealth dicing score should dominate for GaN-on-Si"

    # --- bow from epitaxy -----------------------------------------------------
    def test_bow_keys(self):
        r = self.bow_from_epitaxy("GaN-on-Si", 200.0, 5.0, 1050.0, 25.0)
        for k in ("bow_mm", "sigma_epi_MPa", "R_curvature_m",
                  "delta_alpha_ppm", "handling_risk"):
            assert k in r, f"missing key: {k}"

    def test_bow_positive(self):
        r = self.bow_from_epitaxy("GaN-on-Si", 200.0, 5.0, 1050.0, 25.0)
        assert r["bow_mm"] > 0

    def test_bow_si_greater_than_sic(self):
        r_si  = self.bow_from_epitaxy("GaN-on-Si",  200.0, 5.0, 1050.0, 25.0)
        r_sic = self.bow_from_epitaxy("GaN-on-SiC", 200.0, 5.0, 1050.0, 25.0)
        assert r_si["bow_mm"] > r_sic["bow_mm"], \
            "GaN-on-Si bow > GaN-on-SiC (larger CTE mismatch)"

    def test_bow_larger_wafer_more_bow(self):
        r150 = self.bow_from_epitaxy("GaN-on-Si", 150.0, 5.0, 1050.0, 25.0)
        r200 = self.bow_from_epitaxy("GaN-on-Si", 200.0, 5.0, 1050.0, 25.0)
        assert r200["bow_mm"] > r150["bow_mm"], \
            "Larger diameter -> more bow (L^2 dependence)"

    # --- dicing method comparison ---------------------------------------------
    def test_comparison_keys(self):
        r = self.dicing_method_comparison("GaN-on-Si", 2.0, 80.0, 150.0)
        for k in ("blade", "stealth", "laser"):
            assert k in r

    def test_comparison_yield_in_range(self):
        r = self.dicing_method_comparison("GaN-on-Si", 2.0, 80.0, 150.0)
        for method in ("blade", "stealth", "laser"):
            y = r[method]["yield_est"]
            assert 0.0 < y <= 1.0, f"{method} yield out of range: {y}"

    # --- market context -------------------------------------------------------
    def test_market_context_keys(self):
        r = self.gan_market_context()
        for k in ("rf_market", "power_market", "disco_tools", "growth_driver"):
            assert k in r, f"missing key: {k}"


# ══════════════════════════════════════════════════════════════════════════════
# Ga₂O₃ dicing
# ══════════════════════════════════════════════════════════════════════════════

class TestGa2O3:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("materials._ga2o3_kernel",
                             reason="ga2o3_kernel .so not built")
        from materials._ga2o3_kernel import (
            ga2o3_crystal_properties, cleavage_risk,
            ga2o3_vs_sic_dicing, material_benchmark,
        )
        self.ga2o3_crystal_properties = ga2o3_crystal_properties
        self.cleavage_risk            = cleavage_risk
        self.ga2o3_vs_sic_dicing      = ga2o3_vs_sic_dicing
        self.material_benchmark       = material_benchmark

    # --- crystal properties ---------------------------------------------------
    def test_crystal_props_keys(self):
        r = self.ga2o3_crystal_properties()
        for k in ("bandgap_eV", "E_avg_GPa", "K_IC_100_MPa_sqrtm",
                  "H_GPa", "cleavage_planes", "recommended_method"):
            assert k in r, f"missing key: {k}"

    def test_bandgap_wider_than_sic(self):
        r = self.ga2o3_crystal_properties()
        # SiC bandgap = 3.3 eV, Ga2O3 = 4.8 eV
        assert r["bandgap_eV"] > 3.3, "Ga2O3 bandgap should exceed SiC (3.3 eV)"

    def test_K_IC_lower_than_sic(self):
        r = self.ga2o3_crystal_properties()
        # SiC K_IC ~2.0, Ga2O3 ~0.5
        assert r["K_IC_100_MPa_sqrtm"] < 1.0, "Ga2O3 K_IC(100) should be < 1.0 MPa√m"

    # --- cleavage risk --------------------------------------------------------
    def test_cleavage_risk_keys(self):
        r = self.cleavage_risk(0.0, 100.0, 5.0)
        for k in ("cleavage_risk_score", "risk_grade", "optimal_angle_deg",
                  "recommendation"):
            assert k in r, f"missing key: {k}"

    def test_cleavage_risk_score_range(self):
        r = self.cleavage_risk(30.0, 100.0, 5.0)
        assert 0.0 <= r["cleavage_risk_score"] <= 1.0

    def test_cleavage_risk_parallel_higher_than_perpendicular(self):
        r0  = self.cleavage_risk(0.0, 100.0, 5.0)   # parallel to cleavage
        r90 = self.cleavage_risk(90.0, 100.0, 5.0)  # perpendicular
        assert r0["cleavage_risk_score"] > r90["cleavage_risk_score"], \
            "Street parallel to cleavage plane has higher fracture risk"

    def test_optimal_angle_is_45(self):
        r = self.cleavage_risk(45.0, 100.0, 5.0)
        assert r["optimal_angle_deg"] == pytest.approx(45.0)

    # --- Ga₂O₃ vs SiC dicing -------------------------------------------------
    def test_dicing_comparison_keys(self):
        r = self.ga2o3_vs_sic_dicing(2000.0, 5.0, 45.0)
        assert "comparison" in r
        assert "ga2o3_cleavage_penalty" in r
        assert r["ga2o3_cleavage_penalty"] >= 1.0

    def test_comparison_has_three_materials(self):
        r = self.ga2o3_vs_sic_dicing(2000.0, 5.0, 45.0)
        names = {d["material"] for d in r["comparison"]}
        assert "Si" in names and "SiC" in names and "Ga2O3" in names

    def test_ga2o3_chip_larger_than_sic(self):
        r = self.ga2o3_vs_sic_dicing(2000.0, 5.0, 45.0)
        chips = {d["material"]: d["chip_um_estimated"] for d in r["comparison"]}
        # Ga2O3 K_IC=0.5 vs SiC K_IC=2.0 → Ga2O3 chips MORE
        assert chips["Ga2O3"] > chips["SiC"], \
            "Ga2O3 lower K_IC → larger chipping than SiC"

    # --- material benchmark ---------------------------------------------------
    def test_benchmark_has_all_materials(self):
        r = self.material_benchmark()
        for mat in ("Si", "SiC", "GaN", "Ga2O3", "Diamond"):
            assert mat in r, f"missing material: {mat}"

    def test_diamond_highest_thermal(self):
        r = self.material_benchmark()
        assert r["Diamond"]["thermal_W_mK"] > r["SiC"]["thermal_W_mK"], \
            "Diamond thermal conductivity should exceed SiC"

    def test_ga2o3_highest_bfm(self):
        r = self.material_benchmark()
        # Ga2O3 BFM theoretical > SiC (wide bandgap + high breakdown)
        assert r["Ga2O3"]["BFM_vs_Si"] > r["SiC"]["BFM_vs_Si"]


# ══════════════════════════════════════════════════════════════════════════════
# InP photonic chip dicing
# ══════════════════════════════════════════════════════════════════════════════

class TestInPDicing:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("materials._inp_dicing_kernel",
                             reason="inp_dicing_kernel .so not built")
        from materials._inp_dicing_kernel import (
            inp_photonic_chip_properties, laser_haz_model,
            blade_vs_laser_yield, iown_market_context,
        )
        self.inp_photonic_chip_properties = inp_photonic_chip_properties
        self.laser_haz_model              = laser_haz_model
        self.blade_vs_laser_yield         = blade_vs_laser_yield
        self.iown_market_context          = iown_market_context

    # --- chip properties ------------------------------------------------------
    def test_chip_props_keys(self):
        r = self.inp_photonic_chip_properties()
        for k in ("E_GPa", "K_IC_MPa_sqrtm", "H_GPa",
                  "cleavage_planes", "recommended_method"):
            assert k in r, f"missing key: {k}"

    def test_inp_K_IC_very_low(self):
        r = self.inp_photonic_chip_properties()
        assert r["K_IC_MPa_sqrtm"] < 0.5, "InP K_IC should be < 0.5 MPa√m"

    def test_inp_direct_bandgap(self):
        r = self.inp_photonic_chip_properties()
        assert "Direct" in r["transition_type"]

    # --- laser HAZ model ------------------------------------------------------
    def test_haz_keys(self):
        r = self.laser_haz_model(200.0, 5.0, 100.0, 10.0)
        for k in ("haz_cw_um", "haz_pulsed_um", "kerf_width_um", "quality_grade"):
            assert k in r, f"missing key: {k}"

    def test_pulsed_haz_less_than_cw(self):
        r = self.laser_haz_model(200.0, 5.0, 100.0, 10.0)
        assert r["haz_pulsed_um"] <= r["haz_cw_um"], \
            "Pulsed laser HAZ should not exceed CW HAZ"

    def test_slower_feed_more_haz(self):
        r_fast = self.laser_haz_model(200.0, 20.0, 100.0, 0.0)
        r_slow = self.laser_haz_model(200.0,  1.0, 100.0, 0.0)
        assert r_slow["haz_cw_um"] > r_fast["haz_cw_um"], \
            "Slower feed → longer interaction time → larger HAZ"

    def test_haz_positive(self):
        r = self.laser_haz_model(100.0, 10.0, 100.0, 5.0)
        assert r["haz_pulsed_um"] > 0
        assert r["kerf_width_um"] > 0

    # --- blade vs laser yield -------------------------------------------------
    def test_yield_keys(self):
        r = self.blade_vs_laser_yield(1.0, 40.0, "laser", 100.0)
        for k in ("p_yield_per_chip", "chip_um_estimated",
                  "kerf_um", "recommendation"):
            assert k in r, f"missing key: {k}"

    def test_laser_narrower_kerf_than_blade(self):
        r_laser = self.blade_vs_laser_yield(1.0, 40.0, "laser", 100.0)
        r_blade = self.blade_vs_laser_yield(1.0, 40.0, "blade", 100.0)
        assert r_laser["kerf_um"] < r_blade["kerf_um"], \
            "Laser kerf (~10µm) narrower than blade kerf (street×0.5)"

    def test_yield_in_range(self):
        r = self.blade_vs_laser_yield(1.5, 40.0, "laser", 100.0)
        assert 0.0 < r["p_yield_per_chip"] <= 1.0

    # --- IOWN market context --------------------------------------------------
    def test_iown_keys(self):
        r = self.iown_market_context()
        for k in ("iown_full_name", "inp_role", "chip_type",
                  "disco_role", "demand_driver", "interview_point"):
            assert k in r, f"missing key: {k}"

    def test_iown_mentions_eml(self):
        r = self.iown_market_context()
        assert "EML" in r["chip_type"]

    def test_iown_disco_role_mentions_dfl(self):
        r = self.iown_market_context()
        assert "DFL" in r["disco_role"], \
            "DISCO's InP laser tool (DFL7560) should be mentioned"
