"""Tests for AP試験 sixth wave:
  materials/diamond_kernel.cpp        → ダイヤモンド半導体 (CVD限界, レーザー刻, 熱比較)
  materials/aln_rf_kernel.cpp         → AlN基板 (RF熱モデル, mmWave/DUV市場)
  materials/twod_substrate_kernel.cpp → 2D材料 + サファイアダイシング + 将来ロードマップ
"""
from __future__ import annotations
import sys, os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ══════════════════════════════════════════════════════════════════════════════
# Diamond semiconductor
# ══════════════════════════════════════════════════════════════════════════════

class TestDiamond:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("materials._diamond_kernel",
                             reason="diamond_kernel .so not built")
        from materials._diamond_kernel import (
            diamond_properties, cvd_wafer_limits,
            laser_scribe_diamond, thermal_comparison,
        )
        self.diamond_properties   = diamond_properties
        self.cvd_wafer_limits     = cvd_wafer_limits
        self.laser_scribe_diamond = laser_scribe_diamond
        self.thermal_comparison   = thermal_comparison

    # --- properties -----------------------------------------------------------
    def test_diamond_props_keys(self):
        r = self.diamond_properties()
        for k in ("E_GPa", "H_GPa", "K_IC_MPa_sqrtm",
                  "thermal_conductivity_W_mK", "bandgap_eV",
                  "blade_dicing", "recommended_method"):
            assert k in r, f"missing key: {k}"

    def test_diamond_hardness_highest(self):
        r = self.diamond_properties()
        assert r["mohs_hardness"] == pytest.approx(10.0)
        assert r["H_GPa"] > 50.0, "Diamond Vickers hardness should be > 50 GPa"

    def test_diamond_thermal_conductivity_very_high(self):
        r = self.diamond_properties()
        assert r["thermal_conductivity_W_mK"] >= 1500.0, \
            "Diamond k should be >= 1500 W/mK"

    def test_diamond_bandgap_wider_than_sic(self):
        r = self.diamond_properties()
        assert r["bandgap_eV"] > 3.3, "Diamond bandgap (5.5) must exceed SiC (3.3)"

    def test_blade_dicing_not_feasible(self):
        r = self.diamond_properties()
        assert "IMPOSSIBLE" in r["blade_dicing"].upper() or "impossible" in r["blade_dicing"].lower()

    # --- CVD wafer limits -----------------------------------------------------
    def test_cvd_keys(self):
        r = self.cvd_wafer_limits(10.0, 300.0)
        for k in ("growth_rate_um_h", "growth_time_h",
                  "cost_usd_estimate", "feasibility_2024"):
            assert k in r, f"missing key: {k}"

    def test_cvd_growth_time_positive(self):
        r = self.cvd_wafer_limits(5.0, 500.0)
        assert r["growth_time_h"] > 0

    def test_cvd_large_wafer_not_feasible(self):
        r = self.cvd_wafer_limits(300.0, 500.0)
        assert "Not feasible" in r["feasibility_2024"] or "not" in r["feasibility_2024"].lower()

    def test_cvd_small_single_crystal_feasible(self):
        r = self.cvd_wafer_limits(8.0, 300.0)
        assert "feasible" in r["feasibility_2024"].lower()

    # --- laser scribing -------------------------------------------------------
    def test_laser_keys(self):
        r = self.laser_scribe_diamond(500.0, 355.0, 10.0, 1.0, 300.0)
        for k in ("fluence_J_cm2", "threshold_J_cm2", "ablates",
                  "kerf_um", "quality", "disco_relevance"):
            assert k in r, f"missing key: {k}"

    def test_uv_detected_correctly(self):
        r = self.laser_scribe_diamond(500.0, 355.0, 10.0, 1.0, 300.0)
        assert r["is_uv"] is True

    def test_ir_is_not_uv(self):
        r = self.laser_scribe_diamond(500.0, 1064.0, 0.001, 1.0, 300.0)
        assert r["is_uv"] is False

    # --- thermal comparison ---------------------------------------------------
    def test_thermal_keys(self):
        r = self.thermal_comparison(50.0, 1.0, 300.0)
        assert "results" in r
        assert len(r["results"]) >= 3

    def test_diamond_lowest_temperature_rise(self):
        r = self.thermal_comparison(50.0, 1.0, 300.0)
        temps = {d["substrate"]: d["junction_dT_K"] for d in r["results"]}
        diamond_key = [k for k in temps if "Diamond" in k]
        sic_key     = [k for k in temps if "SiC" in k]
        if diamond_key and sic_key:
            assert temps[diamond_key[0]] < temps[sic_key[0]], \
                "Diamond substrate should give lower junction temperature than SiC"


# ══════════════════════════════════════════════════════════════════════════════
# AlN RF substrate
# ══════════════════════════════════════════════════════════════════════════════

class TestAlNRF:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("materials._aln_rf_kernel",
                             reason="aln_rf_kernel .so not built")
        from materials._aln_rf_kernel import (
            aln_substrate_properties, rf_thermal_model,
            aln_dicing_model, aln_market_context,
        )
        self.aln_substrate_properties = aln_substrate_properties
        self.rf_thermal_model         = rf_thermal_model
        self.aln_dicing_model         = aln_dicing_model
        self.aln_market_context       = aln_market_context

    # --- substrate properties -------------------------------------------------
    def test_props_keys(self):
        r = self.aln_substrate_properties("AlN_single_crystal")
        for k in ("E_GPa", "K_IC_MPa_sqrtm", "H_GPa",
                  "thermal_conductivity_W_mK", "bandgap_eV"):
            assert k in r, f"missing key: {k}"

    def test_single_crystal_bandgap_widest(self):
        r = self.aln_substrate_properties("AlN_single_crystal")
        # AlN bandgap 6.2 eV > GaN 3.4 eV > SiC 3.3 eV
        assert r["bandgap_eV"] > 6.0

    def test_ceramic_has_no_bandgap(self):
        r = self.aln_substrate_properties("AlN_ceramic")
        # ceramic grade: bandgap field is NaN (not a semiconductor)
        assert math.isnan(float(r["bandgap_eV"]))

    def test_three_types_have_different_applications(self):
        apps = set()
        for t in ("AlN_ceramic", "AlN_single_crystal", "AlN_GaN_HEMT"):
            r = self.aln_substrate_properties(t)
            apps.add(r["application"])
        assert len(apps) == 3, "Each substrate type should have a distinct application"

    # --- RF thermal model -----------------------------------------------------
    def test_rf_thermal_keys(self):
        r = self.rf_thermal_model(5.0, 1.0, "GaN-on-SiC")
        assert "results" in r
        assert len(r["results"]) >= 3

    def test_diamond_substrate_coolest(self):
        r = self.rf_thermal_model(5.0, 1.0, "GaN-on-SiC")
        temps = {d["substrate"]: d["T_rise_K"] for d in r["results"]}
        dia_key = [k for k in temps if "Diamond" in k]
        si_key  = [k for k in temps if "Si" in k and "SiC" not in k]
        if dia_key and si_key:
            assert temps[dia_key[0]] < temps[si_key[0]]

    def test_gaN_on_si_hottest(self):
        r = self.rf_thermal_model(5.0, 1.0, "GaN-on-SiC")
        temps = {d["substrate"]: d["T_rise_K"] for d in r["results"]}
        si_key = [k for k in temps if k == "GaN-on-Si"]
        sic_key = [k for k in temps if k == "GaN-on-SiC"]
        if si_key and sic_key:
            assert temps[si_key[0]] > temps[sic_key[0]], \
                "GaN-on-Si should be hotter than GaN-on-SiC"

    # --- AlN dicing model -----------------------------------------------------
    def test_dicing_keys(self):
        r = self.aln_dicing_model(800.0, 5.0, 20.0, 0.635)
        for k in ("Ra_um", "chip_um", "surface_grade", "recommendation"):
            assert k in r, f"missing key: {k}"

    def test_finer_grit_smoother_surface(self):
        r_coarse = self.aln_dicing_model(320.0,  5.0, 20.0, 0.635)
        r_fine   = self.aln_dicing_model(2000.0, 5.0, 20.0, 0.635)
        assert r_fine["Ra_um"] < r_coarse["Ra_um"]

    # --- market context -------------------------------------------------------
    def test_market_context_keys(self):
        r = self.aln_market_context()
        for k in ("duv_led_market", "mmwave_5g", "power_module_substrate",
                  "disco_tools"):
            assert k in r, f"missing key: {k}"


# ══════════════════════════════════════════════════════════════════════════════
# 2D materials + sapphire substrate dicing
# ══════════════════════════════════════════════════════════════════════════════

class TestTwoDSubstrate:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("materials._twod_substrate_kernel",
                             reason="twod_substrate_kernel .so not built")
        from materials._twod_substrate_kernel import (
            twod_material_properties, sapphire_properties,
            sapphire_dicing, future_semiconductor_roadmap,
        )
        self.twod_material_properties   = twod_material_properties
        self.sapphire_properties        = sapphire_properties
        self.sapphire_dicing            = sapphire_dicing
        self.future_semiconductor_roadmap = future_semiconductor_roadmap

    # --- 2D material properties -----------------------------------------------
    def test_graphene_zero_bandgap(self):
        r = self.twod_material_properties("graphene")
        assert r["bandgap_eV"] == pytest.approx(0.0)

    def test_mos2_direct_bandgap(self):
        r = self.twod_material_properties("MoS2")
        assert r["bandgap_eV"] > 1.5, "MoS2 monolayer bandgap should be ~1.8 eV"
        assert "Direct" in r["bandgap_note"] or "direct" in r["bandgap_note"]

    def test_hbn_wide_bandgap(self):
        r = self.twod_material_properties("hBN")
        assert r["bandgap_eV"] >= 5.5, "hBN bandgap should be >= 5.5 eV"

    def test_graphene_high_mobility(self):
        r = self.twod_material_properties("graphene")
        assert r["electron_mobility_cm2_Vs"] > 100000.0, \
            "Graphene mobility > 100,000 cm2/Vs (theoretical)"

    def test_all_materials_have_disco_relevance(self):
        for mat in ("graphene", "MoS2", "hBN", "WS2"):
            r = self.twod_material_properties(mat)
            assert "disco_relevance" in r

    # --- sapphire properties --------------------------------------------------
    def test_sapphire_props_keys(self):
        r = self.sapphire_properties()
        for k in ("E_c_GPa", "H_GPa", "K_IC_MPa_sqrtm",
                  "mohs_hardness", "LED_use"):
            assert k in r, f"missing key: {k}"

    def test_sapphire_mohs_9(self):
        r = self.sapphire_properties()
        assert r["mohs_hardness"] == pytest.approx(9.0)

    def test_sapphire_transparent_uv(self):
        r = self.sapphire_properties()
        assert "UV" in r["transmission_range"] or "0.17" in r["transmission_range"]

    # --- sapphire dicing ------------------------------------------------------
    def test_blade_dicing_keys(self):
        r = self.sapphire_dicing(150.0, 430.0, "blade", 2000.0, 5.0)
        for k in ("chip_um", "Ra_um", "kerf_um"):
            assert k in r, f"missing key: {k}"

    def test_laser_dicing_keys(self):
        r = self.sapphire_dicing(150.0, 430.0, "laser", 2000.0, 5.0)
        for k in ("kerf_um", "scribe_depth_um", "breaking_force_N"):
            assert k in r, f"missing key: {k}"

    def test_laser_narrower_kerf_than_blade(self):
        r_blade = self.sapphire_dicing(150.0, 430.0, "blade", 2000.0, 5.0)
        r_laser = self.sapphire_dicing(150.0, 430.0, "laser", 2000.0, 5.0)
        assert r_laser["kerf_um"] < r_blade["kerf_um"], \
            "Laser scribing gives narrower kerf than blade for sapphire"

    def test_blade_coarse_more_chipping(self):
        r_fine   = self.sapphire_dicing(150.0, 430.0, "blade", 2000.0, 5.0)
        r_coarse = self.sapphire_dicing(150.0, 430.0, "blade",  320.0, 5.0)
        assert r_coarse["chip_um"] > r_fine["chip_um"]

    # --- roadmap --------------------------------------------------------------
    def test_roadmap_has_five_eras(self):
        r = self.future_semiconductor_roadmap()
        era_keys = [k for k in r if k.startswith("era")]
        assert len(era_keys) >= 5

    def test_roadmap_summary_mentions_moat(self):
        r = self.future_semiconductor_roadmap()
        assert "moat" in r["summary"].lower() or "know" in r["summary"].lower()

    def test_roadmap_wbg_era_mentions_sic_gan(self):
        r = self.future_semiconductor_roadmap()
        era2 = r["era2_wbg_power"]
        assert "SiC" in era2["key_material"] or "GaN" in era2["key_material"]
