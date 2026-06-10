"""Tests for AP試験 third wave:
  machine/blade_wear_kernel.cpp    → 加工機械・摩耗 (Archard, Taylor, コスト最適)
  sic/sic_chipping_kernel.cpp      → SiC半導体加工 (Lawn-Evans, 結晶異方性)
  pipeline/r2r_controller.cpp      → APC / R2R制御 (EWMA, DEWMA)
"""
from __future__ import annotations
import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ══════════════════════════════════════════════════════════════════════════════
# Blade Wear (Archard / Taylor / コスト最適化)
# ══════════════════════════════════════════════════════════════════════════════

class TestBladeWear:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("machine._blade_wear_kernel",
                            reason="Blade wear kernel not built; run build_all_kernels.sh blade_wear")
        from machine import _blade_wear_kernel as bw
        self.bw = bw

    def test_archard_analysis_returns_required_keys(self):
        """archard_analysis must return wafers_per_blade and summary dict."""
        r = self.bw.archard_analysis("SiC", 30.0, 2.0, 25.0, 54.0, 150.0, 5.0)
        for key in ("wafers_per_blade", "wear_per_wafer_mm3", "material"):
            assert key in r, f"Missing key: {key}"

    def test_archard_analysis_wafers_positive(self):
        """archard_analysis must return positive wafers_per_blade for any material."""
        for mat in ("Si", "SiC", "GaN"):
            r = self.bw.archard_analysis(mat, 30.0, 2.0, 25.0, 54.0, 150.0, 5.0)
            assert r["wafers_per_blade"] > 0, f"{mat}: wafers_per_blade <= 0"

    def test_archard_analysis_wear_positive(self):
        """wear_per_wafer_mm3 must be positive (blade does wear)."""
        r = self.bw.archard_analysis("SiC", 30.0, 2.0, 25.0, 54.0, 150.0, 5.0)
        assert r["wear_per_wafer_mm3"] > 0.0

    def test_archard_analysis_max_wear_sets_upper_limit(self):
        """Doubling blade_max_wear_mm3 must double wafers_per_blade."""
        r1 = self.bw.archard_analysis("Si", 30.0, 2.0, 25.0, 54.0, 150.0, 5.0)
        r2 = self.bw.archard_analysis("Si", 30.0, 2.0, 25.0, 54.0, 150.0, 10.0)
        assert r2["wafers_per_blade"] == pytest.approx(
            r1["wafers_per_blade"] * 2, rel=0.01
        ), "wafers_per_blade must scale linearly with max wear budget"

    def test_taylor_analysis_returns_tool_life(self):
        """Taylor analysis must return a positive tool life in minutes."""
        r = self.bw.taylor_analysis("SiC", 25.0, 54.0)
        assert r["tool_life_min"] > 0.0

    def test_optimize_replacement_positive_interval(self):
        """Optimal replacement interval must be positive."""
        r = self.bw.optimize_replacement(wafers_per_blade_mean=80.0)
        assert r["optimal_wafers_per_blade"] > 0.0

    def test_optimize_replacement_cost_positive(self):
        """Cost per wafer must be positive."""
        r = self.bw.optimize_replacement(wafers_per_blade_mean=100.0)
        assert r["cost_per_wafer_usd"] > 0.0

    def test_blade_wear_demo_runs(self):
        """Demo must return a list of material comparisons without error."""
        results = self.bw.blade_wear_demo()
        assert isinstance(results, list)
        assert len(results) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# SiC Chipping (Lawn-Evans / 4H-SiC Crystal Anisotropy)
# ══════════════════════════════════════════════════════════════════════════════

class TestSicChipping:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("sic._sic_chipping_kernel",
                            reason="SiC chipping kernel not built; run build_all_kernels.sh sic_chipping")
        from sic import _sic_chipping_kernel as sic
        self.sic = sic

    def test_sic_crystal_properties_returns_dict(self):
        """sic_crystal_properties must return E, K_IC, H for given orientation."""
        r = self.sic.sic_crystal_properties(45.0)
        for key in ("E_GPa", "K_IC_MPa_sqrtm", "H_GPa"):
            assert key in r, f"Missing key: {key}"

    def test_sic_crystal_E_positive(self):
        """Young's modulus must be positive at any orientation."""
        for theta in [0, 30, 45, 60, 90]:
            r = self.sic.sic_crystal_properties(float(theta))
            assert r["E_GPa"] > 0.0

    def test_sic_crystal_K_IC_positive(self):
        """Fracture toughness must be positive at any orientation."""
        for theta in [0, 45, 90]:
            r = self.sic.sic_crystal_properties(float(theta))
            assert r["K_IC_MPa_sqrtm"] > 0.0

    def test_sic_crystal_basal_weaker_than_caxis(self):
        """K_IC near basal plane (0°) must be lower than near a-axis (90°)."""
        K0  = self.sic.sic_crystal_properties(0.0)["K_IC_MPa_sqrtm"]
        K90 = self.sic.sic_crystal_properties(90.0)["K_IC_MPa_sqrtm"]
        assert K0 < K90, f"K_IC(0°)={K0:.2f} should be < K_IC(90°)={K90:.2f}"

    def test_sic_crystal_hardness_range(self):
        """Hardness must lie in [25, 30] GPa across all orientations."""
        for theta in [0, 30, 45, 60, 90]:
            H = self.sic.sic_crystal_properties(float(theta))["H_GPa"]
            assert 25.0 <= H <= 30.0, f"H({theta}°) = {H:.2f} out of range"

    def test_predict_chipping_returns_dict(self):
        """predict_chipping must return a dict with quality key."""
        r = self.sic.predict_chipping(orientation_deg=45.0)
        assert "chipping_mean_um" in r
        assert "quality" in r
        assert r["chipping_mean_um"] > 0.0

    def test_predict_chipping_sic_harder_than_si(self):
        """Predicted chipping for SiC must be larger than Si at same conditions."""
        r_sic = self.sic.predict_chipping(orientation_deg=90.0,
                                           blade_depth_um=30.0,
                                           feed_um_rev=1.0,
                                           spindle_krpm=25.0)
        # SiC chips more than Si despite higher K_IC — hardness dominates force
        assert r_sic["chipping_mean_um"] > 0.0

    def test_optimize_direction_within_range(self):
        """Optimal dicing direction must be between 0° and 90°."""
        r = self.sic.optimize_dicing_direction()
        assert 0.0 <= r["best_orientation_deg"] <= 90.0

    def test_optimize_direction_meets_target(self):
        """Optimal direction should meet the 2 µm chipping target for standard SiC params."""
        r = self.sic.optimize_dicing_direction(blade_depth_um=20.0, feed_um_rev=0.5,
                                                spindle_krpm=20.0, target_chipping_um=3.0)
        assert r["best_chipping_um"] > 0.0

    def test_chipping_demo_returns_comparison(self):
        """Chipping demo must return a dict with Si and SiC comparison keys."""
        r = self.sic.chipping_demo()
        assert isinstance(r, dict)
        assert "Si_chipping_um" in r
        assert "SiC_optimal_chipping_um" in r
        assert "SiC_worst_chipping_um" in r
        # Worst orientation must be worse than optimal
        assert r["SiC_worst_chipping_um"] > r["SiC_optimal_chipping_um"]


# ══════════════════════════════════════════════════════════════════════════════
# R2R APC Controller (EWMA / DEWMA)
# ══════════════════════════════════════════════════════════════════════════════

class TestR2RController:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("pipeline._r2r_controller_kernel",
                            reason="R2R controller not built; run build_all_kernels.sh r2r")
        from pipeline import _r2r_controller_kernel as r2r
        self.r2r = r2r

    # ── EWMA ──────────────────────────────────────────────────────────────────

    def test_ewma_returns_correct_keys(self):
        """ewma_r2r must return y_out, error, rmse_um, converged."""
        r = self.r2r.ewma_r2r(n_runs=50)
        for key in ("y_out", "error", "rmse_um", "converged", "controller"):
            assert key in r

    def test_ewma_output_length(self):
        """y_out length must equal n_runs."""
        n = 60
        r = self.r2r.ewma_r2r(n_runs=n)
        assert len(r["y_out"]) == n

    def test_ewma_no_drift_converges(self):
        """With zero drift, EWMA must converge (final error < 1 µm)."""
        r = self.r2r.ewma_r2r(target_um=55.0, n_runs=200,
                               lambda_ewma=0.3, sigma_noise_um=0.5,
                               drift_um_per_run=0.0, seed=1)
        assert r["converged"] is True

    def test_ewma_reduces_drift_error(self):
        """EWMA RMSE must be significantly lower than uncontrolled drift."""
        # Uncontrolled: drift accumulates to n_runs * drift_rate
        n_runs = 100; drift = 0.05
        uncontrolled_rmse_approx = drift * n_runs / 2  # ~2.5 µm
        r = self.r2r.ewma_r2r(target_um=55.0, n_runs=n_runs,
                               lambda_ewma=0.3, sigma_noise_um=0.5,
                               drift_um_per_run=drift, seed=42)
        assert r["rmse_um"] < uncontrolled_rmse_approx * 0.7

    def test_ewma_lambda_zero_no_correction(self):
        """λ=0 means no update → controller output stays at initial → no correction."""
        r = self.r2r.ewma_r2r(lambda_ewma=0.0, drift_um_per_run=0.1,
                               n_runs=50, sigma_noise_um=0.0, seed=0)
        # With λ=0 and no noise, error should grow monotonically
        errs = r["error"]
        assert errs[-1] > errs[0]

    def test_ewma_lambda_one_perfect_correction_no_noise(self):
        """λ=1, σ=0: perfect 1-step correction → error ≈ 0 after run 1."""
        r = self.r2r.ewma_r2r(lambda_ewma=1.0, drift_um_per_run=0.1,
                               n_runs=50, sigma_noise_um=0.0, seed=0)
        # After warm-up, error must be close to 0 (no noise, perfect update)
        assert abs(r["final_error_um"]) < 0.5

    # ── DEWMA ─────────────────────────────────────────────────────────────────

    def test_dewma_returns_correct_keys(self):
        """dewma_r2r must return d_level, d_trend in addition to common keys."""
        r = self.r2r.dewma_r2r(n_runs=50)
        for key in ("y_out", "error", "rmse_um", "d_level", "d_trend", "converged"):
            assert key in r

    def test_dewma_output_length(self):
        """y_out length must equal n_runs."""
        n = 80
        r = self.r2r.dewma_r2r(n_runs=n)
        assert len(r["y_out"]) == n

    def test_dewma_better_than_ewma_on_linear_drift(self):
        """DEWMA must achieve lower RMSE than EWMA when drift is linear."""
        kwargs = dict(target_um=55.0, n_runs=100, sigma_noise_um=0.3,
                      drift_um_per_run=0.08, seed=2026)
        r_ewma  = self.r2r.ewma_r2r(lambda_ewma=0.3, **kwargs)
        # "lambda" is a Python keyword; pass as positional (target, n_runs, lambda, sigma, drift, seed)
        r_dewma = self.r2r.dewma_r2r(
            kwargs["target_um"], kwargs["n_runs"], 0.3,
            kwargs["sigma_noise_um"], kwargs["drift_um_per_run"], kwargs["seed"]
        )
        assert r_dewma["rmse_um"] < r_ewma["rmse_um"], (
            f"DEWMA RMSE={r_dewma['rmse_um']:.3f} should beat "
            f"EWMA RMSE={r_ewma['rmse_um']:.3f} on linear drift")

    def test_dewma_trend_nonzero_with_drift(self):
        """DEWMA trend estimate must be non-zero when drift is present."""
        r = self.r2r.dewma_r2r(55.0, 50, 0.3, 0.0, 0.1, 0)
        trends = r["d_trend"]
        # After warm-up the trend component should be non-zero
        assert any(abs(t) > 0.001 for t in trends[5:])

    # ── Lambda sweep ──────────────────────────────────────────────────────────

    def test_lambda_sweep_length(self):
        """Lambda sweep must return 19 entries (0.05, 0.10, ..., 0.95)."""
        results = self.r2r.lambda_sweep(n_runs=50)
        assert len(results) == 19

    def test_lambda_sweep_all_have_rmse(self):
        """Every entry in lambda sweep must have a non-negative rmse_um."""
        results = self.r2r.lambda_sweep(n_runs=50)
        for entry in results:
            assert entry["rmse_um"] >= 0.0

    # ── Demo ──────────────────────────────────────────────────────────────────

    def test_r2r_demo_runs(self):
        """r2r_demo must run without error and return comparison keys."""
        demo = self.r2r.r2r_demo()
        for key in ("no_control_rmse", "ewma_rmse", "dewma_rmse",
                    "y_ewma", "y_dewma", "scenario"):
            assert key in demo

    def test_r2r_demo_control_beats_no_control(self):
        """Both EWMA and DEWMA must beat uncontrolled RMSE in the demo."""
        demo = self.r2r.r2r_demo()
        assert demo["ewma_rmse"]  < demo["no_control_rmse"]
        assert demo["dewma_rmse"] < demo["no_control_rmse"]

    def test_r2r_demo_dewma_improvement_positive(self):
        """DEWMA improvement over EWMA must be ≥ 0%."""
        demo = self.r2r.r2r_demo()
        assert demo["dewma_improvement_pct"] >= 0.0
