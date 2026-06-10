"""Tests for AP試験 fourth wave:
  fem/wafer_warp_kernel.cpp            → 研削反り (Stoney式, TAIKO®リブ抑制)
  sensors/in_situ_monitor_kernel.cpp   → インプロセス計測 (AE, 電流, カルマン, 融合)
  pipeline/recipe_optimizer_kernel.cpp → ベイズ最適化 (GP+EI/UCB, Cp/Cpk)
"""
from __future__ import annotations
import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ══════════════════════════════════════════════════════════════════════════════
# Wafer Warp (Stoney / TAIKO® / grit sweep)
# ══════════════════════════════════════════════════════════════════════════════

class TestWaferWarp:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("fem._wafer_warp_kernel",
                             reason="wafer_warp_kernel .so not built")
        from fem._wafer_warp_kernel import (
            stoney_warp, taiko_warp_reduction, grit_sweep, wafer_warp_demo,
        )
        self.stoney_warp         = stoney_warp
        self.taiko_warp_reduction = taiko_warp_reduction
        self.grit_sweep          = grit_sweep
        self.wafer_warp_demo     = wafer_warp_demo

    # --- stoney_warp ----------------------------------------------------------
    def test_stoney_warp_keys(self):
        r = self.stoney_warp("Si", 300.0, 25.0, 5.0, 2000.0, 5.0)
        for k in ("sigma_residual_MPa", "sigma_thermal_MPa", "sigma_total_MPa",
                  "R_curvature_m", "bow_mm", "warp_um", "ttv_um", "handling_risk"):
            assert k in r, f"missing key: {k}"

    def test_stoney_warp_positive(self):
        r = self.stoney_warp("Si", 300.0, 25.0, 5.0, 2000.0, 5.0)
        assert r["warp_um"] > 0
        assert r["bow_mm"] > 0
        assert r["sigma_residual_MPa"] > 0

    def test_stoney_warp_finer_grit_less_warp(self):
        coarse = self.stoney_warp("Si", 300.0, 25.0, 5.0, 320.0, 5.0)
        fine   = self.stoney_warp("Si", 300.0, 25.0, 5.0, 8000.0, 5.0)
        assert fine["warp_um"] < coarse["warp_um"], \
            "finer grit should produce less warp"

    def test_stoney_warp_thinner_more_warp(self):
        thick = self.stoney_warp("Si", 300.0, 200.0, 5.0, 2000.0, 0.0)
        thin  = self.stoney_warp("Si", 300.0,  25.0, 5.0, 2000.0, 0.0)
        assert thin["warp_um"] > thick["warp_um"], \
            "thinner wafer should bow more (smaller h_s -> larger warp)"

    def test_stoney_warp_ttv_fraction_of_warp(self):
        r = self.stoney_warp("Si", 300.0, 25.0, 5.0, 2000.0, 5.0)
        assert r["ttv_um"] == pytest.approx(r["warp_um"] * 0.10, rel=1e-6)

    def test_stoney_warp_handling_risk_field(self):
        r = self.stoney_warp("Si", 300.0, 25.0, 5.0, 2000.0, 5.0)
        assert r["handling_risk"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")

    # --- taiko_warp_reduction -------------------------------------------------
    def test_taiko_keys(self):
        r = self.taiko_warp_reduction("Si", 300.0, 25.0, 3.5, 775.0, 2000.0, 5.0)
        for k in ("warp_full_grind_um", "suppression_fraction",
                  "warp_taiko_um", "stiffness_ratio",
                  "feasible_25um", "hbm4_spec_1mm"):
            assert k in r, f"missing key: {k}"

    def test_taiko_suppression_fraction_range(self):
        r = self.taiko_warp_reduction("Si", 300.0, 25.0, 3.5, 775.0, 2000.0, 5.0)
        sup = r["suppression_fraction"]
        assert 0 < sup < 1, f"suppression_fraction out of range: {sup}"

    def test_taiko_reduces_warp(self):
        r = self.taiko_warp_reduction("Si", 300.0, 25.0, 3.5, 775.0, 2000.0, 5.0)
        assert r["warp_taiko_um"] < r["warp_full_grind_um"], \
            "TAIKO® should reduce warp vs full grind"

    def test_taiko_stiffness_ratio_gt1(self):
        r = self.taiko_warp_reduction("Si", 300.0, 25.0, 3.5, 775.0, 2000.0, 5.0)
        assert r["stiffness_ratio"] > 1.0, "rim thicker than thin center → I_rim > I_center"

    def test_taiko_hbm4_field(self):
        r = self.taiko_warp_reduction("Si", 300.0, 25.0, 3.5, 775.0, 2000.0, 5.0)
        assert r["hbm4_spec_1mm"] in ("PASS (< 1 mm)", "FAIL")

    # --- grit_sweep -----------------------------------------------------------
    def test_grit_sweep_keys(self):
        r = self.grit_sweep("Si", 300.0, 25.0, 1000.0, True, 3.5)
        for k in ("sweep", "min_grit_needed", "achievable"):
            assert k in r, f"missing key: {k}"

    def test_grit_sweep_has_entries(self):
        r = self.grit_sweep("Si", 300.0, 25.0, 1000.0, True, 3.5)
        assert len(r["sweep"]) >= 5

    def test_grit_sweep_achievable_loose_spec(self):
        r = self.grit_sweep("Si", 300.0, 200.0, 1e7, False, 3.5)
        assert r["achievable"] is True

    # --- wafer_warp_demo ------------------------------------------------------
    def test_wafer_warp_demo_two_materials(self):
        results = self.wafer_warp_demo()
        assert len(results) == 2
        mats = {d["material"] for d in results}
        assert "Si" in mats and "SiC" in mats

    def test_wafer_warp_demo_taiko_reduction_positive(self):
        results = self.wafer_warp_demo()
        for d in results:
            assert d["taiko_reduction_pct"] > 0, \
                f"{d['material']}: TAIKO® should always reduce warp"


# ══════════════════════════════════════════════════════════════════════════════
# In-situ Monitor (AE sensor / spindle current / Kalman / fusion)
# ══════════════════════════════════════════════════════════════════════════════

class TestInSituMonitor:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("sensors._in_situ_monitor_kernel",
                             reason="in_situ_monitor_kernel .so not built")
        from sensors._in_situ_monitor_kernel import (
            ae_sensor_sim, spindle_current_monitor,
            kalman_kerf_filter, sensor_fusion_demo,
        )
        self.ae_sensor_sim           = ae_sensor_sim
        self.spindle_current_monitor = spindle_current_monitor
        self.kalman_kerf_filter      = kalman_kerf_filter
        self.sensor_fusion_demo      = sensor_fusion_demo

    # --- AE sensor ------------------------------------------------------------
    def test_ae_sensor_keys(self):
        r = self.ae_sensor_sim(30.0, 10.0, 100, 0.1, 1.0, 42)
        for k in ("ae_rms", "chip_events", "n_true_chips",
                  "detection_rate", "false_alarm_rate", "threshold"):
            assert k in r, f"missing key: {k}"

    def test_ae_rms_length(self):
        r = self.ae_sensor_sim(30.0, 10.0, 200, 0.1, 1.0, 42)
        assert len(r["ae_rms"]) == 200

    def test_ae_chipping_rate_zero_no_chips(self):
        r = self.ae_sensor_sim(30.0, 10.0, 200, 0.0, 1.0, 42)
        assert r["n_true_chips"] == 0
        assert len(r["chip_events"]) == 0

    def test_ae_high_chipping_many_chips(self):
        r = self.ae_sensor_sim(30.0, 10.0, 500, 1.0, 0.5, 42)
        assert r["n_true_chips"] > 400, "chipping_rate=1.0 should chip almost every cut"

    def test_ae_detection_positive(self):
        r = self.ae_sensor_sim(30.0, 10.0, 500, 1.0, 1.0, 42)
        assert r["detection_rate"] > 0.5, "should detect majority of chips"

    # --- spindle current monitor ---------------------------------------------
    def test_spindle_keys(self):
        r = self.spindle_current_monitor(3.5, 200, 0.4, 0.08, 4.5, 7)
        for k in ("current_trace", "alarm_run", "alarm_triggered",
                  "I_base_A", "I_final_A", "blade_wear_pct_at_alarm",
                  "remaining_life_pct"):
            assert k in r, f"missing key: {k}"

    def test_spindle_current_increases(self):
        r = self.spindle_current_monitor(3.5, 200, 0.4, 0.0, 999.0, 7)
        # no noise, no alarm → I_final should be higher than I_base
        assert r["I_final_A"] > r["I_base_A"]

    def test_spindle_alarm_triggers_when_reachable(self):
        # threshold = 4.0 < I_base*(1+wear_factor) = 3.5*1.4 = 4.9
        r = self.spindle_current_monitor(3.5, 200, 0.4, 0.0, 4.0, 7)
        assert r["alarm_triggered"] is True

    def test_spindle_trace_length(self):
        r = self.spindle_current_monitor(3.5, 100, 0.4, 0.08, 999.0, 7)
        assert len(r["current_trace"]) == 100

    # --- Kalman filter --------------------------------------------------------
    def test_kalman_keys(self):
        r = self.kalman_kerf_filter(55.0, 100, 0.05, 0.2, 0.8, 99)
        for k in ("y_meas", "x_est_offset", "x_est_drift",
                  "true_drift", "estimated_drift", "converged"):
            assert k in r, f"missing key: {k}"

    def test_kalman_trace_length(self):
        r = self.kalman_kerf_filter(55.0, 80, 0.05, 0.2, 0.8, 99)
        assert len(r["y_meas"]) == 80
        assert len(r["x_est_offset"]) == 80

    def test_kalman_drift_estimation_converges(self):
        r = self.kalman_kerf_filter(55.0, 200, 0.10, 0.1, 0.5, 99)
        assert r["converged"] is True, \
            f"Kalman should converge; drift_error_pct={r['drift_error_pct']:.1f}%"

    # --- sensor fusion --------------------------------------------------------
    def test_fusion_keys(self):
        r = self.sensor_fusion_demo(150, 100.0, 314)
        for k in ("anomaly_scores", "alarms", "fault_start_run",
                  "n_true_alarm_runs", "detection_precision", "threshold"):
            assert k in r, f"missing key: {k}"

    def test_fusion_scores_length(self):
        r = self.sensor_fusion_demo(100, 80.0, 314)
        assert len(r["anomaly_scores"]) == 100

    def test_fusion_detects_late_fault(self):
        # fault from run 80 out of 150 → should get some correct alarms
        r = self.sensor_fusion_demo(150, 80.0, 314)
        assert r["n_true_alarm_runs"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# Recipe Optimizer (Bayesian optimization / Cp/Cpk)
# ══════════════════════════════════════════════════════════════════════════════

class TestRecipeOptimizer:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("pipeline._recipe_optimizer_kernel",
                             reason="recipe_optimizer_kernel .so not built")
        from pipeline._recipe_optimizer_kernel import (
            bayesian_optimize_1d, process_capability, recipe_optimizer_demo,
        )
        self.bayesian_optimize_1d = bayesian_optimize_1d
        self.process_capability   = process_capability
        self.recipe_optimizer_demo = recipe_optimizer_demo

    # --- bayesian_optimize_1d ------------------------------------------------
    def test_bo_keys(self):
        r = self.bayesian_optimize_1d(10.0, 50.0, 30.0, 5, 15, 0.05,
                                       "EI", 8.0, 2.0, 2026)
        for k in ("X_obs", "y_obs", "best_x", "best_y",
                  "true_optimal", "final_regret", "iterations",
                  "acq_fn", "converged"):
            assert k in r, f"missing key: {k}"

    def test_bo_observation_count(self):
        r = self.bayesian_optimize_1d(10.0, 50.0, 30.0, 5, 15, 0.05,
                                       "EI", 8.0, 2.0, 2026)
        assert len(r["X_obs"]) == 5 + 15  # n_init + n_iter

    def test_bo_iterations_count(self):
        r = self.bayesian_optimize_1d(10.0, 50.0, 30.0, 5, 10, 0.05,
                                       "UCB", 8.0, 2.0, 2026)
        assert len(r["iterations"]) == 10

    def test_bo_ei_regret_positive(self):
        r = self.bayesian_optimize_1d(10.0, 50.0, 30.0, 5, 20, 0.05,
                                       "EI", 8.0, 2.0, 2026)
        assert r["final_regret"] >= 0

    def test_bo_ucb_regret_positive(self):
        r = self.bayesian_optimize_1d(10.0, 50.0, 30.0, 5, 20, 0.05,
                                       "UCB", 8.0, 2.0, 2026)
        assert r["final_regret"] >= 0

    def test_bo_ei_best_x_in_range(self):
        r = self.bayesian_optimize_1d(10.0, 50.0, 30.0, 5, 20, 0.05,
                                       "EI", 8.0, 2.0, 2026)
        assert 10.0 <= r["best_x"] <= 50.0

    def test_bo_acq_fn_label_preserved(self):
        for fn in ("EI", "UCB"):
            r = self.bayesian_optimize_1d(10.0, 50.0, 30.0, 3, 5, 0.05,
                                           fn, 8.0, 2.0, 2026)
            assert r["acq_fn"] == fn

    # --- process_capability --------------------------------------------------
    def test_capability_keys(self):
        data = [55.0 + 0.3 * (i % 5 - 2) for i in range(20)]
        r = self.process_capability(data, 57.0, 53.0)
        for k in ("mean", "sigma", "Cp", "Cpk", "ppm_est", "grade"):
            assert k in r, f"missing key: {k}"

    def test_capability_centered_cpk_positive(self):
        # Mean exactly centered between spec limits
        data = [0.0 + 0.1 * (i % 3 - 1) for i in range(30)]
        r = self.process_capability(data, 1.0, -1.0)
        assert r["Cpk"] > 0

    def test_capability_cp_formula(self):
        vals = [(-1)**i * 0.1 for i in range(20)]  # alternates ±0.1, sigma ≈ 0.103
        r = self.process_capability(vals, 1.0, -1.0)
        # Cp = (USL-LSL)/(6*sigma)
        expected_cp = (1.0 - (-1.0)) / (6.0 * r["sigma"])
        assert r["Cp"] == pytest.approx(expected_cp, rel=1e-6)

    def test_capability_grade_field(self):
        data = [0.0 + 0.01 * i for i in range(30)]
        r = self.process_capability(data, 10.0, -10.0)
        assert r["grade"] in ("Six Sigma", "Capable (1.33)", "Marginal", "Not capable")

    # --- recipe_optimizer_demo ------------------------------------------------
    def test_demo_keys(self):
        r = self.recipe_optimizer_demo()
        for k in ("scenario", "search_range", "true_optimal_krpm",
                  "ei_best_krpm", "ucb_best_krpm",
                  "ei_regret", "ucb_regret",
                  "ei_converged", "ucb_converged",
                  "best_recipe_Cpk", "best_recipe_grade"):
            assert k in r, f"missing key: {k}"

    def test_demo_true_optimal_is_30(self):
        r = self.recipe_optimizer_demo()
        assert r["true_optimal_krpm"] == pytest.approx(30.0)

    def test_demo_best_recipe_cpk_numeric(self):
        r = self.recipe_optimizer_demo()
        assert isinstance(float(r["best_recipe_Cpk"]), float)

    def test_demo_regrets_nonnegative(self):
        r = self.recipe_optimizer_demo()
        assert float(r["ei_regret"]) >= 0
        assert float(r["ucb_regret"]) >= 0
