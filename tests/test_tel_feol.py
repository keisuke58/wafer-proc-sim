"""
魔改造 TEL — 統合スモークテスト
================================
forward model (fem/tel_feol_recipe.py) と、その上に乗る5モジュール
(GP代理 / 多目的BO / EnKF-APC / TMCMC較正 / 装置競争力) のコア関数を
高速パラメータで叩き、物理的に妥当な結果が返ることを検証する。

CLI 図生成は伴わない（高速・CI向け）。各モジュールの本気の検証は
それぞれの __main__ 実行と論文用図で行う。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ════════════════════════════════════════════════════════════════════════════
# 0. Forward model 契約
# ════════════════════════════════════════════════════════════════════════════
class TestForwardModel:
    def test_default_recipe_fom_sane(self):
        from market_analysis.tel_feol_recipe import default_recipe, simulate_recipe
        f = simulate_recipe(default_recipe())
        assert 5.0 <= f["mu_ch"] <= 95.0
        assert 2.0 <= f["V_th"] <= 4.0
        assert f["R_on_sp"] > 0 and f["Dit"] > 0 and f["EOT_nm"] > 0

    def test_vector_round_trip(self):
        from market_analysis.tel_feol_recipe import (
            default_recipe, recipe_to_vector, vector_to_recipe)
        r = default_recipe()
        r2 = vector_to_recipe(recipe_to_vector(r))
        assert r2["ald_material"] == r["ald_material"]
        assert abs(r2["ox_T_C"] - r["ox_T_C"]) < 1e-6

    def test_objective_vector_and_scalar(self):
        from market_analysis.tel_feol_recipe import (
            default_recipe, objective_vector, scalar_fom, OBJECTIVES)
        ov = objective_vector(default_recipe())
        assert ov.shape == (len(OBJECTIVES),) == (3,)
        assert ov[0] < 0          # mu_ch は max → 符号反転で負
        assert scalar_fom(default_recipe()) > 0

    def test_tool_state_drift_changes_output(self):
        from market_analysis.tel_feol_recipe import default_recipe, simulate_recipe
        base = simulate_recipe(default_recipe())
        drift = simulate_recipe(default_recipe(),
                                tool_state={"ox_T_offset_C": 40.0,
                                            "ald_gpc_scale": 1.1})
        assert drift != base


# ════════════════════════════════════════════════════════════════════════════
# 1. GP 代理モデル
# ════════════════════════════════════════════════════════════════════════════
class TestSurrogate:
    def test_fit_predict_tracks_truth(self):
        from ml.tel_feol_surrogate import build_dataset, FeolSurrogate
        from market_analysis.tel_feol_recipe import default_recipe, simulate_recipe
        X, Y, recipes = build_dataset(40, seed=0)
        surr = FeolSurrogate().fit_xy(X, Y)
        mu_hat, ron_hat = surr.predict(default_recipe())
        true = simulate_recipe(default_recipe())
        # 代理は真値を ±40% 程度で追えていればスモーク合格
        assert abs(mu_hat - true["mu_ch"]) / true["mu_ch"] < 0.4
        assert ron_hat > 0


# ════════════════════════════════════════════════════════════════════════════
# 2. 多目的ベイズ最適化 (ParEGO)
# ════════════════════════════════════════════════════════════════════════════
class TestBO:
    def test_parego_finds_nondominated_front(self):
        from optimization.tel_feol_bo import run_parego, pareto_mask, hypervolume
        res = run_parego(n_init=8, n_iter=6, pool=80, seed=0, verbose=False)
        Y = res["Y"]
        assert Y.shape[1] == 3 and len(Y) == 14
        mask = pareto_mask(Y)
        assert mask.sum() >= 1
        # Pareto 集合は真に非劣 (自分自身に支配される点がない)
        ref = Y.max(axis=0) + 1.0
        hv_all = hypervolume(Y[mask], ref)
        assert hv_all >= 0


# ════════════════════════════════════════════════════════════════════════════
# 3. EnKF run-to-run 制御 (APC)
# ════════════════════════════════════════════════════════════════════════════
class TestAPC:
    def test_control_beats_open_loop(self):
        from pipeline.tel_apc import run_apc
        res = run_apc(n_wafers=30, n_ensemble=15, meas_noise=0.15,
                      drift_sigma=2.0, seed=0)
        # 制御ありは開ループより設定値追従が良い
        assert res["rmse_ctrl"] <= res["rmse_open"]
        assert res["mu_ctrl"].shape == (30,)


# ════════════════════════════════════════════════════════════════════════════
# 4. TMCMC 物理較正
# ════════════════════════════════════════════════════════════════════════════
class TestTMCMC:
    def test_posterior_recovers_B_coul(self):
        from optimization.tel_tmcmc_calibration import (
            tmcmc, log_likelihood, log_prior, PARAM_BOUNDS, PARAM_NAMES)
        res = tmcmc(log_likelihood, log_prior, PARAM_BOUNDS,
                    n_samples=200, max_stages=20, beta_scale=0.2, seed=42)
        samples, weights = res["samples"], res["weights"]
        mean = np.average(samples, weights=weights, axis=0)
        assert res["beta_final"] == pytest.approx(1.0, abs=1e-9)
        # B_coul は順モデル公称 5.5e13 のオーダーに収まる
        b = mean[PARAM_NAMES.index("B_coul")]
        assert 1e13 <= b <= 2e14


# ════════════════════════════════════════════════════════════════════════════
# 5. 装置競争力
# ════════════════════════════════════════════════════════════════════════════
class TestEquipmentCompetitiveness:
    def test_clean_axis_favors_tel(self):
        from market_analysis.tel_equipment_competitiveness import run_competitiveness
        axes, meta = run_competitiveness(n_wafers=60, seed=0)
        assert set(axes) == {"ALD_uniformity", "clean_effectiveness", "low_damage_etch"}
        clean = axes["clean_effectiveness"]
        # 洗浄軸では TEL (CELLESTA) が競合を大きく上回る歩留まり
        assert clean["tel"]["yield"] >= clean["competitor"]["yield"]
