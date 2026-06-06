"""
TEL APC — EnKF run-to-run 先進プロセス制御 (Advanced Process Control)
=====================================================================
ALD / 熱酸化ツールの「隠れ装置状態ドリフト」を Ensemble Kalman Filter (EnKF)
で推定し、ウェハ毎にレシピノブをフィードバック調整して µ_ch を設定値に保持する
run-to-run (R2R) 制御を実装する。

魔改造 TEL スタックの制御層:
    fem/tel_feol_recipe.py          forward model (recipe + tool_state → FOM)
    ml/tel_feol_surrogate.py        GP 代理モデル
    optimization/tel_feol_bo.py     多目的ベイズ最適化 (Pareto)
    pipeline/tel_apc.py  ←ここ      EnKF run-to-run 制御

方法 (各ウェハ k):
    [1] 真の装置状態がランダムウォークでドリフト
            x_k = x_{k-1} + w_k,   w_k ~ N(0, Q)
            x = (ox_T_offset_C, ald_gpc_scale)
            - ox_T_offset_C : 酸化炉温度ドリフト [°C] (µ_ch を直接動かす)
            - ald_gpc_scale : ALD 成膜速度ドリフト (EOT/V_th 経由の二次効果)
    [2] 計測: y_k = µ_ch(recipe_k, x_k) + v_k,  v_k ~ N(0, R)   (計測ノイズ)
    [3] EnFK 更新: ~30 メンバのアンサンブルで隠れ状態 x を推定
            - predict : 各メンバを Q でドリフトさせる
            - update  : 計測 y_k で Kalman ゲイン補正 (アンサンブル共分散)
    [4] R2R フィードバック: 推定ドリフト ox_T_offset_C を
            酸化時間ノブ ox_t_hr で打ち消す (解析ゲイン dµ/dox_t で逆算)。
            target µ_ch ≈ default recipe の µ_ch (~75)。

評価:
    CONTROLLED (EnKF R2R) vs UNCONTROLLED (open-loop, ノブ固定) を
    µ_ch の対設定値 RMSE で比較。制御側が明確に良いことを示す。

実行:
    python pipeline/tel_apc.py                 # 80 ウェハ, デフォルト設定
    python pipeline/tel_apc.py --wafers 120 --ensemble 40 --seed 7
    python pipeline/tel_apc.py --meas-noise 0.4 --drift-sigma 1.5

参考:
    Evensen (2003) Ocean Dyn — Ensemble Kalman Filter
    Moyne, del Castillo & Hurwitz (2001) — Run-to-Run Control in Semiconductor Mfg
    Qin et al. (2006) — Semiconductor APC/R2R control review
    Kimoto & Cooper (2014) — SiC MOSFET technology
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/nishioka/git/wafer-proc-sim")

from fem.tel_feol_recipe import simulate_recipe, default_recipe

OUT_DIR = "/home/nishioka/git/wafer-proc-sim/results"
FIG_PATH = os.path.join(OUT_DIR, "tel_apc_control.png")

# 隠れ装置状態ベクトル順序
STATE_NAMES = ["ox_T_offset_C", "ald_gpc_scale"]

# 酸化時間ノブの可動域 (forward model の CONTINUOUS["ox_t_hr"] に整合)
OX_T_HR_LO, OX_T_HR_HI = 0.5, 4.0


def _measure_mu(recipe, true_state):
    """真の装置状態下での µ_ch (ノイズ無し真値)。"""
    return simulate_recipe(recipe, true_state)["mu_ch"]


def estimate_gains(recipe, eps_off=5.0, eps_oxt=0.1):
    """
    制御設計用の局所感度を有限差分で取得。
        g_off = dµ_ch / d(ox_T_offset_C)   [cm²/Vs per °C]
        g_oxt = dµ_ch / d(ox_t_hr)         [cm²/Vs per hr]
    これらから「推定温度ドリフトを ox_t で打ち消す」補正量を逆算する。
    """
    base = dict(recipe)
    mu_p = _measure_mu(base, {"ox_T_offset_C": +eps_off})
    mu_m = _measure_mu(base, {"ox_T_offset_C": -eps_off})
    g_off = (mu_p - mu_m) / (2.0 * eps_off)

    rp = dict(base); rp["ox_t_hr"] = base["ox_t_hr"] + eps_oxt
    rm = dict(base); rm["ox_t_hr"] = base["ox_t_hr"] - eps_oxt
    g_oxt = (_measure_mu(rp, None) - _measure_mu(rm, None)) / (2.0 * eps_oxt)
    return g_off, g_oxt


# ════════════════════════════════════════════════════════════════════════════
# Ensemble Kalman Filter
# ════════════════════════════════════════════════════════════════════════════

class ToolStateEnKF:
    """
    隠れ装置状態 x = (ox_T_offset_C, ald_gpc_scale) を µ_ch 計測から推定する EnKF。

    観測演算子 h(x) は forward model 経由 (非線形)。EnKF はアンサンブル統計で
    h の Jacobian を陽に持たずに Kalman ゲインを構成する。
    """

    def __init__(self, recipe0, n_ensemble=30, Q=None, R=0.09, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n = int(n_ensemble)
        # プロセスノイズ共分散 Q (ドリフトモデル)
        self.Q = np.array(Q) if Q is not None else np.diag([1.0, 0.0025])
        self.R = float(R)  # 計測ノイズ分散 (µ_ch)
        # 初期アンサンブル: 公称 (offset 0, scale 1) 周りに散らす
        mean0 = np.array([0.0, 1.0])
        spread = np.array([2.0, 0.05])
        self.ens = mean0 + spread * self.rng.standard_normal((self.n, 2))
        self.recipe0 = dict(recipe0)

    def state_mean(self):
        return self.ens.mean(axis=0)

    def _ens_to_ts(self, member):
        return {"ox_T_offset_C": float(member[0]),
                "ald_gpc_scale": float(member[1])}

    def predict(self):
        """ランダムウォークドリフトをアンサンブルに伝播 (forecast step)。"""
        w = self.rng.multivariate_normal(np.zeros(2), self.Q, size=self.n)
        self.ens = self.ens + w
        # 物理的に妥当な範囲に緩くクリップ (scale > 0)
        self.ens[:, 1] = np.clip(self.ens[:, 1], 0.5, 1.6)

    def update(self, recipe_applied, y_meas):
        """
        計測 y_meas (ノイズ込み µ_ch) で解析更新 (analysis step)。
        observation perturbation 方式の EnKF。
        """
        # 各メンバの予測観測 h(x_i) = µ_ch(recipe_applied, x_i)
        hx = np.array([
            _measure_mu(recipe_applied, self._ens_to_ts(m)) for m in self.ens
        ])
        x_mean = self.ens.mean(axis=0)
        h_mean = hx.mean()

        dX = self.ens - x_mean            # (n, 2)
        dH = hx - h_mean                  # (n,)

        # アンサンブル共分散
        P_xh = (dX * dH[:, None]).sum(axis=0) / (self.n - 1)   # (2,)
        P_hh = (dH * dH).sum() / (self.n - 1) + self.R         # scalar

        K = P_xh / P_hh                   # Kalman gain (2,)

        # 観測摂動付き更新
        y_pert = y_meas + np.sqrt(self.R) * self.rng.standard_normal(self.n)
        innov = y_pert - hx               # (n,)
        self.ens = self.ens + innov[:, None] * K[None, :]
        self.ens[:, 1] = np.clip(self.ens[:, 1], 0.5, 1.6)
        return self.state_mean()


# ════════════════════════════════════════════════════════════════════════════
# Run-to-run シミュレーション
# ════════════════════════════════════════════════════════════════════════════

def run_apc(n_wafers=80, n_ensemble=30, meas_noise=0.15, drift_sigma=2.0,
            gpc_drift_sigma=0.01, drift_cap=30.0, seed=0):
    """
    EnKF R2R 制御 (CONTROLLED) と open-loop (UNCONTROLLED) を同一ドリフト系列で比較。

    返り値 dict:
        wafers, setpoint,
        mu_open, mu_ctrl                 各ウェハの真 µ_ch
        true_off, est_off                真/推定 ox_T_offset_C
        true_scale, est_scale            真/推定 ald_gpc_scale
        ox_t_ctrl                        制御が打った酸化時間ノブ
        rmse_open, rmse_ctrl, improve_pct
    """
    rng = np.random.default_rng(seed)
    recipe0 = default_recipe()

    # 設定値 = 公称レシピ・公称装置状態での µ_ch
    setpoint = _measure_mu(recipe0, None)

    # 制御設計用の局所ゲイン
    g_off, g_oxt = estimate_gains(recipe0)

    # 共通の真ドリフト系列 (両ケースで同一)
    Q_true = np.diag([drift_sigma ** 2, gpc_drift_sigma ** 2])
    true_states = np.zeros((n_wafers, 2))
    x = np.array([0.0, 1.0])
    for k in range(n_wafers):
        x = x + rng.multivariate_normal(np.zeros(2), Q_true)
        # 真ドリフトもノブ権限内に収まるよう緩くキャップ
        x[0] = np.clip(x[0], -drift_cap, drift_cap)
        x[1] = np.clip(x[1], 0.6, 1.5)
        true_states[k] = x

    # EnKF (制御器が Q をやや過大評価 = 追従性確保)
    enkf = ToolStateEnKF(
        recipe0, n_ensemble=n_ensemble,
        Q=np.diag([(drift_sigma * 1.3) ** 2, (gpc_drift_sigma * 1.3) ** 2]),
        R=meas_noise ** 2, seed=seed + 1,
    )

    mu_open = np.zeros(n_wafers)
    mu_ctrl = np.zeros(n_wafers)
    est_off = np.zeros(n_wafers)
    est_scale = np.zeros(n_wafers)
    ox_t_ctrl = np.zeros(n_wafers)

    # 計測ノイズ系列 (両ケースで同一にして公平比較)
    meas_v = meas_noise * rng.standard_normal(n_wafers)

    for k in range(n_wafers):
        ts_true = {"ox_T_offset_C": float(true_states[k, 0]),
                   "ald_gpc_scale": float(true_states[k, 1])}

        # ── UNCONTROLLED: ノブ固定 (open-loop) ──────────────────
        mu_open[k] = _measure_mu(recipe0, ts_true)

        # ── CONTROLLED: EnKF predict → 前ウェハ推定でノブ補正 → 計測 → update ─
        enkf.predict()
        x_hat = enkf.state_mean()         # 計測前の事前推定 (forecast)

        # R2R フィードフォワード補正:
        #   推定温度ドリフトが µ_ch を g_off*Δoff だけ動かす → ox_t で打ち消す
        #   Δox_t = -(g_off / g_oxt) * est_offset
        delta_ox_t = -(g_off / g_oxt) * x_hat[0]
        ox_t_cmd = float(np.clip(recipe0["ox_t_hr"] + delta_ox_t,
                                 OX_T_HR_LO, OX_T_HR_HI))
        recipe_k = dict(recipe0)
        recipe_k["ox_t_hr"] = ox_t_cmd
        ox_t_ctrl[k] = ox_t_cmd

        # 実プロセス実行 (真の装置状態)
        mu_true_ctrl = _measure_mu(recipe_k, ts_true)
        mu_ctrl[k] = mu_true_ctrl

        # 計測 (ノイズ込み) → EnKF 解析更新
        y_meas = mu_true_ctrl + meas_v[k]
        x_post = enkf.update(recipe_k, y_meas)
        est_off[k] = x_post[0]
        est_scale[k] = x_post[1]

    rmse_open = float(np.sqrt(np.mean((mu_open - setpoint) ** 2)))
    rmse_ctrl = float(np.sqrt(np.mean((mu_ctrl - setpoint) ** 2)))
    improve_pct = 100.0 * (rmse_open - rmse_ctrl) / rmse_open

    return {
        "wafers": np.arange(1, n_wafers + 1),
        "setpoint": setpoint,
        "mu_open": mu_open,
        "mu_ctrl": mu_ctrl,
        "true_off": true_states[:, 0],
        "est_off": est_off,
        "true_scale": true_states[:, 1],
        "est_scale": est_scale,
        "ox_t_ctrl": ox_t_ctrl,
        "rmse_open": rmse_open,
        "rmse_ctrl": rmse_ctrl,
        "improve_pct": improve_pct,
        "g_off": g_off,
        "g_oxt": g_oxt,
    }


# ════════════════════════════════════════════════════════════════════════════
# 図
# ════════════════════════════════════════════════════════════════════════════

def make_figure(res, path=FIG_PATH):
    w = res["wafers"]
    sp = res["setpoint"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # ── パネル1: µ_ch vs wafer (open-loop vs EnKF制御) ──────────
    ax = axes[0]
    ax.axhline(sp, color="k", ls="--", lw=1.4,
               label=f"setpoint = {sp:.2f}")
    ax.plot(w, res["mu_open"], "o-", color="tab:red", ms=3, lw=1.0,
            alpha=0.8, label=f"open-loop (RMSE={res['rmse_open']:.3f})")
    ax.plot(w, res["mu_ctrl"], "s-", color="tab:blue", ms=3, lw=1.2,
            alpha=0.9, label=f"EnKF R2R (RMSE={res['rmse_ctrl']:.3f})")
    ax.set_xlabel("wafer index")
    ax.set_ylabel(r"$\mu_{ch}$  [cm$^2$/Vs]")
    ax.set_title(
        f"TEL APC: EnKF run-to-run control of channel mobility "
        f"(improvement {res['improve_pct']:.1f}%)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # ── パネル2: 真 vs EnKF推定の装置ドリフト ───────────────────
    ax2 = axes[1]
    ax2.plot(w, res["true_off"], "-", color="tab:green", lw=1.8,
             label="true ox_T_offset [°C]")
    ax2.plot(w, res["est_off"], "x--", color="tab:purple", lw=1.0, ms=4,
             label="EnKF estimate [°C]")
    ax2.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel("wafer index")
    ax2.set_ylabel("oxidation furnace temp drift  [°C]")
    ax2.set_title("Hidden tool-state drift: true vs EnKF estimate")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)

    # 推定誤差 RMSE を注記
    drift_rmse = float(np.sqrt(np.mean((res["true_off"] - res["est_off"]) ** 2)))
    ax2.text(0.98, 0.04,
             f"drift est. RMSE = {drift_rmse:.2f} °C",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="TEL APC — EnKF run-to-run control of ALD/oxidation tools")
    ap.add_argument("--wafers", type=int, default=80, help="シミュレーションウェハ数")
    ap.add_argument("--ensemble", type=int, default=30, help="EnKF アンサンブルメンバ数")
    ap.add_argument("--meas-noise", type=float, default=0.15,
                    help="µ_ch 計測ノイズ標準偏差 [cm²/Vs]")
    ap.add_argument("--drift-sigma", type=float, default=2.0,
                    help="ox_T_offset ランダムウォーク標準偏差 [°C/wafer]")
    ap.add_argument("--gpc-drift-sigma", type=float, default=0.01,
                    help="ald_gpc_scale ランダムウォーク標準偏差 /wafer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = run_apc(
        n_wafers=args.wafers,
        n_ensemble=args.ensemble,
        meas_noise=args.meas_noise,
        drift_sigma=args.drift_sigma,
        gpc_drift_sigma=args.gpc_drift_sigma,
        seed=args.seed,
    )

    print("=" * 64)
    print("TEL APC — EnKF run-to-run process control")
    print("=" * 64)
    print(f"  wafers              {args.wafers}")
    print(f"  ensemble members    {args.ensemble}")
    print(f"  setpoint µ_ch       {res['setpoint']:.3f} cm²/Vs")
    print(f"  control gains       dµ/dox_T_off={res['g_off']:+.4f},"
          f" dµ/dox_t={res['g_oxt']:+.4f}")
    print("-" * 64)
    print(f"  RMSE open-loop      {res['rmse_open']:.4f} cm²/Vs")
    print(f"  RMSE EnKF R2R       {res['rmse_ctrl']:.4f} cm²/Vs")
    print(f"  improvement         {res['improve_pct']:.1f} %")
    drift_rmse = float(np.sqrt(np.mean((res["true_off"] - res["est_off"]) ** 2)))
    print(f"  drift est. RMSE     {drift_rmse:.2f} °C")
    print("=" * 64)

    path = make_figure(res)
    print(f"[OK] saved -> {path}")


if __name__ == "__main__":
    main()
