"""
Keyence APC — VK-X フィードバックによる EnKF R2R ブレードダイシング制御
======================================================================
ブレード摩耗がダイシングウェーハ間でランダムウォークドリフトする状況を想定し、
Keyence VK-X 共焦点計測によるリアルタイムフィードバックで摩耗を推定し、
フィード速度ノブで補正するラン・ツー・ラン (R2R) 制御を実装する。

3 シナリオを比較:
  CTRL-VKX  — EnKF + VK-X 100× (σ_meas=0.30 µm): タイトな Kalman ゲイン
  CTRL-OPT  — EnKF + 光学顕微鏡 (σ_meas=1.50 µm): ゆるい Kalman ゲイン
  OPEN-LOOP — フィード固定、フィードバックなし

チッピング前進モデル (Lawn–Evans + blade wear 拡張):
    chip(recipe, x) = chip0
                      × (1 + wear_scale × x[0])   # ブレード摩耗
                      × x[1]                        # 材料不均一性
    chip0 = 5.0 µm (公称: feed=3 mm/s, depth=50 µm, spindle=30 krpm)
    wear_scale = 0.12 /（正規化摩耗単位）

装置状態 x = [blade_wear, material_var]:
    blade_wear : ブレード摩耗量 (単調増加傾向 + ランダムウォーク)
    material_var: 材料不均一性スケール (ランダムウォーク, 初期 1.0)

制御ノブ: feed_rate [mm/s]  (chip ∝ feed^0.6, Lawn–Evans 近似)

評価:
    目標チッピング chip_sp = 5.0 µm に対する RMSE を 3 シナリオで比較。

実行:
    python pipeline/keyence_apc.py
    python pipeline/keyence_apc.py --wafers 120 --ensemble 40 --seed 7
    python pipeline/keyence_apc.py --no-vkx --no-opt     # open-loop のみ

参考文献:
    Evensen G. (2003) Ocean Dyn — Ensemble Kalman Filter
    Moyne et al. (2001) — Run-to-Run Control in Semiconductor Mfg
    Lawn B.R. & Evans A.G. (1977) — lateral crack chipping model
    Keyence VK-X3100 仕様 (公開情報, 2024)
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fem.keyence_metrology_model import TOOL_SIGMA

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ────────────────────────────────────────────────────────────────────────────
# プロセス定数
# ────────────────────────────────────────────────────────────────────────────
CHIP0_UM        = 5.0    # 公称チッピング幅 [µm]
CHIP_SETPOINT   = 5.0    # 目標チッピング幅 [µm]
WEAR_SCALE      = 0.12   # 摩耗→チッピング増加係数
FEED_NOMINAL    = 3.0    # 公称フィード速度 [mm/s]
FEED_LO         = 1.0    # フィード下限 [mm/s]
FEED_HI         = 6.0    # フィード上限 [mm/s]
FEED_EXPONENT   = 0.60   # chip ∝ feed^alpha (Lawn–Evans 近似)


# ════════════════════════════════════════════════════════════════════════════
# 1. チッピング前進モデル
# ════════════════════════════════════════════════════════════════════════════

def chip_forward(feed: float, blade_wear: float,
                 material_var: float) -> float:
    """
    Lawn–Evans 拡張チッピングモデル:
        chip = chip0 × (feed/feed_nom)^α × (1 + wear_scale×wear) × mat_var
    """
    feed_factor = (max(feed, 0.1) / FEED_NOMINAL) ** FEED_EXPONENT
    return CHIP0_UM * feed_factor * (1.0 + WEAR_SCALE * blade_wear) * material_var


def feed_from_setpoint(blade_wear: float, material_var: float,
                       setpoint: float = CHIP_SETPOINT) -> float:
    """推定摩耗・材料変動から目標チッピングを達成するフィードを逆算."""
    denom = (1.0 + WEAR_SCALE * blade_wear) * material_var
    if denom < 1e-6:
        return FEED_NOMINAL
    ratio = setpoint / (CHIP0_UM * denom)
    feed  = FEED_NOMINAL * ratio ** (1.0 / FEED_EXPONENT)
    return float(np.clip(feed, FEED_LO, FEED_HI))


# ════════════════════════════════════════════════════════════════════════════
# 2. Ensemble Kalman Filter
# ════════════════════════════════════════════════════════════════════════════

class BladeWearEnKF:
    """
    隠れ装置状態 x = [blade_wear, material_var] をチッピング計測から推定する EnKF。

    観測演算子 h(x, feed) = chip_forward(feed, x[0], x[1])
    計測ノイズ R = σ_meas² (VK-X: 0.09, 光学: 2.25 µm²)
    """

    def __init__(self, sigma_meas: float, n_ensemble: int = 30,
                 Q: np.ndarray = None, seed: int = 0):
        self.rng   = np.random.default_rng(seed)
        self.n     = n_ensemble
        self.R     = sigma_meas**2
        self.Q     = Q if Q is not None else np.diag([0.004, 0.0001])

        # 初期アンサンブル: wear=0 ± 0.2, mat_var=1 ± 0.05
        mean0   = np.array([0.0, 1.0])
        spread0 = np.array([0.2, 0.05])
        self.ens = mean0 + spread0 * self.rng.standard_normal((self.n, 2))
        self.ens[:, 0] = np.maximum(self.ens[:, 0], 0.0)
        self.ens[:, 1] = np.clip(self.ens[:, 1], 0.5, 1.5)

    def state_mean(self) -> np.ndarray:
        return self.ens.mean(axis=0)

    def predict(self) -> None:
        """ランダムウォークドリフトをアンサンブルに伝播 (forecast step)."""
        w = self.rng.multivariate_normal(np.zeros(2), self.Q, size=self.n)
        self.ens = self.ens + w
        # 物理制約: wear ≥ 0, mat_var > 0
        self.ens[:, 0] = np.maximum(self.ens[:, 0], 0.0)
        self.ens[:, 1] = np.clip(self.ens[:, 1], 0.5, 1.5)

    def update(self, feed_applied: float, y_meas: float) -> np.ndarray:
        """計測 y_meas (ノイズ込みチッピング幅) で解析更新 (analysis step)."""
        hx     = np.array([chip_forward(feed_applied, m[0], m[1])
                            for m in self.ens])
        x_mean = self.ens.mean(axis=0)
        h_mean = hx.mean()

        dX = self.ens - x_mean          # (n, 2)
        dH = hx - h_mean                # (n,)

        P_xh = (dX * dH[:, None]).sum(axis=0) / (self.n - 1)   # (2,)
        P_hh = (dH**2).sum() / (self.n - 1) + self.R            # scalar

        K = P_xh / P_hh                 # Kalman gain (2,)

        # 観測摂動付き更新 (perturbed-observations EnKF)
        y_pert = y_meas + np.sqrt(self.R) * self.rng.standard_normal(self.n)
        innov  = y_pert - hx
        self.ens = self.ens + innov[:, None] * K[None, :]
        self.ens[:, 0] = np.maximum(self.ens[:, 0], 0.0)
        self.ens[:, 1] = np.clip(self.ens[:, 1], 0.5, 1.5)
        return self.state_mean()


# ════════════════════════════════════════════════════════════════════════════
# 3. R2R シミュレーション
# ════════════════════════════════════════════════════════════════════════════

def run_apc(
    n_wafers:      int   = 80,
    n_ensemble:    int   = 30,
    sigma_vkx:     float = TOOL_SIGMA["vkx_100x"],
    sigma_opt:     float = TOOL_SIGMA["opt_50x"],
    wear_drift:    float = 0.015,  # ウェーハ毎の平均摩耗増加 [/wafer]
    wear_noise:    float = 0.008,  # 摩耗ランダムウォーク σ [/wafer]
    mat_noise:     float = 0.015,  # 材料変動ランダムウォーク σ
    seed:          int   = 0,
) -> dict:
    """
    3 制御シナリオを同一ドリフト系列で比較して結果 dict を返す。
    """
    rng = np.random.default_rng(seed)

    # ── 真の装置状態生成 ──────────────────────────────────────────────────
    wear  = np.zeros(n_wafers)
    mat_v = np.ones(n_wafers)
    for k in range(1, n_wafers):
        wear[k]  = max(0.0, wear[k-1] + wear_drift
                       + rng.normal(0, wear_noise))
        mat_v[k] = np.clip(mat_v[k-1] + rng.normal(0, mat_noise), 0.5, 1.5)

    # ── EnKF × 2 + open-loop ─────────────────────────────────────────────
    enkf_vkx = BladeWearEnKF(sigma_vkx, n_ensemble, seed=1)
    enkf_opt = BladeWearEnKF(sigma_opt, n_ensemble, seed=2)

    chip_vkx  = np.zeros(n_wafers)
    chip_opt  = np.zeros(n_wafers)
    chip_open = np.zeros(n_wafers)
    feed_vkx  = np.full(n_wafers, FEED_NOMINAL)
    feed_opt  = np.full(n_wafers, FEED_NOMINAL)
    est_wear_vkx = np.zeros(n_wafers)
    est_wear_opt = np.zeros(n_wafers)

    for k in range(n_wafers):
        # ── VK-X 制御 ──────────────────────────────────────────────────
        chip_true_vkx  = chip_forward(feed_vkx[k], wear[k], mat_v[k])
        y_meas_vkx     = chip_true_vkx + rng.normal(0, sigma_vkx)
        enkf_vkx.predict()
        state_vkx      = enkf_vkx.update(feed_vkx[k], y_meas_vkx)
        chip_vkx[k]    = chip_true_vkx
        est_wear_vkx[k]= state_vkx[0]
        if k + 1 < n_wafers:
            feed_vkx[k+1] = feed_from_setpoint(state_vkx[0], state_vkx[1])

        # ── 光学顕微鏡制御 ───────────────────────────────────────────
        chip_true_opt  = chip_forward(feed_opt[k], wear[k], mat_v[k])
        y_meas_opt     = chip_true_opt + rng.normal(0, sigma_opt)
        enkf_opt.predict()
        state_opt      = enkf_opt.update(feed_opt[k], y_meas_opt)
        chip_opt[k]    = chip_true_opt
        est_wear_opt[k]= state_opt[0]
        if k + 1 < n_wafers:
            feed_opt[k+1] = feed_from_setpoint(state_opt[0], state_opt[1])

        # ── Open-loop (フィード固定) ──────────────────────────────────
        chip_open[k] = chip_forward(FEED_NOMINAL, wear[k], mat_v[k])

    rmse_vkx  = float(np.sqrt(np.mean((chip_vkx  - CHIP_SETPOINT)**2)))
    rmse_opt  = float(np.sqrt(np.mean((chip_opt  - CHIP_SETPOINT)**2)))
    rmse_open = float(np.sqrt(np.mean((chip_open - CHIP_SETPOINT)**2)))

    return dict(
        wafers=np.arange(1, n_wafers + 1),
        setpoint=CHIP_SETPOINT,
        chip_vkx=chip_vkx,   chip_opt=chip_opt,   chip_open=chip_open,
        feed_vkx=feed_vkx,   feed_opt=feed_opt,
        est_wear_vkx=est_wear_vkx, est_wear_opt=est_wear_opt,
        true_wear=wear,       true_mat=mat_v,
        rmse_vkx=rmse_vkx,   rmse_opt=rmse_opt,   rmse_open=rmse_open,
        improve_vkx_pct=(rmse_open - rmse_vkx) / rmse_open * 100,
        improve_opt_pct=(rmse_open - rmse_opt) / rmse_open * 100,
        sigma_vkx=sigma_vkx, sigma_opt=sigma_opt,
    )


# ════════════════════════════════════════════════════════════════════════════
# 4. 可視化
# ════════════════════════════════════════════════════════════════════════════

def plot_apc(res: dict, save_dir: str = OUT_DIR) -> None:
    os.makedirs(save_dir, exist_ok=True)
    w = res["wafers"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax1, ax2, ax3, ax4 = axes.flat

    # (A) チッピング幅トレース
    ax = ax1
    ax.plot(w, res["chip_open"], "C3-",  lw=1.2, alpha=0.7,  label="Open-loop")
    ax.plot(w, res["chip_opt"],  "C1-",  lw=1.4, alpha=0.85,
            label=f"EnKF + Opt. (σ={res['sigma_opt']:.2f}µm)")
    ax.plot(w, res["chip_vkx"],  "C0-",  lw=1.8,
            label=f"EnKF + VK-X (σ={res['sigma_vkx']:.2f}µm)")
    ax.axhline(res["setpoint"], color="k", lw=0.9, ls="--", label=f"Setpoint {res['setpoint']}µm")
    ax.set_xlabel("Wafer #")
    ax.set_ylabel("Chipping width [µm]")
    ax.set_title("(A) R2R controlled chipping width")
    ax.legend(fontsize=8)
    ax.set_ylim(0, None)

    # (B) RMSE 棒グラフ
    ax = ax2
    labels  = ["Open-loop", f"Opt. scope\n(σ={res['sigma_opt']:.2f}µm)",
               f"VK-X\n(σ={res['sigma_vkx']:.2f}µm)"]
    rmses   = [res["rmse_open"], res["rmse_opt"], res["rmse_vkx"]]
    colors  = ["C3", "C1", "C0"]
    bars = ax.bar(labels, rmses, color=colors, alpha=0.85, edgecolor="k", lw=0.7)
    for b, v in zip(bars, rmses):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE vs setpoint [µm]")
    ax.set_title("(B) Control performance comparison")
    ax.set_ylim(0, max(rmses) * 1.35)
    ax.text(0.98, 0.97,
            f"VK-X improvement: {res['improve_vkx_pct']:.1f}%\n"
            f"Opt. improvement: {res['improve_opt_pct']:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # (C) ブレード摩耗推定 vs 真値
    ax = ax3
    ax.plot(w, res["true_wear"],     "k-",  lw=1.5, label="True blade wear")
    ax.plot(w, res["est_wear_vkx"],  "C0--", lw=1.4,
            label=f"EnKF estimate (VK-X σ={res['sigma_vkx']:.2f}µm)")
    ax.plot(w, res["est_wear_opt"],  "C1--", lw=1.4,
            label=f"EnKF estimate (Opt. σ={res['sigma_opt']:.2f}µm)")
    ax.set_xlabel("Wafer #")
    ax.set_ylabel("Blade wear [a.u.]")
    ax.set_title("(C) Blade wear state estimation")
    ax.legend(fontsize=8)

    # (D) フィード速度調整
    ax = ax4
    ax.plot(w, res["feed_vkx"],  "C0-", lw=1.8,
            label=f"Feed (VK-X ctrl) σ={res['sigma_vkx']:.2f}µm")
    ax.plot(w, res["feed_opt"],  "C1-", lw=1.4,
            label=f"Feed (Opt. ctrl) σ={res['sigma_opt']:.2f}µm")
    ax.axhline(FEED_NOMINAL, color="k", lw=0.8, ls="--",
               label=f"Nominal {FEED_NOMINAL} mm/s")
    ax.axhline(FEED_LO,  color="gray", lw=0.5, ls=":")
    ax.axhline(FEED_HI,  color="gray", lw=0.5, ls=":")
    ax.set_xlabel("Wafer #")
    ax.set_ylabel("Feed rate [mm/s]")
    ax.set_title("(D) Corrective feed rate (control input)")
    ax.legend(fontsize=8)
    ax.set_ylim(FEED_LO * 0.8, FEED_HI * 1.05)

    fig.suptitle(
        "Keyence VK-X APC — EnKF R2R Blade Dicing Control\n"
        f"(n_wafers={len(w)},  setpoint={CHIP_SETPOINT}µm,  "
        f"VK-X RMSE={res['rmse_vkx']:.3f}µm  "
        f"vs Open-loop {res['rmse_open']:.3f}µm "
        f"[{res['improve_vkx_pct']:.1f}% improvement])",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "keyence_apc_control.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[keyence_apc] saved → {path}")


def print_report(res: dict) -> None:
    print("\n" + "=" * 60)
    print(f"{'Keyence APC — Run-to-Run Control Summary':^60}")
    print("=" * 60)
    print(f"\n  Wafers simulated : {len(res['wafers'])}")
    print(f"  Chip setpoint    : {res['setpoint']:.1f} µm")
    print(f"\n  CTRL-VKX  (σ={res['sigma_vkx']:.2f}µm)  "
          f"RMSE = {res['rmse_vkx']:.3f} µm  "
          f"(-{res['improve_vkx_pct']:.1f}% vs open)")
    print(f"  CTRL-OPT  (σ={res['sigma_opt']:.2f}µm)  "
          f"RMSE = {res['rmse_opt']:.3f} µm  "
          f"(-{res['improve_opt_pct']:.1f}% vs open)")
    print(f"  OPEN-LOOP            "
          f"RMSE = {res['rmse_open']:.3f} µm")
    print()


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Keyence VK-X APC simulation")
    ap.add_argument("--wafers",   type=int,   default=80)
    ap.add_argument("--ensemble", type=int,   default=30)
    ap.add_argument("--seed",     type=int,   default=0)
    ap.add_argument("--no-plot",  action="store_true")
    ap.add_argument("--meas-noise-vkx", type=float, default=TOOL_SIGMA["vkx_100x"])
    ap.add_argument("--meas-noise-opt", type=float, default=TOOL_SIGMA["opt_50x"])
    args = ap.parse_args()

    print(f"[keyence_apc] Simulating {args.wafers} wafers ...")
    res = run_apc(n_wafers=args.wafers, n_ensemble=args.ensemble,
                  sigma_vkx=args.meas_noise_vkx, sigma_opt=args.meas_noise_opt,
                  seed=args.seed)
    print_report(res)
    if not args.no_plot:
        plot_apc(res)
    print("[keyence_apc] Done.")


if __name__ == "__main__":
    main()
