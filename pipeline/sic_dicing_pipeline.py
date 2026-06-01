"""
SiC Dicing Optimization Pipeline — Classical ML Stack
======================================================
End-to-end pipeline for 4H-SiC blade dicing optimization.

Architecture:
    Step 1  Experimental data   (quality-aware, N=25, quality A/B/C/D)
    Step 2  Heteroscedastic GP  (per-point noise from data quality grades)
    Step 3  EnKF state estimation (blade wear + crystal variation tracking)
    Step 4  Recipe optimization  (Simulated Annealing / QUBO-ready)
    Step 5  4-panel results figure

Usage:
    python pipeline/sic_dicing_pipeline.py
    python pipeline/sic_dicing_pipeline.py --no-plots
    python pipeline/sic_dicing_pipeline.py --quality AB   # high-quality data only
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from validation.experimental_data import CHIPPING_DATA, point_noise

OUT_DIR = os.path.join(ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = ["cut_depth_um", "blade_W_um", "feed_mm_s", "spindle_rpm"]
TARGET_COL   = "chipping_um"

# Reference recipe (Micro2026 conditions)
REF = {"cut_depth_um": 200.0, "blade_W_um": 23.0,
       "feed_mm_s": 1.0, "spindle_rpm": 30000.0}


# ════════════════════════════════════════════════════════════════════════════
# Step 1: Data loading
# ════════════════════════════════════════════════════════════════════════════

def load_experimental_data(quality_filter: str = "all") -> tuple:
    """
    Load experimental chipping data with quality filtering.

    quality_filter: "all" | "AB" | "A"
        "all": all grades (N≈25)
        "AB" : only directly measured + digitized (N≈11)
        "A"  : only directly measured with reported std (N≈6)

    Returns:
        X          : (N, 4) feature matrix [depth, bw, feed, rpm]
        y          : (N,)   chipping [µm]
        alpha_vec  : (N,)   per-point noise variance [µm²]
        sources    : list of source labels for plotting
        quality    : list of quality grades
    """
    allowed = {"all": set("ABCD"), "AB": set("AB"), "A": {"A"}}
    ok_grades = allowed.get(quality_filter, set("ABCD"))

    rows, alpha_list, sources, grades = [], [], [], []
    for entry in CHIPPING_DATA:
        if entry.get("cut_type") == "complete":
            continue
        if entry.get("quality", "B") not in ok_grades:
            continue
        rows.append(entry)
        alpha_list.append(point_noise(entry) ** 2)

    import pandas as pd
    df = pd.DataFrame(rows)
    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COL].values.astype(float)
    alpha_vec = np.array(alpha_list)
    return X, y, alpha_vec, df["source"].tolist(), df["quality"].tolist()


# ════════════════════════════════════════════════════════════════════════════
# Step 2: Heteroscedastic GP Surrogate
# ════════════════════════════════════════════════════════════════════════════

def _build_kernel():
    return (
        ConstantKernel(10.0, (0.1, 1e3))
        * RBF(
            length_scale=[50.0, 50.0, 2.0, 10000.0],
            length_scale_bounds=[(5, 500), (5, 500), (0.1, 20), (500, 50000)],
        )
        + WhiteKernel(noise_level=0.5, noise_level_bounds=(0.01, 30.0))
    )


class HeteroscedasticGP:
    """
    GP surrogate with per-point noise derived from data quality grades.

    Quality A (direct measurement): noise = reported std
    Quality B (digitized figure):   noise = 1.5 µm
    Quality C (interpolated):       noise = 2.5 µm
    Quality D (rough estimate):     noise = 4.0 µm
    """

    def __init__(self):
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.gp = GaussianProcessRegressor(
            kernel=_build_kernel(),
            n_restarts_optimizer=15,
            normalize_y=False,
            alpha=0.0,  # overridden per-point in fit()
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            alpha_vec: np.ndarray = None) -> "HeteroscedasticGP":
        Xs = self.scaler_x.fit_transform(X)
        ys = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        scale_sq = float(self.scaler_y.scale_[0]) ** 2
        if alpha_vec is not None:
            self.gp.alpha = alpha_vec / scale_sq
        else:
            self.gp.alpha = 1e-6
        self.gp.fit(Xs, ys)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray,
                return_std: bool = False):
        Xs = self.scaler_x.transform(X)
        mu, sigma = self.gp.predict(Xs, return_std=True)
        mu    = self.scaler_y.inverse_transform(mu.reshape(-1, 1)).ravel()
        sigma = sigma * float(self.scaler_y.scale_[0])
        return (mu, sigma) if return_std else mu

    def loo_cv(self, X: np.ndarray, y: np.ndarray,
               alpha_vec: np.ndarray = None) -> dict:
        loo   = LeaveOneOut()
        preds = np.zeros_like(y)
        for tr, te in loo.split(X):
            tmp = HeteroscedasticGP()
            tmp.fit(X[tr], y[tr],
                    alpha_vec[tr] if alpha_vec is not None else None)
            preds[te] = tmp.predict(X[te])
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        r2   = float(r2_score(y, preds))
        return {"rmse": rmse, "r2": r2, "loo_preds": preds}


# ════════════════════════════════════════════════════════════════════════════
# Step 3: EnKF State Estimation (GP-coupled)
# ════════════════════════════════════════════════════════════════════════════

# Physics-based base chipping model (for EnKF forward model)
def _base_chipping(depth_um: float, blade_w_um: float) -> float:
    depth_term  = 1.2 + 0.035 * (depth_um - 80.0) / 310.0 * 10.0
    width_term  = 0.8 + 0.40  * (blade_w_um - 20.0) / 35.0
    interaction = 1.0 + 0.12  * (
        (depth_um - 80.0) / 310.0 * (blade_w_um - 20.0) / 35.0
    )
    return depth_term * width_term * interaction * 3.5


def run_enkf(n_wafers: int = 60,
             obs_noise: float = 1.5,
             n_ensemble: int = 100,
             seed: int = 42) -> dict:
    """
    Simulate n_wafers of blade dicing and estimate hidden state with EnKF.

    State: x = [d_eff, alpha_w, k_ic]
        d_eff   : depth ratio (actual / nominal), starts ~1.0
        alpha_w : blade wear factor, 1.0 → ~0.8 over 60 wafers
        k_ic    : local K_Ic factor (crystal orientation), ~0.84–1.00

    Returns dict with time series for analysis.
    """
    rng = np.random.default_rng(seed)
    D_NOM, BLADE_W = REF["cut_depth_um"], REF["blade_W_um"]

    def obs_model(state: np.ndarray) -> float:
        d_eff, alpha_w, k_ic = state
        actual = D_NOM * np.clip(d_eff, 0.5, 1.5)
        base   = _base_chipping(actual, BLADE_W)
        return base / (np.clip(alpha_w, 0.1, 2.0) * np.clip(k_ic, 0.5, 1.5))

    # Process noise Q and measurement noise R
    Q = np.diag([1e-4, 5e-5, 1e-4])
    R = obs_noise ** 2
    gamma_w = 2e-3

    # Initialise ensemble
    x0  = np.array([1.0, 1.0, 1.0])
    P0  = np.diag([0.02 ** 2, 0.02 ** 2, 0.03 ** 2])
    L0  = np.linalg.cholesky(P0)
    ens = x0 + (L0 @ rng.standard_normal((3, n_ensemble))).T  # (N,3)

    # True hidden trajectory (ground truth for simulation)
    true_d   = np.ones(n_wafers)
    true_aw  = 1.0 - gamma_w * np.arange(n_wafers)           # linear wear
    true_kic = 1.0 - 0.06 * np.sin(np.linspace(0, 2*np.pi, n_wafers))

    # Storage
    means    = np.zeros((n_wafers, 3))
    stds     = np.zeros((n_wafers, 3))
    chip_obs = np.zeros(n_wafers)
    chip_est = np.zeros(n_wafers)
    chip_nom = np.zeros(n_wafers)

    for t in range(n_wafers):
        # Simulate observation from true state
        true_state = np.array([true_d[t], true_aw[t], true_kic[t]])
        y_true = obs_model(true_state) + rng.normal(0, obs_noise)
        chip_obs[t] = y_true
        chip_nom[t] = obs_model(x0)

        # ── Forecast step ────────────────────────────────────────────────
        xf       = ens.copy()
        xf[:, 1] -= gamma_w                                   # wear drift
        LQ       = np.linalg.cholesky(Q)
        xf      += (LQ @ rng.standard_normal((3, n_ensemble))).T

        # ── Analysis step ────────────────────────────────────────────────
        v     = rng.normal(0, obs_noise, n_ensemble)
        yf    = np.array([obs_model(xf[i]) for i in range(n_ensemble)])
        yf_b  = yf.mean()
        dX    = xf - xf.mean(axis=0, keepdims=True)
        dY    = yf - yf_b
        Pxy   = (dX.T @ dY) / (n_ensemble - 1)
        Pyy   = float((dY @ dY) / (n_ensemble - 1))
        K     = Pxy / (Pyy + R)
        ens   = xf + np.outer(y_true - (yf + v), K)

        means[t] = ens.mean(axis=0)
        stds[t]  = ens.std(axis=0)
        chip_est[t] = obs_model(means[t])

    rmse_enkf = float(np.sqrt(np.mean((chip_obs - chip_est) ** 2)))
    rmse_nom  = float(np.sqrt(np.mean((chip_obs - chip_nom) ** 2)))

    return {
        "means": means, "stds": stds,
        "chip_obs": chip_obs, "chip_est": chip_est, "chip_nom": chip_nom,
        "true_d": true_d, "true_aw": true_aw, "true_kic": true_kic,
        "rmse_enkf": rmse_enkf, "rmse_nom": rmse_nom,
        "n_wafers": n_wafers,
    }


# ════════════════════════════════════════════════════════════════════════════
# Step 4: Recipe Optimization (SA on GP surface)
# ════════════════════════════════════════════════════════════════════════════

def optimize_recipe(gp_model: HeteroscedasticGP,
                    n_restarts: int = 5,
                    seed: int = 0) -> dict:
    """
    Find the (depth, blade_W) that minimizes predicted chipping.
    Feed and spindle fixed at reference conditions.

    Uses random-restart gradient-free search (scipy L-BFGS-B on GP mean).
    """
    from scipy.optimize import minimize

    best_x, best_val = None, np.inf
    rng = np.random.default_rng(seed)

    depth_bounds = (80.0, 390.0)
    width_bounds = (20.0, 55.0)

    def objective(params):
        d, w = params
        x = np.array([[d, w, REF["feed_mm_s"], REF["spindle_rpm"]]])
        return float(gp_model.predict(x)[0])

    for _ in range(n_restarts):
        x0 = [
            rng.uniform(*depth_bounds),
            rng.uniform(*width_bounds),
        ]
        res = minimize(objective, x0,
                       bounds=[depth_bounds, width_bounds],
                       method="L-BFGS-B")
        if res.fun < best_val:
            best_val = res.fun
            best_x   = res.x

    # Full 2D response surface for plotting
    d_grid = np.linspace(*depth_bounds, 80)
    w_grid = np.linspace(*width_bounds, 80)
    D, W   = np.meshgrid(d_grid, w_grid)
    X_grid = np.column_stack([
        D.ravel(), W.ravel(),
        np.full(D.size, REF["feed_mm_s"]),
        np.full(D.size, REF["spindle_rpm"]),
    ])
    mu_grid, std_grid = gp_model.predict(X_grid, return_std=True)

    return {
        "opt_depth": float(best_x[0]),
        "opt_width": float(best_x[1]),
        "opt_chip":  float(best_val),
        "D": D, "W": W,
        "mu_grid": mu_grid.reshape(D.shape),
        "std_grid": std_grid.reshape(D.shape),
    }


# ════════════════════════════════════════════════════════════════════════════
# Step 5: Visualisation
# ════════════════════════════════════════════════════════════════════════════

QUALITY_COLOR = {"A": "#2166ac", "B": "#4dac26", "C": "#f1a340", "D": "#d73027"}
SOURCE_MARKER = {"Micro2026": "o", "Mat2022": "s", "AIP2021": "^"}


def plot_pipeline_results(
    gp_loo: dict,
    y: np.ndarray,
    sources: list,
    quality: list,
    enkf: dict,
    opt: dict,
    save: bool = True,
) -> str:
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── Panel 1: LOO scatter ─────────────────────────────────────────────────
    preds = gp_loo["loo_preds"]
    for i, (src, q) in enumerate(zip(sources, quality)):
        ax1.scatter(y[i], preds[i],
                    color=QUALITY_COLOR.get(q, "gray"),
                    marker=SOURCE_MARKER.get(src, "o"),
                    s=70, edgecolors="k", lw=0.5, zorder=3)
    lim = [0, max(y.max(), preds.max()) * 1.1]
    ax1.plot(lim, lim, "k--", lw=1.2)
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("Measured chipping [µm]", fontsize=10)
    ax1.set_ylabel("LOO predicted [µm]", fontsize=10)
    ax1.set_title(
        f"(a) GP Surrogate LOO\n"
        f"RMSE={gp_loo['rmse']:.2f} µm  R²={gp_loo['r2']:.3f}",
        fontsize=10)
    # Quality legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=QUALITY_COLOR[q], markeredgecolor="k",
               markersize=7, label=f"Grade {q}")
        for q in ["A", "B", "C", "D"]
    ]
    ax1.legend(handles=handles, fontsize=7, loc="upper left")
    ax1.grid(alpha=0.2)

    # ── Panel 2: EnKF state tracking ────────────────────────────────────────
    wafers = np.arange(enkf["n_wafers"])
    names  = ["d_eff", r"$\alpha_w$", r"$k_{Ic}$"]
    true_vals = [enkf["true_d"], enkf["true_aw"], enkf["true_kic"]]
    colors    = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for k in range(3):
        mu  = enkf["means"][:, k]
        sig = enkf["stds"][:, k]
        ax2.plot(wafers, mu, color=colors[k], lw=1.8, label=f"EnKF {names[k]}")
        ax2.fill_between(wafers, mu - sig, mu + sig,
                         color=colors[k], alpha=0.18)
        ax2.plot(wafers, true_vals[k], color=colors[k],
                 ls="--", lw=1.0, alpha=0.6)
    ax2.set_xlabel("Wafer index", fontsize=10)
    ax2.set_ylabel("State value", fontsize=10)
    ax2.set_title(
        f"(b) EnKF State Estimation (N={enkf['n_wafers']} wafers)\n"
        f"RMSE: EnKF {enkf['rmse_enkf']:.2f} µm  vs Nominal {enkf['rmse_nom']:.2f} µm",
        fontsize=10)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.2)

    # ── Panel 3: Chipping prediction vs observation ──────────────────────────
    wafers = np.arange(enkf["n_wafers"])
    ax3.scatter(wafers, enkf["chip_obs"], s=18, color="#666",
                alpha=0.7, label="Observed", zorder=3)
    ax3.plot(wafers, enkf["chip_est"], color="#1f77b4",
             lw=2.0, label=f"EnKF ({enkf['rmse_enkf']:.2f} µm)")
    ax3.plot(wafers, enkf["chip_nom"], color="#d62728",
             lw=1.5, ls="--", label=f"Nominal ({enkf['rmse_nom']:.2f} µm)")
    ax3.axhline(15.0, color="#d73027", lw=1.0, ls=":", alpha=0.8,
                label="15 µm threshold")
    ax3.set_xlabel("Wafer index", fontsize=10)
    ax3.set_ylabel("Chipping [µm]", fontsize=10)
    ax3.set_title(
        "(c) EnKF Chipping Tracking\n"
        f"RMSE improvement: {(1 - enkf['rmse_enkf']/enkf['rmse_nom'])*100:.0f}%",
        fontsize=10)
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.2)

    # ── Panel 4: GP response surface + optimum ──────────────────────────────
    im = ax4.contourf(opt["D"], opt["W"], opt["mu_grid"],
                      levels=20, cmap="YlOrRd")
    plt.colorbar(im, ax=ax4, label="Predicted chipping [µm]")
    ax4.contour(opt["D"], opt["W"], opt["mu_grid"],
                levels=[15.0], colors="white", linewidths=1.8, linestyles="--")
    ax4.scatter(opt["opt_depth"], opt["opt_width"],
                s=180, color="lime", edgecolors="k", lw=1.5,
                zorder=5, label=f"Optimum ({opt['opt_chip']:.1f} µm)")
    ax4.set_xlabel("Cut Depth [µm]", fontsize=10)
    ax4.set_ylabel("Blade Width [µm]", fontsize=10)
    ax4.set_title(
        f"(d) GP Response Surface\n"
        f"Optimal: depth={opt['opt_depth']:.0f} µm, "
        f"width={opt['opt_width']:.0f} µm",
        fontsize=10)
    ax4.legend(fontsize=8)
    ax4.text(opt["D"].min() + 10, opt["W"].max() - 3,
             "── 15 µm limit", color="white", fontsize=8, va="top")

    fig.suptitle(
        "SiC Dicing Optimization Pipeline\n"
        "Experimental Data → Heteroscedastic GP → EnKF → Recipe Optimization",
        fontsize=12, fontweight="bold")

    if save:
        path = os.path.join(OUT_DIR, "sic_pipeline_results.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Saved: {path}")
        plt.close()
        return path
    plt.show()
    return ""


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def run(quality: str = "all", make_plots: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    log = print if verbose else lambda *a, **k: None

    log(f"\n{'═'*60}")
    log(f"  SiC Dicing Pipeline  (quality={quality})")
    log(f"{'═'*60}")

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    log("\n[Step 1] Loading experimental data …")
    X, y, alpha_vec, sources, qual = load_experimental_data(quality)
    log(f"  N={len(y)}  chipping {y.min():.1f}–{y.max():.1f} µm")
    log(f"  Grade breakdown: " +
        "  ".join(f"{g}:{qual.count(g)}" for g in "ABCD" if qual.count(g)))

    # ── Step 2: GP surrogate ─────────────────────────────────────────────────
    log("\n[Step 2] Heteroscedastic GP surrogate …")
    gp = HeteroscedasticGP()
    loo = gp.loo_cv(X, y, alpha_vec)
    log(f"  LOO RMSE = {loo['rmse']:.2f} µm   R² = {loo['r2']:.4f}")
    gp.fit(X, y, alpha_vec)
    log(f"  Kernel: {gp.gp.kernel_}")

    # ── Step 3: EnKF ─────────────────────────────────────────────────────────
    log("\n[Step 3] EnKF state estimation …")
    enkf = run_enkf(n_wafers=80, obs_noise=1.5, n_ensemble=100)
    improv = (1 - enkf["rmse_enkf"] / enkf["rmse_nom"]) * 100
    log(f"  RMSE: EnKF={enkf['rmse_enkf']:.2f} µm  Nominal={enkf['rmse_nom']:.2f} µm")
    log(f"  Improvement: {improv:.0f}%")

    # ── Step 4: Recipe optimization ──────────────────────────────────────────
    log("\n[Step 4] Recipe optimization (GP + L-BFGS-B) …")
    opt = optimize_recipe(gp)
    log(f"  Optimal depth = {opt['opt_depth']:.0f} µm")
    log(f"  Optimal width = {opt['opt_width']:.0f} µm")
    log(f"  Predicted chipping = {opt['opt_chip']:.2f} µm")

    # ── Step 5: Plots ─────────────────────────────────────────────────────────
    if make_plots:
        log("\n[Step 5] Saving results figure …")
        plot_pipeline_results(loo, y, sources, qual, enkf, opt, save=True)

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.1f}s")

    return {
        "n_data": len(y),
        "gp_rmse": loo["rmse"],
        "gp_r2":   loo["r2"],
        "enkf_rmse": enkf["rmse_enkf"],
        "nom_rmse":  enkf["rmse_nom"],
        "enkf_improvement_pct": improv,
        "opt_depth": opt["opt_depth"],
        "opt_width": opt["opt_width"],
        "opt_chip":  opt["opt_chip"],
    }


def main():
    ap = argparse.ArgumentParser(
        description="SiC Dicing Optimization Pipeline")
    ap.add_argument("--quality", default="all",
                    choices=["all", "AB", "A"],
                    help="Data quality filter (default: all)")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip figure generation")
    args = ap.parse_args()

    results = run(quality=args.quality, make_plots=not args.no_plots)

    print(f"\n{'═'*60}")
    print(f"  Pipeline Summary")
    print(f"{'═'*60}")
    print(f"  Data points      : {results['n_data']}")
    print(f"  GP LOO RMSE      : {results['gp_rmse']:.2f} µm")
    print(f"  GP LOO R²        : {results['gp_r2']:.4f}")
    print(f"  EnKF RMSE        : {results['enkf_rmse']:.2f} µm")
    print(f"  Nominal RMSE     : {results['nom_rmse']:.2f} µm")
    print(f"  EnKF improvement : {results['enkf_improvement_pct']:.0f}%")
    print(f"  Optimal depth    : {results['opt_depth']:.0f} µm")
    print(f"  Optimal width    : {results['opt_width']:.0f} µm")
    print(f"  Optimal chipping : {results['opt_chip']:.2f} µm")


if __name__ == "__main__":
    main()
