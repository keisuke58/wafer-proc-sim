"""
論文用 Figure 生成スクリプト
paper/sic_gp_quality/

Figure 2: LOO scatter plot — All grades (R²=0.32) vs A/B grades (R²=0.82)
Figure 1: Data quality grade distribution (bar chart + noise assignment)

Usage:
    python paper/sic_gp_quality/gen_figures.py
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from validation.experimental_data import CHIPPING_DATA
from validation.experimental_data import point_noise

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = ["cut_depth_um", "blade_W_um", "feed_mm_s", "spindle_rpm"]
TARGET_COL   = "chipping_um"

GRADE_COLORS = {"A": "#2166ac", "B": "#4dac26", "C": "#f4a582", "D": "#d6604d"}

# ── Data loading ──────────────────────────────────────────────────────────────

def load(quality_filter: str = "all"):
    ok = {"all": set("ABCD"), "AB": set("AB"), "A": {"A"}}[quality_filter]
    rows, alphas = [], []
    for e in CHIPPING_DATA:
        if e.get("cut_type") == "complete":
            continue
        if e.get("quality", "B") not in ok:
            continue
        rows.append(e)
        alphas.append(point_noise(e) ** 2)
    df = pd.DataFrame(rows)
    X  = df[FEATURE_COLS].values.astype(float)
    y  = df[TARGET_COL].values.astype(float)
    return X, y, df, np.array(alphas)


def build_kernel():
    return ConstantKernel(10.0, (0.1, 1e3)) * RBF(
        length_scale=[50., 50., 2., 10000.],
        length_scale_bounds=[(5,500),(5,500),(0.1,20),(500,50000)],
    ) + WhiteKernel(noise_level=0.5, noise_level_bounds=(0.05, 30.))


def loo_predict(X, y, alpha_vec):
    loo   = LeaveOneOut()
    preds = np.zeros_like(y)
    for tr, te in loo.split(X):
        sx = StandardScaler(); sy = StandardScaler()
        Xs = sx.fit_transform(X[tr])
        ys = sy.fit_transform(y[tr].reshape(-1,1)).ravel()
        alpha_s = alpha_vec[tr] / sy.scale_[0]**2
        gp = GaussianProcessRegressor(
            kernel=build_kernel(), n_restarts_optimizer=10,
            normalize_y=False, alpha=alpha_s,
        )
        gp.fit(Xs, ys)
        mu = gp.predict(sx.transform(X[te]))
        preds[te] = sy.inverse_transform(mu.reshape(-1,1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2   = float(r2_score(y, preds))
    return preds, rmse, r2


# ════════════════════════════════════════════════════════════════════════════
# Figure 1: Data quality distribution
# ════════════════════════════════════════════════════════════════════════════

def fig1_quality_distribution():
    rows = [e for e in CHIPPING_DATA if e.get("cut_type") != "complete"]
    df   = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: grade counts by source
    ax = axes[0]
    sources = df["source"].unique()
    grades  = ["A", "B", "C", "D"]
    x_pos   = np.arange(len(grades))
    width   = 0.35
    src_colors = {"Micro2026": "#2166ac", "Mat2022": "#d6604d",
                  "AIP2021": "#4dac26"}
    bottom = np.zeros(len(grades))
    for src in sources:
        counts = [len(df[(df["source"]==src) & (df["quality"]==g)]) for g in grades]
        ax.bar(x_pos, counts, width=0.6, bottom=bottom,
               label=src, color=src_colors.get(src,"gray"),
               edgecolor="k", linewidth=0.5)
        bottom += np.array(counts)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Grade {g}" for g in grades], fontsize=11)
    ax.set_ylabel("Number of data points", fontsize=11)
    ax.set_title("(a) Data quality grade distribution", fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # Right: noise sigma per grade
    ax2 = axes[1]
    noise_map = {"A": 1.0, "B": 1.5, "C": 2.5, "D": 4.0}
    bars = ax2.bar(grades,
                   [noise_map[g] for g in grades],
                   color=[GRADE_COLORS[g] for g in grades],
                   edgecolor="k", linewidth=0.6, width=0.5)
    for bar, g in zip(bars, grades):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 f"$\\sigma$={noise_map[g]} µm",
                 ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("Measurement noise $\\sigma$ [µm]", fontsize=11)
    ax2.set_title("(b) Per-grade noise assignment", fontsize=11)
    ax2.set_ylim(0, 5); ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Figure 1: Experimental dataset composition\n"
        "Micro2026 (Micromachines, DOI:10.3390/mi17020187) + "
        "Mat2022 (Materials, DOI:10.3390/ma15228083)",
        fontsize=10,
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_quality_distribution.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Fig1 -> {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: LOO scatter — All vs AB
# ════════════════════════════════════════════════════════════════════════════

def fig2_loo_scatter():
    print("Running LOO for 'all' grades ...")
    Xa, ya, dfa, aa = load("all")
    preds_all, rmse_all, r2_all = loo_predict(Xa, ya, aa)

    print("Running LOO for 'AB' grades ...")
    Xab, yab, dfab, aab = load("AB")
    preds_ab, rmse_ab, r2_ab = loo_predict(Xab, yab, aab)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    lim = (0, 30)

    for ax, y_true, y_pred, df_q, rmse, r2, title, n in [
        (axes[0], ya,  preds_all, dfa,  rmse_all, r2_all,
         "All grades (A–D)", len(ya)),
        (axes[1], yab, preds_ab,  dfab, rmse_ab,  r2_ab,
         "High-quality only (A–B)", len(yab)),
    ]:
        # scatter by grade
        for grade, grp in df_q.groupby("quality"):
            idx = grp.index
            ax.scatter(
                y_true[idx], y_pred[idx],
                color=GRADE_COLORS.get(grade, "gray"),
                edgecolors="k", linewidths=0.5, s=70, zorder=5,
                label=f"Grade {grade} (n={len(idx)})",
            )

        # error bars from noise
        ax.errorbar(
            y_true, y_pred,
            xerr=[point_noise(e) for e in
                  [r for r in CHIPPING_DATA
                   if r.get("cut_type")!="complete"
                   and r.get("quality","B") in
                   (set("ABCD") if "All" in title else set("AB"))]],
            fmt="none", ecolor="gray", alpha=0.4, capsize=2,
        )

        # diagonal
        ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
        ax.fill_between(lim,
                         [l - 3 for l in lim],
                         [l + 3 for l in lim],
                         alpha=0.07, color="gray", label="±3 µm band")

        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Measured chipping [µm]", fontsize=11)
        ax.set_ylabel("LOO-predicted chipping [µm]", fontsize=11)
        ax.set_title(
            f"({['a','b'][axes.tolist().index(ax)]}) {title}\n"
            f"n={n},  LOO RMSE = {rmse:.2f} µm,  R² = {r2:.2f}",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)

        # annotation box
        ax.text(0.97, 0.05,
                f"RMSE = {rmse:.2f} µm\nR² = {r2:.2f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="gray", alpha=0.8))

    fig.suptitle(
        "Figure 2: LOO cross-validation — impact of data quality stratification\n"
        "Heteroscedastic GP, 4H-SiC blade dicing (Micro2026 + Mat2022)",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_loo_scatter.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Fig2 -> {path}")
    print(f"     All: RMSE={rmse_all:.2f} µm, R²={r2_all:.3f}")
    print(f"     AB:  RMSE={rmse_ab:.2f} µm, R²={r2_ab:.3f}")
    return path, rmse_all, r2_all, rmse_ab, r2_ab


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Generating paper figures ===\n")
    fig1_quality_distribution()
    fig2_loo_scatter()
    print("\nDone. Figures in:", OUT_DIR)
