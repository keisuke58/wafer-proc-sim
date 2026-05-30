"""
Fusion GP: experimental chipping data (26 pts) + FEM deletion_fraction (5 pts).

Strategy:
  1. Calibrate: fit deletion_fraction → chipping_um using depth-matched
     Micro2026 points (blade_W=23µm, feed=1mm/s, spindle=30krpm).
  2. Convert all 5 FEM points to chipping_um scale.
  3. Retrain ExperimentalGPSurrogate on 26 exp + 5 FEM points (~31 total).
  4. Compare LOO metrics vs. experimental-only model.
  5. Plot: depth sweep (exp GP vs fusion GP) + depth heatmap.

Usage:
    python ml/train_fusion_gp.py [--plot] [--loo]
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from validation.experimental_data import CHIPPING_DATA
from ml.train_from_experimental import (
    ExperimentalGPSurrogate, FEATURE_COLS, TARGET_COL, MODEL_PATH, REF,
    _sweep_array,
)

FEM_CSV   = os.path.join(os.path.dirname(__file__), "..", "results",
                          "parametric_summary_extended.csv")
FUSION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..",
                                  "results", "gp_fusion.pkl")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Step 1: Calibration ────────────────────────────────────────────────────────

def calibrate_del_frac_to_chipping(fem_df: pd.DataFrame,
                                    exp_df: pd.DataFrame) -> LinearRegression:
    """
    Fit chipping_um = a * deletion_fraction + b using Micro2026 depth sweep
    points that match FEM conditions (blade_W=23, feed=1.0, spindle=30000).
    """
    ref_mask = (
        (exp_df["blade_W_um"] == 23) &
        (exp_df["feed_mm_s"]  == 1.0) &
        (exp_df["spindle_rpm"] == 30000)
    )
    ref_exp = exp_df[ref_mask].copy()

    # Interpolate FEM deletion_fraction at each experimental depth
    fem_sorted = fem_df.sort_values("cut_depth_um")
    depths_fem = fem_sorted["cut_depth_um"].values.astype(float)
    delfrac_fem = fem_sorted["deletion_fraction"].values.astype(float)

    # Only use exp points within FEM depth range
    d_min, d_max = depths_fem.min(), depths_fem.max()
    ref_exp = ref_exp[
        (ref_exp["cut_depth_um"] >= d_min) &
        (ref_exp["cut_depth_um"] <= d_max)
    ]

    if len(ref_exp) < 2:
        print("[!] Not enough calibration points — using identity scaling")
        return None

    df_interp = np.interp(ref_exp["cut_depth_um"].values, depths_fem, delfrac_fem)
    chip_vals  = ref_exp["chipping_um"].values

    cal = LinearRegression()
    cal.fit(df_interp.reshape(-1, 1), chip_vals)
    print(f"  Calibration: chipping = {cal.coef_[0]:.2f} × del_frac "
          f"+ {cal.intercept_:.2f}  (n={len(ref_exp)})")
    return cal


# ── Step 2: Build fused dataset ───────────────────────────────────────────────

def build_fused_dataset(fem_df: pd.DataFrame,
                         exp_df: pd.DataFrame,
                         cal):
    """
    Append FEM rows (converted to chipping_um) to experimental DataFrame.
    FEM conditions: feed=0.5 mm/s, spindle=N/A → use Micro2026 reference values.
    """
    fem_rows = []
    for _, row in fem_df.iterrows():
        del_frac = row["deletion_fraction"]
        if del_frac == 0.0:
            continue  # skip empty/zero rows

        if cal is not None:
            chip_est = float(cal.predict([[del_frac]])[0])
        else:
            chip_est = del_frac * 20.0  # rough scale if no calibration

        chip_est = max(chip_est, 0.5)  # physical lower bound

        fem_rows.append({
            "source":         "FEM",
            "material":       "4H-SiC",
            "blade_W_um":     float(row["blade_W_um"]),
            "cut_depth_um":   float(row["cut_depth_um"]),
            "feed_mm_s":      0.5,   # FEM plunge feed speed
            "spindle_rpm":    30000.0,  # assume Micro2026 reference
            "chipping_um":    chip_est,
            "chipping_std_um": None,
            "notes":          f"FEM-derived, del_frac={del_frac:.4f}",
            "deletion_fraction": del_frac,
        })

    fem_extra = pd.DataFrame(fem_rows)
    fused = pd.concat([exp_df, fem_extra], ignore_index=True)
    print(f"  Fused dataset: {len(exp_df)} exp + {len(fem_extra)} FEM "
          f"= {len(fused)} total")
    return fused, fem_extra


# ── Step 3: Plot comparison ───────────────────────────────────────────────────

def plot_comparison(model_exp: ExperimentalGPSurrogate,
                    model_fus: ExperimentalGPSurrogate,
                    exp_df: pd.DataFrame,
                    fem_extra: pd.DataFrame,
                    out_dir: str = RESULTS_DIR):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    depths = np.linspace(60, 420, 120)
    X_sw   = _sweep_array("cut_depth_um", depths,
                           {f: REF[f] for f in FEATURE_COLS if f != "cut_depth_um"})

    mu_exp, sig_exp = model_exp.predict(X_sw, return_std=True)
    mu_fus, sig_fus = model_fus.predict(X_sw, return_std=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(depths, mu_exp, color="#2166ac", lw=2, label="Exp-only GP (26 pts)")
    ax.fill_between(depths, mu_exp - 2*sig_exp, mu_exp + 2*sig_exp,
                    alpha=0.15, color="#2166ac")

    ax.plot(depths, mu_fus, color="#d6604d", lw=2, ls="--",
            label=f"Fusion GP ({len(exp_df)}+{len(fem_extra)} pts)")
    ax.fill_between(depths, mu_fus - 2*sig_fus, mu_fus + 2*sig_fus,
                    alpha=0.12, color="#d6604d")

    # Experimental scatter (blade_W=23, feed=1.0, spindle=30k)
    ref_mask = (
        (exp_df["blade_W_um"] == 23) &
        (exp_df["feed_mm_s"]  == 1.0) &
        (exp_df["spindle_rpm"] == 30000)
    )
    sub = exp_df[ref_mask]
    ax.scatter(sub["cut_depth_um"], sub["chipping_um"],
               color="#2166ac", s=70, zorder=5, edgecolors="k", lw=0.6,
               label="Micro2026 (exp)")

    # FEM points
    if len(fem_extra):
        ax.scatter(fem_extra["cut_depth_um"], fem_extra["chipping_um"],
                   color="#d6604d", s=80, marker="^", zorder=6, edgecolors="k",
                   lw=0.6, label="FEM-derived")

    ax.axhline(15.0, color="gray", ls=":", lw=1.2, label="15 µm threshold")
    ax.set_xlabel("Cut Depth [µm]", fontsize=12)
    ax.set_ylabel("Front Chipping [µm]", fontsize=12)
    ax.set_title("Fusion GP: Experimental + FEM Data\n"
                 "(blade_W=23µm, feed=1mm/s, spindle=30krpm fixed)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)

    fig_path = os.path.join(out_dir, "gp_fusion_depth_sweep.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Fusion depth sweep → {fig_path}")

    # ── Second plot: deletion_fraction vs chipping calibration ────────────────
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fem_sorted = fem_extra.sort_values("cut_depth_um")
    ax2.scatter(fem_sorted["deletion_fraction"], fem_sorted["chipping_um"],
                c=fem_sorted["cut_depth_um"], cmap="viridis", s=100,
                edgecolors="k", lw=0.6, zorder=5)
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(fem_sorted["cut_depth_um"].min(),
                                                   fem_sorted["cut_depth_um"].max()))
    plt.colorbar(sm, ax=ax2, label="Cut Depth [µm]")
    ax2.set_xlabel("FEM Deletion Fraction", fontsize=11)
    ax2.set_ylabel("Calibrated Chipping [µm]", fontsize=11)
    ax2.set_title("FEM Calibration: deletion_fraction → chipping", fontsize=10)
    ax2.grid(alpha=0.25)
    fig2_path = os.path.join(out_dir, "gp_fusion_calibration.png")
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Calibration plot → {fig2_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--loo",  action="store_true")
    args = parser.parse_args()

    # Load experimental data
    exp_df = pd.DataFrame(CHIPPING_DATA)
    X_exp  = exp_df[FEATURE_COLS].values.astype(float)
    y_exp  = exp_df[TARGET_COL].values.astype(float)
    print(f"[*] Experimental: {len(exp_df)} points, chipping {y_exp.min():.1f}–{y_exp.max():.1f} µm")

    # Load FEM data (filter rows with valid deletion_fraction)
    fem_df = pd.read_csv(FEM_CSV)
    fem_df = fem_df[fem_df["deletion_fraction"] > 0.0].copy()
    print(f"[*] FEM: {len(fem_df)} valid points")
    print(fem_df[["cut_depth_um", "blade_W_um", "deletion_fraction"]].to_string(index=False))

    # Step 1: calibrate
    print("\n[*] Calibrating deletion_fraction → chipping_um ...")
    cal = calibrate_del_frac_to_chipping(fem_df, exp_df)

    # Step 2: build fused dataset
    print("\n[*] Building fused dataset ...")
    fused_df, fem_extra = build_fused_dataset(fem_df, exp_df, cal)

    X_fus = fused_df[FEATURE_COLS].values.astype(float)
    y_fus = fused_df[TARGET_COL].values.astype(float)

    # Experimental-only model (baseline)
    print("\n[*] Training experimental-only GP ...")
    model_exp = ExperimentalGPSurrogate()
    if args.loo:
        print("  LOO (exp-only):")
        model_exp.loo_cv(X_exp, y_exp)
    model_exp.fit(X_exp, y_exp)
    model_exp.save(MODEL_PATH)

    # Fusion model
    print("\n[*] Training fusion GP ...")
    model_fus = ExperimentalGPSurrogate()
    if args.loo:
        print("  LOO (fusion):")
        model_fus.loo_cv(X_fus, y_fus)
    model_fus.fit(X_fus, y_fus)
    model_fus.save(FUSION_MODEL_PATH)

    if args.plot:
        print("\n[*] Plotting ...")
        plot_comparison(model_exp, model_fus, exp_df, fem_extra)

    # Summary predictions at key conditions
    print("\n[*] Prediction comparison (blade_W=23, feed=1.0, spindle=30k):")
    print(f"  {'depth':>6}  {'exp GP':>10}  {'fusion GP':>10}")
    for d in [80, 150, 220, 290, 360, 390]:
        tc = np.array([[d, 23.0, 1.0, 30000.0]])
        me, se = model_exp.predict(tc, return_std=True)
        mf, sf = model_fus.predict(tc, return_std=True)
        print(f"  {d:>6}µm  {me[0]:>6.1f}±{se[0]:.1f}  {mf[0]:>6.1f}±{sf[0]:.1f} µm")


if __name__ == "__main__":
    main()
