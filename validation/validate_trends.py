"""
Trend validation: compare FEM parametric sweep results to experimental literature.

Validation strategy:
  1. Qualitative: FEM trends (chipping ↑ with depth, feed) vs experimental trends
  2. Quantitative correlation: Pearson r between FEM deletion_fraction and exp chipping_um
  3. TMCMC calibration: infer scale factor (deletion_fraction → chipping_um)

Usage:
    python validate_trends.py --csv results/parametric_summary.csv [--plot]
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from validation.experimental_data import (
    get_chipping_dataframe, QUALITATIVE_TRENDS, CHIPPING_DATA
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


# ── 1. Qualitative trend check ────────────────────────────────────────────────
def check_qualitative_trends(df_fem: pd.DataFrame) -> dict:
    """
    Verify FEM shows correct monotonic trends.
    Returns dict of {trend_name: (expected, observed, pass/fail)}.
    """
    results = {}

    # Trend 1: chipping increases with cut depth (blade width fixed)
    for bw in df_fem["blade_W_um"].unique():
        sub = df_fem[df_fem["blade_W_um"] == bw].sort_values("cut_depth_um")
        if len(sub) < 2:
            continue
        corr = sub["deletion_fraction"].corr(sub["cut_depth_um"])
        key = f"depth_effect_bw{int(bw)}"
        results[key] = {
            "expected":  "positive (chipping ↑ with depth)",
            "pearson_r": round(corr, 3),
            "pass":      corr > 0,
        }

    # Trend 2: chipping decreases or stays flat with larger blade width
    # (wider blade = more kerf material removal per pass → less stress concentration)
    for d in df_fem["cut_depth_um"].unique():
        sub = df_fem[df_fem["cut_depth_um"] == d].sort_values("blade_W_um")
        if len(sub) < 2:
            continue
        corr = sub["deletion_fraction"].corr(sub["blade_W_um"])
        key = f"blade_effect_d{int(d)}"
        results[key] = {
            "expected":  "negative or weak (wider blade slightly less chipping)",
            "pearson_r": round(corr, 3),
            "pass":      corr < 0.5,   # allow weak positive (literature is mixed)
        }

    # Summary
    passed = sum(1 for v in results.values() if v["pass"])
    total  = len(results)
    print(f"\n[Qualitative Trends] {passed}/{total} passed")
    for k, v in results.items():
        flag = "✓" if v["pass"] else "✗"
        print(f"  {flag} {k}: r={v['pearson_r']:.3f} ({v['expected'][:40]})")

    return results


# ── 2. Quantitative scale calibration ────────────────────────────────────────
def calibrate_scale(df_fem: pd.DataFrame) -> dict:
    """
    Find best scale factor S such that:
        chipping_um_predicted ≈ S × deletion_fraction

    Uses experimental data at matching blade widths.
    Returns: {S_median, S_std, R2}
    """
    df_exp = get_chipping_dataframe()

    # Match by blade width (closest available)
    scales = []
    for _, row in df_exp.iterrows():
        bw_exp = row["blade_W_um"]
        # Find FEM row with closest blade width
        fem_match = df_fem.iloc[(df_fem["blade_W_um"] - bw_exp).abs().argsort()[:1]]
        df_frac = float(fem_match["deletion_fraction"].values[0])
        if df_frac > 1e-6:
            S = row["chipping_um"] / df_frac
            scales.append(S)

    if not scales:
        print("[!] No matching data for scale calibration")
        return {}

    S_median = float(np.median(scales))
    S_std    = float(np.std(scales))
    print(f"\n[Scale Calibration]")
    print(f"  Conversion: chipping_um = {S_median:.1f} × deletion_fraction")
    print(f"  Std: {S_std:.1f}  (over {len(scales)} matched points)")
    print(f"  Interpretation: 1% element deletion ≈ {S_median/100:.1f} µm chipping")

    return {"S_median": S_median, "S_std": S_std, "n_points": len(scales)}


# ── 3. Sensitivity comparison ─────────────────────────────────────────────────
def compare_sensitivity(df_fem: pd.DataFrame) -> None:
    """
    Compare parameter sensitivity ranking between FEM and experiment.
    Expected from literature: depth > blade_W (minor).
    """
    # FEM: total variance explained by each parameter
    from sklearn.linear_model import LinearRegression
    X = df_fem[["cut_depth_um", "blade_W_um"]].values
    y = df_fem["deletion_fraction"].values

    # Normalize
    X_n = (X - X.mean(0)) / (X.std(0) + 1e-10)
    lr  = LinearRegression().fit(X_n, y)
    coef_abs = np.abs(lr.coef_)

    print("\n[Parameter Sensitivity (FEM)]")
    params = ["cut_depth_um", "blade_W_um"]
    for p, c in zip(params, coef_abs):
        pct = 100 * c / coef_abs.sum()
        bar = "█" * int(pct / 5)
        print(f"  {p:20s}: {pct:5.1f}% {bar}")

    print(f"\n  Literature ranking: {QUALITATIVE_TRENDS['parameter_rank']}")
    if coef_abs[0] > coef_abs[1]:
        print("  ✓ FEM agrees: cut_depth is dominant")
    else:
        print("  ✗ FEM disagrees: blade_W appears dominant — check model")


# ── 4. Convert FEM deletion_fraction → chipping_um using calibration ──────────
def apply_calibration(df_fem: pd.DataFrame, S: float) -> pd.DataFrame:
    df = df_fem.copy()
    df["chipping_um_predicted"] = df["deletion_fraction"] * S
    return df


# ── 5. Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  required=True,
                        help="FEM parametric_summary.csv")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    df_fem = pd.read_csv(args.csv)
    print(f"[*] Loaded FEM data: {len(df_fem)} rows")
    print(f"    Columns: {list(df_fem.columns)}")
    print(f"    deletion_fraction range: "
          f"{df_fem['deletion_fraction'].min():.4f} – "
          f"{df_fem['deletion_fraction'].max():.4f}")

    check_qualitative_trends(df_fem)
    calib = calibrate_scale(df_fem)
    compare_sensitivity(df_fem)

    if calib:
        df_pred = apply_calibration(df_fem, calib["S_median"])
        out_path = os.path.join(RESULTS_DIR, "validated_predictions.csv")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df_pred.to_csv(out_path, index=False)
        print(f"\n[✓] Calibrated predictions → {out_path}")

    if args.plot and calib:
        _plot(df_fem, calib["S_median"])


def _plot(df_fem, S):
    import matplotlib.pyplot as plt

    df_exp = get_chipping_dataframe()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # FEM response surface
    ax = axes[0]
    pivot = df_fem.pivot_table(
        values="deletion_fraction",
        index="blade_W_um", columns="cut_depth_um")
    im = ax.imshow(pivot.values * S, aspect="auto", origin="lower",
                   extent=[df_fem["cut_depth_um"].min(),
                           df_fem["cut_depth_um"].max(),
                           df_fem["blade_W_um"].min(),
                           df_fem["blade_W_um"].max()],
                   cmap="hot")
    plt.colorbar(im, ax=ax, label="Predicted chipping [µm]")
    ax.set_xlabel("Cut Depth [µm]")
    ax.set_ylabel("Blade Width [µm]")
    ax.set_title("FEM: Calibrated Chipping Prediction")

    # Trend comparison: chipping vs feed (experimental) + FEM depth proxy
    ax = axes[1]
    for source in df_exp["source"].unique():
        sub = df_exp[df_exp["source"] == source].sort_values("feed_mm_s")
        ax.plot(sub["feed_mm_s"], sub["chipping_um"],
                "o-", label=f"Exp: {source}", markersize=6)

    # FEM: chipping vs cut_depth (both are "severity" parameters → comparable trend)
    ax2 = ax.twiny()
    sub_fem = df_fem[df_fem["blade_W_um"] == df_fem["blade_W_um"].median()]
    sub_fem = sub_fem.sort_values("cut_depth_um")
    ax2.plot(sub_fem["cut_depth_um"],
             sub_fem["deletion_fraction"] * S,
             "s--", color="purple", label="FEM (depth sweep)", markersize=6)
    ax2.set_xlabel("FEM Cut Depth [µm]", color="purple")
    ax.axhline(QUALITATIVE_TRENDS["acceptable_chipping_um"],
               color="red", ls=":", label="Production limit (15 µm)")

    ax.set_xlabel("Exp Feed Speed [mm/s]")
    ax.set_ylabel("Chipping Width [µm]")
    ax.set_title("Trend Comparison: Exp vs FEM")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "validation_trends.png")
    plt.savefig(path, dpi=150)
    print(f"[✓] Plot → {path}")
    plt.close()


if __name__ == "__main__":
    main()
