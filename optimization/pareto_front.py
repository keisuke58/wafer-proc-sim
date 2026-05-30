"""
Pareto front: minimize chipping vs maximize material removal rate (MRR).

MRR ∝ cut_depth × feed_speed × blade_W  [µm² · mm/s]
Chipping = GP prediction [µm]

Fixed: blade_W=23µm, spindle=30,000rpm (Micro2026 reference)
Vary : cut_depth_um ∈ [60, 420], feed_mm_s ∈ [0.3, 3.2]

Usage:
    python optimization/pareto_front.py [--plot]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.train_from_experimental import ExperimentalGPSurrogate, FEATURE_COLS, MODEL_PATH

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Fixed conditions (Micro2026 blade)
BLADE_W_UM   = 23.0
SPINDLE_RPM  = 30000.0
CHIP_LIMIT   = 15.0    # µm production threshold


def compute_grid(model: ExperimentalGPSurrogate,
                 n: int = 100) -> pd.DataFrame:
    """Evaluate GP over depth × feed grid; compute MRR."""
    depths = np.linspace(60.0,  420.0, n)
    feeds  = np.linspace(0.3,   3.2,   n)
    D, F   = np.meshgrid(depths, feeds)

    X = np.column_stack([
        D.ravel(),
        np.full(D.size, BLADE_W_UM),
        F.ravel(),
        np.full(D.size, SPINDLE_RPM),
    ])
    mu, sigma = model.predict(X, return_std=True)

    mrr = D.ravel() * F.ravel() * BLADE_W_UM   # µm² · mm/s (relative MRR)

    return pd.DataFrame({
        "cut_depth_um": D.ravel(),
        "feed_mm_s":    F.ravel(),
        "chipping_um":  mu,
        "chip_std_um":  sigma,
        "mrr":          mrr,
    })


def extract_pareto(df: pd.DataFrame,
                   chip_col: str = "chipping_um",
                   mrr_col:  str = "mrr") -> pd.DataFrame:
    """Return the Pareto-optimal set (min chipping, max MRR)."""
    pts = df[[chip_col, mrr_col]].values
    n   = len(pts)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pts[j, 0] <= pts[i, 0] and pts[j, 1] >= pts[i, 1]:
                if pts[j, 0] < pts[i, 0] or pts[j, 1] > pts[i, 1]:
                    dominated[i] = True
                    break
    return df[~dominated].sort_values(chip_col)


def plot_pareto(df: pd.DataFrame,
                pareto_df: pd.DataFrame,
                out_dir: str = RESULTS_DIR) -> str:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: depth×feed heatmap with Pareto front overlay ──────────────────
    ax = axes[0]
    depths = np.sort(df["cut_depth_um"].unique())
    feeds  = np.sort(df["feed_mm_s"].unique())
    n      = int(np.sqrt(len(df)))
    D      = df["cut_depth_um"].values.reshape(n, n)
    F      = df["feed_mm_s"].values.reshape(n, n)
    C      = df["chipping_um"].values.reshape(n, n)

    im = ax.contourf(D, F, C, levels=20, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="GP Mean Chipping [µm]")
    ax.contour(D, F, C, levels=[CHIP_LIMIT], colors="white",
               linewidths=2, linestyles="--")

    safe = df[df["chipping_um"] <= CHIP_LIMIT]
    ax.scatter(safe["cut_depth_um"], safe["feed_mm_s"],
               c=safe["mrr"], cmap="cool", s=4, alpha=0.4, label="Safe zone")
    ax.scatter(pareto_df["cut_depth_um"], pareto_df["feed_mm_s"],
               c="lime", s=40, zorder=5, edgecolors="k", lw=0.5,
               label="Pareto front")

    ax.text(65, 3.05, f"← {CHIP_LIMIT} µm threshold", color="white", fontsize=9, va="top")
    ax.set_xlabel("Cut Depth [µm]", fontsize=11)
    ax.set_ylabel("Feed Speed [mm/s]", fontsize=11)
    ax.set_title("Chipping Response + Pareto Front\n(blade_W=23µm, spindle=30krpm)", fontsize=11)
    ax.legend(fontsize=9)

    # ── Right: chipping vs MRR Pareto curve ──────────────────────────────────
    ax = axes[1]
    safe_sorted = safe.sort_values("chipping_um")
    ax.scatter(safe["mrr"] / 1e3, safe["chipping_um"],
               c=safe["cut_depth_um"], cmap="viridis", s=8, alpha=0.3,
               label="Safe operating points")
    pf_sorted = pareto_df.sort_values("chipping_um")
    ax.plot(pf_sorted["mrr"] / 1e3, pf_sorted["chipping_um"],
            "r-o", ms=5, lw=2, label="Pareto front", zorder=5)
    ax.axhline(CHIP_LIMIT, color="gray", ls="--", lw=1.2,
               label=f"Threshold {CHIP_LIMIT} µm")

    # Annotate a few Pareto points
    for _, row in pf_sorted.iloc[::max(1, len(pf_sorted)//4)].iterrows():
        ax.annotate(
            f"d={row.cut_depth_um:.0f}µm\n f={row.feed_mm_s:.1f}mm/s",
            xy=(row["mrr"]/1e3, row["chipping_um"]),
            xytext=(8, 4), textcoords="offset points", fontsize=8,
        )

    ax.set_xlabel("Material Removal Rate [×10³ µm²·mm/s]", fontsize=11)
    ax.set_ylabel("GP Mean Chipping [µm]", fontsize=11)
    ax.set_title("Chipping vs MRR Tradeoff\n(Pareto: min chipping, max MRR)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.suptitle("4H-SiC Blade Dicing — Process Optimisation (blade_W=23µm)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "pareto_front.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Pareto plot → {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--n",    type=int, default=100, help="Grid size per axis")
    args = parser.parse_args()

    model = ExperimentalGPSurrogate.load(MODEL_PATH)

    print(f"[*] Evaluating GP on {args.n}×{args.n} grid …")
    df = compute_grid(model, n=args.n)

    safe = df[df["chipping_um"] <= CHIP_LIMIT]
    print(f"  Safe points (chip<{CHIP_LIMIT}µm): {len(safe)}/{len(df)} "
          f"({100*len(safe)/len(df):.1f}%)")

    print("\n[*] Extracting Pareto front (min chipping, max MRR) …")
    # Use only safe region for Pareto
    if len(safe) > 0:
        pareto_df = extract_pareto(safe)
        print(f"  Pareto points: {len(pareto_df)}")
        print(pareto_df[["cut_depth_um", "feed_mm_s", "chipping_um", "mrr"]].to_string(index=False))
    else:
        print("  No safe operating points found.")
        pareto_df = pd.DataFrame()

    if args.plot and len(pareto_df) > 0:
        plot_pareto(df, pareto_df)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "grid_chipping_mrr.csv"), index=False)
    if len(pareto_df) > 0:
        pareto_df.to_csv(os.path.join(RESULTS_DIR, "pareto_front.csv"), index=False)
    print("[✓] CSVs saved to results/")


if __name__ == "__main__":
    main()
