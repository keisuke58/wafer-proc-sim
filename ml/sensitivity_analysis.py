"""
Variance-based sensitivity analysis for the experimental GP surrogate.

Computes first-order Sobol indices S_i and total-effect indices S_Ti via
Monte Carlo sampling (Saltelli 2010 estimator, N=4096 base samples).

Also reports analytical gradient sensitivity at the Micro2026 reference point.

Usage:
    python ml/sensitivity_analysis.py [--plot]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.train_from_experimental import (
    ExperimentalGPSurrogate, FEATURE_COLS, MODEL_PATH, REF,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Feature display names and ranges for sampling
FEAT_META = {
    "cut_depth_um": {"label": "Cut Depth",    "unit": "µm",   "lo":  60.0, "hi": 420.0},
    "blade_W_um":   {"label": "Blade Width",  "unit": "µm",   "lo":  23.0, "hi":  48.0},
    "feed_mm_s":    {"label": "Feed Speed",   "unit": "mm/s", "lo":   0.3, "hi":   3.2},
    "spindle_rpm":  {"label": "Spindle Speed","unit": "rpm",  "lo":18000.0, "hi":42000.0},
}


# ── Sobol estimator (Saltelli 2010) ───────────────────────────────────────────

def _saltelli_matrices(n: int, rng: np.random.Generator) -> tuple:
    """Return sample matrices A, B and AB_i for each feature (i=0..3)."""
    lo  = np.array([FEAT_META[f]["lo"] for f in FEATURE_COLS])
    hi  = np.array([FEAT_META[f]["hi"] for f in FEATURE_COLS])
    A   = rng.uniform(lo, hi, size=(n, len(FEATURE_COLS)))
    B   = rng.uniform(lo, hi, size=(n, len(FEATURE_COLS)))
    AB  = []
    for i in range(len(FEATURE_COLS)):
        m      = A.copy()
        m[:, i] = B[:, i]
        AB.append(m)
    return A, B, AB


def sobol_indices(model: ExperimentalGPSurrogate,
                  n: int = 4096,
                  seed: int = 0) -> pd.DataFrame:
    """Saltelli (2010) first-order S_i and total S_Ti via GP mean predictions."""
    rng     = np.random.default_rng(seed)
    A, B, AB_list = _saltelli_matrices(n, rng)

    yA = model.predict(A)
    yB = model.predict(B)
    f0 = (yA.mean() + yB.mean()) / 2.0
    V  = np.var(np.concatenate([yA, yB]))

    rows = []
    for i, feat in enumerate(FEATURE_COLS):
        yAB = model.predict(AB_list[i])
        S_i  = float(np.mean(yB * (yAB - yA)) / V)
        S_Ti = float(np.mean((yA - yAB) ** 2) / (2 * V))
        rows.append({
            "feature":   feat,
            "label":     FEAT_META[feat]["label"],
            "unit":      FEAT_META[feat]["unit"],
            "S_i":       max(S_i, 0.0),   # clip numerical noise below 0
            "S_Ti":      max(S_Ti, 0.0),
        })
    return pd.DataFrame(rows).sort_values("S_Ti", ascending=False)


# ── Gradient sensitivity at reference point ───────────────────────────────────

def gradient_sensitivity(model: ExperimentalGPSurrogate,
                          eps_frac: float = 0.01) -> pd.DataFrame:
    """
    Normalised gradient |∂mu/∂x_i| × range_i at Micro2026 reference.
    Gives 'chipping change [µm] per full range of each parameter'.
    """
    x0   = np.array([[REF[f] for f in FEATURE_COLS]])
    mu0  = model.predict(x0)[0]
    rows = []
    for i, feat in enumerate(FEATURE_COLS):
        lo, hi = FEAT_META[feat]["lo"], FEAT_META[feat]["hi"]
        delta  = (hi - lo) * eps_frac
        xp = x0.copy(); xp[0, i] += delta
        xm = x0.copy(); xm[0, i] -= delta
        grad = (model.predict(xp)[0] - model.predict(xm)[0]) / (2 * delta)
        sensitivity = abs(grad) * (hi - lo)   # µm chipping per full range
        rows.append({
            "feature":      feat,
            "label":        FEAT_META[feat]["label"],
            "unit":         FEAT_META[feat]["unit"],
            "grad":         float(grad),
            "sensitivity_um": float(sensitivity),
        })
    return pd.DataFrame(rows).sort_values("sensitivity_um", ascending=False)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_sensitivity(sobol_df: pd.DataFrame,
                     grad_df: pd.DataFrame,
                     out_dir: str = RESULTS_DIR) -> str:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Sobol indices
    ax = axes[0]
    labels = sobol_df["label"].tolist()
    x      = np.arange(len(labels))
    w      = 0.35
    bars1 = ax.bar(x - w/2, sobol_df["S_i"],  w, label="First-order $S_i$",    color="#2166ac", alpha=0.85)
    bars2 = ax.bar(x + w/2, sobol_df["S_Ti"], w, label="Total-effect $S_{Ti}$", color="#d6604d", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Sobol Index", fontsize=11)
    ax.set_title("Variance-Based Sensitivity (Saltelli 2010)\nN=4096 GP samples", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9)

    # Gradient sensitivity
    ax = axes[1]
    gdf = grad_df.sort_values("sensitivity_um", ascending=True)
    colors = ["#d6604d" if g > 0 else "#2166ac" for g in gdf["grad"]]
    bars = ax.barh(gdf["label"], gdf["sensitivity_um"], color=colors, alpha=0.85)
    ax.set_xlabel("Chipping change over full parameter range [µm]", fontsize=11)
    ax.set_title("Local Gradient Sensitivity\n(at Micro2026 reference: depth=390µm, feed=1mm/s)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, gdf["sensitivity_um"]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f} µm", va="center", fontsize=10)

    plt.suptitle("4H-SiC Blade Dicing — GP Surrogate Sensitivity Analysis", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "sensitivity_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Sensitivity plot → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--n",    type=int, default=4096, help="Sobol sample size")
    args = parser.parse_args()

    model = ExperimentalGPSurrogate.load(MODEL_PATH)

    print("\n[*] Sobol indices (N=%d) …" % args.n)
    sobol_df = sobol_indices(model, n=args.n)
    print(sobol_df[["label", "S_i", "S_Ti"]].to_string(index=False))

    print("\n[*] Gradient sensitivity at Micro2026 reference …")
    grad_df = gradient_sensitivity(model)
    print(grad_df[["label", "grad", "sensitivity_um"]].to_string(index=False))

    print("\n[*] Ranking (by total-effect index):")
    for _, row in sobol_df.iterrows():
        print("  %s: S_Ti=%.3f  (S_i=%.3f)" % (row["label"], row["S_Ti"], row["S_i"]))

    if args.plot:
        plot_sensitivity(sobol_df, grad_df)


if __name__ == "__main__":
    main()
