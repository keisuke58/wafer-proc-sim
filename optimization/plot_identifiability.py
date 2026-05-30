"""
Identifiability analysis for Gc / dp_cohesion / eps_fracture_scale calibration.

Shows the degenerate parameter manifold where different (Gc, dp, eps) combinations
produce indistinguishable RF2 curves — a key limitation of single-objective
scalar fitting that motivates multi-condition experiments.

Outputs:
  results/identifiability_slices.png   — 2D RMSE slices through true params
  results/identifiability_manifold.png — iso-RMSE surface in (Gc, dp) space
  results/identifiability_report.json  — manifold width statistics

Usage:
    python plot_identifiability.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from optimization.bayesian_opt_gc_fit import (
    _physics_rf2, SyntheticFEMRunner, CUT_DEPTHS_UM, BOUNDS_ARRAY,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TRUE        = SyntheticFEMRunner.TRUE_PARAMS
RF2_EXP     = _physics_rf2(CUT_DEPTHS_UM, **TRUE)   # noise-free target


def rmse(Gc, dp, eps) -> float:
    rf2 = _physics_rf2(CUT_DEPTHS_UM, Gc, dp, eps)
    return float(np.sqrt(np.mean((rf2 - RF2_EXP) ** 2)))


# ── 2D RMSE slices ────────────────────────────────────────────────────────────

def plot_slices(n_grid: int = 80):
    """Three 2D slices through the true param point."""
    Gc_range  = np.linspace(*BOUNDS_ARRAY[0], n_grid)
    dp_range  = np.linspace(*BOUNDS_ARRAY[1], n_grid)
    eps_range = np.linspace(*BOUNDS_ARRAY[2], n_grid)

    # Slice 1: Gc vs dp (eps fixed at true)
    Z1 = np.array([[rmse(g, d, TRUE["eps_fracture_scale"])
                    for d in dp_range] for g in Gc_range])

    # Slice 2: Gc vs eps (dp fixed at true)
    Z2 = np.array([[rmse(g, TRUE["dp_cohesion_GPa"], e)
                    for e in eps_range] for g in Gc_range])

    # Slice 3: dp vs eps (Gc fixed at true)
    Z3 = np.array([[rmse(TRUE["Gc"], d, e)
                    for e in eps_range] for d in dp_range])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cmap = "plasma_r"

    configs = [
        (Gc_range,  dp_range,  Z1, "Gc [J/m²]",        "dp_cohesion [GPa]",
         TRUE["Gc"], TRUE["dp_cohesion_GPa"]),
        (Gc_range,  eps_range, Z2, "Gc [J/m²]",        "eps_fracture_scale",
         TRUE["Gc"], TRUE["eps_fracture_scale"]),
        (dp_range,  eps_range, Z3, "dp_cohesion [GPa]", "eps_fracture_scale",
         TRUE["dp_cohesion_GPa"], TRUE["eps_fracture_scale"]),
    ]

    for ax, (xv, yv, Z, xl, yl, tx, ty) in zip(axes, configs):
        vmax = np.percentile(Z, 90)
        im   = ax.contourf(xv, yv, Z.T, levels=30, cmap=cmap, vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="RMSE [N]")
        ax.contour(xv, yv, Z.T, levels=[RF2_EXP.mean() * 0.05],
                   colors="white", linewidths=1.5, linestyles="--")
        ax.scatter(tx, ty, marker="*", s=300, c="lime", zorder=10,
                   label="True params")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"RMSE slice  ({yl.split()[0]} fixed)")
        ax.legend(fontsize=8)

    plt.suptitle("Identifiability: RMSE landscape slices\n"
                 "(white dashed = 5% relative error contour)", fontsize=11)
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "identifiability_slices.png")
    plt.savefig(path, dpi=150)
    print(f"[✓] {path}")
    plt.close()
    return Z1, Z2, Z3


# ── Degenerate manifold: product Gc × dp^0.35 × eps_f = const ────────────────

def plot_manifold(n_grid: int = 100):
    """
    In the physics model RF2 ∝ sqrt(Gc) × dp^0.35 × log_eps.
    Plot iso-RMSE curve in (Gc, dp) plane for multiple eps values
    to visualize the degenerate manifold.
    """
    Gc_range = np.linspace(3, 80, n_grid)
    dp_range = np.linspace(0.3, 3.0, n_grid)
    GC, DP   = np.meshgrid(Gc_range, dp_range)

    eps_values = [5, 15, 25, 50, 80]
    colors     = plt.cm.cool(np.linspace(0, 1, len(eps_values)))

    fig, ax = plt.subplots(figsize=(8, 6))
    threshold = RF2_EXP.mean() * 0.03   # 3% relative tolerance

    for eps, col in zip(eps_values, colors):
        Z = np.vectorize(lambda g, d: rmse(g, d, eps))(GC, DP)
        ax.contour(Gc_range, dp_range, Z, levels=[threshold],
                   colors=[col], linewidths=2)
        ax.plot([], [], color=col, lw=2, label=f"eps_scale = {eps}")

    ax.scatter(TRUE["Gc"], TRUE["dp_cohesion_GPa"],
               marker="*", s=400, c="red", zorder=10, label="True params")
    ax.set_xlabel("Gc [J/m²]", fontsize=12)
    ax.set_ylabel("dp_cohesion [GPa]", fontsize=12)
    ax.set_title("Degenerate parameter manifold\n"
                 f"(iso-RMSE = {threshold/1e3:.1f} kN  per eps_fracture_scale)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(RESULTS_DIR, "identifiability_manifold.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[✓] {path}")
    plt.close()


# ── Manifold width statistics ─────────────────────────────────────────────────

def compute_manifold_stats(n_mc: int = 5000, threshold_frac: float = 0.05) -> dict:
    """
    Monte Carlo: fraction of parameter space within threshold_frac relative RMSE.
    Quantifies how degenerate the problem is.
    """
    rng      = np.random.default_rng(0)
    lo, hi   = BOUNDS_ARRAY[:, 0], BOUNDS_ARRAY[:, 1]
    samples  = rng.uniform(lo, hi, size=(n_mc, 3))
    threshold = RF2_EXP.mean() * threshold_frac

    rmse_vals = np.array([rmse(*s) for s in samples])
    frac_below = (rmse_vals < threshold).mean()

    stats = {
        "threshold_N":         round(threshold, 1),
        "threshold_frac":      threshold_frac,
        "n_mc_samples":        n_mc,
        "fraction_below_threshold": round(float(frac_below), 4),
        "rmse_percentiles": {
            "p10": round(float(np.percentile(rmse_vals, 10)), 1),
            "p50": round(float(np.percentile(rmse_vals, 50)), 1),
            "p90": round(float(np.percentile(rmse_vals, 90)), 1),
        },
        "note": (
            "fraction_below_threshold > 0.01 indicates strong non-identifiability: "
            "many parameter combinations fit the data equally well."
        ),
    }
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[*] Computing 2D RMSE slices...")
    Z1, Z2, Z3 = plot_slices(n_grid=80)

    print("[*] Plotting degenerate manifold...")
    plot_manifold(n_grid=100)

    print("[*] Monte Carlo manifold statistics...")
    stats = compute_manifold_stats(n_mc=4000)
    print(f"    Fraction within 5% RMSE threshold: {stats['fraction_below_threshold']*100:.2f}%")
    print(f"    → {'High' if stats['fraction_below_threshold'] > 0.01 else 'Low'} degeneracy")

    path = os.path.join(RESULTS_DIR, "identifiability_report.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[✓] {path}")


if __name__ == "__main__":
    main()
