"""
Bayesian optimization of SiC dicing parameters.

Objective : minimize chipping (deletion_fraction)
Constraint: max_principal_stress < 0.95 × sigma_fracture

Algorithm : Expected Improvement (EI) with GP surrogate
            → suggests next FEM experiment OR finds global optimum

Usage:
    # Suggest next experiment to run
    python bayesian_opt.py --csv results/parametric_summary.csv --suggest

    # Full optimization (no new FEM, uses surrogate only)
    python bayesian_opt.py --csv results/parametric_summary.csv --optimize

    # Active learning loop (iterative: suggest → run FEM → update GP)
    python bayesian_opt.py --csv results/parametric_summary.csv --active --n-iter 5
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.surrogate_gp import DicingGPSurrogate, FEATURE_COLS, TARGET_COLS, MODEL_PATH
from data.materials.material_properties import SiC

# ── Parameter bounds ──────────────────────────────────────────────────────────
BOUNDS = {
    "cut_depth_um": (10.0, 70.0),
    "blade_W_um":   (15.0, 50.0),
}
BOUNDS_LIST = [BOUNDS[k] for k in FEATURE_COLS]

# Fracture stress constraint (95 % of material value for safety margin)
SIGMA_FRACTURE = SiC["sigma_y"] * 0.95 / 1e9   # [GPa] after normalisation in GP


# ── Acquisition functions ─────────────────────────────────────────────────────
def expected_improvement(X: np.ndarray,
                          model: DicingGPSurrogate,
                          y_best: float,
                          xi: float = 0.01) -> np.ndarray:
    """EI w.r.t. chipping fraction (target 0 = deletion_fraction)."""
    mu, sigma = model.predict(X, return_std=True)
    mu_chip    = mu[:, 0]
    sigma_chip = sigma[:, 0]
    sigma_chip = np.maximum(sigma_chip, 1e-9)

    improvement = y_best - mu_chip - xi
    Z   = improvement / sigma_chip
    ei  = improvement * norm.cdf(Z) + sigma_chip * norm.pdf(Z)
    ei[sigma_chip < 1e-10] = 0.0
    return ei


def constrained_ei(X: np.ndarray,
                    model: DicingGPSurrogate,
                    y_best: float,
                    xi: float = 0.01) -> np.ndarray:
    """EI × P(constraint satisfied): stress < SIGMA_FRACTURE."""
    mu, sigma = model.predict(X, return_std=True)
    mu_chip    = mu[:, 0]
    sigma_chip = np.maximum(sigma[:, 0], 1e-9)
    mu_stress  = mu[:, 1]
    sigma_stress = np.maximum(sigma[:, 1], 1e-9)

    improvement = y_best - mu_chip - xi
    Z   = improvement / sigma_chip
    ei  = improvement * norm.cdf(Z) + sigma_chip * norm.pdf(Z)

    # Probability of constraint: stress < threshold
    p_feasible = norm.cdf((SIGMA_FRACTURE - mu_stress) / sigma_stress)
    return ei * p_feasible


# ── Optimizer ─────────────────────────────────────────────────────────────────
def maximize_ei(model: DicingGPSurrogate,
                y_best: float,
                constrained: bool = True,
                seed: int = 42) -> dict:
    """Maximize EI over the parameter space using differential evolution."""

    def neg_cei(x):
        X = np.array(x).reshape(1, -1)
        if constrained:
            return -constrained_ei(X, model, y_best)[0]
        return -expected_improvement(X, model, y_best)[0]

    result = differential_evolution(
        neg_cei,
        bounds=BOUNDS_LIST,
        seed=seed,
        maxiter=500,
        tol=1e-8,
        workers=1,
        popsize=20)

    x_opt = result.x
    mu, sigma = model.predict(x_opt.reshape(1, -1), return_std=True)
    return {
        "cut_depth_um":             round(x_opt[0], 2),
        "blade_W_um":               round(x_opt[1], 2),
        "pred_deletion_fraction":   round(float(mu[0, 0]), 6),
        "pred_stress_GPa":          round(float(mu[0, 1]), 4),
        "ei_value":                 round(-result.fun, 8),
        "current_best_chip":        round(y_best, 6),
    }


# ── Pareto front (chipping vs. stress) ───────────────────────────────────────
def compute_pareto_front(model: DicingGPSurrogate,
                          n_grid: int = 2000,
                          out_dir: str = "results") -> pd.DataFrame:
    """Sample dense grid, extract Pareto-optimal points."""
    rng = np.random.default_rng(0)
    X_samp = rng.uniform(
        low =[b[0] for b in BOUNDS_LIST],
        high=[b[1] for b in BOUNDS_LIST],
        size=(n_grid, 2))
    mu, _ = model.predict(X_samp, return_std=True)

    chip   = mu[:, 0]
    stress = mu[:, 1]

    # Pareto: point i dominates j if chip_i <= chip_j AND stress_i <= stress_j
    pareto_mask = np.ones(n_grid, dtype=bool)
    for i in range(n_grid):
        for j in range(n_grid):
            if i == j:
                continue
            if chip[j] <= chip[i] and stress[j] <= stress[i]:
                if chip[j] < chip[i] or stress[j] < stress[i]:
                    pareto_mask[i] = False
                    break

    df_pareto = pd.DataFrame({
        "cut_depth_um": X_samp[pareto_mask, 0],
        "blade_W_um":   X_samp[pareto_mask, 1],
        "deletion_fraction":      chip[pareto_mask],
        "max_principal_stress_GPa": stress[pareto_mask],
    }).sort_values("deletion_fraction")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pareto_front.csv")
    df_pareto.to_csv(path, index=False)
    print(f"[✓] Pareto front ({len(df_pareto)} points) → {path}")
    return df_pareto


def plot_pareto(df_pareto: pd.DataFrame, out_dir: str = "results"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Objective space
    ax = axes[0]
    ax.scatter(df_pareto["deletion_fraction"],
               df_pareto["max_principal_stress_GPa"],
               c="steelblue", s=40, zorder=5, label="Pareto front")
    ax.set_xlabel("Chipping Fraction (lower = better)")
    ax.set_ylabel("Max Principal Stress [GPa]")
    ax.set_title("Pareto Front: Chipping vs. Stress")
    ax.axhline(SIGMA_FRACTURE, color="red", ls="--",
               label=f"Fracture limit ({SIGMA_FRACTURE:.1f} GPa)")
    ax.legend()

    # Parameter space
    ax = axes[1]
    sc = ax.scatter(df_pareto["cut_depth_um"],
                    df_pareto["blade_W_um"],
                    c=df_pareto["deletion_fraction"],
                    cmap="plasma", s=40)
    plt.colorbar(sc, ax=ax, label="Chipping Fraction")
    ax.set_xlabel("Cut Depth [µm]")
    ax.set_ylabel("Blade Width [µm]")
    ax.set_title("Pareto Points in Parameter Space")

    plt.tight_layout()
    path = os.path.join(out_dir, "pareto_front.png")
    plt.savefig(path, dpi=150)
    print(f"[✓] Plot → {path}")
    plt.close()


# ── Active learning loop ──────────────────────────────────────────────────────
def active_learning_loop(csv_path: str, n_iter: int = 5):
    """
    Iterative loop:
      1. Load current CSV
      2. Fit GP
      3. Suggest next point via EI
      4. Print suggestion → user runs FEM → appends result to CSV
      5. Repeat
    """
    for iteration in range(1, n_iter + 1):
        print(f"\n{'='*60}")
        print(f"  Active Learning Iteration {iteration}/{n_iter}")
        print(f"{'='*60}")

        df   = pd.read_csv(csv_path)
        X    = df[FEATURE_COLS].values.astype(float)
        Y    = df[TARGET_COLS].values.astype(float)
        Y[:, 1] /= 1e9   # Pa → GPa

        model = DicingGPSurrogate()
        model.fit(X, Y)
        model.save()

        y_best = Y[:, 0].min()
        suggestion = maximize_ei(model, y_best, constrained=True)

        print(f"\n[→] Suggested next experiment:")
        for k, v in suggestion.items():
            print(f"    {k}: {v}")

        print(f"\n[!] Run ABAQUS with these parameters, then append result to:")
        print(f"    {csv_path}")
        input("[Enter] to continue after updating CSV...")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",      required=True)
    parser.add_argument("--suggest",  action="store_true",
                        help="Suggest one next experiment via EI")
    parser.add_argument("--optimize", action="store_true",
                        help="Find global optimum on surrogate")
    parser.add_argument("--pareto",   action="store_true",
                        help="Compute and plot Pareto front")
    parser.add_argument("--active",   action="store_true",
                        help="Run active learning loop")
    parser.add_argument("--n-iter",   type=int, default=5)
    args = parser.parse_args()

    out_dir = os.path.dirname(args.csv)

    if args.active:
        active_learning_loop(args.csv, n_iter=args.n_iter)
        return

    # Load data and fit GP
    df = pd.read_csv(args.csv)
    X  = df[FEATURE_COLS].values.astype(float)
    Y  = df[TARGET_COLS].values.astype(float)
    Y[:, 1] /= 1e9

    if os.path.exists(MODEL_PATH):
        model = DicingGPSurrogate.load()
    else:
        model = DicingGPSurrogate()
        model.fit(X, Y)
        model.save()

    y_best = Y[:, 0].min()
    print(f"[*] Current best chipping fraction: {y_best:.6f}")

    if args.suggest or args.optimize:
        print("\n[*] Optimizing acquisition function...")
        result = maximize_ei(model, y_best, constrained=True)
        print("\n[→] Optimal / next suggested parameters:")
        for k, v in result.items():
            print(f"    {k}: {v}")

    if args.pareto:
        print("\n[*] Computing Pareto front...")
        df_pareto = compute_pareto_front(model, out_dir=out_dir)
        print(df_pareto.head(10).to_string(index=False))
        try:
            plot_pareto(df_pareto, out_dir=out_dir)
        except ImportError:
            print("[!] matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
