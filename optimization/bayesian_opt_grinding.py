"""
Bayesian optimization of Si wafer grinding parameters.

Objectives : minimize warpage_um  (primary)
             minimize max_residual_stress_MPa  (secondary)
Constraint : MRR >= MRR_MIN  (throughput floor)

MRR is computed analytically:
    MRR [mm³/min] = feed_rate_um_min × cut_depth_um × wheel_width_mm × 1e-3

Algorithm : Expected Improvement (EI) with GP surrogate + Pareto front

Usage:
    python bayesian_opt_grinding.py --csv results/parametric_summary_grinding.csv --suggest
    python bayesian_opt_grinding.py --csv results/parametric_summary_grinding.csv --pareto
    python bayesian_opt_grinding.py --csv results/parametric_summary_grinding.csv --active --n-iter 5
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.surrogate_grinding import GrindingGPSurrogate, FEATURE_COLS, TARGET_COLS, MODEL_PATH

# ── Parameter bounds ──────────────────────────────────────────────────────────
BOUNDS = {
    "wheel_speed_rpm":  (1000.0, 10000.0),
    "feed_rate_um_min": (100.0,  1000.0),
    "cut_depth_um":     (5.0,    60.0),
}
BOUNDS_LIST = [BOUNDS[k] for k in FEATURE_COLS]

# MRR floor: minimum acceptable material removal rate [mm³/min]
# Set to 10% below the mean MRR of the training sweep as a default.
# Override via --mrr-min argument.
MRR_MIN_DEFAULT = 0.5   # mm³/min (conservative; adjust per process requirement)
WHEEL_WIDTH_MM  = 5.0   # must match grinding_warpage_2d.py DEFAULT


def compute_mrr(X: np.ndarray) -> np.ndarray:
    """
    Analytical MRR [mm³/min] from parameter array X (N, 3).
    MRR = feed_rate_um_min × cut_depth_um × wheel_width_mm × 1e-3
    """
    feed  = X[:, 1]   # µm/min
    depth = X[:, 2]   # µm
    return feed * depth * WHEEL_WIDTH_MM * 1e-3   # µm² * mm / min = 1e-3 mm³/min


# ── Acquisition functions ─────────────────────────────────────────────────────
def expected_improvement(X: np.ndarray,
                          model: GrindingGPSurrogate,
                          y_best: float,
                          xi: float = 0.01) -> np.ndarray:
    """EI w.r.t. warpage (target = 0, lower is better)."""
    mu, sigma = model.predict(X, return_std=True)
    mu_w    = mu[:, 0]
    sigma_w = np.maximum(sigma[:, 0], 1e-9)

    improvement = y_best - mu_w - xi
    Z  = improvement / sigma_w
    ei = improvement * norm.cdf(Z) + sigma_w * norm.pdf(Z)
    ei[sigma_w < 1e-10] = 0.0
    return ei


def constrained_ei(X: np.ndarray,
                    model: GrindingGPSurrogate,
                    y_best: float,
                    mrr_min: float,
                    xi: float = 0.01) -> np.ndarray:
    """EI × P(MRR >= mrr_min)."""
    ei  = expected_improvement(X, model, y_best, xi=xi)
    mrr = compute_mrr(X)
    # MRR is deterministic — hard constraint rather than probabilistic
    feasible = (mrr >= mrr_min).astype(float)
    return ei * feasible


# ── Optimizer ─────────────────────────────────────────────────────────────────
def maximize_ei(model: GrindingGPSurrogate,
                y_best: float,
                mrr_min: float = MRR_MIN_DEFAULT,
                seed: int = 42) -> dict:
    """Find next suggested experiment via constrained EI."""

    def neg_cei(x):
        X = np.array(x).reshape(1, -1)
        return -constrained_ei(X, model, y_best, mrr_min)[0]

    result = differential_evolution(
        neg_cei,
        bounds=BOUNDS_LIST,
        seed=seed,
        maxiter=500,
        tol=1e-8,
        workers=1,
        popsize=20)

    x_opt  = result.x
    mu, sigma = model.predict(x_opt.reshape(1, -1), return_std=True)
    mrr    = float(compute_mrr(x_opt.reshape(1, -1))[0])
    return {
        "wheel_speed_rpm":          round(x_opt[0], 1),
        "feed_rate_um_min":         round(x_opt[1], 1),
        "cut_depth_um":             round(x_opt[2], 2),
        "pred_warpage_um":          round(float(mu[0, 0]), 4),
        "pred_stress_MPa":          round(float(mu[0, 1]), 2),
        "pred_mrr_mm3_min":         round(mrr, 4),
        "ei_value":                 round(-result.fun, 8),
        "current_best_warpage_um":  round(y_best, 4),
    }


# ── Pareto front (warpage vs. stress) ─────────────────────────────────────────
def compute_pareto_front(model: GrindingGPSurrogate,
                          mrr_min: float = MRR_MIN_DEFAULT,
                          n_grid: int = 3000,
                          out_dir: str = "results") -> pd.DataFrame:
    """Sample parameter space, return Pareto-optimal (warpage, stress) pairs."""
    rng = np.random.default_rng(0)
    X_samp = rng.uniform(
        low =[b[0] for b in BOUNDS_LIST],
        high=[b[1] for b in BOUNDS_LIST],
        size=(n_grid, 3))

    # Filter infeasible (MRR < floor)
    mrr  = compute_mrr(X_samp)
    mask = mrr >= mrr_min
    X_samp = X_samp[mask]
    if len(X_samp) == 0:
        print("[!] No feasible points with MRR >= %.2f mm³/min" % mrr_min)
        return pd.DataFrame()

    mu, _ = model.predict(X_samp)
    warp   = mu[:, 0]
    stress = mu[:, 1]

    # Pareto: point i is dominated if both objectives worse than some j
    n = len(X_samp)
    pareto_mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if warp[j] <= warp[i] and stress[j] <= stress[i]:
                if warp[j] < warp[i] or stress[j] < stress[i]:
                    pareto_mask[i] = False
                    break

    df_pareto = pd.DataFrame({
        "wheel_speed_rpm":        X_samp[pareto_mask, 0],
        "feed_rate_um_min":       X_samp[pareto_mask, 1],
        "cut_depth_um":           X_samp[pareto_mask, 2],
        "warpage_um":             warp[pareto_mask],
        "max_residual_stress_MPa": stress[pareto_mask],
        "mrr_mm3_min":            compute_mrr(X_samp[pareto_mask]),
    }).sort_values("warpage_um")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pareto_front_grinding.csv")
    df_pareto.to_csv(path, index=False)
    print(f"[✓] Pareto front ({len(df_pareto)} points) → {path}")
    return df_pareto


def plot_pareto(df_pareto: pd.DataFrame, out_dir: str = "results"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sc = ax.scatter(df_pareto["warpage_um"],
                    df_pareto["max_residual_stress_MPa"],
                    c=df_pareto["mrr_mm3_min"], cmap="viridis", s=50, zorder=5)
    plt.colorbar(sc, ax=ax, label="MRR [mm³/min]")
    ax.set_xlabel("Warpage [µm]  ← minimize")
    ax.set_ylabel("Max Residual Stress [MPa]  ← minimize")
    ax.set_title("Pareto Front: Warpage vs. Stress")

    ax = axes[1]
    sc2 = ax.scatter(df_pareto["wheel_speed_rpm"],
                     df_pareto["feed_rate_um_min"],
                     c=df_pareto["warpage_um"], cmap="plasma", s=50)
    plt.colorbar(sc2, ax=ax, label="Warpage [µm]")
    ax.set_xlabel("Wheel Speed [rpm]")
    ax.set_ylabel("Feed Rate [µm/min]")
    ax.set_title("Pareto Points in Parameter Space")

    plt.tight_layout()
    path = os.path.join(out_dir, "pareto_front_grinding.png")
    plt.savefig(path, dpi=150)
    print(f"[✓] Plot → {path}")
    plt.close()


# ── Active learning loop ──────────────────────────────────────────────────────
def active_learning_loop(csv_path: str, n_iter: int = 5,
                          mrr_min: float = MRR_MIN_DEFAULT):
    """Suggest → (user runs FEM) → update → repeat."""
    for iteration in range(1, n_iter + 1):
        print(f"\n{'='*60}")
        print(f"  Active Learning Iteration {iteration}/{n_iter}")
        print(f"{'='*60}")

        df = pd.read_csv(csv_path)
        X  = df[FEATURE_COLS].values.astype(float)
        Y  = df[TARGET_COLS].values.astype(float)

        model = GrindingGPSurrogate()
        model.fit(X, Y)
        model.save()

        y_best     = Y[:, 0].min()
        suggestion = maximize_ei(model, y_best, mrr_min=mrr_min)

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
                        help="Suggest one next experiment via constrained EI")
    parser.add_argument("--pareto",   action="store_true",
                        help="Compute and plot Pareto front")
    parser.add_argument("--active",   action="store_true",
                        help="Run active learning loop")
    parser.add_argument("--n-iter",   type=int, default=5)
    parser.add_argument("--mrr-min",  type=float, default=MRR_MIN_DEFAULT,
                        help="Minimum MRR constraint [mm³/min]")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.csv)

    if args.active:
        active_learning_loop(args.csv, n_iter=args.n_iter, mrr_min=args.mrr_min)
        return

    df = pd.read_csv(args.csv)
    X  = df[FEATURE_COLS].values.astype(float)
    Y  = df[TARGET_COLS].values.astype(float)

    if os.path.exists(MODEL_PATH):
        model = GrindingGPSurrogate.load()
    else:
        model = GrindingGPSurrogate()
        model.fit(X, Y)
        model.save()

    y_best = Y[:, 0].min()
    print(f"[*] Current best warpage: {y_best:.4f} µm")

    if args.suggest:
        print("\n[*] Optimizing acquisition function...")
        result = maximize_ei(model, y_best, mrr_min=args.mrr_min)
        print("\n[→] Suggested next experiment:")
        for k, v in result.items():
            print(f"    {k}: {v}")

    if args.pareto:
        print("\n[*] Computing Pareto front...")
        df_pareto = compute_pareto_front(model, mrr_min=args.mrr_min,
                                          out_dir=out_dir)
        if not df_pareto.empty:
            print(df_pareto.head(10).to_string(index=False))
            try:
                plot_pareto(df_pareto, out_dir=out_dir)
            except ImportError:
                print("[!] matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
