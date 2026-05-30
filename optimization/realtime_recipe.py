"""
Real-time Recipe Correction — Digital Twin for SiC Blade Dicing.

Workflow:
  1. Operator measures chipping on the current wafer (observed_chip_um).
  2. TMCMC infers the posterior over (cut_depth, feed_speed) that produced it.
  3. GP predicts chipping over a fine recipe grid.
  4. Optimal recipe is returned: the lowest-chipping point within the safe zone.

This closes the feedback loop:
    Sensor (chipping measurement)
        → TMCMC (inverse problem: what recipe produced this?)
        → GP (forward: what recipe minimises chipping next?)
        → Machine (apply corrected recipe)

Usage:
    python optimization/realtime_recipe.py --chip 8.5
    python optimization/realtime_recipe.py --chip 12.0 --plot
    python optimization/realtime_recipe.py --demo          # simulate 5 wafers
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.train_from_experimental import ExperimentalGPSurrogate, MODEL_PATH, FEATURE_COLS
from optimization.tmcmc_dicing import tmcmc

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# Production recipe constraints
RECIPE_BOUNDS = {
    "cut_depth_um": (80.0,   390.0),
    "blade_W_um":   (20.0,    55.0),
    "feed_mm_s":    (0.5,     3.0),
    "spindle_rpm":  (22000., 38000.),
}
CHIP_LIMIT = 15.0  # µm — production threshold


# ── TMCMC-based inverse inference ─────────────────────────────────────────────

def infer_recipe_from_chip(observed_chip_um: float,
                            model: ExperimentalGPSurrogate,
                            fixed=None,
                            n_samples: int = 600,
                            noise_std: float = 1.0) -> dict:
    """
    Given observed chipping, infer posterior over (depth, feed).
    Fixed: blade_W_um and spindle_rpm held at nominal values.
    """
    fixed = fixed or {"blade_W_um": 23.0, "spindle_rpm": 30000.0}

    def log_likelihood(theta):
        depth, feed = theta
        x = np.array([[depth, fixed["blade_W_um"], feed, fixed["spindle_rpm"]]])
        mu, sigma = model.predict(x, return_std=True)
        total_std = np.sqrt(sigma[0]**2 + noise_std**2)
        return -0.5 * ((observed_chip_um - mu[0]) / total_std) ** 2

    def log_prior(theta):
        depth, feed = theta
        if not (RECIPE_BOUNDS["cut_depth_um"][0] <= depth <= RECIPE_BOUNDS["cut_depth_um"][1]):
            return -np.inf
        if not (RECIPE_BOUNDS["feed_mm_s"][0] <= feed <= RECIPE_BOUNDS["feed_mm_s"][1]):
            return -np.inf
        return 0.0

    # Initial samples: uniform prior
    rng = np.random.default_rng(0)
    d_init = rng.uniform(*RECIPE_BOUNDS["cut_depth_um"], n_samples)
    f_init = rng.uniform(*RECIPE_BOUNDS["feed_mm_s"],    n_samples)
    theta0 = np.column_stack([d_init, f_init])

    bounds = np.array([
        [RECIPE_BOUNDS["cut_depth_um"][0], RECIPE_BOUNDS["cut_depth_um"][1]],
        [RECIPE_BOUNDS["feed_mm_s"][0],    RECIPE_BOUNDS["feed_mm_s"][1]],
    ])
    result = tmcmc(log_likelihood, log_prior,
                   n_samples=n_samples, n_steps=15,
                   param_bounds=bounds, random_seed=0)
    samples = result["samples"] if isinstance(result, dict) else result

    return {
        "depth_mean":  float(np.mean(samples[:, 0])),
        "depth_std":   float(np.std(samples[:, 0])),
        "feed_mean":   float(np.mean(samples[:, 1])),
        "feed_std":    float(np.std(samples[:, 1])),
        "samples":     samples,
    }


# ── GP-based recipe optimisation ──────────────────────────────────────────────

def find_optimal_recipe(model: ExperimentalGPSurrogate,
                         fixed=None,
                         n_grid: int = 60) -> dict:
    """Find (depth, feed) minimising chipping while staying below CHIP_LIMIT."""
    fixed = fixed or {"blade_W_um": 23.0, "spindle_rpm": 30000.0}

    depths = np.linspace(*RECIPE_BOUNDS["cut_depth_um"], n_grid)
    feeds  = np.linspace(*RECIPE_BOUNDS["feed_mm_s"],    n_grid)
    D, F   = np.meshgrid(depths, feeds)
    X_grid = np.column_stack([
        D.ravel(),
        np.full(D.size, fixed["blade_W_um"]),
        F.ravel(),
        np.full(D.size, fixed["spindle_rpm"]),
    ])
    mu, sigma = model.predict(X_grid, return_std=True)

    # Safe zone: predicted chipping + 2σ < threshold
    safe = (mu + 2 * sigma) < CHIP_LIMIT
    if not safe.any():
        safe = mu < CHIP_LIMIT  # relax to mean

    # Among safe candidates, maximise MRR = depth × feed
    mrr = D.ravel() * F.ravel()
    mrr_safe = np.where(safe, mrr, -np.inf)
    best = int(np.argmax(mrr_safe))

    return {
        "opt_depth_um": float(D.ravel()[best]),
        "opt_feed_mm_s": float(F.ravel()[best]),
        "opt_chip_pred": float(mu[best]),
        "opt_chip_std":  float(sigma[best]),
        "opt_mrr":       float(mrr[best]),
        "safe_fraction": float(safe.mean()),
        "mu_grid":       mu.reshape(D.shape),
        "D":             D,
        "F":             F,
    }


# ── Demo: simulate sequential wafer correction ────────────────────────────────

def demo_sequential(n_wafers=5):
    model = ExperimentalGPSurrogate.load(MODEL_PATH)
    rng   = np.random.default_rng(1)

    print(f"\n{'Wafer':>6}  {'Obs chip':>9}  {'Inferred depth':>14}  "
          f"{'Inferred feed':>13}  {'→ New depth':>11}  {'→ New feed':>10}  "
          f"{'Pred chip':>10}")
    print("  " + "-" * 90)

    current_depth = 200.0
    current_feed  = 1.0
    fixed = {"blade_W_um": 23.0, "spindle_rpm": 30000.0}

    for w in range(1, n_wafers + 1):
        x = np.array([[current_depth, fixed["blade_W_um"],
                        current_feed, fixed["spindle_rpm"]]])
        mu, sig = model.predict(x, return_std=True)
        obs_chip = float(mu[0] + rng.normal(0, 1.0))
        obs_chip = max(obs_chip, 0.5)

        post = infer_recipe_from_chip(obs_chip, model, fixed=fixed, n_samples=400)
        opt  = find_optimal_recipe(model, fixed=fixed)

        print(f"  {w:>4}    {obs_chip:>8.1f}µm  "
              f"  {post['depth_mean']:>7.0f}±{post['depth_std']:.0f}µm   "
              f"  {post['feed_mean']:>6.2f}±{post['feed_std']:.2f}mm/s   "
              f"  {opt['opt_depth_um']:>9.0f}µm  "
              f"  {opt['opt_feed_mm_s']:>8.2f}mm/s  "
              f"  {opt['opt_chip_pred']:>6.1f}±{opt['opt_chip_std']:.1f}µm")

        current_depth = opt["opt_depth_um"]
        current_feed  = opt["opt_feed_mm_s"]


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_correction(observed_chip, post, opt, out_dir=RESULTS):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: posterior distribution
    ax1 = axes[0]
    samples = post["samples"]
    h = ax1.hist2d(samples[:, 0], samples[:, 1], bins=30,
                   cmap="Blues", density=True)
    plt.colorbar(h[3], ax=ax1, label="Posterior density")
    ax1.axvline(post["depth_mean"], color="#d62728", lw=1.5, ls="--",
                label=f"MAP depth={post['depth_mean']:.0f}µm")
    ax1.axhline(post["feed_mean"], color="#2166ac", lw=1.5, ls="--",
                label=f"MAP feed={post['feed_mean']:.2f}mm/s")
    ax1.set_xlabel("Cut Depth [µm]", fontsize=11)
    ax1.set_ylabel("Feed Speed [mm/s]", fontsize=11)
    ax1.set_title(f"TMCMC Posterior\n(observed chipping = {observed_chip:.1f} µm)",
                  fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2)

    # Panel 2: GP chipping + safe zone + optimal point
    ax2 = axes[1]
    im = ax2.contourf(opt["D"], opt["F"], opt["mu_grid"], levels=20, cmap="YlOrRd")
    plt.colorbar(im, ax=ax2, label="Predicted Chipping [µm]")
    ax2.contour(opt["D"], opt["F"], opt["mu_grid"],
                levels=[CHIP_LIMIT], colors="white", linewidths=2, linestyles="--")
    ax2.scatter(opt["opt_depth_um"], opt["opt_feed_mm_s"],
                color="lime", s=200, marker="*", zorder=5, edgecolors="k",
                lw=0.8, label=f"Optimal: {opt['opt_chip_pred']:.1f}µm")
    ax2.scatter(post["depth_mean"], post["feed_mean"],
                color="cyan", s=100, marker="D", zorder=5, edgecolors="k",
                lw=0.6, label=f"Inferred current")
    ax2.set_xlabel("Cut Depth [µm]", fontsize=11)
    ax2.set_ylabel("Feed Speed [mm/s]", fontsize=11)
    ax2.set_title(f"GP Response + Corrected Recipe\n"
                  f"(Safe zone: {opt['safe_fraction']*100:.0f}% of space)",
                  fontsize=10)
    ax2.legend(fontsize=8)

    fig.suptitle("Real-time Recipe Correction — Digital Twin\n"
                 "(Sensor → TMCMC → GP → Optimal Recipe)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "realtime_recipe_correction.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Plot → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip",  type=float, default=8.5,
                        help="Observed chipping [µm]")
    parser.add_argument("--plot",  action="store_true")
    parser.add_argument("--demo",  action="store_true",
                        help="Simulate 5 sequential wafers")
    args = parser.parse_args()

    model = ExperimentalGPSurrogate.load(MODEL_PATH)
    fixed = {"blade_W_um": 23.0, "spindle_rpm": 30000.0}

    if args.demo:
        demo_sequential()
        return

    print(f"[*] Observed chipping: {args.chip:.1f} µm")
    print("[*] Running TMCMC inference ...")
    post = infer_recipe_from_chip(args.chip, model, fixed=fixed)
    print(f"  Inferred depth : {post['depth_mean']:.0f} ± {post['depth_std']:.0f} µm")
    print(f"  Inferred feed  : {post['feed_mean']:.2f} ± {post['feed_std']:.2f} mm/s")

    print("[*] Finding optimal corrected recipe ...")
    opt = find_optimal_recipe(model, fixed=fixed)
    print(f"  Optimal depth  : {opt['opt_depth_um']:.0f} µm")
    print(f"  Optimal feed   : {opt['opt_feed_mm_s']:.2f} mm/s")
    print(f"  Predicted chip : {opt['opt_chip_pred']:.1f} ± {opt['opt_chip_std']:.1f} µm")
    print(f"  Safe zone      : {opt['safe_fraction']*100:.0f}% of parameter space")

    if args.plot:
        plot_correction(args.chip, post, opt)


if __name__ == "__main__":
    main()
