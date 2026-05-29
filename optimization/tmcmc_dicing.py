"""
TMCMC (Transitional Markov Chain Monte Carlo) for dicing parameter inference.

Infers optimal cutting parameters from observed chipping data using
Bayesian inference with the GP surrogate as the likelihood model.

Two use cases:
  1. CALIBRATION: given observed chipping fraction, infer (cut_depth, blade_W)
  2. OPTIMIZATION: find parameters minimising E[chipping] with uncertainty bounds

Reference: Ching & Chen (2007), "Transitional Markov Chain Monte Carlo Method
           for Bayesian Model Updating, Model Class Selection, and Model Averaging"

Usage:
    python tmcmc_dicing.py --mode calibrate --observed-chip 0.03
    python tmcmc_dicing.py --mode optimize   --target-chip   0.01
"""

import argparse
import os
import sys
import numpy as np
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ml.surrogate_gp import DicingGPSurrogate, FEATURE_COLS, MODEL_PATH

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# ── Parameter bounds (normalised to [0,1] internally) ────────────────────────
PARAM_BOUNDS = np.array([
    [10.0, 70.0],   # cut_depth_um
    [15.0, 50.0],   # blade_W_um
])
N_PARAMS = 2


# ── Log-likelihood: how well params explain observed chipping ─────────────────
def log_likelihood(theta: np.ndarray,
                   model: DicingGPSurrogate,
                   observed_chip: float,
                   sigma_obs: float = 0.005) -> float:
    """
    Gaussian likelihood: P(data | theta) where
        data      = observed chipping fraction
        theta     = [cut_depth_um, blade_W_um]
        sigma_obs = measurement noise std
    """
    X = theta.reshape(1, -1)
    mu, std = model.predict(X, return_std=True)
    mu_chip  = float(mu[0, 0])
    # Combine model uncertainty and observation noise
    total_std = np.sqrt(float(std[0, 0]) ** 2 + sigma_obs ** 2)
    return -0.5 * ((observed_chip - mu_chip) / total_std) ** 2 - np.log(total_std)


# ── Log-prior: uniform over parameter bounds ─────────────────────────────────
def log_prior(theta: np.ndarray) -> float:
    if np.all(theta >= PARAM_BOUNDS[:, 0]) and np.all(theta <= PARAM_BOUNDS[:, 1]):
        return 0.0
    return -np.inf


# ── TMCMC core ────────────────────────────────────────────────────────────────

def tmcmc(log_likelihood_fn,
          log_prior_fn,
          n_samples:    int   = 1000,
          n_steps:      int   = 20,
          cov_scale:    float = 0.2,
          random_seed:  int   = 42) -> dict:
    """
    Transitional MCMC (Ching & Chen 2007).

    Returns:
        samples  : (n_samples, n_params) final posterior samples
        weights  : (n_samples,) normalised importance weights
        log_evidence : log marginal likelihood estimate
    """
    rng = np.random.default_rng(random_seed)

    # ── Stage 0: sample from prior ────────────────────────────────────────────
    samples = rng.uniform(
        low  = PARAM_BOUNDS[:, 0],
        high = PARAM_BOUNDS[:, 1],
        size = (n_samples, N_PARAMS))

    log_likes = np.array([log_likelihood_fn(s) for s in samples])
    log_evidence = 0.0
    beta = 0.0   # tempering parameter (0 → 1)

    for stage in range(n_steps):
        # ── Find next beta such that ESS > n_samples / 2 ─────────────────────
        beta_prev = beta
        lo, hi = beta, 1.0
        for _ in range(50):
            beta_try = (lo + hi) / 2
            delta_beta = beta_try - beta_prev
            log_w = delta_beta * log_likes
            log_w -= log_w.max()
            w = np.exp(log_w)
            w /= w.sum()
            ess = 1.0 / (w ** 2).sum()
            if ess > n_samples / 2:
                lo = beta_try
            else:
                hi = beta_try
            if hi - lo < 1e-6:
                break

        beta = lo
        delta_beta = beta - beta_prev
        log_w = delta_beta * log_likes
        log_w -= log_w.max()
        w = np.exp(log_w)
        log_evidence += np.log(w.mean()) + log_w.max()  # log Z contribution
        w /= w.sum()

        print("  Stage %2d: beta=%.4f  ESS=%.0f" % (
            stage + 1, beta, 1.0 / (w ** 2).sum()))

        if beta >= 1.0:
            break

        # ── Resample ──────────────────────────────────────────────────────────
        idx = rng.choice(n_samples, size=n_samples, p=w)
        samples   = samples[idx]
        log_likes = log_likes[idx]

        # ── MCMC move (Gaussian proposal, adaptive cov) ───────────────────────
        cov = np.cov(samples.T) * cov_scale ** 2 + np.eye(N_PARAMS) * 1e-6
        L   = np.linalg.cholesky(cov)

        for i in range(n_samples):
            proposal  = samples[i] + L @ rng.standard_normal(N_PARAMS)
            lp_prop   = log_prior_fn(proposal)
            if lp_prop == -np.inf:
                continue
            ll_prop   = log_likelihood_fn(proposal)
            log_alpha = (beta * ll_prop + lp_prop) - (beta * log_likes[i] + log_prior_fn(samples[i]))
            if np.log(rng.random()) < log_alpha:
                samples[i]   = proposal
                log_likes[i] = ll_prop

    # Final importance weights at beta=1
    log_w = log_likes - log_likes.max()
    w = np.exp(log_w)
    w /= w.sum()

    return {
        "samples":      samples,
        "weights":      w,
        "log_evidence": log_evidence,
        "beta_final":   beta,
    }


# ── Calibration: infer params from observed chipping ─────────────────────────

def calibrate(observed_chip: float,
              model: DicingGPSurrogate,
              n_samples: int = 2000,
              sigma_obs: float = 0.005) -> dict:

    print("[*] TMCMC calibration: observed chipping = %.4f" % observed_chip)
    ll_fn = lambda theta: log_likelihood(theta, model, observed_chip, sigma_obs)

    result = tmcmc(ll_fn, log_prior, n_samples=n_samples)

    samples = result["samples"]
    weights = result["weights"]

    # Weighted mean and std
    mean_params = np.average(samples, weights=weights, axis=0)
    var_params  = np.average((samples - mean_params) ** 2, weights=weights, axis=0)
    std_params  = np.sqrt(var_params)

    # MAP estimate (highest weight)
    map_idx    = np.argmax(weights)
    map_params = samples[map_idx]

    summary = {
        "observed_chip":    observed_chip,
        "mean_cut_depth_um": float(mean_params[0]),
        "mean_blade_W_um":   float(mean_params[1]),
        "std_cut_depth_um":  float(std_params[0]),
        "std_blade_W_um":    float(std_params[1]),
        "map_cut_depth_um":  float(map_params[0]),
        "map_blade_W_um":    float(map_params[1]),
        "log_evidence":      float(result["log_evidence"]),
    }

    print("\n[Results] Posterior mean:")
    print("  cut_depth = %.2f ± %.2f µm" % (mean_params[0], std_params[0]))
    print("  blade_W   = %.2f ± %.2f µm" % (mean_params[1], std_params[1]))
    print("  MAP: depth=%.2f, kerf=%.2f" % (map_params[0], map_params[1]))

    _save_and_plot(samples, weights, summary, tag="calibrate")
    return summary


# ── Optimization: find params minimizing E[chipping] ─────────────────────────

def optimize(target_chip: float,
             model: DicingGPSurrogate,
             n_samples: int = 2000) -> dict:
    """
    Treat optimization as inference: find theta where predicted chipping ~ target.
    Lower target → tighter constraint on posterior → more concentrated near optimum.
    """
    print("[*] TMCMC optimization: target chipping ≤ %.4f" % target_chip)
    # Likelihood: prefer low chipping, penalize target violations
    sigma_obs = target_chip * 0.1 + 1e-5
    ll_fn = lambda theta: log_likelihood(theta, model, target_chip * 0.5, sigma_obs)

    result = tmcmc(ll_fn, log_prior, n_samples=n_samples)
    samples = result["samples"]
    weights = result["weights"]

    # Best (lowest predicted chipping)
    mu, _ = model.predict(samples, return_std=False)
    chip_pred = mu[:, 0]
    best_idx  = np.argmin(chip_pred)
    best = {
        "cut_depth_um": float(samples[best_idx, 0]),
        "blade_W_um":   float(samples[best_idx, 1]),
        "pred_chip":    float(chip_pred[best_idx]),
        "target_chip":  target_chip,
    }

    print("\n[Results] Best parameters:")
    for k, v in best.items():
        print("  %s: %.4f" % (k, v))

    _save_and_plot(samples, weights, best, tag="optimize")
    return best


# ── Save results + posterior plot ─────────────────────────────────────────────

def _save_and_plot(samples, weights, summary, tag="tmcmc"):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save samples
    np.save(os.path.join(RESULTS_DIR, "tmcmc_%s_samples.npy" % tag), samples)
    np.save(os.path.join(RESULTS_DIR, "tmcmc_%s_weights.npy" % tag), weights)
    with open(os.path.join(RESULTS_DIR, "tmcmc_%s_summary.json" % tag), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Weighted scatter
        ax = axes[0]
        sc = ax.scatter(samples[:, 0], samples[:, 1],
                        c=weights, cmap="plasma", s=10, alpha=0.6)
        plt.colorbar(sc, ax=ax, label="Posterior weight")
        ax.set_xlabel("Cut Depth [µm]")
        ax.set_ylabel("Blade Width [µm]")
        ax.set_title("TMCMC Posterior (%s)" % tag)

        # Marginals
        ax = axes[1]
        bins = 30
        ax.hist(samples[:, 0], bins=bins, weights=weights,
                alpha=0.6, label="Cut Depth", density=True)
        ax2 = ax.twinx()
        ax2.hist(samples[:, 1], bins=bins, weights=weights,
                 alpha=0.6, color="orange", label="Blade Width", density=True)
        ax.set_xlabel("Parameter value [µm]")
        ax.set_title("Posterior Marginals")
        ax.legend(loc="upper left")

        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "tmcmc_%s_posterior.png" % tag)
        plt.savefig(path, dpi=150)
        plt.close()
        print("[✓] Plot → %s" % path)
    except ImportError:
        pass


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["calibrate", "optimize"],
                        default="calibrate")
    parser.add_argument("--observed-chip", type=float, default=0.03,
                        help="Observed chipping fraction (calibrate mode)")
    parser.add_argument("--target-chip",   type=float, default=0.01,
                        help="Target chipping fraction (optimize mode)")
    parser.add_argument("--n-samples",     type=int,   default=2000)
    parser.add_argument("--sigma-obs",     type=float, default=0.005)
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print("[!] GP model not found at %s" % MODEL_PATH)
        print("    Run: python ml/surrogate_gp.py --csv results/parametric_summary.csv")
        sys.exit(1)

    model = DicingGPSurrogate.load()

    if args.mode == "calibrate":
        calibrate(args.observed_chip, model,
                  n_samples=args.n_samples,
                  sigma_obs=args.sigma_obs)
    else:
        optimize(args.target_chip, model, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
