"""
Benchmark: Bayesian Optimization vs TMCMC for Gc parameter calibration.

Both methods target the same synthetic RF2 curve (true params fixed).
Metrics:
  - Convergence of best RMSE vs number of FEM evaluations
  - Wall-clock time
  - Final calibrated parameter error vs ground truth

Usage:
    python benchmark_bayesopt_vs_tmcmc.py
    python benchmark_bayesopt_vs_tmcmc.py --n-iter-bo 40 --n-samples-tmcmc 2000
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
from scipy.stats import qmc

sys_path = os.path.join(os.path.dirname(__file__), "..")
import sys
sys.path.insert(0, sys_path)

from optimization.bayesian_opt_gc_fit import (
    _physics_rf2, SyntheticFEMRunner, GcBayesOpt,
    CUT_DEPTHS_UM, BOUNDS_ARRAY, PARAM_NAMES,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TRUE_PARAMS  = SyntheticFEMRunner.TRUE_PARAMS


# ── TMCMC Gc calibration ──────────────────────────────────────────────────────

def _log_likelihood_gc(theta: np.ndarray, rf2_exp: np.ndarray,
                       sigma_obs: float = 5e4) -> float:
    Gc, dp, eps = theta
    if Gc <= 0 or dp <= 0 or eps <= 0:
        return -np.inf
    rf2_sim  = _physics_rf2(CUT_DEPTHS_UM, Gc, dp, eps)
    residual = rf2_exp - rf2_sim
    return float(-0.5 * np.sum((residual / sigma_obs) ** 2))


def _log_prior_gc(theta: np.ndarray) -> float:
    if np.all(theta >= BOUNDS_ARRAY[:, 0]) and np.all(theta <= BOUNDS_ARRAY[:, 1]):
        return 0.0
    return -np.inf


def run_tmcmc(rf2_exp: np.ndarray,
              n_samples:   int   = 1000,
              n_steps_max: int   = 30,
              cov_scale:   float = 0.2,
              seed:        int   = 42) -> dict:
    """
    TMCMC over (Gc, dp_cohesion_GPa, eps_fracture_scale).
    Returns samples, weights, convergence history.
    """
    rng   = np.random.default_rng(seed)
    lo, hi = BOUNDS_ARRAY[:, 0], BOUNDS_ARRAY[:, 1]
    n_p   = 3

    samples   = rng.uniform(lo, hi, size=(n_samples, n_p))
    ll_fn     = lambda th: _log_likelihood_gc(th, rf2_exp)
    log_likes = np.array([ll_fn(s) for s in samples])

    beta       = 0.0
    n_evals    = n_samples          # prior sampling counts
    history    = []                 # (n_evals, best_rmse)

    def _best_rmse(samp, w):
        map_i = np.argmax(w)
        Gc, dp, eps = samp[map_i]
        rf2_sim = _physics_rf2(CUT_DEPTHS_UM, Gc, dp, eps)
        return float(np.sqrt(np.mean((rf2_sim - rf2_exp) ** 2)))

    for stage in range(n_steps_max):
        beta_prev = beta
        lo_b, hi_b = beta, 1.0
        for _ in range(50):
            beta_try   = (lo_b + hi_b) / 2
            db         = beta_try - beta_prev
            log_w      = db * log_likes
            log_w     -= log_w.max()
            w          = np.exp(log_w); w /= w.sum()
            ess        = 1.0 / (w ** 2).sum()
            if ess > n_samples / 2:
                lo_b = beta_try
            else:
                hi_b = beta_try
            if hi_b - lo_b < 1e-6:
                break

        beta      = lo_b
        db        = beta - beta_prev
        log_w     = db * log_likes
        log_w    -= log_w.max()
        w         = np.exp(log_w); w /= w.sum()

        best_r = _best_rmse(samples, w)
        history.append((n_evals, best_r))
        print(f"  TMCMC stage {stage+1:2d}: β={beta:.4f}  "
              f"ESS={1/(w**2).sum():.0f}  best_rmse={best_r:.1f} N")

        if beta >= 1.0:
            break

        idx       = rng.choice(n_samples, size=n_samples, p=w)
        samples   = samples[idx]
        log_likes = log_likes[idx]

        raw_cov = np.cov(samples.T)
        cov     = raw_cov * cov_scale ** 2 + np.eye(n_p) * 1e-6
        L       = np.linalg.cholesky(cov)

        for i in range(n_samples):
            proposal  = samples[i] + L @ rng.standard_normal(n_p)
            lp        = _log_prior_gc(proposal)
            if lp == -np.inf:
                continue
            ll_prop   = ll_fn(proposal)
            n_evals  += 1
            log_alpha = (beta * ll_prop + lp) - (beta * log_likes[i] + _log_prior_gc(samples[i]))
            if np.log(rng.random()) < log_alpha:
                samples[i]   = proposal
                log_likes[i] = ll_prop
            history.append((n_evals, _best_rmse(samples, w / w.sum())))

    log_w  = log_likes - log_likes.max()
    w_fin  = np.exp(log_w); w_fin /= w_fin.sum()
    mean_p = np.average(samples, weights=w_fin, axis=0)
    map_i  = np.argmax(w_fin)

    return {
        "samples":  samples,
        "weights":  w_fin,
        "mean":     mean_p,
        "map":      samples[map_i],
        "history":  history,
        "n_evals":  n_evals,
    }


# ── BayesOpt wrapper that records history ─────────────────────────────────────

class TrackedGcBayesOpt(GcBayesOpt):
    """GcBayesOpt with per-evaluation convergence tracking."""

    def __init__(self, runner, rf2_exp):
        super().__init__(runner, rf2_exp)
        self.eval_history: list[tuple[int, float]] = []  # (cumulative evals, best rmse)

    def _record(self):
        best = min(self.Y_rmse)
        self.eval_history.append((len(self.Y_rmse), best))

    def run(self, n_init=6, n_iter=20, seed=42, history_csv=None):
        # LHS init
        X_init = self._lhs_init(n_init, seed=seed)
        print(f"\n  BayesOpt init ({n_init} LHS evals)")
        for i, x in enumerate(X_init, 1):
            Gc, dp, eps = x
            rmse = self._rmse(Gc, dp, eps)
            self.X_raw.append(x); self.Y_rmse.append(rmse)
            self._record()

        # BO loop
        print(f"  BayesOpt loop ({n_iter} iters)")
        for it in range(1, n_iter + 1):
            gp        = self._fit_gp()
            Y_s       = self._sy.transform(np.array(self.Y_rmse).reshape(-1,1)).ravel()
            x_next    = self._maximize_ei(gp, Y_s.min(), seed=seed+it)
            Gc, dp, eps = x_next
            rmse      = self._rmse(Gc, dp, eps)
            self.X_raw.append(x_next); self.Y_rmse.append(rmse)
            self._record()
            best = min(self.Y_rmse)
            print(f"    iter {it:2d}/{n_iter}  RMSE={rmse:.1f}  best={best:.1f} N")

        return self._best_result()


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_benchmark(bo_history, tmcmc_history,
                   bo_result, tmcmc_map,
                   rf2_exp: np.ndarray,
                   out_dir: str = RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── A: Convergence curve ──────────────────────────────────────────────
    ax = axes[0]
    bo_x,   bo_y   = zip(*bo_history)
    tmcmc_x, tmcmc_y = zip(*tmcmc_history)
    ax.semilogy(bo_x,    bo_y,    "b-o",  ms=4, lw=1.5, label="BayesOpt (GP+EI)")
    ax.semilogy(tmcmc_x, tmcmc_y, "r--s", ms=4, lw=1.5, label="TMCMC")
    ax.set_xlabel("FEM evaluations")
    ax.set_ylabel("Best RMSE [N]  (log scale)")
    ax.set_title("Convergence: BayesOpt vs TMCMC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── B: RF2 curve fit ──────────────────────────────────────────────────
    ax = axes[1]
    depths = CUT_DEPTHS_UM
    rf2_true = _physics_rf2(depths, **TRUE_PARAMS)
    rf2_bo   = _physics_rf2(depths, bo_result["Gc_J_m2"],
                             bo_result["dp_cohesion_GPa"],
                             bo_result["eps_fracture_scale"])
    rf2_tm   = _physics_rf2(depths, *tmcmc_map)

    ax.plot(depths, rf2_exp  / 1e3, "ko-",  lw=2,   ms=6,  label="Experiment (target)")
    ax.plot(depths, rf2_true / 1e3, "g--",  lw=1.5,         label=f"True params")
    ax.plot(depths, rf2_bo   / 1e3, "b-^",  lw=1.5, ms=5,  label="BayesOpt fit")
    ax.plot(depths, rf2_tm   / 1e3, "r-s",  lw=1.5, ms=5,  label="TMCMC fit (MAP)")
    ax.set_xlabel("Cut depth [µm]")
    ax.set_ylabel("Peak RF2 [kN]")
    ax.set_title("RF2 Curve Fit")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── C: Parameter error bar chart ─────────────────────────────────────
    ax    = axes[2]
    names = ["Gc [J/m²]", "dp_cohesion [GPa]", "eps_frac_scale"]
    true  = [TRUE_PARAMS["Gc"], TRUE_PARAMS["dp_cohesion_GPa"],
             TRUE_PARAMS["eps_fracture_scale"]]
    bo_p  = [bo_result["Gc_J_m2"], bo_result["dp_cohesion_GPa"],
              bo_result["eps_fracture_scale"]]
    tm_p  = list(tmcmc_map)

    x     = np.arange(len(names))
    w     = 0.3
    ax.bar(x - w/2,
           [abs(b - t) / t * 100 for b, t in zip(bo_p, true)],
           w, label="BayesOpt", color="steelblue")
    ax.bar(x + w/2,
           [abs(m - t) / t * 100 for m, t in zip(tm_p, true)],
           w, label="TMCMC (MAP)", color="tomato")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Relative error [%]")
    ax.set_title("Parameter Estimation Error")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"BayesOpt ({bo_result['n_evaluations']} evals, "
        f"RMSE={bo_result['rmse_N']:.0f} N)  vs  "
        f"TMCMC (n_evals≈{tmcmc_history[-1][0]}, "
        f"RMSE={tmcmc_history[-1][1]:.0f} N)",
        fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_bo_vs_tmcmc.png")
    plt.savefig(path, dpi=150)
    print(f"[✓] Plot → {path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-init-bo",       type=int, default=8)
    parser.add_argument("--n-iter-bo",       type=int, default=30)
    parser.add_argument("--n-samples-tmcmc", type=int, default=800)
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    runner  = SyntheticFEMRunner(noise_std=0.0, seed=args.seed)
    rf2_exp = runner.get_experimental_rf2()

    print("=" * 60)
    print("  Target RF2 [N]:", np.round(rf2_exp / 1e3, 2), "kN")
    print("  True params:   ", TRUE_PARAMS)
    print("=" * 60)

    # ── BayesOpt ──────────────────────────────────────────────────────────
    print("\n[*] Running BayesOpt...")
    t0       = time.time()
    bo_opt   = TrackedGcBayesOpt(runner=runner, rf2_exp=rf2_exp)
    bo_result = bo_opt.run(n_init=args.n_init_bo, n_iter=args.n_iter_bo,
                           seed=args.seed)
    bo_time  = time.time() - t0
    print(f"\n  BayesOpt done in {bo_time:.1f}s")
    print(f"  Best: {bo_result}")

    # ── TMCMC ─────────────────────────────────────────────────────────────
    print("\n[*] Running TMCMC...")
    t0          = time.time()
    tmcmc_res   = run_tmcmc(rf2_exp, n_samples=args.n_samples_tmcmc,
                             seed=args.seed)
    tmcmc_time  = time.time() - t0
    print(f"\n  TMCMC done in {tmcmc_time:.1f}s  ({tmcmc_res['n_evals']} evals)")
    print(f"  MAP: Gc={tmcmc_res['map'][0]:.2f}  dp={tmcmc_res['map'][1]:.3f}  "
          f"eps={tmcmc_res['map'][2]:.1f}")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        "bayesopt": {**bo_result, "wall_time_s": round(bo_time, 2)},
        "tmcmc": {
            "Gc_J_m2":            round(float(tmcmc_res["map"][0]), 4),
            "dp_cohesion_GPa":    round(float(tmcmc_res["map"][1]), 5),
            "eps_fracture_scale": round(float(tmcmc_res["map"][2]), 3),
            "n_evaluations":      tmcmc_res["n_evals"],
            "wall_time_s":        round(tmcmc_time, 2),
        },
    }
    path = os.path.join(RESULTS_DIR, "benchmark_summary.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] Summary → {path}")

    plot_benchmark(bo_opt.eval_history, tmcmc_res["history"],
                   bo_result, tmcmc_res["map"], rf2_exp)


if __name__ == "__main__":
    main()
