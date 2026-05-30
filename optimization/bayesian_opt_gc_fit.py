"""
Bayesian optimization for material parameter calibration (Gc fitting).

Calibrates fracture energy Gc, Drucker-Prager cohesion, and fracture strain
scale by fitting simulated peak RF2 (thrust force) vs cut-depth to
experimental data.

Algorithm:
    GP surrogate (sklearn) + Expected Improvement over 3-parameter space.
    Each evaluation = one ABAQUS parametric sweep (all cut depths).

Usage:
    # Synthetic test (no ABAQUS needed):
    python bayesian_opt_gc_fit.py --synthetic --n-init 6 --n-iter 20

    # Real mode with experimental RF2 CSV:
    python bayesian_opt_gc_fit.py --exp-csv data/experimental/rf2_exp.csv --n-init 8 --n-iter 30

    # Resume from saved history:
    python bayesian_opt_gc_fit.py --exp-csv ... --resume results/gc_fit_history.csv
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

# ── Search space ──────────────────────────────────────────────────────────────
BOUNDS = {
    "Gc":                 (3.0,   80.0),   # J/m²  (SiC nominal ~17.5)
    "dp_cohesion_GPa":    (0.3,    3.0),   # GPa   (SiC nominal  1.0)
    "eps_fracture_scale": (1.0,  100.0),   # –     (default 25.0)
}
PARAM_NAMES   = list(BOUNDS.keys())
BOUNDS_ARRAY  = np.array([BOUNDS[k] for k in PARAM_NAMES])   # (3, 2)

# Cut depths used for the RF2 curve fit (must match Micro2026 sweep)
CUT_DEPTHS_UM = np.array([80.0, 150.0, 220.0, 290.0, 360.0])

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Synthetic FEM runner ──────────────────────────────────────────────────────

def _physics_rf2(depths_um: np.ndarray,
                 Gc: float,
                 dp_cohesion_GPa: float,
                 eps_fracture_scale: float) -> np.ndarray:
    """
    Physics-inspired surrogate for RF2 (thrust force peak [N]).

    Scaling: RF2 ~ K_Ic * sqrt(d) * blade_bw * f(dp) * g(eps_scale)
    where K_Ic = sqrt(Gc * E_SiC).
    """
    E_SiC   = 410e9
    bw_m    = 23e-6
    d_m     = depths_um * 1e-6
    K_Ic    = np.sqrt(Gc * E_SiC)
    dp_f    = dp_cohesion_GPa ** 0.35
    eps_f   = np.log1p(eps_fracture_scale / 8.0) / np.log1p(25.0 / 8.0)
    return K_Ic * np.sqrt(d_m) * bw_m * dp_f * eps_f * 1.5e6   # [N]


class SyntheticFEMRunner:
    """Generates RF2 from the physics model + noise (no ABAQUS needed)."""

    TRUE_PARAMS = {"Gc": 17.5, "dp_cohesion_GPa": 1.0, "eps_fracture_scale": 25.0}

    def __init__(self, noise_std: float = 0.3, seed: int = 0):
        self.noise_std = noise_std
        self.rng       = np.random.default_rng(seed)
        self.rf2_ref   = _physics_rf2(CUT_DEPTHS_UM, **self.TRUE_PARAMS)

    def evaluate(self, Gc: float, dp_cohesion_GPa: float,
                 eps_fracture_scale: float) -> np.ndarray:
        rf2 = _physics_rf2(CUT_DEPTHS_UM, Gc, dp_cohesion_GPa, eps_fracture_scale)
        return rf2 + self.rng.normal(0.0, self.noise_std, size=rf2.shape)

    def get_experimental_rf2(self) -> np.ndarray:
        return self.rf2_ref.copy()


# ── ABAQUS FEM runner ─────────────────────────────────────────────────────────

class AbaqusFEMRunner:
    """Calls ABAQUS for each evaluation; one parametric sweep per call."""

    def __init__(self,
                 fem_dir:    str,
                 work_dir:   str,
                 abaqus_cmd: str = "abaqus"):
        self.fem_dir    = os.path.abspath(fem_dir)
        self.work_dir   = os.path.abspath(work_dir)
        self.abaqus_cmd = abaqus_cmd
        os.makedirs(work_dir, exist_ok=True)

    def _write_config(self, run_dir: str, depth_um: float,
                      Gc: float, dp_cohesion_GPa: float,
                      eps_fracture_scale: float,
                      job_name: str) -> None:
        cfg = {
            "material":            "SiC",
            "cut_depth_um":        depth_um,
            "blade_W_um":          23.0,
            "feed_speed_m_s":      0.5,
            "mesh_global_um":      8.0,
            "mesh_fine_um":        2.0,
            "target_dt_s":         1e-8,
            "eps_fracture_scale":  eps_fracture_scale,
            "gc_override":         Gc,
            "dp_cohesion_override": dp_cohesion_GPa * 1e9,
            "job_name":            job_name,
            "num_cpus":            4,
        }
        with open(os.path.join(run_dir, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    def _run_abaqus(self, run_dir: str) -> bool:
        script = os.path.join(self.fem_dir, "dicing_blade_2d.py")
        cmd = [self.abaqus_cmd, "cae", f"noGUI={script}"]
        result = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True,
                                timeout=3600)
        if result.returncode != 0:
            print(f"[!] ABAQUS error:\n{result.stderr[-500:]}")
            return False
        return True

    def _extract_rf2(self, run_dir: str, job_name: str) -> float | None:
        """Extract peak |RF2| from forces.csv written by extract_results.py."""
        forces_csv = os.path.join(self.work_dir, "results", job_name, "forces.csv")
        if not os.path.exists(forces_csv):
            extractor = os.path.join(self.fem_dir, "extract_results.py")
            cmd = [self.abaqus_cmd, "python", extractor, "--", "--job", job_name]
            subprocess.run(cmd, cwd=run_dir, capture_output=True, timeout=600)
        if not os.path.exists(forces_csv):
            return None
        df = pd.read_csv(forces_csv)
        return float(df["RF2_N"].abs().max()) if "RF2_N" in df.columns else None

    def evaluate(self, Gc: float, dp_cohesion_GPa: float,
                 eps_fracture_scale: float) -> np.ndarray:
        rf2_peaks = np.full(len(CUT_DEPTHS_UM), np.nan)
        ts = int(time.time())
        for i, depth in enumerate(CUT_DEPTHS_UM):
            job = f"gcfit_{ts}_d{int(depth):03d}"
            run_dir = os.path.join(self.work_dir, job)
            os.makedirs(run_dir, exist_ok=True)
            self._write_config(run_dir, depth, Gc, dp_cohesion_GPa,
                               eps_fracture_scale, job)
            ok = self._run_abaqus(run_dir)
            if ok:
                val = self._extract_rf2(run_dir, job)
                if val is not None:
                    rf2_peaks[i] = val
        return rf2_peaks


# ── GP kernel ─────────────────────────────────────────────────────────────────

def _build_kernel():
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(3), length_scale_bounds=[(0.01, 100.0)] * 3,
                 nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
    )


# ── BayesOpt core ─────────────────────────────────────────────────────────────

class GcBayesOpt:
    """
    Bayesian optimization loop for Gc/cohesion/eps_scale calibration.

    GP is fitted to (normalized params) → RMSE of RF2 curve fit.
    Next candidate is chosen by maximizing Expected Improvement.
    """

    def __init__(self, runner, rf2_exp: np.ndarray):
        self.runner  = runner
        self.rf2_exp = np.asarray(rf2_exp, dtype=float)
        self.X_raw:  list[np.ndarray] = []   # (3,) arrays in original units
        self.Y_rmse: list[float]      = []
        self._sx = StandardScaler()
        self._sy = StandardScaler()

    # ── Objective ──────────────────────────────────────────────────────────
    def _rmse(self, Gc, dp_cohesion_GPa, eps_fracture_scale) -> float:
        rf2_sim = self.runner.evaluate(Gc, dp_cohesion_GPa, eps_fracture_scale)
        mask    = np.isfinite(rf2_sim) & np.isfinite(self.rf2_exp)
        if mask.sum() == 0:
            return 1e9
        return float(np.sqrt(np.mean((rf2_sim[mask] - self.rf2_exp[mask]) ** 2)))

    # ── GP helpers ─────────────────────────────────────────────────────────
    def _fit_gp(self) -> GaussianProcessRegressor:
        X = np.array(self.X_raw)
        Y = np.array(self.Y_rmse).reshape(-1, 1)
        self._sx.fit(X)
        self._sy.fit(Y)
        Xs = self._sx.transform(X)
        Ys = self._sy.transform(Y).ravel()
        gp = GaussianProcessRegressor(kernel=_build_kernel(),
                                      n_restarts_optimizer=5,
                                      normalize_y=False)
        gp.fit(Xs, Ys)
        return gp

    def _ei(self, x_raw: np.ndarray, gp: GaussianProcessRegressor,
            y_best_scaled: float, xi: float = 0.01) -> np.ndarray:
        Xs   = self._sx.transform(x_raw.reshape(-1, len(PARAM_NAMES)))
        mu, sigma = gp.predict(Xs, return_std=True)
        sigma     = np.maximum(sigma, 1e-9)
        imp   = y_best_scaled - mu - xi
        Z     = imp / sigma
        return (imp * norm.cdf(Z) + sigma * norm.pdf(Z))

    def _maximize_ei(self, gp: GaussianProcessRegressor,
                     y_best_scaled: float,
                     seed: int = 0) -> np.ndarray:
        def neg_ei(x):
            return -self._ei(np.array(x), gp, y_best_scaled)[0]

        res = differential_evolution(neg_ei,
                                     bounds=BOUNDS_ARRAY.tolist(),
                                     seed=seed, maxiter=500,
                                     tol=1e-8, popsize=20, workers=1)
        return res.x

    # ── Initialization (Latin Hypercube) ───────────────────────────────────
    def _lhs_init(self, n: int, seed: int = 42) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=3, seed=seed)
        unit    = sampler.random(n)
        lo, hi  = BOUNDS_ARRAY[:, 0], BOUNDS_ARRAY[:, 1]
        return lo + unit * (hi - lo)

    # ── Main loop ──────────────────────────────────────────────────────────
    def run(self, n_init: int = 6, n_iter: int = 20,
            seed: int = 42, history_csv: str | None = None) -> dict:
        """
        Run BayesOpt calibration.

        Args:
            n_init       : LHS initialization evaluations
            n_iter       : BayesOpt iterations after init
            seed         : random seed
            history_csv  : path to save/load history CSV

        Returns:
            dict with best params and RMSE
        """
        # ── Resume from existing history ──────────────────────────────────
        if history_csv and os.path.exists(history_csv):
            df_hist = pd.read_csv(history_csv)
            for _, row in df_hist.iterrows():
                self.X_raw.append(
                    np.array([row["Gc"], row["dp_cohesion_GPa"],
                               row["eps_fracture_scale"]]))
                self.Y_rmse.append(float(row["rmse"]))
            print(f"[✓] Resumed {len(self.Y_rmse)} evaluations from {history_csv}")
        else:
            # ── LHS initialization ────────────────────────────────────────
            X_init = self._lhs_init(n_init, seed=seed)
            print(f"\n{'='*60}")
            print(f"  Phase 1: LHS initialization ({n_init} evaluations)")
            print(f"{'='*60}")
            for i, x in enumerate(X_init, 1):
                Gc, dp, eps = x
                print(f"\n  [{i}/{n_init}] Gc={Gc:.2f} J/m²  dp={dp:.3f} GPa  "
                      f"eps_scale={eps:.1f}")
                rmse = self._rmse(Gc, dp, eps)
                self.X_raw.append(x)
                self.Y_rmse.append(rmse)
                print(f"         → RMSE = {rmse:.4f} N")
                self._save_history(history_csv)

        # ── BayesOpt loop ─────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Phase 2: Bayesian optimization ({n_iter} iterations)")
        print(f"{'='*60}")
        for iteration in range(1, n_iter + 1):
            gp          = self._fit_gp()
            Y_scaled    = self._sy.transform(
                np.array(self.Y_rmse).reshape(-1, 1)).ravel()
            y_best_s    = Y_scaled.min()
            x_next      = self._maximize_ei(gp, y_best_s, seed=seed + iteration)
            Gc, dp, eps = x_next
            print(f"\n  Iter {iteration:2d}/{n_iter}  "
                  f"Gc={Gc:.2f} J/m²  dp={dp:.3f} GPa  eps_scale={eps:.1f}")
            rmse = self._rmse(Gc, dp, eps)
            self.X_raw.append(x_next)
            self.Y_rmse.append(rmse)
            best_i = int(np.argmin(self.Y_rmse))
            print(f"    → RMSE = {rmse:.4f} N  |  best so far = "
                  f"{self.Y_rmse[best_i]:.4f} N  "
                  f"(Gc={self.X_raw[best_i][0]:.2f}, "
                  f"dp={self.X_raw[best_i][1]:.3f}, "
                  f"eps={self.X_raw[best_i][2]:.1f})")
            self._save_history(history_csv)

        return self._best_result()

    # ── Helpers ────────────────────────────────────────────────────────────
    def _save_history(self, path: str | None) -> None:
        if path is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        rows = [{"Gc": x[0], "dp_cohesion_GPa": x[1],
                 "eps_fracture_scale": x[2], "rmse": y}
                for x, y in zip(self.X_raw, self.Y_rmse)]
        pd.DataFrame(rows).to_csv(path, index=False)

    def _best_result(self) -> dict:
        i = int(np.argmin(self.Y_rmse))
        x = self.X_raw[i]
        return {
            "Gc_J_m2":             round(float(x[0]), 4),
            "dp_cohesion_GPa":     round(float(x[1]), 5),
            "eps_fracture_scale":  round(float(x[2]), 3),
            "rmse_N":              round(float(self.Y_rmse[i]), 6),
            "n_evaluations":       len(self.Y_rmse),
        }

    def plot(self, out_path: str | None = None) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[!] matplotlib not available, skipping plot")
            return

        X  = np.array(self.X_raw)
        Y  = np.array(self.Y_rmse)
        best_i = int(np.argmin(Y))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        labels  = ["Gc [J/m²]", "dp_cohesion [GPa]", "eps_fracture_scale"]
        true_v  = [17.5, 1.0, 25.0]   # SiC nominal values

        for k, ax in enumerate(axes):
            sc = ax.scatter(X[:, k], Y, c=np.arange(len(Y)),
                            cmap="viridis", s=40, zorder=5)
            ax.scatter(X[best_i, k], Y[best_i],
                       marker="*", s=300, c="red", zorder=10, label="best")
            ax.axvline(true_v[k], color="gray", ls="--", lw=1, label="nominal")
            ax.set_xlabel(labels[k])
            ax.set_ylabel("RMSE [N]")
            ax.set_title(f"RMSE vs {labels[k]}")
            ax.legend(fontsize=8)

        plt.colorbar(sc, ax=axes[-1], label="Iteration")
        plt.suptitle("BayesOpt Gc Calibration", fontsize=12)
        plt.tight_layout()

        path = out_path or os.path.join(RESULTS_DIR, "gc_fit_bayesopt.png")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        plt.savefig(path, dpi=150)
        print(f"[✓] Plot → {path}")
        plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BayesOpt Gc calibration for dicing blade FEM")
    parser.add_argument("--synthetic",  action="store_true",
                        help="Use physics surrogate instead of ABAQUS")
    parser.add_argument("--exp-csv",    default=None,
                        help="CSV with columns: cut_depth_um, RF2_peak_N")
    parser.add_argument("--n-init",     type=int, default=6,
                        help="LHS initialization evaluations (default 6)")
    parser.add_argument("--n-iter",     type=int, default=20,
                        help="BayesOpt iterations (default 20)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--resume",     default=None,
                        help="Path to existing gc_fit_history.csv to resume")
    parser.add_argument("--abaqus-cmd", default="abaqus")
    parser.add_argument("--work-dir",   default=None,
                        help="Directory for ABAQUS jobs (default: results/gc_fit_runs)")
    parser.add_argument("--plot",       action="store_true")
    args = parser.parse_args()

    history_csv = args.resume or os.path.join(RESULTS_DIR, "gc_fit_history.csv")

    # ── Runner ────────────────────────────────────────────────────────────
    if args.synthetic:
        runner  = SyntheticFEMRunner(noise_std=0.3, seed=args.seed)
        rf2_exp = runner.get_experimental_rf2()
        print("[*] Synthetic mode: target params =", SyntheticFEMRunner.TRUE_PARAMS)
        print("[*] Target RF2 [N]:", np.round(rf2_exp, 2))
    elif args.exp_csv:
        df_exp  = pd.read_csv(args.exp_csv)
        rf2_exp = df_exp.sort_values("cut_depth_um")["RF2_peak_N"].values
        fem_dir = os.path.join(os.path.dirname(__file__), "..", "fem")
        work_dir = args.work_dir or os.path.join(RESULTS_DIR, "gc_fit_runs")
        runner  = AbaqusFEMRunner(fem_dir=fem_dir, work_dir=work_dir,
                                  abaqus_cmd=args.abaqus_cmd)
        print(f"[*] Experimental RF2 loaded from {args.exp_csv}")
        print(f"    cut_depths = {CUT_DEPTHS_UM} µm")
        print(f"    RF2_exp    = {np.round(rf2_exp, 2)} N")
    else:
        parser.error("Specify --synthetic or --exp-csv <path>")

    # ── Run BayesOpt ──────────────────────────────────────────────────────
    opt    = GcBayesOpt(runner=runner, rf2_exp=rf2_exp)
    result = opt.run(n_init=args.n_init, n_iter=args.n_iter,
                     seed=args.seed, history_csv=history_csv)

    print(f"\n{'='*60}")
    print("  Calibration result")
    print(f"{'='*60}")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Save final summary
    summary_path = os.path.join(RESULTS_DIR, "gc_fit_summary.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[✓] Summary → {summary_path}")
    print(f"[✓] History → {history_csv}")

    if args.plot:
        opt.plot()


if __name__ == "__main__":
    main()
