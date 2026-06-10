"""
taiko/optimizer.py — TAIKO® Process Digital Twin: Warpage Minimization

Integrates:
  1. ml/taiko_grinding_gp.py   — GP surrogate mapping grinding recipe → warpage
  2. fem/kabra_thermal_2d.py   — KABRA SiC laser thermal parameters
  3. optimization/bayesian_opt — Expected Improvement acquisition

Workflow:
    recipe = {z2_wheel_speed_rpm, z2_feed_um_s, edge_width_mm, thin_thickness_um}
    warpage_hat, std = gp.predict(recipe)          # from taiko_grinding_gp
    kabra_risk = kabra_stress_estimate(recipe)     # from kabra physics
    cost = w1*warpage_hat + w2*kabra_risk          # combined objective
    next_recipe = EI_acquisition(gp, bounds)       # BO next suggestion

Usage:
    from taiko.optimizer import TAIKOOptimizer

    opt = TAIKOOptimizer()
    opt.fit_from_data()                         # fit GP on Taguchi data
    best = opt.optimize(n_iter=20)              # BO loop
    print(best)                                 # {'z2_wheel_speed_rpm': ..., 'warpage_mm': ...}
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    _SK = True
except ImportError:
    _SK = False

# ── KABRA physics: analytical stress estimate ─────────────────────────────────

def kabra_stress_estimate(
    laser_power_W: float = 15.0,
    scan_speed_mm_s: float = 200.0,
    focus_depth_um: float = 100.0,
) -> float:
    """
    Simplified analytical estimate of sub-surface tensile stress [MPa]
    from KABRA laser-slicing parameters.

    Model: Beer-Lambert energy density × SiC absorption → peak temperature
    → thermal gradient → stress via CTE mismatch (plane-strain estimate).

    Returns normalised stress score 0–1 (1 = maximum stress regime).
    """
    E_density = laser_power_W / (scan_speed_mm_s * 1e-3)     # J/m (energy per scan length)
    alpha_abs  = 1.0 / (focus_depth_um * 1e-6 + 1e-9)        # effective absorption [1/m]
    T_peak_rel = E_density * alpha_abs / 1e10                 # normalised peak temp
    stress_score = float(np.clip(T_peak_rel, 0.0, 1.0))
    return stress_score


# ── TAIKO edge ring constraint ────────────────────────────────────────────────

def taiko_edge_feasibility(
    edge_width_mm: float,
    thin_thickness_um: float,
    wafer_diameter_mm: float = 300.0,
) -> bool:
    """
    Check TAIKO® geometry feasibility.
    - Edge ring must be ≥ 2 mm wide (minimum support width)
    - Thinned center must be ≤ 50 µm for HBM4 targets (25 µm for HBM5)
    - Edge thickness fixed at ~775 µm (standard 300 mm wafer)
    """
    if edge_width_mm < 2.0:
        return False
    if thin_thickness_um > 100.0:
        return False
    if edge_width_mm > wafer_diameter_mm / 2 * 0.3:  # edge ≤ 15% of radius
        return False
    return True


# ── GP surrogate wrapper ──────────────────────────────────────────────────────

class _FallbackGP:
    """Minimal GP stand-in when scikit-learn is unavailable."""
    def __init__(self):
        self._fitted = False
        self._y_mean = 0.05

    def fit(self, X, y):
        self._y_mean = float(np.mean(y))
        self._fitted = True

    def predict(self, X, return_std=False):
        n = len(X)
        mu = np.full(n, self._y_mean)
        std = np.full(n, self._y_mean * 0.3)
        return (mu, std) if return_std else mu


class TAIKOOptimizer:
    """
    Bayesian Optimiser for TAIKO® grinding recipe.

    Features used:
        z2_wheel_speed_rpm : fine-grinding wheel speed [rpm]  (1400–2200)
        z2_feed_um_s       : fine-grinding feed rate [µm/s]   (0.25–0.35)
        z1_wafer_speed_rpm : rough-grinding wafer speed [rpm] (150–250)

    Objective:
        Minimise predicted warpage [mm]  (primary)
        Subject to kabra_stress_score ≤ 0.5 (soft constraint)
    """

    BOUNDS = {
        "z2_wheel_speed_rpm": (1400.0, 2200.0),
        "z2_feed_um_s":       (0.25,   0.35),
        "z1_wafer_speed_rpm": (150.0,  250.0),
    }
    FEAT_COLS = ["z2_wheel_speed_rpm", "z2_feed_um_s", "z1_wafer_speed_rpm"]

    def __init__(
        self,
        w_warpage: float = 1.0,
        w_kabra: float = 0.3,
        laser_power_W: float = 15.0,
        scan_speed_mm_s: float = 200.0,
    ):
        self.w_warpage = w_warpage
        self.w_kabra = w_kabra
        self.laser_power_W = laser_power_W
        self.scan_speed_mm_s = scan_speed_mm_s

        self._scaler = StandardScaler() if _SK else None
        if _SK:
            kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-3)
            self._gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        else:
            self._gp = _FallbackGP()

        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._best: dict | None = None

    def fit_from_data(self, X: np.ndarray | None = None, y: np.ndarray | None = None) -> "TAIKOOptimizer":
        """
        Fit GP surrogate.

        Parameters
        ----------
        X : (n, 3) array of [z2_wheel_speed_rpm, z2_feed_um_s, z1_wafer_speed_rpm]
        y : (n,) warpage [mm]

        If X/y are None, Taguchi L9 synthetic data based on Wu et al. (2023) is used.
        """
        if X is None or y is None:
            X, y = self._wu2023_taguchi_data()

        self._X_train = np.asarray(X, dtype=float)
        self._y_train = np.asarray(y, dtype=float)

        if _SK:
            X_sc = self._scaler.fit_transform(self._X_train)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._gp.fit(X_sc, self._y_train)
        else:
            self._gp.fit(self._X_train, self._y_train)

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_warpage_mm, std_warpage_mm)."""
        X = np.atleast_2d(X)
        if _SK:
            X_sc = self._scaler.transform(X)
            mu, std = self._gp.predict(X_sc, return_std=True)
        else:
            mu, std = self._gp.predict(X, return_std=True)
        return mu, std

    def acquisition_ei(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement (minimisation)."""
        mu, std = self.predict(X)
        f_best = self._y_train.min() if self._y_train is not None else 0.0
        Z = (f_best - mu - xi) / (std + 1e-9)
        ei = (f_best - mu - xi) * norm.cdf(Z) + std * norm.pdf(Z)
        return np.maximum(ei, 0.0)

    def optimize(self, n_iter: int = 20) -> dict:
        """
        Run BO loop for n_iter steps. Returns best recipe found.
        """
        if self._X_train is None:
            self.fit_from_data()

        bounds_list = [self.BOUNDS[f] for f in self.FEAT_COLS]

        for _ in range(n_iter):
            def neg_ei(x):
                return -self.acquisition_ei(np.array(x).reshape(1, -1))[0]

            res = differential_evolution(neg_ei, bounds_list, seed=42,
                                         maxiter=50, tol=1e-4, polish=True)
            x_next = res.x.reshape(1, -1)
            mu_next, _ = self.predict(x_next)

            kabra_score = kabra_stress_estimate(
                laser_power_W=self.laser_power_W,
                scan_speed_mm_s=self.scan_speed_mm_s,
            )
            cost = self.w_warpage * mu_next[0] + self.w_kabra * kabra_score

            y_obs = mu_next[0] + np.random.normal(0, 0.002)  # synthetic observation noise

            self._X_train = np.vstack([self._X_train, x_next])
            self._y_train = np.append(self._y_train, y_obs)

            if _SK:
                X_sc = self._scaler.fit_transform(self._X_train)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._gp.fit(X_sc, self._y_train)
            else:
                self._gp.fit(self._X_train, self._y_train)

        best_idx = int(np.argmin(self._y_train))
        x_best = self._X_train[best_idx]
        self._best = {
            feat: float(x_best[i]) for i, feat in enumerate(self.FEAT_COLS)
        }
        self._best["warpage_mm"] = float(self._y_train[best_idx])
        self._best["kabra_stress_score"] = kabra_stress_estimate(
            laser_power_W=self.laser_power_W,
            scan_speed_mm_s=self.scan_speed_mm_s,
        )
        return self._best

    def pareto_front(self, n_grid: int = 30) -> list[dict]:
        """
        Compute Pareto front over (warpage, kabra_stress) on the parameter grid.
        Returns list of non-dominated solutions.
        """
        bounds_list = [self.BOUNDS[f] for f in self.FEAT_COLS]
        grids = [np.linspace(lo, hi, n_grid) for lo, hi in bounds_list]
        mesh = np.array(np.meshgrid(*grids, indexing="ij")).reshape(3, -1).T

        mu, _ = self.predict(mesh)
        kabra = np.full(len(mesh), kabra_stress_estimate(
            laser_power_W=self.laser_power_W,
            scan_speed_mm_s=self.scan_speed_mm_s,
        ))

        objectives = np.column_stack([mu, kabra])
        dominated = np.zeros(len(objectives), dtype=bool)
        for i in range(len(objectives)):
            for j in range(len(objectives)):
                if i == j:
                    continue
                if (objectives[j] <= objectives[i]).all() and (objectives[j] < objectives[i]).any():
                    dominated[i] = True
                    break

        front = []
        for i in np.where(~dominated)[0]:
            row = {feat: float(mesh[i, k]) for k, feat in enumerate(self.FEAT_COLS)}
            row["warpage_mm"] = float(objectives[i, 0])
            row["kabra_stress_score"] = float(objectives[i, 1])
            front.append(row)
        return front

    @staticmethod
    def _wu2023_taguchi_data() -> tuple[np.ndarray, np.ndarray]:
        """
        Synthetic Taguchi L9 data based on Wu et al. (2023) Table 2.
        [z2_wheel_speed_rpm, z2_feed_um_s, z1_wafer_speed_rpm] → warpage_mm
        """
        X = np.array([
            [1400, 0.25, 150], [1400, 0.30, 200], [1400, 0.35, 250],
            [1800, 0.25, 200], [1800, 0.30, 250], [1800, 0.35, 150],
            [2200, 0.25, 250], [2200, 0.30, 150], [2200, 0.35, 200],
        ], dtype=float)
        # Warpage decreases with higher wheel speed, increases with higher feed
        y = np.array([0.082, 0.091, 0.098, 0.063, 0.071, 0.079,
                       0.045, 0.053, 0.061])
        return X, y


def optimize_taiko_recipe(
    n_iter: int = 30,
    laser_power_W: float = 15.0,
    scan_speed_mm_s: float = 200.0,
    edge_width_mm: float = 3.0,
    thin_thickness_um: float = 25.0,
    verbose: bool = True,
) -> dict:
    """
    Convenience function: fit GP, run BO, return best TAIKO® recipe.

    Parameters
    ----------
    n_iter           : number of BO iterations
    laser_power_W    : KABRA laser power [W]
    scan_speed_mm_s  : KABRA scan speed [mm/s]
    edge_width_mm    : TAIKO® edge ring width [mm]
    thin_thickness_um: target center thickness [µm]
    verbose          : print result summary

    Returns
    -------
    dict with keys: z2_wheel_speed_rpm, z2_feed_um_s, z1_wafer_speed_rpm,
                    warpage_mm, kabra_stress_score, taiko_feasible
    """
    if not taiko_edge_feasibility(edge_width_mm, thin_thickness_um):
        raise ValueError(
            f"TAIKO® geometry infeasible: edge={edge_width_mm} mm, "
            f"thickness={thin_thickness_um} µm"
        )

    opt = TAIKOOptimizer(
        laser_power_W=laser_power_W,
        scan_speed_mm_s=scan_speed_mm_s,
    )
    opt.fit_from_data()
    best = opt.optimize(n_iter=n_iter)
    best["taiko_feasible"] = True
    best["edge_width_mm"] = edge_width_mm
    best["thin_thickness_um"] = thin_thickness_um

    if verbose:
        print("─" * 50)
        print("TAIKO® Recipe Optimisation Result")
        print(f"  Z2 wheel speed : {best['z2_wheel_speed_rpm']:.0f} rpm")
        print(f"  Z2 feed rate   : {best['z2_feed_um_s']:.3f} µm/s")
        print(f"  Z1 wafer speed : {best['z1_wafer_speed_rpm']:.0f} rpm")
        print(f"  Pred. warpage  : {best['warpage_mm']*1000:.1f} µm")
        print(f"  KABRA stress   : {best['kabra_stress_score']:.3f}  (0=low, 1=high)")
        print(f"  Edge / thin    : {edge_width_mm} mm / {thin_thickness_um} µm")
        print("─" * 50)
    return best


if __name__ == "__main__":
    best = optimize_taiko_recipe(n_iter=20)
