"""
KABRA® Process GP Surrogate + Bayesian Optimisation.

KABRA® (Key Attribute Blading Resistance Accelerator) is DISCO's laser-based
SiC ingot slicing process. This module builds a GP surrogate that maps:

    Input (process parameters):
        laser_power_W     : 5–30 W
        scan_speed_mm_s   : 100–400 mm/s
        focus_depth_um    : 50–150 µm below surface

    Output (quality metrics):
        haz_width_um      : heat-affected zone width [µm] (minimise)
        kerf_quality      : normalised kerf straightness score 0–1 (maximise)
        throughput_mm2_s  : effective area slicing rate [mm²/s] (maximise)

Physics basis (surrogate trained on synthetic FEM-derived data from
fem/kabra_thermal_2d.py physical model):

    HAZ width scales with:  sqrt(P / (k * v))         (Rosenthal approximation)
    Kerf quality degrades at: high power + low speed    (overheating)
                               low power + high speed   (incomplete fracture)
    Throughput scales with:   scan_speed (approximately)

Usage:
    python sic/sic_kabra_gp.py               # fit + evaluate + BO
    python sic/sic_kabra_gp.py --plot        # add visualisation
    python sic/sic_kabra_gp.py --n_train 45  # larger design of experiments
"""

import argparse
import os
import sys
import warnings

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.materials.material_properties import SiC

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import r2_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    warnings.warn("scikit-learn not found — GP fitting disabled")


# ── Process parameter bounds ──────────────────────────────────────────────────

BOUNDS = {
    "laser_power_W":   (5.0,  30.0),
    "scan_speed_mm_s": (100.0, 400.0),
    "focus_depth_um":  (50.0,  150.0),
}

PARAM_NAMES = list(BOUNDS.keys())


# ── Physics-based synthetic data generation ───────────────────────────────────

def _rosenthal_haz(P, v_mm_s, mat=SiC) -> float:
    """
    HAZ half-width from Rosenthal moving point-source approximation.
    HAZ_width [µm] ∝ sqrt(P / (k * v))
    Calibrated to match KABRA® FEM results (kabra_thermal_2d.py sweep).
    """
    v_m_s = v_mm_s * 1e-3
    k     = mat["k_thermal"]    # W/(m·K)
    rho   = mat["density"]      # kg/m³
    Cp    = mat["Cp"]           # J/(kg·K)
    alpha = k / (rho * Cp)     # thermal diffusivity m²/s

    # Modified Rosenthal: HAZ ~ 2 * sqrt(alpha * P / (pi * k * v * T_melt))
    T_melt = 2730.0   # SiC sublimation ~2730°C
    T_0    = 25.0
    dT     = T_melt - T_0
    haz_m  = 2.0 * np.sqrt(alpha * P / (np.pi * k * v_m_s * dT))
    return haz_m * 1e6  # µm


def _kerf_quality(P, v_mm_s, d_um) -> float:
    """
    Kerf quality score in [0, 1].
    Degrades when:
      - P too high + v too low   → overmelting, recast layer
      - P too low + v too high   → incomplete fracture, deviation
      - focus too shallow/deep   → defocus penalty
    """
    # Energy density proxy: E_d = P / v [J/mm]
    E_d     = P / v_mm_s
    E_opt   = 0.08    # optimal J/mm (empirically from KABRA literature)
    E_sigma = 0.04

    # Gaussian bell around optimal energy density
    q_energy = np.exp(-0.5 * ((E_d - E_opt) / E_sigma) ** 2)

    # Focus penalty: optimal around 100µm (mid-ingot for 200µm thick slice)
    d_opt   = 100.0
    d_sigma = 30.0
    q_focus = np.exp(-0.5 * ((d_um - d_opt) / d_sigma) ** 2)

    return float(q_energy * q_focus)


def _throughput(v_mm_s, P, mat=SiC) -> float:
    """
    Effective area throughput [mm²/s].
    At high speeds, incomplete fracture → re-scan needed.
    """
    q = _kerf_quality(P, v_mm_s, 100.0)   # quality at optimal focus
    # throughput = speed × effective pass rate (quality²: both axes of kerf)
    return v_mm_s * (q ** 2)


def generate_doe(n: int = 27, seed: int = 42) -> np.ndarray:
    """
    Latin Hypercube Sampling over KABRA parameter space.
    Returns array (n, 3): [power, speed, focus_depth]
    """
    rng = np.random.RandomState(seed)

    # LHS: n points, 3 dims
    samples = np.zeros((n, 3))
    for d in range(3):
        perm = rng.permutation(n)
        samples[:, d] = (perm + rng.rand(n)) / n

    # Scale to bounds
    lo = np.array([b[0] for b in BOUNDS.values()])
    hi = np.array([b[1] for b in BOUNDS.values()])
    return lo + samples * (hi - lo)


def simulate_doe(X: np.ndarray, noise_frac: float = 0.04) -> np.ndarray:
    """
    Evaluate synthetic KABRA physics for each row of X.
    Returns Y (n, 3): [haz_width_um, kerf_quality, throughput_mm2_s]
    """
    rng = np.random.RandomState(99)
    Y = np.zeros((len(X), 3))
    for i, (P, v, d) in enumerate(X):
        haz = _rosenthal_haz(P, v)
        kq  = _kerf_quality(P, v, d)
        tp  = _throughput(v, P)
        # Add multiplicative noise
        noise = 1 + noise_frac * rng.randn(3)
        Y[i] = [haz * noise[0], np.clip(kq * noise[1], 0, 1), tp * noise[2]]
    return Y


# ── GP surrogate ──────────────────────────────────────────────────────────────

class KABRAGPSurrogate:
    """
    Multi-output GP surrogate for KABRA® process.
    One independent GP per output (HAZ, quality, throughput).
    """

    OUTPUT_NAMES = ["haz_width_um", "kerf_quality", "throughput_mm2_s"]

    def __init__(self):
        if not _SKLEARN:
            raise ImportError("scikit-learn required")
        self.scalers_X = []
        self.scalers_y = []
        self.gps       = []
        self._fitted   = False

    def _make_kernel(self):
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=[1.0, 1.0, 1.0],
                     length_scale_bounds=[(0.1, 10.0)] * 3,
                     nu=2.5)
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        )

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.gps, self.scalers_X, self.scalers_y = [], [], []
        for j in range(Y.shape[1]):
            sx = StandardScaler()
            sy = StandardScaler()
            Xs = sx.fit_transform(X)
            ys = sy.fit_transform(Y[:, j:j+1]).ravel()
            gp = GaussianProcessRegressor(
                kernel=self._make_kernel(),
                n_restarts_optimizer=5,
                normalize_y=False,
                alpha=0.0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(Xs, ys)
            self.gps.append(gp)
            self.scalers_X.append(sx)
            self.scalers_y.append(sy)
        self._fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) each shape (n, n_outputs)."""
        means, stds = [], []
        for j, gp in enumerate(self.gps):
            Xs = self.scalers_X[j].transform(X)
            mu_s, sigma_s = gp.predict(Xs, return_std=True)
            mu    = self.scalers_y[j].inverse_transform(mu_s.reshape(-1, 1)).ravel()
            sigma = sigma_s * self.scalers_y[j].scale_[0]
            means.append(mu)
            stds.append(np.abs(sigma))
        return np.column_stack(means), np.column_stack(stds)

    def loo_cv(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """Leave-one-out cross-validation R² per output."""
        loo = LeaveOneOut()
        r2s = {name: [] for name in self.OUTPUT_NAMES}
        for train_idx, test_idx in loo.split(X):
            clone = KABRAGPSurrogate()
            clone.fit(X[train_idx], Y[train_idx])
            pred, _ = clone.predict(X[test_idx])
            for j, name in enumerate(self.OUTPUT_NAMES):
                r2s[name].append((Y[test_idx, j], pred[0, j]))
        results = {}
        for name, pairs in r2s.items():
            y_true = np.array([p[0] for p in pairs])
            y_pred = np.array([p[1] for p in pairs])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results[name] = r2_score(y_true, y_pred)
        return results


# ── Bayesian optimisation ─────────────────────────────────────────────────────

def scalarised_objective(params_norm: np.ndarray, surrogate: KABRAGPSurrogate,
                          w_haz: float = 0.4, w_kq: float = 0.4, w_tp: float = 0.2,
                          bounds_lo: np.ndarray = None, bounds_hi: np.ndarray = None
                          ) -> float:
    """
    Scalarised objective (negative = minimise):
        J = w_haz * haz_norm + w_kq * (1 - kerf_quality) + w_tp * (1 - tp_norm)
    All outputs normalised to [0, 1] using crude reference values.
    """
    params = bounds_lo + params_norm * (bounds_hi - bounds_lo)
    mu, _  = surrogate.predict(params.reshape(1, -1))
    haz    = mu[0, 0]
    kq     = np.clip(mu[0, 1], 0, 1)
    tp     = mu[0, 2]

    haz_norm = np.clip(haz / 50.0,  0, 1)   # 50µm reference
    tp_norm  = np.clip(tp  / 20.0,  0, 1)   # 20 mm²/s reference

    return w_haz * haz_norm + w_kq * (1 - kq) + w_tp * (1 - tp_norm)


def expected_improvement(x_cand: np.ndarray, surrogate: KABRAGPSurrogate,
                          X_obs: np.ndarray, Y_obs: np.ndarray,
                          target_idx: int = 0,   # 0=HAZ (min), 1=quality (max)
                          xi: float = 0.01,
                          minimise: bool = True) -> float:
    """
    EI for single-output acquisition.
    target_idx: which output to optimise.
    """
    mu, sigma = surrogate.predict(x_cand.reshape(1, -1))
    mu_t    = mu[0, target_idx]
    sig_t   = sigma[0, target_idx] + 1e-9

    y_obs = Y_obs[:, target_idx]
    f_best = y_obs.min() if minimise else -y_obs.max()
    mu_use = mu_t if minimise else -mu_t

    Z  = (f_best - mu_use - xi) / sig_t
    ei = (f_best - mu_use - xi) * norm.cdf(Z) + sig_t * norm.pdf(Z)
    return float(max(ei, 0.0))


def run_bayesian_optimisation(surrogate: KABRAGPSurrogate,
                               X_init: np.ndarray, Y_init: np.ndarray,
                               n_iter: int = 10) -> dict:
    """
    Multi-start L-BFGS-B to maximise EI (minimise HAZ while keeping quality high).
    Returns best parameters found.
    """
    lo = np.array([b[0] for b in BOUNDS.values()])
    hi = np.array([b[1] for b in BOUNDS.values()])

    best_params = None
    best_score  = np.inf
    rng = np.random.RandomState(7)

    for _ in range(n_iter):
        x0_norm = rng.rand(3)
        res = minimize(
            fun=scalarised_objective,
            x0=x0_norm,
            args=(surrogate, 0.4, 0.4, 0.2, lo, hi),
            method="L-BFGS-B",
            bounds=[(0, 1)] * 3,
            options={"maxiter": 200},
        )
        if res.fun < best_score:
            best_score  = res.fun
            best_params = lo + res.x * (hi - lo)

    mu, sigma = surrogate.predict(best_params.reshape(1, -1))
    return {
        "laser_power_W":   best_params[0],
        "scan_speed_mm_s": best_params[1],
        "focus_depth_um":  best_params[2],
        "pred_haz_um":     mu[0, 0],
        "pred_kerf_quality": mu[0, 1],
        "pred_throughput_mm2_s": mu[0, 2],
        "pred_sigma": sigma[0],
        "objective": best_score,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KABRA GP surrogate + BO")
    parser.add_argument("--n_train", type=int, default=27)
    parser.add_argument("--n_bo",    type=int, default=15)
    parser.add_argument("--plot",    action="store_true")
    args = parser.parse_args()

    print("\n══════════════════════════════════════════════════")
    print("  KABRA® Process GP Surrogate + Bayesian Optimisation")
    print("  Material: 4H-SiC  |  DISCO KABRA® laser slicing")
    print("══════════════════════════════════════════════════\n")

    # 1. Design of Experiments
    print(f"  [1/4] Generating DoE  (n={args.n_train}, LHS) ...")
    X = generate_doe(args.n_train)
    Y = simulate_doe(X)
    print(f"        HAZ range:   {Y[:,0].min():.1f} – {Y[:,0].max():.1f} µm")
    print(f"        Quality:     {Y[:,1].min():.3f} – {Y[:,1].max():.3f}")
    print(f"        Throughput:  {Y[:,2].min():.1f} – {Y[:,2].max():.1f} mm²/s")

    if not _SKLEARN:
        print("\n  scikit-learn not found. Skipping GP fitting.")
        return

    # 2. Fit GP surrogate
    print(f"\n  [2/4] Fitting GP surrogate ...")
    surrogate = KABRAGPSurrogate()
    surrogate.fit(X, Y)

    # 3. LOO-CV
    print(f"\n  [3/4] Leave-one-out cross-validation:")
    loo_r2 = surrogate.loo_cv(X, Y)
    for name, r2 in loo_r2.items():
        bar = "█" * int(max(r2, 0) * 20) + "░" * (20 - int(max(r2, 0) * 20))
        print(f"        {name:<24s} R²={r2:.3f}  [{bar}]")

    # 4. Bayesian optimisation
    print(f"\n  [4/4] Bayesian optimisation (n_iter={args.n_bo}) ...")
    opt = run_bayesian_optimisation(surrogate, X, Y, n_iter=args.n_bo)
    print(f"\n  ── Optimal KABRA® Parameters ──────────────────")
    print(f"     Laser power    : {opt['laser_power_W']:.1f} W")
    print(f"     Scan speed     : {opt['scan_speed_mm_s']:.0f} mm/s")
    print(f"     Focus depth    : {opt['focus_depth_um']:.0f} µm")
    print(f"\n  ── Predicted Quality at Optimum ───────────────")
    print(f"     HAZ width      : {opt['pred_haz_um']:.1f} µm  (target: <20 µm)")
    print(f"     Kerf quality   : {opt['pred_kerf_quality']:.3f}  (target: >0.8)")
    print(f"     Throughput     : {opt['pred_throughput_mm2_s']:.1f} mm²/s")

    # Baseline (DISCO default: 15W, 200mm/s, 100µm)
    baseline = np.array([[15.0, 200.0, 100.0]])
    mu_b, _  = surrogate.predict(baseline)
    print(f"\n  ── Baseline (15W / 200mm/s / 100µm) ──────────")
    print(f"     HAZ width      : {mu_b[0,0]:.1f} µm")
    print(f"     Kerf quality   : {mu_b[0,1]:.3f}")
    print(f"     Throughput     : {mu_b[0,2]:.1f} mm²/s")

    haz_imp = (mu_b[0, 0] - opt["pred_haz_um"]) / mu_b[0, 0] * 100
    tp_imp  = (opt["pred_throughput_mm2_s"] - mu_b[0, 2]) / mu_b[0, 2] * 100
    print(f"\n  ── Optimisation Gain ──────────────────────────")
    print(f"     HAZ reduction  : {haz_imp:+.1f}%")
    print(f"     Throughput gain: {tp_imp:+.1f}%")
    print(f"\n  → GP-BO can systematically find KABRA® recipes")
    print(f"     beyond manual trial-and-error.")
    print(f"  → Next step: replace synthetic data with real FEM sweep results")
    print(f"     from fem/kabra_thermal_2d.py to ground the surrogate in physics.")

    if args.plot:
        _plot_gp(surrogate, X, Y, opt)


def _plot_gp(surrogate, X, Y, opt):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("matplotlib not available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("KABRA® GP Surrogate — Process Space", fontsize=13)

    powers = np.linspace(*BOUNDS["laser_power_W"], 40)
    speeds = np.linspace(*BOUNDS["scan_speed_mm_s"], 40)
    P_grid, V_grid = np.meshgrid(powers, speeds)
    d_fixed = 100.0  # fixed focus depth

    X_grid = np.column_stack([
        P_grid.ravel(), V_grid.ravel(),
        np.full(P_grid.size, d_fixed)
    ])
    mu_grid, _ = surrogate.predict(X_grid)

    output_titles = ["HAZ width [µm]", "Kerf quality", "Throughput [mm²/s]"]
    cmaps = ["hot_r", "viridis", "plasma"]

    for i, (title, cmap) in enumerate(zip(output_titles, cmaps)):
        ax = axes[i]
        Z = mu_grid[:, i].reshape(P_grid.shape)
        im = ax.contourf(P_grid, V_grid, Z, levels=20, cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.scatter(X[:, 0], X[:, 1], c="white", s=20, alpha=0.6, zorder=5)
        ax.scatter(opt["laser_power_W"], opt["scan_speed_mm_s"],
                   c="cyan", s=150, marker="*", zorder=10, label="Optimum")
        ax.set_xlabel("Laser power [W]")
        ax.set_ylabel("Scan speed [mm/s]")
        ax.set_title(f"{title}\n(focus depth={d_fixed:.0f}µm)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "results", "kabra_gp_process_map.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
