"""
Hybrid laser+plasma process GP surrogate.

Inputs  (8 features):
    laser_W        [W]      5–30
    scan_mm_s      [mm/s]   100–500
    pulse_kHz      [kHz]    50–200
    n_passes       [-]      1–5
    plasma_s       [s]      30–300
    n_cycles       [-]      10–100
    mask_um        [µm]     3–20
    wafer_um       [µm]     100–400

Outputs (4 targets):
    street_width_um   [µm]   smaller = better (die yield)
    die_strength_MPa  [MPa]  larger  = better
    HAZ_depth_um      [µm]   smaller = better
    throughput_mm2_s  [mm²/s] larger = better

Usage:
    # Generate synthetic data then train
    python ml/hybrid_process_gp.py --generate --n 400
    python ml/hybrid_process_gp.py --train --csv data/experimental/hybrid_synthetic.csv
    python ml/hybrid_process_gp.py --train --plot

    # Predict from saved model
    from ml.hybrid_process_gp import HybridGPSurrogate
    model = HybridGPSurrogate.load()
    mu, sigma = model.predict(X)
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_PATH   = os.path.join(RESULTS_DIR, "hybrid_gp.pkl")
DEFAULT_CSV  = os.path.join(os.path.dirname(__file__), "..",
                             "data", "experimental", "hybrid_synthetic.csv")

FEATURE_COLS = [
    "laser_W", "scan_mm_s", "pulse_kHz", "n_passes",
    "plasma_s", "n_cycles", "mask_um", "wafer_um",
    "pulse_regime_code",   # ① 0=ns / 1=ps / 2=fs
    "duty_cycle",          # ② pulsed plasma
    "beol_k",              # ③ Low-k dielectric constant
    "beol_thickness_um",   # ③ BEOL stack thickness
]

TARGET_COLS = [
    "street_width_um",
    "die_strength_MPa",
    "HAZ_depth_um",
    "throughput_mm2_s",
    "delamination_risk",   # ③ 0=safe → 1=delamination
]

# Physical bounds for normalisation / constraint checking
BOUNDS = np.array([
    [5.0,    30.0],    # laser_W
    [100.0,  500.0],   # scan_mm_s
    [50.0,   200.0],   # pulse_kHz
    [1.0,    5.0],     # n_passes
    [600.0,  3600.0],  # plasma_s [s] (10–60 min)
    [10.0,   100.0],   # n_cycles
    [3.0,    20.0],    # mask_um
    [25.0,   400.0],   # wafer_um (④ thin wafer: down to 25µm)
    [0.0,    2.0],     # pulse_regime_code (continuous relaxation for GP)
    [0.1,    1.0],     # duty_cycle
    [1.8,    3.5],     # beol_k
    [2.0,    10.0],    # beol_thickness_um
])


def _build_kernel(n_features: int = 8):
    """Anisotropic RBF kernel with per-feature length scales."""
    ls_init  = [50.0] * n_features
    ls_bound = [(0.1, 5000.0)] * n_features
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=ls_init, length_scale_bounds=ls_bound)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
    )


class HybridGPSurrogate:
    """
    Multi-output GP surrogate for hybrid laser+plasma process.
    One independent GP per output target.
    """

    def __init__(self):
        n = len(FEATURE_COLS)
        self.scaler_x  = StandardScaler()
        self.scalers_y = [StandardScaler() for _ in TARGET_COLS]
        self.gps = [
            GaussianProcessRegressor(
                kernel=_build_kernel(n),
                n_restarts_optimizer=5,
                normalize_y=False,
                alpha=0.0)
            for _ in TARGET_COLS
        ]
        self.is_fitted = False

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        X : (N, 8) feature matrix
        Y : (N, 4) target matrix  [street_width, die_strength, HAZ_depth, throughput]
        """
        Xs = self.scaler_x.fit_transform(X)
        for i, (gp, scaler, name) in enumerate(
                zip(self.gps, self.scalers_y, TARGET_COLS)):
            ys = scaler.fit_transform(Y[:, i:i+1]).ravel()
            gp.fit(Xs, ys)
            print(f"  [{name}] kernel: {gp.kernel_}")
        self.is_fitted = True
        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray,
                return_std: bool = False) -> tuple:
        """
        Returns (mu, sigma) if return_std else mu.
        mu, sigma : (N, 4)
        """
        assert self.is_fitted, "Call fit() first."
        Xs   = self.scaler_x.transform(X)
        mus  = []
        stds = []
        for gp, scaler in zip(self.gps, self.scalers_y):
            mu_s, std_s = gp.predict(Xs, return_std=True)
            mu  = scaler.inverse_transform(mu_s.reshape(-1, 1)).ravel()
            std = std_s * scaler.scale_[0]
            mus.append(mu); stds.append(std)
        mu_arr  = np.column_stack(mus)
        std_arr = np.column_stack(stds)
        return (mu_arr, std_arr) if return_std else mu_arr

    # ── LOO cross-validation ──────────────────────────────────────────────────
    def loo_cv(self, X: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
        """Leave-one-out CV; returns per-target RMSE and R²."""
        loo  = LeaveOneOut()
        preds = np.zeros_like(Y)

        for train_idx, test_idx in loo.split(X):
            surrogate_loo = HybridGPSurrogate()
            surrogate_loo.fit(X[train_idx], Y[train_idx])
            preds[test_idx] = surrogate_loo.predict(X[test_idx])

        rows = []
        for i, name in enumerate(TARGET_COLS):
            rmse = np.sqrt(mean_squared_error(Y[:, i], preds[:, i]))
            r2   = r2_score(Y[:, i], preds[:, i])
            rows.append({"target": name, "LOO_RMSE": round(rmse, 4), "R2": round(r2, 4)})
        return pd.DataFrame(rows)

    # ── Persist ───────────────────────────────────────────────────────────────
    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "HybridGPSurrogate":
        return joblib.load(path)


# ── Training helpers ──────────────────────────────────────────────────────────

def load_data(csv_path: str) -> tuple:
    df = pd.read_csv(csv_path)
    X  = df[FEATURE_COLS].values.astype(float)
    Y  = df[TARGET_COLS].values.astype(float)
    return X, Y, df


def train(csv_path: str = DEFAULT_CSV,
          run_loo: bool = True,
          plot: bool = False) -> HybridGPSurrogate:
    X, Y, df = load_data(csv_path)
    print(f"Data: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Targets: {TARGET_COLS}")

    model = HybridGPSurrogate()
    print("\nFitting GPs...")
    model.fit(X, Y)

    if run_loo:
        print("\nRunning LOO CV (may take a minute)...")
        cv = model.loo_cv(X, Y)
        print(cv.to_string(index=False))
        os.makedirs(RESULTS_DIR, exist_ok=True)
        cv.to_csv(os.path.join(RESULTS_DIR, "hybrid_gp_loo_cv.csv"), index=False)

    model.save()

    if plot:
        _plot_sensitivity(model, X, Y)
        _plot_loo_scatter(model, X, Y)

    return model


def _plot_sensitivity(model: HybridGPSurrogate,
                       X: np.ndarray, Y: np.ndarray):
    """1D sensitivity sweeps for each feature."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_feat  = len(FEATURE_COLS)
    n_targ  = len(TARGET_COLS)
    fig, axes = plt.subplots(n_feat, n_targ, figsize=(4 * n_targ, 3 * n_feat),
                              sharey='col')

    X_mean = X.mean(axis=0)
    for fi, fname in enumerate(FEATURE_COLS):
        sweep = np.linspace(BOUNDS[fi, 0], BOUNDS[fi, 1], 50)
        X_sw  = np.tile(X_mean, (50, 1))
        X_sw[:, fi] = sweep
        mu, sigma = model.predict(X_sw, return_std=True)

        for ti, tname in enumerate(TARGET_COLS):
            ax = axes[fi, ti]
            ax.plot(sweep, mu[:, ti], 'b-')
            ax.fill_between(sweep,
                             mu[:, ti] - 2 * sigma[:, ti],
                             mu[:, ti] + 2 * sigma[:, ti],
                             alpha=0.2, color='b')
            ax.set_xlabel(fname, fontsize=8)
            if fi == 0:
                ax.set_title(tname, fontsize=9)
            ax.tick_params(labelsize=7)

    plt.suptitle("Hybrid GP — 1D Sensitivity (others at mean)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "hybrid_gp_sensitivity.png")
    plt.savefig(out, dpi=150)
    print(f"Sensitivity plot → {out}")


def _plot_loo_scatter(model: HybridGPSurrogate,
                       X: np.ndarray, Y: np.ndarray):
    """Predicted vs actual scatter for each target."""
    import matplotlib.pyplot as plt

    mu, _ = model.predict(X)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (ax, name) in enumerate(zip(axes, TARGET_COLS)):
        ax.scatter(Y[:, i], mu[:, i], alpha=0.4, s=15)
        lo = min(Y[:, i].min(), mu[:, i].min())
        hi = max(Y[:, i].max(), mu[:, i].max())
        ax.plot([lo, hi], [lo, hi], 'r--')
        r2 = r2_score(Y[:, i], mu[:, i])
        ax.set(xlabel=f"Actual {name}", ylabel="Predicted", title=f"R²={r2:.3f}")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "hybrid_gp_scatter.png")
    plt.savefig(out, dpi=150)
    print(f"Scatter plot → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true",
                    help="Generate synthetic CSV via plasma_bosch_model")
    ap.add_argument("--n",        type=int,   default=400)
    ap.add_argument("--train",    action="store_true")
    ap.add_argument("--csv",      default=DEFAULT_CSV)
    ap.add_argument("--no-loo",   action="store_true")
    ap.add_argument("--plot",     action="store_true")
    args = ap.parse_args()

    if args.generate:
        from fem.plasma_bosch_model import generate_dataset
        df = generate_dataset(n=args.n)
        os.makedirs(os.path.dirname(DEFAULT_CSV), exist_ok=True)
        df.to_csv(DEFAULT_CSV, index=False)
        print(f"Generated {len(df)} samples → {DEFAULT_CSV}")

    if args.train:
        train(csv_path=args.csv,
              run_loo=not args.no_loo,
              plot=args.plot)
