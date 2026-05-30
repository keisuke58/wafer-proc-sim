"""
TAIKO® Process GP Surrogate — Warpage Prediction.

Data: Wu et al. (2023) Journal of Mechanics 39:191-198
      DOI: 10.1093/jom/ufad018
      9 Taguchi L9 experiments, 300 mm wafer, two-stage grinding.

Features (Z2 fine-grinding stage — dominant for final warpage):
    z2_wheel_speed_rpm, z2_wafer_speed_rpm, z2_feed_um_s,
    z1_wheel_speed_rpm (rough stage influence)

Target: actual_warpage_mm

TAIKO® relevance:
    DISCO TAIKO® leaves a ~3 mm thick edge ring to support the ultra-thin
    center. Minimising warpage during grinding is critical — excessive
    warpage causes die-attach failures in HBM stacking.
    This GP maps grinding recipe → warpage, enabling BO-based recipe
    optimisation directly relevant to TAIKO® process development.

Usage:
    python ml/taiko_grinding_gp.py [--loo] [--plot] [--optimise]
"""

import argparse
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data",
                           "experimental", "oxford2023_taguchi_warpage.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                           "gp_taiko_warpage.pkl")
RESULTS    = os.path.join(os.path.dirname(__file__), "..", "results")

FEATURE_COLS = ["z1_wafer_speed_rpm", "z2_wheel_speed_rpm", "z2_feed_um_s"]
TARGET_COL   = "actual_warpage_mm"

# Parameter bounds for optimisation (TAIKO® typical ranges)
BOUNDS = {
    "z1_wafer_speed_rpm": (150, 250),
    "z2_wheel_speed_rpm": (1400, 2200),
    "z2_feed_um_s":       (0.25, 0.35),
}


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_PATH, comment="#")
    X  = df[FEATURE_COLS].values.astype(float)
    y  = df[TARGET_COL].values.astype(float)
    return X, y, df


# ── Model ─────────────────────────────────────────────────────────────────────

class TaikoWarpageGP:
    def __init__(self):
        kernel = (
            ConstantKernel(1.0, (0.1, 1e3))
            * Matern(length_scale=[50., 300., 0.05],
                     length_scale_bounds=[(5, 500), (10, 2000), (0.005, 0.5)],
                     nu=2.5)
            + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 0.5))
        )
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=20,
            normalize_y=False, alpha=0.0)
        self.is_fitted = False

    def fit(self, X, y):
        Xs = self.scaler_x.fit_transform(X)
        ys = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(Xs, ys)
        self.is_fitted = True
        print(f"  kernel: {self.gp.kernel_}")
        return self

    def predict(self, X, return_std=False):
        Xs = self.scaler_x.transform(X)
        mu, sigma = self.gp.predict(Xs, return_std=True)
        mu = self.scaler_y.inverse_transform(mu.reshape(-1, 1)).ravel()
        sigma = sigma * float(self.scaler_y.scale_[0])
        return (mu, sigma) if return_std else mu

    def loo_cv(self, X, y):
        loo = LeaveOneOut()
        preds = np.zeros_like(y)
        for tr, te in loo.split(X):
            tmp = TaikoWarpageGP()
            tmp.fit(X[tr], y[tr])
            preds[te] = tmp.predict(X[te])
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        r2   = float(r2_score(y, preds))
        print(f"  LOO RMSE = {rmse:.3f} mm    R² = {r2:.4f}")
        return {"rmse": rmse, "r2": r2, "loo_preds": preds}

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[✓] Saved → {path}")

    @staticmethod
    def load(path=MODEL_PATH):
        return joblib.load(path)


# ── Bayesian Optimisation (minimise warpage) ──────────────────────────────────

def expected_improvement(X_cand, model, y_best, xi=0.01):
    mu, sigma = model.predict(X_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    imp   = y_best - mu - xi
    Z     = imp / sigma
    ei    = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


def optimise_recipe(model, y_best, n_random=2000):
    rng = np.random.default_rng(42)
    lows  = np.array([BOUNDS[f][0] for f in FEATURE_COLS])
    highs = np.array([BOUNDS[f][1] for f in FEATURE_COLS])
    X_cand = rng.uniform(lows, highs, size=(n_random, len(FEATURE_COLS)))
    ei = expected_improvement(X_cand, model, y_best)
    top5 = np.argsort(ei)[::-1][:5]
    return X_cand[top5], ei[top5]


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_results(model, X, y, df, out_dir=RESULTS):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    sweep_params = [
        ("z1_wafer_speed_rpm", np.linspace(150, 250, 80),   "Z1 Wafer Speed [rpm]"),
        ("z2_wheel_speed_rpm", np.linspace(1400, 2200, 80), "Z2 Wheel Speed [rpm]"),
        ("z2_feed_um_s",       np.linspace(0.25, 0.35, 80), "Z2 Feed Rate [µm/s]"),
    ]
    ref = {f: float(np.mean([BOUNDS[f][0], BOUNDS[f][1]])) for f in FEATURE_COLS}

    for ax, (feat, x_vals, xlabel) in zip(axes, sweep_params):
        X_sw = np.tile([ref[f] for f in FEATURE_COLS], (len(x_vals), 1))
        X_sw[:, FEATURE_COLS.index(feat)] = x_vals
        mu, sig = model.predict(X_sw, return_std=True)
        ax.plot(x_vals, mu, color="#d62728", lw=2, label="GP mean")
        ax.fill_between(x_vals, mu - 2*sig, mu + 2*sig,
                        alpha=0.18, color="#d62728", label="±2σ")
        ax.scatter(df[feat], df[TARGET_COL],
                   color="#2166ac", s=70, zorder=5, edgecolors="k",
                   lw=0.6, label="Oxford 2023")
        ax.axhline(1.0, color="gray", ls=":", lw=1.2, label="1.0 mm target")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Warpage [mm]", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "TAIKO® Back-Grinding GP Surrogate — Warpage Prediction\n"
        "(300 mm wafer, 780→360 µm, Drucker-Prager + Wu 2023 Taguchi L9)",
        fontsize=10,
    )
    plt.tight_layout()
    out = os.path.join(out_dir, "taiko_warpage_gp.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Plot → {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loo",      action="store_true")
    parser.add_argument("--plot",     action="store_true")
    parser.add_argument("--optimise", action="store_true")
    args = parser.parse_args()

    X, y, df = load_data()
    print(f"[*] Loaded {len(df)} Taguchi L9 data points")
    print(f"    Warpage: {y.min():.2f} – {y.max():.2f} mm  "
          f"(mean {y.mean():.2f})")

    model = TaikoWarpageGP()

    if args.loo:
        print("\n[*] LOO cross-validation ...")
        model.loo_cv(X, y)

    print("\n[*] Fitting on full dataset ...")
    model.fit(X, y)
    model.save()

    if args.plot:
        plot_results(model, X, y, df)

    if args.optimise:
        print("\n[*] Bayesian optimisation — minimise warpage ...")
        y_best = float(y.min())
        print(f"    Current best: {y_best:.3f} mm")
        X_opt, ei_vals = optimise_recipe(model, y_best)
        print("\n  Top-5 recommended grinding recipes:")
        header = "  " + "  ".join(f"{f:>22}" for f in FEATURE_COLS)
        print(header + "   warpage(pred)    EI")
        for xr, ei in zip(X_opt, ei_vals):
            mu, sig = model.predict(xr.reshape(1, -1), return_std=True)
            vals = "  ".join(f"{v:>22.1f}" for v in xr)
            print(f"  {vals}   {mu[0]:.3f}±{sig[0]:.3f} mm   {ei:.4f}")

    # Always show spot predictions
    print("\n[*] Prediction demo (fixed z2_wafer=250rpm):")
    demos = [
        [150, 2200, 0.25],  # slow wafer, high wheel, low feed (low warpage expected)
        [250, 2200, 0.35],  # fast wafer, high feed (high warpage expected)
        [150, 1800, 0.30],  # slow wafer, mid conditions
    ]
    for tc in demos:
        mu, s = model.predict(np.array([tc]), return_std=True)
        labels = dict(zip(FEATURE_COLS, tc))
        print(f"  z1_wafer={labels['z1_wafer_speed_rpm']:.0f}rpm  "
              f"z2_wheel={labels['z2_wheel_speed_rpm']:.0f}rpm  "
              f"feed={labels['z2_feed_um_s']:.2f}µm/s  "
              f"→ warpage = {mu[0]:.3f} ± {s[0]:.3f} mm")


if __name__ == "__main__":
    main()
