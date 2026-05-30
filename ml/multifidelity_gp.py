"""
Multi-Fidelity GP (AR1 co-kriging) for SiC Dicing.

Model (Kennedy & O'Hagan 2000):
    y_high(x) = rho * y_low(x) + delta(x)

Where:
    y_low  = FEM deletion_fraction (fast, 5 pts, 2 features: depth, blade_W)
    y_high = experimental chipping  (slow, 26 pts, 4 features)
    rho    = linear scaling factor learned from data
    delta  = discrepancy GP capturing what FEM misses

Benefit over plain fusion GP:
  - Explicitly models the low/high fidelity relationship
  - Propagates FEM uncertainty into the high-fidelity prediction
  - Interpretable: rho tells you how well FEM tracks experiments

Usage:
    python ml/multifidelity_gp.py [--loo] [--plot]
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from validation.experimental_data import CHIPPING_DATA
from ml.train_from_experimental import (
    ExperimentalGPSurrogate, FEATURE_COLS, TARGET_COL, MODEL_PATH, REF,
    _sweep_array,
)

FEM_CSV   = os.path.join(os.path.dirname(__file__), "..", "results",
                          "parametric_summary_extended.csv")
MF_MODEL  = os.path.join(os.path.dirname(__file__), "..", "results",
                          "gp_multifidelity.pkl")
RESULTS   = os.path.join(os.path.dirname(__file__), "..", "results")

# Low-fidelity features (FEM): subset of high-fidelity feature space
LF_FEATURES = ["cut_depth_um", "blade_W_um"]


# ── Low-fidelity GP (FEM) ─────────────────────────────────────────────────────

class LowFidelityGP:
    """GP on FEM deletion_fraction (2-feature)."""
    def __init__(self):
        kernel = (
            ConstantKernel(1.0, (0.1, 1e2))
            * Matern(length_scale=[50., 10.],
                     length_scale_bounds=[(5, 500), (1, 100)],
                     nu=2.5)
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 0.01))
        )
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=15,
            normalize_y=False, alpha=0.0)

    def fit(self, X_lf, y_lf):
        self.scaler_x.fit(X_lf)
        self.scaler_y.fit(y_lf.reshape(-1, 1))
        Xs = self.scaler_x.transform(X_lf)
        ys = self.scaler_y.transform(y_lf.reshape(-1, 1)).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(Xs, ys)
        return self

    def predict(self, X, return_std=False):
        Xs = self.scaler_x.transform(X)
        mu, sigma = self.gp.predict(Xs, return_std=True)
        mu    = self.scaler_y.inverse_transform(mu.reshape(-1, 1)).ravel()
        sigma = sigma * float(self.scaler_y.scale_[0])
        return (mu, sigma) if return_std else mu


# ── AR1 Multi-Fidelity model ──────────────────────────────────────────────────

class AR1MultiFidelityGP:
    """
    Kennedy-O'Hagan AR1 model:
        y_h(x) = rho * y_l(x_lf) + delta(x)

    rho is optimised by MLE over the high-fidelity residuals.
    delta is a GP on the full 4-feature space.
    """

    def __init__(self):
        self.lf_gp  = LowFidelityGP()
        self.rho    = 1.0
        self.delta_gp = ExperimentalGPSurrogate()
        self.is_fitted = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        # Step 1: fit LF GP
        self.lf_gp.fit(X_lf, y_lf)

        # Step 2: optimise rho via MLE on HF residuals
        X_lf_at_hf = X_hf[:, :len(LF_FEATURES)]
        mu_lf = self.lf_gp.predict(X_lf_at_hf)

        def neg_log_ml(log_rho):
            rho = np.exp(log_rho[0])
            residuals = y_hf - rho * mu_lf
            return float(np.sum(residuals ** 2))

        res = minimize(neg_log_ml, x0=[0.0], method="Nelder-Mead",
                       options={"maxiter": 500, "xatol": 1e-4})
        self.rho = float(np.exp(res.x[0]))
        print(f"  AR1 rho (LF→HF scaling) = {self.rho:.4f}")

        # Step 3: fit delta GP on residuals
        residuals = y_hf - self.rho * mu_lf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.delta_gp.fit(X_hf, residuals)

        self.is_fitted = True
        return self

    def predict(self, X_hf, return_std=False):
        X_lf = X_hf[:, :len(LF_FEATURES)]
        mu_lf, sig_lf = self.lf_gp.predict(X_lf, return_std=True)
        mu_d,  sig_d  = self.delta_gp.predict(X_hf, return_std=True)

        mu_hf  = self.rho * mu_lf + mu_d
        sig_hf = np.sqrt((self.rho * sig_lf) ** 2 + sig_d ** 2)

        return (mu_hf, sig_hf) if return_std else mu_hf

    def loo_cv(self, X_lf, y_lf, X_hf, y_hf):
        loo = LeaveOneOut()
        preds = np.zeros(len(y_hf))
        for tr, te in loo.split(X_hf):
            tmp = AR1MultiFidelityGP()
            tmp.fit(X_lf, y_lf, X_hf[tr], y_hf[tr])
            preds[te] = tmp.predict(X_hf[te])
        rmse = float(np.sqrt(mean_squared_error(y_hf, preds)))
        r2   = float(r2_score(y_hf, preds))
        print(f"  MF LOO RMSE = {rmse:.3f} µm    R² = {r2:.4f}")
        return {"rmse": rmse, "r2": r2}

    def save(self, path=MF_MODEL):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[✓] MF model saved → {path}")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_comparison(mf_model, exp_model, exp_df, out_dir=RESULTS):
    import matplotlib.pyplot as plt

    depths = np.linspace(60, 420, 120)
    X_sw   = _sweep_array("cut_depth_um", depths,
                           {f: REF[f] for f in FEATURE_COLS if f != "cut_depth_um"})

    mu_exp, sig_exp = exp_model.predict(X_sw, return_std=True)
    mu_mf,  sig_mf  = mf_model.predict(X_sw, return_std=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, mu_exp, color="#2166ac", lw=2, label="Standard GP (exp-only)")
    ax.fill_between(depths, mu_exp-2*sig_exp, mu_exp+2*sig_exp,
                    alpha=0.12, color="#2166ac")

    ax.plot(depths, mu_mf, color="#d62728", lw=2, ls="--",
            label=f"AR1 Multi-Fidelity GP (ρ={mf_model.rho:.2f})")
    ax.fill_between(depths, mu_mf-2*sig_mf, mu_mf+2*sig_mf,
                    alpha=0.10, color="#d62728")

    mask = ((exp_df["blade_W_um"]==23) & (exp_df["feed_mm_s"]==1.0) &
            (exp_df["spindle_rpm"]==30000))
    sub = exp_df[mask]
    ax.scatter(sub["cut_depth_um"], sub["chipping_um"],
               color="#2166ac", s=70, zorder=5, edgecolors="k", lw=0.6,
               label="Micro2026 (exp)")

    ax.axhline(15.0, color="gray", ls=":", lw=1.2, label="15 µm threshold")
    ax.set_xlabel("Cut Depth [µm]", fontsize=12)
    ax.set_ylabel("Front Chipping [µm]", fontsize=12)
    ax.set_title(f"AR1 Multi-Fidelity GP vs Standard GP\n"
                 f"(ρ={mf_model.rho:.3f}: FEM-to-experiment scaling)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)

    out = os.path.join(out_dir, "gp_multifidelity.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] MF comparison plot → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loo",  action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    # Load data
    exp_df = pd.DataFrame(CHIPPING_DATA)
    X_hf   = exp_df[FEATURE_COLS].values.astype(float)
    y_hf   = exp_df[TARGET_COL].values.astype(float)

    fem_df = pd.read_csv(FEM_CSV)
    fem_df = fem_df[fem_df["deletion_fraction"] > 0].copy()
    X_lf   = fem_df[LF_FEATURES].values.astype(float)
    y_lf   = fem_df["deletion_fraction"].values.astype(float)

    print(f"[*] HF data: {len(y_hf)} experimental pts")
    print(f"[*] LF data: {len(y_lf)} FEM pts")

    # Baseline: standard GP
    exp_model = ExperimentalGPSurrogate()
    if args.loo:
        print("\n[Standard GP LOO]")
        exp_model.loo_cv(X_hf, y_hf)
    exp_model.fit(X_hf, y_hf)

    # AR1 multi-fidelity
    mf_model = AR1MultiFidelityGP()
    if args.loo:
        print("\n[AR1 MF GP LOO]")
        mf_model.loo_cv(X_lf, y_lf, X_hf, y_hf)
    print("\n[*] Fitting AR1 MF GP ...")
    mf_model.fit(X_lf, y_lf, X_hf, y_hf)
    mf_model.save()

    if args.plot:
        plot_comparison(mf_model, exp_model, exp_df)

    # Demo predictions
    print("\n[*] Prediction comparison (blade_W=23, feed=1.0, spindle=30k):")
    print(f"  {'depth':>6}  {'std GP':>12}  {'AR1 MF GP':>12}")
    for d in [100, 200, 300, 390]:
        tc = np.array([[d, 23.0, 1.0, 30000.0]])
        me, se = exp_model.predict(tc, return_std=True)
        mm, sm = mf_model.predict(tc, return_std=True)
        print(f"  {d:>6}µm  {me[0]:>5.1f}±{se[0]:.1f}µm  {mm[0]:>5.1f}±{sm[0]:.1f}µm")


if __name__ == "__main__":
    main()
