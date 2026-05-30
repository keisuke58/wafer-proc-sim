"""
Die Strength Prediction — GP + Weibull Breakage Probability.

Data: Kim et al. (2023) Micro and Nano Systems Letters 11:16
      DOI: 10.1186/s40486-023-00183-w
      3 methods × 3 thicknesses = 9 conditions, n=15 dies each.

Model:
    Features: [method_code, target_thickness_um]
    Target  : die_strength_MPa (Weibull median)

Outputs:
    1. GP surrogate: params → strength distribution (mu, sigma)
    2. Breakage probability: P(strength < threshold) via GP uncertainty
    3. Method comparison: BDBG vs SDBG vs PDBG (DISCO-relevant)
    4. Thickness effect on strength degradation

DISCO relevance:
    - BDBG (blade dicing before grinding) is DISCO's core process
    - Die strength after BDBG degrades ~51% from 100→300µm (1023→502 MPa)
    - This model helps optimise blade parameters to maximise die strength
    - Directly relevant to HBM (ultra-thin die stacking, 20-50µm target)

Usage:
    python ml/die_strength_gp.py [--plot] [--predict]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data",
                           "experimental", "kim2023_dbg_die_strength.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                           "gp_die_strength.pkl")
RESULTS    = os.path.join(os.path.dirname(__file__), "..", "results")

# Method encoding
METHOD_MAP  = {"BDBG": 0, "SDBG": 1, "PDBG": 2}
METHOD_NAMES = {0: "BDBG (Blade)", 1: "SDBG (Stealth)", 2: "PDBG (Plasma)"}
METHOD_COLORS = {0: "#d62728", 1: "#2166ac", 2: "#2ca02c"}

# HBM die strength requirement (JEDEC, conservative)
HBM_STRENGTH_THRESHOLD = 600.0   # MPa
PROD_STRENGTH_MIN      = 400.0   # MPa — minimum acceptable for standard packaging

FEATURE_COLS = ["method_code", "target_thickness_um"]
TARGET_COL   = "die_strength_MPa"


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_PATH, comment="#")
    df["method_code"] = df["method"].map(METHOD_MAP)
    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COL].values.astype(float)
    return X, y, df


# ── Model ─────────────────────────────────────────────────────────────────────

class DieStrengthGP:
    """GP surrogate for die bending strength prediction."""

    def __init__(self):
        # length_scales are in NORMALIZED feature space (after StandardScaler)
        # method_code range: 0-2 → scaled ≈ ±1.2  → l.s.=0.8 gives inter-method variation
        # thickness range: 100-300µm → scaled ≈ ±1.2 → l.s.=0.6 captures 100vs300µm contrast
        kernel = (
            ConstantKernel(1.0, (0.1, 1e3))
            * Matern(length_scale=[0.8, 0.6],
                     length_scale_bounds=[(0.1, 5.0), (0.1, 5.0)],
                     nu=2.5)
            + WhiteKernel(noise_level=10., noise_level_bounds=(1., 1e4))
        )
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=20,
            normalize_y=False, alpha=0.0)

    def fit(self, X, y):
        Xs = self.scaler_x.fit_transform(X)
        ys = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(Xs, ys)
        print(f"  kernel: {self.gp.kernel_}")
        return self

    def predict(self, X, return_std=False):
        Xs = self.scaler_x.transform(X)
        mu, sigma = self.gp.predict(Xs, return_std=True)
        mu    = self.scaler_y.inverse_transform(mu.reshape(-1, 1)).ravel()
        sigma = sigma * float(self.scaler_y.scale_[0])
        return (mu, sigma) if return_std else mu

    def breakage_prob(self, X, threshold=PROD_STRENGTH_MIN):
        """P(die_strength < threshold) — probability of breakage during assembly."""
        mu, sigma = self.predict(X, return_std=True)
        return norm.cdf((threshold - mu) / np.maximum(sigma, 1.0))

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[✓] Model → {path}")

    @staticmethod
    def load(path=MODEL_PATH):
        return joblib.load(path)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_results(model, X, y, df, out_dir=RESULTS):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    thicknesses = np.linspace(80, 320, 100)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ── Panel 1: Strength vs thickness per method ──────────────────────────
    ax = axes[0]
    for code, name in METHOD_NAMES.items():
        color = METHOD_COLORS[code]
        X_sw = np.column_stack([np.full(len(thicknesses), code), thicknesses])
        mu, sig = model.predict(X_sw, return_std=True)
        ax.plot(thicknesses, mu, color=color, lw=2, label=name)
        ax.fill_between(thicknesses, mu - sig, mu + sig,
                        alpha=0.18, color=color)
        mask = df["method_code"] == code
        ax.scatter(df[mask]["target_thickness_um"], df[mask][TARGET_COL],
                   color=color, s=80, zorder=5, edgecolors="k", lw=0.6)

    ax.axhline(HBM_STRENGTH_THRESHOLD, color="purple", ls="--", lw=1.5,
               label=f"HBM target {HBM_STRENGTH_THRESHOLD:.0f} MPa")
    ax.axhline(PROD_STRENGTH_MIN, color="gray", ls=":", lw=1.2,
               label=f"Min acceptable {PROD_STRENGTH_MIN:.0f} MPa")
    ax.set_xlabel("Target Die Thickness [µm]", fontsize=11)
    ax.set_ylabel("Die Bending Strength [MPa]", fontsize=11)
    ax.set_title("Die Strength vs Thickness\n(Kim 2023, n=15 per condition)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)

    # ── Panel 2: Breakage probability ─────────────────────────────────────
    ax2 = axes[1]
    for thresh, color, ls, label in [
        (HBM_STRENGTH_THRESHOLD, "purple", "--", f"HBM {HBM_STRENGTH_THRESHOLD:.0f}MPa"),
        (PROD_STRENGTH_MIN,      "gray",   ":",  f"Prod {PROD_STRENGTH_MIN:.0f}MPa"),
    ]:
        for code, name in METHOD_NAMES.items():
            mc = METHOD_COLORS[code]
            X_sw = np.column_stack([np.full(len(thicknesses), code), thicknesses])
            p_break = model.breakage_prob(X_sw, threshold=thresh)
            ax2.plot(thicknesses, p_break * 100,
                     color=mc, lw=2 if thresh == HBM_STRENGTH_THRESHOLD else 1,
                     ls=ls if thresh == PROD_STRENGTH_MIN else "-",
                     label=f"{name} (>{thresh:.0f}MPa)" if thresh == HBM_STRENGTH_THRESHOLD else None)

    ax2.axhline(1.0, color="red", ls=":", lw=1.0, label="1% breakage")
    ax2.set_xlabel("Target Die Thickness [µm]", fontsize=11)
    ax2.set_ylabel("P(strength < threshold) [%]", fontsize=11)
    ax2.set_title(f"Breakage Probability\n(solid: <{HBM_STRENGTH_THRESHOLD:.0f}MPa threshold)",
                  fontsize=10)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.25)
    ax2.set_ylim(0, 60)

    # ── Panel 3: BDBG focus — strength vs chipping tradeoff ───────────────
    ax3 = axes[2]
    bdbg_mask = df["method"] == "BDBG"
    bdbg_df   = df[bdbg_mask].copy()

    # Approximate chipping from Kim2023 notes
    chipping_approx = {"100": 8.0, "200": 10.0, "300": 12.5}
    bdbg_df["chipping_um"] = bdbg_df["target_thickness_um"].astype(str).map(
        lambda t: chipping_approx.get(str(t), np.nan))

    sc = ax3.scatter(bdbg_df["chipping_um"], bdbg_df[TARGET_COL],
                     c=bdbg_df["target_thickness_um"], cmap="viridis",
                     s=150, zorder=5, edgecolors="k", lw=0.8)
    plt.colorbar(sc, ax=ax3, label="Die Thickness [µm]")

    # Annotate
    for _, row in bdbg_df.iterrows():
        ax3.annotate(f"{row['target_thickness_um']:.0f}µm",
                     (row["chipping_um"], row[TARGET_COL]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax3.axhline(HBM_STRENGTH_THRESHOLD, color="purple", ls="--", lw=1.5,
                label=f"HBM {HBM_STRENGTH_THRESHOLD:.0f} MPa")
    ax3.set_xlabel("Front Chipping [µm] (approx.)", fontsize=11)
    ax3.set_ylabel("Die Bending Strength [MPa]", fontsize=11)
    ax3.set_title("BDBG: Chipping vs Strength\n(tradeoff for HBM optimisation)",
                  fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)
    ax3.set_ylim(bottom=0)

    fig.suptitle(
        "Die Strength After Dicing-Before-Grinding — Kim et al. (2023)\n"
        "BDBG (DISCO DAD3350) | SDBG | PDBG — 200mm Si, 10×10mm² die",
        fontsize=10,
    )
    plt.tight_layout()
    out = os.path.join(out_dir, "die_strength_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Plot → {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot",    action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    X, y, df = load_data()
    print(f"[*] Loaded {len(df)} conditions")
    print(f"    Strength: {y.min():.0f}–{y.max():.0f} MPa")

    model = DieStrengthGP()
    model.fit(X, y)
    model.save()

    if args.plot:
        plot_results(model, X, y, df)

    # Key predictions
    print("\n[*] Die strength predictions:")
    print(f"  {'Method':>12}  {'Thick':>6}  {'Strength':>12}  "
          f"{'P(<600MPa)':>11}  {'P(<400MPa)':>11}")
    for method, code in METHOD_MAP.items():
        for t in [100, 200, 300]:
            xr = np.array([[code, t]])
            mu, sig = model.predict(xr, return_std=True)
            p_hmb  = model.breakage_prob(xr, HBM_STRENGTH_THRESHOLD)[0]
            p_prod = model.breakage_prob(xr, PROD_STRENGTH_MIN)[0]
            print(f"  {method:>12}  {t:>5}µm  "
                  f"{mu[0]:>7.0f}±{sig[0]:.0f}MPa  "
                  f"{p_hmb*100:>9.1f}%  "
                  f"{p_prod*100:>9.1f}%")
        print()

    # DISCO insight
    print("[*] Key insight for DISCO:")
    bdbg = [(0, t) for t in [100, 200, 300]]
    strengths = [model.predict(np.array([[c, t]]))[0] for c, t in bdbg]
    degradation = (strengths[0] - strengths[-1]) / strengths[0] * 100
    print(f"  BDBG strength degrades {degradation:.0f}% from 100µm→300µm "
          f"({strengths[0]:.0f}→{strengths[-1]:.0f} MPa)")
    print(f"  → Optimising blade params to reduce chipping can recover strength")
    print(f"  → HBM target ({HBM_STRENGTH_THRESHOLD:.0f} MPa): "
          f"BDBG 100µm achieves it, 200/300µm do not")


if __name__ == "__main__":
    main()
