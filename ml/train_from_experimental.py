"""
Train GP surrogate on digitized experimental chipping data (Micro2026 + Mat2022).

Features: cut_depth_um, blade_W_um, feed_mm_s, spindle_rpm
Target  : chipping_um

Usage:
    python ml/train_from_experimental.py           # train + save
    python ml/train_from_experimental.py --loo     # LOO cross-validation
    python ml/train_from_experimental.py --plot    # save sweep plots
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from validation.experimental_data import CHIPPING_DATA

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "gp_experimental.pkl"
)

FEATURE_COLS = ["cut_depth_um", "blade_W_um", "feed_mm_s", "spindle_rpm"]
TARGET_COL = "chipping_um"

# Micro2026 reference conditions (depth=390µm, bw=23µm, feed=1mm/s, spindle=30krpm)
REF = {"cut_depth_um": 390.0, "blade_W_um": 23.0, "feed_mm_s": 1.0, "spindle_rpm": 30000.0}


# ── Kernel ────────────────────────────────────────────────────────────────────

def _build_kernel():
    # Per-feature length scales: depth[µm], bw[µm], feed[mm/s], spindle[rpm]
    # Bounds widened past observed optima to avoid boundary convergence warnings.
    return ConstantKernel(10.0, (0.1, 1e3)) * RBF(
        length_scale=[50.0, 50.0, 2.0, 10000.0],
        length_scale_bounds=[(5, 500), (5, 500), (0.1, 20), (500, 50000)],
    ) + WhiteKernel(noise_level=0.5, noise_level_bounds=(0.05, 30.0))


# ── Model class ───────────────────────────────────────────────────────────────

class ExperimentalGPSurrogate:
    """GP surrogate trained on experimental chipping data."""

    def __init__(self):
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.gp = GaussianProcessRegressor(
            kernel=_build_kernel(),
            n_restarts_optimizer=15,
            normalize_y=False,
            alpha=0.0,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExperimentalGPSurrogate":
        Xs = self.scaler_x.fit_transform(X)
        ys = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        self.gp.fit(Xs, ys)
        self.is_fitted = True
        print(f"  kernel: {self.gp.kernel_}")
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        Xs = self.scaler_x.transform(X)
        mu, sigma = self.gp.predict(Xs, return_std=True)
        mu = self.scaler_y.inverse_transform(mu.reshape(-1, 1)).ravel()
        sigma = sigma * float(self.scaler_y.scale_[0])
        return (mu, sigma) if return_std else mu

    def loo_cv(self, X: np.ndarray, y: np.ndarray) -> dict:
        loo = LeaveOneOut()
        preds = np.zeros_like(y)
        for train_idx, test_idx in loo.split(X):
            tmp = ExperimentalGPSurrogate()
            tmp.fit(X[train_idx], y[train_idx])
            preds[test_idx] = tmp.predict(X[test_idx])
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        r2 = float(r2_score(y, preds))
        print(f"  LOO RMSE = {rmse:.2f} µm    R² = {r2:.4f}")
        return {"rmse": rmse, "r2": r2, "loo_preds": preds}

    def save(self, path: str = MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[✓] Saved → {path}")

    @staticmethod
    def load(path: str = MODEL_PATH) -> "ExperimentalGPSurrogate":
        model = joblib.load(path)
        print(f"[✓] Loaded ← {path}")
        return model


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(quality_filter: str = "all"):
    """
    quality_filter: "all" | "AB" | "A"
        "all": all grades (N≈25)
        "AB" : directly measured + digitized (N≈11)
        "A"  : directly measured with reported std (N≈6)
    """
    from validation.experimental_data import point_noise
    ok_grades = {"all": set("ABCD"), "AB": set("AB"), "A": {"A"}}
    allowed = ok_grades.get(quality_filter, set("ABCD"))

    rows, alpha_list = [], []
    for entry in CHIPPING_DATA:
        if entry.get("cut_type") == "complete":
            continue
        if entry.get("quality", "B") not in allowed:
            continue
        rows.append(entry)
        alpha_list.append(point_noise(entry) ** 2)

    df = pd.DataFrame(rows)
    X  = df[FEATURE_COLS].values.astype(float)
    y  = df[TARGET_COL].values.astype(float)
    return X, y, df, np.array(alpha_list)


# ── Sweep helpers ─────────────────────────────────────────────────────────────

def _sweep_array(vary_feat: str, x_vals: np.ndarray, fixed: dict) -> np.ndarray:
    X = np.zeros((len(x_vals), len(FEATURE_COLS)))
    for i, f in enumerate(FEATURE_COLS):
        X[:, i] = x_vals if f == vary_feat else fixed[f]
    return X


def plot_sweeps(
    model: ExperimentalGPSurrogate,
    df_train: pd.DataFrame,
    out_dir: str = "results",
) -> str:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    sweeps = [
        ("cut_depth_um",  np.linspace(60, 420, 120),   "Cut Depth [µm]"),
        ("feed_mm_s",     np.linspace(0.3, 3.2, 100),  "Feed Speed [mm/s]"),
        ("spindle_rpm",   np.linspace(18000, 42000, 100), "Spindle Speed [rpm]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"Micro2026": "#2166ac", "Mat2022": "#d6604d"}

    for ax, (vary_feat, x_vals, xlabel) in zip(axes, sweeps):
        # Fix others at Micro2026 reference
        fixed = {f: REF[f] for f in FEATURE_COLS if f != vary_feat}
        X_sweep = _sweep_array(vary_feat, x_vals, fixed)
        mu, sigma = model.predict(X_sweep, return_std=True)

        ax.plot(x_vals, mu, color="#1a1a2e", lw=2, label="GP mean")
        ax.fill_between(
            x_vals, mu - 2 * sigma, mu + 2 * sigma,
            alpha=0.18, color="#1a1a2e", label="±2σ",
        )

        for src, grp in df_train.groupby("source"):
            ax.scatter(
                grp[vary_feat], grp[TARGET_COL],
                color=colors.get(src, "gray"), s=60, zorder=5,
                edgecolors="k", linewidths=0.6, label=src,
            )

        ax.axhline(15.0, color="#d73027", ls="--", lw=1.2, label="Threshold 15 µm")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Front Chipping [µm]", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "GP Surrogate — Experimental Data (Micro2026 + Mat2022)\n"
        "Fixed: blade_W=23 µm, others at Micro2026 reference",
        fontsize=11,
    )
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "gp_experimental_sweeps.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Sweep plot → {fig_path}")
    plt.close()
    return fig_path


def plot_heatmap(
    model: ExperimentalGPSurrogate,
    out_dir: str = "results",
) -> str:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    depths = np.linspace(60, 420, 80)
    feeds = np.linspace(0.3, 3.2, 80)
    D, F = np.meshgrid(depths, feeds)
    # Fix blade_W=23µm, spindle=30000rpm
    X_grid = np.column_stack([
        D.ravel(),
        np.full(D.size, 23.0),
        F.ravel(),
        np.full(D.size, 30000.0),
    ])
    mu, sigma = model.predict(X_grid, return_std=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, data, title, cmap in [
        (axes[0], mu,    "GP Mean — Front Chipping [µm]", "YlOrRd"),
        (axes[1], sigma, "GP Std [µm]",                   "viridis"),
    ]:
        im = ax.contourf(D, F, data.reshape(D.shape), levels=20, cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.contour(D, F, mu.reshape(D.shape), levels=[15.0],
                   colors="white", linewidths=1.5, linestyles="--")
        ax.set_xlabel("Cut Depth [µm]", fontsize=11)
        ax.set_ylabel("Feed Speed [mm/s]", fontsize=11)
        ax.set_title(title, fontsize=11)

    axes[0].text(65, 3.0, "← 15 µm threshold", color="white",
                 fontsize=9, va="top")
    fig.suptitle("4H-SiC Blade Dicing — Chipping Response Surface\n"
                 "(blade_W=23 µm, spindle=30 krpm, GP trained on Micro2026)", fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "gp_experimental_heatmap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Heatmap → {fig_path}")
    plt.close()
    return fig_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loo",     action="store_true", help="LOO cross-validation")
    parser.add_argument("--plot",    action="store_true", help="Save sweep + heatmap plots")
    parser.add_argument("--quality", default="all", choices=["all", "AB", "A"],
                        help="Data quality filter (default: all)")
    args = parser.parse_args()

    X, y, df, alpha_vec = load_data(args.quality)
    print(f"[*] Loaded {len(df)} data points  (quality={args.quality})")
    print(f"    Sources  : {df['source'].unique().tolist()}")
    print(f"    Features : {FEATURE_COLS}")
    print(f"    Chipping : {y.min():.1f} – {y.max():.1f} µm")
    if "quality" in df.columns:
        from collections import Counter
        print(f"    Grades   : {dict(Counter(df['quality']))}")

    model = ExperimentalGPSurrogate()

    if args.loo:
        print("\n[*] LOO cross-validation (with per-point quality noise) …")
        # Temporarily pass alpha_vec to gp.alpha for LOO
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import mean_squared_error, r2_score
        loo   = LeaveOneOut()
        preds = np.zeros_like(y)
        scaler_y_tmp = model.scaler_y
        for tr, te in loo.split(X):
            tmp = ExperimentalGPSurrogate()
            Xs = tmp.scaler_x.fit_transform(X[tr])
            ys = tmp.scaler_y.fit_transform(y[tr].reshape(-1,1)).ravel()
            tmp.gp.alpha = alpha_vec[tr] / tmp.scaler_y.scale_[0]**2
            tmp.gp.fit(Xs, ys)
            Xte = tmp.scaler_x.transform(X[te])
            mu, _ = tmp.gp.predict(Xte, return_std=True)
            preds[te] = tmp.scaler_y.inverse_transform(mu.reshape(-1,1)).ravel()
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        r2   = float(r2_score(y, preds))
        print(f"  LOO RMSE = {rmse:.2f} µm    R² = {r2:.4f}")

    print("\n[*] Fitting on full dataset …")
    model.fit(X, y)

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    model.save(MODEL_PATH)

    if args.plot:
        plot_sweeps(model, df, out_dir=results_dir)
        plot_heatmap(model, out_dir=results_dir)

    # Spot predictions
    demos = [
        [390, 23, 1.0, 30000],   # Micro2026 reference
        [390, 23, 2.5, 30000],   # high feed
        [200, 48, 5.0, 22000],   # Mat2022 condition
        [100, 23, 1.0, 38000],   # shallow + fast spindle
    ]
    print("\n[*] Prediction demo:")
    for tc in demos:
        mu, s = model.predict(np.array([tc]), return_std=True)
        print(
            f"  depth={tc[0]:>3}µm  bw={tc[1]}µm  feed={tc[2]}mm/s  "
            f"rpm={tc[3]:.0f}  →  chipping = {mu[0]:.1f} ± {s[0]:.1f} µm"
        )


if __name__ == "__main__":
    main()
