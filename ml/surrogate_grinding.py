"""
GP surrogate model for grinding parameter optimization.

Input  : parametric_summary_grinding.csv
           columns: wheel_speed_rpm, feed_rate_um_min, cut_depth_um,
                    warpage_um, max_residual_stress_MPa
Outputs: warpage_um, max_residual_stress_MPa

Usage:
    python surrogate_grinding.py --csv results/parametric_summary_grinding.csv
    python surrogate_grinding.py --csv results/parametric_summary_grinding.csv --plot
    python surrogate_grinding.py --csv results/parametric_summary_grinding.csv --loo
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                          "gp_grinding_model.pkl")

FEATURE_COLS = ["wheel_speed_rpm", "feed_rate_um_min", "cut_depth_um"]
TARGET_COLS  = ["warpage_um", "max_residual_stress_MPa"]


def build_kernel():
    """Anisotropic RBF + constant amplitude + noise."""
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=[500.0, 100.0, 10.0],
              length_scale_bounds=[(10.0, 5000.0),
                                   (10.0, 1000.0),
                                   (1.0,  200.0)])
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )


class GrindingGPSurrogate:
    """Multi-output GP surrogate for grinding (one independent GP per target)."""

    def __init__(self):
        self.scalers_x = StandardScaler()
        self.scalers_y = [StandardScaler() for _ in TARGET_COLS]
        self.gps = [
            GaussianProcessRegressor(
                kernel=build_kernel(),
                n_restarts_optimizer=10,
                normalize_y=False,
                alpha=0.0)
            for _ in TARGET_COLS]
        self.is_fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """X: (N, 3), Y: (N, 2)."""
        Xs = self.scalers_x.fit_transform(X)
        for i, (gp, scaler) in enumerate(zip(self.gps, self.scalers_y)):
            ys = scaler.fit_transform(Y[:, i:i+1]).ravel()
            gp.fit(Xs, ys)
            print(f"  [{TARGET_COLS[i]}] kernel: {gp.kernel_}")
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, return_std=False):
        """Returns (N, 2) mean; optionally (N, 2) std."""
        Xs = self.scalers_x.transform(X)
        means, stds = [], []
        for gp, scaler in zip(self.gps, self.scalers_y):
            mu, sigma = gp.predict(Xs, return_std=True)
            mu    = scaler.inverse_transform(mu.reshape(-1, 1)).ravel()
            sigma = sigma * scaler.scale_[0]
            means.append(mu)
            stds.append(sigma)
        if return_std:
            return np.column_stack(means), np.column_stack(stds)
        return np.column_stack(means)

    def loo_cv(self, X: np.ndarray, Y: np.ndarray) -> dict:
        loo   = LeaveOneOut()
        preds = np.zeros_like(Y)
        for train_idx, test_idx in loo.split(X):
            tmp = GrindingGPSurrogate()
            tmp.fit(X[train_idx], Y[train_idx])
            preds[test_idx] = tmp.predict(X[test_idx])
        metrics = {}
        for i, col in enumerate(TARGET_COLS):
            rmse = np.sqrt(mean_squared_error(Y[:, i], preds[:, i]))
            r2   = r2_score(Y[:, i], preds[:, i])
            metrics[col] = {"rmse": rmse, "r2": r2}
            print(f"  LOO [{col}]  RMSE={rmse:.4g}  R²={r2:.4f}")
        return metrics

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[✓] Model saved → {path}")

    @staticmethod
    def load(path=MODEL_PATH) -> "GrindingGPSurrogate":
        model = joblib.load(path)
        print(f"[✓] Model loaded ← {path}")
        return model


def plot_response_surface(model: GrindingGPSurrogate,
                           X_train: np.ndarray,
                           Y_train: np.ndarray,
                           out_dir: str = "results"):
    """Slice plots: warpage and stress vs (wheel_speed, feed_rate) at fixed cut_depth."""
    import matplotlib.pyplot as plt

    cd_mid = X_train[:, 2].mean()

    ws_grid = np.linspace(2000, 8000, 50)
    fr_grid = np.linspace(200,  800,  50)
    WS, FR  = np.meshgrid(ws_grid, fr_grid)
    X_grid  = np.column_stack([WS.ravel(), FR.ravel(),
                                np.full(WS.size, cd_mid)])
    Y_pred, Y_std = model.predict(X_grid, return_std=True)

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    specs = [
        ("Mean Warpage [µm]",      Y_pred[:, 0], "viridis"),
        ("Std Warpage [µm]",       Y_std[:, 0],  "plasma"),
        ("Mean Max Stress [MPa]",  Y_pred[:, 1], "viridis"),
        ("Std Max Stress [MPa]",   Y_std[:, 1],  "plasma"),
    ]
    for ax, (title, data, cmap) in zip(axes.ravel(), specs):
        im = ax.contourf(WS, FR, data.reshape(WS.shape), levels=20, cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.scatter(X_train[:, 0], X_train[:, 1],
                   c="red", s=60, zorder=5, label="FEM data")
        ax.set_xlabel("Wheel Speed [rpm]")
        ax.set_ylabel("Feed Rate [µm/min]")
        ax.set_title(f"{title}  (cut_depth={cd_mid:.0f} µm)")
        ax.legend(fontsize=8)

    plt.suptitle("Grinding GP Surrogate — Response Surface", fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "gp_grinding_surface.png")
    plt.savefig(fig_path, dpi=150)
    print(f"[✓] Plot → {fig_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--loo",  action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[*] Loaded {len(df)} samples from {args.csv}")
    print(df[FEATURE_COLS + TARGET_COLS].describe())

    X = df[FEATURE_COLS].values.astype(float)
    Y = df[TARGET_COLS].values.astype(float)

    model = GrindingGPSurrogate()

    if args.loo:
        print("\n[*] Running LOO cross-validation...")
        model.loo_cv(X, Y)

    print("\n[*] Fitting on full dataset...")
    model.fit(X, Y)
    model.save()

    if args.plot:
        out_dir = os.path.dirname(args.csv)
        plot_response_surface(model, X, Y, out_dir=out_dir)

    # Demo predictions
    test_cases = np.array([
        [4000, 300, 20],
        [6000, 500, 30],
        [2000, 200, 10],
    ])
    mu, sigma = model.predict(test_cases, return_std=True)
    print("\n[*] Prediction demo:")
    for tc, m, s in zip(test_cases, mu, sigma):
        print(f"  ws={tc[0]}rpm  f={tc[1]}µm/min  d={tc[2]}µm  "
              f"→  warp={m[0]:.2f}±{s[0]:.2f} µm  "
              f"stress={m[1]:.1f}±{s[1]:.1f} MPa")


if __name__ == "__main__":
    main()
