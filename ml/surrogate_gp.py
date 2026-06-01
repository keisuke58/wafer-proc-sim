"""
GP surrogate model for SiC dicing parameter optimization.

Input  : parametric_summary.csv  (cut_depth_um, blade_W_um, ...)
Outputs: deletion_fraction (chipping proxy), max_principal_stress_Pa

Usage:
    python surrogate_gp.py --csv results/parametric_summary.csv
    python surrogate_gp.py --csv results/parametric_summary.csv --plot
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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "gp_model.pkl")

# ── Feature / target columns ──────────────────────────────────────────────────
FEATURE_COLS = ["cut_depth_um", "blade_W_um"]
# max_principal_stress_Pa は FEM結果CSVでは max_RF2_N (推力) に対応。
# どちらか存在する方を使う（後方互換のため両方定義）
TARGET_COLS  = ["deletion_fraction", "max_principal_stress_Pa"]
TARGET_COLS_ALT = ["deletion_fraction", "max_RF2_N"]  # parametric_summary.csv の実列名


def resolve_target_cols(df) -> list[str]:
    """CSVに合わせて TARGET_COLS を動的解決する"""
    if all(c in df.columns for c in TARGET_COLS):
        return TARGET_COLS
    if all(c in df.columns for c in TARGET_COLS_ALT):
        return TARGET_COLS_ALT
    num_cols = df.select_dtypes(include="number").columns.tolist()
    available = [c for c in num_cols if c not in FEATURE_COLS][:2]
    if len(available) < 2:
        raise ValueError(
            f"Cannot resolve 2 target columns from CSV.  "
            f"Expected one of {TARGET_COLS} or {TARGET_COLS_ALT}.  "
            f"Found numeric columns: {num_cols}"
        )
    return available


def build_kernel():
    """Anisotropic RBF + constant amplitude + noise."""
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=[10.0, 10.0], length_scale_bounds=[(1.0, 200.0)] * 2)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )


class DicingGPSurrogate:
    """Multi-output GP surrogate (one independent GP per output)."""

    def __init__(self, target_cols=None):
        self.target_cols = target_cols or list(TARGET_COLS)
        n = len(self.target_cols)
        self.scalers_x = StandardScaler()
        self.scalers_y = [StandardScaler() for _ in range(n)]
        self.gps = [
            GaussianProcessRegressor(
                kernel=build_kernel(),
                n_restarts_optimizer=10,
                normalize_y=False,
                alpha=1e-10)   # small jitter for numerical stability
            for _ in range(n)]
        self.is_fitted = False

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """X: (N, n_features), Y: (N, n_targets)."""
        Xs = self.scalers_x.fit_transform(X)
        for i, (gp, scaler, col) in enumerate(zip(self.gps, self.scalers_y, self.target_cols)):
            ys = scaler.fit_transform(Y[:, i:i+1]).ravel()
            gp.fit(Xs, ys)
            print(f"  [{col}] kernel: {gp.kernel_}")
        self.is_fitted = True
        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray, return_std=False):
        """Returns (N, n_targets) mean; optionally (N, n_targets) std."""
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

    # ── LOO cross-validation ──────────────────────────────────────────────────
    def loo_cv(self, X: np.ndarray, Y: np.ndarray) -> dict:
        loo = LeaveOneOut()
        preds = np.zeros_like(Y)
        for train_idx, test_idx in loo.split(X):
            tmp = DicingGPSurrogate(self.target_cols)
            tmp.fit(X[train_idx], Y[train_idx])
            preds[test_idx] = tmp.predict(X[test_idx])

        metrics = {}
        for i, col in enumerate(self.target_cols):
            rmse = np.sqrt(mean_squared_error(Y[:, i], preds[:, i]))
            r2   = r2_score(Y[:, i], preds[:, i])
            metrics[col] = {"rmse": rmse, "r2": r2}
            print(f"  LOO [{col}]  RMSE={rmse:.4g}  R²={r2:.4f}")
        return metrics

    # ── Save / load ───────────────────────────────────────────────────────────
    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[✓] Model saved → {path}")

    @staticmethod
    def load(path=MODEL_PATH) -> "DicingGPSurrogate":
        model = joblib.load(path)
        print(f"[✓] Model loaded ← {path}")
        return model


# ── Plot response surface ─────────────────────────────────────────────────────
def plot_response_surface(model: DicingGPSurrogate,
                           X_train: np.ndarray,
                           Y_train: np.ndarray,
                           out_dir: str = "results"):
    import matplotlib.pyplot as plt

    d_grid  = np.linspace(15, 65, 60)   # cut depth µm
    bw_grid = np.linspace(15, 45, 60)   # blade width µm
    D, BW   = np.meshgrid(d_grid, bw_grid)
    X_grid  = np.column_stack([D.ravel(), BW.ravel()])
    Y_pred, Y_std = model.predict(X_grid, return_std=True)

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = [("Mean Chipping Fraction", "deletion_fraction", 0),
              ("Std Chipping Fraction",  "deletion_fraction_std", 0),
              ("Mean Max Stress [GPa]",  "max_principal_stress", 1),
              ("Std Max Stress [GPa]",   "max_principal_stress_std", 1)]

    for ax, (title, _, idx) in zip(axes.ravel(), titles):
        is_std = "std" in title.lower()
        data = Y_std[:, idx] if is_std else Y_pred[:, idx]
        if "Stress" in title:
            data = data / 1e9
        im = ax.contourf(D, BW, data.reshape(D.shape), levels=20, cmap="viridis")
        plt.colorbar(im, ax=ax)
        # Overlay training points
        ax.scatter(X_train[:, 0], X_train[:, 1],
                   c="red", s=60, zorder=5, label="FEM data")
        ax.set_xlabel("Cut Depth [µm]")
        ax.set_ylabel("Blade Width [µm]")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "gp_response_surface.png")
    plt.savefig(fig_path, dpi=150)
    print(f"[✓] Plot saved → {fig_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  required=True, help="parametric_summary.csv path")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--loo",  action="store_true", help="Run LOO cross-validation")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[*] Loaded {len(df)} samples from {args.csv}")
    active_targets = resolve_target_cols(df)
    print(f"[*] Using targets: {active_targets}")
    print(df[FEATURE_COLS + active_targets].describe())

    X = df[FEATURE_COLS].values.astype(float)
    Y = df[active_targets].values.astype(float)

    # Scale second target to GPa-range for numerical stability.
    # max_principal_stress_Pa is in Pa → ÷1e9; max_RF2_N is in N → no scaling needed.
    Y_fit = Y.copy()
    if active_targets[1] == "max_principal_stress_Pa":
        Y_fit[:, 1] = Y[:, 1] / 1e9

    model = DicingGPSurrogate(active_targets)

    if args.loo:
        print("\n[*] Running LOO cross-validation...")
        model.loo_cv(X, Y_fit)

    print("\n[*] Fitting on full dataset...")
    model.fit(X, Y_fit)
    model.save()

    if args.plot:
        plot_response_surface(model, X, Y_fit,
                               out_dir=os.path.join(os.path.dirname(args.csv)))

    # Quick prediction demo
    test_cases = np.array([[30, 25], [50, 30], [60, 40]])
    mu, sigma = model.predict(test_cases, return_std=True)
    print("\n[*] Prediction demo:")
    for i, (tc, m, s) in enumerate(zip(test_cases, mu, sigma)):
        print(f"  depth={tc[0]}µm, kerf={tc[1]}µm → "
              f"chip={m[0]:.4f}±{s[0]:.4f}, "
              f"stress={m[1]:.3f}±{s[1]:.3f} GPa")


if __name__ == "__main__":
    main()
