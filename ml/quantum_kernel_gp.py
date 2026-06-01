"""
Quantum Kernel GP — SiC Dicing Process Surrogate
=================================================
ZZFeatureMap (PennyLane) fidelity kernel + sklearn GPR.
Layer 2 of the Quantum Stack: surrogate model for chipping prediction.

Quantum kernel k(x, x') = |<0|U†(x')U(x)|0>|²
  U(x) = ZZFeatureMap: AngleEmbedding(Y) × n_layers + CNOT ring per layer

Connection to Muramatsu Lab (2026):
    "Data Assimilation Based on the Ensemble Kalman Filter
     for Dislocation Motion" → EKF feeds state estimates into this
     surrogate for closed-loop recipe correction.

Usage:
    python ml/quantum_kernel_gp.py          # full LOO + plots
    python ml/quantum_kernel_gp.py --quick  # fit only, skip LOO
    python ml/quantum_kernel_gp.py --kernel-only  # show kernel matrix only
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel, WhiteKernel, Kernel,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pennylane as qml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from validation.experimental_data import CHIPPING_DATA

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

FEATURE_COLS = ["cut_depth_um", "blade_W_um", "feed_mm_s", "spindle_rpm"]
TARGET_COL   = "chipping_um"
N_QUBITS     = 4   # one qubit per feature


# ════════════════════════════════════════════════════════════════════════════
# ZZFeatureMap Quantum Kernel
# ════════════════════════════════════════════════════════════════════════════

class ZZFeatureMapKernel(Kernel):
    """
    Fidelity kernel via ZZFeatureMap circuit.

    Circuit depth = n_layers × (AngleEmbedding + CNOT ring).
    k(x, x') = Prob(|0...0⟩ after U(x) ∘ U†(x')) on default.qubit.

    No trainable hyperparameters — kernel is fully determined by
    the feature map structure (compatible with NISQ devices).
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self._build()

    def _build(self):
        dev = qml.device("default.qubit", wires=self.n_qubits)
        n_q = self.n_qubits
        n_l = self.n_layers

        def _embed(x):
            for _ in range(n_l):
                qml.AngleEmbedding(x, wires=range(n_q), rotation="Y")
                for i in range(n_q - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_q - 1, 0])  # ring closure

        @qml.qnode(dev)
        def _circuit(x1, x2):
            _embed(x1)
            qml.adjoint(_embed)(x2)
            return qml.probs(wires=range(n_q))

        self._circuit = _circuit

    def _k(self, x1, x2) -> float:
        return float(self._circuit(np.asarray(x1), np.asarray(x2))[0])

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X)
        Y = X if Y is None else np.asarray(Y)
        K = np.array([[self._k(x, y) for y in Y] for x in X])
        if eval_gradient:
            return K, np.empty((len(X), len(X), 0))
        return K

    def diag(self, X):
        return np.array([self._k(x, x) for x in X])

    def is_stationary(self) -> bool:
        return False

    @property
    def theta(self):
        return np.empty(0)

    @theta.setter
    def theta(self, v):
        pass

    @property
    def bounds(self):
        return np.empty((0, 2))

    def get_params(self, deep=True):
        return {"n_qubits": self.n_qubits, "n_layers": self.n_layers}

    def clone_with_theta(self, theta):
        return ZZFeatureMapKernel(self.n_qubits, self.n_layers)

    def __repr__(self):
        return f"ZZFeatureMap(n_qubits={self.n_qubits}, n_layers={self.n_layers})"


# ════════════════════════════════════════════════════════════════════════════
# Quantum Kernel GP Surrogate
# ════════════════════════════════════════════════════════════════════════════

class QuantumKernelGP:
    """
    GP surrogate using ZZFeatureMap quantum kernel.

    Input features are MinMax-scaled to [0, π] before passing to the
    quantum circuit (angle embedding range).
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 2,
                 alpha: float = 1.0):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.alpha    = alpha
        self.scaler_x = MinMaxScaler(feature_range=(0.0, np.pi))
        self.scaler_y = StandardScaler()
        self.kernel   = ZZFeatureMapKernel(n_qubits, n_layers)
        self.gp       = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=alpha,
            optimizer=None,     # no hyperparameter optimization
            normalize_y=False,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            alpha_vec=None) -> "QuantumKernelGP":
        """
        alpha_vec: per-point noise variance σ_i² (shape N,).
        If provided, overrides the scalar self.alpha for heteroscedastic noise.
        """
        Xs = self.scaler_x.fit_transform(X)
        ys = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        if alpha_vec is not None:
            scale = float(self.scaler_y.scale_[0])
            self.gp.alpha = alpha_vec / (scale ** 2)   # rescale to normalised space
        t0 = time.time()
        print(f"  Computing {len(X)}×{len(X)} quantum kernel matrix …")
        self.gp.fit(Xs, ys)
        print(f"  Fit done in {time.time()-t0:.1f}s")
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        assert self.is_fitted
        Xs = self.scaler_x.transform(X)
        mu, sigma = self.gp.predict(Xs, return_std=True)
        mu    = self.scaler_y.inverse_transform(mu.reshape(-1, 1)).ravel()
        sigma = sigma * float(self.scaler_y.scale_[0])
        return (mu, sigma) if return_std else mu


# ════════════════════════════════════════════════════════════════════════════
# Classical GP baseline (for comparison)
# ════════════════════════════════════════════════════════════════════════════

def build_classical_gp(alpha: float = 1.0) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(10.0, (0.1, 1e3))
        * RBF(length_scale=[50, 5, 2, 1e4],
              length_scale_bounds=[(5, 500), (1, 100), (0.1, 20), (500, 5e4)])
        + WhiteKernel(noise_level=alpha, noise_level_bounds=(0.05, 30.0))
    )
    return GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10, normalize_y=True
    )


# ════════════════════════════════════════════════════════════════════════════
# Quality-aware data loading
# ════════════════════════════════════════════════════════════════════════════

def load_data(quality_filter: str = "all",
              cut_type: str = "incomplete"):
    """
    Load experimental chipping data with optional quality filtering.

    quality_filter:
        "all"    — all data points (original behaviour)
        "AB"     — only grade A (direct measurement) + B (digitized)
        "A"      — only directly measured with reported std
    cut_type:
        "incomplete" — only partial-penetration cuts (default, consistent regime)
        "all"        — include complete cuts (different fracture mode)

    Returns X, y, alpha_vec where alpha_vec[i] = σ_i² (per-point noise variance).
    alpha_vec can be passed directly as `alpha` to GaussianProcessRegressor.
    """
    import pandas as pd
    from validation.experimental_data import CHIPPING_DATA, point_noise

    df = pd.DataFrame(CHIPPING_DATA)
    df = df[df["material"].isin(["4H-SiC", "SiC"])]

    if cut_type == "incomplete":
        df = df[df["cut_type"] == "incomplete"]

    if quality_filter == "AB":
        df = df[df["quality"].isin(["A", "B"])]
    elif quality_filter == "A":
        df = df[df["quality"] == "A"]

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COL].values.astype(float)

    raw = df.to_dict("records")
    sigma = np.array([point_noise(r) for r in raw])
    alpha_vec = sigma ** 2   # variance for GPR alpha parameter

    return X, y, alpha_vec


# ════════════════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# LOO cross-validation
# ════════════════════════════════════════════════════════════════════════════

def loo_eval(model_factory, X, y, label=""):
    loo = LeaveOneOut()
    preds, truths = [], []
    scaler_x = MinMaxScaler(feature_range=(0.0, np.pi))
    scaler_y = StandardScaler()
    Xs = scaler_x.fit_transform(X)
    ys_std = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        m = model_factory()
        m.fit(X[train_idx], y[train_idx])
        mu = m.predict(X[test_idx])
        preds.append(float(mu[0]))
        truths.append(float(y[test_idx[0]]))
        if (fold + 1) % 10 == 0:
            print(f"    [{label}] LOO {fold+1}/{len(X)}")

    preds, truths = np.array(preds), np.array(truths)
    rmse = np.sqrt(mean_squared_error(truths, preds))
    r2   = r2_score(truths, preds)
    print(f"  [{label}] LOO RMSE={rmse:.2f}µm  R²={r2:.3f}")
    return preds, truths, rmse, r2


# ════════════════════════════════════════════════════════════════════════════
# Plots
# ════════════════════════════════════════════════════════════════════════════

def plot_kernel_matrix(X, y, save=True):
    """Visualize quantum vs classical kernel matrices."""
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    Xs = scaler.fit_transform(X)
    sort_idx = np.argsort(y)
    Xs_s = Xs[sort_idx]

    qkernel = ZZFeatureMapKernel()
    ckernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

    print("  Computing quantum kernel matrix …")
    t0 = time.time()
    K_q = qkernel(Xs_s)
    print(f"  Done in {time.time()-t0:.1f}s")
    K_c = ckernel(Xs_s)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    im0 = axes[0].imshow(K_q, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Quantum Kernel\n(ZZFeatureMap, 4 qubits)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(K_c, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Classical Kernel\n(RBF, normalized)")
    plt.colorbar(im1, ax=axes[1])

    diff = K_q - K_c
    lim  = np.abs(diff).max()
    im2  = axes[2].imshow(diff, cmap="RdBu", vmin=-lim, vmax=lim)
    axes[2].set_title("Quantum − Classical")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("Sample (sorted by chipping)")
        ax.set_ylabel("Sample")

    plt.suptitle("Kernel matrix comparison — SiC dicing (4H-SiC, N="
                 + str(len(X)) + ")", fontsize=11)
    plt.tight_layout()
    if save:
        path = os.path.join(OUT_DIR, "quantum_kernel_matrix.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close()


def plot_loo_comparison(results: dict, save=True):
    """LOO scatter + residual for quantum vs classical GP."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    colors = {"Quantum GP": "#2166ac", "Classical GP": "#d62728"}
    markers = {"Quantum GP": "o", "Classical GP": "s"}

    for label, (preds, truths, rmse, r2) in results.items():
        axes[0].scatter(truths, preds, label=f"{label} (R²={r2:.3f})",
                        color=colors[label], marker=markers[label],
                        alpha=0.75, s=50)

    lim = [0, max(p.max() for p, _, _, _ in results.values()) * 1.1]
    axes[0].plot(lim, lim, "k--", lw=1, label="Perfect")
    axes[0].set_xlabel("Measured chipping [µm]")
    axes[0].set_ylabel("LOO predicted chipping [µm]")
    axes[0].set_title("Leave-One-Out Cross-Validation")
    axes[0].legend()
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)

    x = np.arange(len(list(results.values())[0][1]))
    w = 0.35
    for i, (label, (preds, truths, rmse, r2)) in enumerate(results.items()):
        residuals = preds - truths
        axes[1].bar(x + i * w, np.abs(residuals), w,
                    label=f"{label} (RMSE={rmse:.2f}µm)",
                    color=colors[label], alpha=0.7)

    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("|Residual| [µm]")
    axes[1].set_title("Absolute residuals")
    axes[1].legend()

    plt.suptitle("Quantum Kernel GP vs Classical GP — SiC Dicing", fontsize=11)
    plt.tight_layout()
    if save:
        path = os.path.join(OUT_DIR, "quantum_kernel_gp_loo.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close()


def plot_prediction_surface(model_q, model_c, X, y, save=True):
    """2D prediction surface: cut_depth vs feed_mm_s (blade_W, spindle fixed)."""
    bw_fix  = 23.0
    rpm_fix = 30000.0

    d_grid = np.linspace(80, 390, 30)
    f_grid = np.linspace(0.5, 2.5, 30)
    D, F   = np.meshgrid(d_grid, f_grid)

    X_grid = np.column_stack([
        D.ravel(),
        np.full(D.size, bw_fix),
        F.ravel(),
        np.full(D.size, rpm_fix),
    ])

    mu_q = model_q.predict(X_grid).reshape(D.shape)

    scaler_c = StandardScaler()
    Xsc = scaler_c.fit_transform(X)
    X_grid_sc = scaler_c.transform(X_grid)
    yc_std = StandardScaler()
    ycs = yc_std.fit_transform(y.reshape(-1, 1)).ravel()
    model_c.fit(Xsc, ycs)
    mu_c_std = model_c.predict(X_grid_sc)
    mu_c = yc_std.inverse_transform(mu_c_std.reshape(-1, 1)).ravel().reshape(D.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    vmin = min(mu_q.min(), mu_c.min())
    vmax = max(mu_q.max(), mu_c.max())

    for ax, mu, title in [
        (axes[0], mu_q, "Quantum Kernel GP"),
        (axes[1], mu_c, "Classical RBF GP"),
    ]:
        im = ax.contourf(D, F, mu, levels=20, cmap="YlOrRd",
                         vmin=vmin, vmax=vmax)
        ax.scatter(X[:, 0], X[:, 2], c=y, cmap="YlOrRd",
                   vmin=vmin, vmax=vmax, edgecolors="k", s=60, zorder=5)
        plt.colorbar(im, ax=ax, label="Chipping [µm]")
        ax.set_xlabel("Cut depth [µm]")
        ax.set_ylabel("Feed speed [mm/s]")
        ax.set_title(title)

    plt.suptitle(f"Chipping surface (blade={bw_fix}µm, spindle={rpm_fix:.0f}rpm)",
                 fontsize=11)
    plt.tight_layout()
    if save:
        path = os.path.join(OUT_DIR, "quantum_kernel_surface.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# Learning curve ablation
# ════════════════════════════════════════════════════════════════════════════

def _make_mlp():
    """Small MLP surrogate (sklearn). Hidden: 64-32-16, ReLU, early stopping."""
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=30,
        random_state=0,
        alpha=0.01,       # L2 regularisation
    )


def learning_curve_ablation(X: np.ndarray, y: np.ndarray,
                             n_layers: int = 2,
                             alpha: float = 1.0,
                             n_trials: int = 10,
                             save: bool = True):
    """
    Ablation: RMSE vs training-set size N_train.
    Models: Quantum Kernel GP / Classical RBF-GP / MLP (64-32-16).

    For each N_train in [4, 6, 8, 10, 12, 14]:
        Repeat n_trials random train/test splits.
        Record mean ± std RMSE on the held-out test set.
    """
    from sklearn.neural_network import MLPRegressor  # noqa: F401 (ensure import)

    N_total  = len(X)
    N_values = [n for n in [4, 6, 8, 10, 12, 14] if n < N_total]
    rng      = np.random.default_rng(0)

    rmse_q = {n: [] for n in N_values}
    rmse_c = {n: [] for n in N_values}
    rmse_n = {n: [] for n in N_values}   # NN

    for n_train in N_values:
        print(f"  N_train={n_train} …", flush=True)
        for trial in range(n_trials):
            idx    = rng.permutation(N_total)
            tr, te = idx[:n_train], idx[n_train:]
            if len(te) == 0:
                continue

            # ── Quantum GP ────────────────────────────────────────────────
            qgp = QuantumKernelGP(n_layers=n_layers, alpha=alpha)
            qgp.fit(X[tr], y[tr])
            rmse_q[n_train].append(
                np.sqrt(np.mean((qgp.predict(X[te]) - y[te])**2)))

            # ── Classical GP ──────────────────────────────────────────────
            class _CGP:
                def __init__(self):
                    self._gp = build_classical_gp()
                    self._sx = StandardScaler()
                    self._sy = StandardScaler()
                def fit(self, Xtr, ytr):
                    Xs = self._sx.fit_transform(Xtr)
                    ys = self._sy.fit_transform(ytr.reshape(-1,1)).ravel()
                    self._gp.fit(Xs, ys)
                def predict(self, Xte):
                    Xs = self._sx.transform(Xte)
                    ys = self._gp.predict(Xs)
                    return self._sy.inverse_transform(ys.reshape(-1,1)).ravel()

            cgp = _CGP(); cgp.fit(X[tr], y[tr])
            rmse_c[n_train].append(
                np.sqrt(np.mean((cgp.predict(X[te]) - y[te])**2)))

            # ── MLP ───────────────────────────────────────────────────────
            sx = StandardScaler(); sy = StandardScaler()
            Xs = sx.fit_transform(X[tr])
            ys = sy.fit_transform(y[tr].reshape(-1,1)).ravel()

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # early_stopping needs validation split; disable for very small N
                use_es = n_train >= 8
                mlp = MLPRegressor(
                    hidden_layer_sizes=(64, 32, 16),
                    activation="relu",
                    max_iter=2000,
                    early_stopping=use_es,
                    validation_fraction=0.2 if use_es else 0.0,
                    n_iter_no_change=30,
                    random_state=trial,
                    alpha=0.05,
                )
                mlp.fit(Xs, ys)

            mu_n = sy.inverse_transform(
                mlp.predict(sx.transform(X[te])).reshape(-1,1)).ravel()
            rmse_n[n_train].append(
                np.sqrt(np.mean((mu_n - y[te])**2)))

    # ── Plot ──────────────────────────────────────────────────────────────────
    ns = N_values
    def _stats(d): return (
        np.array([np.mean(d[n]) for n in ns]),
        np.array([np.std(d[n])  for n in ns]),
    )
    mq, sq = _stats(rmse_q)
    mc, sc = _stats(rmse_c)
    mn, sn = _stats(rmse_n)

    models = [
        ("Quantum Kernel GP", mq, sq, "o-", "#2166ac"),
        ("Classical RBF-GP",  mc, sc, "s-", "#d62728"),
        ("MLP (64-32-16)",     mn, sn, "^-", "#1a9641"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, m, s, style, col in models:
        ax.plot(ns, m, style, color=col, lw=2, ms=7, label=label)
        ax.fill_between(ns, m - s, m + s, alpha=0.15, color=col)

    # Mark quantum crossovers vs each baseline
    for label, m_base, _, _, col in models[1:]:
        cross = [i for i in range(len(ns)-1)
                 if (mq[i]-m_base[i]) * (mq[i+1]-m_base[i+1]) < 0]
        if cross:
            ci = cross[0]
            ax.axvline(ns[ci], color=col, ls=":", lw=1.0, alpha=0.6)
            ax.annotate(f"Q×{label.split()[0]} ~N={ns[ci]}",
                        xy=(ns[ci], max(mq[ci], m_base[ci])),
                        xytext=(ns[ci]+0.3, max(mq[ci], m_base[ci])*1.1),
                        fontsize=7, color=col,
                        arrowprops=dict(arrowstyle="->", color=col))

    ax.set_xlabel("Training set size N")
    ax.set_ylabel("RMSE [µm]  (mean ± std, 10 trials)")
    ax.set_title("Learning curves — Quantum GP vs Classical GP vs MLP\n"
                 "(SiC dicing, 4H-SiC experimental data)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(ns)
    plt.tight_layout()

    if save:
        path = os.path.join(OUT_DIR, "quantum_learning_curves.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close()

    print(f"\n  {'N':>2}  |  {'Quantum':>14}  |  {'Classical GP':>14}  |  {'MLP':>14}")
    print("  " + "-" * 58)
    for n in ns:
        print(f"  {n:2d}  |  "
              f"{np.mean(rmse_q[n]):5.2f} ± {np.std(rmse_q[n]):.2f} µm  |  "
              f"{np.mean(rmse_c[n]):5.2f} ± {np.std(rmse_c[n]):.2f} µm  |  "
              f"{np.mean(rmse_n[n]):5.2f} ± {np.std(rmse_n[n]):.2f} µm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",       action="store_true",
                        help="Skip LOO (just fit + surface plot)")
    parser.add_argument("--kernel-only", action="store_true",
                        help="Only plot kernel matrices, no GP training")
    parser.add_argument("--ablation",    action="store_true",
                        help="Run learning curve ablation (quantum vs classical)")
    parser.add_argument("--n-layers",    type=int, default=2,
                        help="ZZFeatureMap layers (default: 2)")
    parser.add_argument("--alpha",       type=float, default=1.0,
                        help="GP noise regularisation (default: 1.0)")
    parser.add_argument("--n-trials",    type=int, default=10,
                        help="Trials per N in ablation (default: 10)")
    parser.add_argument("--quality",     choices=["all", "AB", "A"], default="all",
                        help="Data quality filter: all / AB / A (default: all)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("Quantum Kernel GP — SiC Dicing Surrogate")
    print(f"  n_qubits={N_QUBITS}  n_layers={args.n_layers}  "
          f"alpha={args.alpha}  quality={args.quality}")
    print("=" * 60)

    X, y, alpha_vec = load_data(quality_filter=args.quality)
    print(f"Data: {len(X)} points, features={FEATURE_COLS}")
    print(f"Chipping range: {y.min():.1f}–{y.max():.1f} µm")
    print(f"Noise σ range: {np.sqrt(alpha_vec).min():.1f}–"
          f"{np.sqrt(alpha_vec).max():.1f} µm (per-point)")

    # ── Kernel matrix visualisation ──────────────────────────────────────────
    plot_kernel_matrix(X, y)

    if args.kernel_only:
        return

    # ── Fit both models on all data (for surface plot) ────────────────────────
    print("\n--- Training Quantum Kernel GP (quality-aware α) ---")
    qgp = QuantumKernelGP(n_layers=args.n_layers, alpha=args.alpha)
    qgp.fit(X, y, alpha_vec=alpha_vec)

    cgp = build_classical_gp(alpha=args.alpha)
    plot_prediction_surface(qgp, cgp, X, y)

    if args.quick:
        print("\nQuick mode: skipping LOO.")
        return

    # ── LOO cross-validation ─────────────────────────────────────────────────
    print("\n--- LOO Cross-Validation ---")
    print("  [Quantum GP] this may take ~2–5 min on CPU …")

    q_results = loo_eval(
        lambda: QuantumKernelGP(n_layers=args.n_layers, alpha=args.alpha),
        X, y, label="Quantum GP"
    )

    scaler_ref = StandardScaler()
    X_sc = scaler_ref.fit_transform(X)

    def classical_factory():
        class _CGP:
            def __init__(self):
                self._gp = build_classical_gp()
                self._sx = StandardScaler()
                self._sy = StandardScaler()
            def fit(self, Xtr, ytr):
                Xs = self._sx.fit_transform(Xtr)
                ys = self._sy.fit_transform(ytr.reshape(-1,1)).ravel()
                self._gp.fit(Xs, ys)
            def predict(self, Xte):
                Xs = self._sx.transform(Xte)
                ys = self._gp.predict(Xs)
                return self._sy.inverse_transform(ys.reshape(-1,1)).ravel()
        return _CGP()

    c_results = loo_eval(classical_factory, X, y, label="Classical GP")

    results = {
        "Quantum GP":   q_results,
        "Classical GP": c_results,
    }
    plot_loo_comparison(results)

    print("\n=== Summary ===")
    for label, (_, _, rmse, r2) in results.items():
        print(f"  {label:15s}  RMSE={rmse:.2f}µm  R²={r2:.3f}")

    # ── Ablation (optional) ───────────────────────────────────────────────────
    if args.ablation:
        print("\n--- Learning Curve Ablation ---")
        print(f"  n_trials={args.n_trials} per N  …")
        learning_curve_ablation(X, y, n_layers=args.n_layers,
                                alpha=args.alpha, n_trials=args.n_trials)


def main_ablation_only(quality: str = "all"):
    """Entry point for ablation-only run (skips LOO)."""
    X, y, _ = load_data(quality_filter=quality)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Learning Curve Ablation (quality={quality}, N={len(X)})")
    learning_curve_ablation(X, y, n_trials=10)


if __name__ == "__main__":
    main()
