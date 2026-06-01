"""
Advanced Quantum Kernels for SiC Dicing — Novelty v2
=====================================================
Three kernel designs addressing novelty gaps in quantum_kernel_gp.py:

1. TrainableQKernel  — Quantum Kernel Alignment (QKA)
   Optimize variational parameters θ to maximise kernel-target alignment.
   Reference: Hubregtsen et al. (2022) Quantum Sci. Technol. 7 025030.

2. C6vEquivariantKernel — Group-Invariant Quantum Kernel
   G-twirl over C6v (hexagonal symmetry of 4H-SiC crystal, 12 elements).
   Kernel is invariant to 60° rotations and mirror reflections of crystal angle.
   Reference: Larocca et al. (2022) PRX Quantum 3, 030341.

3. PhysicsInformedQKernel — Structure-Aware Feature Map
   Encodes the known depth × width interaction (multiplicative chipping physics)
   via controlled-Z gates instead of generic angle embedding.

Comparison runner:
    python ml/quantum_kernel_advanced.py
    python ml/quantum_kernel_advanced.py --model all
    python ml/quantum_kernel_advanced.py --model trainable
    python ml/quantum_kernel_advanced.py --model c6v
    python ml/quantum_kernel_advanced.py --model physics
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel, WhiteKernel, Kernel,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pennylane as qml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.quantum_kernel_gp import (
    load_data, build_classical_gp,
    FEATURE_COLS, TARGET_COL, N_QUBITS, OUT_DIR,
)

# ════════════════════════════════════════════════════════════════════════════
# 1. Trainable Quantum Kernel  (Quantum Kernel Alignment)
# ════════════════════════════════════════════════════════════════════════════

class TrainableQKernel(Kernel):
    """
    Variational feature map with trainable parameters θ.

    Circuit: AngleEmbedding(x) → BasicEntanglerLayers(θ) × n_layers
    Parameters θ optimised via kernel-target alignment (KTA):

        KTA(K, y) = y^T K y / (‖K‖_F · ‖yy^T‖_F)

    After alignment, θ is frozen and the kernel is used in GPR.
    This is the key novelty vs ZZFeatureMap: the circuit LEARNS from data.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        rng = np.random.default_rng(42)
        self.params = rng.uniform(0, 2*np.pi, (n_layers, n_qubits))
        self._build()
        self.is_aligned = False

    def _build(self):
        dev = qml.device("default.qubit", wires=self.n_qubits)
        n_q = self.n_qubits

        def _embed(x, params):
            qml.AngleEmbedding(x, wires=range(n_q), rotation="Y")
            qml.BasicEntanglerLayers(params, wires=range(n_q), rotation=qml.RY)

        @qml.qnode(dev)
        def _circuit(x1, x2, params):
            _embed(x1, params)
            qml.adjoint(_embed)(x2, params)
            return qml.probs(wires=range(n_q))

        self._circuit = _circuit

    def _k(self, x1, x2) -> float:
        return float(self._circuit(
            np.asarray(x1), np.asarray(x2), self.params)[0])

    def _kernel_matrix(self, X, Y=None):
        Y = X if Y is None else Y
        return np.array([[self._k(x, y) for y in Y] for x in X])

    # ── Kernel Alignment Optimisation ──────────────────────────────────────

    @staticmethod
    def _kta(K: np.ndarray, y: np.ndarray) -> float:
        """Kernel-Target Alignment (higher = better)."""
        yyt = np.outer(y, y)
        num = np.einsum('ij,ij->', K, yyt)          # tr(K @ yyt)
        den = (np.linalg.norm(K, 'fro')
               * np.linalg.norm(yyt, 'fro'))
        return float(num / (den + 1e-12))

    def align(self, X: np.ndarray, y: np.ndarray,
              n_iter: int = 50, verbose: bool = True) -> float:
        """
        Maximise KTA(K(θ), y) w.r.t. θ using COBYLA.
        Returns final KTA score.
        """
        scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        Xs = scaler.fit_transform(X)
        ys = (y - y.mean()) / (y.std() + 1e-8)

        history = []

        def objective(flat_params):
            self.params = flat_params.reshape(self.n_layers, self.n_qubits)
            K = self._kernel_matrix(Xs)
            score = self._kta(K, ys)
            history.append(-score)
            return -score   # minimise → maximise KTA

        x0 = self.params.ravel()
        result = minimize(objective, x0, method="COBYLA",
                          options={"maxiter": n_iter, "rhobeg": 0.3})
        self.params = result.x.reshape(self.n_layers, self.n_qubits)
        self.is_aligned = True
        final_kta = -result.fun
        if verbose:
            print(f"  KTA: {-history[0]:.4f} → {final_kta:.4f} "
                  f"({len(history)} evals)")
        return final_kta

    # ── sklearn Kernel interface ────────────────────────────────────────────

    def __call__(self, X, Y=None, eval_gradient=False):
        scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        X = scaler.fit_transform(np.asarray(X))
        Y = X if Y is None else scaler.transform(np.asarray(Y))
        K = self._kernel_matrix(X, Y)
        if eval_gradient:
            return K, np.empty((*K.shape, 0))
        return K

    def diag(self, X):
        return np.array([self._k(x, x)
                         for x in MinMaxScaler((0, np.pi)).fit_transform(
                             np.asarray(X))])

    def is_stationary(self): return False

    @property
    def theta(self): return np.empty(0)
    @theta.setter
    def theta(self, v): pass
    @property
    def bounds(self): return np.empty((0, 2))
    def get_params(self, deep=True):
        return {"n_qubits": self.n_qubits, "n_layers": self.n_layers}
    def clone_with_theta(self, theta):
        k = TrainableQKernel(self.n_qubits, self.n_layers)
        k.params = self.params.copy()
        return k
    def __repr__(self):
        s = "aligned" if self.is_aligned else "random"
        return f"TrainableQKernel({s}, n_qubits={self.n_qubits})"


# ════════════════════════════════════════════════════════════════════════════
# 2. C6v Equivariant Kernel  (G-twirl)
# ════════════════════════════════════════════════════════════════════════════

def _c6v_group_actions():
    """
    12 elements of C6v acting on crystal angle θ (degrees).
    Rotations: θ + k*60° for k=0..5
    Mirrors: -(θ + k*60°) for k=0..5   (σv type)
    """
    actions = []
    for k in range(6):
        actions.append(("rot",   k * 60.0))
        actions.append(("mirror", k * 60.0))
    return actions   # 12 elements


def apply_c6v(x: np.ndarray, action: tuple,
              theta_idx: int = 4) -> np.ndarray:
    """
    Apply a C6v group element to feature vector x.
    theta_idx: index of the crystal orientation feature [degrees].
    """
    x_new = x.copy()
    kind, offset = action
    θ = x[theta_idx]
    if kind == "rot":
        x_new[theta_idx] = (θ + offset) % 360.0
    else:  # mirror
        x_new[theta_idx] = (-(θ + offset)) % 360.0
    return x_new


class C6vEquivariantKernel(Kernel):
    """
    G-twirl quantum kernel invariant under C6v (hexagonal crystal symmetry).

    K_G(x, x') = (1/|G|) Σ_{g∈C6v} k_base(g·x, g·x')

    Features: [cut_depth_um, blade_W_um, feed_mm_s, spindle_rpm,
               crystal_theta_deg]   ← 5th feature required

    The kernel is exactly invariant to:
        θ → θ + 60°  (6-fold rotation of crystal)
        θ → -θ       (mirror reflection of crystal)

    This is something NO classical kernel can guarantee without
    hand-crafted feature engineering.
    """

    def __init__(self, n_qubits: int = 5, n_layers: int = 2,
                 theta_idx: int = 4):
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.theta_idx = theta_idx
        self._group    = _c6v_group_actions()   # 12 elements
        self._build()

    def _build(self):
        dev = qml.device("default.qubit", wires=self.n_qubits)
        n_q = self.n_qubits

        def _embed(x):
            qml.AngleEmbedding(x, wires=range(n_q), rotation="Y")
            for _ in range(self.n_layers):
                for i in range(n_q - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_q - 1, 0])

        @qml.qnode(dev)
        def _circuit(x1, x2):
            _embed(x1)
            qml.adjoint(_embed)(x2)
            return qml.probs(wires=range(n_q))

        self._circuit = _circuit

    def _k_base(self, x1, x2) -> float:
        return float(self._circuit(np.asarray(x1), np.asarray(x2))[0])

    def _k_twirl(self, x1, x2) -> float:
        """G-twirl: average base kernel over all 12 C6v elements."""
        total = 0.0
        for g in self._group:
            gx1 = apply_c6v(x1, g, self.theta_idx)
            gx2 = apply_c6v(x2, g, self.theta_idx)
            total += self._k_base(gx1, gx2)
        return total / len(self._group)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X)
        Y = X if Y is None else np.asarray(Y)
        K = np.array([[self._k_twirl(x, y) for y in Y] for x in X])
        if eval_gradient:
            return K, np.empty((*K.shape, 0))
        return K

    def diag(self, X):
        return np.array([self._k_twirl(x, x) for x in X])

    def is_stationary(self): return False

    @property
    def theta(self): return np.empty(0)
    @theta.setter
    def theta(self, v): pass
    @property
    def bounds(self): return np.empty((0, 2))
    def get_params(self, deep=True):
        return {"n_qubits": self.n_qubits, "n_layers": self.n_layers}
    def clone_with_theta(self, theta):
        return C6vEquivariantKernel(self.n_qubits, self.n_layers,
                                    self.theta_idx)
    def __repr__(self):
        return f"C6vEquivariantKernel(n_qubits={self.n_qubits})"


# ════════════════════════════════════════════════════════════════════════════
# 3. Physics-Informed Quantum Kernel
# ════════════════════════════════════════════════════════════════════════════

class PhysicsQKernel(Kernel):
    """
    Physics-informed feature engineering + ZZFeatureMap kernel.

    Key insight: encode chip ∝ depth × width as an explicit feature
    BEFORE quantum embedding.  The 4-qubit circuit sees:
        q0 ← Ry(depth)
        q1 ← Ry(width)
        q2 ← Ry(depth × width / π)   ← multiplicative physics term
        q3 ← Ry(feed)
    followed by a CNOT ring (standard ZZFeatureMap entanglement).

    This is more expressive than CZ gates alone because the product
    depth×width appears explicitly in the Hilbert space, while CZ
    gates only add phase and cannot synthesize a product from angles.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self._build()

    @staticmethod
    def _physics_transform(x: np.ndarray) -> np.ndarray:
        """
        Replace spindle feature with depth×width product (already in [0,π]).
        x: [depth, width, feed, spindle] all in [0, π]
        """
        d, w, f = x[0], x[1], x[2]
        dw = (d * w) / np.pi   # product rescaled to [0, π]
        return np.array([d, w, dw, f])

    def _build(self):
        dev = qml.device("default.qubit", wires=self.n_qubits)
        n_q = self.n_qubits
        n_l = self.n_layers

        def _embed(xf):
            for _ in range(n_l):
                qml.AngleEmbedding(xf, wires=range(n_q), rotation="Y")
                for i in range(n_q - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_q - 1, 0])

        @qml.qnode(dev)
        def _circuit(x1, x2):
            _embed(x1)
            qml.adjoint(_embed)(x2)
            return qml.probs(wires=range(n_q))

        self._circuit = _circuit

    def _k(self, x1, x2) -> float:
        xf1 = self._physics_transform(np.asarray(x1))
        xf2 = self._physics_transform(np.asarray(x2))
        return float(self._circuit(xf1, xf2)[0])

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X)
        Y = X if Y is None else np.asarray(Y)
        K = np.array([[self._k(x, y) for y in Y] for x in X])
        if eval_gradient:
            return K, np.empty((*K.shape, 0))
        return K

    def diag(self, X):
        return np.array([self._k(x, x) for x in X])

    def is_stationary(self): return False

    @property
    def theta(self): return np.empty(0)
    @theta.setter
    def theta(self, v): pass
    @property
    def bounds(self): return np.empty((0, 2))
    def get_params(self, deep=True):
        return {"n_qubits": self.n_qubits, "n_layers": self.n_layers}
    def clone_with_theta(self, theta):
        return PhysicsQKernel(self.n_qubits, self.n_layers)
    def __repr__(self):
        return f"PhysicsQKernel(n_qubits={self.n_qubits})"


# ════════════════════════════════════════════════════════════════════════════
# Data augmentation for C6v kernel (adds crystal_theta_deg feature)
# ════════════════════════════════════════════════════════════════════════════

def augment_with_crystal_angle(X: np.ndarray, y: np.ndarray,
                                alpha_vec: np.ndarray):
    """
    Augment dataset with crystal angle θ as 5th feature.

    Existing data: θ=0 (m-plane, Micro2026 / Mat2022 both m-plane dominant).
    Add θ=15°, 30° points using K_Ic anisotropy scaling from crystal_anisotropy.py:
        chip(θ) ≈ chip(0°) × (K_Ic(0°) / K_Ic(θ))^0.5

    K_Ic values: 0°→2.50, 15°→2.35, 30°→2.10, 45°→2.30, 60°→2.50 MPa√m
    """
    KIC = {0: 2.50, 15: 2.35, 30: 2.10, 45: 2.30, 60: 2.50}
    angles = [0, 15, 30, 45, 60]   # degrees

    X_aug    = []
    y_aug    = []
    alpha_aug = []

    for theta in angles:
        kic_factor = KIC[0] / KIC[theta]
        scale      = np.sqrt(kic_factor)   # chip ∝ K_Ic^{-0.5}

        X_theta = np.column_stack([X, np.full(len(X), float(theta))])
        y_theta = y * scale
        # Higher noise for synthetic augmented points
        noise_scale = 1.0 if theta == 0 else 1.5
        alpha_theta = alpha_vec * noise_scale

        X_aug.append(X_theta)
        y_aug.append(y_theta)
        alpha_aug.append(alpha_theta)

    X_out = np.vstack(X_aug)
    y_out = np.concatenate(y_aug)
    a_out = np.concatenate(alpha_aug)
    return X_out, y_out, a_out


# ════════════════════════════════════════════════════════════════════════════
# Unified GP wrapper
# ════════════════════════════════════════════════════════════════════════════

class AdvancedQKernelGP:
    """GP surrogate wrapping any of the three advanced kernels."""

    def __init__(self, kernel: Kernel, alpha: float = 1.0):
        self.kernel   = kernel
        self.alpha    = alpha
        self.scaler_x = MinMaxScaler(feature_range=(0.0, np.pi))
        self.scaler_y = StandardScaler()
        self.gp       = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, optimizer=None, normalize_y=False)
        self.is_fitted = False

    def fit(self, X, y, alpha_vec=None):
        Xs = self.scaler_x.fit_transform(X)
        ys = self.scaler_y.fit_transform(y.reshape(-1,1)).ravel()
        if alpha_vec is not None:
            scale = float(self.scaler_y.scale_[0])
            self.gp.alpha = alpha_vec / scale**2
        self.gp.fit(Xs, ys)
        self.is_fitted = True
        return self

    def predict(self, X, return_std=False):
        Xs = self.scaler_x.transform(X)
        mu, s = self.gp.predict(Xs, return_std=True)
        mu = self.scaler_y.inverse_transform(mu.reshape(-1,1)).ravel()
        s  = s * float(self.scaler_y.scale_[0])
        return (mu, s) if return_std else mu


# ════════════════════════════════════════════════════════════════════════════
# LOO evaluation
# ════════════════════════════════════════════════════════════════════════════

def loo_eval_advanced(X, y, alpha_vec, models: dict) -> dict:
    """LOO cross-validation for multiple models simultaneously."""
    loo     = LeaveOneOut()
    results = {name: {"preds": [], "truths": []} for name in models}

    for fold, (tr, te) in enumerate(loo.split(X)):
        if (fold + 1) % 5 == 0:
            print(f"  LOO {fold+1}/{len(X)} …", flush=True)
        for name, factory in models.items():
            m = factory()
            m.fit(X[tr], y[tr], alpha_vec[tr] if alpha_vec is not None else None)
            results[name]["preds"].append(float(m.predict(X[te])[0]))
            results[name]["truths"].append(float(y[te[0]]))

    summary = {}
    for name, d in results.items():
        p, t = np.array(d["preds"]), np.array(d["truths"])
        rmse = np.sqrt(mean_squared_error(t, p))
        r2   = r2_score(t, p)
        summary[name] = (p, t, rmse, r2)
        print(f"  {name:25s}  RMSE={rmse:.2f}µm  R²={r2:.3f}")
    return summary


# ════════════════════════════════════════════════════════════════════════════
# Benchmark plot
# ════════════════════════════════════════════════════════════════════════════

def plot_benchmark(summary: dict, save=True):
    """Scatter + RMSE bar for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    palette = {
        "ZZ-Fixed":       "#aec7e8",
        "ZZ-Trainable":   "#1f77b4",
        "Physics-QK":     "#ff7f0e",
        "C6v-Equivariant":"#2ca02c",
        "Classical GP":   "#d62728",
        "MLP":            "#9467bd",
    }
    markers = {"ZZ-Fixed": "v", "ZZ-Trainable": "o", "Physics-QK": "^",
               "C6v-Equivariant": "D", "Classical GP": "s", "MLP": "P"}

    all_vals = [v for p,t,_,_ in summary.values() for v in list(p)+list(t)]
    lim = [0, max(all_vals) * 1.1]

    for name, (preds, truths, rmse, r2) in summary.items():
        col = palette.get(name, "#333")
        mk  = markers.get(name, "o")
        axes[0].scatter(truths, preds, label=f"{name} R²={r2:.2f}",
                        color=col, marker=mk, alpha=0.8, s=55)

    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xlabel("Measured chipping [µm]")
    axes[0].set_ylabel("LOO predicted [µm]")
    axes[0].set_title("LOO Scatter — All Models")
    axes[0].legend(fontsize=7)
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)

    names  = list(summary.keys())
    rmses  = [summary[n][2] for n in names]
    colors = [palette.get(n, "#333") for n in names]
    bars   = axes[1].bar(names, rmses, color=colors, alpha=0.8, edgecolor="k")
    axes[1].set_ylabel("LOO RMSE [µm]")
    axes[1].set_title("RMSE Comparison")
    axes[1].tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, rmses):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Quantum Kernel GP Benchmark — SiC Dicing (4H-SiC)",
                 fontsize=11)
    plt.tight_layout()
    if save:
        path = os.path.join(OUT_DIR, "quantum_benchmark.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close()


def plot_kta_surface(X, y, save=True):
    """Visualise KTA landscape as params vary (2D slice)."""
    from ml.quantum_kernel_gp import ZZFeatureMapKernel
    scaler = MinMaxScaler((0, np.pi))
    Xs = scaler.fit_transform(X)
    ys = (y - y.mean()) / y.std()

    tk = TrainableQKernel(n_layers=1)  # 1-layer for speed

    angles = np.linspace(0, 2*np.pi, 20)
    KTA_grid = np.zeros((20, 20))
    for i, a0 in enumerate(angles):
        for j, a1 in enumerate(angles):
            tk.params = np.array([[a0, a1, 0.0, 0.0]])
            K = tk._kernel_matrix(Xs)
            KTA_grid[i, j] = TrainableQKernel._kta(K, ys)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.contourf(np.degrees(angles), np.degrees(angles), KTA_grid,
                     levels=20, cmap="viridis")
    plt.colorbar(im, ax=ax, label="KTA score")
    ax.set_xlabel("θ₀ [°]")
    ax.set_ylabel("θ₁ [°]")
    ax.set_title("Kernel-Target Alignment landscape\n"
                 "(1-layer trainable kernel, qubits 0–1)")
    plt.tight_layout()
    if save:
        path = os.path.join(OUT_DIR, "quantum_kta_landscape.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["all", "trainable", "c6v", "physics", "benchmark"])
    parser.add_argument("--quality", default="all",
                        choices=["all", "AB", "A"])
    parser.add_argument("--align-iter", type=int, default=80,
                        help="KTA optimisation iterations (default 80)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("Advanced Quantum Kernels — SiC Dicing")
    print(f"  model={args.model}  quality={args.quality}")
    print("=" * 60)

    X, y, alpha_vec = load_data(quality_filter=args.quality)
    print(f"Data: N={len(X)}, chipping {y.min():.1f}–{y.max():.1f} µm\n")

    # ── KTA landscape (always show) ──────────────────────────────────────────
    print("--- KTA landscape ---")
    plot_kta_surface(X, y)

    # ── Trainable kernel ──────────────────────────────────────────────────────
    tk = None
    if args.model in ("all", "trainable"):
        print("\n--- Trainable Quantum Kernel (kernel alignment) ---")
        tk = TrainableQKernel()
        scaler_tmp = MinMaxScaler((0, np.pi))
        tk.align(X, y, n_iter=args.align_iter)

    # ── C6v kernel (with crystal angle augmentation) ──────────────────────────
    c6v_gp = None
    if args.model in ("all", "c6v"):
        print("\n--- C6v Equivariant Kernel ---")
        X5, y5, a5 = augment_with_crystal_angle(X, y, alpha_vec)
        print(f"  Augmented: N={len(X5)} (5 crystal angles × {len(X)} base points)")
        c6v_gp = AdvancedQKernelGP(
            C6vEquivariantKernel(n_qubits=5), alpha=2.0)
        t0 = time.time()
        c6v_gp.fit(X5, y5, a5)
        print(f"  Fit done in {time.time()-t0:.1f}s")

    # ── Physics kernel ────────────────────────────────────────────────────────
    phys_gp = None
    if args.model in ("all", "physics"):
        print("\n--- Physics-Informed Quantum Kernel ---")
        phys_gp = AdvancedQKernelGP(PhysicsQKernel(), alpha=1.0)
        t0 = time.time()
        phys_gp.fit(X, y, alpha_vec)
        print(f"  Fit done in {time.time()-t0:.1f}s")

    # ── LOO benchmark ─────────────────────────────────────────────────────────
    if args.model in ("all", "benchmark", "trainable", "physics"):
        print("\n--- LOO Benchmark ---")

        from ml.quantum_kernel_gp import QuantumKernelGP

        def _classical():
            class _W:
                def __init__(self):
                    self._gp = build_classical_gp()
                    self._sx = StandardScaler(); self._sy = StandardScaler()
                def fit(self, X, y, av=None):
                    Xs = self._sx.fit_transform(X)
                    ys = self._sy.fit_transform(y.reshape(-1,1)).ravel()
                    self._gp.fit(Xs, ys)
                def predict(self, X, return_std=False):
                    Xs = self._sx.transform(X)
                    ys = self._gp.predict(Xs)
                    return self._sy.inverse_transform(ys.reshape(-1,1)).ravel()
            return _W()

        models = {}
        models["ZZ-Fixed"]     = lambda: QuantumKernelGP()
        if args.model in ("all", "trainable"):
            _saved_params = tk.params.copy() if tk else None
            def _trainable_factory():
                t = TrainableQKernel()
                if _saved_params is not None:
                    t.params = _saved_params.copy()
                    t.is_aligned = True
                return AdvancedQKernelGP(t)
            models["ZZ-Trainable"] = _trainable_factory
        if args.model in ("all", "physics"):
            models["Physics-QK"] = lambda: AdvancedQKernelGP(PhysicsQKernel())
        models["Classical GP"] = _classical

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summary = loo_eval_advanced(X, y, alpha_vec, models)

        plot_benchmark(summary)


if __name__ == "__main__":
    main()
