"""
Quantum Annealing Recipe Optimiser — SiC Blade Dicing
======================================================
QUBO (Quadratic Unconstrained Binary Optimisation) formulation of the
dicing recipe problem.  Solved via simulated annealing; architecture
maps directly to Ising hardware (D-Wave / Fixstars Amplify).

Layer 3 of the Quantum Stack:
    Layer 1: EKF data assimilation  (Muramatsu Lab 2026)
    Layer 2: Quantum Kernel GP      (ml/quantum_kernel_gp.py)
    Layer 3: THIS — QUBO recipe optimiser

QUBO Formulation (2-parameter, one-hot encoding):
    Binary variables:
        x_d ∈ {0,1}^{N_d}  — one-hot over cut_depth grid
        x_w ∈ {0,1}^{N_w}  — one-hot over blade_W grid

    QUBO objective (minimise):
        H = H_obj + λ H_c1 + λ H_c2

        H_obj = Σ_{i,j} C_{ij} · x_d_i · x_w_j          (GP chipping)
        H_c1  = (Σ_i x_d_i − 1)²                          (one-hot depth)
        H_c2  = (Σ_j x_w_j − 1)²                          (one-hot width)

    → Q matrix (N_d+N_w) × (N_d+N_w)  (upper triangular QUBO)

Ising mapping (for D-Wave):
    s_i = 2x_i − 1 ∈ {−1,+1}
    H_Ising = Σ_{i<j} J_{ij} s_i s_j + Σ_i h_i s_i

Solvers implemented:
    1. Brute force   (exact, ≤24 variables)
    2. Simulated Annealing (numpy, no extra deps)
    [Future: swap SA for amplify.BinaryQuadraticModel + Fixstars solver]

Usage:
    python optimization/quantum_annealing_recipe.py
    python optimization/quantum_annealing_recipe.py --n-levels 10
    python optimization/quantum_annealing_recipe.py --compare  # vs classical BO
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Parameter grids ───────────────────────────────────────────────────────────
# Keep 2 free parameters (cut_depth, blade_W) matching surrogate_gp.py baseline.
# Feed and spindle fixed at reference conditions.
DEPTH_RANGE = (80.0,  390.0)
WIDTH_RANGE = (20.0,   55.0)
FEED_FIX    = 1.0       # mm/s
RPM_FIX     = 30000.0   # rpm

PENALTY_LAMBDA = 8.0    # constraint penalty ≫ max chipping ≈ 25µm (normalised)


# ════════════════════════════════════════════════════════════════════════════
# Physics-based surrogate (no external file dependency)
# Same parametric model as calibrated in validation/experimental_data.py.
# ════════════════════════════════════════════════════════════════════════════

def chipping_model(cut_depth_um: np.ndarray,
                   blade_W_um:   np.ndarray) -> np.ndarray:
    """
    Analytical chipping surrogate calibrated to Micro2026 experimental data.

    Returns chipping width in µm.  Vectorised over arrays of the same shape.
    """
    # Depth sensitivity: shallow cuts → small chipping, deep → larger
    depth_term = 1.2 + 0.035 * (cut_depth_um - 80.0) / 310.0 * 10.0

    # Width sensitivity: wide blade → more material removed → more chipping
    width_term = 0.8 + 0.4 * (blade_W_um - 20.0) / 35.0

    # Nonlinear interaction (wide blade + deep cut → super-linear chipping)
    interaction = 1.0 + 0.12 * (
        (cut_depth_um - 80.0) / 310.0 * (blade_W_um - 20.0) / 35.0
    )

    return depth_term * width_term * interaction * 3.5  # baseline 3.5 µm


# ════════════════════════════════════════════════════════════════════════════
# QUBO Builder
# ════════════════════════════════════════════════════════════════════════════

def build_qubo(n_d: int, n_w: int, lam: float = PENALTY_LAMBDA):
    """
    Build QUBO matrix Q ∈ R^{(n_d+n_w) × (n_d+n_w)}.

    Layout:  indices [0..n_d) = depth variables,
                     [n_d..n_d+n_w) = width variables.
    """
    n = n_d + n_w
    Q = np.zeros((n, n))

    depth_grid = np.linspace(*DEPTH_RANGE, n_d)
    width_grid = np.linspace(*WIDTH_RANGE, n_w)

    # ── Objective: C_{ij} = normalised chipping at (depth_i, width_j) ───────
    D_mesh, W_mesh = np.meshgrid(depth_grid, width_grid, indexing="ij")
    C = chipping_model(D_mesh, W_mesh)
    C_norm = C / C.max()   # [0, 1], ensures lam >> objective

    # off-diagonal block: depth × width
    for i in range(n_d):
        for j in range(n_w):
            Q[i, n_d + j] += C_norm[i, j]

    # ── Constraint 1: (Σ x_d_i − 1)² ───────────────────────────────────────
    # diagonal: -λ per depth variable
    # off-diagonal (i<j, depth×depth): +2λ
    for i in range(n_d):
        Q[i, i] -= lam
    for i in range(n_d):
        for j in range(i + 1, n_d):
            Q[i, j] += 2 * lam

    # ── Constraint 2: (Σ x_w_j − 1)² ───────────────────────────────────────
    for j in range(n_d, n):
        Q[j, j] -= lam
    for i in range(n_d, n):
        for j in range(i + 1, n):
            Q[i, j] += 2 * lam

    return Q, depth_grid, width_grid, C


def qubo_energy(x: np.ndarray, Q: np.ndarray) -> float:
    """H = x^T Q x  (upper-triangular QUBO convention)."""
    return float(x @ Q @ x)


# ════════════════════════════════════════════════════════════════════════════
# Solvers
# ════════════════════════════════════════════════════════════════════════════

def solve_brute_force(Q: np.ndarray, n_d: int, n_w: int):
    """Exact solver by exhaustive search.  O(2^n) — feasible for n≤24."""
    n = Q.shape[0]
    best_E = np.inf
    best_x = None
    for state in range(2 ** n):
        x = np.array([(state >> k) & 1 for k in range(n)], dtype=float)
        E = qubo_energy(x, Q)
        if E < best_E:
            best_E = E
            best_x = x.copy()
    return best_x, best_E


def solve_simulated_annealing(Q: np.ndarray, n_d: int, n_w: int,
                               T_init: float = 5.0,
                               T_final: float = 0.01,
                               n_steps: int = 50_000,
                               seed: int = 0) -> tuple:
    """
    Simulated annealing for QUBO.  Single bit-flip Metropolis moves.

    Architecture note: replace this function body with
        client = amplify.FixstarsClient(token=API_KEY)
        result = client.solve(bqm)
    to run on Fixstars Amplify (same QUBO matrix Q, no other changes).
    """
    rng = np.random.default_rng(seed)
    n   = Q.shape[0]

    # Warm start: one-hot in each group
    x = np.zeros(n)
    x[rng.integers(0, n_d)] = 1.0
    x[n_d + rng.integers(0, n_w)] = 1.0

    E_cur  = qubo_energy(x, Q)
    E_best = E_cur
    x_best = x.copy()
    history = []

    T = T_init
    alpha = (T_final / T_init) ** (1.0 / n_steps)

    for step in range(n_steps):
        k = rng.integers(0, n)
        x_new     = x.copy()
        x_new[k]  = 1.0 - x_new[k]
        E_new     = qubo_energy(x_new, Q)
        dE        = E_new - E_cur

        if dE < 0 or rng.random() < np.exp(-dE / (T + 1e-15)):
            x     = x_new
            E_cur = E_new
            if E_cur < E_best:
                E_best = E_cur
                x_best = x.copy()

        T *= alpha

        if step % 5000 == 0:
            history.append(E_best)

    return x_best, E_best, np.array(history)


def decode_solution(x: np.ndarray, n_d: int, depth_grid: np.ndarray,
                    width_grid: np.ndarray):
    """Map binary solution vector to physical recipe parameters."""
    d_idx = int(np.argmax(x[:n_d]))
    w_idx = int(np.argmax(x[n_d:]))
    return {
        "cut_depth_um": depth_grid[d_idx],
        "blade_W_um":   width_grid[w_idx],
        "feed_mm_s":    FEED_FIX,
        "spindle_rpm":  RPM_FIX,
        "d_idx":        d_idx,
        "w_idx":        w_idx,
    }


def to_ising(Q: np.ndarray):
    """
    Convert QUBO Q to Ising {J, h} using x = (s+1)/2.

    H_ising = Σ_{i<j} J_ij s_i s_j + Σ_i h_i s_i  + const

    Returned: J (upper triangular), h (vector), offset (constant).
    """
    n = Q.shape[0]
    J = np.zeros((n, n))
    h = np.zeros(n)

    # Q_ij x_i x_j  → Q_ij/4 s_i s_j + Q_ij/4 s_i + Q_ij/4 s_j + Q_ij/4
    for i in range(n):
        for j in range(n):
            if i == j:
                h[i] += Q[i, i] / 2.0
            elif i < j:
                J[i, j] += Q[i, j] / 4.0
                h[i]    += Q[i, j] / 4.0
                h[j]    += Q[i, j] / 4.0

    offset = (np.sum(np.triu(Q, k=1)) + np.sum(np.diag(Q))) / 4.0
    return J, h, offset


# ════════════════════════════════════════════════════════════════════════════
# Classical BO baseline
# ════════════════════════════════════════════════════════════════════════════

def solve_classical_de():
    """Differential evolution on the continuous chipping surface."""
    bounds = [DEPTH_RANGE, WIDTH_RANGE]

    def obj(x):
        return float(chipping_model(np.array([x[0]]), np.array([x[1]])))

    result = differential_evolution(obj, bounds, seed=42, maxiter=500, tol=1e-6)
    return {
        "cut_depth_um": result.x[0],
        "blade_W_um":   result.x[1],
        "chipping_um":  result.fun,
    }


# ════════════════════════════════════════════════════════════════════════════
# Plots
# ════════════════════════════════════════════════════════════════════════════

def plot_qubo_matrix(Q, depth_grid, width_grid, save=True):
    """Visualise QUBO matrix and chipping landscape."""
    n_d, n_w = len(depth_grid), len(width_grid)
    D, W = np.meshgrid(depth_grid, width_grid, indexing="ij")
    C = chipping_model(D, W)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: chipping landscape (oracle)
    im0 = axes[0].contourf(depth_grid, width_grid, C.T, levels=20,
                            cmap="YlOrRd")
    plt.colorbar(im0, ax=axes[0], label="Chipping [µm]")
    axes[0].set_xlabel("Cut depth [µm]")
    axes[0].set_ylabel("Blade width [µm]")
    axes[0].set_title("Chipping oracle (physics model)")

    # Right: QUBO matrix heatmap
    lim = np.abs(Q).max()
    im1 = axes[1].imshow(Q, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    plt.colorbar(im1, ax=axes[1], label="Q_{ij}")
    axes[1].axhline(n_d - 0.5, color="k", lw=1.5, ls="--")
    axes[1].axvline(n_d - 0.5, color="k", lw=1.5, ls="--")
    axes[1].set_title(f"QUBO matrix Q ({n_d+n_w}×{n_d+n_w})")
    axes[1].set_xlabel("Variable index")
    axes[1].set_ylabel("Variable index")

    plt.suptitle("QUBO Formulation — Dicing Recipe Optimisation", fontsize=11)
    plt.tight_layout()
    if save:
        path = os.path.join(OUT_DIR, "qubo_matrix.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close()


def plot_sa_convergence(history: np.ndarray,
                        E_exact: float, save=True):
    """Simulated annealing convergence vs exact solution."""
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(len(history)) * 5000
    ax.plot(steps, history, "-o", ms=4, label="SA best energy", color="#2166ac")
    ax.axhline(E_exact, color="#d62728", ls="--", lw=1.5,
               label=f"Exact optimum ({E_exact:.3f})")
    ax.set_xlabel("SA step")
    ax.set_ylabel("QUBO energy H")
    ax.set_title("Simulated Annealing convergence\n"
                 "(→ swap SA body for Fixstars Amplify client)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(OUT_DIR, "qubo_sa_convergence.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close()


def plot_solution_comparison(recipes: dict,
                              depth_grid, width_grid, save=True):
    """Compare QUBO-SA, exact, and classical DE solutions on landscape."""
    D, W = np.meshgrid(depth_grid, width_grid, indexing="ij")
    C    = chipping_model(D, W)

    d_fine = np.linspace(*DEPTH_RANGE, 200)
    w_fine = np.linspace(*WIDTH_RANGE, 200)
    Df, Wf = np.meshgrid(d_fine, w_fine, indexing="ij")
    Cf = chipping_model(Df, Wf)

    fig, ax = plt.subplots(figsize=(9, 6))
    cf = ax.contourf(d_fine, w_fine, Cf.T, levels=30, cmap="YlOrRd", alpha=0.85)
    plt.colorbar(cf, ax=ax, label="Chipping [µm]")

    markers = {"QUBO-SA": ("o", "#2166ac", 120),
               "QUBO-Exact": ("*", "#1a9641", 200),
               "Classical DE": ("s", "#d62728", 120)}

    for name, rec in recipes.items():
        m, c, s = markers.get(name, ("^", "k", 100))
        chip = chipping_model(np.array([[rec["cut_depth_um"]]]),
                              np.array([[rec["blade_W_um"]]]))[0, 0]
        ax.scatter(rec["cut_depth_um"], rec["blade_W_um"],
                   marker=m, color=c, s=s, zorder=10,
                   label=f"{name}: depth={rec['cut_depth_um']:.0f}µm, "
                         f"W={rec['blade_W_um']:.1f}µm, "
                         f"chip={chip:.1f}µm")

    ax.set_xlabel("Cut depth [µm]")
    ax.set_ylabel("Blade width [µm]")
    ax.set_title("Optimal recipes — QUBO vs Classical optimiser")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save:
        path = os.path.join(OUT_DIR, "qubo_solution_comparison.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close()


def plot_ising_coupling(J, h, n_d, n_w, save=True):
    """Ising coupling J_{ij} matrix for D-Wave visualisation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    lim = max(np.abs(J).max(), np.abs(h).max())
    im0 = axes[0].imshow(J, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    axes[0].axhline(n_d - 0.5, color="k", lw=1, ls="--")
    axes[0].axvline(n_d - 0.5, color="k", lw=1, ls="--")
    plt.colorbar(im0, ax=axes[0], label="J_{ij}")
    axes[0].set_title("Ising couplings J")
    axes[0].set_xlabel("Spin index j")
    axes[0].set_ylabel("Spin index i")

    x = np.arange(len(h))
    colors = ["#2166ac"] * n_d + ["#d62728"] * n_w
    axes[1].bar(x, h, color=colors, alpha=0.8)
    axes[1].axvline(n_d - 0.5, color="k", lw=1.5, ls="--")
    axes[1].set_title("Ising local fields h")
    axes[1].set_xlabel("Spin index")
    axes[1].set_ylabel("h_i")

    plt.suptitle("Ising Hamiltonian — ready for D-Wave / Fixstars Amplify",
                 fontsize=11)
    plt.tight_layout()
    if save:
        path = os.path.join(OUT_DIR, "ising_hamiltonian.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-levels", type=int, default=8,
                        help="Grid levels per parameter (default 8)")
    parser.add_argument("--lam",      type=float, default=PENALTY_LAMBDA,
                        help=f"Constraint penalty λ (default {PENALTY_LAMBDA})")
    parser.add_argument("--compare",  action="store_true",
                        help="Also run classical differential evolution")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    n_d = n_w = args.n_levels
    n_vars = n_d + n_w
    print("=" * 60)
    print("Quantum Annealing Recipe Optimiser — SiC Blade Dicing")
    print(f"  Grid: {n_d} depth × {n_w} width = {n_vars} QUBO variables")
    print(f"  Penalty λ = {args.lam}")
    print("=" * 60)

    # ── Build QUBO ────────────────────────────────────────────────────────────
    Q, depth_grid, width_grid, C = build_qubo(n_d, n_w, lam=args.lam)
    print(f"\nQUBO matrix: {Q.shape}  |Q|_max = {np.abs(Q).max():.3f}")
    plot_qubo_matrix(Q, depth_grid, width_grid)

    # ── Ising mapping ─────────────────────────────────────────────────────────
    J, h, offset = to_ising(Q)
    print(f"Ising: |J|_max={np.abs(J).max():.3f}  |h|_max={np.abs(h).max():.3f}")
    plot_ising_coupling(J, h, n_d, n_w)

    # ── Exact solver ──────────────────────────────────────────────────────────
    if n_vars <= 20:
        print(f"\n--- Brute-force exact solver ({2**n_vars} states) ---")
        x_exact, E_exact = solve_brute_force(Q, n_d, n_w)
        rec_exact = decode_solution(x_exact, n_d, depth_grid, width_grid)
        chip_exact = chipping_model(
            np.array([[rec_exact["cut_depth_um"]]]),
            np.array([[rec_exact["blade_W_um"]]])
        )[0, 0]
        print(f"  Optimal: depth={rec_exact['cut_depth_um']:.1f}µm  "
              f"W={rec_exact['blade_W_um']:.1f}µm  "
              f"chipping={chip_exact:.2f}µm  H={E_exact:.4f}")
    else:
        E_exact = None
        rec_exact = None
        print("  (Brute-force skipped: n_vars > 20)")

    # ── Simulated annealing ───────────────────────────────────────────────────
    print("\n--- Simulated Annealing ---")
    x_sa, E_sa, history = solve_simulated_annealing(Q, n_d, n_w)
    rec_sa = decode_solution(x_sa, n_d, depth_grid, width_grid)
    chip_sa = chipping_model(
        np.array([[rec_sa["cut_depth_um"]]]),
        np.array([[rec_sa["blade_W_um"]]])
    )[0, 0]
    print(f"  SA:     depth={rec_sa['cut_depth_um']:.1f}µm  "
          f"W={rec_sa['blade_W_um']:.1f}µm  "
          f"chipping={chip_sa:.2f}µm  H={E_sa:.4f}")

    if E_exact is not None:
        gap = abs(E_sa - E_exact)
        print(f"  Gap from exact: {gap:.4f}  "
              f"({'OPTIMAL' if gap < 1e-6 else 'suboptimal'})")
        plot_sa_convergence(history, E_exact)

    recipes = {}
    if rec_exact is not None:
        recipes["QUBO-Exact"] = rec_exact
    recipes["QUBO-SA"] = rec_sa

    # ── Classical comparison ──────────────────────────────────────────────────
    if args.compare:
        print("\n--- Classical Differential Evolution ---")
        rec_de = solve_classical_de()
        print(f"  DE:     depth={rec_de['cut_depth_um']:.1f}µm  "
              f"W={rec_de['blade_W_um']:.1f}µm  "
              f"chipping={rec_de['chipping_um']:.2f}µm")
        recipes["Classical DE"] = rec_de

    plot_solution_comparison(recipes, depth_grid, width_grid)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  QUBO size      : {n_vars} variables ({n_d}+{n_w} one-hot)")
    print(f"  Ising spins    : {n_vars}  (D-Wave Advantage supports >5000)")
    print(f"  SA result      : depth={rec_sa['cut_depth_um']:.1f}µm  "
          f"W={rec_sa['blade_W_um']:.1f}µm  chip={chip_sa:.2f}µm")
    if E_exact is not None:
        print(f"  Exact result   : depth={rec_exact['cut_depth_um']:.1f}µm  "
              f"W={rec_exact['blade_W_um']:.1f}µm  chip={chip_exact:.2f}µm")
    print()
    print("  To run on Fixstars Amplify, replace solve_simulated_annealing()")
    print("  with: amplify.BinaryQuadraticModel(Q) → FixstarsClient(token=...)")
    print("  (same Q matrix, no other changes)")


if __name__ == "__main__":
    main()
