"""
Toy implementations of 5 landmark high-energy physics papers.

Papers:
  [1] ADD  — Arkani-Hamed, Dimopoulos, Dvali (1998) hep-ph/9803315
  [2] RS   — Randall & Sundrum (1999) hep-ph/9905221
  [3] AdS  — Maldacena (1997) hep-th/9711200  (AdS/CFT)
  [4] LQG  — Rovelli & Smolin (1994) gr-qc/9411005
  [5] QSup — Arute et al. / Google (2019) arXiv:1905.10526

Run:  python physics_toy_models.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Constants ─────────────────────────────────────────────────────────────────
hbar = 1.055e-34   # J·s
c    = 3e8         # m/s
G    = 6.674e-11   # m³/(kg·s²)
l_P  = np.sqrt(hbar * G / c**3)   # Planck length ≈ 1.6e-35 m
gamma_Immirzi = 0.2375             # Barbero-Immirzi parameter (LQG)


# ═════════════════════════════════════════════════════════════════════════════
# [1]  ADD — Extra Dimensions: gravitational potential
# ═════════════════════════════════════════════════════════════════════════════
def add_potential(r, n_extra, R_extra=1e-6):
    """
    Gravitational potential in ADD model.
      r      : distance [m]
      n_extra: number of extra dimensions
      R_extra: compactification radius [m]

    V(r) ~ 1/r         for r >> R_extra  (4D Newton)
    V(r) ~ 1/r^{1+n}   for r << R_extra  (4+n dimensional)
    """
    V = np.where(
        r > R_extra,
        1.0 / r,
        (R_extra**n_extra) / r**(1 + n_extra),
    )
    return V / V[0]   # normalize


# ═════════════════════════════════════════════════════════════════════════════
# [2]  Randall-Sundrum — Warped Extra Dimension
# ═════════════════════════════════════════════════════════════════════════════
def rs_warp_factor(y, k=1.0, r_c=np.pi):
    """Warp factor a(y) = exp(-k|y|), y ∈ [-pi*r_c, pi*r_c]."""
    return np.exp(-k * np.abs(y))


def rs_kk_masses(n_max=8, k=1.0, r_c=np.pi, kr_c=11.27):
    """
    KK graviton masses in RS1:
      m_n = x_n * k * exp(-k*pi*r_c)
    where x_n are zeros of Bessel J_1.
    Hierarchy suppression factor: exp(-k*pi*r_c) ≈ exp(-11.27) ≈ 10^{-5}
    """
    from scipy.special import jn_zeros
    x_n = jn_zeros(1, n_max)
    suppression = np.exp(-kr_c)
    return x_n * k * suppression


# ═════════════════════════════════════════════════════════════════════════════
# [3]  Maldacena — AdS/CFT: bulk-to-boundary propagator & 2-point function
# ═════════════════════════════════════════════════════════════════════════════
def ads_bulk_boundary(z, x, x0=0.0, Delta=2.0):
    """
    Bulk-to-boundary propagator in Euclidean AdS_3:
      K(z, x; x') = (z / (z² + (x-x')²))^Δ
    z > 0 : radial AdS coordinate (z→0: boundary)
    """
    return (z / (z**2 + (x - x0)**2)) ** Delta


def cft_two_point(x, Delta=2.0):
    """CFT 2-point function: <O(x)O(0)> ∝ 1/|x|^{2Δ}"""
    return 1.0 / np.abs(x) ** (2 * Delta)


# ═════════════════════════════════════════════════════════════════════════════
# [4]  LQG — Rovelli-Smolin: area spectrum from spin networks
# ═════════════════════════════════════════════════════════════════════════════
def lqg_area_spectrum(j_max=5):
    """
    Area eigenvalues of spin-network edges:
      A_j = 8π γ l_P² √(j(j+1))
    j = 0, 1/2, 1, 3/2, ...  (half-integer spin)
    Returns j values and corresponding area eigenvalues in units of l_P².
    """
    j_vals = np.arange(0, j_max + 0.5, 0.5)
    A = 8 * np.pi * gamma_Immirzi * np.sqrt(j_vals * (j_vals + 1))
    return j_vals, A   # A in units of l_P²


# ═════════════════════════════════════════════════════════════════════════════
# [5]  Google Quantum Supremacy — Random quantum circuit simulation
# ═════════════════════════════════════════════════════════════════════════════
def random_quantum_circuit(n_qubits=4, depth=10, seed=42):
    """
    Toy simulation of a random quantum circuit on n_qubits.
    State vector evolves under random single-qubit unitaries + CNOT layers.
    Returns final state probabilities (|ψ|²).

    Real Google experiment: 53 qubits, depth 20 — classical simulation
    would take ~10,000 years. Here n=4 is tractable in numpy.
    """
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0   # |0...0>

    def random_unitary(rng):
        """Haar-random 2×2 unitary via QR decomposition."""
        Z = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))) / np.sqrt(2)
        Q, R = np.linalg.qr(Z)
        phases = np.diag(R) / np.abs(np.diag(R))
        return Q * phases

    def apply_single_qubit(state, U, qubit, n):
        """Apply 2×2 gate U to `qubit` in n-qubit state vector."""
        state = state.reshape([2] * n)
        state = np.tensordot(U, state, axes=([1], [qubit]))
        state = np.moveaxis(state, 0, qubit)
        return state.reshape(dim)

    def apply_cnot(state, ctrl, tgt, n):
        """Apply CNOT gate."""
        state = state.reshape([2] * n)
        # swap |1>_ctrl ⊗ |0>_tgt ↔ |1>_ctrl ⊗ |1>_tgt
        idx0 = [slice(None)] * n
        idx1 = [slice(None)] * n
        idx0[ctrl] = 1; idx0[tgt] = 0
        idx1[ctrl] = 1; idx1[tgt] = 1
        tmp = state[tuple(idx0)].copy()
        state[tuple(idx0)] = state[tuple(idx1)]
        state[tuple(idx1)] = tmp
        return state.reshape(dim)

    for _ in range(depth):
        for q in range(n_qubits):
            U = random_unitary(rng)
            state = apply_single_qubit(state, U, q, n_qubits)
        for q in range(n_qubits - 1):
            state = apply_cnot(state, q, q + 1, n_qubits)

    probs = np.abs(state)**2
    return probs


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════
def plot_all():
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Toy Models: 5 Landmark Physics Papers", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── [1] ADD ───────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    r = np.logspace(-8, -4, 500)
    for n in [0, 2, 6]:
        label = f"n={n} ({'4D Newton' if n == 0 else 'extra dim'})"
        ax1.loglog(r * 1e6, add_potential(r, n_extra=n, R_extra=1e-6), label=label)
    ax1.axvline(1.0, ls="--", color="gray", lw=0.8, label="R_extra=1μm")
    ax1.set_xlabel("r [μm]")
    ax1.set_ylabel("V(r) [normalized]")
    ax1.set_title("[1] ADD: Extra-dim gravity")
    ax1.legend(fontsize=7)

    # ── [2] RS warp factor ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    y = np.linspace(-np.pi, np.pi, 300)
    for k in [0.5, 1.0, 2.0]:
        ax2.plot(y, rs_warp_factor(y, k=k), label=f"k={k}")
    ax2.set_xlabel("y (extra dimension)")
    ax2.set_ylabel("a(y) = exp(-k|y|)")
    ax2.set_title("[2] RS: Warp factor")
    ax2.legend(fontsize=8)

    # ── [3] AdS/CFT bulk-to-boundary propagator ────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.linspace(-5, 5, 300)
    for z in [0.1, 0.5, 1.0, 2.0]:
        ax3.plot(x, ads_bulk_boundary(z, x, Delta=2.0), label=f"z={z}")
    ax3.set_xlabel("x (boundary coordinate)")
    ax3.set_ylabel("K(z, x; 0)")
    ax3.set_title("[3] AdS/CFT: Bulk→boundary propagator")
    ax3.legend(fontsize=8)

    # ── [3b] CFT 2-point function ──────────────────────────────────────────
    ax3b = fig.add_subplot(gs[1, 0])
    x_pos = np.linspace(0.1, 5, 300)
    for Delta in [1.0, 1.5, 2.0, 3.0]:
        ax3b.semilogy(x_pos, cft_two_point(x_pos, Delta), label=f"Δ={Delta}")
    ax3b.set_xlabel("|x|")
    ax3b.set_ylabel("<O(x)O(0)>")
    ax3b.set_title("[3] CFT 2-point function")
    ax3b.legend(fontsize=8)

    # ── [4] LQG area spectrum ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    j_vals, A_vals = lqg_area_spectrum(j_max=5)
    ax4.stem(j_vals, A_vals, basefmt=" ")
    ax4.set_xlabel("Spin j")
    ax4.set_ylabel("Area eigenvalue [8πγ l_P² units]")
    ax4.set_title("[4] LQG: Area spectrum (Rovelli-Smolin)")

    # ── [5] Quantum supremacy circuit output ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    probs = random_quantum_circuit(n_qubits=4, depth=20, seed=2019)
    states = [f"|{i:04b}>" for i in range(len(probs))]
    ax5.bar(range(len(probs)), probs, color="steelblue")
    ax5.set_xticks(range(len(probs)))
    ax5.set_xticklabels(states, rotation=90, fontsize=7)
    ax5.set_ylabel("Probability")
    ax5.set_title("[5] Google: Random circuit output\n(n=4 qubits, depth=20)")
    ax5.axhline(1 / len(probs), ls="--", color="red", lw=0.8, label="uniform")
    ax5.legend(fontsize=7)

    out = "results/physics_toy_models.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    # ── Print key numbers ────────────────────────────────────────────────────
    print("=" * 60)
    print("Planck length l_P =", f"{l_P:.2e}", "m")

    print("\n[2] RS KK graviton masses (normalized to k):")
    try:
        masses = rs_kk_masses()
        print("  m_n =", np.round(masses, 4))
    except ImportError:
        print("  scipy not available — skipping KK masses")

    print("\n[4] LQG area eigenvalues [8πγ l_P² units]:")
    j_vals, A_vals = lqg_area_spectrum()
    for j, A in zip(j_vals, A_vals):
        print(f"  j={j:.1f}  →  A={A:.4f}")

    print("\n[5] Quantum circuit output probabilities:")
    probs = random_quantum_circuit(n_qubits=4, depth=20, seed=2019)
    print("  Sum =", round(probs.sum(), 6), "(should be 1.0)")
    print("  Max prob =", round(probs.max(), 4), f"  (uniform would be {1/len(probs):.4f})")

    print("\nPlotting...")
    plot_all()
