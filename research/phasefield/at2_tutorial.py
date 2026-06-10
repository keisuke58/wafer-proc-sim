"""
AT2 Phase-Field Fracture — FEniCSx (dolfinx) Tutorial
======================================================

Self-study guide for Keio / Muramatsu lab thesis prep.
Target: understand the full AT2 formulation before April 2027.

Physics background (Miehe et al. 2010)
----------------------------------------
Total energy functional:
    Ψ[u, d] = ∫_Ω g(d) ψ_e⁺(ε) dΩ  (elastic, tensile)
             + ∫_Ω ψ_e⁻(ε) dΩ       (elastic, compressive; NOT degraded)
             + ∫_Ω Gc/2 [ d²/ℓ + ℓ |∇d|² ] dΩ   (crack surface energy)

    g(d) = (1-d)² + k_res    (degradation function; k_res ~ 1e-6 stability)
    d ∈ [0,1]:  0=intact, 1=fully broken

Governing PDEs (Euler-Lagrange of Ψ):
    (1) Momentum:  div[ g(d) σ(u) ] = 0            in Ω
    (2) Damage:    Gc/ℓ [d - ℓ² Δd] = 2(1-d) H⁺   in Ω
                   ∂d/∂n = 0                         on ∂Ω

    H⁺ = max_{τ≤t} ψ_e⁺(τ)   (history variable — irreversibility)

Staggered (alternating minimisation) scheme:
    Given d_n (previous step):
        Step A: Solve u  with d = d_n  (linear elastic)
        Step B: Solve d  with u fixed  (linear damage)
        Repeat until ||u_k+1 - u_k|| < tol  (usually 1 iteration per load step is enough)

Benchmark: Single-Edge Notch Shear Test (SENST)
    Domain: [0,1]×[0,1] mm², notch from left at mid-height
    Bottom: u=0 (clamped)
    Top: u_x = δ (imposed horizontal displacement)
    Reference: Miehe et al. (2010) IJNME Fig. 10

Requirements
-----------
    pip install fenics-dolfinx  (or use conda-forge)
    pip install pyvista matplotlib numpy

Note: dolfinx ≥ 0.7 required for the API used below.
Run:  python at2_tutorial.py
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

# ── ① Pure-NumPy reference implementation (no FEniCSx needed to run) ──────────
# This lets you understand AT2 on a 1D bar BEFORE installing dolfinx.

def at2_1d_bar(n_elem=200, E=1.0, Gc=1.0, ell=0.05, n_steps=50, u_max=0.02):
    """
    Minimal AT2 phase-field on a 1D bar [0, L=1].
    Imposed displacement u(L) = u_max (ramp).
    Homogeneous damage nucleation → crack at x=L/2 (symmetric).

    Returns: load-displacement curve (u_top, F_react)
    """
    L = 1.0
    dx = L / n_elem
    x = np.linspace(0, L, n_elem + 1)

    u = np.zeros(n_elem + 1)
    d = np.zeros(n_elem + 1)
    H = np.zeros(n_elem + 1)   # history variable (max ψ⁺ seen so far)
    k_res = 1e-6

    # FEM matrices (1D linear, assembled analytically)
    # Stiffness K_uu entry: ∫ g(d) E (du/dx)(dv/dx) dx
    # Damage K_dd entry: ∫ [Gc/ell psi psi + Gc*ell dpsi dpsi] dx
    # where psi = shape function

    load_disp = []

    for step in range(n_steps):
        u_imposed = u_max * (step + 1) / n_steps

        # ── Staggered loop (1 iteration here; converged for small steps) ──
        # Step A: solve u (direct sparse solve)
        K = np.zeros((n_elem + 1, n_elem + 1))
        f = np.zeros(n_elem + 1)
        for i in range(n_elem):
            g_mid = 0.5 * ((1 - d[i])**2 + (1 - d[i+1])**2) / 2 + k_res
            # Actually: g at midpoint = average
            g_mid = ((1 - d[i])**2 + (1 - d[i+1])**2) * 0.5 + k_res
            ke = E * g_mid / dx * np.array([[1, -1], [-1, 1]])
            K[i:i+2, i:i+2] += ke
        # BCs: u[0]=0, u[n_elem]=u_imposed
        K_free = K[1:n_elem, 1:n_elem]
        f_free = -K[1:n_elem, 0] * 0.0 - K[1:n_elem, n_elem] * u_imposed
        u[1:n_elem] = np.linalg.solve(K_free, f_free)
        u[0] = 0.0
        u[n_elem] = u_imposed

        # Elastic strain energy (tensile only; 1D: always positive if ε>0)
        eps = np.diff(u) / dx        # strain at each element midpoint
        psi_plus = 0.5 * E * np.maximum(eps, 0)**2   # tension only
        # Nodal H: average of adjacent elements
        H_node = np.zeros(n_elem + 1)
        H_node[0] = psi_plus[0]
        H_node[n_elem] = psi_plus[-1]
        H_node[1:n_elem] = 0.5 * (psi_plus[:-1] + psi_plus[1:])
        H = np.maximum(H, H_node)   # irreversibility

        # Step B: solve d
        # AT2 PDE (weak form, nodal):
        # ∑ [ Gc/ell * d_j * phi_j*phi_i + Gc*ell * dd_j/dx * dphi/dx ]
        #   = 2*(1-d_j)*H_j*phi_j*phi_i
        # Linearised (fixed H): A*d = b
        A = np.zeros((n_elem + 1, n_elem + 1))
        b = np.zeros(n_elem + 1)
        for i in range(n_elem):
            # mass-like term: Gc/ell * ∫ phi_i phi_j dx  (lumped)
            A[i, i]   += Gc / ell * dx / 2
            A[i+1,i+1]+= Gc / ell * dx / 2
            # stiffness-like term: Gc*ell * ∫ dphi_i/dx dphi_j/dx dx
            A[i:i+2, i:i+2] += Gc * ell / dx * np.array([[1,-1],[-1,1]])
            # RHS: 2*H * ∫ phi dx  (lumped)
            b[i]   += 2 * H[i]   * dx / 2
            b[i+1] += 2 * H[i+1] * dx / 2
        # No BCs on d (Neumann ∂d/∂n=0 naturally satisfied)
        # Add mass to RHS: AT2 b = 2H*phi + Gc/ell * 0 (for AT2: linear in d)
        # Full AT2 equation: (Gc/ell + 2H)*d - Gc*ell*Δd = 2H
        # Add 2*H to diagonal:
        for i in range(n_elem + 1):
            A[i, i] += 2 * H[i] * (dx if 0 < i < n_elem else dx/2)
            b[i]    += 2 * H[i] * (dx if 0 < i < n_elem else dx/2)
        # Clip for safety:
        A += 1e-14 * np.eye(n_elem + 1)
        d = np.linalg.solve(A, b)
        d = np.clip(d, 0, 1)

        # Reaction force at top node
        F_react = sum(E * ((1 - d[i])**2 + k_res) * (u[i+1]-u[i]) / dx
                      for i in range(n_elem))

        load_disp.append((u_imposed, F_react))

    return np.array(load_disp)


def demo_1d():
    """Run 1D demo and plot load-displacement curve."""
    print("Running 1D AT2 bar demo...")
    ld = at2_1d_bar(n_elem=200, E=1.0, Gc=1.0, ell=0.04, n_steps=80, u_max=0.025)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ld[:, 0], ld[:, 1], 'b-', lw=2)
    ax.set_xlabel("Imposed displacement u_top")
    ax.set_ylabel("Reaction force F")
    ax.set_title("AT2 Phase-Field: 1D Bar\n(snap-back after crack nucleation)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig("at2_1d_demo.png", dpi=150)
    print("Saved: at2_1d_demo.png")
    plt.show()


# ── ② FEniCSx 2D implementation skeleton ──────────────────────────────────────
# Run this AFTER installing dolfinx:
#   conda create -n fenicsx -c conda-forge fenics-dolfinx pyvista matplotlib
#   conda activate fenicsx

FENICS_2D_CODE = '''
"""
AT2 Phase-Field Fracture — 2D FEniCSx Implementation
Single-Edge Notch Tension Test (SENT)
Ref: Miehe, Welschinger, Hofacker (2010) IJNME
"""
import numpy as np
from dolfinx import mesh, fem, log
from dolfinx.fem import (FunctionSpace, Function, Constant,
                          VectorFunctionSpace, form)
from dolfinx.fem.petsc import (LinearProblem, assemble_matrix,
                                 assemble_vector, apply_lifting, set_bc)
from dolfinx.mesh import create_rectangle, CellType
from mpi4py import MPI
import ufl
from petsc4py import PETSc

log.set_log_level(log.LogLevel.WARNING)

# ── Mesh ──────────────────────────────────────────────────────────────────────
# [0,1] x [0,1] mm², 100×100 quad mesh
Lx, Ly = 1.0, 1.0
nx, ny = 80, 80
domain = create_rectangle(MPI.COMM_WORLD,
                            [[0.0, 0.0], [Lx, Ly]],
                            [nx, ny], CellType.quadrilateral)
dim = domain.topology.dim

# ── Function spaces ───────────────────────────────────────────────────────────
V  = VectorFunctionSpace(domain, ("Lagrange", 1))   # displacement
W  = FunctionSpace(domain, ("Lagrange", 1))          # damage

u  = Function(V, name="displacement")
d  = Function(W, name="damage")
d_prev = Function(W, name="damage_prev")           # for irreversibility
H  = Function(W, name="history")                   # max tensile energy

# ── Material parameters ───────────────────────────────────────────────────────
E_mod  = 210e3   # Young's modulus [MPa]
nu     = 0.3     # Poisson ratio
Gc     = 2.7e-3  # Fracture energy [kJ/m² = kN/mm]
ell    = 0.0075  # Regularisation length [mm]  (~ 3-5 elements)
k_res  = 1e-6    # Residual stiffness

mu_lame  = E_mod / (2 * (1 + nu))
lam_lame = E_mod * nu / ((1 + nu) * (1 - 2*nu))

# ── Kinematics ────────────────────────────────────────────────────────────────
def eps(v):
    return ufl.sym(ufl.grad(v))

def sigma_0(v):
    return lam_lame * ufl.tr(eps(v)) * ufl.Identity(dim) + 2 * mu_lame * eps(v)

# Spectral split for tension/compression (Miehe 2010)
# Simplified: use volumetric-deviatoric split for isotropic case
def psi_plus(v):
    """Tensile elastic energy density (Amor et al. 2009 vol-dev split)."""
    treps = ufl.tr(eps(v))
    K_bulk = lam_lame + 2*mu_lame/dim
    psi_vol = 0.5 * K_bulk * (ufl.max_value(treps, 0))**2
    eps_dev = eps(v) - (treps/dim)*ufl.Identity(dim)
    psi_dev = mu_lame * ufl.inner(eps_dev, eps_dev)
    return psi_vol + psi_dev

def psi_minus(v):
    """Compressive elastic energy density."""
    treps = ufl.tr(eps(v))
    K_bulk = lam_lame + 2*mu_lame/dim
    psi_vol = 0.5 * K_bulk * (ufl.min_value(treps, 0))**2
    return psi_vol

# Degradation
def g_deg(d_val):
    return (1 - d_val)**2 + k_res

# ── Variational forms ─────────────────────────────────────────────────────────
du = ufl.TrialFunction(V);  v_test = ufl.TestFunction(V)
dphi = ufl.TrialFunction(W); w_test = ufl.TestFunction(W)

# Step A: displacement (linear, bilinear form)
a_u = ufl.inner(g_deg(d) * sigma_0(du), eps(v_test)) * ufl.dx
L_u = ufl.inner(ufl.Constant(domain, (0.0, 0.0)), v_test) * ufl.dx

# Step B: damage (linear; H fixed from previous step)
a_d = (Gc/ell * dphi * w_test
       + Gc * ell * ufl.dot(ufl.grad(dphi), ufl.grad(w_test))
       + 2 * H * dphi * w_test) * ufl.dx
L_d = 2 * H * w_test * ufl.dx

# ── Boundary conditions ───────────────────────────────────────────────────────
def bottom(x): return np.isclose(x[1], 0.0)
def top(x):    return np.isclose(x[1], Ly)
def notch_tip(x):
    # Notch: x ∈ [0, 0.5], y = 0.5 (half-height)
    return np.logical_and(np.isclose(x[1], Ly/2, atol=1.5/ny),
                           x[0] <= Lx/2 + 1.5/nx)

# Clamp bottom (u=0)
bc_bot = fem.dirichletbc(np.zeros(2), fem.locate_dofs_geometrical(V, bottom), V)

# Top: imposed vertical displacement (tension test)
u_top_val = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))

bc_top = fem.dirichletbc(u_top_val,
                          fem.locate_dofs_geometrical(V, top), V)

bcs_u = [bc_bot, bc_top]

# ── Staggered time loop ───────────────────────────────────────────────────────
n_steps = 100
u_max_y = 0.01  # mm total displacement

load_disp = []

for step in range(n_steps):
    delta = u_max_y * (step + 1) / n_steps
    u_top_val.value = (0.0, delta)

    # Step A: Solve displacement
    problem_u = LinearProblem(a_u, L_u, bcs=bcs_u,
                               petsc_options={"ksp_type": "preonly",
                                              "pc_type": "lu"})
    u = problem_u.solve()

    # Update history variable H = max(H, psi_plus(u))
    psi_n = fem.Expression(psi_plus(u), W.element.interpolation_points())
    psi_fn = Function(W)
    psi_fn.interpolate(psi_n)
    H.x.array[:] = np.maximum(H.x.array, psi_fn.x.array)

    # Step B: Solve damage
    problem_d = LinearProblem(a_d, L_d, bcs=[],
                               petsc_options={"ksp_type": "preonly",
                                              "pc_type": "lu"})
    d_new = problem_d.solve()

    # Enforce irreversibility: d >= d_prev
    d.x.array[:] = np.maximum(d_new.x.array, d_prev.x.array)
    d_prev.x.array[:] = d.x.array.copy()

    # Reaction force at top (integrate traction)
    n_vec = ufl.FacetNormal(domain)
    F_react = fem.assemble_scalar(form(
        ufl.inner(g_deg(d) * sigma_0(u) * n_vec,
                  ufl.as_vector([0, 1])) * ufl.ds
    ))
    load_disp.append((delta, F_react))

    if step % 10 == 0:
        d_max = d.x.array.max()
        print(f"  step {step:3d}: δ={delta:.5f}mm  F={F_react:.4f}kN  d_max={d_max:.3f}")

print("Done. Load-disp data in load_disp list.")
'''

# ── ③ Anisotropic AT2 extension (for SiC / Ga₂O₃) ───────────────────────────
ANISOTROPIC_EXTENSION = '''
# Extension: Anisotropic fracture energy (Teichtmeister et al. 2017)
# Replace isotropic Gc surface energy term with anisotropic version:
#
#   Gc_eff(∇d) = Gc * (1 + β · (n⊗n : ∇d⊗∇d / |∇d|²))
#
# where n = preferred crack plane normal (cleavage plane normal)
# β = anisotropy ratio (β=0 → isotropic; β>0 → preferred crack direction)
#
# For 4H-SiC: n = [0,0,1] (basal plane normal), β ≈ 0.5
# For β-Ga₂O₃: n = [1,0,0] (100 plane normal), β ≈ 1.5 (strong anisotropy)
#
# Fractional Energy:
#   Gamma_crack(d) = ∫_Ω Gc/2 [d²/ℓ + ℓ A(n):∇d⊗∇d] dΩ
#
# where A(n) = I + β (I - n⊗n)   (penalises cracks NOT on preferred plane)
#
# Implementation in FEniCSx:
#   n_crack = ufl.as_vector([1.0, 0.0])  # (100) cleavage normal in 2D
#   beta = 1.5
#   A_anis = ufl.Identity(2) + beta * (ufl.outer(n_crack, n_crack))
#   a_d_anis = (Gc/ell * dphi * w_test
#               + Gc * ell * ufl.inner(A_anis * ufl.grad(dphi), ufl.grad(w_test))
#               + 2 * H * dphi * w_test) * ufl.dx
#
# This is the KEY extension for the Keio / Muramatsu thesis:
#   → Infer β and Gc from experimental chipping data via TMCMC
#   → TMCMC likelihood: L(Gc, β, ℓ | {x_chip, F_chip}) ∝ exp(-||model - data||²)
'''

# ── ④ TMCMC interface skeleton ────────────────────────────────────────────────
TMCMC_INTERFACE = '''
"""
Interface: FEniCSx forward model ↔ TMCMC (JAX, from LUH thesis)
"""
import numpy as np
import subprocess

def run_fenics_forward(params: dict) -> dict:
    """
    Wraps FEniCSx AT2 simulation as a callable for TMCMC likelihood.

    params: {"Gc": float, "ell": float, "beta_aniso": float}
    returns: {"peak_force_kN": float, "crack_angle_deg": float, "load_disp": array}
    """
    # Option 1: in-process (dolfinx Python API — cleanest)
    # from at2_fenics_2d import solve_at2  # your FEniCSx module
    # result = solve_at2(**params)

    # Option 2: subprocess (if using mpirun)
    cmd = ["python", "at2_fenics_2d.py",
           f"--Gc={params['Gc']}", f"--ell={params['ell']}",
           f"--beta={params['beta_aniso']}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    result = eval(out.stdout.strip())  # parse dict output
    return result


def log_likelihood(params: np.ndarray, obs_data: dict) -> float:
    """
    TMCMC likelihood function.
    params: [log_Gc, log_ell, log_beta]
    obs_data: {"peak_force": float, "crack_angle": float, "sigma_noise": float}

    Returns: log p(data | params)
    """
    Gc   = np.exp(params[0])
    ell  = np.exp(params[1])
    beta = np.exp(params[2])

    if ell < 1e-4 or ell > 0.5 or Gc < 1e-4 or beta < 0 or beta > 5:
        return -np.inf

    sim = run_fenics_forward({"Gc": Gc, "ell": ell, "beta_aniso": beta})

    sigma = obs_data["sigma_noise"]
    ll  = -0.5 * ((sim["peak_force_kN"] - obs_data["peak_force"]) / sigma)**2
    ll += -0.5 * ((sim["crack_angle_deg"] - obs_data["crack_angle"]) / 10.0)**2
    return float(ll)

# Log prior: flat in log-space (Jeffrey-like)
def log_prior(params: np.ndarray) -> float:
    """Weakly informative prior: Gc ~ LogNormal(log(2.7e-3), 0.5), etc."""
    Gc   = np.exp(params[0])
    ell  = np.exp(params[1])
    beta = np.exp(params[2])
    lp  = -0.5 * ((params[0] - np.log(2.7e-3)) / 0.5)**2   # Gc
    lp += -0.5 * ((params[1] - np.log(0.0075)) / 0.5)**2   # ell
    lp += -0.5 * ((params[2] - np.log(1.0))    / 1.0)**2   # beta
    return lp
'''


def main():
    demo_1d()
    print("\n=== FEniCSx 2D skeleton (copy to separate file after installing dolfinx) ===")
    print(FENICS_2D_CODE[:300], "...(truncated)")
    print("\n=== Anisotropic extension for SiC/Ga₂O₃ ===")
    print(ANISOTROPIC_EXTENSION[:300], "...(truncated)")
    print("\n=== TMCMC interface skeleton ===")
    print(TMCMC_INTERFACE[:300], "...(truncated)")
    print("\nFull code is in this file — copy FENICS_2D_CODE/ANISOTROPIC_EXTENSION/TMCMC_INTERFACE strings to .py files")


if __name__ == "__main__":
    main()
