"""
at2_simulator_2d.py  —  2D plane-strain AT2 phase-field fracture

Extends at2_simulator.py to 2D, adding:
  - Anisotropic Gc tensor: A = I + beta*(c⊗c), c = cleavage-plane normal
  - Crack deflection angle φ as a continuous observation (Phase 2 thesis)

Solver: finite-difference on a rectangular grid, scipy.sparse.linalg.spsolve.
No FEniCS dependency — prototype before the Keio M1 FEniCS implementation.

Physical setup:
  - Domain: [0,Lx] × [0,Ly]
  - Mode I loading: u_y applied at y=Ly, fixed at y=0
  - Horizontal pre-notch from x=0 to x=notch_x*Lx at y=Ly/2
  - Crack propagates from notch tip; deflects toward cleavage plane if beta<0

Key new observable vs 1D:
  - φ (crack deflection angle): angle of crack path relative to horizontal [deg]
  - Gaussian observation model: φ_obs ~ N(φ_sim(params), sigma_phi²)
  - Unlike binary sign-only (Phase 1), φ is continuous → Gc0 AND beta identifiable
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ─────────────────────────────────────────────────────────────────────────────
#  Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PFParams2D:
    Gc:   float          # isotropic fracture energy [N/m]
    ell:  float          # phase-field length scale [m]
    E:    float          # Young's modulus [Pa]
    nu:   float = 0.25   # Poisson ratio (plane-strain)
    beta: float = 0.0    # anisotropy: Gc_eff(n)=Gc*(1+beta*(n·c)²)
    theta_cut: float = 0.0  # cut angle relative to cleavage plane [rad]

    def cleavage_normal(self) -> np.ndarray:
        """Unit normal to the cleavage plane in the (x,y) cutting plane."""
        # Cleavage plane normal rotates with cut angle:
        # theta_cut=0 → c=(0,1) → cut along cleavage, Gc_eff=Gc (no penalty)
        # theta_cut=pi/2 → c=(1,0) → cut across cleavage, Gc_eff=Gc*(1+beta)
        return np.array([-np.sin(self.theta_cut), np.cos(self.theta_cut)])

    def anisotropy_tensor(self) -> np.ndarray:
        """2×2 anisotropy tensor A = I + beta*(c⊗c)."""
        c = self.cleavage_normal()
        return np.eye(2) + self.beta * np.outer(c, c)


@dataclass
class SimConfig2D:
    Lx:          float = 1.0
    Ly:          float = 0.5
    nx:          int   = 80    # nodes in x
    ny:          int   = 40    # nodes in y
    n_steps:     int   = 100
    u_max_factor: float = 3.0  # u_max = u_max_factor * ell
    kappa_res:   float = 1e-6  # residual stiffness
    notch_y_frac: float = 0.5  # notch at y = Ly * notch_y_frac
    notch_x_frac: float = 0.30 # notch from x=0 to x=Lx*notch_x_frac


@dataclass
class ForwardResult2D:
    U:           np.ndarray   # load displacement steps (n_steps+1,)
    F:           np.ndarray   # reaction force steps (n_steps+1,)
    d_field:     np.ndarray   # final damage field (ny, nx)
    init_U:      float        # displacement at crack initiation
    peak_F:      float        # peak reaction force
    crack_angle: float        # crack deflection angle [deg], 0=horizontal
    converged:   bool


# ─────────────────────────────────────────────────────────────────────────────
#  2D plane-strain elasticity — full (u_x, u_y) FD solver
#
#  Equilibrium (no body forces, spatially varying degradation g):
#    x: ∂/∂x[(λ+2μ)g ε_xx] + ∂/∂y[μg γ_xy] + ∂/∂x[gλ ε_yy] = 0
#    y: ∂/∂y[(λ+2μ)g ε_yy] + ∂/∂x[μg γ_xy] + ∂/∂y[gλ ε_xx] = 0
#
#  DOF ordering: for node k=i+j*nx, DOF 2k=u_x, DOF 2k+1=u_y.
#  BCs: bottom fixed (u_x=u_y=0), top Mode-I (u_x=0, u_y=u_applied),
#       sides Neumann (free traction).
#
#  The softened notch (g≈κ) creates a proper Mode-I stress concentration
#  at the notch tip that drives directional crack propagation.
# ─────────────────────────────────────────────────────────────────────────────

def _lame(params: PFParams2D) -> tuple[float, float]:
    E, nu = params.E, params.nu
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = E / (2.0 * (1.0 + nu))
    return lam, mu


def _dof(i: int, j: int, c: int, nx: int) -> int:
    """DOF index: c=0 → u_x, c=1 → u_y."""
    return 2 * (i + j * nx) + c


def _build_full_elasticity_matrix(g: np.ndarray, params: PFParams2D,
                                   cfg: SimConfig2D) -> sp.csr_matrix:
    """
    Full 2D plane-strain FD elasticity with (u_x, u_y) DOFs.

    Neumann BCs use the flux-based formulation: zero-flux faces contribute
    nothing.  For each direction, only active (non-boundary) faces are
    included, ensuring row sums = 0 and exact satisfaction of the constant
    solution on uniform meshes.

    Cross terms (λ+μ)·g·∂²u/∂x∂y vanish at Neumann boundary nodes.
    """
    lam, mu = _lame(params)
    nx, ny = cfg.nx, cfg.ny
    hx = cfg.Lx / (nx - 1)
    hy = cfg.Ly / (ny - 1)
    M  = 2 * nx * ny

    rows_l, cols_l, vals_l = [], [], []

    def add(r, c, v):
        rows_l.append(r); cols_l.append(c); vals_l.append(v)

    def _face_term(row, self_c, nbr_c, coeff, g_face, h2):
        """Add ∂/∂ξ[coeff·g_face·∂u/∂ξ] for ONE face: diagonal and neighbour."""
        add(row, self_c, -coeff * g_face / h2)
        add(row, nbr_c,  coeff * g_face / h2)

    for j in range(ny):
        for i in range(nx):
            ip = i + 1 if i < nx - 1 else i   # clamped (unused for Neumann)
            im = i - 1 if i > 0      else i
            jp = j + 1 if j < ny - 1 else j
            jm = j - 1 if j > 0      else j

            # Face-averaged g values
            gxp = 0.5*(g[j, i] + g[j, min(i+1,nx-1)])   # right face  (active if i<nx-1)
            gxm = 0.5*(g[j, i] + g[j, max(i-1,0)])       # left  face  (active if i>0)
            gyp = 0.5*(g[j, i] + g[min(j+1,ny-1), i])    # top   face  (active if j<ny-1)
            gym = 0.5*(g[j, i] + g[max(j-1,0),    i])    # bot   face  (active if j>0)
            gc  = g[j, i]

            at_xL = (i == 0);      at_xR = (i == nx - 1)
            at_yB = (j == 0);      at_yT = (j == ny - 1)
            # Cross terms vanish at any Neumann boundary node
            cross_ok = not (at_xL or at_xR or at_yB or at_yT)

            rx = _dof(i, j, 0, nx)   # x-equation DOF
            ry = _dof(i, j, 1, nx)   # y-equation DOF

            a = lam + 2 * mu

            # ── x-equilibrium ──────────────────────────────────────────────
            # (λ+2μ) ∂/∂x[g ∂u_x/∂x]  — only active faces
            if not at_xL:
                _face_term(rx, _dof(i,j,0,nx), _dof(im,j,0,nx), a, gxm, hx**2)
            if not at_xR:
                _face_term(rx, _dof(i,j,0,nx), _dof(ip,j,0,nx), a, gxp, hx**2)

            # μ ∂/∂y[g ∂u_x/∂y]
            if not at_yB:
                _face_term(rx, _dof(i,j,0,nx), _dof(i,jm,0,nx), mu, gym, hy**2)
            if not at_yT:
                _face_term(rx, _dof(i,j,0,nx), _dof(i,jp,0,nx), mu, gyp, hy**2)

            # (λ+μ) g ∂²u_y/∂x∂y
            if cross_ok:
                cc = (lam + mu) * gc / (4.0 * hx * hy)
                add(rx, _dof(ip, jp, 1, nx),  cc)
                add(rx, _dof(im, jm, 1, nx),  cc)
                add(rx, _dof(ip, jm, 1, nx), -cc)
                add(rx, _dof(im, jp, 1, nx), -cc)

            # ── y-equilibrium ──────────────────────────────────────────────
            # μ ∂/∂x[g ∂u_y/∂x]
            if not at_xL:
                _face_term(ry, _dof(i,j,1,nx), _dof(im,j,1,nx), mu, gxm, hx**2)
            if not at_xR:
                _face_term(ry, _dof(i,j,1,nx), _dof(ip,j,1,nx), mu, gxp, hx**2)

            # (λ+2μ) ∂/∂y[g ∂u_y/∂y]
            if not at_yB:
                _face_term(ry, _dof(i,j,1,nx), _dof(i,jm,1,nx), a, gym, hy**2)
            if not at_yT:
                _face_term(ry, _dof(i,j,1,nx), _dof(i,jp,1,nx), a, gyp, hy**2)

            # (λ+μ) g ∂²u_x/∂x∂y
            if cross_ok:
                cc = (lam + mu) * gc / (4.0 * hx * hy)
                add(ry, _dof(ip, jp, 0, nx),  cc)
                add(ry, _dof(im, jm, 0, nx),  cc)
                add(ry, _dof(ip, jm, 0, nx), -cc)
                add(ry, _dof(im, jp, 0, nx), -cc)

    return sp.csr_matrix((vals_l, (rows_l, cols_l)), shape=(M, M))


def _solve_displacement_2d(d: np.ndarray, u_applied: float,
                            params: PFParams2D, cfg: SimConfig2D,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve for (u_x, u_y) given damage and applied top displacement.

    BCs (Mode I roller-grip):
      Bottom (j=0):          u_y=0  (normal constraint, roller)
      Top    (j=ny-1):       u_y=u_applied
      One corner (0,0):      u_x=0  (eliminates rigid-body x-translation)
      Left/Right/top-ux/bottom-ux:  Neumann (free Poisson contraction)
    """
    nx, ny = cfg.nx, cfg.ny
    M = 2 * nx * ny

    g = (1.0 - d) ** 2 + cfg.kappa_res
    K = _build_full_elasticity_matrix(g, params, cfg).tolil()
    rhs = np.zeros(M)

    # u_y=0 at bottom
    for i in range(nx):
        r = _dof(i, 0, 1, nx)
        K[r, :] = 0; K[r, r] = 1.0; rhs[r] = 0.0

    # u_y=u_applied at top
    for i in range(nx):
        r = _dof(i, ny - 1, 1, nx)
        K[r, :] = 0; K[r, r] = 1.0; rhs[r] = u_applied

    # u_x=0 at notch-level center — pins rigid-body x-translation while
    # preserving Mode-I antisymmetry (u_x=0 on the crack symmetry plane).
    notch_j = int(round(cfg.notch_y_frac * (ny - 1)))
    r = _dof(nx // 2, notch_j, 0, nx)
    K[r, :] = 0; K[r, r] = 1.0; rhs[r] = 0.0

    u_flat = spla.spsolve(K.tocsr(), rhs)
    u_x = u_flat[0::2].reshape(ny, nx)
    u_y = u_flat[1::2].reshape(ny, nx)
    return u_x, u_y


def _compute_H_field(u_x: np.ndarray, u_y: np.ndarray,
                     params: PFParams2D, cfg: SimConfig2D,
                     d: np.ndarray | None = None) -> np.ndarray:
    """
    Full plane-strain elastic energy density ψ_e⁺:
      ψ_e = (λ+2μ)/2 ε_xx² + (λ+2μ)/2 ε_yy² + λ ε_xx ε_yy + 2μ ε_xy²

    If d is provided, ε_yy is computed with crack-aware one-sided differences
    to avoid measuring crack-face opening at nodes adjacent to the crack:
      - Node just above cracked row (d[j-1] ≥ 0.99): use forward diff (j, j+1)
      - Node just below cracked row (d[j+1] ≥ 0.99): use backward diff (j-1, j)
    """
    hx = cfg.Lx / (cfg.nx - 1)
    hy = cfg.Ly / (cfg.ny - 1)
    lam, mu = _lame(params)

    eps_xx = np.gradient(u_x, hx, axis=1)
    eps_yy = np.gradient(u_y, hy, axis=0)   # central diff (default)
    eps_xy = 0.5 * (np.gradient(u_x, hy, axis=0) + np.gradient(u_y, hx, axis=1))

    if d is not None:
        cracked = d >= 0.99
        ny = cfg.ny
        # Vectorised one-sided ε_yy near crack faces.
        # Avoids measuring crack-face opening via central diff crossing d≥0.99.
        # lower neighbor (j-1) cracked → forward diff (j, j+1) for interior rows
        mask_lo = cracked[:-2, :] & ~cracked[1:-1, :]   # shape (ny-2, nx)
        eps_yy[1:-1][mask_lo] = (u_y[2:][mask_lo] - u_y[1:-1][mask_lo]) / hy
        # upper neighbor (j+1) cracked → backward diff (j-1, j)
        mask_hi = cracked[2:, :] & ~cracked[1:-1, :]
        eps_yy[1:-1][mask_hi] = (u_y[1:-1][mask_hi] - u_y[:-2][mask_hi]) / hy

    psi = (0.5 * (lam + 2*mu) * (eps_xx**2 + eps_yy**2)
           + lam * eps_xx * eps_yy
           + 2.0 * mu * eps_xy**2)
    return np.maximum(psi, 0.0)


def _reaction_force(u_x: np.ndarray, u_y: np.ndarray, d: np.ndarray,
                    params: PFParams2D, cfg: SimConfig2D) -> float:
    """Integrate degraded σ_yy over bottom boundary."""
    hx = cfg.Lx / (cfg.nx - 1)
    hy = cfg.Ly / (cfg.ny - 1)
    lam, mu = _lame(params)
    eps_xx = np.gradient(u_x, hx, axis=1)[0, :]
    eps_yy = np.gradient(u_y, hy, axis=0)[0, :]
    g      = (1.0 - d[0, :]) ** 2 + cfg.kappa_res
    sig_yy = g * ((lam + 2*mu) * eps_yy + lam * eps_xx)
    return float(np.sum(sig_yy) * hx)


# ─────────────────────────────────────────────────────────────────────────────
#  2D anisotropic AT2 damage solver
# ─────────────────────────────────────────────────────────────────────────────

def _build_2d_laplacian(nx: int, ny: int, hx: float, hy: float,
                        A: np.ndarray) -> sp.csr_matrix:
    """
    Sparse matrix for anisotropic 2D Laplacian: ∇·(A·∇d).

    A = [[Axx, Axy],[Axy, Ayy]] is the 2×2 anisotropy tensor (constant).
    Neumann BCs (∂d/∂n=0) via ghost-node reflection at all boundaries.

    Node ordering: k = i + j*nx  (i=x-index, j=y-index)
    """
    N    = nx * ny
    Axx  = A[0, 0]
    Ayy  = A[1, 1]
    Axy  = A[0, 1]   # = A[1,0]

    diag_val  = -2.0 * Axx / hx**2 - 2.0 * Ayy / hy**2
    cx        = Axx / hx**2   # x-neighbor coefficient
    cy        = Ayy / hy**2   # y-neighbor coefficient
    cxy       = Axy / (2.0 * hx * hy)  # cross-term coefficient

    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    for j in range(ny):
        for i in range(nx):
            k = i + j * nx

            # Diagonal
            add(k, k, diag_val)

            # x-neighbors (with Neumann ghost-node at boundary)
            ip = i + 1 if i < nx - 1 else i - 1   # ghost: reflect
            im = i - 1 if i > 0      else i + 1
            add(k, ip + j * nx, cx)
            add(k, im + j * nx, cx)

            # y-neighbors
            jp = j + 1 if j < ny - 1 else j - 1
            jm = j - 1 if j > 0      else j + 1
            add(k, i + jp * nx, cy)
            add(k, i + jm * nx, cy)

            # Cross terms (2nd-order central difference for ∂²d/∂x∂y)
            if abs(Axy) > 1e-14:
                add(k, ip + jp * nx,  cxy)
                add(k, im + jm * nx,  cxy)
                add(k, ip + jm * nx, -cxy)
                add(k, im + jp * nx, -cxy)

    # Sum duplicate (i,j)=(im+ghost=i+1,j) entries properly
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def _solve_damage_2d(H: np.ndarray, d_prev: np.ndarray,
                     params: PFParams2D, cfg: SimConfig2D) -> np.ndarray:
    """
    Solve 2D AT2 damage equation (one staggered step).

    Equation:
        -Gc*ell * ∇·(A·∇d)  +  (Gc/ell + 2H) * d  =  2H
        d ≥ d_prev  (irreversibility)

    Returns d_new (ny, nx).
    """
    nx, ny = cfg.nx, cfg.ny
    hx = cfg.Lx / (nx - 1)
    hy = cfg.Ly / (ny - 1)
    N  = nx * ny

    A = params.anisotropy_tensor()

    # Laplacian part (scaled by -Gc*ell)
    L = _build_2d_laplacian(nx, ny, hx, hy, A)
    L_scaled = -params.Gc * params.ell * L   # LHS contribution

    # Diagonal part: (Gc/ell + 2H)
    diag = params.Gc / params.ell + 2.0 * H.ravel()
    D    = sp.diags(diag, format="csr")

    K   = L_scaled + D   # full LHS matrix
    rhs = 2.0 * H.ravel()

    d_flat = spla.spsolve(K, rhs)
    d_new  = np.clip(d_flat.reshape(ny, nx), d_prev, 1.0)
    return d_new


# ─────────────────────────────────────────────────────────────────────────────
#  Crack path and deflection angle
# ─────────────────────────────────────────────────────────────────────────────

def crack_deflection_angle(d_field: np.ndarray, cfg: SimConfig2D,
                           threshold: float = 0.4) -> float:
    """
    Measure crack deflection angle from 2D damage field.

    Algorithm:
      1. At each x-column (i), find centroid y of nodes where d > threshold.
      2. Collect (x_i, y_i) pairs for columns with any d > threshold.
      3. Fit a line; return slope angle [deg] relative to horizontal (0=straight).

    Returns 0.0 if crack hasn't propagated or is too short to measure.
    """
    nx, ny = cfg.nx, cfg.ny
    xs = np.linspace(0, cfg.Lx, nx)
    ys = np.linspace(0, cfg.Ly, ny)   # shape (ny,)

    x_pts, y_pts = [], []
    for i in range(nx):
        col = d_field[:, i]   # damage along y at this x
        mask = col > threshold
        if mask.sum() >= 2:
            y_centroid = float(np.average(ys[mask], weights=col[mask]))
            x_pts.append(xs[i])
            y_pts.append(y_centroid)

    if len(x_pts) < 4:
        return 0.0

    x_arr = np.array(x_pts)
    y_arr = np.array(y_pts)
    slope, _ = np.polyfit(x_arr, y_arr, 1)
    return float(np.degrees(np.arctan(slope)))


# ─────────────────────────────────────────────────────────────────────────────
#  Main forward simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_forward_2d(params: PFParams2D, cfg: SimConfig2D | None = None
                   ) -> ForwardResult2D:
    """
    2D plane-strain AT2 forward simulation (staggered).

    Steps:
      1. Initialize d=0 (except notch), H=0.
      2. For each load step u:
         a. Update elastic energy density ψ_e(u) uniformly.
         b. Update history field H = max(H, ψ_e⁺).
         c. Solve 2D anisotropic AT2 damage equation → d.
         d. Record reaction force F.
      3. Detect crack initiation (max(d) ≥ 0.5).
      4. Measure crack deflection angle from final d_field.

    Returns ForwardResult2D.
    """
    cfg = cfg or SimConfig2D()

    nx, ny = cfg.nx, cfg.ny
    n_steps = cfg.n_steps
    u_max   = cfg.u_max_factor * params.ell

    # Grid
    U_arr = np.linspace(0.0, u_max, n_steps + 1)
    F_arr = np.zeros(n_steps + 1)

    # Initialize damage and history
    d = np.zeros((ny, nx))
    H = np.zeros((ny, nx))

    # Pre-notch: d=1 along horizontal notch
    notch_j = int(round(cfg.notch_y_frac * (ny - 1)))
    notch_i  = int(round(cfg.notch_x_frac * (nx - 1)))
    d[notch_j, :notch_i + 1] = 1.0   # horizontal notch from left edge

    # Tracking
    init_U     = u_max   # default: no crack detected
    converged  = True
    crack_found = False

    for step, u_y in enumerate(U_arr):
        # ── Step A: solve full 2D displacement (u_x, u_y) with current damage
        ux_field, uy_field = _solve_displacement_2d(d, u_y, params, cfg)

        # ── Step B: compute elastic energy density field ψ_e⁺(x,y) ────────
        # Pass d so crack-aware one-sided ε_yy is used near crack faces.
        psi_field = _compute_H_field(ux_field, uy_field, params, cfg, d=d)

        # ── Step C: update irreversible history field ────────────────────────
        # Zero out at fully-damaged nodes: they cannot fracture further.
        psi_masked = psi_field.copy()
        psi_masked[d >= 0.99] = 0.0
        H = np.maximum(H, psi_masked)

        # ── Step D: solve 2D anisotropic AT2 damage equation ────────────────
        d_prev = d.copy()
        d = _solve_damage_2d(H, d_prev, params, cfg)
        d[notch_j, :notch_i + 1] = 1.0   # re-enforce notch

        # ── Reaction force ────────────────────────────────────────────────────
        F_arr[step] = _reaction_force(ux_field, uy_field, d, params, cfg)

        # Crack initiation (first time max(d) > 0.5 outside notch zone)
        d_bulk = d.copy()
        d_bulk[notch_j, :notch_i + 1] = 0.0
        if not crack_found and np.max(d_bulk) >= 0.5:
            init_U      = float(u_y)
            crack_found = True

    peak_F = float(np.max(F_arr))
    phi    = crack_deflection_angle(d, cfg)

    return ForwardResult2D(
        U=U_arr, F=F_arr, d_field=d,
        init_U=init_U, peak_F=peak_F,
        crack_angle=phi, converged=converged,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Gaussian observation model (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

def crack_angle_log_likelihood(
    params:     PFParams2D,
    obs:        list[tuple[float, float]],  # [(theta_cut_deg, phi_measured_deg), ...]
    sigma_phi:  float = 2.0,                 # measurement noise [deg]
    cfg:        SimConfig2D | None = None,
) -> float:
    """
    Gaussian log-likelihood for 2D crack deflection angle observations.

    Phase 2 observation model:
        phi_obs ~ N(phi_sim(params, theta_cut), sigma_phi²)

    Unlike binary sign-only (Phase 1), this continuous obs makes
    Gc0 AND beta jointly identifiable.

    Args:
        obs: list of (theta_cut_deg, phi_measured_deg) pairs
        sigma_phi: crack-angle measurement noise [deg]
    """
    cfg_ = cfg or SimConfig2D()
    ll   = 0.0
    for theta_deg, phi_obs in obs:
        p = PFParams2D(Gc=params.Gc, ell=params.ell, E=params.E,
                       nu=params.nu, beta=params.beta,
                       theta_cut=np.radians(theta_deg))
        result = run_forward_2d(p, cfg_)
        phi_sim = result.crack_angle
        ll += -0.5 * ((phi_obs - phi_sim) / sigma_phi) ** 2
    ll -= len(obs) * np.log(sigma_phi * np.sqrt(2 * np.pi))
    return ll


def generate_synthetic_dataset_2d(
    true_params:  PFParams2D,
    theta_list:   list[float],
    sigma_phi:    float = 2.0,
    cfg:          SimConfig2D | None = None,
    seed:         int = 0,
) -> list[tuple[float, float]]:
    """
    Generate synthetic (theta_cut_deg, phi_measured_deg) observations.
    Adds Gaussian noise sigma_phi to the simulated crack angle.
    """
    cfg_ = cfg or SimConfig2D()
    rng  = np.random.default_rng(seed)
    obs  = []
    for theta_deg in theta_list:
        p = PFParams2D(Gc=true_params.Gc, ell=true_params.ell, E=true_params.E,
                       nu=true_params.nu, beta=true_params.beta,
                       theta_cut=np.radians(theta_deg))
        result = run_forward_2d(p, cfg_)
        phi_noisy = result.crack_angle + rng.normal(0.0, sigma_phi)
        obs.append((theta_deg, float(phi_noisy)))
    return obs


# ─────────────────────────────────────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    CFG  = SimConfig2D(nx=80, ny=40, n_steps=80, u_max_factor=3.0)
    TRUE = PFParams2D(Gc=8e-4, ell=0.04, E=1.0, nu=0.25, beta=-0.40)

    print("=== 2D AT2 Phase-Field Demo ===\n")
    print(f"Grid: {CFG.nx}×{CFG.ny}  steps={CFG.n_steps}")
    print(f"True params: Gc={TRUE.Gc:.1e}  ell={TRUE.ell}  beta={TRUE.beta}\n")

    # Sweep cut angles
    theta_list = [0, 15, 30, 45, 60, 75, 90]
    print(f"{'theta':>8}  {'init_U':>8}  {'peak_F':>8}  {'phi [deg]':>10}  {'time [ms]':>10}")
    print("-" * 56)

    for theta_deg in theta_list:
        p = PFParams2D(Gc=TRUE.Gc, ell=TRUE.ell, E=TRUE.E,
                       nu=TRUE.nu, beta=TRUE.beta,
                       theta_cut=np.radians(theta_deg))
        t0 = time.perf_counter()
        r  = run_forward_2d(p, CFG)
        dt = (time.perf_counter() - t0) * 1e3
        print(f"{theta_deg:>8}  {r.init_U:>8.4f}  {r.peak_F:>8.4f}  "
              f"{r.crack_angle:>10.2f}  {dt:>10.1f}")

    print("\nComparing isotropic vs anisotropic at theta=45°:")
    for beta, label in [(0.0, "isotropic"), (-0.40, "SiC-like")]:
        p = PFParams2D(Gc=TRUE.Gc, ell=TRUE.ell, E=TRUE.E, nu=TRUE.nu,
                       beta=beta, theta_cut=np.radians(45))
        r = run_forward_2d(p, CFG)
        print(f"  beta={beta:+.2f} ({label:12s}):  phi={r.crack_angle:+.2f}°  "
              f"init_U={r.init_U:.4f}")

    print("\nGenerating synthetic 2D observations (sigma_phi=2°):")
    obs = generate_synthetic_dataset_2d(
        TRUE, theta_list=[0, 30, 45, 60, 90], sigma_phi=2.0, cfg=CFG, seed=42)
    for theta, phi in obs:
        print(f"  theta={theta:3.0f}°  phi_obs={phi:+.2f}°")
