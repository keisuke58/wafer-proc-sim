"""
at2_simulator_3d.py — 3D AT2 phase-field fracture

Extends at2_simulator_2d.py to full 3D, adding:
  - Full 3D FD elasticity (u_x, u_y, u_z)
  - 3D anisotropic Gc tensor: A = I + beta*(c⊗c), c is a 3D unit vector
  - Two crack observables: deflection angle phi_xy (x-y plane) + twist phi_xz (x-z plane)
  - Phase 3 thesis forward model for (Gc, beta, theta, psi) identification

Physical setup:
  - Domain: [0,Lx] × [0,Ly] × [0,Lz]
  - Mode I loading: u_y applied at y=Ly face, u_y=0 at y=0 face
  - Rectangular pre-notch at y=Ly/2, x in [0, notch_x*Lx], spanning all z
  - Crack propagates in x-direction; deflects in y (phi_xy) and twists in z (phi_xz)

DOF/field ordering:
  - Damage d_field[k, j, i] → node flat index = i + j*nx + k*nx*ny
  - Displacement DOF = 3*(i + j*nx + k*nx*ny) + c, c=0,1,2 for u_x,u_y,u_z
  - Axes: axis=2 → x, axis=1 → y, axis=0 → z  (C-order)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ─────────────────────────────────────────────────────────────────────────────
#  Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PFParams3D:
    Gc:        float
    ell:       float
    E:         float
    nu:        float = 0.25
    beta:      float = 0.0
    theta_cut: float = 0.0  # tilt of c in x-y plane [rad]
    psi_cut:   float = 0.0  # tilt of c toward z-axis [rad]

    def cleavage_normal(self) -> np.ndarray:
        """3D unit normal to the cleavage plane.

        At (theta=0, psi=0): c=(0,1,0), cleavage planes horizontal (y=const).
        theta rotates c in x-y plane; psi tilts c toward z.
        """
        ct, st = np.cos(self.theta_cut), np.sin(self.theta_cut)
        cp, sp = np.cos(self.psi_cut),   np.sin(self.psi_cut)
        return np.array([-st * cp, ct * cp, sp])

    def anisotropy_tensor(self) -> np.ndarray:
        """3×3 anisotropy tensor A = I + beta*(c⊗c)."""
        c = self.cleavage_normal()
        return np.eye(3) + self.beta * np.outer(c, c)


@dataclass
class SimConfig3D:
    Lx:           float = 1.0
    Ly:           float = 0.5
    Lz:           float = 0.5
    nx:           int   = 25
    ny:           int   = 13   # odd → exact notch center at (ny-1)/2
    nz:           int   = 13   # odd → exact z-midplane at (nz-1)/2
    n_steps:      int   = 80
    u_max_factor: float = 3.0
    kappa_res:    float = 1e-6
    notch_y_frac: float = 0.5
    notch_x_frac: float = 0.30


@dataclass
class ForwardResult3D:
    U:            np.ndarray   # load steps (n_steps+1,)
    F:            np.ndarray   # reaction force steps (n_steps+1,)
    d_field:      np.ndarray   # final damage (nz, ny, nx)
    init_U:       float
    peak_F:       float
    crack_angle_xy: float      # deflection in x-y plane [deg]
    crack_angle_xz: float      # twist in x-z plane [deg]
    converged:    bool


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: Lamé constants, DOF index
# ─────────────────────────────────────────────────────────────────────────────

def _lame(params: PFParams3D) -> tuple[float, float]:
    E, nu = params.E, params.nu
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = E / (2.0 * (1.0 + nu))
    return lam, mu


def _dof3(i: int, j: int, k: int, c: int, nx: int, ny: int) -> int:
    """DOF index for node (i,j,k), component c=0(ux),1(uy),2(uz)."""
    return 3 * (i + j * nx + k * nx * ny) + c


# ─────────────────────────────────────────────────────────────────────────────
#  3D FD elasticity matrix
# ─────────────────────────────────────────────────────────────────────────────

def _build_elasticity_3d(g: np.ndarray, params: PFParams3D,
                          cfg: SimConfig3D) -> sp.csr_matrix:
    """
    3D isotropic FD elasticity matrix with (u_x, u_y, u_z) DOFs.

    g[k,j,i] = degradation field, shape (nz, ny, nx).
    DOF ordering: 3*(i + j*nx + k*nx*ny) + c.

    Neumann BCs via missing-face: boundary faces carry no flux.
    Cross terms zeroed at boundary nodes (correct for Neumann).
    """
    lam, mu = _lame(params)
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    hx = cfg.Lx / (nx - 1)
    hy = cfg.Ly / (ny - 1)
    hz = cfg.Lz / (nz - 1)
    M  = 3 * nx * ny * nz

    rows_l, cols_l, vals_l = [], [], []

    def add(r, c, v):
        rows_l.append(r); cols_l.append(c); vals_l.append(v)

    def face_term(row, s_dof, n_dof, coeff, g_face, h2):
        """Adds ∂/∂ξ[g·∂u/∂ξ] one-face contribution: diagonal + neighbour."""
        v = coeff * g_face / h2
        add(row, s_dof, -v)
        add(row, n_dof,  v)

    a = lam + 2 * mu  # (λ+2μ) for diagonal direction

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                ip = min(i + 1, nx - 1);  im = max(i - 1, 0)
                jp = min(j + 1, ny - 1);  jm = max(j - 1, 0)
                kp = min(k + 1, nz - 1);  km = max(k - 1, 0)

                gxp = 0.5 * (g[k, j, i] + g[k, j, min(i+1, nx-1)])
                gxm = 0.5 * (g[k, j, i] + g[k, j, max(i-1, 0)])
                gyp = 0.5 * (g[k, j, i] + g[k, min(j+1, ny-1), i])
                gym = 0.5 * (g[k, j, i] + g[k, max(j-1, 0), i])
                gzp = 0.5 * (g[k, j, i] + g[min(k+1, nz-1), j, i])
                gzm = 0.5 * (g[k, j, i] + g[max(k-1, 0), j, i])
                gc  = g[k, j, i]

                at_xL = (i == 0);       at_xR = (i == nx - 1)
                at_yB = (j == 0);       at_yT = (j == ny - 1)
                at_zF = (k == 0);       at_zB = (k == nz - 1)
                cross_ok = not (at_xL or at_xR or at_yB or at_yT or at_zF or at_zB)

                # Precompute neighbour DOFs for all 3 components
                n0  = i  + j  * nx + k  * nx * ny
                nxp = ip + j  * nx + k  * nx * ny
                nxm = im + j  * nx + k  * nx * ny
                nyp = i  + jp * nx + k  * nx * ny
                nym = i  + jm * nx + k  * nx * ny
                nzp = i  + j  * nx + kp * nx * ny
                nzm = i  + j  * nx + km * nx * ny

                # Corner nodes for cross terms
                nxypp = ip + jp * nx + k  * nx * ny
                nxymm = im + jm * nx + k  * nx * ny
                nxypm = ip + jm * nx + k  * nx * ny
                nxymp = im + jp * nx + k  * nx * ny

                nxzpp = ip + j * nx + kp * nx * ny
                nxzmm = im + j * nx + km * nx * ny
                nxzpm = ip + j * nx + km * nx * ny
                nxzmp = im + j * nx + kp * nx * ny

                nyzpp = i + jp * nx + kp * nx * ny
                nyzmm = i + jm * nx + km * nx * ny
                nyzpm = i + jp * nx + km * nx * ny
                nyzmp = i + jm * nx + kp * nx * ny

                # ── x-equilibrium ─────────────────────────────────────────
                rx = 3 * n0 + 0
                # (λ+2μ)·∂/∂x[gx·∂u_x/∂x]
                if not at_xL: face_term(rx, 3*n0+0, 3*nxm+0, a, gxm, hx**2)
                if not at_xR: face_term(rx, 3*n0+0, 3*nxp+0, a, gxp, hx**2)
                # μ·∂/∂y[gy·∂u_x/∂y]
                if not at_yB: face_term(rx, 3*n0+0, 3*nym+0, mu, gym, hy**2)
                if not at_yT: face_term(rx, 3*n0+0, 3*nyp+0, mu, gyp, hy**2)
                # μ·∂/∂z[gz·∂u_x/∂z]
                if not at_zF: face_term(rx, 3*n0+0, 3*nzm+0, mu, gzm, hz**2)
                if not at_zB: face_term(rx, 3*n0+0, 3*nzp+0, mu, gzp, hz**2)
                # (λ+μ)·gc·∂²u_y/∂x∂y
                if cross_ok:
                    cc = (lam + mu) * gc / (4.0 * hx * hy)
                    add(rx, 3*nxypp+1,  cc)
                    add(rx, 3*nxymm+1,  cc)
                    add(rx, 3*nxypm+1, -cc)
                    add(rx, 3*nxymp+1, -cc)
                # (λ+μ)·gc·∂²u_z/∂x∂z
                if cross_ok:
                    cc = (lam + mu) * gc / (4.0 * hx * hz)
                    add(rx, 3*nxzpp+2,  cc)
                    add(rx, 3*nxzmm+2,  cc)
                    add(rx, 3*nxzpm+2, -cc)
                    add(rx, 3*nxzmp+2, -cc)

                # ── y-equilibrium ─────────────────────────────────────────
                ry = 3 * n0 + 1
                # μ·∂/∂x[gx·∂u_y/∂x]
                if not at_xL: face_term(ry, 3*n0+1, 3*nxm+1, mu, gxm, hx**2)
                if not at_xR: face_term(ry, 3*n0+1, 3*nxp+1, mu, gxp, hx**2)
                # (λ+2μ)·∂/∂y[gy·∂u_y/∂y]
                if not at_yB: face_term(ry, 3*n0+1, 3*nym+1, a, gym, hy**2)
                if not at_yT: face_term(ry, 3*n0+1, 3*nyp+1, a, gyp, hy**2)
                # μ·∂/∂z[gz·∂u_y/∂z]
                if not at_zF: face_term(ry, 3*n0+1, 3*nzm+1, mu, gzm, hz**2)
                if not at_zB: face_term(ry, 3*n0+1, 3*nzp+1, mu, gzp, hz**2)
                # (λ+μ)·gc·∂²u_x/∂x∂y
                if cross_ok:
                    cc = (lam + mu) * gc / (4.0 * hx * hy)
                    add(ry, 3*nxypp+0,  cc)
                    add(ry, 3*nxymm+0,  cc)
                    add(ry, 3*nxypm+0, -cc)
                    add(ry, 3*nxymp+0, -cc)
                # (λ+μ)·gc·∂²u_z/∂y∂z
                if cross_ok:
                    cc = (lam + mu) * gc / (4.0 * hy * hz)
                    add(ry, 3*nyzpp+2,  cc)
                    add(ry, 3*nyzmm+2,  cc)
                    add(ry, 3*nyzpm+2, -cc)
                    add(ry, 3*nyzmp+2, -cc)

                # ── z-equilibrium ─────────────────────────────────────────
                rz = 3 * n0 + 2
                # μ·∂/∂x[gx·∂u_z/∂x]
                if not at_xL: face_term(rz, 3*n0+2, 3*nxm+2, mu, gxm, hx**2)
                if not at_xR: face_term(rz, 3*n0+2, 3*nxp+2, mu, gxp, hx**2)
                # μ·∂/∂y[gy·∂u_z/∂y]
                if not at_yB: face_term(rz, 3*n0+2, 3*nym+2, mu, gym, hy**2)
                if not at_yT: face_term(rz, 3*n0+2, 3*nyp+2, mu, gyp, hy**2)
                # (λ+2μ)·∂/∂z[gz·∂u_z/∂z]
                if not at_zF: face_term(rz, 3*n0+2, 3*nzm+2, a, gzm, hz**2)
                if not at_zB: face_term(rz, 3*n0+2, 3*nzp+2, a, gzp, hz**2)
                # (λ+μ)·gc·∂²u_x/∂x∂z
                if cross_ok:
                    cc = (lam + mu) * gc / (4.0 * hx * hz)
                    add(rz, 3*nxzpp+0,  cc)
                    add(rz, 3*nxzmm+0,  cc)
                    add(rz, 3*nxzpm+0, -cc)
                    add(rz, 3*nxzmp+0, -cc)
                # (λ+μ)·gc·∂²u_y/∂y∂z
                if cross_ok:
                    cc = (lam + mu) * gc / (4.0 * hy * hz)
                    add(rz, 3*nyzpp+1,  cc)
                    add(rz, 3*nyzmm+1,  cc)
                    add(rz, 3*nyzpm+1, -cc)
                    add(rz, 3*nyzmp+1, -cc)

    return sp.csr_matrix((vals_l, (rows_l, cols_l)), shape=(M, M))


def _solve_displacement_3d(d: np.ndarray, u_applied: float,
                            params: PFParams3D, cfg: SimConfig3D,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve for (u_x, u_y, u_z) given damage field and applied top displacement.

    BCs (Mode I):
      Bottom (j=0):                u_y = 0  (all i,k)
      Top    (j=ny-1):             u_y = u_applied  (all i,k)
      Notch center (i=nx//2, j=notch_j, k=nz//2):
          u_x = 0  (Mode I antisymmetry — preserves symmetry about x=Lx/2)
          u_z = 0  (z-translation pin — by z-symmetry of uniform crack)
    """
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    M = 3 * nx * ny * nz

    g = (1.0 - d) ** 2 + cfg.kappa_res   # shape (nz, ny, nx)
    K = _build_elasticity_3d(g, params, cfg).tolil()
    rhs = np.zeros(M)

    # u_y = 0 at bottom face (j=0, all i,k)
    for k in range(nz):
        for i in range(nx):
            r = _dof3(i, 0, k, 1, nx, ny)
            K[r, :] = 0; K[r, r] = 1.0; rhs[r] = 0.0

    # u_y = u_applied at top face (j=ny-1, all i,k)
    for k in range(nz):
        for i in range(nx):
            r = _dof3(i, ny - 1, k, 1, nx, ny)
            K[r, :] = 0; K[r, r] = 1.0; rhs[r] = u_applied

    # u_x = 0 at notch-level center — Mode I antisymmetry
    notch_j = int(round(cfg.notch_y_frac * (ny - 1)))
    r = _dof3(nx // 2, notch_j, nz // 2, 0, nx, ny)
    K[r, :] = 0; K[r, r] = 1.0; rhs[r] = 0.0

    # u_z = 0 at z-midplane center — z-rigid body pin
    r = _dof3(nx // 2, notch_j, nz // 2, 2, nx, ny)
    K[r, :] = 0; K[r, r] = 1.0; rhs[r] = 0.0

    u_flat = spla.spsolve(K.tocsr(), rhs)
    u_x = u_flat[0::3].reshape(nz, ny, nx)
    u_y = u_flat[1::3].reshape(nz, ny, nx)
    u_z = u_flat[2::3].reshape(nz, ny, nx)
    return u_x, u_y, u_z


def _compute_H_field_3d(u_x: np.ndarray, u_y: np.ndarray, u_z: np.ndarray,
                         params: PFParams3D, cfg: SimConfig3D,
                         d: np.ndarray | None = None) -> np.ndarray:
    """
    Full 3D elastic energy density ψ_e⁺:
      ψ = λ/2·(tr ε)² + μ·(ε_xx²+ε_yy²+ε_zz²+2ε_xy²+2ε_xz²+2ε_yz²)

    Axes convention: fields[k, j, i] → axis=2 for x, axis=1 for y, axis=0 for z.
    Crack-aware one-sided ε_yy near cracked rows (same as 2D fix, extended to 3D).
    """
    hx = cfg.Lx / (cfg.nx - 1)
    hy = cfg.Ly / (cfg.ny - 1)
    hz = cfg.Lz / (cfg.nz - 1)
    lam, mu = _lame(params)

    eps_xx = np.gradient(u_x, hx, axis=2)
    eps_yy = np.gradient(u_y, hy, axis=1)   # central diff (overridden near crack)
    eps_zz = np.gradient(u_z, hz, axis=0)
    eps_xy = 0.5 * (np.gradient(u_x, hy, axis=1) + np.gradient(u_y, hx, axis=2))
    eps_xz = 0.5 * (np.gradient(u_x, hz, axis=0) + np.gradient(u_z, hx, axis=2))
    eps_yz = 0.5 * (np.gradient(u_y, hz, axis=0) + np.gradient(u_z, hy, axis=1))

    if d is not None:
        cracked = d >= 0.99   # shape (nz, ny, nx)
        # One-sided ε_yy near horizontal crack faces (axis=1 is y-direction).
        # lower y-neighbor cracked → forward diff in y
        mask_lo = cracked[:, :-2, :] & ~cracked[:, 1:-1, :]  # (nz, ny-2, nx)
        eps_yy[:, 1:-1, :][mask_lo] = (
            (u_y[:, 2:, :][mask_lo] - u_y[:, 1:-1, :][mask_lo]) / hy
        )
        # upper y-neighbor cracked → backward diff in y
        mask_hi = cracked[:, 2:, :] & ~cracked[:, 1:-1, :]
        eps_yy[:, 1:-1, :][mask_hi] = (
            (u_y[:, 1:-1, :][mask_hi] - u_y[:, :-2, :][mask_hi]) / hy
        )

    tr_eps = eps_xx + eps_yy + eps_zz
    psi = (0.5 * lam * tr_eps ** 2
           + mu * (eps_xx**2 + eps_yy**2 + eps_zz**2
                   + 2.0 * eps_xy**2 + 2.0 * eps_xz**2 + 2.0 * eps_yz**2))
    return np.maximum(psi, 0.0)


def _reaction_force_3d(u_x: np.ndarray, u_y: np.ndarray,
                        d: np.ndarray, params: PFParams3D,
                        cfg: SimConfig3D) -> float:
    """Integrate degraded σ_yy over bottom face (j=0)."""
    hx = cfg.Lx / (cfg.nx - 1)
    hy = cfg.Ly / (cfg.ny - 1)
    hz = cfg.Lz / (cfg.nz - 1)
    lam, mu = _lame(params)
    # Strains at j=0 (bottom face): shape (nz, nx)
    eps_xx = np.gradient(u_x, hx, axis=2)[:, 0, :]
    eps_yy = np.gradient(u_y, hy, axis=1)[:, 0, :]
    eps_zz = np.zeros_like(eps_xx)   # approximately zero at bottom face
    g      = (1.0 - d[:, 0, :]) ** 2 + cfg.kappa_res
    sig_yy = g * ((lam + 2*mu) * eps_yy + lam * (eps_xx + eps_zz))
    return float(np.sum(sig_yy) * hx * hz)


# ─────────────────────────────────────────────────────────────────────────────
#  3D anisotropic AT2 damage solver
# ─────────────────────────────────────────────────────────────────────────────

def _build_3d_laplacian(nx: int, ny: int, nz: int,
                         hx: float, hy: float, hz: float,
                         A: np.ndarray) -> sp.csr_matrix:
    """
    Sparse matrix for 3D anisotropic Laplacian: ∇·(A·∇d).

    A is the 3×3 anisotropy tensor (constant). Neumann BCs via ghost-node
    reflection at all boundaries. Node ordering: n = i + j*nx + k*nx*ny.

    Cross terms: ∇·(A·∇d) includes 2*Axy·∂²d/∂x∂y + 2*Axz·∂²d/∂x∂z +
    2*Ayz·∂²d/∂y∂z for symmetric A.
    """
    N   = nx * ny * nz
    Axx = A[0, 0];  Ayy = A[1, 1];  Azz = A[2, 2]
    Axy = A[0, 1];  Axz = A[0, 2];  Ayz = A[1, 2]

    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n = i + j * nx + k * nx * ny

                # Neumann ghost-node reflection
                ip = i + 1 if i < nx - 1 else i - 1
                im = i - 1 if i > 0      else i + 1
                jp = j + 1 if j < ny - 1 else j - 1
                jm = j - 1 if j > 0      else j + 1
                kp = k + 1 if k < nz - 1 else k - 1
                km = k - 1 if k > 0      else k + 1

                # Diagonal
                add(n, n, -2.0*Axx/hx**2 - 2.0*Ayy/hy**2 - 2.0*Azz/hz**2)

                # x-neighbors
                add(n, ip + j*nx + k*nx*ny, Axx / hx**2)
                add(n, im + j*nx + k*nx*ny, Axx / hx**2)
                # y-neighbors
                add(n, i + jp*nx + k*nx*ny, Ayy / hy**2)
                add(n, i + jm*nx + k*nx*ny, Ayy / hy**2)
                # z-neighbors
                add(n, i + j*nx + kp*nx*ny, Azz / hz**2)
                add(n, i + j*nx + km*nx*ny, Azz / hz**2)

                # xy cross: 2*Axy·∂²d/∂x∂y
                if abs(Axy) > 1e-14:
                    cxy = Axy / (2.0 * hx * hy)
                    add(n, ip + jp*nx + k*nx*ny,  cxy)
                    add(n, im + jm*nx + k*nx*ny,  cxy)
                    add(n, ip + jm*nx + k*nx*ny, -cxy)
                    add(n, im + jp*nx + k*nx*ny, -cxy)

                # xz cross: 2*Axz·∂²d/∂x∂z
                if abs(Axz) > 1e-14:
                    cxz = Axz / (2.0 * hx * hz)
                    add(n, ip + j*nx + kp*nx*ny,  cxz)
                    add(n, im + j*nx + km*nx*ny,  cxz)
                    add(n, ip + j*nx + km*nx*ny, -cxz)
                    add(n, im + j*nx + kp*nx*ny, -cxz)

                # yz cross: 2*Ayz·∂²d/∂y∂z
                if abs(Ayz) > 1e-14:
                    cyz = Ayz / (2.0 * hy * hz)
                    add(n, i + jp*nx + kp*nx*ny,  cyz)
                    add(n, i + jm*nx + km*nx*ny,  cyz)
                    add(n, i + jp*nx + km*nx*ny, -cyz)
                    add(n, i + jm*nx + kp*nx*ny, -cyz)

    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def _solve_damage_3d(H: np.ndarray, d_prev: np.ndarray,
                     params: PFParams3D, cfg: SimConfig3D) -> np.ndarray:
    """
    Solve 3D AT2 damage (one staggered step):
        -Gc*ell · ∇·(A·∇d) + (Gc/ell + 2H) · d = 2H
        d ≥ d_prev  (irreversibility)
    """
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    hx = cfg.Lx / (nx - 1)
    hy = cfg.Ly / (ny - 1)
    hz = cfg.Lz / (nz - 1)
    N  = nx * ny * nz

    A = params.anisotropy_tensor()
    L = _build_3d_laplacian(nx, ny, nz, hx, hy, hz, A)
    L_scaled = -params.Gc * params.ell * L

    diag = params.Gc / params.ell + 2.0 * H.ravel()
    D    = sp.diags(diag, format="csr")

    K   = L_scaled + D
    rhs = 2.0 * H.ravel()

    d_flat = spla.spsolve(K, rhs)
    d_new  = np.clip(d_flat.reshape(nz, ny, nx), d_prev, 1.0)
    return d_new


# ─────────────────────────────────────────────────────────────────────────────
#  Crack path and deflection/twist angles
# ─────────────────────────────────────────────────────────────────────────────

def crack_deflection_angles_3d(d_field: np.ndarray, cfg: SimConfig3D,
                                threshold: float = 0.4) -> tuple[float, float]:
    """
    Measure crack deflection (phi_xy) and twist (phi_xz) from 3D damage field.

    phi_xy: crack deflection in x-y plane (averaged over z slices).
      At each x-column i, find y-centroid of d>threshold averaged over all k.
      Fit linear slope; return angle in degrees.

    phi_xz: crack twist in x-z plane (z-centroid vs x position, averaged over y).
      At each x-column i, find z-centroid of d>threshold averaged over all j.
      Fit linear slope; return angle in degrees.

    Both return 0.0 if crack is too short to measure.
    """
    nz, ny, nx = d_field.shape
    xs = np.linspace(0, cfg.Lx, nx)
    ys = np.linspace(0, cfg.Ly, ny)
    zs = np.linspace(0, cfg.Lz, nz)

    # phi_xy: average over z, then measure y-centroid vs x
    d_avg_z = d_field.mean(axis=0)   # (ny, nx)
    xp, yp = [], []
    for i in range(nx):
        col = d_avg_z[:, i]           # (ny,)
        mask = col > threshold
        if mask.sum() >= 2:
            y_cen = float(np.average(ys[mask], weights=col[mask]))
            xp.append(xs[i])
            yp.append(y_cen)

    phi_xy = 0.0
    if len(xp) >= 4:
        slope, _ = np.polyfit(xp, yp, 1)
        phi_xy = float(np.degrees(np.arctan(slope)))

    # phi_xz: average over y, then measure z-centroid vs x
    d_avg_y = d_field.mean(axis=1)   # (nz, nx)
    xp2, zp = [], []
    for i in range(nx):
        col = d_avg_y[:, i]           # (nz,)
        mask = col > threshold
        if mask.sum() >= 2:
            z_cen = float(np.average(zs[mask], weights=col[mask]))
            xp2.append(xs[i])
            zp.append(z_cen)

    phi_xz = 0.0
    if len(xp2) >= 4:
        slope2, _ = np.polyfit(xp2, zp, 1)
        phi_xz = float(np.degrees(np.arctan(slope2)))

    return phi_xy, phi_xz


# ─────────────────────────────────────────────────────────────────────────────
#  Main 3D forward simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_forward_3d(params: PFParams3D,
                   cfg: SimConfig3D | None = None) -> ForwardResult3D:
    """
    3D AT2 forward simulation (staggered elasticity + damage).

    Returns ForwardResult3D with crack deflection (phi_xy) and twist (phi_xz).
    """
    cfg = cfg or SimConfig3D()

    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    n_steps = cfg.n_steps
    u_max   = cfg.u_max_factor * params.ell

    U_arr = np.linspace(0.0, u_max, n_steps + 1)
    F_arr = np.zeros(n_steps + 1)

    # Initialize damage (nz, ny, nx) and history
    d = np.zeros((nz, ny, nx))
    H = np.zeros((nz, ny, nx))

    # Pre-notch: rectangular slit at j=notch_j, i=0..notch_i, all k
    notch_j = int(round(cfg.notch_y_frac * (ny - 1)))
    notch_i = int(round(cfg.notch_x_frac * (nx - 1)))
    d[:, notch_j, :notch_i + 1] = 1.0

    init_U      = u_max
    converged   = True
    crack_found = False

    for step, u_app in enumerate(U_arr):
        # ── A: solve 3D displacement
        ux, uy, uz = _solve_displacement_3d(d, u_app, params, cfg)

        # ── B: elastic energy density (crack-aware ε_yy)
        psi = _compute_H_field_3d(ux, uy, uz, params, cfg, d=d)

        # ── C: update history (zero out at fully-damaged nodes)
        psi_m = psi.copy()
        psi_m[d >= 0.99] = 0.0
        H = np.maximum(H, psi_m)

        # ── D: solve 3D damage
        d_prev = d.copy()
        d = _solve_damage_3d(H, d_prev, params, cfg)
        d[:, notch_j, :notch_i + 1] = 1.0   # re-enforce notch

        # ── Reaction force
        F_arr[step] = _reaction_force_3d(ux, uy, d, params, cfg)

        # Crack initiation
        d_bulk = d.copy()
        d_bulk[:, notch_j, :notch_i + 1] = 0.0
        if not crack_found and np.max(d_bulk) >= 0.5:
            init_U      = float(u_app)
            crack_found = True

    peak_F = float(np.max(F_arr))
    phi_xy, phi_xz = crack_deflection_angles_3d(d, cfg)

    return ForwardResult3D(
        U=U_arr, F=F_arr, d_field=d,
        init_U=init_U, peak_F=peak_F,
        crack_angle_xy=phi_xy, crack_angle_xz=phi_xz,
        converged=converged,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Gaussian observation model (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

def crack_angle_log_likelihood_3d(
    params:    PFParams3D,
    obs:       list[tuple[float, float, float, float]],
    sigma_phi: float = 2.0,
    cfg:       SimConfig3D | None = None,
) -> float:
    """
    Gaussian log-likelihood for 3D crack angle observations.

    obs: list of (theta_cut_deg, psi_cut_deg, phi_xy_obs_deg, phi_xz_obs_deg)
    Both angles are modelled as independent Gaussians with std sigma_phi.

    Phase 3 observation model:
        phi_xy_obs ~ N(phi_xy_sim, sigma²)
        phi_xz_obs ~ N(phi_xz_sim, sigma²)
    Jointly identifiable: (Gc, beta, theta, psi) from multi-orientation observations.
    """
    cfg_ = cfg or SimConfig3D()
    ll   = 0.0
    for theta_deg, psi_deg, phi_xy_obs, phi_xz_obs in obs:
        p = PFParams3D(
            Gc=params.Gc, ell=params.ell, E=params.E, nu=params.nu,
            beta=params.beta,
            theta_cut=np.radians(theta_deg),
            psi_cut=np.radians(psi_deg),
        )
        r = run_forward_3d(p, cfg_)
        ll += -0.5 * ((phi_xy_obs - r.crack_angle_xy) / sigma_phi) ** 2
        ll += -0.5 * ((phi_xz_obs - r.crack_angle_xz) / sigma_phi) ** 2
    ll -= 2 * len(obs) * np.log(sigma_phi * np.sqrt(2 * np.pi))
    return ll


# ─────────────────────────────────────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    CFG  = SimConfig3D(nx=25, ny=13, nz=13, n_steps=60, u_max_factor=3.0)
    BASE = PFParams3D(Gc=8e-4, ell=0.04, E=1.0, nu=0.25, beta=0.0)

    print("=== 3D AT2 Phase-Field Demo ===\n")
    print(f"Grid: {CFG.nx}×{CFG.ny}×{CFG.nz}  steps={CFG.n_steps}")
    print(f"DOFs: {3*CFG.nx*CFG.ny*CFG.nz}")

    # Isotropic baseline
    t0 = time.perf_counter()
    r0 = run_forward_3d(BASE, CFG)
    dt = time.perf_counter() - t0
    print(f"\nIsotropic: phi_xy={r0.crack_angle_xy:+.2f}°  "
          f"phi_xz={r0.crack_angle_xz:+.2f}°  "
          f"initU={r0.init_U:.4f}  ({dt:.1f}s)")

    # Anisotropy sweep: theta at 45°, psi=0 (in-plane cleavage, same as 2D)
    print("\nIn-plane sweep (psi=0, theta varies):")
    for theta_deg in [0, 45, 90]:
        p = PFParams3D(Gc=8e-4, ell=0.04, E=1.0, nu=0.25,
                       beta=-0.8, theta_cut=np.radians(theta_deg))
        t0 = time.perf_counter()
        r  = run_forward_3d(p, CFG)
        dt = time.perf_counter() - t0
        print(f"  theta={theta_deg:3d}°  phi_xy={r.crack_angle_xy:+.2f}°  "
              f"phi_xz={r.crack_angle_xz:+.2f}°  ({dt:.1f}s)")

    # Out-of-plane sweep: psi varies (crack twist)
    print("\nOut-of-plane sweep (theta=0, psi varies — new 3D observable):")
    for psi_deg in [0, 30, 45, 60, 90]:
        p = PFParams3D(Gc=8e-4, ell=0.04, E=1.0, nu=0.25,
                       beta=-0.8, psi_cut=np.radians(psi_deg))
        t0 = time.perf_counter()
        r  = run_forward_3d(p, CFG)
        dt = time.perf_counter() - t0
        print(f"  psi={psi_deg:3d}°  phi_xy={r.crack_angle_xy:+.2f}°  "
              f"phi_xz={r.crack_angle_xz:+.2f}°  ({dt:.1f}s)")
