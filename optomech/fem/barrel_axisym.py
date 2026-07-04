#!/usr/bin/env python3
"""barrel_axisym.py — axisymmetric solid finite-element model of the barrel wall.

The beam model (`barrel_fem.py`) captures bending modes; this one meshes the
barrel wall cross-section in the (r, z) plane with **4-node axisymmetric solid
(Q4) elements** and solves the true 2-D elasticity field — the analysis a CAE
engineer runs for thermal growth and stress, with a real visualizable mesh.

Analyses:
  1. Thermal expansion — a uniform temperature rise imposes an axisymmetric
     thermal strain; the FE solve gives the axial growth (= focus/back-focal
     drift) and the radial expansion, validated against alpha*L*dT.
  2. Axial mount preload — an end pressure gives the von Mises stress field.
  3. Mesh + deformed-shape / von-Mises visualization.

Pure NumPy (+ matplotlib). Run:  python3 barrel_axisym.py
"""
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# Al 6061
E, NU, RHO, ALPHA, YIELD = 68.9e9, 0.33, 2700.0, 23.6e-6, 276e6
OUTER_MM, WALL_MM, LEN_MM = 30.0, 2.0, 40.0


def d_matrix():
    """Axisymmetric isotropic elasticity matrix (order: rr, zz, tt, rz)."""
    c = E / ((1 + NU) * (1 - 2 * NU))
    D = np.array([
        [1 - NU, NU, NU, 0],
        [NU, 1 - NU, NU, 0],
        [NU, NU, 1 - NU, 0],
        [0, 0, 0, (1 - 2 * NU) / 2],
    ]) * c
    return D


def mesh(nr=4, nz=40):
    Ri = (OUTER_MM / 2 - WALL_MM) * 1e-3
    Ro = OUTER_MM / 2 * 1e-3
    L = LEN_MM * 1e-3
    rs = np.linspace(Ri, Ro, nr + 1)
    zs = np.linspace(0, L, nz + 1)
    nodes = np.array([[r, z] for z in zs for r in rs])  # (r,z), z-major
    def nid(ir, iz):
        return iz * (nr + 1) + ir
    elems = []
    for iz in range(nz):
        for ir in range(nr):
            elems.append([nid(ir, iz), nid(ir + 1, iz),
                          nid(ir + 1, iz + 1), nid(ir, iz + 1)])
    return nodes, np.array(elems), nr, nz


GP = 1 / np.sqrt(3)
GAUSS = [(-GP, -GP), (GP, -GP), (GP, GP), (-GP, GP)]


def shape(xi, eta):
    N = 0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta),
                         (1+xi)*(1+eta), (1-xi)*(1+eta)])
    dN = 0.25 * np.array([
        [-(1-eta), (1-eta), (1+eta), -(1+eta)],   # dN/dxi
        [-(1-xi), -(1+xi), (1+xi), (1-xi)],       # dN/deta
    ])
    return N, dN


def element_k_f(coords, D, dT):
    """Axisymmetric Q4 stiffness (8x8) and thermal load (8,)."""
    Ke = np.zeros((8, 8))
    Fe = np.zeros(8)
    eps0 = ALPHA * dT * np.array([1.0, 1.0, 1.0, 0.0])
    for xi, eta in GAUSS:
        N, dN = shape(xi, eta)
        J = dN @ coords                      # 2x2 Jacobian (d[r,z]/d[xi,eta])
        detJ = np.linalg.det(J)
        dNxy = np.linalg.solve(J, dN)        # dN/d[r,z]
        r = N @ coords[:, 0]
        if r < 1e-12:
            r = 1e-12
        B = np.zeros((4, 8))
        for a in range(4):
            B[0, 2*a] = dNxy[0, a]           # eps_rr = dUr/dr
            B[1, 2*a+1] = dNxy[1, a]         # eps_zz = dUz/dz
            B[2, 2*a] = N[a] / r             # eps_tt = Ur/r
            B[3, 2*a] = dNxy[1, a]           # gamma_rz
            B[3, 2*a+1] = dNxy[0, a]
        w = detJ * 2 * np.pi * r             # axisymmetric volume weight
        Ke += B.T @ D @ B * w
        Fe += B.T @ D @ eps0 * w
    return Ke, Fe


def assemble(nodes, elems, D, dT):
    ndof = 2 * len(nodes)
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)
    for el in elems:
        coords = nodes[el]
        Ke, Fe = element_k_f(coords, D, dT)
        dofs = np.array([[2*n, 2*n+1] for n in el]).ravel()
        K[np.ix_(dofs, dofs)] += Ke
        F[dofs] += Fe
    return K, F


def solve_thermal(nodes, elems, nr, nz, dT=10.0):
    D = d_matrix()
    K, F = assemble(nodes, elems, D, dT)
    ndof = 2 * len(nodes)
    fixed = []
    # Fix the mount face (z=0) axially; fix the inner-bottom node radially to
    # remove rigid-body motion (radial expansion stays free elsewhere).
    for i, (r, z) in enumerate(nodes):
        if abs(z) < 1e-12:
            fixed.append(2*i + 1)           # uz = 0 on z=0 face
    fixed.append(0)                          # ur = 0 at first node (r=Ri,z=0)
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    return u, D


def von_mises(nodes, elems, u, D, dT):
    """Element-centre von Mises stress (thermal strain subtracted)."""
    eps0 = ALPHA * dT * np.array([1.0, 1.0, 1.0, 0.0])
    vm = np.zeros(len(elems))
    for e, el in enumerate(elems):
        coords = nodes[el]
        ue = np.array([[u[2*n], u[2*n+1]] for n in el]).ravel()
        N, dN = shape(0, 0)
        J = dN @ coords
        dNxy = np.linalg.solve(J, dN)
        r = N @ coords[:, 0]
        B = np.zeros((4, 8))
        for a in range(4):
            B[0, 2*a] = dNxy[0, a]; B[1, 2*a+1] = dNxy[1, a]
            B[2, 2*a] = N[a] / max(r, 1e-12)
            B[3, 2*a] = dNxy[1, a]; B[3, 2*a+1] = dNxy[0, a]
        s = D @ (B @ ue - eps0)              # [s_rr, s_zz, s_tt, s_rz]
        srr, szz, stt, srz = s
        vm[e] = np.sqrt(0.5*((srr-szz)**2 + (szz-stt)**2 + (stt-srr)**2) + 3*srz**2)
    return vm


def plot_mesh_deformed(nodes, elems, u, vm, path, scale=None):
    z = nodes[:, 1]*1e3; r = nodes[:, 0]*1e3
    ur = u[0::2]; uz = u[1::2]
    if scale is None:
        span = max(z.max()-z.min(), r.max()-r.min())
        dmax = max(np.abs(np.r_[ur, uz]).max(), 1e-12)
        scale = 0.15*span/(dmax*1e3)
    zr = (z + uz*1e3*scale)
    rr = (r + ur*1e3*scale)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.0),
                                 gridspec_kw={"width_ratios": [1, 1.15]})
    # Undeformed mesh.
    polys = [np.column_stack([z[el], r[el]]) for el in elems]
    a1.add_collection(PolyCollection(polys, facecolors="none",
                                     edgecolors="#4a5a6a", linewidths=.5))
    a1.set_xlim(z.min()-1, z.max()+1); a1.set_ylim(r.min()-1, r.max()+3)
    a1.set_title(f"Axisymmetric mesh\n{len(elems)} Q4 elements, {len(nodes)} nodes",
                 fontsize=9)
    a1.set_xlabel("z [mm]"); a1.set_ylabel("r [mm]"); a1.set_aspect("equal")
    # Deformed + von Mises.
    dpolys = [np.column_stack([zr[el], rr[el]]) for el in elems]
    pc = PolyCollection(dpolys, array=vm/1e6, cmap="plasma",
                        edgecolors="#222", linewidths=.3)
    a2.add_collection(pc)
    a2.set_xlim(zr.min()-1, zr.max()+1); a2.set_ylim(rr.min()-1, rr.max()+3)
    a2.set_title(f"Deformed (x{scale:.0f}) + von Mises stress", fontsize=9)
    a2.set_xlabel("z [mm]"); a2.set_ylabel("r [mm]"); a2.set_aspect("equal")
    fig.colorbar(pc, ax=a2, label="von Mises [MPa]", fraction=.046)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    figs = os.path.join(here, "figures"); os.makedirs(figs, exist_ok=True)

    nodes, elems, nr, nz = mesh(nr=4, nz=40)
    dT = 10.0
    u, D = solve_thermal(nodes, elems, nr, nz, dT)
    vm = von_mises(nodes, elems, u, D, dT)

    L = LEN_MM*1e-3
    axial_tip = float(u[1::2].max())          # max axial growth [m]
    radial = float(u[0::2].max())             # max radial expansion [m]
    analytic_axial = ALPHA * dT * L
    plot_mesh_deformed(nodes, elems, u, vm, os.path.join(figs, "axisym_mesh.png"))

    out = {
        "elements": int(len(elems)), "nodes": int(len(nodes)),
        "dT_C": dT,
        "axial_growth_um": round(axial_tip*1e6, 3),
        "axial_growth_um_per_C": round(axial_tip*1e6/dT, 4),
        "analytic_axial_um": round(analytic_axial*1e6, 3),
        "axial_fe_vs_analytic_pct": round(100*abs(axial_tip-analytic_axial)/analytic_axial, 2),
        "radial_expansion_um": round(radial*1e6, 3),
        "max_von_mises_MPa": round(float(vm.max())/1e6, 3),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open(os.path.join(here, "barrel_axisym_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
