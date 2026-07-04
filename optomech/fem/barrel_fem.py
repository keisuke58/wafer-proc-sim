#!/usr/bin/env python3
"""barrel_fem.py — a real finite-element model of a camera lens barrel.

Unlike the closed-form C++ `fea` module, this is a mesh-based Euler-Bernoulli
beam FE model (2 DOF/node: transverse displacement + rotation) assembled from
element stiffness/consistent-mass matrices, solving three genuine analyses that
a lens-barrel mechanical designer runs:

  1. Modal analysis  — generalized eigenproblem (K - w^2 M) for natural
     frequencies and mode shapes (validated against the closed-form cantilever).
  2. Thermal-structural coupling — a radial temperature gradient across the
     barrel imposes a thermal curvature; the FE solve gives the resulting
     boresight (tip) deflection.
  3. Drop-shock transient — Newmark-beta time integration of the barrel under a
     half-sine base-acceleration pulse, giving the peak root bending stress and
     a safety factor.

The barrel is modelled as a cantilever tube (fixed at the mount, free at the
front) carrying a lumped lens-group mass at the tip.

Pure NumPy (+ SciPy eigensolver if available, NumPy fallback otherwise).
Run:  python3 barrel_fem.py   ->  writes figures + prints JSON metrics.
"""
import json
import os
import numpy as np

try:
    from scipy.linalg import eigh as _sp_eigh
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.80665


# ── Material / geometry ──────────────────────────────────────────────────────
class Material:
    def __init__(self, name, E_GPa, rho, yield_MPa, cte, endurance_MPa):
        self.name = name
        self.E = E_GPa * 1e9
        self.rho = rho
        self.yield_ = yield_MPa * 1e6
        self.cte = cte
        self.endurance = endurance_MPa * 1e6


AL6061 = Material("Al 6061", 68.9, 2700.0, 276.0, 23.6e-6, 96.0)
MG_AZ31 = Material("Mg AZ31", 45.0, 1770.0, 200.0, 26.0e-6, 70.0)


class Barrel:
    def __init__(self, mat=AL6061, outer_dia_mm=30.0, wall_mm=2.0,
                 length_mm=40.0, lens_mass_g=8.0, n_elem=20):
        self.mat = mat
        self.Do = outer_dia_mm * 1e-3
        self.Di = (outer_dia_mm - 2.0 * wall_mm) * 1e-3
        self.L = length_mm * 1e-3
        self.m_lens = lens_mass_g * 1e-3
        self.n_elem = n_elem

    @property
    def area(self):
        return np.pi / 4.0 * (self.Do**2 - self.Di**2)

    @property
    def I(self):
        return np.pi / 64.0 * (self.Do**4 - self.Di**4)

    @property
    def c(self):
        return self.Do / 2.0


# ── Element matrices (Euler-Bernoulli beam) ──────────────────────────────────
def beam_ke(EI, le):
    return EI / le**3 * np.array([
        [12,     6*le,    -12,     6*le],
        [6*le,   4*le**2, -6*le,   2*le**2],
        [-12,   -6*le,     12,    -6*le],
        [6*le,   2*le**2, -6*le,   4*le**2],
    ])


def beam_me(rhoA, le):
    return rhoA * le / 420.0 * np.array([
        [156,    22*le,    54,    -13*le],
        [22*le,  4*le**2,  13*le, -3*le**2],
        [54,     13*le,    156,   -22*le],
        [-13*le, -3*le**2, -22*le, 4*le**2],
    ])


def assemble(barrel):
    """Assemble global K, M for a cantilever tube with a tip lens mass.

    Returns the reduced (free-DOF) K, M after applying the fixed-root BC, plus
    the free-DOF index map. DOF ordering per node: [v (transverse), theta].
    """
    n = barrel.n_elem
    le = barrel.L / n
    EI = barrel.mat.E * barrel.I
    rhoA = barrel.mat.rho * barrel.area
    ndof = 2 * (n + 1)
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    ke, me = beam_ke(EI, le), beam_me(rhoA, le)
    for e in range(n):
        d = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K[np.ix_(d, d)] += ke
        M[np.ix_(d, d)] += me
    # Lumped lens mass on the tip translational DOF.
    M[ndof-2, ndof-2] += barrel.m_lens
    # Cantilever: node 0 fixed (v0 = theta0 = 0) -> drop DOF 0,1.
    free = np.arange(2, ndof)
    return K[np.ix_(free, free)], M[np.ix_(free, free)], free, le, EI


# ── Analyses ─────────────────────────────────────────────────────────────────
def modal(barrel, n_modes=3):
    K, M, free, le, EI = assemble(barrel)
    if _HAVE_SCIPY:
        w2, phi = _sp_eigh(K, M)
    else:  # generalized eig via M^-1 K
        w2, phi = np.linalg.eig(np.linalg.solve(M, K))
        idx = np.argsort(w2.real)
        w2, phi = w2.real[idx], phi.real[:, idx]
    w2 = np.clip(w2, 0, None)
    freqs = np.sqrt(w2) / (2*np.pi)
    return freqs[:n_modes], phi[:, :n_modes], free


def closed_form_f1(barrel):
    """Cantilever first bending frequency with a tip mass (Rayleigh)."""
    EI = barrel.mat.E * barrel.I
    k = 3*EI / barrel.L**3
    m_eff = barrel.m_lens + 0.24 * barrel.mat.rho * barrel.area * barrel.L
    return np.sqrt(k/m_eff) / (2*np.pi)


def thermal_bending(barrel, dT_gradient=5.0):
    """Radial temperature gradient dT across the diameter -> thermal curvature
    kappa = cte*dT/Do imposed on every element; solve for the boresight tip
    deflection (a coupled thermal-structural solve)."""
    K, M, free, le, EI = assemble(barrel)
    kappa = barrel.mat.cte * dT_gradient / barrel.Do
    # Element thermal moment M_th = EI*kappa -> nodal force vector [0, +M, 0, -M].
    n = barrel.n_elem
    ndof = 2*(n+1)
    F = np.zeros(ndof)
    for e in range(n):
        d = [2*e, 2*e+1, 2*e+2, 2*e+3]
        F[d] += EI*kappa*np.array([0.0, 1.0, 0.0, -1.0])
    Ff = F[free]
    u = np.linalg.solve(K, Ff)
    tip = u[-2]                          # tip transverse deflection [m]
    analytic = kappa * barrel.L**2 / 2.0  # uniform-curvature cantilever tip
    return tip, analytic, u, free


def drop_shock(barrel, drop_m=1.0, tau_ms=0.5, zeta=0.02, dt_us=2.0):
    """Newmark-beta transient of the barrel under a half-sine base-acceleration
    pulse from a drop. Returns peak tip displacement, peak root bending stress,
    safety factor, and the tip time history."""
    K, M, free, le, EI = assemble(barrel)
    ndof = K.shape[0]
    # Rayleigh damping tuned to ~zeta at the first mode.
    f1 = modal(barrel, 1)[0][0]
    w1 = 2*np.pi*f1
    alpha, beta = zeta*w1, zeta/w1          # C = alpha*M + beta*K
    C = alpha*M + beta*K
    # Influence vector: base motion excites translational DOF only.
    r = np.zeros(ndof)
    r[0::2] = 1.0
    v_impact = np.sqrt(2*G*drop_m)
    tau = tau_ms*1e-3
    a_peak = np.pi*v_impact/(2*tau)         # half-sine peak accel [m/s^2]

    dt = dt_us*1e-6
    t_end = tau + 3.0/f1
    nsteps = int(t_end/dt)
    # Newmark-beta (average acceleration): gamma=0.5, beta_n=0.25.
    gamma, bn = 0.5, 0.25
    Keff = K + (gamma/(bn*dt))*C + (1.0/(bn*dt**2))*M
    Keff_inv = np.linalg.inv(Keff)
    u = np.zeros(ndof); v = np.zeros(ndof)
    a_base0 = 0.0
    acc = np.linalg.solve(M, -M@r*a_base0 - C@v - K@u)
    tip_hist = np.zeros(nsteps)
    times = np.zeros(nsteps)
    peak_stress = 0.0
    for i in range(nsteps):
        t = i*dt
        a_base = a_peak*np.sin(np.pi*t/tau) if t < tau else 0.0
        Feff = (-M@r*a_base
                + M@((1.0/(bn*dt**2))*u + (1.0/(bn*dt))*v + (1.0/(2*bn)-1.0)*acc)
                + C@((gamma/(bn*dt))*u + (gamma/bn-1.0)*v
                     + dt*(gamma/(2*bn)-1.0)*acc))
        u_new = Keff_inv @ Feff
        acc_new = (1.0/(bn*dt**2))*(u_new-u) - (1.0/(bn*dt))*v - (1.0/(2*bn)-1.0)*acc
        v = v + dt*((1-gamma)*acc + gamma*acc_new)
        u, acc = u_new, acc_new
        tip_hist[i] = u[-2]
        times[i] = t
        # Root element curvature -> bending stress at the fixed end.
        v1, th1, v2, th2 = 0.0, 0.0, u[0], u[1]
        kappa_root = (6/le**2)*(v2-v1) - (2/le)*(2*th1+th2)
        sigma = EI/barrel.I * abs(kappa_root) * barrel.c   # = E*c*kappa
        peak_stress = max(peak_stress, sigma)
    peak_tip = np.max(np.abs(tip_hist))
    sf = barrel.mat.yield_/peak_stress if peak_stress > 0 else 0.0
    return dict(a_peak_g=a_peak/G, peak_tip_um=peak_tip*1e6,
                peak_stress_MPa=peak_stress/1e6, safety_factor=sf,
                times=times, tip_hist=tip_hist)


# ── Figures ──────────────────────────────────────────────────────────────────
def node_x(barrel):
    return np.linspace(0, barrel.L, barrel.n_elem+1)


def plot_modes(barrel, freqs, phi, free, path):
    x = node_x(barrel)*1e3
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    colors = ["#2a78d6", "#e0873a", "#12a150"]
    for m in range(min(3, phi.shape[1])):
        full = np.zeros(2*(barrel.n_elem+1))
        full[free] = phi[:, m]
        v = full[0::2]
        v = v/np.max(np.abs(v))
        ax.plot(x, v, color=colors[m], lw=2,
                label=f"Mode {m+1}: {freqs[m]/1e3:.2f} kHz")
    ax.axhline(0, color="#aab", lw=.8)
    ax.set_xlabel("barrel axis [mm]"); ax.set_ylabel("normalized deflection")
    ax.set_title("Lens-barrel bending mode shapes (beam FE)")
    ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_thermal(barrel, u, free, tip, path):
    x = node_x(barrel)*1e3
    full = np.zeros(2*(barrel.n_elem+1)); full[free] = u
    v = full[0::2]*1e6
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(x, v, color="#b5532a", lw=2)
    ax.fill_between(x, 0, v, color="#b5532a", alpha=.12)
    ax.set_xlabel("barrel axis [mm]"); ax.set_ylabel("boresight deflection [µm]")
    ax.set_title(f"Thermal-gradient bending (tip = {tip*1e6:.2f} µm)")
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_shock(shock, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(shock["times"]*1e3, shock["tip_hist"]*1e6, color="#2a78d6", lw=1.4)
    ax.axhline(0, color="#aab", lw=.8)
    ax.set_xlabel("time [ms]"); ax.set_ylabel("tip displacement [µm]")
    ax.set_title(f"Drop-shock transient (peak {shock['peak_tip_um']:.1f} µm, "
                 f"{shock['peak_stress_MPa']:.1f} MPa, SF {shock['safety_factor']:.0f})")
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    figs = os.path.join(here, "figures")
    os.makedirs(figs, exist_ok=True)

    barrel = Barrel(mat=AL6061, n_elem=24)
    freqs, phi, free = modal(barrel, 3)
    cf = closed_form_f1(barrel)
    tip, analytic, u_th, free_th = thermal_bending(barrel, dT_gradient=5.0)
    shock = drop_shock(barrel)

    plot_modes(barrel, freqs, phi, free, os.path.join(figs, "barrel_modes.png"))
    plot_thermal(barrel, u_th, free_th, tip, os.path.join(figs, "barrel_thermal.png"))
    plot_shock(shock, os.path.join(figs, "barrel_shock.png"))

    out = {
        "material": barrel.mat.name,
        "n_elements": barrel.n_elem,
        "modal_hz": [round(float(f), 1) for f in freqs],
        "first_mode_hz": round(float(freqs[0]), 1),
        "closed_form_f1_hz": round(float(cf), 1),
        "modal_vs_closed_form_pct": round(100*abs(freqs[0]-cf)/cf, 2),
        "thermal_tip_um_per_5C_grad": round(float(tip*1e6), 3),
        "thermal_analytic_um": round(float(analytic*1e6), 3),
        "shock_a_peak_g": round(shock["a_peak_g"], 1),
        "shock_peak_tip_um": round(shock["peak_tip_um"], 2),
        "shock_peak_stress_MPa": round(shock["peak_stress_MPa"], 2),
        "shock_safety_factor": round(shock["safety_factor"], 1),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open(os.path.join(here, "barrel_fem_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
