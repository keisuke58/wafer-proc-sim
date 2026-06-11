"""Laser-induced bubble/void formation — the physics behind DISCO Stealth
Dicing (SD) and laser cavitation.

Two toy models:

1. SD thermal runaway (0D focal volume)
   Si is semi-transparent at 1064 nm, but its absorption coefficient grows
   steeply with temperature. Above a laser-intensity threshold the focal
   volume enters a positive feedback (absorb -> heat -> absorb more) and
   melts within the ns pulse — the origin of the SD modified layer whose
   resolidification leaves voids ("bubbles") + microcracks.
   Reference mechanism: Ohmura et al. (temperature-dependent absorption),
   Kumagai et al., "Advanced Dicing Technology for Semiconductor Wafer —
   Stealth Dicing", IEEE Trans. Semicond. Manuf. 20 (2007).

2. Rayleigh-Plesset cavitation bubble
   A laser focused in water creates a plasma whose expansion drives a vapor
   bubble: growth, violent collapse, rebounds. Classic R-P dynamics:
       rho ( R R'' + 3/2 R'^2 ) = p_B(R) - p_inf - 4 mu R'/R - 2 S/R

Run:  python demos/laser_bubble.py   -> results/laser_bubble.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# ----------------------------------------------------------------------
# 1. SD thermal runaway (lumped focal volume, ns pulse)
# ----------------------------------------------------------------------
T_ROOM = 300.0          # K
T_MELT = 1687.0         # K, silicon melting point
RHO_SI = 2329.0         # kg/m^3
CP_SI = 700.0           # J/(kg K)
ALPHA0 = 1.0e3          # 1/m  weak linear absorption of Si at 1064 nm, 300 K
T_ACT = 430.0           # K   activation scale of alpha(T) (fit-like toy value)
H_COND = 2.5e9          # W/(m^3 K) lumped conduction loss of ~um focal volume


def absorption_coeff(T):
    """Temperature-dependent absorption coefficient alpha(T) [1/m].

    Si band-edge absorption at 1064 nm rises steeply with temperature
    (band-gap narrowing + free carriers). Exponential toy form, capped to
    keep the ODE finite after melt onset.
    """
    return np.minimum(ALPHA0 * np.exp((np.asarray(T) - T_ROOM) / T_ACT), 1.0e8)


def focal_temperature(intensity_w_m2, t_pulse=150e-9, n_steps=3000):
    """Integrate dT/dt = [alpha(T) I - h (T - T_room)] / (rho cp).

    Returns (t, T) arrays; T is clipped at 1.5*T_MELT (model invalid beyond).
    """
    t = np.linspace(0.0, t_pulse, n_steps)
    dt = t[1] - t[0]
    T = np.empty_like(t)
    T[0] = T_ROOM
    for i in range(1, n_steps):
        heat = absorption_coeff(T[i - 1]) * intensity_w_m2
        loss = H_COND * (T[i - 1] - T_ROOM)
        T[i] = T[i - 1] + dt * (heat - loss) / (RHO_SI * CP_SI)
        if T[i] > 1.5 * T_MELT:
            T[i:] = 1.5 * T_MELT
            break
    return t, T


def runaway_threshold(i_lo=1e12, i_hi=1e15, tol=0.01):
    """Bisect the laser intensity [W/m^2] above which the focal volume melts
    within the pulse (thermal runaway -> void formation)."""
    def melts(intensity):
        return focal_temperature(intensity)[1].max() >= T_MELT

    if melts(i_lo) or not melts(i_hi):
        raise ValueError("bracket does not contain threshold")
    while i_hi / i_lo > 1.0 + tol:
        mid = np.sqrt(i_lo * i_hi)
        if melts(mid):
            i_hi = mid
        else:
            i_lo = mid
    return np.sqrt(i_lo * i_hi)


# ----------------------------------------------------------------------
# 2. Rayleigh-Plesset laser cavitation bubble
# ----------------------------------------------------------------------
RHO_W = 998.0       # kg/m^3
MU_W = 1.0e-3       # Pa s
S_W = 0.0728        # N/m surface tension
P_INF = 101325.0    # Pa
P_VAP = 2339.0      # Pa vapor pressure (20 C)
GAMMA = 1.4         # polytropic exponent of bubble gas


def rayleigh_plesset(t, y, r0, p_gas0):
    """R-P ODE. y = [R, dR/dt]; gas pressure p_g = p_gas0 (r0/R)^{3 gamma}."""
    r, v = y
    p_gas = p_gas0 * (r0 / r) ** (3.0 * GAMMA)
    p_b = p_gas + P_VAP - 2.0 * S_W / r - 4.0 * MU_W * v / r
    acc = (p_b - P_INF) / (RHO_W * r) - 1.5 * v**2 / r
    return [v, acc]


def simulate_bubble(r0=20e-6, p_gas0=5e6, t_end=60e-6, n_eval=4000):
    """Laser breakdown -> hot high-pressure nucleus of radius r0 expands.

    Returns (t, R). Default 5 MPa nucleus grows to ~x10 radius then
    collapses and rebounds.
    """
    sol = solve_ivp(
        rayleigh_plesset, (0.0, t_end), [r0, 0.0], args=(r0, p_gas0),
        t_eval=np.linspace(0.0, t_end, n_eval), method="LSODA",
        rtol=1e-8, atol=1e-12, max_step=t_end / 2000,
    )
    return sol.t, sol.y[0]


def max_radius(p_gas0, r0=20e-6):
    """Maximum bubble radius reached for a given nucleus pressure."""
    _, r = simulate_bubble(r0=r0, p_gas0=p_gas0)
    return float(r.max())


# ----------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------
def main():
    os.makedirs(RESULTS, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) thermal runaway: T(t) for intensities around the threshold
    i_th = runaway_threshold()
    print(f"SD thermal-runaway threshold: {i_th:.3e} W/m^2 "
          f"({i_th/1e4/1e9:.2f} GW/cm^2)")
    for f, c in [(0.6, "#4477aa"), (0.9, "#66ccee"), (1.1, "#ee6677"),
                 (1.5, "#aa3377")]:
        t, T = focal_temperature(f * i_th)
        axes[0].plot(t * 1e9, T, color=c, label=f"{f:.1f} x I_th")
    axes[0].axhline(T_MELT, ls="--", c="k", lw=0.8)
    axes[0].text(2, T_MELT * 1.03, "Si melt (1687 K)", fontsize=8)
    axes[0].set_xlabel("time [ns]")
    axes[0].set_ylabel("focal temperature [K]")
    axes[0].set_title("(a) SD thermal runaway\n(absorb->heat->absorb feedback)")
    axes[0].legend(fontsize=8)

    # (b) threshold sharpness: peak T vs intensity
    scan = np.linspace(0.4, 1.6, 60) * i_th
    peak = [focal_temperature(i)[1].max() for i in scan]
    axes[1].plot(scan / 1e13, peak, "k-")
    axes[1].axvline(i_th / 1e13, ls="--", c="r", lw=0.8, label="threshold")
    axes[1].axhline(T_MELT, ls="--", c="k", lw=0.8)
    axes[1].set_xlabel("laser intensity [$10^{13}$ W/m$^2$]")
    axes[1].set_ylabel("peak temperature [K]")
    axes[1].set_title("(b) melt threshold — why only\nthe focal point is processed")
    axes[1].legend(fontsize=8)

    # (c) cavitation bubble R(t)
    for p0, c in [(2e6, "#4477aa"), (5e6, "#ee6677"), (10e6, "#aa3377")]:
        t, r = simulate_bubble(p_gas0=p0)
        axes[2].plot(t * 1e6, r * 1e6, color=c, label=f"{p0/1e6:.0f} MPa nucleus")
    axes[2].set_xlabel("time [us]")
    axes[2].set_ylabel("bubble radius [um]")
    axes[2].set_title("(c) laser cavitation bubble\n(Rayleigh-Plesset: grow/collapse/rebound)")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out = os.path.abspath(os.path.join(RESULTS, "laser_bubble.png"))
    fig.savefig(out, dpi=150)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
