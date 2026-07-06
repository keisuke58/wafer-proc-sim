#!/usr/bin/env python3
"""resist_develop.py — from dissolution chemistry to the sidewall angle:
the Mack model, implemented and closed against its analytic limits.

The second half of the resist story (the first half is
fujifilm/car_stochastics.py): once exposure has produced a deprotection
map m(x, z), DEVELOPMENT is a rate process that turns chemistry contrast
into a geometric profile.

  MACK MODEL   dissolution rate vs remaining protection m:
                 r(m) = r_max (a+1)(1-m)^n / (a + (1-m)^n) + r_min
               with a = (n+1)/(n-1) (1-m_th)^n. The exponent n is the
               dissolution SELECTIVITY — the resist designer's main knob.

  EXPOSURE     m(x, z) = exp(-C E I(x) e^{-alpha z}): first-order Dill
               kinetics; absorption alpha makes the bottom see less light,
               so the latent image is weaker at the resist foot.

  CONTRAST     gamma = dlog(thickness remaining)/dlog(E) at the clearing
               dose. The theoretical characteristic curve is swept from
               the same rate model — the closed loop: dose-to-clear E0
               must satisfy integral dz / r(m(E0, z)) = t_dev exactly.

  PROFILE      per-column development front: time to clear depth z is
               t(x, z) = integral_0^z dz'/r(m(x, z')); the contour
               t = t_dev is the resist profile, and its slope at
               mid-height is the SIDEWALL ANGLE. Sweep n: a low-n resist
               prints a ramp, a high-n resist prints a wall — same aerial
               image, different chemistry.

Pure numpy + matplotlib.  Run:  python3 fujifilm/resist_develop.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "resist_develop_results.json")

PITCH_NM = 400.0         # relaxed line/space (profile physics, not res.)
K_IMG = 0.8              # aerial image contrast
THICK_NM = 300.0         # resist thickness
ALPHA_UM = 0.6           # absorption [1/um]
DILL_C = 0.9             # exposure rate constant [cm^2/mJ-ish, relative]
R_MAX = 120.0            # nm/s
R_MIN = 0.08             # nm/s
M_TH = 0.6               # Mack threshold protection
T_DEV = 45.0             # develop time [s]
NZ = 240
NX = 320


def mack_rate(m, n=8.0, r_max=R_MAX, r_min=R_MIN, m_th=M_TH):
    """Mack development rate [nm/s]."""
    a = (n + 1.0) / (n - 1.0) * (1.0 - m_th) ** n
    w = (1.0 - np.asarray(m, dtype=float)) ** n
    return r_max * (a + 1.0) * w / (a + w) + r_min


def m_map(dose, x=None, z=None):
    """Deprotection map m(x, z) after exposure (Dill first-order)."""
    if x is None:
        x = np.linspace(-PITCH_NM / 2, PITCH_NM / 2, NX)
    if z is None:
        z = np.linspace(0, THICK_NM, NZ)
    I = 0.5 * (1.0 + K_IMG * np.cos(2 * np.pi * x / PITCH_NM))
    att = np.exp(-ALPHA_UM * z[:, None] / 1000.0)   # alpha[1/um]*z[nm]
    return x, z, np.exp(-DILL_C * dose * I[None, :] * att)


def clear_time(dose, n=8.0):
    """Time to clear the full thickness at each x: integral dz / r."""
    x, z, m = m_map(dose)
    dz = z[1] - z[0]
    r = mack_rate(m, n)
    return x, z, np.cumsum(dz / r, axis=0)


def dose_to_clear(n=8.0):
    """E0 of the BRIGHT column (x = 0, the cosine peak): bisection on
    full-thickness clear time = T_DEV."""
    lo, hi = 0.1, 30.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        _, _, t = clear_time(mid, n)
        if t[-1, NX // 2] < T_DEV:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def characteristic_curve(n=8.0, doses=None):
    """Thickness remaining after T_DEV in open frame (I=1) vs dose;
    gamma = -dlog10(T)/dlog10(E) slope at half thickness."""
    doses = np.logspace(-1.6, 0.6, 220) if doses is None else doses
    z = np.linspace(0, THICK_NM, NZ)
    dz = z[1] - z[0]
    rem = np.empty(len(doses))
    for i, e in enumerate(doses):
        m = np.exp(-DILL_C * e * np.exp(-ALPHA_UM * z / 1000.0))
        t = np.cumsum(dz / mack_rate(m, n))
        cleared = np.searchsorted(t, T_DEV)
        rem[i] = max(THICK_NM - cleared * dz, 0.0) / THICK_NM
    # gamma = slope of the remaining-thickness cliff on the log-dose axis,
    # from the interpolated doses where 90 % and 10 % remain
    ld = np.log10(doses)
    e90 = np.interp(-0.9, -rem, ld)      # rem is decreasing in dose
    e10 = np.interp(-0.1, -rem, ld)
    g = 0.8 / (e10 - e90) if e10 > e90 else float("nan")
    return doses, rem, float(g)


def profile(dose, n=8.0):
    """Resist profile: for each x the depth cleared at T_DEV. Returns
    (x, depth_cleared[x]) and the mid-height sidewall angle [deg]."""
    x, z, t = clear_time(dose, n)
    dz = z[1] - z[0]
    depth = np.array([np.searchsorted(t[:, j], T_DEV) * dz
                      for j in range(t.shape[1])])
    depth = np.minimum(depth, THICK_NM)
    # sidewall angle where the front crosses mid-thickness
    half = THICK_NM / 2
    j = np.argmin(np.abs(depth - half))
    if 0 < j < len(x) - 1:
        slope = (depth[j + 1] - depth[j - 1]) / (x[j + 1] - x[j - 1])
        ang = np.degrees(np.arctan(np.abs(slope)))
    else:
        ang = float("nan")
    return x, depth, float(ang)


def run():
    out = {"mack": {"n_sweep": {}, "r_max": R_MAX, "r_min": R_MIN,
                    "m_th": M_TH, "t_dev_s": T_DEV}}
    for n in (2.0, 8.0, 16.0):
        e0 = dose_to_clear(n)
        _, _, g = characteristic_curve(n)
        _, _, ang = profile(1.35 * e0, n)
        out["mack"]["n_sweep"][f"n{n:.0f}"] = {
            "dose_to_clear": round(e0, 3),
            "gamma": round(g, 2),
            "sidewall_deg": round(ang, 1)}
    # closed loop: at E0 the brightest column clears in exactly T_DEV
    e0 = dose_to_clear(8.0)
    _, _, t = clear_time(e0, 8.0)
    out["closed_loop_clear_time_err_pct"] = round(
        100.0 * abs(t[-1, -1] - T_DEV) / T_DEV, 2)
    return out


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.5, 4.3))
    ax = fig.add_subplot(1, 3, 1)
    mm = np.linspace(0, 1, 300)
    for n, c in ((2.0, "#888"), (8.0, "#b4740a"), (16.0, "#00794f")):
        ax.semilogy(mm, mack_rate(mm, n), lw=1.8, color=c,
                    label=f"n={n:.0f}")
    ax.set_xlabel("protection m (1 = unexposed)")
    ax.set_ylabel("develop rate r(m) [nm/s]")
    ax.set_title("Mack dissolution: selectivity n is\nthe chemistry knob",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(1, 3, 2)
    for n, c in ((2.0, "#888"), (8.0, "#b4740a"), (16.0, "#00794f")):
        d, rem, g = characteristic_curve(n)
        ax.semilogx(d, rem, lw=1.8, color=c,
                    label=f"n={n:.0f}: gamma={g:.1f}")
    ax.set_xlabel("dose [rel]")
    ax.set_ylabel("thickness remaining [frac]")
    ax.set_title("Characteristic curve: contrast gamma\nfrom the same "
                 "rate model", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(1, 3, 3)
    for n, c in ((2.0, "#888"), (8.0, "#b4740a"), (16.0, "#00794f")):
        e0 = dose_to_clear(n)
        x, depth, ang = profile(1.35 * e0, n)
        ax.plot(x, THICK_NM - depth, lw=1.8, color=c,
                label=f"n={n:.0f}: sidewall {ang:.0f} deg")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("resist height [nm]")
    ax.set_title("Same aerial image, different chemistry:\nramp vs wall",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "ff_develop.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
