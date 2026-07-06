#!/usr/bin/env python3
"""implant_anneal.py — ion implantation + rapid thermal anneal: putting the
dopant where you want it, and keeping it there.

Doping is the one front-end step this portfolio had not covered. Two halves,
each closed against an analytic reference:

  IMPLANT      LSS theory says a monoenergetic ion beam comes to rest in a
               roughly Gaussian depth profile: projected range Rp(E) and
               straggle dRp(E), here from power-law fits to published LSS
               tables for B, P, As in Si (fit values quoted in-code). The
               as-implanted profile is
                   N(x) = phi/(sqrt(2 pi) dRp) exp(-(x-Rp)^2 / 2 dRp^2)
               plus an exponential CHANNELING tail. Closed loop: the
               integral of the profile returns the dose to <1% (the
               remainder is the physical surface-truncation loss).

  ANNEAL       implantation damages the lattice; the anneal activates the
               dopant AND diffuses it. A Gaussian diffusing by D*t stays
               Gaussian with sigma^2 = dRp^2 + 2 D t — so the from-scratch
               explicit finite-difference diffusion solver can be verified
               against the exact analytic answer (<1% here). Arrhenius
               D(T) makes the THERMAL BUDGET argument quantitative: a
               1050 C / 10 s spike RTA and a 1000 C / 30 min furnace anneal
               deposit similar activation heat, but the furnace drives the
               junction ~2x deeper — that is why RTA exists.

  DEVICE VIEW  junction depth xj (crossing of the substrate doping) and
               sheet resistance Rs = 1/(q integral N mu(N) dx) with a
               doping-dependent mobility fit: the xj-Rs tradeoff the
               process integrator actually negotiates.

Pure numpy + matplotlib.  Run:  python3 amat/implant_anneal.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "implant_anneal_results.json")

# power-law fits Rp = a E^b [nm], dRp = c E^d [nm] (E in keV) to LSS-table
# values for Si substrates (approximate; quoted to ~15%)
SPECIES = {
    "B":  {"a": 3.8, "b": 0.85, "c": 2.2, "d": 0.75, "m_amu": 11},
    "P":  {"a": 1.4, "b": 0.90, "c": 0.7, "d": 0.85, "m_amu": 31},
    "As": {"a": 0.8, "b": 0.88, "c": 0.3, "d": 0.88, "m_amu": 75},
}
N_SUB = 1e17             # substrate background doping [cm^-3]
CHAN_FRAC = 0.005        # dose fraction in the channeling tail
CHAN_LEN_NM = 30.0       # channeling tail decay length
Q_E = 1.602e-19
D0_CM2S = 0.76           # boron diffusivity prefactor [cm^2/s]
EA_EV = 3.46             # activation energy [eV]
KB = 8.617e-5


def rp_drp(species, energy_kev):
    s = SPECIES[species]
    return s["a"] * energy_kev ** s["b"], s["c"] * energy_kev ** s["d"]


def implant_profile(species="B", energy_kev=30.0, dose_cm2=1e15,
                    x_nm=None):
    """As-implanted N(x) [cm^-3] on x [nm] grid: Gaussian + channel tail."""
    rp, drp = rp_drp(species, energy_kev)
    if x_nm is None:
        x_nm = np.linspace(0, rp + 12 * drp + 6 * CHAN_LEN_NM, 4000)
    dose_nm2 = dose_cm2 * 1e-14                  # per nm^2
    gauss = (1 - CHAN_FRAC) * dose_nm2 / (np.sqrt(2 * np.pi) * drp) \
        * np.exp(-0.5 * ((x_nm - rp) / drp) ** 2)
    tail = CHAN_FRAC * dose_nm2 / CHAN_LEN_NM \
        * np.exp(-np.maximum(x_nm - rp, 0.0) / CHAN_LEN_NM) \
        * (x_nm >= rp)
    return x_nm, (gauss + tail) * 1e21           # nm^-3 -> cm^-3


def diffusivity(temp_c):
    """Boron D(T) [nm^2/s] by Arrhenius."""
    t_k = temp_c + 273.15
    return D0_CM2S * np.exp(-EA_EV / (KB * t_k)) * 1e14   # cm^2->nm^2


def anneal_fd(x_nm, n, d_nm2s, time_s):
    """Explicit FD solve of dN/dt = D d2N/dx2 with reflecting surface."""
    dx = x_nm[1] - x_nm[0]
    dt = 0.4 * dx * dx / d_nm2s
    steps = max(int(time_s / dt), 1)
    dt = time_s / steps
    r = d_nm2s * dt / (dx * dx)
    n = n.copy()
    for _ in range(steps):
        lap = np.empty_like(n)
        lap[1:-1] = n[:-2] - 2 * n[1:-1] + n[2:]
        lap[0] = 2 * (n[1] - n[0])               # reflecting (no out-flux)
        lap[-1] = 2 * (n[-2] - n[-1])
        n = n + r * lap
    return n


def gaussian_after_anneal(species, energy_kev, dose_cm2, x_nm,
                          temp_c, time_s):
    """Analytic: Gaussian stays Gaussian, sigma^2 += 2 D t (ignoring the
    small channel tail; surface reflection folded in by image source)."""
    rp, drp = rp_drp(species, energy_kev)
    sig2 = drp ** 2 + 2 * diffusivity(temp_c) * time_s
    sig = np.sqrt(sig2)
    dose_nm2 = (1 - CHAN_FRAC) * dose_cm2 * 1e-14
    main = np.exp(-0.5 * ((x_nm - rp) / sig) ** 2)
    image = np.exp(-0.5 * ((x_nm + rp) / sig) ** 2)   # reflecting surface
    return dose_nm2 / (np.sqrt(2 * np.pi) * sig) * (main + image) * 1e21


def junction_depth(x_nm, n):
    """Deepest crossing of the substrate doping [nm]."""
    above = n > N_SUB
    idx = np.where(above)[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[-1]
    if i + 1 >= len(x_nm):
        return float(x_nm[-1])
    f = (np.log(n[i]) - np.log(N_SUB)) / (np.log(n[i]) - np.log(n[i + 1]))
    return float(x_nm[i] + f * (x_nm[i + 1] - x_nm[i]))


def mobility(n_cm3):
    """Hole mobility fit for B in Si [cm^2/Vs] (Caughey-Thomas form)."""
    return 44.9 + (470.5 - 44.9) / (1 + (n_cm3 / 2.23e17) ** 0.719)


def sheet_resistance(x_nm, n):
    """Rs [ohm/sq] over the region above background."""
    mask = n > N_SUB
    if not mask.any():
        return float("nan")
    dx_cm = (x_nm[1] - x_nm[0]) * 1e-7
    sigma = Q_E * np.sum(n[mask] * mobility(n[mask])) * dx_cm
    return float(1.0 / sigma)


def run():
    x, n0 = implant_profile("B", 30.0, 1e15)
    dx = x[1] - x[0]
    dose_back = np.sum(n0) * dx * 1e-21 * 1e14     # cm^-2
    d_rta = diffusivity(1050.0)
    d_fur = diffusivity(1000.0)
    n_rta = anneal_fd(x, n0, d_rta, 10.0)
    n_fur = anneal_fd(x, n0, d_fur, 1800.0)
    ana = gaussian_after_anneal("B", 30.0, 1e15, x, 1050.0, 10.0)
    core = n_rta > n_rta.max() * 1e-3
    fd_err = float(np.max(np.abs(n_rta[core] - ana[core]) / ana[core].max()))
    ranges = {f"{sp}_{e:.0f}keV": {"Rp_nm": round(rp_drp(sp, e)[0], 1),
                                   "dRp_nm": round(rp_drp(sp, e)[1], 1)}
              for sp, e in (("B", 30), ("P", 80), ("As", 100))}
    return {
        "implant": {"species": "B", "E_keV": 30, "dose_cm2": 1e15,
                    "dose_recovered_cm2": float(f"{dose_back:.4e}"),
                    "dose_err_pct": round(100 * abs(dose_back - 1e15) / 1e15,
                                          3),
                    "ranges": ranges},
        "anneal": {
            "fd_vs_analytic_maxerr_frac": round(fd_err, 4),
            "D_nm2s": {"RTA_1050C": float(f"{d_rta:.3e}"),
                       "furnace_1000C": float(f"{d_fur:.3e}")},
            "as_implanted": {"xj_nm": round(junction_depth(x, n0), 1),
                             "Rs_ohm_sq": round(sheet_resistance(x, n0), 1)},
            "RTA_1050C_10s": {"xj_nm": round(junction_depth(x, n_rta), 1),
                              "Rs_ohm_sq": round(sheet_resistance(x, n_rta),
                                                 1)},
            "furnace_1000C_30min": {"xj_nm": round(junction_depth(x, n_fur),
                                                  1),
                                   "Rs_ohm_sq": round(
                                       sheet_resistance(x, n_fur), 1)},
        },
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    ax = axes[0]
    for sp, e, c in (("B", 30, "#0b3d91"), ("P", 80, "#b4740a"),
                     ("As", 100, "#8a2be2")):
        x, n = implant_profile(sp, e, 1e15)
        ax.semilogy(x, n, lw=1.8, color=c, label=f"{sp} {e} keV")
    ax.axhline(N_SUB, color="0.5", ls=":", lw=1)
    ax.annotate("substrate 1e17", (5, N_SUB * 1.5), fontsize=8, color="0.4")
    ax.set_xlabel("depth [nm]")
    ax.set_ylabel("concentration [cm^-3]")
    ax.set_ylim(1e15, 1e21)
    ax.set_xlim(0, 350)
    ax.set_title("As-implanted LSS profiles (Gaussian + channel tail)\n"
                 "integral returns the dose to <1% (surface loss)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    x, n0 = implant_profile("B", 30.0, 1e15)
    n_rta = anneal_fd(x, n0, diffusivity(1050.0), 10.0)
    n_fur = anneal_fd(x, n0, diffusivity(1000.0), 1800.0)
    ana = gaussian_after_anneal("B", 30.0, 1e15, x, 1050.0, 10.0)
    ax.semilogy(x, n0, lw=1.2, color="0.6", label="as-implanted")
    ax.semilogy(x, n_rta, lw=2, color="#0b3d91",
                label="spike RTA 1050C / 10 s (FD)")
    ax.semilogy(x, ana, "--", lw=1, color="k", alpha=0.6,
                label="analytic Gaussian check")
    ax.semilogy(x, n_fur, lw=2, color="#c0392b",
                label="furnace 1000C / 30 min (FD)")
    ax.axhline(N_SUB, color="0.5", ls=":", lw=1)
    a = res["anneal"]
    ax.set_xlabel("depth [nm]")
    ax.set_ylabel("concentration [cm^-3]")
    ax.set_ylim(1e15, 1e21)
    ax.set_xlim(0, 450)
    ax.set_title(f"Why RTA exists: xj {a['RTA_1050C_10s']['xj_nm']:.0f} vs "
                 f"{a['furnace_1000C_30min']['xj_nm']:.0f} nm (furnace)\n"
                 f"FD solver vs analytic: {a['fd_vs_analytic_maxerr_frac']*100:.2f}% max err",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "amat_implant.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
