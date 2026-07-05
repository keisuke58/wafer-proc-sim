#!/usr/bin/env python3
"""spin_clean.py — single-wafer wet cleaning: the physics inside the chamber.

Semiconductor cleaning looks mundane and is anything but: at advanced nodes
the cleaner must remove sub-30 nm particles WITHOUT eating more than a few
angstroms of film and WITHOUT collapsing 10:1 patterns. Four coupled pieces
of physics, all implemented from scratch:

  1. PARTICLE REMOVAL — adhesion (van der Waals, ∝ d) against hydrodynamic
     drag in the spin boundary layer (∝ d²): drag loses as d shrinks, which
     is WHY small particles are the industry's core problem. Removal is a
     rolling-moment criterion with lognormal adhesion scatter → PRE(d)
     curves. Megasonic assist: modeled as cavitation microjet impulses
     (the accepted dominant mechanism; lognormal strength, hit
     probability) on top of the streaming shear — still ∝ d², so the
     smallest particles remain the hardest.
  2. CHEMICAL UNDERCUT — SC-1 etches under the particle contact; once the
     etch depth passes the contact radius the particle releases. PRE jumps
     with etch amount — but etch IS material loss, and the loss budget at
     advanced nodes is ~0.1 nm/clean. The tradeoff curve is the product.
  3. PATTERN COLLAPSE — capillary pressure 2σcosθ/S bends neighboring
     lines together during drying; cantilever bending gives the maximum
     survivable aspect ratio vs surface tension: water 72 → IPA 21.7 →
     supercritical CO₂ ~0 mN/m.
  4. SPIN DYNAMICS — Emslie-Bonner-Peck film thinning h(t) (validated
     against the analytic solution), rinse/dry timing, wafers-per-hour.

Pure numpy + matplotlib.  Run:  python3 cleaning/spin_clean.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "spin_clean_results.json")

# ── constants ───────────────────────────────────────────────────────────────
HAMAKER = 1.0e-20        # J (SiO2-particle-water)
Z0 = 0.4e-9              # adhesion separation [m]
A_CONTACT = 0.8e-9       # contact/deformation radius [m] (lever arm)
RHO = 1000.0             # water [kg/m3]
MU = 0.9e-3              # water viscosity [Pa s]
NU = MU / RHO
OMEGA_SPIN = 2 * np.pi * 1500 / 60.0   # 1500 rpm
R_EVAL = 0.10            # evaluation radius on the wafer [m]
F_MEGA = 1.0e6           # megasonic frequency [Hz]
U_STREAM = 0.05          # acoustic streaming velocity near surface [m/s]
SIGMA_LN = 0.55          # lognormal scatter of adhesion (surface roughness)
# megasonic cavitation: transient microjet impulse pressure on the particle
# cross-section — the accepted dominant removal mechanism, far stronger than
# streaming shear for small particles (still ∝ d², so smaller still loses)
P_JET_MED = 5.0e4        # median effective microjet pressure [Pa]
P_JET_LN = 1.1           # lognormal spread of jet strength / proximity
P_HIT = 0.9              # probability a cavitation event visits a site

E_SI = 130e9             # line material Young's modulus [Pa]
SIGMA_WATER = 0.072      # surface tension [N/m]
SIGMA_IPA = 0.0217
SIGMA_SCCO2 = 0.0005
COS_THETA = 1.0


def adhesion_force(d_m):
    """vdW sphere-plane: F = A d / (12 z0²)  [N]."""
    return HAMAKER * d_m / (12.0 * Z0 ** 2)


def wall_shear_spin(omega=OMEGA_SPIN, r=R_EVAL):
    """Rotating-disk boundary layer wall shear ~ 0.8 μ r sqrt(ω³/ν)."""
    return 0.8 * MU * r * np.sqrt(omega ** 3 / NU)


def wall_shear_megasonic(u=U_STREAM, f=F_MEGA):
    """Acoustic (Schlichting) streaming: δ = sqrt(2ν/ω), τ = μ u / δ."""
    delta = np.sqrt(2.0 * NU / (2 * np.pi * f))
    return MU * u / delta


def drag_force(d_m, tau):
    """O'Neill drag on a wall-attached sphere in shear flow: F ≈ 32 τ r²."""
    return 32.0 * tau * (d_m / 2.0) ** 2


def removal_efficiency(d_nm, tau, etch_nm=0.0, megasonic=False,
                       n_mc=20000, seed=0):
    """PRE via rolling-moment criterion with lognormal adhesion scatter.

    Rolling: removal moment vs adhesion moment F_a·a_contact. The removal
    moment is hydrodynamic drag (∝ τ d²) plus, with megasonic on, the
    cavitation microjet impulse (∝ P_jet d², lognormal strength, hit
    probability P_HIT). SC-1 undercut shrinks the effective contact:
    a_eff = a - etch (released outright once the etch eats the contact)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d_nm, dtype=float) * 1e-9
    out = np.empty(d.shape)
    for i, dd in np.ndenumerate(d):
        a_eff = max(A_CONTACT - etch_nm * 1e-9, 0.0)
        if a_eff == 0.0:
            out[i] = 1.0
            continue
        fa = adhesion_force(dd) * rng.lognormal(0.0, SIGMA_LN, n_mc)
        f_rem = np.full(n_mc, drag_force(dd, tau))
        if megasonic:
            hit = rng.random(n_mc) < P_HIT
            p_jet = P_JET_MED * rng.lognormal(0.0, P_JET_LN, n_mc)
            f_rem = f_rem + hit * p_jet * np.pi * (dd / 2.0) ** 2
        out[i] = float(np.mean(f_rem * 1.4 * dd / 2.0 > fa * a_eff))
    return out


def material_loss_nm(etch_nm, n_cleans=1):
    return etch_nm * n_cleans


# ── pattern collapse ────────────────────────────────────────────────────────
def max_aspect_ratio(sigma, w_nm=15.0, s_nm=15.0, E=E_SI):
    """Cantilever criterion: capillary ΔP = 2σcosθ/S bends a line of width
    w; tip deflection (uniform load) δ = 3 ΔP H⁴ / (2 E w³). Collapse when
    δ > S/2 → AR_max = H/w with H = (E w³ S / (3·2σcosθ/S·... ))^(1/4)."""
    w = w_nm * 1e-9
    s = s_nm * 1e-9
    dp = 2.0 * sigma * COS_THETA / s
    h4 = (s / 2.0) * (2.0 * E * w ** 3) / (3.0 * dp)
    return float(h4 ** 0.25 / w)


# ── spin film dynamics (Emslie–Bonner–Peck) ────────────────────────────────
def film_thinning(h0_um=50.0, omega=OMEGA_SPIN, t_end=8.0, n=400):
    """dh/dt = -2ρω²h³/(3μ); analytic h(t) = h0/sqrt(1+4ρω²h0²t/(3μ))."""
    t = np.linspace(0, t_end, n)
    h0 = h0_um * 1e-6
    k = RHO * omega ** 2 / MU
    h_analytic = h0 / np.sqrt(1.0 + 4.0 * k * h0 ** 2 * t / 3.0)
    # explicit integration as an independent check
    h_num = np.empty(n)
    h = h0
    dt = t[1] - t[0]
    for i in range(n):
        h_num[i] = h
        for _ in range(20):                      # sub-steps for stiffness
            h = h - dt / 20.0 * (2.0 * k / 3.0) * h ** 3
    return t, h_analytic * 1e6, h_num * 1e6


def throughput_wph(t_chem=30.0, t_rinse=20.0, t_dry=15.0, t_handle=8.0,
                   n_chambers=8):
    t_total = t_chem + t_rinse + t_dry + t_handle
    return 3600.0 / t_total * n_chambers


# ── headline numbers ────────────────────────────────────────────────────────
def run():
    tau_s = wall_shear_spin()
    tau_m = tau_s + wall_shear_megasonic()
    sizes = np.array([20.0, 30.0, 50.0, 100.0])
    res = {
        "wall_shear_spin_pa": round(float(tau_s), 1),
        "wall_shear_megasonic_pa": round(float(wall_shear_megasonic()), 1),
        "pre_spin": {f"{int(s)}nm": round(float(
            removal_efficiency([s], tau_s)[0]), 3) for s in sizes},
        "pre_megasonic": {f"{int(s)}nm": round(float(
            removal_efficiency([s], tau_s, megasonic=True)[0]), 3)
            for s in sizes},
        "pre_mega_sc1_0p3nm": {f"{int(s)}nm": round(float(
            removal_efficiency([s], tau_s, etch_nm=0.3,
                               megasonic=True)[0]), 3)
            for s in sizes},
        "ar_max": {"water": round(max_aspect_ratio(SIGMA_WATER), 2),
                   "ipa": round(max_aspect_ratio(SIGMA_IPA), 2),
                   "scco2": round(max_aspect_ratio(SIGMA_SCCO2), 2)},
        "throughput_wph": round(throughput_wph(), 1),
        # dicing cross-over: kerf debris is micron-scale, spin suffices
        "pre_spin_kerf_debris": {
            "500nm": round(float(removal_efficiency([500.0], tau_s)[0]), 3),
            "1000nm": round(float(
                removal_efficiency([1000.0], tau_s)[0]), 3),
        },
    }
    return res


# ── figures ─────────────────────────────────────────────────────────────────
def _plot_pre(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.linspace(15, 150, 60)
    tau_s = wall_shear_spin()
    tau_m = tau_s + wall_shear_megasonic()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ax = axes[0]
    dd = np.logspace(np.log10(15e-9), np.log10(200e-9), 50)
    ax.loglog(dd * 1e9, adhesion_force(dd) * 1e9, color="#c44", lw=1.8,
              label="adhesion (vdW) ~ d")
    ax.loglog(dd * 1e9, drag_force(dd, tau_s) * 1e9, color="#26a", lw=1.8,
              label="spin drag ~ d^2")
    ax.loglog(dd * 1e9, drag_force(dd, tau_m) * 1e9, color="#2a7", lw=1.8,
              label="spin + megasonic drag")
    ax.set_xlabel("particle diameter [nm]")
    ax.set_ylabel("force [nN]")
    ax.set_title("Why small particles win: d vs d^2 scaling")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(d, removal_efficiency(d, tau_s) * 100, color="#26a", lw=1.8,
            label="spin only (1500 rpm)")
    ax.plot(d, removal_efficiency(d, tau_s, megasonic=True) * 100,
            color="#2a7", lw=1.8, label="+ megasonic (1 MHz, cavitation)")
    ax.plot(d, removal_efficiency(d, tau_s, etch_nm=0.3,
                                  megasonic=True) * 100,
            color="#b4740a", lw=1.8, label="+ SC-1 undercut 0.3 nm")
    ax.axhline(95, color="k", ls=":", lw=1, label="95% PRE target")
    ax.set_xlabel("particle diameter [nm]")
    ax.set_ylabel("removal efficiency PRE [%]")
    ax.set_title("PRE vs size: physics, assist, chemistry")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_chem_collapse(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ax = axes[0]
    tau_m = wall_shear_spin() + wall_shear_megasonic()
    etch = np.linspace(0.0, 1.5, 30)
    pre30 = [removal_efficiency([30.0], wall_shear_spin(), etch_nm=e,
                                megasonic=True)[0] * 100 for e in etch]
    ax.plot(etch, pre30, color="#26a", lw=1.8)
    ax.set_xlabel("SC-1 etch per clean [nm]")
    ax.set_ylabel("PRE at 30 nm [%]", color="#26a")
    ax.axvspan(0, 0.1, color="0.9", label="advanced-node loss budget")
    ax2 = ax.twinx()
    ax2.plot(etch, material_loss_nm(etch, n_cleans=20), color="#c44",
             lw=1.8, ls="--")
    ax2.set_ylabel("cumulative loss after 20 cleans [nm]", color="#c44")
    ax.set_title("Chemistry helps until the loss budget stops you")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    sig = np.logspace(np.log10(3e-4), np.log10(0.075), 60)
    ax.semilogx(sig * 1000, [max_aspect_ratio(s) for s in sig], color="#26a",
                lw=1.8)
    for s, name, c in ((SIGMA_WATER, "water", "#c44"),
                       (SIGMA_IPA, "IPA", "#b4740a"),
                       (SIGMA_SCCO2, "scCO2", "#2a7")):
        ax.plot([s * 1000], [max_aspect_ratio(s)], "o", ms=8, color=c)
        ax.annotate(f"{name}: AR {max_aspect_ratio(s):.1f}",
                    (s * 1000, max_aspect_ratio(s)),
                    textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax.set_xlabel("surface tension [mN/m]")
    ax.set_ylabel("max survivable aspect ratio (15 nm lines)")
    ax.set_title("Pattern collapse: why drying fluid choice IS the spec")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_dicing(path):
    """DISCO cross-over figure: kerf-debris sizes (0.2-5 um) vs front-end
    particle sizes — why a dicer's spinner needs no megasonic while a
    front-end cleaner cannot live without it. Same physics, one curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau = wall_shear_spin()
    d = np.logspace(np.log10(20), np.log10(5000), 60)
    pre_s = removal_efficiency(d, tau) * 100
    pre_m = removal_efficiency(d, tau, megasonic=True) * 100

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.semilogx(d, pre_s, color="#26a", lw=2, label="spin only (1500 rpm)")
    ax.semilogx(d, pre_m, color="#2a7", lw=1.6, ls="--",
                label="+ megasonic")
    ax.axvspan(200, 5000, color="#26a", alpha=0.08)
    ax.axvspan(20, 100, color="#c44", alpha=0.08)
    ax.text(950, 30, "kerf debris after dicing\n(0.2-5 um):\nspin alone"
            " suffices", fontsize=9, ha="center", color="#26a")
    ax.text(45, 30, "front-end\nparticles\n(<100 nm):\nneeds assist",
            fontsize=9, ha="center", color="#c44")
    ax.axhline(95, color="k", ls=":", lw=1)
    ax.set_xlabel("particle diameter [nm]")
    ax.set_ylabel("removal efficiency PRE [%]")
    ax.set_title("One scaling law, two machines: adhesion ~ d vs drag ~ d^2")
    ax.legend(fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_spin(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, ha, hn = film_thinning()
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.semilogy(t, ha, color="#26a", lw=2, label="analytic (EBP)")
    ax.semilogy(t, hn, "--", color="#c44", lw=1.4, label="numerical check")
    ax.set_xlabel("spin time [s]")
    ax.set_ylabel("film thickness [um]")
    ax.set_title(f"Rinse film thinning at 1500 rpm - "
                 f"throughput {throughput_wph():.0f} wafers/h (8 chambers)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot_pre(os.path.join(ASSETS, "clean_pre.png"))
    _plot_chem_collapse(os.path.join(ASSETS, "clean_chem.png"))
    _plot_spin(os.path.join(ASSETS, "clean_spin.png"))
    _plot_dicing(os.path.join(ASSETS, "clean_dicing.png"))
    print("figures ->", ASSETS)


if __name__ == "__main__":
    main()
