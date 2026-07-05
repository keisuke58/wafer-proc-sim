#!/usr/bin/env python3
"""nand_sim.py — 3D NAND from program pulse to endurance rating.

A flash product is a negotiation between four pieces of physics, simulated
here end-to-end at the cell-population level:

  ISPP        Incremental Step Pulse Programming: pulse-verify loops walk
              each cell's Vth up in ΔVpgm steps until it passes verify. The
              resulting per-state distribution is ~uniform(ΔVpgm) ⊕ noise —
              TLC packs 8 such states into ~5 V of window.
  CYCLING     every program/erase cycle generates tunnel-oxide traps:
              distributions WIDEN (σ ∝ cycles^0.6) and the erase state
              creeps up — the window closes from both sides.
  RETENTION   trapped charge detraps over time (∝ log t), Arrhenius-
              accelerated by temperature: distributions slide down and
              smear after bake.
  ECC         the product is not "zero errors" but "correctable errors":
              raw BER from adjacent-state overlap (analytic erfc), then
              BCH-72/1KiB (binomial tail) or an LDPC-class limit turns
              RBER into UBER. ENDURANCE RATING = the P/E count where
              1-year-retention RBER still sits under the ECC limit.

Pure numpy + scipy + matplotlib.  Run:  python3 memdev/nand_sim.py
"""
import json
import os

import numpy as np
from scipy.special import erfc
from scipy.stats import binom

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "nand_sim_results.json")

# ── cell / array constants (TLC) ────────────────────────────────────────────
N_LEVELS = 8
V_ERASE = -2.5            # erased-state mean [V]
V_TARGETS = np.linspace(0.0, 4.2, N_LEVELS - 1)   # program verify levels
DV_PGM = 0.25             # ISPP step [V]
SIGMA0 = 0.05             # intrinsic Vth noise (fresh) [V]
K_CYC_SIG = 5.5e-4        # sigma widening [V] * cycles^0.6
K_CYC_MU = 3.0e-3         # erase-state upshift [V] * cycles^0.6
K_RET = 3.5e-4            # retention downshift [V] * cycles^0.6 * log-time
                          # (applied ∝ state level: full charge loses most)
EA_EV = 1.1               # activation energy [eV]
KB = 8.617e-5
PAGE_BITS = 8192 * 8 + 72 * 14        # ~1KiB payload + parity footprint
BCH_T = 72                # correctable bits per 1KiB codeword
LDPC_RBER_LIMIT = 8.0e-3  # modern LDPC operating limit


def ispp_distribution(n=200_000, seed=0):
    """Monte-Carlo ISPP: each cell walks up in DV_PGM steps from a random
    start until it crosses verify → uniform-over-step + gaussian noise."""
    rng = np.random.default_rng(seed)
    out = []
    for vt in V_TARGETS:
        start = vt - rng.uniform(0.0, DV_PGM, n)     # overshoot position
        vth = start + DV_PGM                         # last pulse crosses
        vth += SIGMA0 * rng.standard_normal(n)
        out.append(vth)
    er = V_ERASE + 2.2 * SIGMA0 * rng.standard_normal(n)
    return [er] + out


def state_params(cycles=0.0, t_hours=0.0, temp_C=55.0):
    """Analytic (mu, sigma) per state after cycling + retention."""
    nt = cycles ** 0.6
    acc = np.exp(-EA_EV / KB * (1 / (temp_C + 273.15) - 1 / 328.15))
    ret = K_RET * nt * np.log1p(t_hours * acc)
    mus, sigmas = [], []
    for i in range(N_LEVELS):
        if i == 0:
            mu = V_ERASE + K_CYC_MU * nt             # erase creeps UP
            sg = np.hypot(2.2 * SIGMA0, K_CYC_SIG * nt)
        else:
            # charge loss scales with stored charge -> higher states slide
            # further: the top of the window compresses
            mu = V_TARGETS[i - 1] + DV_PGM / 2.0 \
                - ret * i / (N_LEVELS - 1)
            sg = np.hypot(np.sqrt(SIGMA0 ** 2 + DV_PGM ** 2 / 12.0),
                          K_CYC_SIG * nt)
        mus.append(float(mu))
        sigmas.append(float(sg))
    return np.array(mus), np.array(sigmas)


def rber(cycles=0.0, t_hours=0.0, temp_C=55.0):
    """Raw BER from pairwise overlap of adjacent states at the optimal
    read point (equal-error crossing), gray-coded (1 bit per boundary)."""
    mu, sg = state_params(cycles, t_hours, temp_C)
    errs = []
    for i in range(N_LEVELS - 1):
        # optimal threshold between states i, i+1
        num = mu[i] * sg[i + 1] + mu[i + 1] * sg[i]
        vt = num / (sg[i] + sg[i + 1])
        p = 0.5 * erfc((vt - mu[i]) / (sg[i] * np.sqrt(2))) \
            + 0.5 * erfc((mu[i + 1] - vt) / (sg[i + 1] * np.sqrt(2)))
        errs.append(p / 2.0)
    # each cell stores 3 bits; each boundary error flips ~1 bit
    return float(np.sum(errs) / 3.0 / (N_LEVELS / 4.0))


def uber_bch(rber_val, t=BCH_T, n_bits=8192 + 72 * 14):
    """P(codeword uncorrectable) = P(#errors > t), binomial tail."""
    return float(binom.sf(t, n_bits, min(rber_val, 0.4)))


def endurance_rating(t_hours=8760.0, temp_C=55.0,
                     limit=LDPC_RBER_LIMIT):
    """Max P/E cycles with 1-year retention RBER still under the ECC
    limit (bisection)."""
    lo, hi = 10.0, 300_000.0
    if rber(hi, t_hours, temp_C) < limit:
        return hi
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if rber(mid, t_hours, temp_C) < limit:
            lo = mid
        else:
            hi = mid
    return lo


def run():
    return {
        "levels": N_LEVELS, "ispp_step_V": DV_PGM,
        "rber_fresh": f"{rber(0, 0):.2e}",
        "rber_3k_1yr_55C": f"{rber(3000, 8760):.2e}",
        "rber_3k_1yr_85C": f"{rber(3000, 8760, 85):.2e}",
        "uber_bch_at_1e-3": f"{uber_bch(1e-3):.2e}",
        "uber_bch_at_6e-3": f"{uber_bch(6e-3):.2e}",
        "endurance_ldpc_1yr_55C": int(endurance_rating()),
        "endurance_ldpc_1yr_85C": int(endurance_rating(temp_C=85)),
    }


def _plot(path_dist, path_life, seed=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # distributions: fresh vs cycled+baked
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.6), sharex=True)
    v = np.linspace(-4, 5.5, 900)
    for ax, (cyc, th, ttl) in zip(axes, [
            (0, 0, "fresh (ISPP just finished)"),
            (3000, 8760, "after 3k P/E + 1 year @ 55 C")]):
        mu, sg = state_params(cyc, th)
        for i in range(N_LEVELS):
            pdf = np.exp(-0.5 * ((v - mu[i]) / sg[i]) ** 2) / sg[i]
            ax.semilogy(v, pdf, lw=1.4,
                        color=plt.cm.viridis(i / (N_LEVELS - 1)))
        for i in range(N_LEVELS - 1):
            num = mu[i] * sg[i + 1] + mu[i + 1] * sg[i]
            ax.axvline(num / (sg[i] + sg[i + 1]), color="0.6", ls=":",
                       lw=0.7)
        ax.set_ylim(1e-6, 20)
        ax.set_ylabel("pdf (log)")
        ax.set_title(f"TLC Vth window - {ttl}  "
                     f"(RBER {rber(cyc, th):.1e})", fontsize=10)
    axes[1].set_xlabel("cell Vth [V]")
    fig.tight_layout()
    fig.savefig(path_dist, dpi=110)
    plt.close(fig)

    # lifetime map
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax = axes[0]
    cycles = np.logspace(1, 4.7, 60)
    for th, lab, c in ((0.0, "read right after program", "#26a"),
                       (720.0, "1 month retention @ 55C", "#b4740a"),
                       (8760.0, "1 year retention @ 55C", "#c44")):
        ax.loglog(cycles, [rber(c_, th) for c_ in cycles], color=c, lw=1.8,
                  label=lab)
    ax.axhline(LDPC_RBER_LIMIT, color="k", ls="--", lw=1,
               label="LDPC correctable limit")
    e = endurance_rating()
    ax.axvline(e, color="0.5", ls=":",
               label=f"endurance rating = {e:.0f} P/E")
    ax.set_xlabel("P/E cycles")
    ax.set_ylabel("raw BER")
    ax.set_title("Endurance is where retention meets ECC")
    ax.legend(fontsize=8)

    ax = axes[1]
    rb = np.logspace(-4, -1.5, 80)
    ax.loglog(rb, [uber_bch(x) for x in rb], color="#26a", lw=1.8,
              label=f"BCH t={BCH_T}/1KiB (binomial tail)")
    ax.axvline(LDPC_RBER_LIMIT, color="#c44", ls="--", lw=1.2,
               label="LDPC-class limit")
    ax.axhline(1e-15, color="k", ls=":", lw=1, label="UBER spec 1e-15")
    ax.set_xlabel("raw BER")
    ax.set_ylabel("uncorrectable BER")
    ax.set_title("ECC turns 1e-3 raw errors into 1e-15:\n"
                 "the product IS the code")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_life, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "nand_vth.png"),
          os.path.join(ASSETS, "nand_life.png"))
    print("figures ->", ASSETS)


if __name__ == "__main__":
    main()
