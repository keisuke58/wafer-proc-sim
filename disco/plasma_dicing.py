#!/usr/bin/env python3
"""plasma_dicing.py — plasma (Bosch) dicing vs the blade: scallops, chipping,
and why "no blade" wins on thin/fragile wafers.

DISCO's future is the bladeless cut. Plasma dicing singulates dies by a Bosch
deep-reactive-ion etch of the street instead of a spinning blade — no
mechanical contact, so no front-side chipping, and a kerf a few microns wide
instead of tens. This module quantifies the trade the process integrator
actually weighs:

  BOSCH SIDEWALL  the alternating etch/passivation cycles leave SCALLOPS on
      the street wall: amplitude a_c per cycle, cycle depth d_c, N = T/d_c
      cycles through a wafer of thickness T. Scallop roughness and sidewall
      angle are the die-edge-strength knobs; more passivation -> shallower
      scallops, straighter wall (throughput cost).

  CHIPPING       a blade removes material by fracture, so front/back-side
      chipping scales with feed rate and grit; plasma removes it chemically,
      so die-edge chipping is ~0. The gap is the whole reason plasma exists
      for thin (<100 um) and brittle (SiC/GaN) wafers that a blade shatters.

  KERF ECONOMICS narrow plasma streets fit MORE DIES per wafer than a wide
      blade kerf + its chipping keep-out. Dies/wafer and the per-die cost
      shift is quantified for a representative die size.

Pure numpy + matplotlib.  Run:  python3 disco/plasma_dicing.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "plasma_dicing_results.json")

# ── Bosch process constants ─────────────────────────────────────────────────
ETCH_PER_CYCLE_UM = 1.2      # vertical etch per Bosch cycle [um]
CYCLE_TIME_S = 6.0           # one etch+passivation cycle [s]
# scallop lateral amplitude vs passivation fraction (more passiv -> smaller)
SCALLOP_A0_UM = 0.42         # bare scallop amplitude [um]


def bosch_sidewall(thickness_um=100.0, passivation=0.5, street_w_um=8.0,
                   n_per_cycle=6):
    """Street wall profile x(z) with Bosch scallops. Returns (z, x_wall,
    scallop_amp_um, sidewall_deg, n_cycles)."""
    d_c = ETCH_PER_CYCLE_UM
    n_cyc = int(np.ceil(thickness_um / d_c))
    a_c = SCALLOP_A0_UM * (1.0 - 0.85 * passivation)      # scallop amplitude
    # slight per-cycle net lateral drift -> sidewall taper
    drift = 0.03 * (1.0 - passivation)                    # um per cycle
    z = np.linspace(0, thickness_um, n_cyc * n_per_cycle)
    x0 = street_w_um / 2.0
    cyc = z / d_c
    # scallops: a half-sine bulge within each cycle, plus cumulative drift
    scallop = a_c * np.abs(np.sin(np.pi * cyc))
    x_wall = x0 + scallop + drift * cyc
    # sidewall angle from net top->bottom widening (90 = vertical)
    dwidth = x_wall[-1] - x_wall[0]
    angle = float(np.degrees(np.arctan2(thickness_um, dwidth)))
    angle = float(np.clip(angle, 60.0, 90.0))
    return z, x_wall, float(a_c), angle, n_cyc


def chipping_blade(feed_mm_s=50.0, grit_um=4.0, thickness_um=100.0):
    """Blade front-side chip size [um] — grows with feed, grit and thinness
    (thin wafers flex and chip more). Empirical monotone form."""
    thin = np.clip(100.0 / thickness_um, 1.0, 4.0)
    return float(2.0 + 0.09 * feed_mm_s + 1.4 * grit_um) * thin


def chipping_plasma():
    """Plasma is contact-free: chip size ~ scallop-scale, effectively no
    fracture chipping."""
    return 0.3      # um (sub-scallop; no mechanical fracture)


def dies_per_wafer(die_mm=5.0, kerf_um=40.0, wafer_mm=300.0):
    """Count square dies across a round wafer, given kerf street width."""
    pitch = die_mm + kerf_um * 1e-3
    r = wafer_mm / 2.0 - 3.0                         # edge exclusion
    n = 0
    k = int(wafer_mm / pitch) + 2
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            cx, cy = (i + 0.5) * pitch, (j + 0.5) * pitch
            if abs(cx) + die_mm / 2 < r and abs(cy) + die_mm / 2 < r \
                    and np.hypot(abs(cx) + die_mm / 2, abs(cy) + die_mm / 2) < r:
                n += 1
    return n


def run():
    T = 80.0        # thin wafer [um] — where plasma shines
    # sidewall scallops for a passivation sweep
    walls = {}
    for p in (0.2, 0.5, 0.8):
        _, _, a, ang, nc = bosch_sidewall(T, p)
        walls[f"passivation_{p}"] = {"scallop_amp_um": round(a, 3),
                                     "sidewall_deg": round(ang, 1),
                                     "n_cycles": nc}
    thr_min = bosch_sidewall(T, 0.5)[4] * CYCLE_TIME_S / 60.0

    # chipping: blade vs plasma
    ch_blade = chipping_blade(thickness_um=T)
    ch_plasma = chipping_plasma()

    # kerf economics: blade kerf (blade width + chip keep-out) vs plasma kerf
    kerf_blade = 30.0 + 2 * ch_blade            # keep-out for chipping [um]
    kerf_plasma = 8.0                            # narrow street [um]
    die = 1.0                                    # small die (LED/MEMS): where
    #                                              plasma dicing pays off most
    n_blade = dies_per_wafer(die, kerf_blade)
    n_plasma = dies_per_wafer(die, kerf_plasma)

    return {
        "wafer_thickness_um": T,
        "bosch_sidewall": {
            "etch_per_cycle_um": ETCH_PER_CYCLE_UM,
            "cycle_time_s": CYCLE_TIME_S,
            "throughput_min_per_street_pass": round(thr_min, 2),
            "passivation_sweep": walls},
        "chipping_um": {"blade": round(ch_blade, 2),
                        "plasma": round(ch_plasma, 2),
                        "reduction_x": round(ch_blade / ch_plasma, 1)},
        "kerf_economics": {
            "die_mm": die,
            "kerf_blade_um": round(kerf_blade, 1),
            "kerf_plasma_um": kerf_plasma,
            "dies_blade": n_blade, "dies_plasma": n_plasma,
            "extra_dies": n_plasma - n_blade,
            "gain_pct": round(100 * (n_plasma - n_blade) / n_blade, 2)},
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))

    # 1: Bosch scalloped sidewalls
    ax = axes[0]
    T = res["wafer_thickness_um"]
    for p, c, lab in ((0.2, "#c0392b", "low passiv."),
                      (0.5, "#b4740a", "mid"),
                      (0.8, "#1a7a4a", "high passiv.")):
        z, x, a, ang, nc = bosch_sidewall(T, p)
        ax.plot(x, z, color=c, lw=1.6, label=f"{lab} (scallop {a:.2f}um)")
        ax.plot(-x, z, color=c, lw=1.6)
    ax.invert_yaxis()
    ax.set_xlabel("street half-width [um]")
    ax.set_ylabel("depth [um]")
    ax.set_title("Bosch plasma dicing: scalloped\nstreet wall (thin wafer)",
                 fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # 2: chipping blade vs plasma
    ax = axes[1]
    ch = res["chipping_um"]
    ax.bar(["blade", "plasma"], [ch["blade"], ch["plasma"]],
           color=["#c0392b", "#1a7a4a"])
    ax.set_ylabel("die-edge chipping [um]")
    ax.set_title(f"Contact-free = no chipping\n{ch['reduction_x']:.0f}x less "
                 "than the blade", fontsize=10)
    for i, v in enumerate([ch["blade"], ch["plasma"]]):
        ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 3: dies per wafer
    ax = axes[2]
    ke = res["kerf_economics"]
    ax.bar(["blade\nkerf %.0fum" % ke["kerf_blade_um"],
            "plasma\nkerf %.0fum" % ke["kerf_plasma_um"]],
           [ke["dies_blade"], ke["dies_plasma"]],
           color=["#8a8f98", "#1f6f8b"])
    ax.set_ylabel(f"dies per 300 mm wafer ({ke['die_mm']:.0f} mm die)")
    ax.set_title(f"Narrow street = more dies\n+{ke['extra_dies']} dies "
                 f"(+{ke['gain_pct']:.1f}%)", fontsize=10)
    for i, v in enumerate([ke["dies_blade"], ke["dies_plasma"]]):
        ax.text(i, v + 20, f"{v}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "disco_plasma_dicing.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
