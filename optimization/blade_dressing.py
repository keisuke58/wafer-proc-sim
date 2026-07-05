#!/usr/bin/env python3
"""blade_dressing.py — dicing-blade dressing policy: when to re-sharpen.

`optimization/blade_wear.py` answers WHEN TO REPLACE a blade (chipping trend
→ USL). This module answers the question that comes long before replacement:
WHEN TO DRESS. A resin-bond dicing blade dulls in two coupled ways —

  GLAZING   grit protrusion wears down (attritious wear), cutting force rises
  LOADING   swarf embeds in the bond, force rises further

Rising force is self-limiting: past a critical force the bond fractures and
sheds grit (self-sharpening), which restores protrusion — but at the cost of
extra blade consumption AND a chipping spike while the force is high. A dress
pass (cutting a dressing board) restores protrusion and clears loading in a
controlled way, consuming a known amount of blade exposure and machine time.

So dressing policy is a genuine optimization:
  dress too often   → blade exposure + machine time wasted
  dress too seldom  → force runaway → chipping violations (yield loss)

Three policies are simulated over the same deterministic product mix
(alternating soft/hard segments, e.g. bare Si vs test-pattern streets):

  NONE   rely on self-sharpening only
  FIXED  dress every N streets (swept over N)
  CBM    condition-based: dress when the EWMA of the RECIPE-NORMALIZED
         spindle-current proxy (measured force ÷ fresh-blade force for the
         material being cut, 3 % noise) crosses θ (swept over θ). The
         normalization is the key trick — raw current confounds material
         hardness with blade dullness; dividing by the recipe baseline
         isolates dullness, so one threshold works across the product mix.

Cost model per street: blade amortization (consumed exposure fraction ×
blade price) + machine time (cut + dress) + yield loss per chipping
violation. CBM adapts dressing to the workpiece actually being cut, so it
beats the best fixed interval on a mixed-product line.

Pure numpy + matplotlib.  Run:  python3 optimization/blade_dressing.py
"""
import json
import os
from dataclasses import dataclass

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "blade_dressing_results.json")

# ── blade / process constants ───────────────────────────────────────────────
S_FRESH = 0.92        # grit protrusion fraction right after dressing
L_FRESH = 0.03        # residual loading right after dressing
F0 = 2.0              # cutting force of a fresh blade on soft material [N]
F_CRIT = 2.6          # force (×F0) at which the bond self-sharpens
P_SHARP = 1.5         # force exponent on dullness: F ∝ (S_FRESH/sharp)^P
BETA_LOAD = 1.2       # force multiplier per unit loading
K_ATTR = 2.5e-4       # protrusion attrition per street per unit (F/F0)
K_LOAD = 6.0e-4       # loading build-up per street (hard material scaled)
SS_RECOVER = 0.5      # self-sharpening: fraction of lost protrusion restored
SS_LOAD_KEEP = 0.4    # loading remaining after a self-sharpening event
EXPOSURE_UM = 550.0   # usable blade exposure budget until replacement [µm]
WEAR_UM_CUT = 0.010   # radial wear per street at F=F0 [µm]
WEAR_UM_SS = 2.0      # extra exposure shed by one self-sharpening event [µm]
DRESS_UM = 4.0        # exposure consumed by one dress pass [µm]

# chipping response (backside chipping, BSC)
CHIP_BASE = 5.0       # BSC of a healthy cut [µm]
CHIP_SLOPE = 9.0      # BSC added per unit (F/F0 - 0.8) [µm]
CHIP_SIGMA = 0.12     # lognormal scatter on BSC
USL_CHIP = 25.0       # BSC spec limit [µm]

# economics (¥)
BLADE_COST = 40_000.0     # hub blade price
MACHINE_RATE = 6_000.0    # machine + operator [¥/h]
T_CUT = 6.0               # seconds per street
T_DRESS = 40.0            # seconds per dress pass
YIELD_LOSS = 1_500.0      # scrap cost of one chipping-violation street

# condition monitoring
EWMA_LAMBDA = 0.2     # spindle-current EWMA smoothing
SENSE_NOISE = 0.03    # spindle-current proxy noise (fraction)
MIN_GAP = 20          # streets: minimum spacing between dress passes


def make_hardness(n_streets: int, seed: int = 3,
                  p_hard: float = 0.5) -> np.ndarray:
    """Deterministic product mix: segments of soft (1.0) and hard (1.6)
    material with irregular lengths — the reason a fixed interval can't be
    right everywhere. p_hard shifts the mix (share of hard segments)."""
    rng = np.random.default_rng(seed)
    hard = np.empty(n_streets)
    i = 0
    while i < n_streets:
        seg = int(rng.integers(300, 700))
        hard[i:i + seg] = 1.6 if rng.random() < p_hard else 1.0
        i += seg
    return hard


@dataclass
class SimResult:
    force: np.ndarray          # per-street cutting force [N]
    chip: np.ndarray           # per-street BSC [µm]
    sharp: np.ndarray          # protrusion state after each street
    load: np.ndarray           # loading state after each street
    dress_at: np.ndarray       # street indices where a dress pass ran
    ss_at: np.ndarray          # street indices of self-sharpening events
    exposure_used_um: float    # total blade exposure consumed
    violations: int            # streets with BSC > USL_CHIP

    def cost_per_1000(self) -> dict:
        n = len(self.force)
        blade = self.exposure_used_um / EXPOSURE_UM * BLADE_COST
        time_s = n * T_CUT + len(self.dress_at) * T_DRESS
        machine = time_s / 3600.0 * MACHINE_RATE
        yield_c = self.violations * YIELD_LOSS
        scale = 1000.0 / n
        return {
            "blade": blade * scale,
            "machine": machine * scale,
            "yield": yield_c * scale,
            "total": (blade + machine + yield_c) * scale,
            "violations_per_1000": self.violations * scale,
            "dresses_per_1000": len(self.dress_at) * scale,
        }


def cutting_force(sharp: float, load: float, hard: float) -> float:
    return F0 * hard * (S_FRESH / sharp) ** P_SHARP * (1.0 + BETA_LOAD * load)


def simulate(policy, hardness: np.ndarray, seed: int = 42) -> SimResult:
    """Run one blade over the street sequence under a dressing policy.

    policy: ("none",) | ("fixed", N) | ("cbm", theta)
    """
    rng = np.random.default_rng(seed)
    n = len(hardness)
    sharp, load = S_FRESH, L_FRESH
    exposure = 0.0
    ewma = 1.0            # recipe-normalized load: 1.0 = fresh blade
    since_dress = 0
    force = np.empty(n)
    chip = np.empty(n)
    sharp_tr = np.empty(n)
    load_tr = np.empty(n)
    dress_at, ss_at = [], []
    violations = 0

    for i in range(n):
        h = hardness[i]
        f = cutting_force(sharp, load, h)
        force[i] = f
        c = (CHIP_BASE + CHIP_SLOPE * max(f / F0 - 0.8, 0.0)) \
            * rng.lognormal(0.0, CHIP_SIGMA)
        chip[i] = c
        if c > USL_CHIP:
            violations += 1

        # wear this street
        sharp = max(sharp - K_ATTR * (f / F0) * h, 0.30)
        load = load + K_LOAD * h * (1.0 - load)
        exposure += WEAR_UM_CUT * (f / F0)
        since_dress += 1

        # bond self-sharpens past the critical force (uncontrolled dressing)
        if f > F_CRIT * F0:
            sharp += SS_RECOVER * (S_FRESH - sharp)
            load *= SS_LOAD_KEEP
            exposure += WEAR_UM_SS
            ss_at.append(i)

        # condition monitor: spindle-current proxy, normalized by the
        # fresh-blade force of the recipe being cut (the machine knows the
        # recipe, so this baseline is available in production).
        meas = f * (1.0 + SENSE_NOISE * rng.standard_normal()) / (F0 * h)
        ewma = EWMA_LAMBDA * meas + (1.0 - EWMA_LAMBDA) * ewma

        # policy decision AFTER the street
        dress = False
        if policy[0] == "fixed":
            dress = since_dress >= policy[1]
        elif policy[0] == "cbm":
            dress = ewma > policy[1] and since_dress >= MIN_GAP
        if dress:
            sharp, load = S_FRESH, L_FRESH
            exposure += DRESS_UM
            ewma = 1.0
            since_dress = 0
            dress_at.append(i)

        sharp_tr[i] = sharp
        load_tr[i] = load

    return SimResult(force, chip, sharp_tr, load_tr,
                     np.asarray(dress_at), np.asarray(ss_at),
                     exposure, violations)


FIXED_GRID = (50, 75, 100, 150, 200, 300, 400, 600, 900)
CBM_GRID = (1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.8)


def sweep(hardness: np.ndarray, seed: int = 42):
    """Sweep both policy families; return (per-policy curves, best summary)."""
    fixed = {N: simulate(("fixed", N), hardness, seed).cost_per_1000()
             for N in FIXED_GRID}
    cbm = {th: simulate(("cbm", th), hardness, seed).cost_per_1000()
           for th in CBM_GRID}
    none = simulate(("none",), hardness, seed).cost_per_1000()
    best_N = min(fixed, key=lambda k: fixed[k]["total"])
    best_th = min(cbm, key=lambda k: cbm[k]["total"])

    # robustness: freeze both tuned policies, shift the product mix (more
    # hard segments, different schedule) — the fixed interval was tuned for
    # a mix that no longer exists, CBM keeps reacting to the blade itself.
    shifted_mix = make_hardness(len(hardness), seed=9, p_hard=0.75)
    shifted = {
        "none": simulate(("none",), shifted_mix, seed).cost_per_1000(),
        "fixed": simulate(("fixed", best_N), shifted_mix, seed)
        .cost_per_1000(),
        "cbm": simulate(("cbm", best_th), shifted_mix, seed).cost_per_1000(),
    }
    return {
        "none": none,
        "fixed": fixed, "best_fixed_N": best_N,
        "cbm": cbm, "best_cbm_theta": best_th,
        "shifted": shifted,
    }


# ── figures + JSON ──────────────────────────────────────────────────────────
def _plot_state(hardness, best_th, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = 3000
    h = hardness[:n_show]
    r_none = simulate(("none",), h)
    r_cbm = simulate(("cbm", best_th), h)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.4), sharex=True)
    x = np.arange(n_show)

    ax = axes[0]
    ax.plot(x, r_none.sharp, color="#c44", lw=1.0, label="no dressing")
    ax.plot(x, r_cbm.sharp, color="#26a", lw=1.0, label="condition-based")
    ax.set_ylabel("grit protrusion s")
    ax.legend(loc="lower left", fontsize=9)
    hard_mask = h > 1.0
    ax.fill_between(x, 0, 1, where=hard_mask, transform=ax.get_xaxis_transform(),
                    color="0.85", zorder=0, label=None)
    ax.set_title("Blade state over 3000 streets — gray bands = hard-material "
                 "segments; dressing keeps the blade out of force runaway")

    ax = axes[1]
    ax.plot(x, r_none.force, color="#c44", lw=0.8)
    ax.plot(x, r_cbm.force, color="#26a", lw=0.8)
    ax.axhline(F_CRIT * F0, color="k", ls="--", lw=1,
               label="self-sharpen threshold")
    for d in r_cbm.dress_at:
        ax.axvline(d, color="#26a", alpha=0.20, lw=1)
    ax.set_ylabel("cutting force [N]")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[2]
    ax.plot(x, r_none.chip, color="#c44", lw=0.6, alpha=0.8)
    ax.plot(x, r_cbm.chip, color="#26a", lw=0.6, alpha=0.8)
    ax.axhline(USL_CHIP, color="k", ls="--", lw=1, label=f"USL {USL_CHIP:.0f} um")
    ax.set_ylabel("BSC chipping [um]")
    ax.set_xlabel("street index")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_policy(sw, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    Ns = list(sw["fixed"].keys())
    tot_f = [sw["fixed"][N]["total"] for N in Ns]
    ax.plot(Ns, tot_f, "o-", color="#888", label="fixed interval")
    bN = sw["best_fixed_N"]
    ax.plot([bN], [sw["fixed"][bN]["total"]], "s", ms=10, color="#555",
            label=f"best fixed (N={bN})")
    bt = sw["best_cbm_theta"]
    ax.axhline(sw["cbm"][bt]["total"], color="#26a", lw=2,
               label=f"condition-based (theta={bt})")
    ax.set_xscale("log")
    ax.set_xlabel("dress interval N [streets]")
    ax.set_ylabel("total cost [yen / 1000 streets]")
    ax.set_title("Dressing policy cost")
    ax.legend(fontsize=9)

    ax = axes[1]
    labels = ["none", f"fixed N={bN}", f"CBM th={bt}"]
    x = np.arange(3)
    rows_a = [sw["none"], sw["fixed"][bN], sw["cbm"][bt]]
    rows_b = [sw["shifted"]["none"], sw["shifted"]["fixed"],
              sw["shifted"]["cbm"]]
    for off, rows, color, lab in ((-0.19, rows_a, "#7a9", "tuning mix"),
                                  (+0.19, rows_b, "#c86", "shifted mix")):
        vals = [r["total"] for r in rows]
        ax.bar(x + off, vals, width=0.36, color=color, label=lab)
        for i, r in enumerate(rows):
            ax.text(i + off, r["total"] * 1.01,
                    f"{r['violations_per_1000']:.1f}v", ha="center",
                    fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("total cost [yen / 1000 streets]")
    ax.set_title("Policies frozen, product mix shifted\n"
                 "(bars annotated with violations per 1000 streets)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    n_streets = 8000
    hardness = make_hardness(n_streets)
    sw = sweep(hardness)

    bN, bt = sw["best_fixed_N"], sw["best_cbm_theta"]
    summary = {
        "n_streets": n_streets,
        "usl_chip_um": USL_CHIP,
        "none": sw["none"],
        "best_fixed": {"N": bN, **sw["fixed"][bN]},
        "best_cbm": {"theta": bt, **sw["cbm"][bt]},
        "cbm_saving_vs_best_fixed_pct": round(
            100.0 * (1.0 - sw["cbm"][bt]["total"]
                     / sw["fixed"][bN]["total"]), 1),
        "cbm_saving_vs_none_pct": round(
            100.0 * (1.0 - sw["cbm"][bt]["total"] / sw["none"]["total"]), 1),
        # one saw at ~6 s/street cuts ~3.5M streets/yr (70% utilization)
        "cbm_saving_yen_per_year_per_machine": round(
            (sw["fixed"][bN]["total"] - sw["cbm"][bt]["total"]) * 3500),
        "shifted_mix": {
            "none": sw["shifted"]["none"],
            "fixed_frozen": sw["shifted"]["fixed"],
            "cbm_frozen": sw["shifted"]["cbm"],
            "cbm_saving_vs_fixed_pct": round(
                100.0 * (1.0 - sw["shifted"]["cbm"]["total"]
                         / sw["shifted"]["fixed"]["total"]), 1),
        },
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    os.makedirs(ASSETS, exist_ok=True)
    _plot_state(hardness, bt, os.path.join(ASSETS, "dress_state.png"))
    _plot_policy(sw, os.path.join(ASSETS, "dress_policy.png"))
    print("figures ->", ASSETS)


if __name__ == "__main__":
    main()
