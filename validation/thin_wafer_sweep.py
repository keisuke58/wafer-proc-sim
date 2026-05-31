"""
④ Thin wafer parametric sweep — 2nm node validation.

Sweeps wafer thickness from 25 µm (HBM4) to 200 µm (standard) across
three laser regimes (ns/ps/fs) and two plasma duty cycles (CW vs pulsed).

Key questions:
    Q1. How does ARDE scale with wafer thickness? (thinner → higher AR → more lag)
    Q2. Does die strength drop below automotive threshold (<100 MPa) for thin dies?
    Q3. How does delamination risk change as wafer thins and laser HAZ fraction rises?
    Q4. Minimum plasma_s to fully dice each thickness?

Usage:
    python validation/thin_wafer_sweep.py
    python validation/thin_wafer_sweep.py --plot
    python validation/thin_wafer_sweep.py --csv results/thin_wafer_sweep.csv

References:
    SK Hynix HBM4 (2026): 16 layers × 25µm/die
    TSMC N2 final die thickness: 50–70 µm (backgrind after bonding)
    Plasma-Therm PDOC (Plasma Dicing on Carrier): thin wafer specialty process
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fem.plasma_bosch_model import hybrid_physics, ETCH_RATE_BASE_UM_MIN

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# Sweep axes
WAFER_THICKNESSES_UM = [25, 50, 75, 100, 150, 200, 300, 400]
PULSE_REGIMES        = ["ns", "ps", "fs"]
DUTY_CYCLES          = [1.0, 0.2]   # CW vs 20% pulsed

# Fixed baseline (representative 2nm-era process)
BASELINE = dict(
    laser_W           = 10.0,
    scan_mm_s         = 300.0,
    pulse_kHz         = 100.0,
    n_passes          = 2,
    n_cycles          = 80,
    mask_um           = 8.0,
    beol_k            = 2.8,     # IBM ALK (2nm)
    beol_thickness_um = 5.0,
)

# Plasma time needed: estimate as 1.5 × (wafer_um / etch_rate_base) in seconds
# Factor 1.5 accounts for ARDE slowdown
def _plasma_s_for_thickness(wafer_um: float) -> float:
    return 1.5 * (wafer_um / ETCH_RATE_BASE_UM_MIN) * 60.0   # s


def run_sweep() -> pd.DataFrame:
    """Run full parametric sweep and return results DataFrame."""
    records = []

    for wafer_um in WAFER_THICKNESSES_UM:
        plasma_s = _plasma_s_for_thickness(wafer_um)

        for regime in PULSE_REGIMES:
            for dc in DUTY_CYCLES:
                res = hybrid_physics(
                    **BASELINE,
                    wafer_um      = wafer_um,
                    plasma_s      = plasma_s,
                    pulse_regime  = regime,
                    duty_cycle    = dc,
                )

                # Aspect ratio of final trench
                eff_mask = max(BASELINE["mask_um"], res.groove_width_um)
                final_AR = res.total_depth_um / eff_mask if eff_mask > 0 else 0

                records.append({
                    "wafer_um":           wafer_um,
                    "pulse_regime":       regime,
                    "duty_cycle":         dc,
                    "plasma_s":           round(plasma_s, 0),
                    "groove_depth_um":    res.groove_depth_um,
                    "groove_width_um":    res.groove_width_um,
                    "HAZ_depth_um":       res.HAZ_depth_um,
                    "plasma_depth_um":    res.plasma_depth_um,
                    "total_depth_um":     res.total_depth_um,
                    "fully_diced":        res.total_depth_um >= wafer_um * 0.98,
                    "street_width_um":    res.street_width_um,
                    "final_AR":           round(final_AR, 2),
                    "die_strength_MPa":   res.die_strength_MPa,
                    "delamination_risk":  res.delamination_risk,
                    "lowk_damage_nm":     res.lowk_damage_depth_nm,
                    "throughput_mm2_s":   res.throughput_mm2_s,
                })

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame):
    print("\n=== Thin Wafer Sweep Summary ===\n")

    # Q1: ARDE scale
    print("── Q1: Final aspect ratio vs wafer thickness (ns, CW) ──")
    q1 = df[(df.pulse_regime == "ns") & (df.duty_cycle == 1.0)][
        ["wafer_um", "final_AR", "plasma_depth_um", "fully_diced"]]
    print(q1.to_string(index=False))

    # Q2: Die strength
    print("\n── Q2: Die strength [MPa] across regimes (wafer=50µm) ──")
    q2 = df[df.wafer_um == 50][
        ["pulse_regime", "duty_cycle", "die_strength_MPa", "HAZ_depth_um"]]
    print(q2.to_string(index=False))

    # Q3: Delamination risk
    print("\n── Q3: Delamination risk (beol_k=2.8) across regimes ──")
    q3 = df[df.duty_cycle == 1.0][
        ["wafer_um", "pulse_regime", "delamination_risk", "lowk_damage_nm"]
    ].pivot_table(index="wafer_um", columns="pulse_regime",
                  values="delamination_risk", aggfunc="mean")
    print(q3.round(4).to_string())

    # Q4: Minimum plasma time
    print("\n── Q4: Minimum plasma_s to fully dice each thickness ──")
    q4 = df[df.pulse_regime == "ps"][
        ["wafer_um", "plasma_s", "fully_diced", "duty_cycle"]].drop_duplicates(
        subset=["wafer_um", "duty_cycle"])
    print(q4.to_string(index=False))

    # 2nm compliance summary
    print("\n── 2nm Compliance Matrix (wafer=50µm, beol_k=2.8) ──")
    spec = df[df.wafer_um == 50].copy()
    spec["sw_ok"]     = spec.street_width_um  < 30.0
    spec["str_ok"]    = spec.die_strength_MPa > 100.0
    spec["delam_ok"]  = spec.delamination_risk < 0.30
    spec["2nm_pass"]  = spec.sw_ok & spec.str_ok & spec.delam_ok
    summary = spec[["pulse_regime", "duty_cycle", "street_width_um",
                     "die_strength_MPa", "delamination_risk", "2nm_pass"]]
    print(summary.to_string(index=False))


def plot_results(df: pd.DataFrame, out_dir: str = RESULTS):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    os.makedirs(out_dir, exist_ok=True)
    regimes  = ["ns", "ps", "fs"]
    colors   = {"ns": "#e74c3c", "ps": "#2980b9", "fs": "#27ae60"}
    lstyles  = {1.0: "-", 0.2: "--"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    metrics   = ["HAZ_depth_um", "die_strength_MPa", "delamination_risk",
                 "final_AR",     "street_width_um",   "throughput_mm2_s"]
    ylabels   = ["HAZ depth [µm]", "Die strength [MPa]", "Delamination risk",
                 "Trench AR",      "Street width [µm]",   "Throughput [mm²/s]"]
    thresholds = {
        "die_strength_MPa": (100.0, "Automotive min"),
        "delamination_risk": (0.30, "2nm spec"),
        "street_width_um":  (30.0,  "2nm spec"),
    }

    for ax, metric, ylabel in zip(axes.ravel(), metrics, ylabels):
        for regime in regimes:
            for dc in DUTY_CYCLES:
                sub = df[(df.pulse_regime == regime) & (df.duty_cycle == dc)]
                label = f"{regime.upper()} dc={dc:.1f}"
                ax.plot(sub.wafer_um, sub[metric],
                        color=colors[regime], ls=lstyles[dc],
                        marker='o', ms=4, label=label)

        if metric in thresholds:
            tval, tlabel = thresholds[metric]
            ax.axhline(tval, color='k', ls=':', lw=1.0, label=tlabel)

        ax.set(xlabel="Wafer thickness [µm]", ylabel=ylabel,
               title=metric.replace("_", " "))
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)

    plt.suptitle("④ Thin Wafer Sweep — 2nm Node Validation\n"
                 f"Baseline: P={BASELINE['laser_W']}W, "
                 f"v={BASELINE['scan_mm_s']}mm/s, "
                 f"beol_k={BASELINE['beol_k']}", fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "thin_wafer_sweep.png")
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")

    # Regime comparison heatmap: delamination risk
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    for ax, dc in zip(axes2, DUTY_CYCLES):
        pivot = df[df.duty_cycle == dc].pivot_table(
            index="wafer_um", columns="pulse_regime",
            values="delamination_risk", aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=0.6, origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set(xlabel="Pulse regime", ylabel="Wafer thickness [µm]",
               title=f"Delamination risk — duty_cycle={dc:.1f}")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i,j]:.2f}",
                        ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax)

    plt.suptitle("③ Low-k Delamination Risk Heatmap (beol_k=2.8)", fontsize=11)
    plt.tight_layout()
    out2 = os.path.join(out_dir, "thin_wafer_delam_heatmap.png")
    plt.savefig(out2, dpi=150)
    print(f"Heatmap saved → {out2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--csv",  default=os.path.join(RESULTS, "thin_wafer_sweep.csv"))
    args = ap.parse_args()

    df = run_sweep()
    print_summary(df)

    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(args.csv, index=False)
    print(f"\nCSV saved → {args.csv}  ({len(df)} rows)")

    if args.plot:
        plot_results(df)
