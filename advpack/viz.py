"""
advpack.viz — figures + CLI for the connected advanced-packaging line.

Headline figure: as the wafer is thinned, Stoney warp grows ~1/t², driving the
bonding overlay error and void risk past spec — i.e. the THINNING recipe decides
whether the bonded stack is manufacturable at all.

Run:
    python -m advpack.viz                       # thinning sweep + figure
    python -m advpack.viz --material SiC
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from advpack import StackConfig, PackagingLine, min_manufacturable_thickness

BOLD = "\033[1m"; RESET = "\033[0m"; GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"

#: the "prescription" — each mitigation and how it changes the base recipe
PRESCRIPTIONS = [
    ("baseline",                  {}),
    ("low-stress film (30→8MPa)", {"film_stress_MPa": 8.0}),
    ("stress-balanced backside",  {"stress_balanced": True}),
    ("carrier wafer 700µm",       {"carrier_thickness_um": 700.0}),
    ("carrier + low-stress",      {"carrier_thickness_um": 700.0, "film_stress_MPa": 8.0}),
]


def prescription_table(material="Si", dice="laser"):
    base = StackConfig(material=material, dice_method=dice)
    rows = []
    for name, over in PRESCRIPTIONS:
        tmin = min_manufacturable_thickness(base.copy(**over))
        rows.append((name, tmin))
    return rows


def plot_prescription(rows, material, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.titleweight": "bold"})
    names = [r[0] for r in rows]
    # un-manufacturable → show as full thickness bar with hatch
    vals = [(r[1] if r[1] is not None else 775.0) for r in rows]
    fail = [r[1] is None for r in rows]
    colors = ["#b5374a" if i == 0 else "#1b7f79" for i in range(len(rows))]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="k", alpha=.9)
    for b, v, f in zip(bars, vals[::-1], fail[::-1]):
        lbl = "no window" if f else f"{v:.0f} µm"
        ax.text(v + 8, b.get_y() + b.get_height() / 2, lbl, va="center", fontsize=10)
    ax.set(xlabel="minimum manufacturable die thickness [µm]  (← thinner is better)",
           title=f"Prescription: how each fix unlocks thinner dies  ({material})")
    ax.invert_xaxis()   # thinner (better) to the right
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def thinning_sweep(material="Si", dice="laser", t_lo=40.0, t_hi=775.0, n=40):
    L = PackagingLine()
    ts = np.linspace(t_hi, t_lo, n)
    rows = []
    for t in ts:
        r = L.run(StackConfig(material=material, target_thickness_um=float(t),
                              dice_method=dice))
        s = r.summary()
        s["yielded"] = r.yielded
        rows.append(s)
    return rows


def plot_sweep(rows, material, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.grid": True, "grid.alpha": .3, "axes.titleweight": "bold"})
    t = [r["target_um"] for r in rows]
    bow = [r["bow_um"] for r in rows]
    overlay = [r["overlay_nm"] for r in rows]
    void = [r["void_risk"] for r in rows]
    yld = [r["yield"] for r in rows]
    budget = StackConfig().overlay_budget_nm

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    ax[0].plot(t, bow, "o-", color="#1b7f79", ms=4)
    ax[0].set(xlabel="die thickness [µm]  (← thinner)", ylabel="warp bow [µm]",
              title="(a) Stoney warp ~ 1/t²")
    ax[0].invert_xaxis(); ax[0].set_yscale("log")

    ax[1].plot(t, overlay, "o-", color="#b5374a", ms=4)
    ax[1].axhline(budget, ls="--", color="gray", label=f"overlay budget {budget:.0f} nm")
    ax[1].set(xlabel="die thickness [µm]  (← thinner)", ylabel="bond overlay error [nm]",
              title="(b) warp → overlay error")
    ax[1].invert_xaxis(); ax[1].set_yscale("log"); ax[1].legend(fontsize=9, frameon=False)

    # yield map: green where the whole line yields
    ax[2].plot(t, void, "o-", color="#c08a1e", ms=4, label="void risk")
    # shade the manufacturable thickness band
    ok_t = [tt for tt, y in zip(t, yld) if y]
    if ok_t:
        ax[2].axvspan(min(ok_t), max(ok_t), color="#9ecae1", alpha=.35,
                      label="stack YIELDS")
    ax[2].axhline(0.5, ls="--", color="gray")
    ax[2].set(xlabel="die thickness [µm]  (← thinner)", ylabel="bond void risk",
              title="(c) manufacturable window")
    ax[2].invert_xaxis(); ax[2].legend(fontsize=9, frameon=False)

    fig.suptitle(f"Advanced packaging line — thin→warp→bond→singulate  ({material})\n"
                 "thinning the die explodes warp → kills bondability "
                 "(overlay & voids out of spec)", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="Advanced-packaging line demo")
    ap.add_argument("--material", default="Si")
    ap.add_argument("--dice", default="laser", choices=["laser", "blade"])
    ap.add_argument("--prescribe", action="store_true",
                    help="show how mitigations lower the min manufacturable thickness")
    args = ap.parse_args()

    if args.prescribe:
        rows = prescription_table(args.material, args.dice)
        print(f"{BOLD}Prescription — min manufacturable thickness ({args.material}){RESET}")
        base = rows[0][1]
        for name, tmin in rows:
            if tmin is None:
                print(f"  {name:28} {RED}no window{RESET}")
            else:
                gain = "" if name == "baseline" or base is None else \
                    f"  ({GREEN}{base - tmin:+.0f}µm vs baseline{RESET})"
                print(f"  {name:28} {GREEN}{tmin:>6.0f} µm{RESET}{gain}")
        out = plot_prescription(rows, args.material,
                                os.path.join(ROOT, "results", "advpack_prescription.png"))
        print(f"\n  {CYAN}wrote {out}{RESET}")
        return

    rows = thinning_sweep(args.material, args.dice)
    print(f"{BOLD}thin→warp→bond→singulate  ({args.material}, {args.dice} dice){RESET}")
    print(f"  {'t_um':>6}{'bow_um':>9}{'overlay_nm':>11}{'void':>7}{'delam':>7}"
          f"{'die_MPa':>8}  yield")
    for r in rows[::5]:
        col = GREEN if r["yield"] else RED
        print(f"  {r['target_um']:>6.0f}{r['bow_um']:>9.0f}{r['overlay_nm']:>11.0f}"
              f"{r['void_risk']:>7.2f}{r['delam_risk']:>7.2f}{r['die_strength_MPa']:>8.0f}"
              f"  {col}{r['yield']}{RESET}")
    ok = [r["target_um"] for r in rows if r["yield"]]
    if ok:
        print(f"\n  {BOLD}manufacturable thickness ≥ ~{min(ok):.0f} µm{RESET}  "
              "(below this, warp kills bondability)")
    else:
        print(f"\n  {RED}no thickness yields at these stress/budget settings{RESET}")
    out = plot_sweep(rows, args.material, os.path.join(ROOT, "results", "advpack_thinning.png"))
    print(f"  {CYAN}wrote {out}{RESET}")


if __name__ == "__main__":
    main()
