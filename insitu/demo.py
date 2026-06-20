"""
insitu.demo — listen to a dicing pass and flag incipient cracks.

    python -m insitu.demo                       # table over blade/plasma/stealth
    python -m insitu.demo --plot insitu_listen.png

Prints, per method, the AE detection rate / false-positive rate / lead time
before the catastrophic chip, and (optionally) plots the AE waveform with
detected events plus the cumulative-AE-energy damage curve.
"""
from __future__ import annotations

import argparse

from insitu import monitor_all, simulate_ae, window_features, MahaDetector
from quantum.coherence import CANONICAL_SIGNATURES


def _print_table(results) -> None:
    print("\nProcess-embedded SHM — self-baselining Mahalanobis AE detector\n")
    print(f"  {'method':<9} {'det.rate':>9} {'FPR':>7} {'lead time':>11} {'catastrophe':>12}")
    print("  " + "-" * 54)
    for name in ("plasma", "stealth", "blade"):
        r = results[name]
        det = "n/a" if r.detection_rate != r.detection_rate else f"{r.detection_rate*100:.0f}%"
        fpr = "n/a" if r.false_pos_rate != r.false_pos_rate else f"{r.false_pos_rate*100:.0f}%"
        lead = "—" if r.lead_time_s is None else f"{r.lead_time_s*1e3:.1f} ms"
        cat = "—" if r.catastrophe_s is None else f"{r.catastrophe_s*1e3:.0f} ms"
        print(f"  {name:<9} {det:>9} {fpr:>7} {lead:>11} {cat:>12}")
    print("\n  → blade radiates a catastrophic chip the detector catches with lead "
          "time;\n    stealth/plasma stay near the healthy baseline (few/no alarms).")


def _plot(path: str, seed: int = 0) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = monitor_all(seed=seed)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 7.0))

    # top: blade AE waveform with detected-event markers
    sig = CANONICAL_SIGNATURES["blade"]
    trace = simulate_ae(sig, seed=seed)
    feats, times = window_features(trace)
    det = MahaDetector().fit(feats)
    alarms = det.alarms(feats)
    ax0.plot(trace.t * 1e3, trace.x, lw=0.4, color="0.5", label="AE waveform (blade)")
    w = times[1] - times[0]
    for ta in times[alarms]:
        ax0.axvspan(ta * 1e3, (ta + w) * 1e3, color="crimson", alpha=0.18)
    if trace.catastrophe_s is not None:
        ax0.axvline(trace.catastrophe_s * 1e3, color="k", ls="--", lw=1.2,
                    label="catastrophic chip")
    ax0.axvspan(0, det.n_baseline * w * 1e3, color="seagreen", alpha=0.12,
                label="healthy baseline")
    ax0.set_xlabel("time [ms]")
    ax0.set_ylabel("AE amplitude")
    ax0.set_title("結晶の聴診器: AE stream with Mahalanobis crack alarms (red)")
    ax0.legend(frameon=False, fontsize=8, loc="upper left")

    # bottom: cumulative AE energy (damage-accumulation curve) per method
    for name, color in (("blade", "crimson"), ("stealth", "seagreen"),
                        ("plasma", "steelblue")):
        r = results[name]
        ax1.plot(r.times * 1e3, r.cum_energy, lw=2.0, color=color, label=name)
    ax1.set_xlabel("time [ms]")
    ax1.set_ylabel("cumulative AE energy  (damage proxy)")
    ax1.set_title("Damage-accumulation curve — blade loads up, dry routes stay flat")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"  saved plot → {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", metavar="PATH", default=None)
    args = ap.parse_args()
    _print_table(monitor_all(seed=args.seed))
    if args.plot:
        _plot(args.plot, seed=args.seed)


if __name__ == "__main__":
    main()
