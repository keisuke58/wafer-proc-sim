"""
quantum.demo — coherence-limited dicing of a quantum-processor chip.

Runs the three singulation methods against a qubit T1 budget and prints the
ranking + the keep-out distance each one needs. Optionally sweeps keep-out and
plots T1(L) per method.

    python -m quantum.demo                         # table, Si substrate, 5 GHz
    python -m quantum.demo --substrate Sapphire
    python -m quantum.demo --plot quantum_coherence.png
"""
from __future__ import annotations

import argparse

from quantum import (
    QubitSpec, QUBIT_SUBSTRATES, CANONICAL_SIGNATURES,
    evaluate_coherence, rank_methods,
)


def _print_table(qubit: QubitSpec, substrate, ranked) -> None:
    print(f"\nCoherence-limited dicing — substrate={substrate.name}, "
          f"f={qubit.f_GHz} GHz, T1 budget={qubit.T1_target_us:.0f} µs, "
          f"keep-out={qubit.keepout_um:.0f} µm\n")
    print(f"  {'method':<9} {'SSD':>5} {'wet':>4} {'T1[µs]':>8} "
          f"{'min keep-out':>13} {'verdict':>8}")
    print("  " + "-" * 56)
    for sig, rep in ranked:
        ko = "∞" if rep.min_keepout_um == float("inf") else f"{rep.min_keepout_um:.0f} µm"
        verdict = "PASS" if rep.feasible else "FAIL"
        print(f"  {sig.method:<9} {sig.ssd_um:>4.1f} {('yes' if sig.wet else 'no'):>4} "
              f"{rep.T1_us:>8.0f} {ko:>13} {verdict:>8}")
        for v in rep.violations:
            print(f"      ! {v}")
    best = ranked[0]
    worst = ranked[-1]
    print(f"\n  → best: {best[0].method} (T1 {best[1].T1_us:.0f} µs), "
          f"worst: {worst[0].method} (T1 {worst[1].T1_us:.0f} µs). "
          f"Coherence ranking is the inverse of the cost ranking.")


def _plot(qubit: QubitSpec, substrate, path: str) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ls = np.linspace(0.0, 600.0, 200)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for sig in CANONICAL_SIGNATURES.values():
        t1 = [evaluate_coherence(sig, QubitSpec(
            f_GHz=qubit.f_GHz, T1_target_us=qubit.T1_target_us, keepout_um=L),
            substrate).T1_us for L in Ls]
        ax.plot(Ls, t1, lw=2.2, label=f"{sig.method} (SSD {sig.ssd_um:.0f} µm"
                + (", wet)" if sig.wet else ", dry)"))
    ax.axhline(qubit.T1_target_us, ls="--", c="0.4", lw=1.3,
               label=f"T1 budget ({qubit.T1_target_us:.0f} µs)")
    ax.set_xlabel("keep-out distance qubit ↔ dice edge  L [µm]")
    ax.set_ylabel("edge-qubit T1 [µs]")
    ax.set_title(f"Coherence-limited dicing — {substrate.name} substrate, "
                 f"{qubit.f_GHz} GHz transmon")
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"  saved plot → {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", default="Si", choices=sorted(QUBIT_SUBSTRATES))
    ap.add_argument("--freq-GHz", type=float, default=5.0)
    ap.add_argument("--t1-target-us", type=float, default=100.0)
    ap.add_argument("--keepout-um", type=float, default=200.0)
    ap.add_argument("--plot", metavar="PATH", default=None)
    args = ap.parse_args()

    substrate = QUBIT_SUBSTRATES[args.substrate]
    qubit = QubitSpec(f_GHz=args.freq_GHz, T1_target_us=args.t1_target_us,
                      keepout_um=args.keepout_um)
    ranked = rank_methods(qubit, substrate)
    _print_table(qubit, substrate, ranked)
    if args.plot:
        _plot(qubit, substrate, args.plot)


if __name__ == "__main__":
    main()
