"""
Coverage / reliability test for the inverse phase-field forensics posterior.

A single recovered posterior that happens to cover the truth proves nothing about
whether the *stated* uncertainty is honest. This script runs the inversion on many
independent ground truths drawn from the prior and asks the calibration question:

    of all the q-credible regions we report, do ~q of them actually contain truth?

If the emulator-noise-calibrated posterior is well calibrated, the empirical
coverage curve sits on the diagonal. Over-confident -> below; under-confident ->
above. This is the "does the UQ mean what it says" check, on real model runs.

Run (slow, ~15-20 min on CPU):
  cd /home/nishioka/git/wafer-proc-sim
  python -m research.phasefield.coverage_pf_forensics
"""
from __future__ import annotations
import json
import time
import numpy as np

from research.phasefield import inverse_pf_forensics as F

# --- config (lighter than the single-shot demo so many truths are feasible) ---
N_TRUTH   = 20
GRID_N    = 12
K_SAMPLES = 4
N_CALIB   = 6
Q_LEVELS  = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
OUT_PNG   = "research/phasefield/results/inverse_pf_coverage.png"
OUT_JSON  = "research/phasefield/results/inverse_pf_coverage.json"
SEED      = 7


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    model = F.load_generator()
    gc_grid, ell_grid = F.make_grids(GRID_N)

    # draw truths from the (log-uniform) prior, kept away from the grid edges
    # so a covering region can exist on both sides
    lo_gc, hi_gc   = np.log(F.GC_RANGE[0]),  np.log(F.GC_RANGE[1])
    lo_el, hi_el   = np.log(F.ELL_RANGE[0]), np.log(F.ELL_RANGE[1])
    pad = 0.12
    truths = []
    for _ in range(N_TRUTH):
        gc  = float(np.exp(rng.uniform(lo_gc + pad * (hi_gc - lo_gc), hi_gc - pad * (hi_gc - lo_gc))))
        ell = float(np.exp(rng.uniform(lo_el + pad * (hi_el - lo_el), hi_el - pad * (hi_el - lo_el))))
        truths.append((gc, ell))

    contains = np.zeros((N_TRUTH, len(Q_LEVELS)), dtype=bool)
    recov = []   # (gc_true, ell_true, gc_mean, ell_mean, gc_std, ell_std)
    for t, (gc_true, ell_true) in enumerate(truths):
        seed0 = SEED + 1000 * (t + 1)
        f_obs, f_sigma, _ = F.calibrate_obs(model, gc_true, ell_true,
                                            n_calib=N_CALIB, k=K_SAMPLES, seed0=seed0 + 500)
        lp   = F.grid_logpost(model, f_obs, f_sigma, gc_grid, ell_grid,
                              k=K_SAMPLES, seed0=seed0)
        post = F.normalise_post(lp)
        s    = F.posterior_summary(post, gc_grid, ell_grid)
        recov.append((gc_true, ell_true, s['gc_mean'], s['ell_mean'], s['gc_std'], s['ell_std']))
        for qi, q in enumerate(Q_LEVELS):
            contains[t, qi] = F.hpd_contains(post, gc_grid, ell_grid, gc_true, ell_true, q)
        print(f"  truth {t+1:2d}/{N_TRUTH}  Gc*={gc_true:.2e} ell*={ell_true:.3f}  "
              f"-> mean Gc={s['gc_mean']:.2e} ell={s['ell_mean']:.3f}  [{time.time()-t0:.0f}s]")

    emp_cov = contains.mean(axis=0)
    recov = np.array(recov)

    print("\n=== reliability (nominal -> empirical coverage) ===")
    for q, e in zip(Q_LEVELS, emp_cov):
        print(f"  q={q:.1f}  empirical={e:.2f}  ({int(round(e*N_TRUTH))}/{N_TRUTH})")
    # mean absolute calibration error
    mace = float(np.mean(np.abs(emp_cov - Q_LEVELS)))
    print(f"  mean abs calibration error = {mace:.3f}  (0 = perfect)")

    out = {
        "n_truth": N_TRUTH, "grid_n": GRID_N, "k_samples": K_SAMPLES,
        "q_levels": Q_LEVELS.tolist(), "empirical_coverage": emp_cov.tolist(),
        "mace": mace, "recovered": recov.tolist(),
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[save] {OUT_JSON}")
    _plot(Q_LEVELS, emp_cov, recov, mace)
    print(f"[save] {OUT_PNG}")
    print(f"[done] total {time.time()-t0:.0f}s")


def _plot(qs, emp, recov, mace):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))

    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax[0].plot(qs, emp, "-o", color="C0", label="empirical")
    ax[0].set_xlabel("nominal credible level q")
    ax[0].set_ylabel("empirical coverage")
    ax[0].set_title(f"reliability curve\n(MACE={mace:.3f}, N={recov.shape[0]})")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)

    # recovered vs true, Gc
    gct, gcm, gcs = recov[:, 0], recov[:, 2], recov[:, 4]
    ax[1].errorbar(gct, gcm, yerr=gcs, fmt="o", ms=4, capsize=2, color="C1")
    lim = [recov[:, 0].min() * 0.8, recov[:, 0].max() * 1.2]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("Gc true"); ax[1].set_ylabel("Gc recovered (mean +/- sd)")
    ax[1].set_title("Gc recovery")

    # recovered vs true, ell
    elt, elm, els = recov[:, 1], recov[:, 3], recov[:, 5]
    ax[2].errorbar(elt, elm, yerr=els, fmt="o", ms=4, capsize=2, color="C2")
    lim = [recov[:, 1].min() * 0.8, recov[:, 1].max() * 1.2]
    ax[2].plot(lim, lim, "k--", lw=1)
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("ell true"); ax[2].set_ylabel("ell recovered (mean +/- sd)")
    ax[2].set_title("ell recovery")

    fig.suptitle("Inverse phase-field forensics: posterior calibration over many truths", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
