"""
Inject Kojima real FLIR/TSA full-field stress into the inverse phase-field
forensics pipeline -- WITH an explicit domain-gap audit.

HONESTY NOTE (this is the whole point):
  The FM emulator was trained on AT2 *damage* fields d in [0,1]. The Kojima data
  is full-field *stress* (DSPSS, MPa) measured on a real perforated interstage.
  These are different physical observables. Feeding stress in and reading off a
  "fracture toughness" would be nonsense. So before trusting any posterior we
  audit whether the real field's descriptor vector even lies inside the emulator's
  REACHABLE feature manifold. If it doesn't, the inversion is extrapolation and we
  say so. If it does (for some features), we report an emulator-equivalent
  best-match landscape, explicitly labelled as morphology-matching, not physics.

Real arrays (copaam_derived/exp_to_FEM_CNN/):
  rect_dspss_exp_totsu.npy  (12, 245, 56)  convex defect specimens
  rect_dspss_exp_ou.npy     (10, 185, 58)  concave defect specimens

Run:
  cd /home/nishioka/git/wafer-proc-sim
  python -m research.phasefield.inject_kojima_pf
"""
from __future__ import annotations
import json
import time
import numpy as np

from research.phasefield import inverse_pf_forensics as F

KOJIMA_DIR = "/home/nishioka/from_kojima/copaam_derived/exp_to_FEM_CNN"
GRID_N     = 14
K_SAMPLES  = 6
OUT_PNG    = "research/phasefield/results/inject_kojima_pf.png"
OUT_JSON   = "research/phasefield/results/inject_kojima_pf.json"
NY, NX     = 40, 80   # emulator field grid


def resize_to(field: np.ndarray, ny=NY, nx=NX) -> np.ndarray:
    """Bilinear resample a 2D array to (ny, nx) using numpy only."""
    sy, sx = field.shape
    yi = np.linspace(0, sy - 1, ny)
    xi = np.linspace(0, sx - 1, nx)
    y0 = np.floor(yi).astype(int); y1 = np.clip(y0 + 1, 0, sy - 1)
    x0 = np.floor(xi).astype(int); x1 = np.clip(x0 + 1, 0, sx - 1)
    wy = (yi - y0)[:, None]; wx = (xi - x0)[None, :]
    f00 = field[np.ix_(y0, x0)]; f01 = field[np.ix_(y0, x1)]
    f10 = field[np.ix_(y1, x0)]; f11 = field[np.ix_(y1, x1)]
    top = f00 * (1 - wx) + f01 * wx
    bot = f10 * (1 - wx) + f11 * wx
    return top * (1 - wy) + bot * wy


def load_real(name="totsu") -> np.ndarray:
    """Mean stress field over specimens, resampled to (NY,NX), robust-normalised [0,1]."""
    fn = f"{KOJIMA_DIR}/rect_dspss_exp_{name}.npy"
    stack = np.asarray(np.load(fn), dtype=float)          # (n, H, W)
    field = np.nanmean(stack, axis=0)                     # mean over specimens
    field = resize_to(field)
    lo, hi = np.nanpercentile(field, 1), np.nanpercentile(field, 99)  # robust clip
    field = np.clip((field - lo) / (hi - lo + 1e-9), 0, 1)
    return field


def feature_cloud(model, gc_grid, ell_grid, k=K_SAMPLES, seed0=0):
    """Emulator feature vector at every grid point -> (n_ell, n_gc, n_feat)."""
    n_ell, n_gc = len(ell_grid), len(gc_grid)
    cloud = np.zeros((n_ell, n_gc, len(F.FEATURE_NAMES)))
    for i, ell in enumerate(ell_grid):
        for j, gc in enumerate(gc_grid):
            cloud[i, j] = F.features(F.forward_field(model, gc, ell, k=k,
                                                     seed=seed0 + i * n_gc + j))
    return cloud


def main() -> None:
    t0 = time.time()
    model = F.load_generator()
    gc_grid, ell_grid = F.make_grids(GRID_N)

    print("[real] loading Kojima totsu (convex) stress field")
    d_real = load_real("totsu")
    f_real = F.features(d_real)

    print(f"[cloud] emulator feature cloud {GRID_N}x{GRID_N} ...")
    cloud = feature_cloud(model, gc_grid, ell_grid, k=K_SAMPLES)
    cflat = cloud.reshape(-1, len(F.FEATURE_NAMES))
    cmean = cflat.mean(axis=0)
    cstd  = cflat.std(axis=0)
    cmin  = cflat.min(axis=0)
    cmax  = cflat.max(axis=0)

    # --- domain-gap audit -----------------------------------------------------
    print("\n=== domain-gap audit (real stress vs emulator damage features) ===")
    print(f"  {'feature':<16} {'real':>8} {'emu_min':>8} {'emu_max':>8} {'z':>7}  in-range")
    z = (f_real - cmean) / (cstd + 1e-9)
    in_range = (f_real >= cmin) & (f_real <= cmax)
    for n, fr, mn, mx, zz, ir in zip(F.FEATURE_NAMES, f_real, cmin, cmax, z, in_range):
        print(f"  {n:<16} {fr:8.3f} {mn:8.3f} {mx:8.3f} {zz:7.2f}  {'YES' if ir else 'NO  <- OOD'}")
    n_in = int(in_range.sum())
    print(f"  -> {n_in}/{len(F.FEATURE_NAMES)} features inside emulator-reachable range")

    # --- best-match landscape (morphology only, NOT physics) ------------------
    # use only in-range features for the match; sigma = local cloud spread
    use = in_range.copy()
    if use.sum() == 0:
        print("  [verdict] ALL features OOD -> inversion is pure extrapolation; no landscape.")
        use[:] = True   # still draw something, fully caveated
    f_sigma = np.maximum(cstd, 0.05 * (np.abs(cmean) + 1e-6))
    diff = (cloud - f_real) / f_sigma
    diff[:, :, ~use] = 0.0
    maha = np.sqrt((diff ** 2).sum(axis=2))           # (n_ell, n_gc)
    match = np.exp(-0.5 * (maha ** 2))
    match = match / match.sum()
    mi, mj = np.unravel_index(np.argmax(match), match.shape)
    gc_bm, ell_bm = float(gc_grid[mj]), float(ell_grid[mi])
    print(f"\n  best emulator-equivalent match (in-range feats only):")
    print(f"    Gc_eq = {gc_bm:.3e}   ell_eq = {ell_bm:.4f}")
    print("  [verdict] This is the emulator config whose NORMALISED MORPHOLOGY best")
    print("            matches the real stress field. It is NOT a physical fracture")
    print("            toughness: the real observable is stress, not damage.")

    out = {
        "real_source": "kojima totsu (convex), mean of 12 specimens, robust-normalised",
        "feature_names": F.FEATURE_NAMES,
        "f_real": f_real.tolist(),
        "emu_cloud_min": cmin.tolist(), "emu_cloud_max": cmax.tolist(),
        "emu_cloud_mean": cmean.tolist(), "z_scores": z.tolist(),
        "in_range": in_range.tolist(), "n_in_range": n_in,
        "best_match_eq": {"Gc": gc_bm, "ell": ell_bm},
        "caveat": "emulator-equivalent morphology match; real observable is STRESS not DAMAGE",
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[save] {OUT_JSON}")

    _plot(d_real, model, gc_grid, ell_grid, match, (gc_bm, ell_bm),
          f_real, cmin, cmax, in_range)
    print(f"[save] {OUT_PNG}")
    print(f"[done] total {time.time()-t0:.0f}s")


def _plot(d_real, model, gc_grid, ell_grid, match, bm, f_real, cmin, cmax, in_range):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 3.8))

    ax0 = fig.add_subplot(1, 4, 1)
    im0 = ax0.imshow(d_real, origin="lower", aspect="auto", cmap="inferno")
    ax0.set_title("Kojima totsu (convex)\nnormalised stress field"); ax0.set_xlabel("x"); ax0.set_ylabel("y")
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    ax1 = fig.add_subplot(1, 4, 2)
    d_bm = F.forward_field(model, bm[0], bm[1], k=K_SAMPLES, seed=1)
    im1 = ax1.imshow(d_bm, origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=1)
    ax1.set_title(f"best emulator match\nGc_eq={bm[0]:.1e}, ell_eq={bm[1]:.3f}")
    ax1.set_xlabel("x"); fig.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(1, 4, 3)
    im2 = ax2.pcolormesh(gc_grid, ell_grid, match, shading="auto", cmap="viridis")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.scatter([bm[0]], [bm[1]], marker="X", s=120, c="cyan", edgecolor="black")
    ax2.set_xlabel("Gc_eq"); ax2.set_ylabel("ell_eq")
    ax2.set_title("morphology-match landscape\n(NOT physical posterior)")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(1, 4, 4)
    idx = np.arange(len(F.FEATURE_NAMES))
    ax3.barh(idx, cmax - cmin, left=cmin, color="lightgray", label="emu range")
    colors = ["C2" if ir else "C3" for ir in in_range]
    ax3.scatter(f_real, idx, color=colors, zorder=5, label="real")
    ax3.set_yticks(idx); ax3.set_yticklabels(F.FEATURE_NAMES, fontsize=8)
    ax3.set_title("domain-gap audit\n(green=in-range, red=OOD)")
    ax3.set_xlabel("feature value"); ax3.legend(fontsize=7)

    fig.suptitle("Real FLIR/TSA injection into inverse PF forensics — with domain-gap audit", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
