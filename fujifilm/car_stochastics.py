#!/usr/bin/env python3
"""car_stochastics.py — chemically amplified resist: where line-edge
roughness comes from, counted photon by photon.

The scanner (canon/litho_imaging.py) delivers an aerial image; the RESIST
turns it into an edge. In a chemically amplified resist (CAR) that
conversion is a chain of counting experiments, and the counting noise IS
the line-edge roughness (LER):

  EXPOSURE     absorbed photons per volume ~ Poisson(dose x image). Each
               absorbed photon converts a photo-acid generator with some
               probability -> the ACID map is a thinned Poisson field.
  PEB          during post-exposure bake the acid random-walks: a Gaussian
               blur of diffusion length L_d. Blur averages the shot noise
               DOWN but averages the image gradient down TOO.
  DEVELOP      deprotection above threshold -> per-row edge position; the
               3-sigma of edge positions along the line is the LER.

  THE PHYSICS  sigma_edge = sigma_acid / (acid gradient at the edge):
               noise over slope. Dose enters only through the counting
               statistics, so LER ~ dose^(-1/2) — verified here by fitting
               the exponent on the Monte-Carlo output. Diffusion gives the
               RLS TRIANGLE: more blur kills shot noise (good) and image
               log-slope (bad) -> an optimal L_d exists, and you cannot buy
               resolution, LER and sensitivity at once. That three-way
               trade is the photoresist designer's whole job.

Pure numpy + matplotlib.  Run:  python3 fujifilm/car_stochastics.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "car_stochastics_results.json")

PITCH_NM = 90.0          # line/space pitch (45 nm half-pitch class)
DX = 0.5                 # grid [nm]
NY = 400                 # rows along the line (LER statistics)
K_IMG = 0.75             # aerial image contrast from the scanner
PHOTONS_PER_NM2 = 8.0    # absorbed photons / nm^2 at dose 1.0 (DUV-scale)
PAG_YIELD = 0.35         # acid generation probability per absorbed photon
THICK_NM = 4.0           # interaction depth lumped into the 2-D sheet
L_DIFF_NM = 5.0          # nominal PEB diffusion length


def aerial(x, dose=1.0):
    """Aerial image intensity (clear-field units) x dose."""
    return dose * 0.5 * (1.0 + K_IMG * np.cos(2 * np.pi * x / PITCH_NM))


def acid_field(dose=1.0, seed=0, ny=NY):
    """Thinned-Poisson acid count map on a (ny, nx) grid."""
    rng = np.random.default_rng(seed)
    x = (np.arange(int(PITCH_NM / DX)) - PITCH_NM / DX / 2) * DX
    lam = aerial(x, dose) * PHOTONS_PER_NM2 * PAG_YIELD \
        * DX * DX * THICK_NM
    return x, rng.poisson(np.broadcast_to(lam, (ny, len(x))))


def peb_blur(acid, l_diff_nm=L_DIFF_NM):
    """Gaussian acid diffusion along both axes (reflect pad in x, wrap
    unnecessary: line is periodic in neither axis at this crop)."""
    if l_diff_nm <= 0:
        return acid.astype(float)
    sig = l_diff_nm / DX
    half = int(4 * sig)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(
        lambda r: np.convolve(np.pad(r, half, mode="reflect"), k, "valid"),
        1, acid.astype(float))
    out = np.apply_along_axis(
        lambda r: np.convolve(np.pad(r, half, mode="reflect"), k, "valid"),
        0, out)
    return out


def edge_positions(x, blurred, thresh=None):
    """Per-row deprotection-threshold crossing on the RIGHT flank: the
    aerial cosine peaks at x=0, so acid falls from the center outward and
    the printed edge is the downward crossing near x = +pitch/4."""
    if thresh is None:
        thresh = blurred.mean()
    n = blurred.shape[0]
    edges = np.full(n, np.nan)
    c = blurred.shape[1] // 2
    for r in range(n):
        row = blurred[r]
        if row[c] <= thresh:
            continue
        i = c + 1
        while i < len(row) and row[i] >= thresh:
            i += 1
        if i >= len(row) or row[i - 1] == row[i]:
            continue
        # row[i-1] >= thresh > row[i]: interpolate the crossing
        f = (row[i - 1] - thresh) / (row[i - 1] - row[i])
        edges[r] = x[i - 1] + f * (x[i] - x[i - 1])
    return edges[np.isfinite(edges)]


def ler_3sigma(dose=1.0, l_diff_nm=L_DIFF_NM, seed=0):
    x, a = acid_field(dose, seed)
    b = peb_blur(a, l_diff_nm)
    e = edge_positions(x, b)
    return float(3.0 * np.std(e))


def dose_exponent(doses=(0.5, 1.0, 2.0, 4.0), l_diff_nm=L_DIFF_NM):
    """Fit LER ~ dose^p; counting statistics predict p = -1/2."""
    lers = [np.mean([ler_3sigma(d, l_diff_nm, s) for s in range(3)])
            for d in doses]
    p = np.polyfit(np.log(doses), np.log(lers), 1)[0]
    return float(p), [round(v, 3) for v in lers]


def run():
    p, lers = dose_exponent()
    sweep = {}
    for ld in (2.0, 5.0, 10.0, 16.0, 22.0, 30.0, 40.0):
        sweep[f"Ld_{ld:.0f}nm"] = round(np.mean(
            [ler_3sigma(1.0, ld, s) for s in range(3)]), 3)
    best = min(sweep, key=sweep.get)
    return {
        "model": {"pitch_nm": PITCH_NM, "image_contrast": K_IMG,
                  "photons_per_nm2": PHOTONS_PER_NM2,
                  "pag_yield": PAG_YIELD},
        "ler_vs_dose_3sig_nm": dict(zip(["d0.5", "d1", "d2", "d4"], lers)),
        "dose_exponent": {"fit": round(p, 3), "theory": -0.5},
        "ler_vs_diffusion_nm": sweep,
        "optimal_Ld": best,
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.5, 4.4))
    ax = fig.add_subplot(1, 3, 1)
    x, a = acid_field(1.0, seed=1, ny=120)
    b = peb_blur(a)
    th = b.mean()
    ax.imshow(b, aspect="auto", cmap="viridis",
              extent=[x[0], x[-1], 0, 120 * DX])
    e = edge_positions(x, b, th)
    ax.plot(e, np.linspace(0, 120 * DX, len(e)), color="w", lw=1)
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y along line [nm]")
    ax.set_title("Blurred acid field + per-row edge:\nthe wiggle IS "
                 "counting noise", fontsize=10)

    ax = fig.add_subplot(1, 3, 2)
    doses = np.array([0.5, 1.0, 2.0, 4.0])
    lers = [res["ler_vs_dose_3sig_nm"][k] for k in
            ("d0.5", "d1", "d2", "d4")]
    ax.loglog(doses, lers, "o", color="#00794f", ms=6)
    pfit = res["dose_exponent"]["fit"]
    ax.loglog(doses, lers[1] * (doses / 1.0) ** (-0.5), "--", color="0.5",
              label="dose^(-1/2) (counting statistics)")
    ax.set_xlabel("relative dose")
    ax.set_ylabel("LER 3-sigma [nm]")
    ax.set_title(f"Shot-noise law: fitted exponent {pfit:.2f}\n(theory "
                 f"-0.50) — sensitivity buys roughness", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(1, 3, 3)
    lds = [2.0, 5.0, 10.0, 16.0, 22.0, 30.0, 40.0]
    vals = [res["ler_vs_diffusion_nm"][f"Ld_{ld:.0f}nm"] for ld in lds]
    ax.plot(lds, vals, "-o", color="#00794f", lw=1.8, ms=5)
    ax.annotate("too little blur:\nshot noise survives", (lds[0], vals[0]),
                textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.annotate("too much blur:\nedge slope dies", (lds[-1], vals[-1]),
                textcoords="offset points", xytext=(-92, -16), fontsize=8)
    ax.set_xlabel("PEB diffusion length L_d [nm]")
    ax.set_ylabel("LER 3-sigma [nm]")
    ax.set_title("The RLS trade-off has an interior optimum:\nresist "
                 "design = choosing where to sit", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "ff_car.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
