#!/usr/bin/env python3
"""euv_printability.py — does this mask defect PRINT? Aerial-image ΔCD.

The existing inspection modules answer "can we SEE the defect?"
(die-to-die difference, adaptive statistics, phase channel). An actinic
EUV mask inspector exists to answer a harder question: "does the defect
PRINT on the wafer?" — a mask owner disposition decision worth real money
(repair vs ignore vs scrap). Detection strength and print impact are NOT
the same thing, and this module quantifies the gap.

Physics implemented from scratch:

  IMAGING     Abbe partially-coherent imaging at λ=13.5 nm, NA 0.33 (wafer
              side), conventional disk source σ=0.8: sum of coherent images
              over a source-point grid, pupil defocus for through-focus.
  MASK        16 nm half-pitch line/space (wafer scale), thin-mask model:
              multilayer reflectance 1.0, absorber 0.15·e^{i0.7π}.
  DEFECTS     amplitude type (absorber particle in the open space) and
              PHASE type (multilayer bump: Gaussian height profile,
              φ = 4πh/λ — invisible in transmission, deadly in print).
  METRICS     ΔCD% of the adjacent line at best focus and through focus
              (phase defects print worst OUT of focus), vs the pixel-level
              detection score an inspector's difference channel sees.

The money plot: detection score vs ΔCD scatter — same-visibility defects
split into printing and non-printing, which is why disposition must be
printability-aware (aerial metric), not pixel-count-aware.

Pure numpy + matplotlib.  Run:  python3 inspection/euv_printability.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "euv_printability_results.json")

# ── optics (wafer-scale nm) ─────────────────────────────────────────────────
WAVELEN = 13.5
NA = 0.33
SIGMA = 0.8            # partial coherence (disk source)
N = 256
DX = 1.0               # nm/pixel
PITCH = 32.0           # line/space pitch (16 nm half-pitch)
ABS_R = 0.15           # absorber field reflectance (amplitude)
ABS_PH = 0.7 * np.pi   # absorber phase
THRESH = 0.60          # print threshold, CALIBRATED so the defect-free
                       # aerial image prints the nominal 16 nm line at best
                       # focus (resist threshold calibration, standard
                       # lithography practice)


def _grid():
    x = (np.arange(N) - N / 2) * DX
    return np.meshgrid(x, x)


def mask_field(defect=None):
    """Complex mask reflectance. defect = (kind, x0, y0, size_nm) with kind
    'particle' (amplitude disk in an open space) or 'bump' (ML phase)."""
    X, Y = _grid()
    lines = (np.mod(X, PITCH) < PITCH / 2)          # True = absorber line
    m = np.where(lines, ABS_R * np.exp(1j * ABS_PH), 1.0).astype(complex)
    if defect is not None:
        kind, x0, y0, size = defect
        r2 = (X - x0) ** 2 + (Y - y0) ** 2
        if kind == "particle":
            m = np.where(r2 <= (size / 2) ** 2,
                         ABS_R * np.exp(1j * ABS_PH), m)
        elif kind == "bump":
            h = 2.0                                  # bump height [nm]
            w = size / 2.355                         # FWHM -> sigma
            phase = (4 * np.pi * h / WAVELEN) * np.exp(-r2 / (2 * w ** 2))
            m = m * np.exp(1j * phase)
    return m


_FX = np.fft.fftfreq(N, DX)


def aerial_image(m, defocus_nm=0.0, n_src=5):
    """Abbe sum over a σ-disk source grid; pupil = circ(NA/λ) + defocus."""
    FX, FY = np.meshgrid(_FX, _FX)
    f_na = NA / WAVELEN
    M = np.fft.fft2(m)
    I = np.zeros((N, N))
    ss = np.linspace(-SIGMA, SIGMA, n_src)
    n_used = 0
    for sx in ss:
        for sy in ss:
            if sx ** 2 + sy ** 2 > SIGMA ** 2:
                continue
            fx0, fy0 = sx * f_na, sy * f_na
            rho2 = ((FX - fx0) ** 2 + (FY - fy0) ** 2) / f_na ** 2
            P = (rho2 <= 1.0).astype(complex)
            # defocus: exp(i k z sqrt(1-(λf)^2)) ~ exp(-iπλz f²)
            P *= np.exp(-1j * np.pi * WAVELEN * defocus_nm
                        * ((FX - fx0) ** 2 + (FY - fy0) ** 2))
            I += np.abs(np.fft.ifft2(M * P)) ** 2
            n_used += 1
    I /= n_used
    return I / I.max()


def measure_cd(I, x_center_nm):
    """CD of the dark line nearest x_center_nm: width under THRESH along
    the horizontal cut through the image center."""
    row = I[N // 2]
    x = (np.arange(N) - N / 2) * DX
    dark = row < THRESH
    # find the dark run containing/nearest x_center
    best = None
    i = 0
    while i < N:
        if dark[i]:
            j = i
            while j < N - 1 and dark[j + 1]:
                j += 1
            c = 0.5 * (x[i] + x[j])
            wd = x[j] - x[i]
            if best is None or abs(c - x_center_nm) < abs(best[0]
                                                          - x_center_nm):
                best = (c, wd, i, j)
            i = j + 1
        else:
            i += 1
    if best is None:
        return np.nan
    i, j = best[2], best[3]

    # sub-pixel: linear interpolation of the two threshold crossings
    def cross(k0, k1):
        y0, y1 = row[k0], row[k1]
        if y1 == y0:
            return x[k1]
        return x[k0] + (THRESH - y0) / (y1 - y0) * (x[k1] - x[k0])

    left = cross(i - 1, i) if i > 0 else x[i]
    right = cross(j + 1, j) if j < N - 1 else x[j]
    return float(right - left)


def detection_score(defect):
    """What a difference-channel inspector sees: peak |I_def - I_ref| at
    best focus, in % of clear intensity."""
    I0 = aerial_image(mask_field(None))
    I1 = aerial_image(mask_field(defect))
    return float(np.max(np.abs(I1 - I0)) * 100.0)


def delta_cd(defect, defocus_nm=0.0):
    """ΔCD [%] of the absorber line adjacent to the defect. Bridging /
    line-merge (CD blow-up) is capped at 100 % — beyond that the defect is
    catastrophic either way and the number stops being a CD."""
    x_line = PITCH / 4              # center of the absorber line at x~8nm
    I0 = aerial_image(mask_field(None), defocus_nm)
    I1 = aerial_image(mask_field(defect), defocus_nm)
    cd0 = measure_cd(I0, x_line)
    cd1 = measure_cd(I1, x_line)
    if not np.isfinite(cd0) or not np.isfinite(cd1):
        return 100.0
    return float(np.clip((cd1 - cd0) / cd0 * 100.0, -100.0, 100.0))


def defect_population(seed=2):
    """A mixed population: particles and ML bumps of assorted sizes placed
    in the open space next to the measured line."""
    rng = np.random.default_rng(seed)
    pop = []
    for size in (5, 6, 8, 12, 16, 20, 26):
        for kind in ("particle", "bump"):
            x0 = PITCH * 0.75 + rng.uniform(-2, 2)   # in the open space
            y0 = rng.uniform(-8, 8)
            pop.append((kind, float(x0), float(y0), float(size)))
    return pop


def run():
    pop = defect_population()
    rows = []
    for d in pop:
        rows.append({
            "kind": d[0], "size_nm": d[3],
            "det_score_pct": round(detection_score(d), 2),
            "dcd_focus_pct": round(delta_cd(d, 0.0), 2),
            # phase defects are through-focus ASYMMETRIC — evaluate both
            # sides and keep the worse one
            "dcd_defoc_pct": round(max(delta_cd(d, +25.0),
                                       delta_cd(d, -25.0),
                                       key=abs), 2),
        })
    # printability = worst |ΔCD| across focus; spec: 10 % CD change
    for r in rows:
        r["dcd_worst_pct"] = round(max(abs(r["dcd_focus_pct"]),
                                       abs(r["dcd_defoc_pct"])), 2)
        r["prints"] = bool(r["dcd_worst_pct"] >= 10.0)
    return rows


def _plot(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.5, 7.6))
    gs = fig.add_gridspec(2, 3)

    # aerial image examples
    ex = [(None, "reference (16 nm HP)"),
          (("particle", PITCH * 0.75, 0.0, 20.0), "absorber particle 20 nm"),
          (("bump", PITCH * 0.75, 0.0, 20.0), "ML phase bump 20 nm (2 nm)")]
    for k, (d, title) in enumerate(ex):
        ax = fig.add_subplot(gs[0, k])
        I = aerial_image(mask_field(d), defocus_nm=-25.0 if d and
                         d[0] == "bump" else 0.0)
        ax.imshow(I, cmap="inferno", extent=[-N / 2, N / 2, -N / 2, N / 2])
        ax.contour(I, levels=[THRESH], colors="c", linewidths=0.7,
                   extent=[-N / 2, N / 2, N / 2, -N / 2])
        ax.set_title(title + ("  (defocus -25 nm)" if d and d[0] == "bump"
                              else ""), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # ΔCD vs size
    ax = fig.add_subplot(gs[1, 0:2])
    for kind, c in (("particle", "#c86"), ("bump", "#26a")):
        rs = [r for r in rows if r["kind"] == kind]
        s = [r["size_nm"] for r in rs]
        ax.plot(s, [abs(r["dcd_focus_pct"]) for r in rs], "o-", color=c,
                label=f"{kind}, best focus")
        ax.plot(s, [abs(r["dcd_defoc_pct"]) for r in rs], "s--", color=c,
                alpha=0.7, label=f"{kind}, worst of +/-25 nm defocus")
    ax.axhline(10, color="k", ls=":", lw=1, label="10% CD spec")
    ax.set_xlabel("defect size [nm]")
    ax.set_ylabel("|dCD| of adjacent line [%]")
    ax.set_title("Printability vs size: phase defects print WORSE out of "
                 "focus (asymmetric)")
    ax.legend(fontsize=8)

    # the money plot
    ax = fig.add_subplot(gs[1, 2])
    for kind, c, mk in (("particle", "#c86", "o"), ("bump", "#26a", "s")):
        rs = [r for r in rows if r["kind"] == kind]
        ax.scatter([r["det_score_pct"] for r in rs],
                   [r["dcd_worst_pct"] for r in rs], c=c, marker=mk, s=42,
                   label=kind)
    ax.axhline(10, color="k", ls=":", lw=1)
    ax.set_xlabel("detection score [% of clear]")
    ax.set_ylabel("worst |dCD| across focus [%]")
    ax.set_title("Seeing != printing:\nsame visibility, different fate")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    rows = run()
    n_print = sum(r["prints"] for r in rows)
    summary = {"pitch_nm": PITCH, "na": NA, "sigma": SIGMA,
               "defects": rows,
               "n_printing": n_print, "n_total": len(rows)}
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["defects"], indent=1)[:1200])
    print(f"printing {n_print}/{len(rows)}")
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "euv_print.png"), rows)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
