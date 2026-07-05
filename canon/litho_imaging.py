#!/usr/bin/env python3
"""litho_imaging.py — KrF projection lithography: Bossung curves, the process
window, and why the illuminator is a resolution knob.

A KrF scanner-class imaging model (λ=248 nm, NA 0.86 — FPA-6300ES class),
implemented as an Abbe sum over the source, with the mask entering as its
discrete diffraction orders (exact for line/space patterns, no FFT grid
error):

  IMAGING       for each source point (sx, sy) on the σ-disk, the coherent
                amplitude is Σ_m c_m · P(m/p + sx·NA/λ, sy·NA/λ) · e^{2πimx/p}
                with c_m the binary-mask Fourier coefficients and P the
                pupil. Defocus enters as the EXACT phase
                exp(i·2πz/λ·√(1-(λf)²)) at each order's absolute pupil
                position — the cross term this keeps (2·fm·fx0) is
                precisely the mechanism by which off-axis illumination
                buys depth of focus. Intensity = source average of
                |amplitude|².

  BOSSUNG       printed CD (threshold resist model, threshold calibrated so
                the nominal feature prints at best focus / nominal dose)
                versus focus, one curve per dose: the lithographer's
                fingerprint plot. Aberration-free ⇒ curves are symmetric in
                focus (verified in tests).

  PROCESS WINDOW at each defocus, the dose range that keeps CD within ±10%
                of target; the through-focus overlap gives DOF at a given
                exposure latitude — THE manufacturability number.

  ILLUMINATION  conventional disk σ, annular, and x-dipole. Off-axis
                illumination re-centers the ±1 orders in the pupil: at
                k1 near the limit it buys BOTH image contrast and DOF —
                the "free" resolution knob that kept 248 nm printing far
                below the naive λ scale.

Closed-loop checks: a coherent-limit pitch cutoff at λ/(NA(1+σ)), CD ≈ mask
CD for a relaxed feature, Bossung symmetry.

Pure numpy + matplotlib.  Run:  python3 canon/litho_imaging.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "litho_imaging_results.json")

WAVELEN = 248.0          # KrF [nm]
NA = 0.86
HP_NOM = 100.0           # nominal half-pitch [nm]  (k1 = 0.347)
CD_TOL = 0.10            # process window: CD within ±10 % of target
N_X = 512                # sample points across one pitch period


def mask_orders(pitch_nm, m_max=7):
    """Fourier coefficients of a binary 50 % duty line/space mask
    (dark line centered at x=0): c_0 = 0.5, c_m = sin(πm/2)/(πm)."""
    ms = np.arange(-m_max, m_max + 1)
    cs = np.where(ms == 0, 0.5,
                  np.sin(np.pi * ms / 2) / (np.pi * np.where(ms == 0, 1, ms)))
    # dark line (absorber) at center: field = 1 - lines(x)
    cs = -cs
    cs[ms == 0] += 1.0
    return ms, cs


def source_points(kind="conv", sigma=0.7, n=13):
    """(sx, sy) source grid in σ units, |s|≤1 refers to the NA edge.
    kinds: conv (disk σ), annular (0.55–0.85), dipole_x (poles at
    ±σ_opt = ±λ/(2·pitch·NA) = ±0.72 for the 200 nm pitch — the pole
    position that lands the 0th and 1st orders symmetric in the pupil,
    killing the linear defocus sensitivity; r 0.2)."""
    ss = np.linspace(-1.0, 1.0, n)
    SX, SY = np.meshgrid(ss, ss)
    r = np.hypot(SX, SY)
    if kind == "conv":
        keep = r <= sigma
    elif kind == "annular":
        keep = (r >= 0.55) & (r <= 0.85)
    elif kind == "dipole_x":
        keep = (np.hypot(np.abs(SX) - 0.72, SY) <= 0.2)
    else:
        raise ValueError(kind)
    return SX[keep], SY[keep]


def aerial_image(pitch_nm, defocus_nm=0.0, kind="conv", sigma=0.7):
    """Normalized 1-D aerial image over one period (order-sum Abbe)."""
    ms, cs = mask_orders(pitch_nm)
    f_na = NA / WAVELEN
    sx, sy = source_points(kind, sigma)
    x = np.linspace(-pitch_nm / 2, pitch_nm / 2, N_X, endpoint=False)
    I = np.zeros(N_X)
    for k in range(len(sx)):
        fx0, fy0 = sx[k] * f_na, sy[k] * f_na
        fm = ms / pitch_nm
        f2 = (fm + fx0) ** 2 + fy0 ** 2
        inside = f2 <= f_na ** 2
        if not inside.any():
            continue
        # exact defocus phase at each order's absolute pupil position
        # (keeps the 2*fm*fx0 cross term = the off-axis DOF mechanism)
        ph = np.exp(2j * np.pi * defocus_nm / WAVELEN
                    * np.sqrt(np.maximum(0.0,
                              1.0 - WAVELEN ** 2 * f2[inside])))
        amp = (cs[inside] * ph)[None, :] \
            * np.exp(2j * np.pi * np.outer(x, fm[inside]))
        I += np.abs(amp.sum(axis=1)) ** 2
    I /= len(sx)
    # I is already in clear-field units (clear field: single unit order)
    return x, I


def measure_cd(x, I, thresh):
    """Width of the dark region (I<thresh) around x=0, by interpolation."""
    n = len(x)
    c = n // 2
    if I[c] >= thresh:
        return float("nan")
    i = c
    while i > 0 and I[i - 1] < thresh:
        i -= 1
    j = c
    while j < n - 1 and I[j + 1] < thresh:
        j += 1
    if i == 0 or j == n - 1:
        return float("nan")
    # linear sub-sample crossings at both edges
    xl = x[i - 1] + (I[i - 1] - thresh) / (I[i - 1] - I[i]) \
        * (x[i] - x[i - 1])
    xr = x[j] + (I[j] - thresh) / (I[j] - I[j + 1]) * (x[j + 1] - x[j])
    return float(xr - xl)


def calibrate_threshold(pitch_nm, kind="conv", sigma=0.7):
    """Resist threshold that prints the nominal CD (=pitch/2) at focus."""
    x, I = aerial_image(pitch_nm, 0.0, kind, sigma)
    target = pitch_nm / 2
    ths = np.linspace(0.05, 0.95, 181)
    cds = np.array([measure_cd(x, I, t) for t in ths])
    ok = np.isfinite(cds)
    return float(np.interp(target, cds[ok], ths[ok]))


def bossung(pitch_nm, kind="conv", sigma=0.7, defoci=None, doses=None):
    """CD(defocus, dose) surface. Dose d scales the image: CD from
    threshold/d."""
    th0 = calibrate_threshold(pitch_nm, kind, sigma)
    defoci = np.linspace(-500, 500, 21) if defoci is None else defoci
    doses = np.linspace(0.85, 1.15, 13) if doses is None else doses
    cd = np.full((len(doses), len(defoci)), np.nan)
    for j, z in enumerate(defoci):
        x, I = aerial_image(pitch_nm, z, kind, sigma)
        for i, d in enumerate(doses):
            cd[i, j] = measure_cd(x, I, th0 / d)
    return defoci, doses, cd, th0


def process_window(pitch_nm, kind="conv", sigma=0.7, el_pct=5.0):
    """DOF [nm] at the given exposure latitude: through-focus overlap of the
    dose interval keeping CD within ±10 % of target."""
    defoci = np.linspace(0, 800, 41)          # symmetric: scan one side
    doses = np.linspace(0.75, 1.30, 45)
    _, _, cd, _ = bossung(pitch_nm, kind, sigma, defoci, doses)
    target = pitch_nm / 2
    lo, hi = target * (1 - CD_TOL), target * (1 + CD_TOL)
    d_lo, d_hi = [], []
    for j in range(len(defoci)):
        ok = np.isfinite(cd[:, j]) & (cd[:, j] >= lo) & (cd[:, j] <= hi)
        if not ok.any():
            d_lo.append(np.nan)
            d_hi.append(np.nan)
        else:
            d_lo.append(doses[ok].min())
            d_hi.append(doses[ok].max())
    d_lo, d_hi = np.array(d_lo), np.array(d_hi)
    need = el_pct / 100.0
    dof = 0.0
    for j in range(len(defoci)):
        band_lo = np.nanmax(d_lo[:j + 1]) if j > 0 else d_lo[0]
        band_hi = np.nanmin(d_hi[:j + 1]) if j > 0 else d_hi[0]
        if not np.isfinite(band_lo) or not np.isfinite(band_hi):
            break
        if (band_hi - band_lo) / ((band_hi + band_lo) / 2) < need:
            break
        dof = defoci[j]
    return 2.0 * dof     # symmetric in focus


def image_contrast(pitch_nm, kind="conv", sigma=0.7, defocus_nm=0.0):
    x, I = aerial_image(pitch_nm, defocus_nm, kind, sigma)
    return float((I.max() - I.min()) / (I.max() + I.min() + 1e-12))


def run():
    hp = HP_NOM
    pitch = 2 * hp
    res = {"optics": {"lambda_nm": WAVELEN, "NA": NA,
                      "k1_at_100nm_hp": round(hp * NA / WAVELEN, 3)},
           "rayleigh": {
               "min_hp_k1_0.25_nm": round(0.25 * WAVELEN / NA, 1),
               "coherent_cutoff_pitch_nm":
                   round(WAVELEN / (NA * (1 + 0.7)), 1)},
           "contrast_at_100nm_hp": {
               k: round(image_contrast(pitch, k), 3)
               for k in ("conv", "annular", "dipole_x")},
           "dof_at_5pct_EL_nm": {
               k: round(process_window(pitch, k), 0)
               for k in ("conv", "annular", "dipole_x")},
           "dof_relaxed_130nm_conv_nm":
               round(process_window(2 * 130.0, "conv"), 0)}
    return res


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pitch = 2 * HP_NOM
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))

    ax = axes[0]
    defoci = np.linspace(-500, 500, 21)
    doses = np.linspace(0.9, 1.1, 5)
    dz, dd, cd, _ = bossung(pitch, "conv", defoci=defoci, doses=doses)
    for i, d in enumerate(dd):
        ax.plot(dz, cd[i], "-o", ms=2.5, lw=1.3,
                label=f"dose {d:.2f}")
    ax.axhline(HP_NOM, color="0.6", lw=0.8)
    ax.axhspan(HP_NOM * 0.9, HP_NOM * 1.1, color="0.85", alpha=0.5)
    ax.set_xlabel("defocus [nm]")
    ax.set_ylabel("printed CD [nm]")
    ax.set_title("Bossung curves (100 nm L/S, conventional σ0.7)\n"
                 "gray band = CD spec ±10%", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1]
    for k, c, lab in (("conv", "#666", "conventional σ0.7"),
                      ("annular", "#b4740a", "annular 0.55–0.85"),
                      ("dipole_x", "#c00000", "dipole σ0.72 = λ/2pNA")):
        zz = np.linspace(0, 700, 15)
        ctr = [image_contrast(pitch, k, defocus_nm=z) for z in zz]
        ax.plot(zz, ctr, "-o", ms=3, lw=1.6, color=c, label=lab)
    ax.set_xlabel("defocus [nm]")
    ax.set_ylabel("image contrast")
    ax.set_title("Illumination is a resolution knob:\ncontrast through "
                 "focus at k1=0.35", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[2]
    dof = res["dof_at_5pct_EL_nm"]
    names = ["conv", "annular", "dipole_x"]
    labels = ["conventional", "annular", "dipole"]
    colors = ["#666", "#b4740a", "#c00000"]
    ax.bar(labels, [dof[n] for n in names], color=colors)
    for i, n in enumerate(names):
        ax.annotate(f"{dof[n]:.0f} nm", (i, dof[n]),
                    ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("DOF @ 5% exposure latitude [nm]")
    ax.set_title("Process window at 100 nm half-pitch:\noff-axis "
                 "illumination buys usable focus budget", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "canon_litho.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
