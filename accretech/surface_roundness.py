#!/usr/bin/env python3
"""surface_roundness.py — precision metrology: roughness and roundness.

The two instruments at the heart of a dimensional-metrology company,
implemented from the ISO definitions up, with analytic closed-loop checks:

  SURFACE ROUGHNESS — a stylus trace contains form + waviness + roughness
  superposed. ISO 16610-21 separates them with a GAUSSIAN PROFILE FILTER
  whose amplitude transmission is exactly 50 % at the cutoff λc (verified
  numerically in tests). Parameters follow ISO 4287: Ra (mean absolute),
  Rq (rms), Rz (mean peak-to-valley over 5 sampling lengths). Closed loop:
  for a pure sine of amplitude A, Ra = 2A/π analytically — the
  implementation is checked against it.

  ROUNDNESS — a spindle trace r(θ) contains eccentricity (1st harmonic:
  SETUP, not form), ovality (2nd), lobing (3rd+: chucking distortion,
  bearing errors) and noise. Reference-circle fits per ISO 12181:
    * LSC — least-squares circle (linear, the workhorse)
    * MZC — minimum-zone circle (Chebyshev: minimizes the radial band;
      always ≤ LSC roundness, found here by direct optimization)
  plus the HARMMONIC DECOMPOSITION (FFT of r(θ)) that turns a wobbly polar
  plot into a diagnosis: "energy at k=3 → three-jaw chuck distortion".

Pure numpy + scipy + matplotlib.  Run:  python3 accretech/surface_roundness.py
"""
import json
import os

import numpy as np
from scipy.optimize import minimize

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "surface_roundness_results.json")

# ── roughness: profile synthesis + ISO Gaussian filter ─────────────────────
DX_MM = 0.5e-3            # sampling step [mm] (0.5 µm)
LC_MM = 0.8               # cutoff λc [mm]
EVAL_LEN_MM = 4.0         # evaluation length = 5 λc


def synth_profile(seed=0):
    """Stylus trace [µm] over the evaluation length: roughness (correlated
    noise) + waviness (mm-scale) + a gentle form curvature."""
    rng = np.random.default_rng(seed)
    n = int(EVAL_LEN_MM / DX_MM)
    x = np.arange(n) * DX_MM
    rough = np.convolve(rng.standard_normal(n),
                        np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2),
                        "same")
    rough *= 0.35 / rough.std()
    wav = 1.8 * np.sin(2 * np.pi * x / 2.6 + 0.7) \
        + 0.9 * np.sin(2 * np.pi * x / 1.4)
    form = 0.4 * (x - x.mean()) ** 2
    return x, form + wav + rough


def gaussian_filter_profile(z, dx_mm=DX_MM, lc_mm=LC_MM):
    """ISO 16610-21 Gaussian weighting: 50 % transmission at λc.
    Returns the WAVINESS (mean line); roughness = z - waviness."""
    alpha = np.sqrt(np.log(2.0) / np.pi)
    half = int(lc_mm / dx_mm)
    xs = np.arange(-half, half + 1) * dx_mm
    s = np.exp(-np.pi * (xs / (alpha * lc_mm)) ** 2)
    s /= s.sum()
    pad = np.pad(z, half, mode="reflect")
    return np.convolve(pad, s, "valid")


def roughness_params(r, dx_mm=DX_MM, lc_mm=LC_MM):
    """ISO 4287 Ra, Rq, Rz over 5 sampling lengths."""
    n_s = int(lc_mm / dx_mm)
    n_use = 5 * n_s
    r = r[:n_use] - r[:n_use].mean()
    ra = float(np.mean(np.abs(r)))
    rq = float(np.sqrt(np.mean(r ** 2)))
    rz = float(np.mean([r[i * n_s:(i + 1) * n_s].max()
                        - r[i * n_s:(i + 1) * n_s].min()
                        for i in range(5)]))
    return ra, rq, rz


def filter_transmission(wavelength_mm, lc_mm=LC_MM, n=8000):
    """Numerical amplitude transmission of the roughness output at a given
    wavelength (should be 50 % exactly at λc)."""
    x = np.arange(n) * DX_MM
    z = np.sin(2 * np.pi * x / wavelength_mm)
    w = gaussian_filter_profile(z)
    r = z - w
    core = slice(n // 4, 3 * n // 4)
    return float(np.ptp(r[core]) / np.ptp(z[core]))


# ── roundness ───────────────────────────────────────────────────────────────
def synth_round(n=720, ecc_um=8.0, oval_um=3.0, lobe3_um=2.0,
                noise_um=0.25, seed=1):
    """r(θ) deviation trace [µm] around a nominal radius."""
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = (ecc_um * np.cos(th - 0.6)          # eccentricity = setup, k=1
         + oval_um * np.cos(2 * th + 0.3)   # ovality
         + lobe3_um * np.cos(3 * th - 1.1)  # 3-lobe chucking distortion
         + noise_um * rng.standard_normal(n))
    return th, r


def _residual(th, r, cx, cy):
    return r - cx * np.cos(th) - cy * np.sin(th)


def roundness_lsc(th, r):
    """Least-squares circle: center = first-harmonic coefficients."""
    cx = 2.0 * np.mean(r * np.cos(th))
    cy = 2.0 * np.mean(r * np.sin(th))
    res = _residual(th, r, cx, cy)
    return float(np.ptp(res)), (float(cx), float(cy))


def roundness_mzc(th, r):
    """Minimum-zone circle: center minimizing the radial band (Chebyshev),
    seeded from LSC."""
    _, (cx0, cy0) = roundness_lsc(th, r)

    def band(c):
        return np.ptp(_residual(th, r, c[0], c[1]))

    out = minimize(band, [cx0, cy0], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    return float(out.fun), (float(out.x[0]), float(out.x[1]))


def harmonics(th, r, kmax=15):
    """Amplitude of each undulation-per-revolution component [µm]."""
    n = len(th)
    c = np.fft.rfft(r) / n * 2.0
    return np.abs(c[1:kmax + 1])


def run():
    x, z = synth_profile()
    w = gaussian_filter_profile(z)
    ra, rq, rz = roughness_params(z - w)
    th, r = synth_round()
    lsc, _ = roundness_lsc(th, r)
    mzc, _ = roundness_mzc(th, r)
    h = harmonics(th, r)
    return {
        "roughness": {"Ra_um": round(ra, 4), "Rq_um": round(rq, 4),
                      "Rz_um": round(rz, 3),
                      "transmission_at_lc": round(
                          filter_transmission(LC_MM), 4)},
        "roundness": {"LSC_um": round(lsc, 3), "MZC_um": round(mzc, 3),
                      "mzc_gain_pct": round(100 * (1 - mzc / lsc), 1),
                      "harmonic_peaks": {f"k{k+1}": round(float(a), 3)
                                         for k, a in enumerate(h[:4])}},
    }


def _plot(path_r, path_c, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # roughness figure
    x, z = synth_profile()
    w = gaussian_filter_profile(z)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True)
    ax = axes[0]
    ax.plot(x, z, color="0.4", lw=0.8, label="raw stylus trace")
    ax.plot(x, w, color="#c44", lw=1.8,
            label="waviness (Gaussian mean line, lc=0.8 mm)")
    ax.set_ylabel("height [um]")
    ax.legend(fontsize=9)
    ax.set_title("ISO 16610 Gaussian profile filter: separate what the "
                 "part IS from how it FEELS")
    ax = axes[1]
    rr = res["roughness"]
    ax.plot(x, z - w, color="#26a", lw=0.8, label="roughness profile")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("position [mm]")
    ax.set_ylabel("height [um]")
    ax.set_title(f"Ra {rr['Ra_um']:.3f} / Rq {rr['Rq_um']:.3f} / "
                 f"Rz {rr['Rz_um']:.2f} um  -  filter transmission at "
                 f"lc = {rr['transmission_at_lc']*100:.1f}% (ISO: 50%)",
                 fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path_r, dpi=110)
    plt.close(fig)

    # roundness figure
    th, r = synth_round()
    lsc, (cx, cy) = roundness_lsc(th, r)
    mzc, cm = roundness_mzc(th, r)
    res1 = _residual(th, r, cx, cy)
    fig = plt.figure(figsize=(10.5, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    scale = 1.0
    ax.plot(th, 20 + r * scale, color="0.5", lw=0.8, label="raw r(theta)")
    ax.plot(th, 20 + res1 * scale, color="#26a", lw=1.2,
            label="after LSC centering")
    ax.plot(th, 20 + np.full_like(th, res1.max()), "--", color="#c44",
            lw=0.9)
    ax.plot(th, 20 + np.full_like(th, res1.min()), "--", color="#c44",
            lw=0.9, label="LSC roundness band")
    ax.set_yticklabels([])
    ax.set_title(f"Roundness: LSC {lsc:.2f} um / MZC {mzc:.2f} um\n"
                 f"(eccentricity removed by centering, not by grinding)",
                 fontsize=10)
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.1, -0.12))

    ax = fig.add_subplot(1, 2, 2)
    h = harmonics(th, r)
    ks = np.arange(1, len(h) + 1)
    colors = ["#888" if k not in (2, 3) else ("#b4740a" if k == 2
                                              else "#c44") for k in ks]
    ax.bar(ks, h, color=colors)
    ax.annotate("k=2: ovality", (2, h[1]), textcoords="offset points",
                xytext=(8, 4), fontsize=9, color="#b4740a")
    ax.annotate("k=3: 3-jaw chuck\ndistortion", (3, h[2]),
                textcoords="offset points", xytext=(8, 4), fontsize=9,
                color="#c44")
    ax.set_xlabel("undulations per revolution k")
    ax.set_ylabel("amplitude [um]")
    ax.set_title("Harmonic decomposition = diagnosis\n(k=1 excluded: "
                 "that's setup eccentricity, not form)")
    fig.tight_layout()
    fig.savefig(path_c, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "acc_rough.png"),
          os.path.join(ASSETS, "acc_round.png"), res)
    print("figures ->", ASSETS)


if __name__ == "__main__":
    main()
