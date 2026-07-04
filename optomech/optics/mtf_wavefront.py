#!/usr/bin/env python3
"""mtf_wavefront.py — wave-optics image-quality (MTF) model for a lens barrel.

The C++ `tolerance` module scores a barrel by geometric boresight spot [µm].
But the real acceptance criterion for a camera lens is **MTF** (contrast at a
spatial frequency), and MTF is set by the *wavefront*, not just ray positions.
This module is the mechanical designer's true optical KPI: it maps a mechanical
error to a wavefront error to an MTF loss.

Pipeline (physical wave optics, pure NumPy):

  mechanical error  ->  Zernike wavefront W(rho,theta)  ->  pupil P = circ*e^{i2piW}
                    ->  PSF = |FFT(P)|^2  ->  MTF = |autocorr(P)| (via IFFT(PSF))

What a barrel designer owns is the **mechanics -> wavefront** map:
  * axial despace / thermal focus drift  -> defocus  (Zernike Z4)
  * element decenter / tilt              -> coma     (Zernike Z7/Z8)
  * (astigmatism Z5/Z6 from squeeze, kept small)
Optical sensitivities (waves of aberration per unit mechanical error) come from
the lens prescription's tolerance table; representative values are used here.

Analyses:
  1. MTF curves — diffraction-limited vs nominal vs thermally-defocused vs a
     tolerance-perturbed sample; the aberration-free FE MTF is validated against
     the closed-form diffraction MTF of a circular aperture.
  2. Thermal defocus — barrel CTE (axial) + lens dn/dT combine into a focus
     shift; its MTF@spec cost is quantified over temperature.
  3. Monte-Carlo tolerance -> MTF@spec distribution, and the decenter tolerance
     that holds MTF@spec above the requirement.

Run:  python3 mtf_wavefront.py
"""
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Optical system (matches the C++ optomech 4-group demo) -----------------
WAVELEN_UM = 0.55          # design wavelength (green) [µm]
F_NUMBER = 2.8             # working f/#
EFL_MM = 33.3              # effective focal length [mm]
SPEC_LPMM = 60.0           # MTF spec spatial frequency [line-pairs/mm]
MTF_SPEC = 0.35            # required MTF at SPEC_LPMM
CUTOFF_LPMM = 1.0e3 / (WAVELEN_UM * F_NUMBER)   # incoherent cutoff [lp/mm]

# ---- Mechanics -> wavefront sensitivities (waves RMS per unit) --------------
# Representative values from a lens tolerance table (would come from the optical
# design's sensitivity analysis for the real prescription).
COMA_PER_DECENTER = 0.055    # waves RMS coma per µm element decenter
ASTIG_PER_TILT = 0.030       # waves RMS astig per arc-min element tilt
# Thermal: barrel axial CTE (from the axisymmetric FE) + lens dn/dT power drift.
BARREL_UM_PER_C = 0.97       # axial back-focal drift [µm/°C] (barrel expansion)
LENS_UM_PER_C = 0.55         # focus drift [µm/°C] from dn/dT of the elements
FOCUS_UM_PER_C = BARREL_UM_PER_C + LENS_UM_PER_C

# ---- Pupil grid -------------------------------------------------------------
NPIX = 512
R_PIX = 128                  # pupil radius in pixels (padding gives fine MTF)


def _pupil_grid():
    y, x = np.mgrid[0:NPIX, 0:NPIX].astype(float)
    cx = cy = NPIX / 2.0
    rho = np.hypot(x - cx, y - cy) / R_PIX
    theta = np.arctan2(y - cy, x - cx)
    mask = rho <= 1.0
    return rho, theta, mask


RHO, THETA, MASK = _pupil_grid()


def zernike_wavefront(c_defocus=0.0, c_coma=0.0, c_astig=0.0):
    """Wavefront [waves] over the pupil from RMS-normalized Zernike coeffs."""
    rho, th = RHO, THETA
    z4 = np.sqrt(3) * (2 * rho**2 - 1)                 # defocus
    z7 = np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.cos(th)   # coma (x)
    z5 = np.sqrt(6) * rho**2 * np.cos(2 * th)          # astigmatism
    return c_defocus * z4 + c_coma * z7 + c_astig * z5


def mtf_from_wavefront(W_waves, n_freq=80):
    """Return (freq_lpmm, mtf) for a wavefront map [waves] over the pupil.

    MTF is the modulus of the normalized optical transfer function, computed as
    the (auto-)correlation of the complex pupil via PSF = |FFT(P)|^2 and
    OTF = IFFT(PSF). The frequency axis is calibrated so the diffraction cutoff
    (support edge of the pupil autocorrelation, at pixel offset 2*R_PIX) maps to
    1/(lambda*N).
    """
    P = MASK * np.exp(1j * 2 * np.pi * W_waves)
    psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(P)))) ** 2
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    c = NPIX // 2
    mtf_line = np.abs(otf[c, c:])
    mtf_line = mtf_line / mtf_line[0]
    # pixel offset k -> frequency: k/(2*R_PIX) * cutoff
    kmax = 2 * R_PIX
    k = np.arange(kmax)
    freq = k / (2.0 * R_PIX) * CUTOFF_LPMM
    mtf = mtf_line[:kmax]
    # resample to n_freq points up to cutoff for tidy curves
    fq = np.linspace(0, CUTOFF_LPMM, n_freq)
    return fq, np.interp(fq, freq, mtf)


def diffraction_mtf(freq_lpmm):
    """Closed-form MTF of an aberration-free circular aperture (for validation)."""
    nu = np.clip(freq_lpmm / CUTOFF_LPMM, 0, 1)
    return (2 / np.pi) * (np.arccos(nu) - nu * np.sqrt(1 - nu**2))


def defocus_waves(focus_shift_um):
    """RMS defocus Zernike coeff [waves] for an image-plane focus shift [µm].

    Peak defocus wavefront (P-V) = dz/(8 N^2); the RMS Z4 coeff = W_pv/(2*sqrt3).
    """
    w_pv_waves = (focus_shift_um / (8.0 * F_NUMBER**2)) / WAVELEN_UM
    return w_pv_waves / (2.0 * np.sqrt(3))


def mtf_at(freq_lpmm, W_waves):
    fq, m = mtf_from_wavefront(W_waves)
    return float(np.interp(freq_lpmm, fq, m))


def _plot_curves(path):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    fq, m_diff = mtf_from_wavefront(np.zeros_like(RHO))
    ax.plot(fq, diffraction_mtf(fq), "k--", lw=1.3, label="Diffraction limit (analytic)")
    ax.plot(fq, m_diff, color="#1f77b4", lw=2.0, label="Nominal FE (wave-optics)")

    # Thermal defocus at +20 °C.
    c4 = defocus_waves(FOCUS_UM_PER_C * 20.0)
    _, m_th = mtf_from_wavefront(zernike_wavefront(c_defocus=c4))
    ax.plot(fq, m_th, color="#d62728", lw=2.0,
            label=f"Thermal +20°C (defocus {c4:.03f}λ rms)")

    # A tolerance-perturbed sample (decenter+tilt).
    W = zernike_wavefront(c_coma=COMA_PER_DECENTER * 3.0,
                          c_astig=ASTIG_PER_TILT * 2.0)
    _, m_tol = mtf_from_wavefront(W)
    ax.plot(fq, m_tol, color="#2ca02c", lw=2.0,
            label="Tol sample (3µm decenter + 2' tilt)")

    ax.axvline(SPEC_LPMM, color="#888", ls=":", lw=1)
    ax.axhline(MTF_SPEC, color="#888", ls=":", lw=1)
    ax.annotate(f"spec: MTF≥{MTF_SPEC:.2f}\n@{SPEC_LPMM:.0f} lp/mm",
                xy=(SPEC_LPMM, MTF_SPEC), xytext=(SPEC_LPMM + 12, MTF_SPEC + 0.12),
                fontsize=8.5, color="#555",
                arrowprops=dict(arrowstyle="->", color="#888", lw=.8))
    ax.set_xlim(0, CUTOFF_LPMM * 0.72)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Spatial frequency [line-pairs/mm]")
    ax.set_ylabel("MTF (modulation)")
    ax.set_title(f"Wave-optics MTF  ·  f/{F_NUMBER}, λ={WAVELEN_UM}µm, "
                 f"cutoff {CUTOFF_LPMM:.0f} lp/mm", fontsize=10)
    ax.legend(fontsize=8.2, loc="upper right", framealpha=.95)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_tolerance(path, seed=7, n=4000):
    rng = np.random.default_rng(seed)
    # 1) Monte-Carlo MTF@spec over a manufacturing tolerance box.
    dec = rng.normal(0, 2.0, n)      # element decenter [µm], sigma 2µm
    tilt = rng.normal(0, 1.5, n)     # element tilt [arc-min], sigma 1.5'
    dT = rng.uniform(-15, 15, n)     # operating temperature swing [°C]
    mtf_mc = np.empty(n)
    for i in range(n):
        W = zernike_wavefront(
            c_defocus=defocus_waves(FOCUS_UM_PER_C * dT[i]),
            c_coma=COMA_PER_DECENTER * dec[i],
            c_astig=ASTIG_PER_TILT * tilt[i])
        mtf_mc[i] = mtf_at(SPEC_LPMM, W)
    p10 = float(np.percentile(mtf_mc, 10))
    frac_pass = float(np.mean(mtf_mc >= MTF_SPEC))

    # 2) Decenter tolerance sweep: median MTF@spec vs |decenter|.
    dvals = np.linspace(0, 8, 25)
    mtf_sweep = np.array([mtf_at(SPEC_LPMM,
                          zernike_wavefront(c_coma=COMA_PER_DECENTER * d))
                          for d in dvals])
    # largest decenter that still meets the spec
    ok = dvals[mtf_sweep >= MTF_SPEC]
    dec_tol = float(ok.max()) if ok.size else 0.0

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    a1.hist(mtf_mc, bins=45, color="#1f77b4", alpha=.85)
    a1.axvline(MTF_SPEC, color="#d62728", ls="--", lw=1.5,
               label=f"spec {MTF_SPEC:.2f}")
    a1.axvline(p10, color="#2ca02c", ls=":", lw=1.5, label=f"P10 {p10:.2f}")
    a1.set_title(f"MC MTF@{SPEC_LPMM:.0f}lp/mm  ·  pass {frac_pass*100:.0f}%",
                 fontsize=9.5)
    a1.set_xlabel("MTF @ spec"); a1.set_ylabel("count")
    a1.legend(fontsize=8.5)

    a2.plot(dvals, mtf_sweep, "-o", color="#1f77b4", ms=3.5)
    a2.axhline(MTF_SPEC, color="#d62728", ls="--", lw=1.5, label=f"spec {MTF_SPEC:.2f}")
    a2.axvline(dec_tol, color="#2ca02c", ls=":", lw=1.5,
               label=f"tol {dec_tol:.1f}µm")
    a2.set_title("Decenter tolerance from MTF spec", fontsize=9.5)
    a2.set_xlabel("Element decenter [µm]"); a2.set_ylabel(f"MTF @ {SPEC_LPMM:.0f}lp/mm")
    a2.legend(fontsize=8.5); a2.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return p10, frac_pass, dec_tol


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    figs = os.path.join(here, "figures")
    os.makedirs(figs, exist_ok=True)

    # Validate the aberration-free FE MTF against the analytic diffraction MTF.
    fq, m_fe = mtf_from_wavefront(np.zeros_like(RHO))
    m_an = diffraction_mtf(fq)
    band = (fq > 0) & (fq < 0.6 * CUTOFF_LPMM)
    val_err = float(np.max(np.abs(m_fe[band] - m_an[band])))

    mtf_nom = mtf_at(SPEC_LPMM, np.zeros_like(RHO))
    c4_20 = defocus_waves(FOCUS_UM_PER_C * 20.0)
    mtf_th20 = mtf_at(SPEC_LPMM, zernike_wavefront(c_defocus=c4_20))

    _plot_curves(os.path.join(figs, "mtf_curves.png"))
    p10, frac_pass, dec_tol = _plot_tolerance(os.path.join(figs, "mtf_tolerance.png"))

    out = {
        "wavelength_um": WAVELEN_UM,
        "f_number": F_NUMBER,
        "cutoff_lpmm": round(CUTOFF_LPMM, 1),
        "spec_lpmm": SPEC_LPMM,
        "mtf_spec": MTF_SPEC,
        "validation_max_abs_err_vs_diffraction": round(val_err, 4),
        "mtf_nominal_at_spec": round(mtf_nom, 3),
        "focus_drift_um_per_C": round(FOCUS_UM_PER_C, 3),
        "defocus_waves_at_20C": round(c4_20, 4),
        "mtf_at_spec_20C": round(mtf_th20, 3),
        "mc_mtf_p10_at_spec": round(p10, 3),
        "mc_fraction_pass": round(frac_pass, 3),
        "decenter_tolerance_um": round(dec_tol, 2),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open(os.path.join(here, "mtf_wavefront_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
