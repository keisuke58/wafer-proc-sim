#!/usr/bin/env python3
"""cmos_signal_chain.py — CMOS image-sensor signal chain from photons to DN.

The optomech modules cover the LENS side (FEM, MTF, servo, CAD). This module
covers the SENSOR side — the device physics an image-sensor engineer
characterizes on every new pixel:

  1. PIXEL MODEL — photon shot noise, QE, full-well saturation, dark current,
     read noise, PRNU/DSNU fixed-pattern noise, conversion gain, 12-bit ADC.
  2. PHOTON TRANSFER CURVE (PTC) — the standard characterization: plot
     temporal variance vs mean signal, read the noise regimes off the slopes
     (read floor → shot ∝ S^0.5 → PRNU ∝ S → full-well collapse), and
     ESTIMATE the conversion gain from the shot-noise slope. The estimate is
     checked against the ground-truth gain the simulator was built with —
     the closed loop that proves the method, not just the pretty curve.
  3. SNR / DYNAMIC RANGE — SNR vs exposure, dynamic range in dB, and the
     SNR10 illuminance (lux needed for SNR=10 — the metric mobile-sensor
     datasheets quote).
  4. SPLIT-EXPOSURE HDR — long/short exposure merge: linearize, blend, and
     show the extended dynamic range with the characteristic SNR dip at the
     stitch point (the cost of HDR that single-exposure plots hide).

Pure numpy + matplotlib.  Run:  python3 optomech/sensor/cmos_signal_chain.py
"""
import json
import os

import numpy as np

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "cmos_signal_chain_results.json")

# ── pixel / readout parameters (1.0 µm-class mobile BSI pixel) ──────────────
QE = 0.75                 # quantum efficiency at 550 nm
FULL_WELL_E = 6000.0      # full-well capacity [e-]
READ_NOISE_E = 1.6        # read noise [e- rms]
DARK_E_S = 3.0            # dark current at 60 °C [e-/s]
PRNU = 0.008              # photo-response non-uniformity (gain FPN, rms)
DSNU_E = 1.0              # dark-signal non-uniformity [e- rms offset FPN]
ADC_BITS = 12
CONV_GAIN_DN_E = 0.65     # conversion gain [DN/e-] (system gain, truth)
ADC_MAX = 2 ** ADC_BITS - 1

# photometry for SNR10 (f/2.0 lens, 1/30 s, 1.0 µm pixel, 550 nm green)
PIX_AREA_UM2 = 1.0
T_EXP_S = 1.0 / 30.0
F_NUM = 2.0
# photons/µm² per lux·s at the sensor for a f/2.0 lens imaging a gray scene
# (standard photometric conversion, luminous efficacy 683 lm/W @555 nm,
# lens transmission 0.9): ~ lux * t * 4e3 / (4*N^2) photons/µm²
PHOTONS_PER_LUX_S_UM2 = 4.0e3 * 0.9 / (4.0 * F_NUM ** 2)


def expose(n_photons, n_pix=20000, t_exp=T_EXP_S, rng=None, prnu_map=None,
           dsnu_map=None):
    """Simulate one exposure of n_pix pixels at a mean photon count.

    Returns digital numbers (DN) after the full chain:
    photons → shot noise → QE → PRNU gain FPN → + dark (shot) → full-well
    clip → + read noise → conversion gain → ADC quantize/clip.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if prnu_map is None:
        prnu_map = 1.0 + PRNU * rng.standard_normal(n_pix)
    if dsnu_map is None:
        dsnu_map = DSNU_E * rng.standard_normal(n_pix)

    # Poisson thinning: photon arrival AND QE selection are one Poisson
    # process with mean ph*QE (a deterministic *QE after the draw would
    # wrongly shrink shot variance by another factor QE).
    sig_e = rng.poisson(max(n_photons, 0.0) * QE * prnu_map)
    dark_e = rng.poisson(DARK_E_S * t_exp, n_pix) + dsnu_map
    e = np.clip(sig_e + dark_e, 0.0, FULL_WELL_E)
    e_read = e + READ_NOISE_E * rng.standard_normal(n_pix)
    dn = np.clip(np.round(e_read * CONV_GAIN_DN_E), 0, ADC_MAX)
    return dn


def photon_transfer_curve(n_levels=40, rng=None):
    """Mean-variance sweep with FPN removed the standard way: the temporal
    variance is taken from the DIFFERENCE of two identical exposures
    (var = var(a-b)/2), which cancels the fixed pattern."""
    if rng is None:
        rng = np.random.default_rng(7)
    n_pix = 20000
    prnu_map = 1.0 + PRNU * rng.standard_normal(n_pix)
    dsnu_map = DSNU_E * rng.standard_normal(n_pix)
    ph_levels = np.logspace(-0.3, 4.2, n_levels)
    mean_dn, var_t_dn, var_tot_dn = [], [], []
    for ph in ph_levels:
        a = expose(ph, n_pix, rng=rng, prnu_map=prnu_map, dsnu_map=dsnu_map)
        b = expose(ph, n_pix, rng=rng, prnu_map=prnu_map, dsnu_map=dsnu_map)
        dark = expose(0.0, n_pix, rng=rng, prnu_map=prnu_map,
                      dsnu_map=dsnu_map)
        s = float(np.mean(a) - np.mean(dark))
        mean_dn.append(s)
        var_t_dn.append(float(np.var(a - b) / 2.0))     # temporal only
        var_tot_dn.append(float(np.var(a)))             # temporal + FPN
    return np.asarray(mean_dn), np.asarray(var_t_dn), np.asarray(var_tot_dn)


def estimate_conversion_gain(mean_dn, var_t_dn):
    """Shot-noise regime: var_DN = K · S_DN (K = conversion gain in DN/e-).
    Fit the slope where shot noise dominates (well above read floor, well
    below full well)."""
    read_var = var_t_dn[np.argmin(mean_dn)]
    lo = 20.0 * read_var                      # shot >> read
    hi = 0.5 * FULL_WELL_E * CONV_GAIN_DN_E   # < half full well
    m = (mean_dn > lo) & (mean_dn < hi)
    k = float(np.sum(mean_dn[m] * (var_t_dn[m] - read_var))
              / np.sum(mean_dn[m] ** 2))
    return k


def sensor_metrics():
    """Datasheet numbers from the model parameters."""
    dr_db = 20.0 * np.log10(FULL_WELL_E / READ_NOISE_E)
    # SNR10: solve SNR(N_e) = 10 with shot + read noise
    # S/sqrt(S + r²) = 10 → S² −100S −100r² = 0
    s10_e = (100.0 + np.sqrt(100.0 ** 2 + 4.0 * 100.0 * READ_NOISE_E ** 2)) \
        / 2.0
    ph10 = s10_e / QE
    lux10 = ph10 / (PHOTONS_PER_LUX_S_UM2 * PIX_AREA_UM2 * T_EXP_S)
    sat_lux = (FULL_WELL_E / QE) / (PHOTONS_PER_LUX_S_UM2 * PIX_AREA_UM2
                                    * T_EXP_S)
    return {"dynamic_range_db": round(dr_db, 1),
            "snr10_signal_e": round(s10_e, 1),
            "snr10_lux": round(lux10, 1),
            "saturation_lux": round(sat_lux, 0)}


def hdr_merge_snr(ratio=16.0, n_points=120):
    """SNR vs scene illuminance for long, short and merged exposures.

    Long exposure t, short t/ratio. Merge: use long where unsaturated,
    else short × ratio (linearized). SNR computed analytically from shot +
    read noise; the stitch point shows the classic SNR dip.
    """
    lux = np.logspace(-1, 4.4, n_points)
    ph_long = lux * PHOTONS_PER_LUX_S_UM2 * PIX_AREA_UM2 * T_EXP_S
    e_long = ph_long * QE
    e_short = e_long / ratio

    def snr(e):
        e = np.minimum(e, FULL_WELL_E)
        return e / np.sqrt(e + READ_NOISE_E ** 2)

    snr_long = np.where(e_long < FULL_WELL_E, snr(e_long), 0.0)
    snr_short = np.where(e_short < FULL_WELL_E, snr(e_short), 0.0)
    merged = np.where(e_long < 0.95 * FULL_WELL_E, snr_long, snr_short)
    # dynamic range of the merged system
    dr_single_db = 20.0 * np.log10(FULL_WELL_E / READ_NOISE_E)
    dr_hdr_db = dr_single_db + 20.0 * np.log10(ratio)
    return lux, snr_long, snr_short, merged, dr_single_db, dr_hdr_db


# ── figures ─────────────────────────────────────────────────────────────────
def _plot_ptc(path, mean_dn, var_t, var_tot, k_est):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    ax = axes[0]
    m = mean_dn > 0
    ax.loglog(mean_dn[m], np.sqrt(var_tot[m]), "o", ms=4, color="#c86",
              label="total noise (with FPN)")
    ax.loglog(mean_dn[m], np.sqrt(var_t[m]), "o", ms=4, color="#26a",
              label="temporal noise (frame diff)")
    s = np.logspace(0, np.log10(mean_dn.max()), 50)
    ax.loglog(s, np.sqrt(k_est * s), "--", color="k", lw=1,
              label=f"shot fit: K = {k_est:.3f} DN/e-")
    ax.axhline(np.sqrt(var_t[np.argmin(mean_dn)]), color="0.5", ls=":",
               label="read floor")
    ax.axvline(FULL_WELL_E * CONV_GAIN_DN_E, color="0.5", ls="--", lw=1)
    ax.text(FULL_WELL_E * CONV_GAIN_DN_E * 0.9, 0.3, "full well",
            rotation=90, fontsize=8, ha="right")
    ax.set_xlabel("mean signal [DN]")
    ax.set_ylabel("noise [DN rms]")
    ax.set_title("Photon transfer curve")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    # exclude the clipped full-well points — variance collapse there makes
    # the naive SNR shoot up artificially
    mm = m & (mean_dn < 0.85 * FULL_WELL_E * CONV_GAIN_DN_E)
    e_sig = mean_dn[mm] / CONV_GAIN_DN_E
    snr = mean_dn[mm] / np.sqrt(var_t[mm])
    ax.loglog(e_sig, snr, "o-", ms=4, color="#26a", label="measured SNR")
    ax.axhline(10, color="0.5", ls=":", label="SNR10")
    ax.set_xlabel("signal [e-]")
    ax.set_ylabel("SNR")
    ax.set_title("SNR vs signal")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_hdr(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lux, s_l, s_s, s_m, dr1, dr2 = hdr_merge_snr()
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.semilogx(lux, s_l, color="#26a", lw=1.5, label="long exposure only")
    ax.semilogx(lux, s_s, color="#c86", lw=1.5, label="short (1/16) only")
    ax.semilogx(lux, s_m, color="k", lw=2.2, alpha=0.75,
                label="HDR merge")
    stitch = lux[np.argmax(np.diff(s_m) < -3)] if np.any(np.diff(s_m) < -3) \
        else None
    if stitch:
        ax.axvline(stitch, color="0.6", ls=":")
        ax.text(stitch * 1.1, 30, "stitch point\n(SNR dip)", fontsize=9)
    ax.axhline(10, color="0.5", ls=":", lw=1)
    ax.set_xlabel("scene illuminance [lux]")
    ax.set_ylabel("SNR")
    ax.set_title(f"Split-exposure HDR: dynamic range "
                 f"{dr1:.0f} dB -> {dr2:.0f} dB")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    mean_dn, var_t, var_tot = photon_transfer_curve()
    k_est = estimate_conversion_gain(mean_dn, var_t)
    met = sensor_metrics()
    _, _, _, _, dr1, dr2 = hdr_merge_snr()
    summary = {
        "pixel": {"full_well_e": FULL_WELL_E, "read_noise_e": READ_NOISE_E,
                  "qe": QE, "prnu_pct": PRNU * 100, "adc_bits": ADC_BITS},
        "conversion_gain_true_dn_per_e": CONV_GAIN_DN_E,
        "conversion_gain_ptc_estimate": round(k_est, 4),
        "conversion_gain_error_pct": round(
            100.0 * abs(k_est - CONV_GAIN_DN_E) / CONV_GAIN_DN_E, 2),
        **met,
        "hdr": {"exposure_ratio": 16, "dr_single_db": round(dr1, 1),
                "dr_hdr_db": round(dr2, 1)},
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)
    for d in (FIGDIR, ASSETS):
        _plot_ptc(os.path.join(d, "sensor_ptc.png"), mean_dn, var_t, var_tot,
                  k_est)
        _plot_hdr(os.path.join(d, "sensor_hdr.png"))
    print("figures ->", FIGDIR, "and", ASSETS)


if __name__ == "__main__":
    main()
