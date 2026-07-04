#!/usr/bin/env python3
"""servo_analysis.py — frequency-domain control design for the OIS / AF servo.

The C++ `ois` and `cam` modules simulate the actuators in the time domain. This
module is the control engineer's companion analysis: it designs and *proves* the
closed-loop servo in the frequency domain — the language a mechatronics reviewer
speaks — for the optical-image-stabilisation (OIS) shift actuator.

Three analyses:
  1. Loop shaping + stability margins — a PI-lead controller is wrapped around
     the 2nd-order actuator plant; the open-loop Bode plot reports the gain
     crossover, phase margin and gain margin.
  2. Disturbance rejection — the sensitivity function S(jw) = 1/(1+L) is applied
     to a hand-tremor motion PSD to get the residual image motion and the
     stabilisation in photographic "stops", plus the closed-loop step response
     (settling time, overshoot).
  3. Actuator-resonance notch compensation — a lightly damped mechanical
     resonance in the actuator eats the gain margin; a notch filter tuned to the
     resonance restores it, letting the loop keep its bandwidth.

Uses scipy.signal. Run:  python3 servo_analysis.py
"""
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

TWO_PI = 2 * np.pi

# ---- Actuator plant (matches the C++ ois module) ----------------------------
ACT_FREQ_HZ = 40.0        # actuator natural frequency
ACT_ZETA = 0.7            # actuator damping
FOCAL_MM = 35.0


def second_order(wn, zeta, dc=1.0):
    """DC-normalised 2nd-order plant: dc * wn^2 / (s^2 + 2 zeta wn s + wn^2)."""
    return np.array([dc * wn**2]), np.array([1.0, 2 * zeta * wn, wn**2])


def series(*tfs):
    """Multiply a list of (num, den) transfer functions."""
    num, den = np.array([1.0]), np.array([1.0])
    for n, d in tfs:
        num = np.polymul(num, n)
        den = np.polymul(den, d)
    return num, den


def pi_lead(kp, wi, wz, wp):
    """PI * lead: kp * (1 + wi/s) * (1 + s/wz)/(1 + s/wp)."""
    pi_n, pi_d = np.array([1.0, wi]), np.array([1.0, 0.0])       # (s+wi)/s
    lead_n, lead_d = np.array([1.0 / wz, 1.0]), np.array([1.0 / wp, 1.0])
    n, d = series((pi_n, pi_d), (lead_n, lead_d))
    return kp * n, d


def notch(wr, zeta_z, zeta_p):
    """Notch at wr: (s^2 + 2 zz wr s + wr^2)/(s^2 + 2 zp wr s + wr^2)."""
    return (np.array([1.0, 2 * zeta_z * wr, wr**2]),
            np.array([1.0, 2 * zeta_p * wr, wr**2]))


# Loop transport delay (gyro read + DSP compute + actuator ZOH); a real OIS runs
# on a sampled controller, so a small delay is present and sets the ultimate
# gain margin. First-order Padé keeps the loop rational for step responses.
DELAY_S = 4.0e-4


def pade1(td):
    """First-order Padé of e^{-s*td}: (1 - s td/2)/(1 + s td/2)."""
    return np.array([-td / 2, 1.0]), np.array([td / 2, 1.0])


def frf(num, den, w):
    """Complex open-loop frequency response L(jw) over rad/s grid w."""
    _, h = signal.freqs(num, den, worN=w)
    return h


def margins(w, L):
    """Phase margin [deg], gain margin [dB], and crossover freqs [Hz]."""
    mag = np.abs(L)
    ph = np.unwrap(np.angle(L))                 # rad
    ph_deg = np.degrees(ph)

    # Gain crossover: first |L| crossing 1 from above.
    pm, wgc_hz = np.nan, np.nan
    s = mag - 1.0
    idx = np.where(np.sign(s[:-1]) != np.sign(s[1:]))[0]
    if idx.size:
        i = idx[0]
        frac = s[i] / (s[i] - s[i + 1])
        wgc = w[i] + frac * (w[i + 1] - w[i])
        phc = ph_deg[i] + frac * (ph_deg[i + 1] - ph_deg[i])
        pm = 180.0 + phc
        wgc_hz = wgc / TWO_PI

    # Phase crossover: phase crossing -180 deg.
    gm_db, wpc_hz = np.inf, np.nan
    p = ph_deg + 180.0
    idx = np.where(np.sign(p[:-1]) != np.sign(p[1:]))[0]
    if idx.size:
        i = idx[0]
        frac = p[i] / (p[i] - p[i + 1])
        wpc = w[i] + frac * (w[i + 1] - w[i])
        magc = np.interp(wpc, w, mag)
        gm_db = -20 * np.log10(magc)
        wpc_hz = wpc / TWO_PI
    return pm, gm_db, wgc_hz, wpc_hz


# ---- Controller design (loop shaping) ---------------------------------------
WN = TWO_PI * ACT_FREQ_HZ
PLANT = second_order(WN, ACT_ZETA)
DELAY = pade1(DELAY_S)
CTRL = pi_lead(kp=2.6, wi=TWO_PI * 26.0, wz=TWO_PI * 45.0, wp=TWO_PI * 300.0)
LOOP = series(CTRL, PLANT, DELAY)                # open loop L = C P e^{-sTd}


def closed_loop(num_L, den_L):
    """T = L/(1+L) as a scipy TransferFunction."""
    num_T = num_L
    den_T = np.polyadd(den_L, num_L)
    return signal.TransferFunction(num_T, den_T)


def tremor_psd(f):
    """Hand-tremor image-motion PSD [µm^2/Hz] vs frequency [Hz].

    Physiological tremor concentrates in ~1-12 Hz with a dominant ~4-6 Hz peak;
    modelled as two Gaussian bands on a low-frequency floor.
    """
    band = (2.2 * np.exp(-0.5 * ((f - 4.5) / 1.6) ** 2) +
            0.8 * np.exp(-0.5 * ((f - 10.0) / 2.5) ** 2))
    floor = 0.25 / (1 + (f / 1.5) ** 2)
    return band + floor


def _plot_bode(path):
    w = TWO_PI * np.logspace(-0.3, 3.2, 4000)
    f = w / TWO_PI
    Lp = frf(*PLANT, w)
    Ll = frf(*LOOP, w)
    pm, gm, wgc, wpc = margins(w, Ll)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.2, 1]})
    a1.semilogx(f, 20 * np.log10(np.abs(Lp)), color="#888", lw=1.5,
                label="Plant P (actuator only)")
    a1.semilogx(f, 20 * np.log10(np.abs(Ll)), color="#1f77b4", lw=2.2,
                label="Open loop L = C·P (compensated)")
    a1.axhline(0, color="#bbb", lw=.8)
    if np.isfinite(wgc):
        a1.axvline(wgc, color="#2ca02c", ls=":", lw=1.2)
    a1.set_ylabel("Magnitude [dB]"); a1.set_ylim(-60, 60)
    a1.set_title(f"OIS open-loop Bode  ·  PM {pm:.0f}°, GM {gm:.0f} dB, "
                 f"crossover {wgc:.0f} Hz", fontsize=10)
    a1.legend(fontsize=8.3, loc="lower left"); a1.grid(which="both", alpha=.25)

    a2.semilogx(f, np.degrees(np.unwrap(np.angle(Lp))), color="#888", lw=1.5)
    a2.semilogx(f, np.degrees(np.unwrap(np.angle(Ll))), color="#1f77b4", lw=2.2)
    a2.axhline(-180, color="#d62728", ls="--", lw=1)
    if np.isfinite(wgc):
        a2.axvline(wgc, color="#2ca02c", ls=":", lw=1.2, label=f"crossover {wgc:.0f} Hz")
        phc = np.interp(wgc * TWO_PI, w, np.degrees(np.unwrap(np.angle(Ll))))
        a2.annotate(f"PM {pm:.0f}°", xy=(wgc, phc), xytext=(wgc * 1.4, phc + 55),
                    fontsize=9, color="#2ca02c",
                    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=.9))
    a2.set_ylabel("Phase [deg]"); a2.set_xlabel("Frequency [Hz]")
    a2.set_ylim(-300, 30); a2.legend(fontsize=8.3, loc="lower left")
    a2.grid(which="both", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return pm, gm, wgc, wpc


def _plot_rejection(path):
    # Sensitivity S = 1/(1+L) over the tremor band.
    f = np.linspace(0.3, 30, 2000)
    w = TWO_PI * f
    Ll = frf(*LOOP, w)
    S = 1.0 / (1.0 + Ll)
    Smag = np.abs(S)

    D = tremor_psd(f)                # demand PSD (OIS off), µm^2/Hz
    R = Smag**2 * D                  # residual PSD (OIS on)
    off_rms = np.sqrt(np.trapezoid(D, f))
    on_rms = np.sqrt(np.trapezoid(R, f))
    stops = np.log2(off_rms / on_rms)

    # Closed-loop step response.
    T = closed_loop(*LOOP)
    tt = np.linspace(0, 0.15, 1500)
    _, y = signal.step(T, T=tt)
    yss = y[-1]
    overshoot = max(0.0, (y.max() - yss) / yss * 100)
    tol = 0.02 * yss
    settled = np.where(np.abs(y - yss) > tol)[0]
    ts_ms = (tt[settled[-1]] * 1e3) if settled.size else 0.0

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 4.0))
    a1.fill_between(f, D, color="#888", alpha=.35, label="Tremor demand (OIS off)")
    a1.fill_between(f, R, color="#1f77b4", alpha=.7, label="Residual (OIS on)")
    a1.set_yscale("log"); a1.set_xlim(0, 20)
    a1.set_xlabel("Frequency [Hz]"); a1.set_ylabel("Motion PSD [µm²/Hz]")
    a1.set_title(f"Disturbance rejection  ·  {stops:.1f} stops "
                 f"({off_rms:.1f}→{on_rms:.2f} µm rms)", fontsize=9.5)
    a1.legend(fontsize=8.3); a1.grid(which="both", alpha=.2)

    a2.plot(tt * 1e3, y, color="#1f77b4", lw=2)
    a2.axhline(yss, color="#888", ls="--", lw=1)
    a2.axhspan(yss - tol, yss + tol, color="#2ca02c", alpha=.12)
    a2.axvline(ts_ms, color="#2ca02c", ls=":", lw=1.3,
               label=f"settle {ts_ms:.0f} ms (±2%)")
    a2.set_xlabel("Time [ms]"); a2.set_ylabel("Normalised response")
    a2.set_title(f"Closed-loop step  ·  overshoot {overshoot:.0f}%", fontsize=9.5)
    a2.legend(fontsize=8.3); a2.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return stops, off_rms, on_rms, overshoot, ts_ms


def _plot_notch(path):
    # Push the loop bandwidth up so a mechanical resonance matters.
    ctrl_hi = pi_lead(kp=2.8, wi=TWO_PI * 20.0, wz=TWO_PI * 70.0, wp=TWO_PI * 450.0)
    wr = TWO_PI * 260.0                          # actuator/flexure resonance [Hz]
    reso = second_order(wr, 0.03)                # lightly damped, DC gain 1
    L_reso = series(ctrl_hi, PLANT, reso, DELAY)
    nfilt = notch(wr, zeta_z=0.015, zeta_p=0.26)
    L_notch = series(ctrl_hi, PLANT, reso, nfilt, DELAY)

    w = TWO_PI * np.logspace(0, 3.4, 5000)
    f = w / TWO_PI
    Lr = frf(*L_reso, w)
    Ln = frf(*L_notch, w)
    pm_r, gm_r, wgc_r, _ = margins(w, Lr)
    pm_n, gm_n, wgc_n, _ = margins(w, Ln)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.2, 1]})
    a1.semilogx(f, 20 * np.log10(np.abs(Lr)), color="#d62728", lw=1.9,
                label=f"With resonance, no notch  (GM {gm_r:.0f} dB)")
    a1.semilogx(f, 20 * np.log10(np.abs(Ln)), color="#1f77b4", lw=2.2,
                label=f"Notch-compensated  (GM {gm_n:.0f} dB)")
    a1.axhline(0, color="#bbb", lw=.8)
    a1.axvline(260, color="#888", ls=":", lw=1, label="resonance 260 Hz")
    a1.set_ylabel("Magnitude [dB]"); a1.set_ylim(-70, 70)
    a1.set_title(f"Actuator resonance & notch  ·  PM {pm_r:.0f}°→{pm_n:.0f}°, "
                 f"GM {gm_r:.0f}→{gm_n:.0f} dB", fontsize=10)
    a1.legend(fontsize=8.2, loc="lower left"); a1.grid(which="both", alpha=.25)

    a2.semilogx(f, np.degrees(np.unwrap(np.angle(Lr))), color="#d62728", lw=1.9)
    a2.semilogx(f, np.degrees(np.unwrap(np.angle(Ln))), color="#1f77b4", lw=2.2)
    a2.axhline(-180, color="#444", ls="--", lw=1)
    a2.axvline(260, color="#888", ls=":", lw=1)
    a2.set_ylabel("Phase [deg]"); a2.set_xlabel("Frequency [Hz]")
    a2.grid(which="both", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return pm_r, gm_r, pm_n, gm_n, wgc_n


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    figs = os.path.join(here, "figures")
    os.makedirs(figs, exist_ok=True)

    pm, gm, wgc, wpc = _plot_bode(os.path.join(figs, "ois_bode.png"))
    stops, off_rms, on_rms, os_pct, ts_ms = _plot_rejection(
        os.path.join(figs, "ois_rejection.png"))
    pm_r, gm_r, pm_n, gm_n, wgc_n = _plot_notch(
        os.path.join(figs, "notch_bode.png"))

    out = {
        "plant_freq_hz": ACT_FREQ_HZ, "plant_zeta": ACT_ZETA,
        "phase_margin_deg": round(float(pm), 1),
        "gain_margin_dB": round(float(gm), 1),
        "crossover_hz": round(float(wgc), 1),
        "phase_crossover_hz": round(float(wpc), 1) if np.isfinite(wpc) else None,
        "rejection_stops": round(float(stops), 2),
        "tremor_off_rms_um": round(float(off_rms), 2),
        "tremor_on_rms_um": round(float(on_rms), 3),
        "step_overshoot_pct": round(float(os_pct), 1),
        "step_settle_ms": round(float(ts_ms), 1),
        "resonance_hz": 260.0,
        "gm_no_notch_dB": round(float(gm_r), 1),
        "gm_with_notch_dB": round(float(gm_n), 1),
        "pm_no_notch_deg": round(float(pm_r), 1),
        "pm_with_notch_deg": round(float(pm_n), 1),
        "notch_bandwidth_hz": round(float(wgc_n), 1),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open(os.path.join(here, "servo_analysis_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
