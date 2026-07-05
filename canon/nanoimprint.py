#!/usr/bin/env python3
"""nanoimprint.py — nanoimprint lithography (NIL): the residual layer is a
fluid-mechanics problem and the drop pattern is the throughput knob.

NIL replaces projection optics with direct contact: inkjet-dispensed resist
droplets spread under a patterned template, features fill by capillarity,
UV cures, template separates. No diffraction limit — the economics live in
the FLUID DYNAMICS:

  SQUEEZE FLOW  each drop of local radius R flattens under the capillary
  pressure ΔP = 2γcosθ/h (thin-film Laplace pressure). Stefan squeeze flow
  gives  dh/dt = -4γcosθ · h² / (3ηR²),  which integrates in closed form to
      1/h(t) = 1/h0 + 4γcosθ/(3ηR²) · t
  (the ODE integration is verified against this analytic solution in
  tests). Time to a target residual layer thickness (RLT) therefore scales
  as η R² / γ: HALVING the drop pitch (4× the drop count, R²∝area/count)
  cuts the spread time 4× — the inkjet pattern, not the press, sets
  throughput.

  NON-FILL DEFECTS  features fill on a fast capillary time-constant, but
  trapped gas dissolves on a slow one (τ_gas): non-fill defect density
  decays as D(t) = D0 · exp(-t_spread/τ_gas). A condensable/He ambient
  shrinks τ_gas ~5× — chemistry, again cheaper than hardware.

  THROUGHPUT    wafers/hour vs the defect spec: fields/wafer × (spread +
  expose + overhead). The Pareto plot shows the two levers (drop pitch,
  ambient) moving the curve in different directions — and why NIL
  engineering is dispense-pattern engineering.

Pure numpy + matplotlib.  Run:  python3 canon/nanoimprint.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "nanoimprint_results.json")

GAMMA = 0.030            # resist surface tension [N/m]
COS_TH = 0.3             # effective wetting (release-coated template)
ETA = 0.025              # resist viscosity [Pa s] (jettable with heating)
H0_NM = 250.0            # film height when drops merge [nm]
RLT_NM = 10.0            # target residual layer thickness [nm]
FIELD_MM = 26.0          # field size [mm] (26 x 33 standard)
FIELD_MM2 = 26.0 * 33.0
FIELDS_PER_WAFER = 84
T_EXPOSE_S = 0.4         # UV cure
T_OVERHEAD_S = 0.6       # align + separate + step
D0_PER_CM2 = 50.0        # non-fill density at t=0 [1/cm^2]
TAU_GAS_AIR_S = 0.55     # trapped-gas dissolution constant, air
TAU_GAS_HE_S = 0.11      # ... with He/condensable ambient (~5x faster)


def drop_radius_um(pitch_um):
    """Local squeeze radius: each drop feeds a pitch² cell; R = cell radius
    equivalent (pitch/sqrt(pi))."""
    return pitch_um / np.sqrt(np.pi)


def spread_ode(pitch_um, t_end_s=2.0, n=4000):
    """Integrate dh/dt = -4γcosθ h²/(3ηR²) numerically (RK2).
    Returns (t [s], h [nm])."""
    R = drop_radius_um(pitch_um) * 1e-6
    k = 4.0 * GAMMA * COS_TH / (3.0 * ETA * R ** 2)      # [1/(m s)]
    t = np.linspace(0.0, t_end_s, n)
    h = np.empty(n)
    h[0] = H0_NM * 1e-9
    dt = t[1] - t[0]
    for i in range(1, n):
        h1 = h[i - 1]
        k1 = -k * h1 ** 2
        h2 = h1 + 0.5 * dt * k1
        h[i] = h1 + dt * (-k * h2 ** 2)
    return t, h * 1e9


def spread_analytic(pitch_um, t_s):
    """Closed form: 1/h = 1/h0 + 4γcosθ/(3ηR²) t."""
    R = drop_radius_um(pitch_um) * 1e-6
    k = 4.0 * GAMMA * COS_TH / (3.0 * ETA * R ** 2)
    return 1e9 / (1.0 / (H0_NM * 1e-9) + k * np.asarray(t_s))


def t_spread(pitch_um, rlt_nm=RLT_NM):
    """Time [s] to reach the target RLT (closed form inverted)."""
    R = drop_radius_um(pitch_um) * 1e-6
    k = 4.0 * GAMMA * COS_TH / (3.0 * ETA * R ** 2)
    return float((1.0 / (rlt_nm * 1e-9) - 1.0 / (H0_NM * 1e-9)) / k)


def defect_density(t_spread_s, tau_gas_s=TAU_GAS_AIR_S, d0=D0_PER_CM2):
    return float(d0 * np.exp(-t_spread_s / tau_gas_s))


def throughput_wph(t_spread_s):
    t_field = t_spread_s + T_EXPOSE_S + T_OVERHEAD_S
    return float(3600.0 / (FIELDS_PER_WAFER * t_field))


def pareto(pitch_um, tau_gas_s):
    """Sweep spread time from t(RLT) down to a 1e-3/cm² defect floor:
    (defects/cm², wafers/hour) curve."""
    t_min = t_spread(pitch_um)
    t_max = max(1.2 * t_min, tau_gas_s * np.log(D0_PER_CM2 / 1e-3))
    ts = np.linspace(t_min, t_max, 60)
    d = np.array([defect_density(t, tau_gas_s) for t in ts])
    w = np.array([throughput_wph(t) for t in ts])
    return ts, d, w


def wph_at_spec(pitch_um, tau_gas_s, spec_per_cm2=0.1):
    """Throughput at the defect spec (spread the minimum time that meets
    both the RLT and the defect spec)."""
    t_need = tau_gas_s * np.log(D0_PER_CM2 / spec_per_cm2)
    t = max(t_spread(pitch_um), t_need)
    return throughput_wph(t), t


def run():
    t, h = spread_ode(200.0, t_end_s=4.0, n=16000)
    ana = spread_analytic(200.0, t)
    ode_err = float(np.max(np.abs(h - ana) / ana))
    w_air, t_air = wph_at_spec(200.0, TAU_GAS_AIR_S)
    w_he, t_he = wph_at_spec(200.0, TAU_GAS_HE_S)
    w_he_fine, t_fine = wph_at_spec(100.0, TAU_GAS_HE_S)
    return {
        "fluid": {"eta_Pas": ETA, "gamma_Nm": GAMMA, "h0_nm": H0_NM,
                  "rlt_target_nm": RLT_NM},
        "ode_vs_analytic_max_err": round(ode_err, 6),
        "t_spread_s": {"pitch_200um": round(t_spread(200.0), 3),
                       "pitch_100um": round(t_spread(100.0), 3),
                       "ratio": round(t_spread(200.0) / t_spread(100.0), 2)},
        "wph_at_0.1_per_cm2": {
            "air_pitch200": round(w_air, 1),
            "He_pitch200": round(w_he, 1),
            "He_pitch100": round(w_he_fine, 1)},
        "spread_used_s": {"air_pitch200": round(t_air, 2),
                          "He_pitch200": round(t_he, 2),
                          "He_pitch100": round(t_fine, 2)},
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    for pitch, c in ((200.0, "#c00000"), (100.0, "#1f618d")):
        t, h = spread_ode(pitch, t_end_s=3.0, n=12000)
        ax.semilogy(t * 1e3, h, lw=2, color=c,
                    label=f"drop pitch {pitch:.0f} um (ODE)")
        ax.semilogy(t * 1e3, spread_analytic(pitch, t), "--", lw=1,
                    color="k", alpha=0.55)
    ax.axhline(RLT_NM, color="0.5", ls=":", lw=1)
    ax.annotate("target RLT 10 nm", (120, RLT_NM * 1.15), fontsize=8,
                color="0.4")
    ax.set_xlabel("spread time [ms]")
    ax.set_ylabel("film thickness h(t) [nm]")
    ax.set_title("Capillary squeeze flow: 1/h = 1/h0 + kt (dashed = "
                 "analytic)\nhalf the drop pitch -> 4x faster", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for pitch, tau, c, lab in (
            (200.0, TAU_GAS_AIR_S, "#888", "pitch 200 um, air"),
            (200.0, TAU_GAS_HE_S, "#b4740a", "pitch 200 um, He ambient"),
            (100.0, TAU_GAS_HE_S, "#c00000", "pitch 100 um, He ambient")):
        _, d, w = pareto(pitch, tau)
        ax.semilogx(d, w, "-o", ms=2.5, lw=1.6, color=c, label=lab)
    ax.axvline(0.1, color="0.5", ls=":", lw=1)
    ax.set_xlim(3e-12, 1.5)
    ax.annotate("defect spec 0.1/cm2", (0.045, 4.5), fontsize=8,
                color="0.4", rotation=90)
    ax.annotate("RLT-limited: defects << spec\nbut capped at 12.1 wph",
                (2e-11, 13.2), fontsize=8, color="#b4740a")
    ax.annotate("gas-limited (air)", (1e-3, 6.6), fontsize=8, color="#666")
    ax.set_xlabel("non-fill defects [1/cm2]")
    ax.set_ylabel("throughput [wafers/hour]")
    w = res["wph_at_0.1_per_cm2"]
    ax.set_title(f"Defect-throughput Pareto: air {w['air_pitch200']} -> He "
                 f"{w['He_pitch200']} -> +fine drops {w['He_pitch100']} wph\n"
                 f"(dispense pattern + ambient, not the press)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "canon_nil.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
