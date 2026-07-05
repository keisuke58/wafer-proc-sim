#!/usr/bin/env python3
"""vacuum_pump.py — dry-pump pumpdown physics: why the last decade is the hard one.

A multi-stage roots dry pump evacuating a process chamber, from the pump's
own datasheet physics up:

  SPEED CURVE S(p) — a dry pump moves a fixed swept volume, but its finite
  stage-to-stage compression ratio lets gas leak backwards. Net speed
      S(p) = S0 * (1 - p_ult0 / p)
  is flat over most of the range and collapses at the pump's intrinsic
  ultimate pressure p_ult0 (back-leakage = forward flow).

  PUMPDOWN ODE — the chamber obeys the gas balance
      V dp/dt = -S(p) * p + Q(t)
  with Q(t) the wall outgassing load. Water desorption from an unbaked
  wall follows the classic 1/t law: Q(t) = q1 * A / (t / 3600 s).

  TWO REGIMES — early on Q is negligible and the solution is the textbook
  exponential p(t) = p0 exp(-S t / V), i.e. t = (V/S) ln(p0/p) (the tests
  verify the implementation against this closed form). Then the chamber
  hangs at the OUTGASSING FLOOR p ~= Q(t)/S, and the time to reach the
  turbo-crossover pressure (1 Pa) is set by the water the walls adsorbed
  during venting. Pumping speed only enters linearly there, while the
  choice of vent gas attacks Q itself: venting with dry N2 instead of
  humid air cuts q1h ~20x. That is the design lesson behind every
  load-lock: a 2x bigger pump buys 2x; a dry-N2 vent buys ~20x, for free.

Integration uses a per-step exponential update (S, Q frozen over the step),
which is unconditionally stable near the stiff equilibrium.

Pure numpy + matplotlib.  Run:  python3 ebara/vacuum_pump.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "vacuum_pump_results.json")

P_ATM = 1.013e5          # [Pa]
V_CHAMBER = 0.20         # chamber volume [m^3]
S0 = 100.0 / 3600.0      # rated speed 100 m^3/h [m^3/s]
P_ULT0 = 0.5             # pump intrinsic ultimate [Pa] (compression limit)
P_CROSS = 1.0            # turbo-crossover / handoff pressure [Pa]
A_WALL = 3.0             # internal wall area incl. fixtures [m^2]
Q1H_WET = 1.0e-3         # q at t=1h after a humid-air vent [Pa m^3/s/m^2]
VENT_FACTOR = 20.0       # dry-N2 vent adsorbs ~20x less water


def pump_speed(p, s0=S0, p_ult0=P_ULT0):
    """Net pumping speed [m^3/s] at chamber pressure p [Pa]."""
    p = np.asarray(p, dtype=float)
    return np.maximum(s0 * (1.0 - p_ult0 / np.maximum(p, 1e-12)), 0.0)


def outgassing(t_s, dry_vent=False, q1h=Q1H_WET, area=A_WALL):
    """Wall outgassing load Q(t) [Pa m^3/s]; 1/t water-desorption law."""
    q = q1h / (VENT_FACTOR if dry_vent else 1.0)
    return q * area / np.maximum(t_s / 3600.0, 1e-3)


def pumpdown(t_end_s=24 * 3600.0, p_start=P_ATM, s0=S0, p_ult0=P_ULT0,
             dry_vent=False, with_outgassing=True, n_per_decade=120):
    """Integrate V dp/dt = -S(p) p + Q(t) on a log-time grid.

    The throughput S(p)*p = s0*(p - p_ult0) is EXACTLY linear in p, so with
    Q frozen over a step the update is the exact exponential relaxation to
    p_eq = p_ult0 + Q/s0 — unconditionally stable however stiff the
    equilibrium is. Returns (t [s], p [Pa])."""
    n = int(n_per_decade * np.log10(t_end_s / 0.01))
    t = np.logspace(-2, np.log10(t_end_s), n)
    p = np.empty(n)
    p[0] = p_start
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        q = float(outgassing(t[i - 1], dry_vent)) if with_outgassing \
            else 0.0
        # V dp/dt = -s0 (p - p_ult0) + q  ->  relax exactly towards p_eq
        p_eq = p_ult0 + q / s0
        p[i] = p_eq + (p[i - 1] - p_eq) * np.exp(-s0 * dt / V_CHAMBER)
    return t, p


def time_to(t, p, p_target):
    """First crossing time [s] of p_target (log interpolation)."""
    idx = np.where(p <= p_target)[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    if i == 0:
        return float(t[0])
    f = (np.log(p[i - 1]) - np.log(p_target)) / (np.log(p[i - 1])
                                                 - np.log(p[i]))
    return float(t[i - 1] + f * (t[i] - t[i - 1]))


def ultimate_pressure(s0=S0, p_ult0=P_ULT0, t_s=24 * 3600.0,
                      dry_vent=False):
    """Equilibrium pressure: outgassing floor Q/S on top of the pump limit."""
    return p_ult0 + float(outgassing(t_s, dry_vent)) / s0


def run():
    t, p_wet = pumpdown()
    _, p_dry = pumpdown(dry_vent=True)
    _, p_2x = pumpdown(s0=2 * S0)
    _, p_vol = pumpdown(with_outgassing=False)

    t_rough = time_to(t, p_vol, 10.0)
    t_rough_ana = V_CHAMBER / S0 * np.log(P_ATM / 10.0)
    return {
        "pump": {"S0_m3h": round(S0 * 3600), "V_m3": V_CHAMBER,
                 "p_ult0_Pa": P_ULT0},
        "roughing": {
            "t_to_10Pa_s": round(t_rough, 2),
            "t_analytic_VS_ln_s": round(t_rough_ana, 2),
            "err_pct": round(100 * abs(t_rough - t_rough_ana) / t_rough_ana,
                             2)},
        "t_to_crossover_1Pa_s": {
            "volume_gas_only": round(time_to(t, p_vol, P_CROSS), 1),
            "humid_air_vent": round(time_to(t, p_wet, P_CROSS), 1),
            "pump_x2": round(time_to(t, p_2x, P_CROSS), 1),
            "dry_N2_vent": round(time_to(t, p_dry, P_CROSS), 1)},
        "ultimate_24h_Pa": round(ultimate_pressure(), 4),
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    pp = np.logspace(np.log10(0.4), 5.2, 400)
    ax.loglog(pp, pump_speed(pp) * 3600, color="#1f618d", lw=2)
    ax.axvline(P_ULT0, color="#c44", ls="--", lw=1)
    ax.annotate("compression-ratio limit:\nback-leakage = forward flow",
                (P_ULT0 * 1.3, 40), fontsize=9, color="#c44")
    ax.set_xlabel("inlet pressure [Pa]")
    ax.set_ylabel("net speed S(p) [m3/h]")
    ax.set_title("Dry pump speed curve: flat until the\ncompression "
                 "ratio runs out", fontsize=10)
    ax.set_ylim(1, 1000)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    t, p_wet = pumpdown()
    _, p_dry = pumpdown(dry_vent=True)
    _, p_2x = pumpdown(s0=2 * S0)
    _, p_vol = pumpdown(with_outgassing=False)
    ax.loglog(t / 60, p_vol, color="0.6", lw=1.2, ls="--",
              label="volume gas only (V/S exponential)")
    ax.loglog(t / 60, p_wet, color="#1f618d", lw=2,
              label="humid-air vent (wet walls)")
    ax.loglog(t / 60, p_2x, color="#b4740a", lw=1.6,
              label="pump x2, still wet")
    ax.loglog(t / 60, p_dry, color="#1a7a4a", lw=2, label="dry-N2 vent")
    ax.axhline(P_CROSS, color="#c44", ls=":", lw=1)
    u = res["t_to_crossover_1Pa_s"]
    ax.annotate("turbo crossover 1 Pa", (0.03, P_CROSS * 1.35), fontsize=8,
                color="#c44")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("chamber pressure [Pa]")
    ax.set_title(f"To 1 Pa: wet {u['humid_air_vent']:.0f} s / pump x2 "
                 f"{u['pump_x2']:.0f} s / dry-N2 vent "
                 f"{u['dry_N2_vent']:.0f} s\n(the free fix beats the "
                 f"expensive one)", fontsize=10)
    ax.set_xlim(0.02, 60)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "eb_pump.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
