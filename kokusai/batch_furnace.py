#!/usr/bin/env python3
"""batch_furnace.py — 150 wafers in one tube: reactant depletion down the
boat, and the temperature ramp that cancels it.

A vertical batch furnace (the KOKUSAI franchise) processes ~150 wafers at
once — but the gas that feeds wafer #150 has already flowed past #1..#149.
LPCVD in a batch tube is therefore a built-in uniformity problem, and the
classic fix is beautiful: tilt the TEMPERATURE instead of fighting the
flow.

  DEPLETION    plug flow past a stack of reacting wafers, first-order
               surface reaction: the reactant concentration decays as
                   dC/dz = -(k(T) a / Q) C  ->  C(z) = C0 exp(-k a z / Q)
               so the deposition rate (~ k C) falls exponentially down the
               boat. Closed loop: total deposited mass equals inlet minus
               outlet flux (mass balance to <1e-9), and k -> 0 recovers
               perfect uniformity.

  COMPENSATION deposition rate ~ k(T(z)) C(z) with Arrhenius k. Solving
               rate(z) = rate(0) for T(z) gives the analytic ramp
                   T(z) = Ea / (Ea/T0 - (kB) ln(C(z)/C0) ... )
               computed self-consistently here (the ramp changes the
               depletion which changes the ramp; fixed-point iterated).
               A few degrees of tilt flatten a double-digit thickness
               slope — which is why every batch recipe has zone setpoints
               a few C apart, not one temperature.

  ECONOMICS    the batch trade: throughput of 150 wafers/run vs the
               single-wafer tools elsewhere in this repo — uniformity is
               the price, zone control is how you pay it.

Pure numpy + matplotlib.  Run:  python3 kokusai/batch_furnace.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "batch_furnace_results.json")

N_WAFERS = 150
T0_C = 700.0             # base temperature [C] (poly-Si LPCVD class)
EA_EV = 1.7              # surface-reaction activation energy [eV]
KB = 8.617e-5
DEPLETION = 0.35         # k a L / Q at T0: total depletion strength


def k_arrh(t_c):
    """Arrhenius rate constant normalized to 1 at T0."""
    t0k, tk = T0_C + 273.15, np.asarray(t_c, dtype=float) + 273.15
    return np.exp(-EA_EV / KB * (1.0 / tk - 1.0 / t0k))


def profile(temp_c=None, n=N_WAFERS, depletion=DEPLETION):
    """March down the boat: local rate r_i ~ k(T_i) C_i, concentration
    update C_{i+1} = C_i (1 - depletion/n * k(T_i)). Returns (rate, C)."""
    t = np.full(n, T0_C) if temp_c is None else np.asarray(temp_c)
    c = 1.0
    rate = np.empty(n)
    cs = np.empty(n)
    for i in range(n):
        ki = float(k_arrh(t[i]))
        cs[i] = c
        rate[i] = ki * c
        c = c * (1.0 - depletion / n * ki)
    return rate, cs


def uniformity(rate):
    """Range / (2 * mean) in % — the fab convention."""
    return float(100 * (rate.max() - rate.min()) / (2 * rate.mean()))


def solve_ramp(n=N_WAFERS, depletion=DEPLETION, iters=40):
    """Temperature profile T(z) making the local rate uniform, found by
    fixed-point iteration (ramp -> depletion -> ramp)."""
    t = np.full(n, T0_C)
    for _ in range(iters):
        _, cs = profile(t, n, depletion)
        # want k(T_i) * C_i = k(T0) * C_0 = 1  ->  k = 1/C_i
        t0k = T0_C + 273.15
        tk = 1.0 / (1.0 / t0k + KB / EA_EV * np.log(cs))
        t = tk - 273.15
    return t


def run():
    rate0, cs0 = profile()
    # mass balance: sum of deposited = inlet - outlet (units of C0*Q)
    dep_sum = float(np.sum(rate0) * DEPLETION / N_WAFERS)
    inout = float(1.0 - cs0[-1] * (1.0 - DEPLETION / N_WAFERS
                                   * float(k_arrh(T0_C))))
    t_ramp = solve_ramp()
    rate1, _ = profile(t_ramp)
    return {
        "config": {"n_wafers": N_WAFERS, "T0_C": T0_C, "Ea_eV": EA_EV,
                   "depletion_kaLQ": DEPLETION},
        "mass_balance_err": round(abs(dep_sum - inout), 12),
        "flat_temperature": {
            "rate_first_last": [round(float(rate0[0]), 4),
                                round(float(rate0[-1]), 4)],
            "uniformity_pct": round(uniformity(rate0), 2)},
        "with_ramp": {
            "ramp_C": round(float(t_ramp[-1] - t_ramp[0]), 2),
            "uniformity_pct": round(uniformity(rate1), 4)},
        "zero_depletion_uniformity_pct": round(
            uniformity(profile(depletion=1e-12)[0]), 6),
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.arange(N_WAFERS)
    rate0, cs0 = profile()
    t_ramp = solve_ramp()
    rate1, _ = profile(t_ramp)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax = axes[0]
    ax.plot(z, rate0 / rate0[0] * 100, lw=2, color="#c0392b",
            label="flat temperature")
    ax.plot(z, rate1 / rate1[0] * 100, lw=2, color="#1a7a4a",
            label="with temperature ramp")
    ax.plot(z, cs0 * 100, ":", lw=1.2, color="0.5",
            label="reactant concentration (flat T)")
    u0 = res["flat_temperature"]["uniformity_pct"]
    u1 = res["with_ramp"]["uniformity_pct"]
    ax.set_xlabel("wafer position along the boat (gas inlet -> outlet)")
    ax.set_ylabel("deposition rate [% of wafer #1]")
    ax.set_title(f"Depletion down the boat: +/-{u0:.1f}% -> "
                 f"+/-{u1:.2f}% with the ramp", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(z, t_ramp, lw=2, color="#b4740a")
    ax.set_xlabel("wafer position along the boat")
    ax.set_ylabel("zone temperature [C]")
    r = res["with_ramp"]["ramp_C"]
    ax.set_title(f"The fix is {r:.1f} C of tilt, not more gas:\nArrhenius "
                 "k(T) cancels the depletion exactly", fontsize=10)
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
    _plot(os.path.join(ASSETS, "kok_furnace.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
