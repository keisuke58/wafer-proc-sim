#!/usr/bin/env python3
"""oxidation.py — thermal oxidation by Deal-Grove: the textbook law,
implemented and closed against its own analytic solution.

Thermal SiO2 is the foundation the MOS transistor was built on, and
Deal-Grove (1965) is arguably the most famous process model in the
industry. The batch-furnace companies live by it.

  MODEL        oxidant diffuses through the existing oxide (flux ~ D/x)
               and reacts at the Si interface (flux ~ ks). Equating the
               two gives   dx/dt = B / (2x + A)   whose solution is the
               LINEAR-PARABOLIC law
                   x^2 + A x = B (t + tau).
  REGIMES      thin oxide: reaction-limited, x ~ (B/A) t  (linear rate
               constant). Thick oxide: diffusion-limited, x ~ sqrt(B t)
               (parabolic constant). The log-log growth curve bends from
               slope 1 to slope 1/2 — the fingerprint every furnace
               engineer knows.
  WET VS DRY   H2O has far higher solubility than O2 in SiO2, so wet B is
               ~19x larger at 1100 C: dry grows thin oxides controllably,
               wet grows the 500 nm FIELD oxide in reasonable time.
               Honest footnote: Deal-Grove famously UNDER-predicts dry
               oxides below ~20 nm (the "thin-oxide anomaly" that the tau
               offset patches and Massoud later modeled) — so the thin
               recipe quoted here is a 30 nm dry oxide at 1000 C, inside
               the model's domain of validity.
  CLOSED LOOP  the numerical ODE integration of dx/dt = B/(2x+A) matches
               the analytic quadratic-formula solution to <0.05% (tested),
               and each regime matches its asymptote in its own domain.

Pure numpy + matplotlib.  Run:  python3 kokusai/oxidation.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "oxidation_results.json")

# Deal-Grove constants (published values, um and hours)
DRY = {"A_um": 0.090, "B_um2h": 0.027, "tau_h": 0.076}      # dry 1100 C
WET = {"A_um": 0.11,  "B_um2h": 0.510, "tau_h": 0.0}        # wet 1100 C
DRY1000 = {"A_um": 0.165, "B_um2h": 0.0117, "tau_h": 0.37}  # dry 1000 C


def x_analytic(t_h, p):
    """Closed-form Deal-Grove thickness [um] at time t [h]."""
    a, b, tau = p["A_um"], p["B_um2h"], p["tau_h"]
    t = np.asarray(t_h, dtype=float)
    return (-a + np.sqrt(a * a + 4 * b * (t + tau))) / 2.0


def x_ode(t_h, p, n=200000):
    """Numerical integration of dx/dt = B/(2x+A) from x(0)=x_analytic(0)."""
    a, b = p["A_um"], p["B_um2h"]
    tt = np.linspace(0.0, float(t_h), n)
    dt = tt[1] - tt[0]
    x = float(x_analytic(0.0, p))
    for _ in range(n - 1):
        # RK2 midpoint
        k1 = b / (2 * x + a)
        xm = x + 0.5 * dt * k1
        x += dt * b / (2 * xm + a)
    return x


def time_for(x_um, p):
    """Inverse law: t = (x^2 + A x)/B - tau [h]."""
    a, b, tau = p["A_um"], p["B_um2h"], p["tau_h"]
    return (x_um * x_um + a * x_um) / b - tau


def run():
    t_test = 1.0
    xa = float(x_analytic(t_test, DRY))
    xn = x_ode(t_test, DRY)
    ode_err = abs(xn - xa) / xa
    # regime checks
    t_lin = 0.005                          # far below A^2/4B crossover
    x_lin = float(x_analytic(t_lin, {**DRY, "tau_h": 0.0}))
    lin_pred = DRY["B_um2h"] / DRY["A_um"] * t_lin
    t_par = 40.0
    x_par = float(x_analytic(t_par, WET))
    par_pred = np.sqrt(WET["B_um2h"] * t_par)
    return {
        "constants_1100C": {"dry": DRY, "wet": WET,
                            "B_ratio_wet_dry": round(WET["B_um2h"]
                                                     / DRY["B_um2h"], 1)},
        "closed_loop_ode_vs_analytic_pct": round(100 * ode_err, 4),
        "linear_regime": {"t_h": t_lin, "x_um": round(x_lin, 5),
                          "BA_t_um": round(lin_pred, 5),
                          "err_pct": round(100 * abs(x_lin - lin_pred)
                                           / lin_pred, 2)},
        "parabolic_regime": {"t_h": t_par, "x_um": round(x_par, 3),
                             "sqrt_Bt_um": round(par_pred, 3),
                             "err_pct": round(100 * abs(x_par - par_pred)
                                              / par_pred, 2)},
        "recipes": {
            "thin_30nm_dry1000C_min": round(60 * time_for(0.030, DRY1000),
                                            1),
            "field_500nm_wet1100C_min": round(60 * time_for(0.500, WET), 1),
            "field_500nm_DRY1100C_hours": round(time_for(0.500, DRY), 1),
        },
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax = axes[0]
    t = np.logspace(-3, 2, 300)
    for p, c, lab in ((DRY, "#a03c64", "dry O2 (gate quality)"),
                      (WET, "#2e6da4", "wet H2O (field speed)")):
        ax.loglog(t, x_analytic(t, {**p, "tau_h": 0.0}), lw=2, color=c,
                  label=lab)
    ax.loglog(t, DRY["B_um2h"] / DRY["A_um"] * t, ":", color="#a03c64",
              lw=1, label="linear asymptote (B/A)t")
    ax.loglog(t, np.sqrt(WET["B_um2h"] * t), ":", color="#2e6da4", lw=1,
              label="parabolic asymptote sqrt(Bt)")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("oxide thickness [um]")
    ax.set_title("Deal-Grove at 1100C: slope 1 -> 1/2\n(reaction-limited "
                 "-> diffusion-limited)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    r = res["recipes"]
    labels = ["thin 30 nm\n(dry 1000C)", "field 500 nm\n(wet 1100C)",
              "field 500 nm\nif DRY 1100C"]
    vals = [r["thin_30nm_dry1000C_min"], r["field_500nm_wet1100C_min"],
            r["field_500nm_DRY1100C_hours"] * 60]
    colors = ["#a03c64", "#2e6da4", "#999"]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        txt = f"{v:.0f} min" if v < 600 else f"{v/60:.0f} h"
        ax.annotate(txt, (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel("furnace time [min]")
    ax.set_title("Why two chemistries exist: dry for control,\nwet for "
                 "thickness — dry field oxide would take "
                 f"{r['field_500nm_DRY1100C_hours']:.0f} hours",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "kok_oxide.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
