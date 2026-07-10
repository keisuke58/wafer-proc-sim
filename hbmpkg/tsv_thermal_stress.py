#!/usr/bin/env python3
"""tsv_thermal_stress.py — a copper via wants to be a different size than
the silicon around it. This is the "TSV keep-out zone" problem.

A through-silicon via (TSV) is copper (alpha ~= 17 ppm/K) drilled into
silicon (alpha ~= 2.6 ppm/K) and bonded near the reflow/anneal temperature.
Cooling to room temperature, the copper wants to shrink far more than the
silicon around it — and cannot, because it is bonded. The result is a
thermal stress field around every via that (a) can crack the silicon in a
mobility-degrading "keep-out zone" that IC designers must leave transistor-
free, and (b) puts the copper into compression that partially recovers on
reheating ("Cu pumping" / via protrusion), a known HBM base-die reliability
failure mode.

Physics implemented from scratch — a composite-cylinder (concentric-
cylinder assemblage) axisymmetric thermoelastic solve:

  Within EACH homogeneous, isotropic material the axisymmetric plane-strain
  Navier equation with no r-dependent source is the Euler-Cauchy ODE
      u'' + u'/r - u/r^2 = 0   =>   u(r) = C1 r + C2/r  (EXACT, not a
  numerical approximation — this ODE has this closed form identically).
  Regularity at r=0 forces C2=0 in the copper core; three linear conditions
  (displacement continuity, radial-stress continuity at the core/shell
  interface, and a traction-free outer unit-cell boundary) close the system
  for the three remaining unknowns, solved exactly via a 3x3 linear system
  built from Duhamel-Neumann thermoelastic stress-strain relations.

  VALIDATION — an exact identity, not an approximation: identical
  core/shell materials must give an EXACTLY zero stress field everywhere
  (free thermal expansion, no mismatch to react against). (A "shrink the
  core to zero" check was tried and deliberately dropped: for a dilute
  inclusion, a/b is already small, and the LOCAL interfacial stress
  converges to a finite dilute-limit value rather than vanishing — shrinking
  the core further does not drive it to zero, which is correct micro-
  mechanics, not a bug. The pitch sweep below shows that convergent
  large-b/a limit directly.)

  The outer "unit cell" radius comes from the TSV pitch (area-matched
  circular cell), the standard way to fold an array's areal density into a
  single-via axisymmetric problem.

Pure numpy + matplotlib.  Run:  python3 packaging/tsv_thermal_stress.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "tsv_thermal_stress_results.json")

# ── material properties: (E [Pa], nu, alpha [1/K]) ──────────────────────────
CU = dict(E=110e9, nu=0.34, alpha=17.0e-6)
SI = dict(E=130e9, nu=0.28, alpha=2.6e-6)

DELTA_T_K = -225.0        # 250C bond/anneal -> 25C room


def _lame(mat):
    E, nu = mat["E"], mat["nu"]
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    beta = (3 * lam + 2 * mu) * mat["alpha"]     # = E*alpha/(1-2nu)
    return lam, mu, beta


def solve_via(a_m, b_m, core=CU, shell=SI, delta_t=DELTA_T_K):
    """Exact composite-cylinder thermoelastic solve. Returns dict with
    P_core, P_shell, Q_shell (sigma_rr = P - Q/r^2, sigma_tt = P + Q/r^2;
    core has Q=0 so its stress state is uniform)."""
    lam_c, mu_c, beta_c = _lame(core)
    lam_s, mu_s, beta_s = _lame(shell)

    # unknowns x = [C1_core, C1_shell, Q_shell]
    # row1: displacement continuity at r=a
    # row2: radial-stress continuity at r=a
    # row3: traction-free outer boundary at r=b
    M = np.array([
        [a_m,                     -a_m,                    -1.0 / (2 * mu_s * a_m)],
        [2 * (lam_c + mu_c),      -2 * (lam_s + mu_s),      1.0 / a_m**2],
        [0.0,                      2 * (lam_s + mu_s),     -1.0 / b_m**2],
    ])
    rhs = np.array([0.0, (beta_c - beta_s) * delta_t, beta_s * delta_t])
    C1_core, C1_shell, Q_shell = np.linalg.solve(M, rhs)

    P_core = 2 * (lam_c + mu_c) * C1_core - beta_c * delta_t
    P_shell = 2 * (lam_s + mu_s) * C1_shell - beta_s * delta_t
    return {"P_core": float(P_core), "P_shell": float(P_shell),
            "Q_shell": float(Q_shell), "a_m": a_m, "b_m": b_m}


def stresses(sol, r):
    """sigma_rr, sigma_tt at radius r (array-safe)."""
    r = np.asarray(r, dtype=float)
    a, b = sol["a_m"], sol["b_m"]
    P_core, P_shell, Q = sol["P_core"], sol["P_shell"], sol["Q_shell"]
    srr = np.where(r <= a, P_core, P_shell - Q / r**2)
    stt = np.where(r <= a, P_core, P_shell + Q / r**2)
    return srr, stt


def unit_cell_radius(pitch_m):
    """Area-matched circular unit cell from a square TSV pitch."""
    return pitch_m / np.sqrt(np.pi)


def run():
    a = 2.5e-6          # 5 um diameter TSV
    pitch = 40e-6        # 40 um pitch
    b = unit_cell_radius(pitch)

    sol = solve_via(a, b)
    interface_hoop_si = float(sol["P_shell"] + sol["Q_shell"] / a**2)
    core_stress = sol["P_core"]

    # ── validation 1: identical materials -> exactly zero stress ──
    sol_same = solve_via(a, b, core=SI, shell=SI)
    same_max_abs = max(abs(sol_same["P_core"]), abs(sol_same["P_shell"]),
                       abs(sol_same["Q_shell"]) / a**2)

    # ── pitch sweep: tighter pitch (denser TSV array) -> higher stress,
    #    converging smoothly to a stable dilute-limit value as pitch grows ──
    pitches_um = [20, 30, 40, 60, 80, 200, 500]
    pitch_rows = {}
    for p_um in pitches_um:
        b_p = unit_cell_radius(p_um * 1e-6)
        s = solve_via(a, b_p)
        hoop = s["P_shell"] + s["Q_shell"] / a**2
        pitch_rows[f"{p_um}um"] = {"si_interface_hoop_MPa": round(hoop / 1e6, 1)}
    dilute_limit_change_pct = 100 * abs(
        pitch_rows["500um"]["si_interface_hoop_MPa"]
        - pitch_rows["200um"]["si_interface_hoop_MPa"]
    ) / abs(pitch_rows["200um"]["si_interface_hoop_MPa"])

    return {
        "config": {"via_diam_um": 2 * a * 1e6, "pitch_um": pitch * 1e6,
                   "unit_cell_radius_um": round(b * 1e6, 2),
                   "delta_T_K": DELTA_T_K},
        "nominal": {
            "cu_core_stress_MPa": round(core_stress / 1e6, 1),
            "si_interface_hoop_stress_MPa": round(interface_hoop_si / 1e6, 1),
        },
        "validation": {
            "identical_materials_max_abs_stress_Pa": same_max_abs,
            "dilute_limit_200um_vs_500um_pct_change": round(
                dilute_limit_change_pct, 3),
        },
        "pitch_sweep": pitch_rows,
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = res["config"]["via_diam_um"] / 2 * 1e-6
    b = res["config"]["unit_cell_radius_um"] * 1e-6
    sol = solve_via(a, b)
    r = np.linspace(0.05 * a, b, 400)
    srr, stt = stresses(sol, r)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax = axes[0]
    ax.plot(r * 1e6, srr / 1e6, lw=2, color="#b4740a", label=r"$\sigma_{rr}$")
    ax.plot(r * 1e6, stt / 1e6, lw=2, color="#c0392b", label=r"$\sigma_{\theta\theta}$")
    ax.axvline(a * 1e6, color="0.5", ls=":", lw=1)
    ax.annotate("Cu/Si\ninterface", (a * 1e6, ax.get_ylim()[1] * 0.7 if
               ax.get_ylim()[1] > 0 else 0), fontsize=8, color="0.4")
    ax.set_xlabel("radius [um]  (Cu core | Si shell)")
    ax.set_ylabel("stress [MPa]")
    hoop = res["nominal"]["si_interface_hoop_stress_MPa"]
    ax.set_title(f"TSV thermal stress: Si interface hoop stress "
                 f"{hoop:.0f} MPa\n(exact composite-cylinder solve, "
                 f"250C->25C cooldown)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ps = sorted(res["pitch_sweep"], key=lambda k: int(k.replace("um", "")))
    xs = [int(k.replace("um", "")) for k in ps]
    ys = [res["pitch_sweep"][k]["si_interface_hoop_MPa"] for k in ps]
    ax.plot(xs, ys, "o-", lw=2, color="#0b3d91")
    ax.set_xlabel("TSV pitch [um]")
    ax.set_ylabel("Si interface hoop stress [MPa]")
    ax.set_title("Denser TSV arrays (tighter pitch) raise the shared\n"
                 "thermal stress each via must react against", fontsize=10)
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
    _plot(os.path.join(ASSETS, "tsv_stress.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
