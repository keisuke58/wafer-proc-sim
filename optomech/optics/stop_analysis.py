#!/usr/bin/env python3
"""stop_analysis.py — STOP chain: thermal FEM → wavefront → MTF vs temperature.

The classic Structural-Thermal-Optical-Performance loop, built entirely from
modules already in this repo instead of three disconnected tools:

  STRUCTURAL  optomech/fem/barrel_axisym.py — axisymmetric thermo-elastic FE
              of the lens barrel, run here at multiple ΔT and for three
              barrel materials (Al 6061 / Ti-6Al-4V / Invar 36) by swapping
              the material constants. Axial growth is checked to be linear
              in ΔT (it must be — the check catches FE regressions).
  THERMAL →   axial growth defocuses the image plane; the lens elements add
  OPTICAL     their own dn/dT focus drift. A radial temperature gradient
              across the barrel (fraction G of ΔT) bends it, tilting the
              lens seat → astigmatism, and sagging it → decenter → coma
              (sensitivities from the optical design's tolerance table in
              mtf_wavefront.py).
  PERFORMANCE optomech/optics/mtf_wavefront.py — Zernike wavefront → pupil
              FFT → MTF at the 60 lp/mm spec frequency.

Output: MTF vs operating temperature for the three barrel materials and the
temperature WINDOW over which the design holds MTF ≥ 0.35 — the number a
systems engineer actually signs up to.

Run:  python3 optomech/optics/stop_analysis.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from optomech.fem import barrel_axisym as ba          # noqa: E402
from optomech.optics import mtf_wavefront as mw       # noqa: E402

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "stop_analysis_results.json")

# barrel materials: (E [Pa], CTE [1/K])
MATERIALS = {
    "Al 6061": (68.9e9, 23.6e-6),
    "Ti-6Al-4V": (113.8e9, 8.6e-6),
    "Invar 36": (141.0e9, 1.2e-6),
}
T_REF_C = 20.0            # design (athermal) temperature
GRAD_FRAC = 0.15          # radial gradient across the barrel, fraction of ΔT
DEC_PER_GRAD_C = 0.12     # µm element decenter per °C of radial gradient
TILT_PER_GRAD_C = 0.08    # arcmin lens-seat tilt per °C of radial gradient


def fe_axial_growth_um(dT_C, E, alpha, nr=4, nz=40):
    """Run the axisymmetric FE with the given material and return the mean
    axial growth of the free (lens-seat) end in µm."""
    E0, a0 = ba.E, ba.ALPHA
    try:
        ba.E, ba.ALPHA = E, alpha
        nodes, elems, nr_, nz_ = ba.mesh(nr, nz)
        u, _ = ba.solve_thermal(nodes, elems, nr_, nz_, dT=dT_C)
        tip = np.where(np.abs(nodes[:, 1] - nodes[:, 1].max()) < 1e-12)[0]
        return float(np.mean(u[2 * tip + 1]) * 1e6)
    finally:
        ba.E, ba.ALPHA = E0, a0


def fe_sensitivity_um_per_C(E, alpha, dTs=(-20.0, 10.0, 25.0, 40.0)):
    """FE growth at several ΔT + linear fit. Returns (µm/°C, r²)."""
    g = np.array([fe_axial_growth_um(dt, E, alpha) for dt in dTs])
    dts = np.asarray(dTs)
    slope = float(np.sum(dts * g) / np.sum(dts ** 2))
    ss_res = float(np.sum((g - slope * dts) ** 2))
    ss_tot = float(np.sum((g - g.mean()) ** 2))
    return slope, 1.0 - ss_res / ss_tot, dts, g


def wavefront_at(dT_from_ref, barrel_um_per_C):
    """Zernike coefficients [waves RMS] at a temperature offset from design."""
    focus_um = (barrel_um_per_C + mw.LENS_UM_PER_C) * dT_from_ref
    c_def = mw.defocus_waves(focus_um)
    grad_C = GRAD_FRAC * abs(dT_from_ref)
    c_coma = mw.COMA_PER_DECENTER * DEC_PER_GRAD_C * grad_C
    c_astig = mw.ASTIG_PER_TILT * TILT_PER_GRAD_C * grad_C
    return c_def, c_coma, c_astig


def mtf_vs_temperature(barrel_um_per_C, temps_C):
    out = []
    for t in temps_C:
        c_def, c_coma, c_astig = wavefront_at(t - T_REF_C, barrel_um_per_C)
        W = mw.zernike_wavefront(c_def, c_coma, c_astig)
        out.append(mw.mtf_at(mw.SPEC_LPMM, W))
    return np.asarray(out)


def temperature_window(temps_C, mtf):
    """Contiguous span around T_REF where MTF >= spec, via interpolation."""
    ok = mtf >= mw.MTF_SPEC
    if not ok.any():
        return 0.0, (T_REF_C, T_REF_C)
    i_ref = int(np.argmin(np.abs(temps_C - T_REF_C)))
    lo = hi = i_ref
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1

    def cross(i0, i1):
        m0, m1 = mtf[i0], mtf[i1]
        if m1 == m0:
            return temps_C[i1]
        f = (mw.MTF_SPEC - m0) / (m1 - m0)
        return temps_C[i0] + f * (temps_C[i1] - temps_C[i0])

    t_lo = temps_C[0] if lo == 0 else cross(lo, lo - 1)
    t_hi = temps_C[-1] if hi == len(ok) - 1 else cross(hi, hi + 1)
    return float(t_hi - t_lo), (float(t_lo), float(t_hi))


def run_stop():
    temps = np.linspace(-30.0, 90.0, 61)
    result = {}
    for name, (E, alpha) in MATERIALS.items():
        slope, r2, dts, growth = fe_sensitivity_um_per_C(E, alpha)
        mtf = mtf_vs_temperature(slope, temps)
        width, (t_lo, t_hi) = temperature_window(temps, mtf)
        result[name] = {
            "fe_um_per_C": round(slope, 4), "fe_linearity_r2": round(r2, 6),
            "fe_points": {str(d): round(g, 3) for d, g in zip(dts, growth)},
            "window_C": round(width, 1),
            "window": [round(t_lo, 1), round(t_hi, 1)],
            "mtf60_at_ref": round(float(
                mtf[np.argmin(np.abs(temps - T_REF_C))]), 3),
            "temps": temps.tolist(), "mtf": mtf.tolist(),
        }
    return temps, result


# ── figure ──────────────────────────────────────────────────────────────────
def _plot(path, temps, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Al 6061": "#c86", "Ti-6Al-4V": "#a7c", "Invar 36": "#26a"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))

    ax = axes[0, 0]
    for name, r in res.items():
        d = np.array([float(k) for k in r["fe_points"]])
        g = np.array(list(r["fe_points"].values()))
        o = np.argsort(d)
        ax.plot(d[o], g[o], "o", color=colors[name], ms=5)
        dd = np.linspace(-25, 45, 10)
        ax.plot(dd, r["fe_um_per_C"] * dd, "-", color=colors[name], lw=1,
                label=f"{name}: {r['fe_um_per_C']:.3f} um/C")
    ax.set_xlabel("barrel dT [C]")
    ax.set_ylabel("FE axial growth [um]")
    ax.set_title("Structural: axisymmetric FE, 3 barrel materials")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    dts = np.linspace(-50, 70, 100)
    slope_al = res["Al 6061"]["fe_um_per_C"]
    comps = np.array([wavefront_at(d, slope_al) for d in dts])
    ax.plot(dts + T_REF_C, np.abs(comps[:, 0]), color="#26a",
            label="defocus (barrel + lens dn/dT)")
    ax.plot(dts + T_REF_C, comps[:, 1], color="#c86",
            label="coma (gradient bend decenter)")
    ax.plot(dts + T_REF_C, comps[:, 2], color="#a7c",
            label="astig (gradient seat tilt)")
    ax.set_xlabel("temperature [C]")
    ax.set_ylabel("Zernike coeff [waves RMS]")
    ax.set_title("Thermal -> optical: wavefront build-up (Al barrel)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for name, r in res.items():
        ax.plot(temps, r["mtf"], color=colors[name], lw=1.8,
                label=f"{name}: window {r['window_C']:.0f} C "
                      f"[{r['window'][0]:.0f}, {r['window'][1]:.0f}]")
    ax.axhline(mw.MTF_SPEC, color="k", ls="--", lw=1,
               label=f"spec {mw.MTF_SPEC} @ {mw.SPEC_LPMM:.0f} lp/mm")
    ax.axvline(T_REF_C, color="0.6", ls=":", lw=1)
    ax.set_xlabel("operating temperature [C]")
    ax.set_ylabel(f"MTF @ {mw.SPEC_LPMM:.0f} lp/mm")
    ax.set_title("Performance: MTF vs temperature")
    ax.legend(fontsize=8, loc="lower center")

    ax = axes[1, 1]
    for t, style in ((T_REF_C, "-"), (60.0, "--"), (85.0, ":")):
        c_def, c_coma, c_astig = wavefront_at(
            t - T_REF_C, res["Al 6061"]["fe_um_per_C"])
        W = mw.zernike_wavefront(c_def, c_coma, c_astig)
        fq, m = mw.mtf_from_wavefront(W)
        ax.plot(fq, m, style, color="#c86", lw=1.5, label=f"Al @ {t:.0f} C")
    fq = np.linspace(0, mw.CUTOFF_LPMM, 80)
    ax.plot(fq, mw.diffraction_mtf(fq), "k:", lw=1, label="diffraction limit")
    ax.plot([mw.SPEC_LPMM], [mw.MTF_SPEC], "k*", ms=12)
    ax.set_xlim(0, 200)
    ax.set_xlabel("spatial frequency [lp/mm]")
    ax.set_ylabel("MTF")
    ax.set_title("Full MTF curves (Al barrel)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    temps, res = run_stop()
    summary = {name: {k: v for k, v in r.items()
                      if k not in ("temps", "mtf")}
               for name, r in res.items()}
    summary["spec"] = {"freq_lpmm": mw.SPEC_LPMM, "mtf_min": mw.MTF_SPEC,
                       "t_ref_C": T_REF_C}
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)
    for d in (FIGDIR, ASSETS):
        _plot(os.path.join(d, "stop_chain.png"), temps, res)
    print("figure ->", FIGDIR, "and", ASSETS)


if __name__ == "__main__":
    main()
