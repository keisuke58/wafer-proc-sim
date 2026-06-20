"""
fem/calibrate_haz_3d.py — calibrate the cheap analytical groove/HAZ model
against the 3-D thermal solver ("ground truth").

The hybrid pipeline's Stage-A HAZ comes from `analytical_groove` (a fast 2-D
per-pulse Beer–Lambert estimate). Here we run matched laser recipes through both
that model and the 3-D transient-thermal solver (same material, SiC) and fit a
linear calibration   y_3D = a · x_analytic + b   so the cheap model can be
corrected to 3-D truth. The thermal HAZ is the apples-to-apples quantity (both
models are thermal); groove depth is reported for context (different mechanisms:
fluence-ablation vs thermal-ablation).

Run:
    python fem/calibrate_haz_3d.py --n 24 --plot
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fem.dicing_thermal_3d_gpu import Sim3DConfig, simulate, get_device, _lhs
from fem.laser_groove_thermal_2d import analytical_groove


SPACE = {"power_W": (10.0, 40.0), "speed_mm_s": (80.0, 400.0),
         "beam_radius_um": (3.0, 8.0)}


def _analytic_excess_haz(power, speed, r0):
    g = analytical_groove({
        "laser_power_W": power, "scan_speed_mm_s": speed,
        "pulse_freq_kHz": 100.0, "beam_radius_um": r0,
        "absorptivity": 0.7, "n_passes": 1, "pulse_regime": "ps",
    })
    excess = max(0.0, g["HAZ_depth_um"] - g["groove_depth_um"])
    return g["groove_depth_um"], excess


def _linfit(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0, float(np.mean(y)), 0.0
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss = ((y - yhat) ** 2).sum(); tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss / tot if tot > 0 else 0.0
    return float(a), float(b), float(r2)


def calibrate(material="SiC", n=24, seed=0, device=None, dx_um=5.0):
    device = device or get_device()
    names = list(SPACE)
    U = _lhs(n, len(names), seed)
    rows = []
    for u in U:
        rec = {nm: float(SPACE[nm][0] + v * (SPACE[nm][1] - SPACE[nm][0]))
               for v, nm in zip(u, names)}
        a_groove, a_haz = _analytic_excess_haz(rec["power_W"], rec["speed_mm_s"],
                                               rec["beam_radius_um"])
        r = simulate(Sim3DConfig(material=material, process="laser",
                                 dx_um=dx_um, **rec), device=device)
        rows.append(dict(**rec, analytic_groove=a_groove, analytic_haz=a_haz,
                         sim3d_groove=r.groove_depth_um, sim3d_haz=r.haz_excess_um))
    return rows


def summarize(rows):
    ax = [r["analytic_haz"] for r in rows]; ay = [r["sim3d_haz"] for r in rows]
    gx = [r["analytic_groove"] for r in rows]; gy = [r["sim3d_groove"] for r in rows]
    a_h, b_h, r2_h = _linfit(ax, ay)
    a_g, b_g, r2_g = _linfit(gx, gy)
    return {
        "haz": dict(slope=a_h, intercept=b_h, r2=r2_h,
                    ratio=float(np.mean(np.array(ay) / np.maximum(ax, 1e-6)))),
        "groove": dict(slope=a_g, intercept=b_g, r2=r2_g),
    }


def plot(rows, cal, material, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, key, title in [(a1, "haz", "excess HAZ [µm]"),
                           (a2, "groove", "groove depth [µm]")]:
        x = [r[f"analytic_{key}"] for r in rows]
        y = [r[f"sim3d_{key}"] for r in rows]
        ax.scatter(x, y, s=28, color="#1b7f79", edgecolor="k", alpha=.8)
        c = cal[key]
        xs = np.linspace(min(x + [0]), max(x + [1e-3]), 50)
        ax.plot(xs, c["slope"] * xs + c["intercept"], "--", color="#b5374a",
                label=f"y={c['slope']:.2f}x+{c['intercept']:.2f}  R²={c['r2']:.2f}")
        lim = max(max(x + [1e-3]), max(y + [1e-3]))
        ax.plot([0, lim], [0, lim], ":", color="gray", lw=1, label="1:1")
        ax.set(xlabel=f"analytic {title}", ylabel=f"3-D sim {title}",
               title=f"({'a' if key=='haz' else 'b'}) {title}")
        ax.legend(fontsize=9, frameon=False); ax.grid(alpha=.3)
    fig.suptitle(f"Analytic ↔ 3-D thermal calibration ({material} laser grooving)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="SiC")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--dx", type=float, default=5.0, help="cell size µm (2 on GPU)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    dev = get_device()
    rows = calibrate(args.material, args.n, args.seed, dev, dx_um=args.dx)
    cal = summarize(rows)
    print(f"=== analytic→3D calibration ({args.material} laser, n={args.n}, dev={dev}) ===")
    for k in ("haz", "groove"):
        c = cal[k]
        print(f"  {k:7}  3D = {c['slope']:.3f}·analytic + {c['intercept']:.3f}   "
              f"R²={c['r2']:.3f}" + (f"   mean ratio={c['ratio']:.3f}" if "ratio" in c else ""))
    if args.plot:
        out = plot(rows, cal, args.material,
                   os.path.join(ROOT, "results", "thermal3d_haz_calibration.png"))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
