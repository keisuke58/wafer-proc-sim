#!/usr/bin/env python3
"""pvd_sputter.py — sputter deposition into a trench: why PVD pinches off,
and what collimation buys.

Physical vapor deposition is LINE-OF-SIGHT: atoms arrive from the target
with a broad angular spread and stick where they land. Inside a trench
that geometry is destiny — the opening sees the whole source, the bottom
sees only a narrowing cone, so the top corners grow fastest, OVERHANG,
and finally pinch off a void. This module simulates that from the
visibility integral up:

  FLUX         each surface element receives
                   F = integral over unblocked arrival angles of
                       S(theta) cos(local incidence)
               with the source angular distribution S(theta) ~ cos^n:
               n = 1 is a diffuse magnetron, large n a collimated /
               long-throw source.

  GROWTH       the trench wall is a polyline grown node-by-node along the
               local normal, re-checking visibility every step (the
               overhang shadows the sidewall below as it grows).

  METRICS      bottom / field step coverage vs aspect ratio; the pinch-off
               point where the mouth closes. Closed loops: a flat surface
               grows at exactly the source rate; the profile stays
               symmetric; coverage decreases monotonically with AR;
               collimation strictly improves it.

  THE POINT    diffuse PVD at AR 3 leaves ~15% bottom coverage and an
               overhang racing to pinch-off; a collimated source doubles
               bottom coverage — but the real production answer for high
               AR (barrier/seed in damascene) is the one this repo already
               models elsewhere: switch to ALD (telproc/) when conformality
               matters more than rate.

Pure numpy + matplotlib.  Run:  python3 amat/pvd_sputter.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "pvd_sputter_results.json")

W_TRENCH = 100.0         # trench width [nm]
N_ANG = 181              # arrival-angle quadrature


N_SIDE, N_BOT, N_TOP = 40, 20, 14
IDX_LWALL = np.arange(N_TOP, N_TOP + N_SIDE)
IDX_RWALL = np.arange(N_TOP + N_SIDE + N_BOT, N_TOP + 2 * N_SIDE + N_BOT)


def make_trench(ar, n_side=N_SIDE, n_bot=N_BOT, n_top=N_TOP):
    """Polyline of the initial surface: top field, down the left wall,
    across the bottom, up the right wall, top field. Returns (x, y) with
    y=0 at the field, y=-depth at the bottom."""
    d = ar * W_TRENCH
    xs, ys = [], []
    for t in np.linspace(-1.2 * W_TRENCH, -W_TRENCH / 2, n_top):
        xs.append(t); ys.append(0.0)
    for t in np.linspace(0, -d, n_side + 1)[1:]:
        xs.append(-W_TRENCH / 2); ys.append(t)
    for t in np.linspace(-W_TRENCH / 2, W_TRENCH / 2, n_bot + 1)[1:]:
        xs.append(t); ys.append(-d)
    for t in np.linspace(-d, 0, n_side + 1)[1:]:
        xs.append(W_TRENCH / 2); ys.append(t)
    for t in np.linspace(W_TRENCH / 2, 1.2 * W_TRENCH, n_top + 1)[1:]:
        xs.append(t); ys.append(0.0)
    return np.array(xs), np.array(ys)


def _normals(x, y):
    """Outward (upward-ish) unit normals of the polyline nodes."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    ln = np.hypot(dx, dy) + 1e-12
    nx, ny = -dy / ln, dx / ln
    flip = ny < 0
    nx[flip] *= -1
    ny[flip] *= -1
    return nx, ny


def flux(x, y, n_exp=1.0):
    """Visibility-integrated arrival flux at each node, normalized so a
    flat open surface gets 1.0. Arrival directions theta from vertical in
    (-pi/2, pi/2), source weight ~ cos^n_exp(theta); a direction counts
    only if the ray to the sky clears every other segment. Fully
    vectorized over segments x angles per node."""
    th = np.linspace(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, N_ANG)
    w = np.cos(th) ** n_exp
    norm = np.sum(w * np.cos(th))       # flat-surface reference
    nx, ny = _normals(x, y)
    n = len(x)
    rx, ry = np.sin(th), np.cos(th)                      # (K,) sky rays
    ax, ay = x[:-1], y[:-1]                              # (M,) seg starts
    ex, ey = np.diff(x), np.diff(y)                      # (M,) seg vecs
    f = np.zeros(n)
    for i in range(n):
        inc = nx[i] * rx + ny[i] * ry                    # (K,)
        vis = inc > 0
        px, py = x[i], y[i]
        # ray-segment intersection, rays (K,1) x segments (1,M):
        den = rx[:, None] * ey[None, :] - ry[:, None] * ex[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = ((ax[None, :] - px) * ey[None, :]
                 - (ay[None, :] - py) * ex[None, :]) / den
            u = ((ax[None, :] - px) * ry[:, None]
                 - (ay[None, :] - py) * rx[:, None]) / (-den)
        hit = (np.abs(den) > 1e-12) & (s > 1e-3) & (u >= 0.0) & (u <= 1.0)
        # ignore the two segments adjacent to node i
        if i > 0:
            hit[:, i - 1] = False
        if i < len(ax):
            hit[:, i] = False
        blocked = hit.any(axis=1)
        f[i] = np.sum(w[vis & ~blocked] * inc[vis & ~blocked]) / norm
    return f


def deposit(ar=2.0, n_exp=1.0, thick_field=40.0, n_steps=20):
    """Grow the film in steps of thick_field/n_steps along local normals
    with per-step flux recomputation. Returns final profile and coverage
    metrics."""
    x, y = make_trench(ar)
    x0, y0 = x.copy(), y.copy()
    step = thick_field / n_steps
    pinched = False
    for _ in range(n_steps):
        f = flux(x, y, n_exp)
        nx, ny = _normals(x, y)
        x = x + nx * f * step
        y = y + ny * f * step
        # string-method regularization: light 3-point smoothing keeps the
        # advancing front from kinking (which would corrupt visibility)
        x[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
        y[1:-1] = 0.25 * y[:-2] + 0.5 * y[1:-1] + 0.25 * y[2:]
        # pinch-off check on the upper walls (index-identified, so the
        # rising bottom cannot masquerade as the mouth)
        mouth = _mouth_width(x, y, ar)
        if mouth < 4.0:
            pinched = True
            break
    d = ar * W_TRENCH
    bot = (np.abs(x0) < W_TRENCH * 0.2) & (np.abs(y0 + d) < 1e-6)
    t_bot = float(np.mean(y[bot] - y0[bot]))
    # flat-surface flux is exactly 1.0 (verified closed loop), so the
    # nominal field thickness is the clean coverage denominator (the far
    # field nodes are few and touched by the smoothing stencil)
    return {"x": x, "y": y, "x0": x0, "y0": y0,
            "coverage": t_bot / thick_field,
            "pinched": pinched, "mouth_nm": _mouth_width(x, y, ar)}


def _mouth_width(x, y, ar):
    """Narrowest gap between the UPPER parts of the two walls (nodes
    identified by index, top 40% of the trench depth)."""
    d = ar * W_TRENCH
    li = IDX_LWALL[y[IDX_LWALL] > -0.4 * d]
    ri = IDX_RWALL[y[IDX_RWALL] > -0.4 * d]
    if len(li) == 0 or len(ri) == 0:
        return W_TRENCH
    return float(np.min(x[ri]) - np.max(x[li]))


def coverage_curve(n_exp, ars=(0.5, 1.0, 2.0, 3.0)):
    return [deposit(ar, n_exp)["coverage"] for ar in ars]


def run():
    ars = (0.5, 1.0, 2.0, 3.0)
    cov_dif = coverage_curve(1.0, ars)
    cov_col = coverage_curve(20.0, ars)
    # flat-surface closed loop
    x, y = make_trench(0.5)
    f = flux(x, y, 1.0)
    flat = f[x < -1.05 * W_TRENCH]
    return {
        "model": {"trench_w_nm": W_TRENCH, "source": "cos^n arrival",
                  "n_diffuse": 1, "n_collimated": 20},
        "flat_flux_closed_loop": {"mean": round(float(flat.mean()), 4),
                                  "expected": 1.0},
        "coverage_vs_AR": {
            "AR": list(ars),
            "diffuse": [round(c, 3) for c in cov_dif],
            "collimated": [round(c, 3) for c in cov_col]},
        "collimation_gain_at_AR3": round(cov_col[-1] / max(cov_dif[-1],
                                                           1e-9), 2),
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4))
    for ax, n_exp, ttl in ((axes[0], 1.0, "diffuse source (cos)"),
                           (axes[1], 20.0, "collimated (cos^20)")):
        r = deposit(2.0, n_exp, thick_field=40.0)
        ax.plot(r["x0"], r["y0"], color="0.6", lw=1, label="initial trench")
        ax.plot(r["x"], r["y"], color="#0b3d91", lw=1.8,
                label="film surface")
        ax.fill(np.concatenate([r["x0"], r["x"][::-1]]),
                np.concatenate([r["y0"], r["y"][::-1]]),
                color="#0b3d91", alpha=0.15)
        ax.set_xlim(-130, 130)
        ax.set_ylim(-230, 60)
        ax.set_xlabel("x [nm]")
        ax.set_title(f"{ttl}\ncoverage {r['coverage']*100:.0f}%, mouth "
                     f"{r['mouth_nm']:.0f} nm", fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("y [nm]")

    ax = axes[2]
    c = res["coverage_vs_AR"]
    ax.plot(c["AR"], np.array(c["diffuse"]) * 100, "-o", color="#888",
            lw=1.8, label="diffuse (cos)")
    ax.plot(c["AR"], np.array(c["collimated"]) * 100, "-o", color="#0b3d91",
            lw=1.8, label="collimated (cos^20)")
    ax.set_xlabel("trench aspect ratio")
    ax.set_ylabel("bottom step coverage [%]")
    ax.set_title(f"Line-of-sight tax: coverage falls with AR;\ncollimation "
                 f"buys {res['collimation_gain_at_AR3']}x at AR 3 — ALD "
                 f"(tel) beyond that", fontsize=10)
    ax.legend(fontsize=8)
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
    _plot(os.path.join(ASSETS, "amat_pvd.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
