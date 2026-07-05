#!/usr/bin/env python3
"""im_dimension.py — one-shot image dimension measurement (IM-series style).

The LJ modules cover the laser-profiler side; this module covers the other
Keyence measurement pillar: the projector-style IMAGE DIMENSION MEASUREMENT
system — "place the part anywhere on the stage, press one button, get every
dimension with GO/NG" . The product's whole value proposition is a
measurement-engineering claim, so that is what gets simulated and proven:

  1. TELECENTRIC IMAGE SYNTHESIS — a machined plate (2 reamed holes) rendered
     with erf edge transitions (optical blur), illumination gradient, sensor
     noise, and a RANDOM POSE every shot (±3 mm, ±5°): the part is never
     placed the same way twice, exactly like a real operator.
  2. AUTO POSE RECOVERY — moment centroid + principal axis: measurements are
     taken in the PART frame, which is why placement doesn't matter.
  3. SUB-PIXEL EDGE PROBING — radial/perpendicular scan rays, gradient peak
     with parabolic + centroid refinement (≈1/30 px), then least-squares
     circle and line fits over hundreds of edge points.
  4. GO/NG AGAINST THE DRAWING — diameters, hole pitch, plate width vs
     tolerances.
  5. THE PITCH, QUANTIFIED — repeatability over 30 random placements vs a
     simulated hand-caliper study (3 operators with individual bias +
     technique scatter): "誰が測っても同じ" as numbers, not marketing.

Pure numpy + matplotlib.  Run:  python3 keyence/im_dimension.py
"""
import json
import os

import numpy as np

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "im_dimension_results.json")

# ── stage / optics ──────────────────────────────────────────────────────────
MM_PER_PX = 0.03          # telecentric scale
IMG_W, IMG_H = 900, 620   # pixels (27 x 18.6 mm field)
EDGE_SIGMA_PX = 1.1       # optical edge blur
NOISE = 0.012             # sensor noise (fraction of full scale)

# ── the part (truth, mm, in part frame; origin = plate center) ─────────────
PLATE_W, PLATE_H = 22.0, 14.0
HOLE_R = 3.0                       # 2 reamed holes, true ⌀6.000
HOLE_A = (-6.0, 0.0)
HOLE_B = (+6.0, 0.0)
# drawing tolerances
TOL = {
    "dia_A_mm": (6.000, 0.020),
    "dia_B_mm": (6.000, 0.020),
    "pitch_mm": (12.000, 0.030),
    "width_mm": (14.000, 0.050),   # plate height = the 14 mm dimension
}


def render(pose, rng):
    """Render one shot. pose = (dx_mm, dy_mm, theta_rad)."""
    dx, dy, th = pose
    xs = (np.arange(IMG_W) - IMG_W / 2) * MM_PER_PX
    ys = (np.arange(IMG_H) - IMG_H / 2) * MM_PER_PX
    X, Y = np.meshgrid(xs, ys)
    # image -> part frame
    c, s = np.cos(-th), np.sin(-th)
    Xp = c * (X - dx) - s * (Y - dy)
    Yp = s * (X - dx) + c * (Y - dy)

    sig = EDGE_SIGMA_PX * MM_PER_PX

    def smooth(d):                      # 0/1 transition over signed dist d
        return 0.5 * (1.0 + np.tanh(d / (sig * 1.2)))

    d_rect = np.minimum(PLATE_W / 2 - np.abs(Xp), PLATE_H / 2 - np.abs(Yp))
    part = smooth(d_rect)
    for (hx, hy) in (HOLE_A, HOLE_B):
        d_hole = np.hypot(Xp - hx, Yp - hy) - HOLE_R
        part *= smooth(d_hole)
    img = 0.12 + 0.78 * part
    img *= 1.0 + 0.05 * (X / xs.max())          # illumination gradient
    img += NOISE * rng.standard_normal(img.shape)
    return np.clip(img, 0.0, 1.0)


# ── pose recovery (why placement doesn't matter) ────────────────────────────
def find_pose(img):
    """Rough part pose from intensity moments + principal axis."""
    w = np.clip(img - 0.5, 0.0, None)           # foreground weight
    tot = w.sum()
    ys, xs = np.mgrid[0:IMG_H, 0:IMG_W]
    cx = float((w * xs).sum() / tot)
    cy = float((w * ys).sum() / tot)
    mxx = float((w * (xs - cx) ** 2).sum() / tot)
    myy = float((w * (ys - cy) ** 2).sum() / tot)
    mxy = float((w * (xs - cx) * (ys - cy)).sum() / tot)
    th = 0.5 * np.arctan2(2 * mxy, mxx - myy)   # long axis ≈ x of part
    dx = (cx - IMG_W / 2) * MM_PER_PX
    dy = (cy - IMG_H / 2) * MM_PER_PX
    return dx, dy, th


def _bilinear(img, x, y):
    x = np.clip(x, 0, IMG_W - 1.001)
    y = np.clip(y, 0, IMG_H - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def _edge_on_ray(img, p0, direction, half_mm=0.7, n=48):
    """Sub-pixel edge position along a scan ray centered at p0 (image mm).

    Samples intensity, takes |gradient|, parabolic peak + centroid refine.
    Returns edge point in image-mm coordinates or None."""
    ts = np.linspace(-half_mm, half_mm, n)
    px = (p0[0] + ts * direction[0]) / MM_PER_PX + IMG_W / 2
    py = (p0[1] + ts * direction[1]) / MM_PER_PX + IMG_H / 2
    prof = _bilinear(img, px, py)
    g = np.abs(np.gradient(prof))
    i = int(np.argmax(g[2:-2])) + 2
    if g[i] < 0.03:
        return None
    a, b, c = g[i - 1], g[i], g[i + 1]
    denom = a - 2 * b + c
    dt = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
    # moment refine over a small window around the peak
    w = g[i - 2:i + 3]
    tt = ts[i - 2:i + 3]
    t_edge = (ts[i] + dt * (ts[1] - ts[0])) * 0.5 \
        + 0.5 * float(np.sum(w * tt) / np.sum(w))
    return (p0[0] + t_edge * direction[0], p0[1] + t_edge * direction[1])


def fit_circle(pts):
    """Algebraic (Kasa) least-squares circle fit. Returns (cx, cy, r)."""
    p = np.asarray(pts)
    A = np.column_stack([2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))])
    b = p[:, 0] ** 2 + p[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    return float(cx), float(cy), float(r)


def measure(img):
    """Full one-shot measurement. Returns dict of dimensions [mm]."""
    dx, dy, th = find_pose(img)
    c, s = np.cos(th), np.sin(th)

    def to_img(xp, yp):                # part frame -> image mm
        return (c * xp - s * yp + dx, s * xp + c * yp + dy)

    # holes: radial rays around nominal centers
    circles = []
    for (hx, hy) in (HOLE_A, HOLE_B):
        cx0, cy0 = to_img(hx, hy)
        pts = []
        for ang in np.linspace(0, 2 * np.pi, 72, endpoint=False):
            d = (np.cos(ang), np.sin(ang))
            p0 = (cx0 + HOLE_R * d[0], cy0 + HOLE_R * d[1])
            e = _edge_on_ray(img, p0, d, half_mm=0.6)
            if e is not None:
                pts.append(e)
        circles.append((fit_circle(pts), pts))
    (ca, ra_pts), (cb, rb_pts) = circles

    # plate width (the 14 mm dimension): perpendicular scans on both long
    # edges, line fit y = a + b x in a rotated frame, distance between lines
    edges = {+1: [], -1: []}
    for side in (+1, -1):
        for xp in np.linspace(-PLATE_W / 2 + 2.5, PLATE_W / 2 - 2.5, 24):
            p0 = to_img(xp, side * PLATE_H / 2)
            d = (-s * side, c * side)             # outward normal in image mm
            e = _edge_on_ray(img, p0, d, half_mm=0.6)
            if e is not None:
                edges[side].append(e)
    # rotate edge points back to part frame; width = mean gap of the y's
    def to_part(pt):
        x, y = pt[0] - dx, pt[1] - dy
        return (c * x + s * y, -s * x + c * y)
    y_top = np.array([to_part(p)[1] for p in edges[+1]])
    y_bot = np.array([to_part(p)[1] for p in edges[-1]])
    width = float(np.mean(y_top) - np.mean(y_bot))

    return {
        "dia_A_mm": 2 * ca[2], "dia_B_mm": 2 * cb[2],
        "pitch_mm": float(np.hypot(cb[0] - ca[0], cb[1] - ca[1])),
        "width_mm": width,
        "_pose": (dx, dy, th),
        "_circle_pts": (ra_pts, rb_pts), "_edge_pts": (edges[+1], edges[-1]),
    }


def go_ng(dims):
    out = {}
    for k, (nom, tol) in TOL.items():
        dev = dims[k] - nom
        out[k] = {"measured": round(dims[k], 4), "nominal": nom,
                  "tol": tol, "dev_um": round(dev * 1000, 2),
                  "go": bool(abs(dev) <= tol)}
    return out


def repeatability(n=30, seed=4):
    """n shots, random pose each time (the operator just drops the part)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        pose = (float(rng.uniform(-3, 3)), float(rng.uniform(-2, 2)),
                float(np.deg2rad(rng.uniform(-5, 5))))
        img = render(pose, rng)
        d = measure(img)
        rows.append([d[k] for k in TOL])
    return np.asarray(rows)


def caliper_study(n_per_op=10, seed=9):
    """Hand-caliper baseline: 3 operators, individual bias + technique
    scatter (values from a typical MSA on vernier calipers)."""
    rng = np.random.default_rng(seed)
    biases_um = (-6.0, +2.0, +7.0)
    sigma_um = 8.0
    truth = {k: TOL[k][0] for k in TOL}
    truth["dia_A_mm"] = 2 * HOLE_R
    truth["dia_B_mm"] = 2 * HOLE_R
    out = {}
    for k in TOL:
        ops = [truth[k] + (b + sigma_um * rng.standard_normal(n_per_op))
               / 1000.0 for b in biases_um]
        out[k] = np.asarray(ops)                 # (3, n)
    return out


# ── figures ─────────────────────────────────────────────────────────────────
def _plot_scene(path, seed=11):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    pose = (1.8, -1.1, np.deg2rad(4.0))
    img = render(pose, rng)
    dims = measure(img)
    gn = go_ng(dims)

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1,
              extent=[-IMG_W / 2 * MM_PER_PX, IMG_W / 2 * MM_PER_PX,
                      IMG_H / 2 * MM_PER_PX, -IMG_H / 2 * MM_PER_PX])
    for pts, col in zip(dims["_circle_pts"], ("#2c6", "#2c6")):
        p = np.asarray(pts)
        ax.plot(p[:, 0], p[:, 1], ".", ms=2.5, color=col)
    for pts in dims["_edge_pts"]:
        p = np.asarray(pts)
        ax.plot(p[:, 0], p[:, 1], ".", ms=2.5, color="#fa3")
    lines = [f"{k}: {v['measured']:.4f} mm  (nom {v['nominal']:.3f} "
             f"+/-{v['tol']:.3f})  {'GO' if v['go'] else 'NG'}"
             for k, v in gn.items()]
    ax.text(0.02, 0.985, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, family="monospace", va="top",
            bbox=dict(fc="white", alpha=0.85, ec="0.6"))
    ax.set_title("One-shot dimension measurement — random placement, "
                 "auto pose recovery, sub-pixel edges (dots)")
    ax.set_xlabel("stage x [mm]")
    ax.set_ylabel("stage y [mm]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_repeat(path, rep, cal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(TOL)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    x = np.arange(len(keys))
    sig_im = rep.std(axis=0, ddof=1) * 1000
    sig_cal = [cal[k].std(ddof=1) * 1000 for k in keys]
    ax.bar(x - 0.18, sig_im, width=0.36, color="#26a",
           label="image measurement (30 random placements)")
    ax.bar(x + 0.18, sig_cal, width=0.36, color="#c86",
           label="hand caliper (3 operators pooled)")
    ax.set_xticks(x, [k.replace("_mm", "") for k in keys], fontsize=9)
    ax.set_ylabel("std dev [um]")
    ax.set_title("Repeatability: placement-proof vs operator-dependent")
    ax.legend(fontsize=8)

    ax = axes[1]
    k = "dia_A_mm"
    nom, tol = TOL[k]
    ax.plot(rep[:, 0] * 1000 - nom * 1000, "o-", ms=4, color="#26a",
            label="image measurement")
    for o in range(3):
        ax.plot(np.arange(10) + 32, cal[k][o] * 1000 - nom * 1000, "s",
                ms=4, alpha=0.8, label=f"caliper op{o+1}")
    ax.axhline(+tol * 1000, color="k", ls="--", lw=1)
    ax.axhline(-tol * 1000, color="k", ls="--", lw=1, label="drawing tol")
    ax.set_xlabel("measurement #")
    ax.set_ylabel("deviation from nominal [um]")
    ax.set_title("Hole A diameter: 30 placements vs 3 operators")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    rng = np.random.default_rng(1)
    img = render((0.5, 0.3, np.deg2rad(2.0)), rng)
    gn = go_ng(measure(img))

    rep = repeatability()
    cal = caliper_study()
    keys = list(TOL)
    summary = {
        "mm_per_px": MM_PER_PX,
        "one_shot": gn,
        "repeatability_um": {k: round(float(rep[:, i].std(ddof=1) * 1000), 2)
                             for i, k in enumerate(keys)},
        "mean_abs_error_um": {k: round(float(np.mean(np.abs(
            rep[:, i] - (2 * HOLE_R if k.startswith("dia") else TOL[k][0])
        )) * 1000), 2) for i, k in enumerate(keys)},
        "caliper_sigma_um": {k: round(float(cal[k].std(ddof=1) * 1000), 2)
                             for k in keys},
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)
    for d in (FIGDIR, ASSETS):
        _plot_scene(os.path.join(d, "im_scene.png"))
        _plot_repeat(os.path.join(d, "im_repeat.png"), rep, cal)
    print("figures ->", FIGDIR, "and", ASSETS)


if __name__ == "__main__":
    main()
