#!/usr/bin/env python3
"""barrel_cad.py — dimension-driven parametric lens barrel + GD&T drawing.

The optics/FEA/control modules reason about the barrel's *behaviour*; this module
is the mechanical-design deliverable itself — the CAD a 鏡筒メカ設計 engineer draws:

  1. A **parametric, dimension-driven 3-D solid** of the barrel (outer wall, mount
     flange, and the internal stepped lens seats), revolved from a named-dimension
     cross-section and exported as **STL** (importable into any CAD/CAM).
  2. A **3-D render** of that solid.
  3. A **GD&T engineering drawing** (cross-section with datums, feature control
     frames, fits and a title block) as a PNG *and* a real **DXF** — and every
     geometric tolerance is tied back to the optical/thermal budget it protects
     (bore coaxiality → element decenter → boresight; seat runout → tilt; mount
     flatness → focus), so the drawing is *derived*, not decorative.

numpy + trimesh (STL) + ezdxf (DXF) + matplotlib. Run: python3 barrel_cad.py
"""
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, FancyArrow
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import trimesh
import ezdxf

# ── Parametric dimensions [mm] — the single source of truth ──────────────────
DIMS = {
    "length": 40.0,
    "outer_dia": 30.0,
    "flange_dia": 40.0,
    "flange_thk": 4.0,
    "wall_nom": 2.0,
    "bolt_circle_dia": 34.0,
    "bolt_hole_dia": 3.4,
    "n_bolts": 3,
    # Internal lens seats: (z_start, z_end, seat_diameter) front→rear.
    "seats": [
        (34.0, 40.0, 27.2),   # element 1 (front, largest)
        (24.0, 34.0, 25.8),   # element 2
        (14.0, 24.0, 24.6),   # element 3
        (6.0, 14.0, 23.6),    # element 4 (rear, smallest)
    ],
    "rear_bore_dia": 26.0,    # rear counterbore / sensor-side clearance
}

MATERIAL = ("Al 6061-T6", 2700.0)   # (name, density kg/m^3)

# ── GD&T schedule — each tolerance linked to the budget it protects ──────────
# (feature, characteristic, tolerance[mm], datum, rationale)
GDT = [
    ("Mount face",      "flatness",       0.008, "—", "Datum A seating; tilts the whole stack → focus/keystone"),
    ("Bore axis",       "datum",          None,  "B", "Datum B: the optical axis reference for all seats"),
    ("Barrel OD",       "coaxiality",     0.020, "B", "Body concentric to bore → even wall, balanced thermal growth"),
    ("Lens seats (×4)", "circular runout",0.005, "B", "Seat wobble → element tilt → coma; drives the boresight budget"),
    ("Seat bores",      "cylindricity",   0.004, "—", "Seat form → element decenter → boresight (H7 slip fit)"),
    ("Front seat face", "perpendicularity",0.006,"B", "Axial reference for element 1 → despace/defocus"),
]
GENERAL_TOL = "±0.05 (linear), ±0.5° (angular) unless noted"


def profile_points():
    """Closed (r, z) cross-section of the solid barrel material."""
    Ro = DIMS["outer_dia"] / 2
    Rf = DIMS["flange_dia"] / 2
    tf = DIMS["flange_thk"]
    L = DIMS["length"]
    r_rear = DIMS["rear_bore_dia"] / 2
    seats = sorted(DIMS["seats"], key=lambda s: s[0])   # rear→front by z

    # Inner boundary (bore) rear→front, stepping at each seat shoulder.
    inner = [(r_rear, 0.0), (r_rear, seats[0][0])]
    for (z0, z1, dia) in seats:
        r = dia / 2
        inner.append((r, z0))
        inner.append((r, z1))
    # Outer boundary front→rear: front face, body, flange step, flange bottom.
    outer = [(Ro, L), (Ro, tf), (Rf, tf), (Rf, 0.0)]
    pts = inner + [(Ro, L)] + outer[1:] + [(r_rear, 0.0)]
    return np.array(pts, float)


def build_solid():
    prof = profile_points()
    mesh = trimesh.creation.revolve(prof, sections=180)
    mesh.fix_normals()
    return mesh


def _bolt_cylinders():
    """Bolt-hole negatives on the flange (for a boolean-capable exporter)."""
    R = DIMS["bolt_circle_dia"] / 2
    d = DIMS["bolt_hole_dia"]
    cyls = []
    for k in range(DIMS["n_bolts"]):
        ang = 2 * np.pi * k / DIMS["n_bolts"] + np.pi / 6
        c = trimesh.creation.cylinder(radius=d / 2, height=DIMS["flange_thk"] * 3,
                                      sections=32)
        c.apply_translation([R * np.cos(ang), R * np.sin(ang), DIMS["flange_thk"] / 2])
        cyls.append(c)
    return cyls


def render_3d(mesh, path):
    fig = plt.figure(figsize=(6.2, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    tri = mesh.triangles                      # (F,3,3)
    # Simple diffuse shading from the z-axis.
    n = mesh.face_normals
    shade = 0.45 + 0.55 * np.clip(n @ np.array([0.3, 0.3, 0.9]), 0, 1)
    colors = np.zeros((len(tri), 4))
    base = np.array([0.42, 0.55, 0.72])
    colors[:, :3] = base * shade[:, None]
    colors[:, 3] = 1.0
    pc = Poly3DCollection(tri, facecolors=colors, edgecolors="none")
    ax.add_collection3d(pc)
    v = mesh.vertices
    ax.set_xlim(v[:, 0].min(), v[:, 0].max())
    ax.set_ylim(v[:, 1].min(), v[:, 1].max())
    ax.set_zlim(v[:, 2].min(), v[:, 2].max())
    ax.set_box_aspect((np.ptp(v[:, 0]), np.ptp(v[:, 1]), np.ptp(v[:, 2])))
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()
    ax.set_title("Parametric lens barrel — revolved solid\n"
                 f"OD {DIMS['outer_dia']:.0f} · L {DIMS['length']:.0f} · "
                 "4 lens seats · Ø40 mount flange", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fcf(ax, x, y, sym, tol, datum, w=17, h=3.4):
    """Draw a feature control frame [sym | tol | datum] at (x,y) top-left."""
    cells = [sym, f"⌀{tol}" if sym == "◎" else f"{tol}"]
    if datum and datum != "—":
        cells.append(datum)
    n = len(cells)
    cw = w / n
    for i, txt in enumerate(cells):
        ax.add_patch(plt.Rectangle((x + i * cw, y - h), cw, h, fill=False,
                                   ec="k", lw=1.0))
        ax.text(x + i * cw + cw / 2, y - h / 2, txt, ha="center", va="center",
                fontsize=7.5)
    return x, y - h / 2


def _datum_tri(ax, x, y, label, up=True):
    dy = 2.4 if up else -2.4
    ax.add_patch(MplPoly([(x, y), (x - 1.1, y + dy), (x + 1.1, y + dy)],
                         closed=True, fc="k"))
    ax.add_patch(plt.Rectangle((x - 1.6, y + dy + (0 if up else -3.2)), 3.2, 3.2,
                               fill=False, ec="k", lw=1.1))
    ax.text(x, y + dy + (1.6 if up else -1.6), label, ha="center", va="center",
            fontsize=8, fontweight="bold")


def draw_gdt(path):
    prof = profile_points()
    r, z = prof[:, 0], prof[:, 1]
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    # Full section: material on both sides of the axis (r and -r), hatched.
    top = np.column_stack([z, r])
    bot = np.column_stack([z, -r])
    ax.add_patch(MplPoly(top, closed=True, fc="#dfe6ee", ec="k", lw=1.4,
                         hatch="////"))
    ax.add_patch(MplPoly(bot, closed=True, fc="#dfe6ee", ec="k", lw=1.4,
                         hatch="////"))
    L = DIMS["length"]
    # Centreline (axis) — dash-dot.
    ax.plot([-4, L + 4], [0, 0], "-.", color="#c0392b", lw=1.0)
    ax.text(L + 4.5, 0, "℄", color="#c0392b", fontsize=11, va="center")

    Ro = DIMS["outer_dia"] / 2
    Rf = DIMS["flange_dia"] / 2

    # --- Dimensions ---------------------------------------------------------
    def dim_h(z0, z1, y, txt):        # horizontal dimension
        ax.annotate("", (z1, y), (z0, y),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=.9))
        ax.text((z0 + z1) / 2, y + .5, txt, ha="center", va="bottom", fontsize=8)

    def dim_v(z0, y0, y1, txt, side=1):   # vertical (diameter) dimension
        ax.annotate("", (z0, y1), (z0, y0),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=.9))
        ax.text(z0 + side * .6, (y0 + y1) / 2, txt, ha="left" if side > 0 else "right",
                va="center", fontsize=8, rotation=90)

    dim_h(0, L, -Rf - 4.0, f"{L:.0f} ±0.1")
    dim_v(L + 2.2, -Ro, Ro, f"⌀{DIMS['outer_dia']:.0f} -0.02/0", side=1)
    dim_v(-2.2, -Rf, Rf, f"⌀{DIMS['flange_dia']:.0f}", side=-1)
    ax.text(4.6, Rf - 1.4, f"flange t{DIMS['flange_thk']:.0f}", fontsize=7)
    # A seat diameter callout (H7 slip fit).
    zs0, zs1, sd = DIMS["seats"][0]
    ax.annotate(f"⌀{sd:.1f} H7", ((zs0 + zs1) / 2, sd / 2 - 0.3),
                ((zs0 + zs1) / 2 + 3, sd / 2 + 5),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=.8))

    # --- Datums -------------------------------------------------------------
    _datum_tri(ax, 0, Rf, "A", up=True)          # mount face = datum A
    ax.text(0, Rf + 6.0, "mount face", ha="center", fontsize=7, color="#333")
    _datum_tri(ax, L + 7.5, 0, "B", up=True)     # bore axis = datum B (on ℄ extension)

    # --- Feature control frames (linked to GDT schedule) --------------------
    y0 = Rf + 4.0
    _fcf(ax, 4, y0, "⏥", 0.008, "—")             # flatness of mount face
    ax.annotate("", (0.3, Rf), (8, y0 - 1.7), arrowprops=dict(arrowstyle="->", lw=.7))
    _fcf(ax, 22, y0, "◎", 0.020, "B")            # coaxiality of OD to B
    ax.annotate("", (30, Ro), (26, y0 - 1.7), arrowprops=dict(arrowstyle="->", lw=.7))
    _fcf(ax, 43, y0 + 4, "↗", 0.005, "B")        # circular runout of seats
    ax.annotate("", (36, sd / 2), (47, y0 + 2.5), arrowprops=dict(arrowstyle="->", lw=.7))
    ax.text(43, y0 + 5.4, "lens seats ×4", fontsize=6.8, color="#333")

    # --- GD&T rationale note (links geometric tol → optical budget) ---------
    note = ("GD&T derived from the boresight budget (tolerance module #4):\n"
            "  ⏥ mount flatness 0.008  → stack tilt → focus / keystone\n"
            "  ◎ OD coaxial ⌀0.02 to B → balanced wall / thermal growth\n"
            "  ↗ seat runout 0.005 to B → element tilt → coma\n"
            "  seat ⌀ H7 slip fit       → decenter → boresight P99 ≤ budget")
    ax.text(1.5, 3.0, note, fontsize=7.2, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f7fb", ec="#8aa0b8", lw=.8))

    # --- Title block --------------------------------------------------------
    tb_x, tb_y, tb_w, tb_h = 20, -Rf - 15.5, 30, 9.5
    ax.add_patch(plt.Rectangle((tb_x, tb_y), tb_w, tb_h, fill=False, ec="k", lw=1.3))
    ax.plot([tb_x, tb_x + tb_w], [tb_y + tb_h * .5] * 2, "k", lw=.7)
    ax.plot([tb_x + tb_w * .55] * 2, [tb_y, tb_y + tb_h], "k", lw=.7)
    ax.text(tb_x + 1, tb_y + tb_h * .72, "LENS BARREL", fontsize=8.5, fontweight="bold")
    ax.text(tb_x + 1, tb_y + tb_h * .28, f"MATL {MATERIAL[0]}", fontsize=7.5)
    ax.text(tb_x + tb_w * .57, tb_y + tb_h * .72, "SCALE 2:1", fontsize=7.5)
    ax.text(tb_x + tb_w * .57, tb_y + tb_h * .28, "3rd ANGLE", fontsize=7.5)
    ax.text(tb_x, tb_y - 1.8, f"General tol {GENERAL_TOL}", fontsize=6.8, color="#333")

    ax.set_xlim(-9, L + 14)
    ax.set_ylim(-Rf - 18, Rf + 11)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Lens barrel — GD&T section: datums A/B, feature control frames, H7 seat fit",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_dxf(path):
    """Emit the section outline + centreline + FCF text as a real DXF drawing."""
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4       # millimetres
    msp = doc.modelspace()
    prof = profile_points()
    top = [(z, r) for r, z in prof]
    bot = [(z, -r) for r, z in prof]
    msp.add_lwpolyline(top, close=True, dxfattribs={"layer": "OUTLINE"})
    msp.add_lwpolyline(bot, close=True, dxfattribs={"layer": "OUTLINE"})
    L = DIMS["length"]
    doc.layers.add("CENTER", color=1)
    msp.add_line((-4, 0), (L + 4, 0), dxfattribs={"layer": "CENTER", "linetype": "CENTER"}) \
        if "CENTER" in doc.linetypes else msp.add_line((-4, 0), (L + 4, 0), dxfattribs={"layer": "CENTER"})
    # GD&T + dimension notes as text (CAD-editable).
    doc.layers.add("GDT", color=5)
    notes = [f"{feat}: {char} {tol if tol is not None else ''} {('|'+dat) if dat not in (None,'—') else ''}".strip()
             for (feat, char, tol, dat, _why) in GDT]
    y = DIMS["flange_dia"] / 2 + 6
    for i, nt in enumerate(notes):
        msp.add_text(nt, dxfattribs={"layer": "GDT", "height": 1.2}).set_placement((-8, y + i * 2.2))
    msp.add_text(f"LENS BARREL   MATL {MATERIAL[0]}   General tol {GENERAL_TOL}",
                 dxfattribs={"layer": "GDT", "height": 1.4}).set_placement((-8, -DIMS["flange_dia"] / 2 - 10))
    doc.saveas(path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    figs = os.path.join(here, "figures"); os.makedirs(figs, exist_ok=True)
    out_dir = os.path.join(here, "out"); os.makedirs(out_dir, exist_ok=True)

    mesh = build_solid()
    stl_path = os.path.join(out_dir, "lens_barrel.stl")
    mesh.export(stl_path)
    render_3d(mesh, os.path.join(figs, "barrel_3d.png"))
    draw_gdt(os.path.join(figs, "barrel_gdt.png"))
    dxf_path = os.path.join(out_dir, "lens_barrel_drawing.dxf")
    write_dxf(dxf_path)

    vol_cm3 = mesh.volume / 1000.0
    mass_g = vol_cm3 * 1e-6 * MATERIAL[1] * 1000.0   # cm^3→m^3 * kg/m^3 → g
    out = {
        "dimensions_mm": {k: v for k, v in DIMS.items() if k != "seats"},
        "n_lens_seats": len(DIMS["seats"]),
        "solid_watertight": bool(mesh.is_watertight),
        "volume_cm3": round(vol_cm3, 2),
        "mass_g": round(mass_g, 1),
        "material": MATERIAL[0],
        "stl_file": os.path.relpath(stl_path, here),
        "dxf_file": os.path.relpath(dxf_path, here),
        "gdt_schedule": [
            {"feature": f, "characteristic": c, "tol_mm": t, "datum": d, "rationale": w}
            for (f, c, t, d, w) in GDT
        ],
        "general_tolerance": GENERAL_TOL,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open(os.path.join(here, "barrel_cad_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
