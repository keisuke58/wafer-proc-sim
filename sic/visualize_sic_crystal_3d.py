"""
4H-SiC Crystal Structure — 3D Atom Visualization
==================================================
Space group: P6_3mc (No. 186), hexagonal
a = 3.073 A,  c = 10.053 A  (4H polytype, 4 bilayers per unit cell)

Stacking: A-B-C-B  (4 unique layers → 4H)
  Layer A: Si at (0,0,0)        + C at (0,0,u)
  Layer B: Si at (1/3,2/3,1/4) + C at (1/3,2/3,1/4+u)
  Layer C: Si at (0,0,1/2)      + C at (0,0,1/2+u)
  Layer B: Si at (2/3,1/3,3/4) + C at (2/3,1/3,3/4+u)
  u = 3/16 (Si-C bond height fraction)

Usage:
    python sic/visualize_sic_crystal_3d.py
    python sic/visualize_sic_crystal_3d.py --save
    python sic/visualize_sic_crystal_3d.py --save --white
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ── Lattice parameters ────────────────────────────────────────────────────────
A_LAT = 3.073   # angstrom
C_LAT = 10.053  # angstrom  (4H: 4 bilayers)
U_SIC = 3.0 / 16.0   # fractional c for Si-C bond (≈ 0.1875)

# Primitive vectors (cartesian, angstrom)
A1 = np.array([A_LAT,          0.0,    0.0])
A2 = np.array([-A_LAT / 2.0,  A_LAT * np.sqrt(3) / 2.0, 0.0])
A3 = np.array([0.0,            0.0,    C_LAT])

def frac_to_cart(u, v, w):
    return u * A1 + v * A2 + w * A3

# ── Fractional positions inside one unit cell ─────────────────────────────────
# Each "bilayer" = 1 Si + 1 C (4 bilayers for 4H)
SI_FRAC = [
    (0,     0,     0.000),   # Layer A
    (1./3., 2./3., 0.250),   # Layer B
    (0,     0,     0.500),   # Layer C
    (2./3., 1./3., 0.750),   # Layer B' (shifted)
]
C_FRAC = [(u, v, w + U_SIC) for u, v, w in SI_FRAC]

# ── Tile into a supercell ─────────────────────────────────────────────────────
def build_supercell(nx=3, ny=3, nz=2):
    si_pos, c_pos = [], []
    for ix in range(-nx, nx + 1):
        for iy in range(-ny, ny + 1):
            for iz in range(nz):
                offset = ix * A1 + iy * A2 + iz * A3
                for u, v, w in SI_FRAC:
                    si_pos.append(frac_to_cart(u, v, w) + offset)
                for u, v, w in C_FRAC:
                    c_pos.append(frac_to_cart(u, v, w) + offset)
    return np.array(si_pos), np.array(c_pos)

# ── Bond drawing ──────────────────────────────────────────────────────────────
BOND_LEN = 1.95   # Si-C bond ~1.89 A, draw within 1.95 A

def get_bonds(si_pos, c_pos, max_len=BOND_LEN):
    segments = []
    for s in si_pos:
        for c in c_pos:
            if np.linalg.norm(s - c) < max_len:
                segments.append([s, c])
    return segments

# ── Hexagonal prism outline ───────────────────────────────────────────────────
def hex_prism_edges(z0=0.0, z1=C_LAT):
    r = A_LAT
    angles = np.linspace(0, 2 * np.pi, 7)
    verts_bot = [(r * np.cos(a), r * np.sin(a), z0) for a in angles]
    verts_top = [(r * np.cos(a), r * np.sin(a), z1) for a in angles]
    edges = []
    for i in range(6):
        edges.append([verts_bot[i], verts_bot[i+1]])
        edges.append([verts_top[i], verts_top[i+1]])
        edges.append([verts_bot[i], verts_top[i]])
    return edges

# ── Theme ─────────────────────────────────────────────────────────────────────
DARK = dict(
    bg="#1A1A2E", pane="#1A1A2E", pane_edge="#333355",
    tick="#888899", axis_label="#AAAACC", spine="#444466",
    bond="#AAAACC", prism="#FFD700",
    si_face="#4FC3F7", si_edge="#1565C0",
    c_face="#EF9A9A",  c_edge="#B71C1C",
    ax_a="#00E676", ax_b="#69F0AE", ax_c="#FF6E40",
    layer="#FFD700", title="white", text="white",
    side_bond="#AAAACC", interlayer="#666688",
    props_title="#FFD700", props_key="#AAAACC", props_val="white",
    dim="#CCCCCC", side_bg="#1A1A2E",
)
LITE = dict(
    bg="white", pane="white", pane_edge="#CCCCCC",
    tick="#555555", axis_label="#333333", spine="#AAAAAA",
    bond="#AAAAAA", prism="#B8860B",
    si_face="#1976D2", si_edge="#0D47A1",
    c_face="#D32F2F",  c_edge="#7F0000",
    ax_a="#2E7D32", ax_b="#43A047", ax_c="#BF360C",
    layer="#B8860B", title="#111111", text="#111111",
    side_bond="#888888", interlayer="#AAAAAA",
    props_title="#333333", props_key="#555555", props_val="#111111",
    dim="#444444", side_bg="white",
)

# ── Main ──────────────────────────────────────────────────────────────────────
def make_figure(theme=DARK):
    si_all, c_all = build_supercell(nx=2, ny=2, nz=1)

    # Clip atoms to a cylinder for clean look
    r_clip = A_LAT * 2.6
    z_clip_lo, z_clip_hi = -0.1, C_LAT + 0.1

    def in_clip(pos):
        return (np.sqrt(pos[:, 0]**2 + pos[:, 1]**2) <= r_clip) & \
               (pos[:, 2] >= z_clip_lo) & (pos[:, 2] <= z_clip_hi)

    si_vis = si_all[in_clip(si_all)]
    c_vis  = c_all[in_clip(c_all)]

    bonds = get_bonds(si_vis, c_vis)

    T = theme
    fig = plt.figure(figsize=(14, 10), facecolor=T["bg"])
    fig.suptitle("4H-SiC Crystal Structure\n"
                 "Stacking: A-B-C-B  (hexagonal, P6₃mc)",
                 fontsize=15, fontweight="bold", color=T["title"], y=0.97)

    # ── 3D main view ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(121, projection="3d")
    ax.set_facecolor(T["pane"])
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(T["pane_edge"])
    ax.yaxis.pane.set_edgecolor(T["pane_edge"])
    ax.zaxis.pane.set_edgecolor(T["pane_edge"])
    ax.grid(False)

    # Bonds
    if bonds:
        lc = Line3DCollection(bonds, linewidths=1.2, colors=T["bond"],
                              alpha=0.55, zorder=1)
        ax.add_collection3d(lc)

    # Atoms
    ax.scatter(si_vis[:, 0], si_vis[:, 1], si_vis[:, 2],
               c=T["si_face"], s=220, depthshade=True,
               edgecolors=T["si_edge"], linewidths=0.5, label="Si", zorder=5)
    ax.scatter(c_vis[:, 0], c_vis[:, 1], c_vis[:, 2],
               c=T["c_face"], s=140, depthshade=True,
               edgecolors=T["c_edge"], linewidths=0.5, label="C", zorder=5)

    # Unit cell hex prism
    prism = hex_prism_edges(0.0, C_LAT)
    edge_col = Line3DCollection(prism, linewidths=1.0, colors=T["prism"],
                                alpha=0.6, linestyles="--", zorder=2)
    ax.add_collection3d(edge_col)

    # Crystal axis arrows
    for vec, lbl, col_key in [
        (A1,                          "a [11-20]", "ax_a"),
        (A2,                          "b [1-100]", "ax_b"),
        (np.array([0, 0, C_LAT*0.45]),"c [0001]",  "ax_c"),
    ]:
        col = T[col_key]
        ax.quiver(0, 0, -1.5, vec[0], vec[1], vec[2],
                  color=col, linewidth=2.0, arrow_length_ratio=0.18)
        ax.text(vec[0]*1.18, vec[1]*1.18, vec[2]*1.0 - 1.5,
                lbl, color=col, fontsize=8.5, fontweight="bold")

    ax.set_xlabel("x  [A]", color=T["axis_label"], fontsize=9, labelpad=4)
    ax.set_ylabel("y  [A]", color=T["axis_label"], fontsize=9, labelpad=4)
    ax.set_zlabel("z  [A]", color=T["axis_label"], fontsize=9, labelpad=4)
    ax.tick_params(colors=T["tick"], labelsize=7)
    for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
        spine.line.set_color(T["spine"])

    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_zlim(-2, C_LAT + 2)
    ax.view_init(elev=20, azim=35)

    leg_fc = "#222244" if T["bg"] != "white" else "white"
    leg_ec = "#555577" if T["bg"] != "white" else "#AAAAAA"
    legend = ax.legend(loc="upper left", fontsize=10, framealpha=0.4,
                       facecolor=leg_fc, edgecolor=leg_ec,
                       labelcolor=T["text"])

    # Layer labels along c-axis
    layer_names = ["A", "B", "C", "B'"]
    for i, (u, v, w) in enumerate(SI_FRAC):
        pos = frac_to_cart(u, v, w)
        ax.text(4.2, 0.0, pos[2],
                f"  Layer {layer_names[i]}", color=T["layer"],
                fontsize=8.5, va="center", fontweight="bold")

    ax.set_title("3D supercell view", color=T["title"], fontsize=11, pad=8)

    # ── Side panel: bond topology + stacking info ─────────────────────────────
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor(T["side_bg"])
    ax2.axis("off")

    layers_z   = [w * C_LAT        for u, v, w in SI_FRAC]
    c_layers_z = [(w + U_SIC)*C_LAT for u, v, w in SI_FRAC]
    x_A  = 0.35; x_B1 = 0.55; x_C = 0.35; x_B2 = 0.55
    xs_si = [x_A, x_B1, x_C, x_B2]

    y_scale = 0.85 / C_LAT
    y0 = 0.07

    for i in range(4):
        ys = layers_z[i]   * y_scale + y0
        yc = c_layers_z[i] * y_scale + y0
        xs = xs_si[i]
        ax2.plot([xs, xs], [ys, yc], color=T["side_bond"], lw=1.5, zorder=1)
        ax2.scatter(xs, ys, s=320, color=T["si_face"], edgecolors=T["si_edge"],
                    linewidths=1.0, zorder=5)
        ax2.text(xs - 0.08, ys, "Si", color=T["text"], fontsize=9,
                 va="center", ha="right", fontweight="bold")
        ax2.scatter(xs, yc, s=200, color=T["c_face"], edgecolors=T["c_edge"],
                    linewidths=1.0, zorder=5)
        ax2.text(xs + 0.06, yc, "C", color=T["text"], fontsize=9,
                 va="center", ha="left", fontweight="bold")
        ax2.text(0.15, ys, layer_names[i], color=T["layer"],
                 fontsize=11, va="center", ha="center", fontweight="bold")
        if i < 3:
            ax2.plot([xs, xs_si[i+1]],
                     [yc, layers_z[i+1]*y_scale + y0],
                     color=T["interlayer"], lw=1.0, ls="--", zorder=0, alpha=0.6)

    # c-axis arrow
    ax2.annotate("", xy=(0.92, y0 + C_LAT*y_scale), xytext=(0.92, y0),
                 arrowprops=dict(arrowstyle="->", color=T["ax_c"], lw=2.0))
    ax2.text(0.94, y0 + C_LAT*y_scale*0.5,
             "c = 10.05 A\n[0001]",
             color=T["ax_c"], fontsize=9, va="center", ha="left",
             fontweight="bold")

    # a-axis annotation
    ax2.annotate("", xy=(x_B1+0.05, y0-0.03), xytext=(x_A-0.05, y0-0.03),
                 arrowprops=dict(arrowstyle="<->", color=T["ax_a"], lw=1.5))
    ax2.text((x_A+x_B1)/2, y0-0.06, "a = 3.07 A",
             color=T["ax_a"], fontsize=9, ha="center", va="top")

    # Bond length annotation
    ys_si0 = layers_z[0]*y_scale + y0
    yc_c0  = c_layers_z[0]*y_scale + y0
    ax2.annotate("", xy=(xs_si[0]+0.14, (ys_si0+yc_c0)/2),
                 xytext=(xs_si[0]+0.04, (ys_si0+yc_c0)/2),
                 arrowprops=dict(arrowstyle="->", color=T["dim"], lw=1.0))
    ax2.text(xs_si[0]+0.15, (ys_si0+yc_c0)/2,
             "d(Si-C)\n= 1.89 A", color=T["dim"], fontsize=8, va="center")

    # Key properties box
    props = [
        ("Space group",  "P6₃mc  (No. 186)"),
        ("Polytype",     "4H  (ABCB stacking)"),
        ("a lattice",    "3.073 A"),
        ("c lattice",    "10.053 A"),
        ("Si-C bond",    "1.89 A"),
        ("Bandgap",      "3.26 eV"),
        ("ue (c-axis)",  "~1000 cm2/Vs"),
        ("Hardness",     "26-28 GPa"),
        ("K_IC",         "1.0-2.8 MPa*sqrtm"),
        ("Atom/cell",    "8  (4 Si + 4 C)"),
    ]
    y_box = 0.40
    ax2.text(0.02, y_box+0.05, "Crystal Properties",
             color=T["props_title"], fontsize=11, fontweight="bold",
             transform=ax2.transAxes)
    for i, (k, v) in enumerate(props):
        y_row = y_box - 0.004 - i*0.038
        ax2.text(0.02, y_row, k+":", color=T["props_key"],
                 fontsize=9, transform=ax2.transAxes)
        ax2.text(0.40, y_row, v, color=T["props_val"],
                 fontsize=9, fontweight="bold", transform=ax2.transAxes)

    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    ax2.set_title("Side view (one unit cell)  +  Properties",
                  color=T["title"], fontsize=11, pad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save",  action="store_true")
    parser.add_argument("--white", action="store_true",
                        help="Use white background instead of dark")
    args = parser.parse_args()

    theme = LITE if args.white else DARK
    fig = make_figure(theme=theme)

    if args.save:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        os.makedirs(out_dir, exist_ok=True)
        suffix = "_white" if args.white else ""
        for ext in ("png", "pdf"):
            path = os.path.join(out_dir, f"sic_crystal_3d{suffix}.{ext}")
            fig.savefig(path, dpi=150, bbox_inches="tight",
                        facecolor=theme["bg"])
            print(f"Saved -> {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
