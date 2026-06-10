"""
4H-SiC Crystal Anisotropy Visualization
========================================
Panels:
  1. Crystal stacking: 4H vs 6H (ABCB / ABCACB)
  2. Polar: Young's modulus E(θ)
  3. Polar: Fracture toughness K_IC(θ)
  4. Chipping depth h_chip(θ) at standard DISCO dicing params
  5. Hexagonal wafer top-view — dicing direction guide (color-coded chipping)
  6. 4H vs 6H vs Si property comparison table

Usage:
    python sic/visualize_sic_anisotropy.py
    python sic/visualize_sic_anisotropy.py --save
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

# ── 4H-SiC physics (mirrors sic_chipping_kernel.cpp) ─────────────────────────
C44 = 163.0   # GPa

def E_SiC(theta_deg):
    """Young's modulus E(θ) [GPa]; θ=0 → c-axis [0001], θ=90 → a-axis [11-20]."""
    th = np.deg2rad(np.asarray(theta_deg, float))
    s2, c2 = np.sin(th)**2, np.cos(th)**2
    S11 = 1.0 / 448.0
    S33 = 1.0 / 476.0
    S13 = -0.10 / 448.0
    S44 = 1.0 / C44
    inv_E = S11*s2**2 + S33*c2**2 + (2*S13 + S44)*s2*c2
    return 1.0 / inv_E   # GPa

def K_IC_SiC(theta_deg):
    """Fracture toughness K_IC(θ) [MPa√m]."""
    th = np.deg2rad(np.asarray(theta_deg, float))
    c2 = np.cos(th)**2
    K_base = 2.0 + (2.8 - 2.0) * (1.0 - c2)
    basal  = np.exp(-(np.asarray(theta_deg, float) / 15.0)**2)
    return K_base * (1 - 0.3*basal) + 1.0 * 0.3*basal

def H_SiC(theta_deg):
    """Vickers hardness H(θ) [GPa]."""
    return 28.0 - 1.4 * np.cos(np.deg2rad(np.asarray(theta_deg, float)))**2

def chipping_um(theta_deg, blade_depth_um=30.0, feed_um_rev=1.0, spindle_krpm=25.0):
    """Lawn-Evans chipping depth [µm]."""
    H = H_SiC(theta_deg)
    E = E_SiC(theta_deg)
    K = K_IC_SiC(theta_deg)
    feed_mm_s = feed_um_rev * 1e-3 * spindle_krpm * 1000.0 / 60.0
    F_N = 0.08 * blade_depth_um * feed_mm_s * (H / 13.0)**0.6
    return 0.015 * np.sqrt(E * 1000 / (H * 1000)) * np.sqrt(F_N) / K

def Si_chipping():
    F_N = 0.08 * 30.0 * (1e-3 * 25e3 / 60) * 1.0
    return 0.015 * np.sqrt(130 * 1000 / (13 * 1000)) * np.sqrt(F_N) / 0.70

# ── Colour palette ────────────────────────────────────────────────────────────
TEAL  = "#00A99D"
RED   = "#E63329"
GOLD  = "#F5A623"
GRAY  = "#4A4A4A"
BG    = "#FAFAFA"

# ── Helpers ───────────────────────────────────────────────────────────────────
def mirror_polar(fn, angles_deg):
    """Build full 360° polar array from a [0°,90°] function via hexagonal symmetry."""
    seg0  = fn(angles_deg[angles_deg <= 90])
    seg1  = fn(180 - angles_deg[(angles_deg > 90)  & (angles_deg <= 180)])
    seg2  = fn(angles_deg[(angles_deg > 180) & (angles_deg <= 270)] - 180)
    seg3  = fn(360 - angles_deg[angles_deg > 270])
    return np.concatenate([seg0, seg1, seg2, seg3])

def _style_polar(ax, title):
    ax.set_facecolor(BG)
    ax.tick_params(labelsize=7, colors=GRAY)
    ax.set_title(title, fontsize=10, fontweight="bold", color=GRAY, pad=16)

def draw_stacking(ax, layers_x, layers_lbl, colors, title, note=""):
    n = len(layers_x)
    for i, (x, lbl, col) in enumerate(zip(layers_x, layers_lbl, colors)):
        c = Circle((x, i), 0.28, color=col, zorder=3, ec="white", lw=1.5)
        ax.add_patch(c)
        ax.text(x, i, lbl, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=4)
        if i > 0:
            ax.plot([layers_x[i-1], x], [i-1, i],
                    color=GRAY, lw=1.2, alpha=0.45, zorder=2)
    ax.set_xlim(-0.8, 1.8); ax.set_ylim(-0.8, n - 0.2)
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", color=GRAY, pad=3)
    if note:
        ax.text(0.5, -0.55, note, ha="center", transform=ax.transAxes,
                fontsize=8, color=RED)

# ── Main figure ───────────────────────────────────────────────────────────────
def make_figure():
    fig = plt.figure(figsize=(19, 11.5), facecolor=BG)
    fig.suptitle("4H-SiC Crystal Anisotropy  ·  Dicing Chipping Model",
                 fontsize=16, fontweight="bold", color=GRAY, y=0.99)

    gs = GridSpec(2, 4, figure=fig,
                  hspace=0.44, wspace=0.36,
                  left=0.04, right=0.97, top=0.93, bottom=0.05)

    angles = np.linspace(0, 360, 721)
    half   = np.linspace(0, 90,  361)

    # ── 1a: 4H stacking ───────────────────────────────────────────────────────
    ax4h = fig.add_subplot(gs[0, 0])
    draw_stacking(ax4h,
                  [0.5, 0.9, 0.5, 0.1],
                  ["A", "B", "C", "B"],
                  [TEAL, GOLD, TEAL, GOLD],
                  "4H-SiC  ABCB\n(EV Power — DISCO main product)")
    # c-axis arrow
    ax4h.annotate("", xy=(1.55, 3.5), xytext=(1.55, -0.3),
                  arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax4h.text(1.65, 1.5, "c\n4.92 A", ha="left", va="center",
              fontsize=8, color=RED)
    ax4h.text(0.5, -0.72, "a = 3.08 A  |  off-axis 4deg\nue ~1000 cm2/Vs  best",
              ha="center", fontsize=8, color=GRAY)

    # ── 1b: 6H stacking ───────────────────────────────────────────────────────
    ax6h = fig.add_subplot(gs[1, 0])
    draw_stacking(ax6h,
                  [0.5, 0.9, 0.5, 0.1, 0.5, 0.9],
                  ["A", "B", "C", "A", "C", "B"],
                  [TEAL]*3 + [GRAY]*3,
                  "6H-SiC  ABCACB\n(RF GaN substrate / niche)")
    ax6h.text(0.5, -0.72,
              "ue ~400 cm2/Vs  anisotropy x100\nnot for EV  =>  DISCO cuts very little",
              ha="center", fontsize=8, color=RED)

    # ── 2: Young's modulus polar ──────────────────────────────────────────────
    ax_E = fig.add_subplot(gs[0, 1], projection="polar")
    _style_polar(ax_E, "Young's Modulus E(θ)  [GPa]")
    th_rad   = np.deg2rad(angles)
    E_polar  = mirror_polar(E_SiC,  angles)
    ax_E.plot(th_rad, E_polar, color=TEAL, lw=2.5)
    ax_E.fill(th_rad, E_polar, color=TEAL, alpha=0.13)
    ax_E.set_rlim(420, 490)
    ax_E.set_rticks([430, 450, 470, 490])
    ax_E.set_thetagrids([0, 90, 180, 270],
                        ["c [0001]", "a [11-20]", "−c", "−a"], fontsize=8)
    ax_E.annotate(f"Ec={E_SiC(0):.0f}",  xy=(0,         E_SiC(0)),
                  xytext=(np.deg2rad(22), 488), fontsize=8.5,
                  color=TEAL, fontweight="bold")
    ax_E.annotate(f"Ea={E_SiC(90):.0f}", xy=(np.pi/2, E_SiC(90)),
                  xytext=(np.deg2rad(68), 488), fontsize=8.5,
                  color=GOLD, fontweight="bold")

    # ── 3: K_IC polar ─────────────────────────────────────────────────────────
    ax_K = fig.add_subplot(gs[1, 1], projection="polar")
    _style_polar(ax_K, "Fracture Toughness K_IC(θ)  [MPa√m]")
    K_polar = mirror_polar(K_IC_SiC, angles)
    ax_K.plot(th_rad, K_polar, color=RED, lw=2.5)
    ax_K.fill(th_rad, K_polar, color=RED, alpha=0.13)
    ax_K.set_rlim(0.8, 3.1)
    ax_K.set_rticks([1.0, 1.5, 2.0, 2.5, 3.0])
    ax_K.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                        ["c", "", "a", "", "−c", "", "−a", ""], fontsize=8)
    ax_K.annotate("Basal\ncleavage\n(brittle)", xy=(0, 1.0),
                  xytext=(np.deg2rad(28), 1.15), fontsize=8,
                  color=RED, fontweight="bold")
    ax_K.annotate("Tough\nzone", xy=(np.pi/2, K_IC_SiC(90)),
                  xytext=(np.deg2rad(65), 2.95), fontsize=9,
                  color=TEAL, fontweight="bold")

    # ── 4: Chipping depth vs orientation ──────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_facecolor(BG)
    chip = chipping_um(half)
    si   = Si_chipping()
    ax_c.plot(half, chip, color=TEAL, lw=2.8, label="4H-SiC (Lawn-Evans)")
    ax_c.axhline(si,  color=GOLD, lw=2.0, ls="--", label=f"Si ref = {si:.1f} µm")
    ax_c.axhline(2.0, color=RED,  lw=1.8, ls=":",  label="DISCO target < 2 µm")
    ax_c.fill_between(half, 0, 2.0, alpha=0.07, color=TEAL)

    best  = np.argmin(chip)
    worst = np.argmax(chip)
    ax_c.scatter(half[best],  chip[best],  s=130, color=TEAL,  zorder=5)
    ax_c.scatter(half[worst], chip[worst], s=130, color=RED, marker="X", zorder=5)
    ax_c.annotate(f"Best θ={half[best]:.0f}°\nh={chip[best]:.2f}µm",
                  xy=(half[best], chip[best]),
                  xytext=(half[best]+7, chip[best]+0.35),
                  fontsize=8.5, color=TEAL, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2))
    ax_c.annotate(f"Worst θ={half[worst]:.0f}°\nh={chip[worst]:.2f}µm",
                  xy=(half[worst], chip[worst]),
                  xytext=(half[worst]+5, chip[worst]-0.5),
                  fontsize=8.5, color=RED, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

    ax_c.set_xlabel("Crystal orientation θ from c-axis [deg]", fontsize=10, color=GRAY)
    ax_c.set_ylabel("Chipping depth [µm]", fontsize=10, color=GRAY)
    ax_c.set_title("Chipping Depth h_chip(θ)\n(depth=30µm, feed=1µm/rev, 25 krpm)",
                   fontsize=10, fontweight="bold", color=GRAY)
    ax_c.set_xlim(0, 90)
    ax_c.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax_c.set_xticklabels(
        ["0deg\nc-axis", "15", "30", "45deg\noff-axis\nstd.", "60", "75", "90deg\na-axis"],
        fontsize=7.5)
    ax_c.legend(fontsize=8, loc="upper right")
    ax_c.grid(True, alpha=0.28)
    ax_c.tick_params(colors=GRAY)
    for sp in ax_c.spines.values(): sp.set_color("#DDDDDD")

    # ── 5: Hexagonal wafer top-view ───────────────────────────────────────────
    ax_h = fig.add_subplot(gs[1, 2])
    ax_h.set_facecolor(BG)
    ax_h.set_aspect("equal")
    ax_h.axis("off")
    ax_h.set_title("Wafer Top View — Dicing Direction vs Chipping\n"
                   "(arc colour: green = low chipping, red = high)",
                   fontsize=10, fontweight="bold", color=GRAY)

    hex_ang = np.linspace(0, 2*np.pi, 7)
    hx = np.cos(hex_ang + np.pi/6)
    hy = np.sin(hex_ang + np.pi/6)
    ax_h.fill(hx, hy, alpha=0.10, color=TEAL)
    ax_h.plot(hx, hy, color=TEAL, lw=1.8)

    def arrow(ang_deg, length, color, label, lw=2.0):
        th = np.deg2rad(ang_deg)
        ax_h.annotate("", xy=(length*np.cos(th), length*np.sin(th)),
                      xytext=(0, 0),
                      arrowprops=dict(arrowstyle="->", color=color, lw=lw))
        ax_h.text(length*1.18*np.cos(th), length*1.18*np.sin(th), label,
                  ha="center", va="center", fontsize=8.5,
                  color=color, fontweight="bold")

    arrow(90,  0.75, GRAY, "[0001]\nc-axis")
    arrow(0,   0.75, TEAL, "[11-20]\na-axis\n* Best dicing", lw=2.5)
    arrow(30,  0.75, GOLD, "[10-10]\nprismatic")

    # 4° off-axis (standard wafer orientation)
    th4 = np.deg2rad(90 - 4)
    ax_h.annotate("", xy=(0.46*np.cos(th4), 0.46*np.sin(th4)),
                  xytext=(0, 0),
                  arrowprops=dict(arrowstyle="->", color=RED, lw=1.8, linestyle="dashed"))
    ax_h.text(0.57*np.cos(th4), 0.57*np.sin(th4),
              "4deg off\n(std. wafer)", ha="left", va="bottom",
              fontsize=8, color=RED)

    # Colour arc (chipping quality)
    th_arc = np.linspace(0, np.pi/2, 91)
    chip_arc = chipping_um(np.rad2deg(th_arc))
    vmin, vmax = 1.5, 5.5
    for i in range(len(th_arc) - 1):
        cv = np.clip((chip_arc[i] - vmin) / (vmax - vmin), 0, 1)
        col = plt.cm.RdYlGn_r(cv)
        r_arc = 0.86
        ax_h.plot([r_arc*np.cos(th_arc[i]), r_arc*np.cos(th_arc[i+1])],
                  [r_arc*np.sin(th_arc[i]), r_arc*np.sin(th_arc[i+1])],
                  color=col, lw=6, solid_capstyle="round", alpha=0.88)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_h, fraction=0.045, pad=0.04)
    cbar.set_label("h_chip [µm]", fontsize=9, color=GRAY)
    cbar.ax.tick_params(labelsize=8)

    ax_h.set_xlim(-1.45, 1.6)
    ax_h.set_ylim(-1.20, 1.45)

    # ── 6: Property comparison table ──────────────────────────────────────────
    ax_t = fig.add_subplot(gs[:, 3])
    ax_t.axis("off")
    ax_t.set_facecolor(BG)

    rows = [
        ["Property",           "4H-SiC",         "6H-SiC",       "Si"],
        ["Bandgap [eV]",       "3.26",            "3.03",         "1.12"],
        ["ue [cm2/Vs]",        "~1000",           "~400",         "1400"],
        ["u isotropy",         "Good",            "Poor x100",    "Isotropic"],
        ["Ec [GPa]",           "476",             "~501",         "130"],
        ["Ea [GPa]",           "448",             "~448",         "130"],
        ["K_IC [MPa*sqrtm]",   "1.0 - 2.8",      "1.0 - 2.8",   "0.70"],
        ["Hardness [GPa]",     "26 - 28",         "26 - 28",      "13"],
        ["Etchrate (KOH)",     "Resistant",       "Resistant",    "Fast"],
        ["EV power device",    "Best (main)",     "Not for EV",   "--"],
        ["GaN-on-X sub.",      "Good (growing)",  "Legacy",       "No"],
        ["DISCO cut vol.",     "Rapidly growing", "Almost none",  "High"],
        ["Blade wear rate",    "2x Si",           "2x Si",        "Baseline"],
    ]

    col_colours = (
        [["#C8F7F3", TEAL, "#F5E6F0", "#FFF8E7"]] +
        [["#FFFFFF", "#D5F5EE", "#FDECEA", "#FFFBF0"]] * (len(rows) - 1)
    )

    tbl = ax_t.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
        cellColours=col_colours[1:],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.52)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor(TEAL)
            cell.set_text_props(color="white", fontweight="bold")
        if c == 1 and r > 0:  # 4H column highlight
            cell.set_text_props(fontweight="bold", color=TEAL)

    ax_t.set_title("4H vs 6H vs Si: Key Properties",
                   fontsize=11, fontweight="bold", color=GRAY, pad=10)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="4H-SiC crystal anisotropy visualization")
    parser.add_argument("--save", action="store_true",
                        help="Save PDF to sic/figures/sic_anisotropy.pdf")
    args = parser.parse_args()

    fig = make_figure()

    if args.save:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "sic_anisotropy.pdf")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"Saved → {path}")
        path_png = path.replace(".pdf", ".png")
        fig.savefig(path_png, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"Saved → {path_png}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
