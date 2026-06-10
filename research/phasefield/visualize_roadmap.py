"""
Research roadmap visualization: 1D → 2D → 3D AT2 + Surrogate.
Outputs: research/phasefield/roadmap.pdf + roadmap.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.patheffects import withStroke

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":False,
})

# Palette
C = {
    "now":     "#1B4F72",   # dark blue  (current)
    "k1":      "#1A5276",   # mid blue   (Keio yr1)
    "k2":      "#0E6655",   # dark green (Keio yr2)
    "obs":     "#7D6608",   # amber
    "surr":    "#6E2F7A",   # purple
    "infer":   "#922B21",   # red
    "arrow":   "#566573",
    "phase":   "#ECF0F1",
    "head_now":"#1B4F72",
    "head_k1": "#1A5276",
    "head_k2": "#0E6655",
    "bg":      "#FDFEFE",
    "lite":    "#EBF5FB",
}

ALPHA_BOX = 0.13

# ── Layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor=C["bg"])
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.set_facecolor(C["bg"])
ax.set_xticks([]); ax.set_yticks([])


def box(ax, x, y, w, h, color, alpha=ALPHA_BOX, lw=1.5, ls="-", radius=0.25):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          linewidth=lw, linestyle=ls,
                          edgecolor=color, facecolor=color, alpha=alpha)
    ax.add_patch(rect)
    return rect


def txt(ax, x, y, s, color="black", size=9, bold=False, ha="center", va="center",
        alpha=1.0):
    w = "bold" if bold else "normal"
    return ax.text(x, y, s, ha=ha, va=va, fontsize=size,
                   color=color, fontweight=w, alpha=alpha)


def arrow(ax, x0, y0, x1, y1, color=C["arrow"], lw=2.0, head=0.3):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=f"->,head_width={head},head_length={head*0.7}",
                                color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"))


# ════════════════════════════════════════════════════════════════════════════
#  COLUMN HEADERS
# ════════════════════════════════════════════════════════════════════════════

col_x   = [1.2, 6.2, 11.2]   # left edge of each column
col_cx  = [3.3, 8.3, 13.3]   # center x
col_w   = 4.0
head_y  = 9.25
head_h  = 0.55

cols = [
    ("Phase 1  —  LUH Thesis (Now)",      C["head_now"]),
    ("Phase 2  —  Keio M1 (2027–28)",     C["head_k1"]),
    ("Phase 3  —  Keio M2 / Paper",       C["head_k2"]),
]

for (label, color), cx, lx in zip(cols, col_cx, col_x):
    box(ax, lx, head_y - 0.05, col_w, head_h + 0.1, color, alpha=0.85)
    txt(ax, cx, head_y + head_h/2, label, color="white", size=10.5, bold=True)


# ════════════════════════════════════════════════════════════════════════════
#  ROW LABELS  (left side)
# ════════════════════════════════════════════════════════════════════════════

row_labels = [
    (8.2, "Forward\nModel"),
    (6.5, "Observations"),
    (4.8, "Surrogate"),
    (3.2, "Inference"),
    (1.5, "Key\nOutputs"),
]
row_colors = [C["now"], C["obs"], C["surr"], C["infer"], "#444"]

for (ry, label), rc in zip(row_labels, row_colors):
    ax.text(0.6, ry, label, ha="center", va="center", fontsize=8.5,
            color=rc, fontweight="bold", style="italic",
            rotation=0)
    ax.plot([0.0, 1.0], [ry - 0.55, ry - 0.55],
            color="#CCCCCC", lw=0.5, ls="--")


# ════════════════════════════════════════════════════════════════════════════
#  HELPER: draw a cell block
# ════════════════════════════════════════════════════════════════════════════

def cell(ax, col_i, cy, lines, color, h=1.0, lw=1.5, ls="-", marker=None):
    lx = col_x[col_i]
    cx = col_cx[col_i]
    ybot = cy - h / 2
    box(ax, lx + 0.08, ybot, col_w - 0.16, h, color, alpha=ALPHA_BOX, lw=lw, ls=ls)
    n = len(lines)
    for k, (text, style) in enumerate(lines):
        dy = (k - (n - 1) / 2) * (h / (n + 0.3))
        if style == "bold":
            bold, size, fc = True,  9.5, color
        elif style == "note":
            bold, size, fc = False, 7.8, C["obs"]   # amber for key notes
        else:
            bold, size, fc = False, 8.5, "#333333"
        txt(ax, cx, cy - dy, text, color=fc, size=size, bold=bold)


# ════════════════════════════════════════════════════════════════════════════
#  ROW 1: FORWARD MODEL  (cy=8.2)
# ════════════════════════════════════════════════════════════════════════════

ry1 = 8.2
cell(ax, 0, ry1, [
    ("1D AT2 Phase-Field", "bold"),
    ("Thomas / tridiag FD", "norm"),
    ("n_elem=200,  ~5 ms/call", "norm"),
], C["now"], h=1.0)

cell(ax, 1, ry1, [
    ("2D Plane-Strain AT2", "bold"),
    ("FEniCS-dolfinx  (hex mesh)", "norm"),
    ("~0.5–2 s/call  (CPU)", "norm"),
], C["k1"], h=1.0)

cell(ax, 2, ry1, [
    ("3D AT2 (Full Wafer)", "bold"),
    ("FEniCS + GPU kernels", "norm"),
    ("~10–60 s/call  →  Surrogate", "norm"),
], C["k2"], h=1.0)


# ════════════════════════════════════════════════════════════════════════════
#  ROW 2: OBSERVATIONS  (cy=6.5)
# ════════════════════════════════════════════════════════════════════════════

ry2 = 6.5
cell(ax, 0, ry2, [
    ("Binary (sign-only) [Novel A]", "bold"),
    ("crack / no-crack  per angle θ", "norm"),
    ("Bernoulli-probit  p=Φ(−z)", "norm"),
    ("★ Gc₀ not identifiable alone", "note"),
], C["obs"], h=1.2)

cell(ax, 1, ry2, [
    ("Continuous: crack deflection", "bold"),
    ("measured deviation angle φ(θ)", "norm"),
    ("Gaussian obs  φ~N(f(params),σ²)", "norm"),
    ("→ Gc₀ + β jointly identifiable", "note"),
], C["obs"], h=1.2)

cell(ax, 2, ry2, [
    ("Multi-modal observations", "bold"),
    ("init_U  +  peak_F  +  φ_3D", "norm"),
    ("→ full (Gc₀, β, ℓ, E) identification", "norm"),
], C["obs"], h=1.2)


# ════════════════════════════════════════════════════════════════════════════
#  ROW 3: SURROGATE  (cy=4.8)
# ════════════════════════════════════════════════════════════════════════════

ry3 = 4.8
cell(ax, 0, ry3, [
    ("JAX MLP  [Novel C]", "bold"),
    ("(log Gc, log ℓ) → init_U", "norm"),
    ("13× speedup  |  3.5% rel err", "norm"),
], C["surr"], h=1.0)

cell(ax, 1, ry3, [
    ("Deeper MLP / GP emulator", "bold"),
    ("(Gc, β, ℓ) → (init_U, φ)", "norm"),
    ("~100× speedup  for SMC pilot", "norm"),
], C["surr"], h=1.0)

cell(ax, 2, ry3, [
    ("DeepONet / FNO  [Novel C+]", "bold"),
    ("(Gc₀, β, ℓ, E, θ) → (U, F, d-field)", "norm"),
    ("Operator learning  +  GPU batch", "norm"),
    ("enables 10⁴-particle TMCMC", "note"),
], C["surr"], h=1.2)


# ════════════════════════════════════════════════════════════════════════════
#  ROW 4: INFERENCE  (cy=3.2)
# ════════════════════════════════════════════════════════════════════════════

ry4 = 3.2
cell(ax, 0, ry4, [
    ("MAP  [Novel B]", "bold"),
    ("L-BFGS-B  5-restart", "norm"),
    ("identifies β (sign recovered)", "norm"),
], C["infer"], h=1.0)

cell(ax, 1, ry4, [
    ("TMCMC / SMC  [Novel D]", "bold"),
    ("posterior p(Gc, β | obs)", "norm"),
    ("credible intervals on anisotropy", "norm"),
], C["infer"], h=1.0)

cell(ax, 2, ry4, [
    ("Full Bayesian  on GPU  [Novel D+]", "bold"),
    ("4× RTX 4090  |  10⁴ particles", "norm"),
    ("p(Gc₀, β, ℓ, E | multi-modal obs)", "norm"),
    ("→ dicing recipe optimization", "note"),
], C["infer"], h=1.2)


# ════════════════════════════════════════════════════════════════════════════
#  ROW 5: KEY OUTPUTS  (cy=1.5)
# ════════════════════════════════════════════════════════════════════════════

ry5 = 1.5
cell(ax, 0, ry5, [
    ("LUH Masterarbeit  (Dec 2026)", "bold"),
    ("Proof-of-concept: binary obs model", "norm"),
    ("identifiability of β (anisotropy)", "norm"),
], "#444", h=1.0, lw=1.2, ls="--")

cell(ax, 1, ry5, [
    ("Keio M1 Interim Report", "bold"),
    ("2D plane-strain extension", "norm"),
    ("Composites B / CMAME paper", "norm"),
], "#444", h=1.0, lw=1.2, ls="--")

cell(ax, 2, ry5, [
    ("Keio Masterarbeit  (Mar 2028)", "bold"),
    ("3D + DeepONet surrogate + TMCMC", "norm"),
    ("Target: CMAME / IJNME / npj CM", "norm"),
], "#444", h=1.0, lw=1.2, ls="--")


# ════════════════════════════════════════════════════════════════════════════
#  VERTICAL PROGRESSION ARROWS between columns
# ════════════════════════════════════════════════════════════════════════════

for row_y in [ry1, ry2, ry3, ry4, ry5]:
    # col 0 → col 1
    arrow(ax, col_x[0] + col_w + 0.08, row_y,
               col_x[1] - 0.05,         row_y,
               color=C["arrow"], lw=2.0, head=0.25)
    # col 1 → col 2
    arrow(ax, col_x[1] + col_w + 0.08, row_y,
               col_x[2] - 0.05,         row_y,
               color=C["arrow"], lw=2.0, head=0.25)


# ════════════════════════════════════════════════════════════════════════════
#  WITHIN-COLUMN DOWN ARROWS (forward → obs → surr → infer → output)
# ════════════════════════════════════════════════════════════════════════════

row_ys = [ry1, ry2, ry3, ry4, ry5]
gaps   = [0.55, 0.55, 0.40, 0.40]
col_colors = [C["now"], C["k1"], C["k2"]]

for ci in range(3):
    cx = col_cx[ci]
    for ri in range(len(row_ys) - 1):
        y0 = row_ys[ri]   - gaps[ri]
        y1 = row_ys[ri+1] + gaps[ri]
        arrow(ax, cx, y0, cx, y1, color=col_colors[ci], lw=1.5, head=0.20)


# ════════════════════════════════════════════════════════════════════════════
#  "★ note" style — recolor the 4th line in obs cell
# ════════════════════════════════════════════════════════════════════════════
# (already handled by color parameter above — the "note" style gets amber)


# ════════════════════════════════════════════════════════════════════════════
#  LEGEND  (bottom)
# ════════════════════════════════════════════════════════════════════════════

legend_items = [
    (C["now"],   "Forward Model (Physics FEM)"),
    (C["obs"],   "Observation Model"),
    (C["surr"],  "Surrogate / Emulator"),
    (C["infer"], "Bayesian Inference"),
    ("#444",     "Deliverable / Paper"),
]
lx0 = 1.5
for k, (c, label) in enumerate(legend_items):
    xi = lx0 + k * 2.8
    ax.add_patch(mpatches.FancyBboxPatch(
        (xi - 0.15, 0.25), 0.28, 0.28,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor=c, edgecolor=c, alpha=0.7))
    ax.text(xi + 0.22, 0.38, label, ha="left", va="center", fontsize=8,
            color="#333")


# ════════════════════════════════════════════════════════════════════════════
#  TITLE
# ════════════════════════════════════════════════════════════════════════════

ax.text(8.0, 9.85,
        "AT2 Phase-Field Fracture × Bayesian Identification  —  Research Roadmap",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#1C2833")

# ════════════════════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════════════════════

out_dir = "research/phasefield"
for ext in ("pdf", "png"):
    path = f"{out_dir}/roadmap.{ext}"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    print(f"Saved: {path}")

plt.close(fig)
