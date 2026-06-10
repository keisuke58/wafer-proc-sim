"""
Research methodology flow — reframed
Problem-centric (not TMCMC-centric):

Core thesis:  Bayesian inversion of anisotropic phase-field fracture
              from binary dicing observations via surrogate-accelerated SMC

Inference method (TMCMC / HMC / SMC / VI) = interchangeable TOOL.
Novel contributions = problem formulation:
  A. Binary likelihood from crack/no-crack observations (sign-only)
  B. Anisotropic Gc(theta) identification for semiconductor materials
  C. DeepONet/PINN surrogate to make Bayes inference tractable

Usage:
    python visualize_research_flow.py          # dark
    python visualize_research_flow.py --white  # white
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

WHITE_MODE = "--white" in sys.argv

DARK = dict(
    bg="#0D1117", col_bg="#161B22", text="#E6EDF3", sub="#8B949E",
    c0="#58A6FF", c1="#3FB950", c2="#F78166", novel="#D2A8FF",
    eq="#FFA657", border="#30363D",
)
LITE = dict(
    bg="#FFFFFF", col_bg="#F6F8FA", text="#24292F", sub="#57606A",
    c0="#0969DA", c1="#1A7F37", c2="#CF222E", novel="#8250DF",
    eq="#953800", border="#D0D7DE",
)
T = LITE if WHITE_MODE else DARK


def rbox(ax, x, y, w, h, fc, ec=None, alpha=0.9, lw=1.4, zorder=2, radius=0.14):
    p = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad={radius}",
                        facecolor=fc, edgecolor=ec or T["border"],
                        linewidth=lw, zorder=zorder, alpha=alpha)
    ax.add_patch(p)


def ctext(ax, x, y, w, h, lines, fsz=9, bold_top=True, color=None, zorder=5, dy=0):
    cx, cy = x + w/2, y + h/2 + dy
    lh = h / (len(lines) + 0.5)
    for i, line in enumerate(lines):
        yi = cy + (len(lines)-1-i)*lh*0.6 - (len(lines)-1)*lh*0.3
        ax.text(cx, yi, line, ha="center", va="center",
                fontsize=fsz, color=color or T["text"], zorder=zorder,
                fontweight="bold" if (bold_top and i == 0) else "normal")


def arr(ax, x1, y1, x2, y2, color, lw=1.8, label="", rad=0):
    style = f"arc3,rad={rad}" if rad else "arc3,rad=0"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=13,
                                connectionstyle=style), zorder=6)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.08, label, ha="center", va="bottom",
                fontsize=7, color=color, fontweight="bold",
                bbox=dict(fc=T["bg"], ec="none", alpha=0.85, pad=1.5),
                zorder=7)


# ── Canvas ────────────────────────────────────────────────────────────────────
FW, FH = 19, 13.5
fig, ax = plt.subplots(figsize=(FW, FH))
fig.patch.set_facecolor(T["bg"])
ax.set_facecolor(T["bg"])
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")


# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(FW/2, FH - 0.55,
        "Bayesian Inversion of Anisotropic Phase-Field Fracture",
        ha="center", va="center", fontsize=14.5, fontweight="bold",
        color=T["text"])
ax.text(FW/2, FH - 1.05,
        "from Binary Dicing Observations  |  Keio / Muramatsu Lab Thesis (2027-28)",
        ha="center", va="center", fontsize=9.5, color=T["sub"], style="italic")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FLOW: 4 column layout
# Col 0: Prior knowledge  Col 1: Forward model  Col 2: Observations
# Col 3: Bayesian inference  →  Output
# ══════════════════════════════════════════════════════════════════════════════

# Layout
BW, BH   = 3.8, 1.65     # standard box
GAP_X    = 0.55
GAP_Y    = 0.40
X_STARTS = [0.45, 0.45 + BW + GAP_X,
             0.45 + 2*(BW + GAP_X),
             0.45 + 3*(BW + GAP_X)]
Y_BOT    = 2.25
Y_MID    = Y_BOT + BH + GAP_Y
Y_TOP    = Y_MID + BH + GAP_Y


# ── Column headers ────────────────────────────────────────────────────────────
headers = [
    (T["c0"],    "Prior Knowledge",    "material physics + wafer-proc-sim"),
    (T["c2"],    "Forward Model",      "AT2 phase-field FEM (FEniCSx)"),
    (T["novel"], "Observations",       "binary crack / no-crack  [NOVEL A]"),
    (T["c1"],    "Bayesian Inference", "SMC / TMCMC / HMC  [tool, not goal]"),
]
for i, (col, title, sub) in enumerate(headers):
    hx = X_STARTS[i]
    hy = Y_TOP + BH + 0.28
    rbox(ax, hx, hy, BW, 0.68, fc=col, ec=col, alpha=0.2, lw=2.0)
    ax.plot([hx+0.1, hx+BW-0.1], [hy+0.65]*2, color=col, lw=2, zorder=4)
    ax.text(hx + BW/2, hy + 0.42, title,
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=col, zorder=5)
    ax.text(hx + BW/2, hy + 0.16, sub,
            ha="center", va="center", fontsize=7.8, color=T["sub"], zorder=5)


# ── ROW TOP: Theoretical foundations ─────────────────────────────────────────
# Col 0: material properties
rbox(ax, X_STARTS[0], Y_TOP, BW, BH, T["col_bg"])
ctext(ax, X_STARTS[0], Y_TOP, BW, BH,
      ["Anisotropic Fracture Properties",
       "SiC 4H:   K_IC(c) = 2.0,  K_IC(a) = 1.4",
       "Ga2O3:   K_IC(100) = 0.50  << SiC",
       "Crystal:  E, H, C11/C33/C44",
       "Source: wafer-proc-sim kernels"],
      fsz=8.6)

# Col 1: AT2 PDE
rbox(ax, X_STARTS[1], Y_TOP, BW, BH, T["col_bg"])
ctext(ax, X_STARTS[1], Y_TOP, BW, BH,
      ["AT2 Phase-Field PDE  (Miehe 2010)",
       "E[u,d] = int g(d)*psi+(eps) + Gc/2[d^2/l + l|grad d|^2]",
       "Anisotropic Gc(theta):  A(n) = I + beta*(I - n x n)",
       "Staggered solve: FEniCSx (dolfinx)",
       "params: Gc, ell, beta  (to infer)"],
      fsz=8.3, color=T["eq"])

# Col 2: Experimental data model
rbox(ax, X_STARTS[2], Y_TOP, BW, BH, T["col_bg"], ec=T["novel"], lw=2)
ctext(ax, X_STARTS[2], Y_TOP, BW, BH,
      ["[NOVEL A]  Binary Observation Model",
       "y_i in {0,1}: crack observed at (theta_i, v_i)?",
       "p(y=1|theta) = Phi( F_sim(theta)/sigma_noise )",
       "log L = sum y_i log Phi(z_i) + (1-y_i) log Phi(-z_i)",
       "N = 30-50 cuts  <<  continuous measurement"],
      fsz=8.5, color=T["novel"])

# Col 3: Inference
rbox(ax, X_STARTS[3], Y_TOP, BW, BH, T["col_bg"])
ctext(ax, X_STARTS[3], Y_TOP, BW, BH,
      ["Bayesian Calibration  (method-agnostic)",
       "p(Gc,beta,ell | y) ~ L(y|theta) * p(theta)",
       "Sampler options:  SMC  /  TMCMC  /  NUTS",
       "  (TMCMC = carry-over from LUH thesis)",
       "Goal: posterior, NOT a specific sampler"],
      fsz=8.5)

arr(ax, X_STARTS[0]+BW, Y_TOP+BH*0.55, X_STARTS[1], Y_TOP+BH*0.55, T["c0"],
    label="prior on Gc")
arr(ax, X_STARTS[1]+BW, Y_TOP+BH*0.55, X_STARTS[2], Y_TOP+BH*0.55, T["c2"],
    label="F_sim(theta)")
arr(ax, X_STARTS[2]+BW, Y_TOP+BH*0.55, X_STARTS[3], Y_TOP+BH*0.55, T["novel"],
    label="log L")


# ── ROW MID: Acceleration layer ───────────────────────────────────────────────
# Col 0: wafer-proc-sim as low-fi model
rbox(ax, X_STARTS[0], Y_MID, BW, BH, T["col_bg"])
ctext(ax, X_STARTS[0], Y_MID, BW, BH,
      ["Low-Fidelity Model  (wafer-proc-sim)",
       "Lawn-Evans:  h_chip ~ (E/H)^0.5 * F^0.5 / K_IC",
       "cos^2(theta) cleavage risk  (Ga2O3)",
       "O(us) per call  — warm-start for BO",
       "Calibrate high-fi on same obs data"],
      fsz=8.5)

# Col 1: Surrogate
rbox(ax, X_STARTS[1], Y_MID, BW, BH, T["col_bg"], ec=T["novel"], lw=2)
ctext(ax, X_STARTS[1], Y_MID, BW, BH,
      ["[NOVEL B]  DeepONet / PINN Surrogate",
       "Train: {theta_j, Gc_j} -> d(x) field from FEniCSx",
       "Eval: O(ms)  vs FEM O(min) per call",
       "Enables SMC with N=10^4 particles",
       "Error bound: ||d_NN - d_FEM|| < eps"],
      fsz=8.5, color=T["novel"])

# Col 2: Experimental design
rbox(ax, X_STARTS[2], Y_MID, BW, BH, T["col_bg"])
ctext(ax, X_STARTS[2], Y_MID, BW, BH,
      ["Optimal Experimental Design (OED)",
       "Which angles theta_i to test next?",
       "BOED:  max E_y[H(prior) - H(post|y)]",
       "Minimize cuts needed for inference",
       "Active learning on binary outcomes"],
      fsz=8.5)

# Col 3: Posterior analysis
rbox(ax, X_STARTS[3], Y_MID, BW, BH, T["col_bg"])
ctext(ax, X_STARTS[3], Y_MID, BW, BH,
      ["Posterior Analysis",
       "p(Gc, beta, ell | data)  — credible intervals",
       "Sensitivity: d log p / d theta via adjoint",
       "Model comparison:  isotropic vs anisotropic",
       "(Bayes factor, LOO from LUH thesis)"],
      fsz=8.5, color=T["c1"])

arr(ax, X_STARTS[0]+BW, Y_MID+BH*0.55, X_STARTS[1], Y_MID+BH*0.55, T["c0"],
    label="multi-fidelity")
arr(ax, X_STARTS[1]+BW, Y_MID+BH*0.55, X_STARTS[2], Y_MID+BH*0.55, T["novel"],
    label="fast surrogate")
arr(ax, X_STARTS[2]+BW, Y_MID+BH*0.55, X_STARTS[3], Y_MID+BH*0.55, T["c2"],
    label="active design")


# ── ROW BOT: Output / Design ──────────────────────────────────────────────────
# Col 0: SiC/Ga2O3 targets
rbox(ax, X_STARTS[0], Y_BOT, BW, BH, T["col_bg"])
ctext(ax, X_STARTS[0], Y_BOT, BW, BH,
      ["Target Materials",
       "4H-SiC:   DISCO KABRA stealth dicing",
       "beta-Ga2O3:  cleavage (100) at beta=103.8",
       "GaN-on-Si:   stealth at 1064nm",
       "AlN:   ceramic blade dicing Ra < 0.5 um"],
      fsz=8.5)

# Col 1: Validation
rbox(ax, X_STARTS[1], Y_BOT, BW, BH, T["col_bg"])
ctext(ax, X_STARTS[1], Y_BOT, BW, BH,
      ["Validation",
       "SENT/SENB benchmark: reproduce Miehe 2010",
       "Converge to known Gc (synthetic data test)",
       "Compare: isotropic / anisotropic posterior",
       "Cross-validate: hold-out angle theta"],
      fsz=8.5)

# Col 2: Novel A+B contribution box
rbox(ax, X_STARTS[2], Y_BOT, BW, BH, T["col_bg"], ec=T["novel"], lw=2.5)
ctext(ax, X_STARTS[2], Y_BOT, BW, BH,
      ["Novel Contribution Summary",
       "A: Binary obs -> probit likelihood  (new for PF)",
       "B: Anisotropic Gc(theta) from dicing data",
       "C: DeepONet accelerates SMC 100-1000x",
       "=> Noii 2022: continuous, isotropic, no surrogate"],
      fsz=8.5, color=T["novel"])

# Col 3: Engineering output
rbox(ax, X_STARTS[3], Y_BOT, BW, BH, T["col_bg"])
ctext(ax, X_STARTS[3], Y_BOT, BW, BH,
      ["Engineering Output",
       "Optimal street angle  theta*(Gc, beta, material)",
       "Confidence interval on crack risk",
       "Feed-speed / blade-grit trade-off map",
       "=> DISCO AP-lab process guideline"],
      fsz=8.5, color=T["c1"])

arr(ax, X_STARTS[0]+BW, Y_BOT+BH*0.55, X_STARTS[1], Y_BOT+BH*0.55, T["c0"])
arr(ax, X_STARTS[1]+BW, Y_BOT+BH*0.55, X_STARTS[2], Y_BOT+BH*0.55, T["c2"])
arr(ax, X_STARTS[2]+BW, Y_BOT+BH*0.55, X_STARTS[3], Y_BOT+BH*0.55, T["novel"])

# ── Vertical arrows ───────────────────────────────────────────────────────────
for col_i, col_c in enumerate([T["c0"], T["c2"], T["novel"], T["c1"]]):
    for yb, yt in [(Y_BOT, Y_MID), (Y_MID, Y_TOP)]:
        arr(ax,
            X_STARTS[col_i] + BW*0.5, yb + BH,
            X_STARTS[col_i] + BW*0.5, yt - 0.05,
            col_c, lw=1.5)

# ── Diagonal: low-fi informs surrogate training ───────────────────────────────
arr(ax,
    X_STARTS[0] + BW, Y_BOT + BH*0.55,   # bottom of col 0
    X_STARTS[1], Y_MID + BH*0.35,         # mid of col 1
    T["c0"], lw=1.4, rad=0.25,
    label="warm start")


# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM STRIP: thesis positioning vs Noii 2022
# ══════════════════════════════════════════════════════════════════════════════
strip_y  = 0.18
strip_x  = 0.45
strip_w  = FW - 0.9
strip_h  = 1.7

rbox(ax, strip_x, strip_y, strip_w, strip_h,
     T["novel"], ec=T["novel"], alpha=0.1, lw=2.0)

ax.text(strip_x + strip_w/2, strip_y + strip_h - 0.25,
        "Positioning vs Noii et al. (2022) CMAME — the closest prior work",
        ha="center", va="center", fontsize=9.5, fontweight="bold",
        color=T["novel"])

cols3 = [strip_x + 0.3, strip_x + strip_w*0.36, strip_x + strip_w*0.68]
widths3 = [strip_w*0.30, strip_w*0.30, strip_w*0.30]

for xi, wi, title, body, col in [
    (cols3[0], widths3[0],
     "Noii et al. 2022  (baseline)",
     "Bayes PF + TMCMC\nContinuous force obs\nIsotropic Gc\nNo surrogate",
     T["sub"]),
    (cols3[1], widths3[1],
     "This thesis  (delta)",
     "Binary obs (probit) [A]\nAnisotropic Gc(theta) [B]\nDeepONet surrog [C]\nSemiconductor dicing",
     T["novel"]),
    (cols3[2], widths3[2],
     "Journal target",
     "CMAME (IF 7.2)  1st choice\nComput. Mech. (IF 4.1)\nInt. J. Fract. (IF 2.4)\nECCOMAS 2028 conf.",
     T["c1"]),
]:
    ax.text(xi + wi/2, strip_y + strip_h - 0.55, title,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=col)
    ax.text(xi + wi/2, strip_y + strip_h - 1.18, body,
            ha="center", va="center", fontsize=8.2, color=T["text"],
            linespacing=1.5)

# dividers
for xd in [cols3[1] - 0.2, cols3[2] - 0.2]:
    ax.plot([xd]*2, [strip_y+0.12, strip_y+strip_h-0.12],
            color=T["novel"], lw=0.8, alpha=0.45)

arrow_x_bt = X_STARTS[3] + BW*0.5
arr(ax, arrow_x_bt, Y_BOT, arrow_x_bt, strip_y + strip_h,
    T["c1"], lw=1.6)

plt.tight_layout(pad=0)
suffix = "_white" if WHITE_MODE else ""
for ext in ("png", "pdf"):
    path = f"research_flow{suffix}.{ext}"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=T["bg"])
    print(f"Saved: {path}")
plt.show()
