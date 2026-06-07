"""
Front-End Semiconductor Equipment — Market Model & Competitive Simulation.

前工程装置（WFE: Wafer Fab Equipment）全体像と主要プレイヤー分析。
DISCOとの相対的な市場ポジションを定量化する。

Segments modelled:
  1. Lithography          (露光)          ASML dominant
  2. Deposition CVD/PVD   (成膜)          Applied Materials / Lam / TEL
  3. Etch                 (エッチング)    Lam Research / TEL
  4. CMP                  (平坦化)        Applied Materials / Ebara
  5. Inspection/Metrology (検査・計測)    KLA / Hitachi High-Tech
  6. Thermal / Cleaning   (熱処理/洗浄)  TEL / Screen

Competitive model:
  - Herfindahl-Hirschman Index (HHI) per segment → monopoly vs. competitive
  - Revenue CAGR from AI/HBM demand shock (2024-2028)
  - Front-end vs Back-end TAM comparison incl. DISCO position
  - Value chain margin flow: foundry capex → equipment revenue split

Usage:
    python ml/frontend_equipment_model.py
"""

import os
import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "figure.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

np.random.seed(0)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Front-end segments ────────────────────────────────────────────────────────

FE_SEGMENTS = [
    "Lithography",
    "Deposition\n(CVD/PVD/ALD)",
    "Etch",
    "CMP",
    "Inspection/\nMetrology",
    "Thermal/\nCleaning",
]
N_FE = len(FE_SEGMENTS)

# TAM 2024 [B USD]
FE_TAM_2024 = np.array([26.0, 24.0, 19.0, 5.5, 9.5, 7.0])
# CAGR 2024-2030 (AI/HBM/EUV ramp)
FE_CAGR     = np.array([0.14, 0.13, 0.12, 0.11, 0.15, 0.10])

# Back-end TAM for comparison
BE_SEGMENTS = ["Blade\nDicing", "Laser\nDicing", "Grinding", "Adv.\nPackaging",
               "Die\nBonding", "Wire\nBonding", "Test"]
BE_TAM_2024 = np.array([1.8, 0.9, 0.7, 3.2, 1.1, 2.4, 7.8])
BE_CAGR     = np.array([0.18, 0.28, 0.15, 0.32, 0.12, 0.08, 0.12])

# ── Front-end players ─────────────────────────────────────────────────────────

FE_COMPANIES = ["ASML", "Applied\nMaterials", "Lam\nResearch", "TEL", "KLA",
                "Nikon", "Others"]
N_FE_CO = len(FE_COMPANIES)

FE_COLORS = {
    "ASML":              "#003087",
    "Applied\nMaterials":"#cc3300",
    "Lam\nResearch":     "#ff8c00",
    "TEL":               "#228b22",
    "KLA":               "#9932cc",
    "Nikon":             "#666666",
    "Others":            "#aaaaaa",
}

# Market share per segment [fraction], rows=company, cols=segment
# Litho  Dep    Etch   CMP    Insp   Thermal
FE_SHARE = np.array([
    [0.90,  0.00,  0.00,  0.00,  0.00,  0.00],  # ASML
    [0.00,  0.35,  0.15,  0.38,  0.05,  0.08],  # Applied Materials
    [0.00,  0.30,  0.50,  0.05,  0.00,  0.05],  # Lam Research
    [0.00,  0.20,  0.18,  0.12,  0.05,  0.42],  # TEL
    [0.00,  0.00,  0.00,  0.05,  0.60,  0.00],  # KLA
    [0.08,  0.00,  0.00,  0.00,  0.00,  0.00],  # Nikon
    [0.02,  0.15,  0.17,  0.40,  0.30,  0.45],  # Others
])

# Revenue 2024 [B USD] — approximate
FE_REV_2024 = np.array([27.0, 26.5, 17.0, 14.0, 9.8, 2.1])  # excl. Others
FE_REV_CO   = FE_COMPANIES[:-1]

# Revenue CAGR per company 2024-2030
FE_REV_CAGR = np.array([0.18, 0.12, 0.11, 0.10, 0.14, 0.04])

# ── HHI (market concentration) ───────────────────────────────────────────────

def hhi(shares: np.ndarray) -> float:
    """Herfindahl-Hirschman Index. HHI > 0.25 = highly concentrated."""
    return float(np.sum(shares**2))


def segment_hhi():
    results = {}
    for s, seg in enumerate(FE_SEGMENTS):
        h = hhi(FE_SHARE[:, s])
        results[seg] = h
    # Back-end blade dicing: DISCO 62%, Accretech 22%, Han's 8%, others 8%
    results["Blade Dicing\n(DISCO)"] = hhi(np.array([0.62, 0.22, 0.08, 0.04, 0.04]))
    return results


# ── AI / HBM demand shock ─────────────────────────────────────────────────────

def hbm_shock(year: float) -> np.ndarray:
    """
    AI/HBM demand multiplier on front-end segment CAGR.
    Lithography and inspection benefit most (extreme precision for HBM TSV).
    """
    ramp = min(1.0, max(0.0, (year - 2024) / 3.0))   # ramp 2024→2027
    return np.array([0.06, 0.04, 0.05, 0.08, 0.10, 0.03]) * ramp


def revenue_forecast(years: np.ndarray, rev_2024: np.ndarray,
                     cagr: np.ndarray, shock_fn, n_mc: int = 500):
    N_yr = len(years)
    N_co = len(rev_2024)
    mc   = np.zeros((n_mc, N_yr, N_co))
    for m in range(n_mc):
        noise = 1 + np.random.normal(0, 0.04, (N_yr, N_co))
        for yi, yr in enumerate(years):
            dt     = yr - 2024
            shock  = shock_fn(yr)[:N_co] if N_co == N_FE else np.zeros(N_co)
            eff_cagr = cagr + shock
            mc[m, yi] = rev_2024 * (1 + eff_cagr)**dt * noise[yi]
    return mc.mean(0), mc.std(0)


# ── Value chain margin model ──────────────────────────────────────────────────

VALUE_CHAIN = {
    "Foundry capex\n(e.g. TSMC)":   {"value": 40.0, "margin": 0.50, "color": "#bbccee"},
    "Front-end\nequipment (WFE)":   {"value": 15.0, "margin": 0.25, "color": "#003087"},
    "Back-end\nequipment (ATME)":   {"value":  5.0, "margin": 0.30, "color": "#cc3300"},
    "  → DISCO\n     (dicing)":     {"value":  1.2, "margin": 0.35, "color": "#ff8c00"},
    "OSAT\n(packaging/test)":       {"value":  8.0, "margin": 0.12, "color": "#228b22"},
    "Chip OEM\n(fabless)":          {"value": 60.0, "margin": 0.45, "color": "#9932cc"},
}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    years = np.linspace(2024, 2030, 60)
    fig = plt.figure(figsize=(20, 15))
    gs  = fig.add_gridspec(3, 4, hspace=0.52, wspace=0.42)

    # ── Panel 1: Front vs Back TAM comparison ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    x_fe = np.arange(N_FE)
    w = 0.35

    b1 = ax1.bar(x_fe - w/2, FE_TAM_2024, w, color="#003087", alpha=0.85,
                 label="Front-end (WFE)")
    b2 = ax1.bar(x_fe + w/2, BE_TAM_2024[:N_FE], w, color="#cc3300", alpha=0.75,
                 label="Back-end (ATME)")

    ax1.set_xticks(x_fe)
    ax1.set_xticklabels(FE_SEGMENTS, fontsize=9)
    ax1.set_ylabel("TAM 2024 [B USD]")
    ax1.set_title(f"(a) Front-End vs Back-End Segment TAM\n"
                  f"WFE total: ${FE_TAM_2024.sum():.0f}B  |  "
                  f"ATME total: ${BE_TAM_2024.sum():.0f}B")
    ax1.legend()
    ax1.grid(alpha=0.25, axis="y")

    # Add total labels
    ax1.text(N_FE - 0.5, FE_TAM_2024.max() * 0.92,
             f"WFE: ${FE_TAM_2024.sum():.0f}B", color="#003087",
             fontsize=11, fontweight="bold")
    ax1.text(N_FE - 0.5, FE_TAM_2024.max() * 0.78,
             f"ATME: ${BE_TAM_2024.sum():.0f}B", color="#cc3300",
             fontsize=11, fontweight="bold")

    # ── Panel 2: HHI concentration per segment ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    hhi_dict = segment_hhi()
    seg_names = list(hhi_dict.keys())
    hhi_vals  = list(hhi_dict.values())
    bar_cols  = ["#003087"] * (len(seg_names) - 1) + ["#ff8c00"]
    bars = ax2.barh(seg_names, hhi_vals, color=bar_cols, alpha=0.85, edgecolor="k",
                    linewidth=0.5)
    ax2.axvline(0.25, color="red", ls="--", lw=1.5, label="HHI=0.25\n(concentrated)")
    ax2.axvline(0.15, color="orange", ls=":", lw=1.2, label="HHI=0.15\n(moderate)")
    ax2.set_xlabel("HHI (Herfindahl-Hirschman Index)")
    ax2.set_title("(b) Market Concentration per Segment\nHigher HHI = more monopolistic")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25, axis="x")
    for bar, val in zip(bars, hhi_vals):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", fontsize=10)

    # ── Panel 3: Revenue forecast — front-end leaders ─────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    rev_mean, rev_std = revenue_forecast(
        years, FE_REV_2024, FE_REV_CAGR, hbm_shock, n_mc=300)

    co_colors = list(FE_COLORS.values())[:6]
    for i, (co, col) in enumerate(zip(FE_REV_CO, co_colors)):
        mu  = rev_mean[:, i]
        sig = rev_std[:, i]
        lw  = 2.5 if co == "ASML" else 1.5
        ax3.plot(years, mu, color=col, lw=lw, label=co.replace("\n", " "))
        ax3.fill_between(years, mu-sig, mu+sig, alpha=0.10, color=col)

    ax3.set_xlabel("Year")
    ax3.set_ylabel("Revenue [B USD]")
    ax3.set_title("(c) Front-End Equipment Revenue Forecast\n(Monte Carlo ±1σ, AI/HBM shock)")
    ax3.legend(ncol=2, fontsize=9)
    ax3.grid(alpha=0.25)

    # ── Panel 4: DISCO position in value chain ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.axis("off")

    vc_items = list(VALUE_CHAIN.items())
    y_positions = np.linspace(0.90, 0.05, len(vc_items))
    bar_h = 0.10

    for (name, info), yp in zip(vc_items, y_positions):
        width = min(info["value"] / 70.0, 0.85)
        rect = mpatches.FancyBboxPatch(
            (0.05, yp - bar_h/2), width, bar_h,
            boxstyle="round,pad=0.005",
            facecolor=info["color"], edgecolor="k", linewidth=0.8,
            transform=ax4.transAxes, alpha=0.85,
        )
        ax4.add_patch(rect)
        ax4.text(0.05 + width/2, yp, f"{name}\n${info['value']:.0f}B  margin {info['margin']:.0%}",
                 ha="center", va="center", fontsize=8.5, color="white",
                 fontweight="bold", transform=ax4.transAxes)

    ax4.set_title("(d) Semiconductor Value Chain\n(revenue + operating margin)", pad=8)
    ax4.text(0.5, -0.02,
             "← bar width ∝ revenue   DISCO (orange) = dicing slice of ATME",
             ha="center", va="top", fontsize=9, transform=ax4.transAxes, color="gray")

    # ── Panel 5: CAGR comparison bubble chart ─────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    # x=TAM 2024, y=CAGR 2024-2030, bubble=HHI
    fe_hhi_vals = np.array([hhi(FE_SHARE[:, s]) for s in range(N_FE)])
    sc = ax5.scatter(FE_TAM_2024, FE_CAGR * 100, s=fe_hhi_vals * 3000,
                     c=range(N_FE), cmap="Blues", alpha=0.7,
                     edgecolors="#003087", linewidths=1.5, label="Front-end")
    for i, seg in enumerate(FE_SEGMENTS):
        ax5.annotate(seg.replace("\n", " "),
                     (FE_TAM_2024[i], FE_CAGR[i]*100),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)

    # DISCO dicing overlay
    disco_tam  = 1.8   # blade dicing TAM
    disco_cagr = 18.0  # %
    disco_hhi  = hhi(np.array([0.62, 0.22, 0.08, 0.04, 0.04]))
    ax5.scatter([disco_tam], [disco_cagr], s=disco_hhi*3000,
                color="#ff8c00", edgecolors="k", linewidths=1.5,
                zorder=5, label="DISCO (blade dicing)")
    ax5.annotate("DISCO\nblade dicing", (disco_tam, disco_cagr),
                 textcoords="offset points", xytext=(6, -14),
                 fontsize=9, color="#cc3300", fontweight="bold")

    ax5.set_xlabel("TAM 2024 [B USD]")
    ax5.set_ylabel("CAGR 2024–2030 [%]")
    ax5.set_title("(e) TAM vs. Growth Rate\n(bubble size = HHI market concentration)")
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.25)
    plt.colorbar(sc, ax=ax5, label="Segment index").ax.tick_params(labelsize=8)

    # ── Panel 6: Strategic summary table ─────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.axis("off")

    rows = [
        ["Metric",              "Front-end (WFE)",  "Back-end (ATME)",   "DISCO (dicing)"],
        ["TAM 2024",            "$91B",             "$18B",              "$1.8B"],
        ["CAGR 2024–30",        "12–15%",           "12–28%",            "18%"],
        ["DISCO share",         "0% (out)",         "~8% total",         ">60%"],
        ["Market structure",    "Oligopoly",        "Mixed",             "Near-monopoly"],
        ["AI/HBM upside",       "High (litho)",     "High (adv. pkg)",   "Moderate (SiC)"],
        ["Barrier to entry",    "Extreme",          "High",              "High"],
        ["DISCO strategy",      "—",                "Expand adv. pkg",   "Defend + grow"],
    ]

    tbl = ax6.table(
        cellText=[r[1:] for r in rows[1:]],
        colLabels=rows[0][1:],
        rowLabels=[r[0] for r in rows[1:]],
        cellLoc="center", loc="center",
        colWidths=[0.28, 0.28, 0.28],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#003087")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 2:   # DISCO column
            cell.set_facecolor("#fff0cc")
        if r == 0 and c < 0:
            cell.set_facecolor("#003087")
            cell.set_text_props(color="white")

    ax6.set_title("(f) Strategic Positioning Summary", pad=10)

    fig.suptitle(
        "Front-End Equipment Market — Context for DISCO's Competitive Position\n"
        "WFE ($91B) vs ATME ($18B): DISCO dominates its niche with >60% dicing share",
        fontsize=13, fontweight="bold",
    )

    out_path = os.path.join(out_dir, "frontend_equipment_model.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Figure → {out_path}")
    plt.close()
    return out_path


def print_summary():
    print(f"\n[TAM comparison]")
    print(f"  Front-end (WFE) total : ${FE_TAM_2024.sum():.0f}B")
    print(f"  Back-end (ATME) total : ${BE_TAM_2024.sum():.0f}B")
    print(f"  Ratio FE/BE           : {FE_TAM_2024.sum()/BE_TAM_2024.sum():.1f}×")

    print(f"\n[HHI per front-end segment]")
    hhi_dict = segment_hhi()
    for seg, h in hhi_dict.items():
        label = ("monopoly" if h > 0.6 else
                 "highly concentrated" if h > 0.25 else
                 "moderate" if h > 0.15 else "competitive")
        print(f"  {seg.replace(chr(10),' '):<28}: HHI={h:.3f}  ({label})")

    print(f"\n[Revenue 2024 — front-end leaders]")
    for co, rev in zip(FE_REV_CO, FE_REV_2024):
        print(f"  {co.replace(chr(10),' '):<20}: ${rev:.1f}B")

    print(f"\n[DISCO context]")
    disco_eff = 1.8 * 0.62   # effective revenue = TAM × share
    print(f"  Blade dicing TAM × share : ${disco_eff:.2f}B effective")
    print(f"  vs ASML revenue          : ${27.0:.1f}B (15× larger)")
    print(f"  but DISCO OP margin ~35% : competitive with any FE player")


def main():
    print("[Front-End Equipment Market Model]")
    print_summary()
    print("\n[Plotting 6-panel figure ...]")
    plot_all()
    print("Done.")


if __name__ == "__main__":
    main()
