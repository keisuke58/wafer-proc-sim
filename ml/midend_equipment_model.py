"""
Middle-End Semiconductor Equipment — DISCO-Centric Market Model.

中工程（前工程と後工程の間）: ウェーハ完成 → ダイ個片化 まで。

DISCOがアドレスする中工程セグメント:
  1. Backgrinding / thinning  バックグラインド（薄化研削）  DISCO ~55%
  2. Blade dicing             ブレードダイシング            DISCO >60%
  3. Laser / stealth dicing   レーザー/ステルスダイシング   DISCO ~45%

隣接セグメント（DISCO不参入、競合認識として重要）:
  4. Wafer probing            ウェーハプロービング（テスト）
  5. TSV / Hybrid bonding     TSV/ハイブリッドボンディング（3D IC）
  6. Dicing tape / frame      ダイシングテープ・フレーム（消耗品）

成長ドライバ:
  - HBM (High Bandwidth Memory): 積層 × 薄化 × ダイシング精度
  - SiC EV: ダイシング難度最高、DISCO優位
  - 3D IC (CoWoS / SoIC): TSV + 薄化需要 → 隣接市場拡大

Usage:
    python ml/midend_equipment_model.py
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
from scipy.integrate import solve_ivp

np.random.seed(7)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Segments ──────────────────────────────────────────────────────────────────

DISCO_SEGS   = ["Backgrinding\n/Thinning", "Blade\nDicing", "Laser/Stealth\nDicing"]
ADJACENT_SEGS= ["Wafer\nProbing", "TSV/Hybrid\nBonding", "Dicing Tape\n/Frame"]
ALL_SEGS     = DISCO_SEGS + ADJACENT_SEGS

# TAM 2024 [B USD]
DISCO_TAM    = np.array([1.5,  1.8,  0.9])
ADJACENT_TAM = np.array([2.0,  1.5,  0.8])
ALL_TAM      = np.hstack([DISCO_TAM, ADJACENT_TAM])

# CAGR 2024-2030
DISCO_CAGR    = np.array([0.22, 0.18, 0.28])   # thin-wafer + SiC + laser
ADJACENT_CAGR = np.array([0.12, 0.38, 0.10])   # TSV exploding from HBM

# DISCO market share 2024
DISCO_SHARE_2024 = np.array([0.55, 0.62, 0.45])

# ── Companies in DISCO core segments ─────────────────────────────────────────

COMPANIES = ["DISCO", "Accretech", "Han's Laser", "OKAMOTO", "Others"]
N_CO = len(COMPANIES)
COLORS = {
    "DISCO":      "#003087",
    "Accretech":  "#cc3300",
    "Han's Laser":"#ff8c00",
    "OKAMOTO":    "#228b22",
    "Others":     "#aaaaaa",
}

# Share matrix (company × DISCO_seg) — rows: DISCO,Accretech,Han's,OKAMOTO,Others
SHARE_2024 = np.array([
    #  Backgrind  Blade   Laser
    [0.55,       0.62,   0.45],   # DISCO
    [0.22,       0.22,   0.10],   # Accretech
    [0.03,       0.08,   0.30],   # Han's Laser
    [0.15,       0.02,   0.02],   # OKAMOTO (grinding specialist)
    [0.05,       0.06,   0.13],   # Others
])

# Growth rates r_i per segment
R_BASE = np.array([
    [0.05,  0.06,  0.10],   # DISCO: steady leader
    [0.04,  0.04,  0.06],   # Accretech
    [0.08,  0.10,  0.20],   # Han's: aggressive laser push
    [0.06,  0.02,  0.02],   # OKAMOTO: grinding focus
    [0.03,  0.03,  0.05],   # Others
])
K_SAT = 0.95

# ── HBM / 3D IC demand shock ──────────────────────────────────────────────────

def hbm_shock(t_yr: float) -> np.ndarray:
    """
    HBM + SiC compound demand shock on DISCO segment growth rates.
    HBM: peaks 2026-2027 (thin-wafer + laser scribing for TSV alignment)
    SiC: sustained through 2030 (blade dicing hard material)
    """
    yr = t_yr + 2024   # absolute year
    # HBM shock: Gaussian peak 2026.5
    hbm = 0.18 * np.exp(-0.5 * ((yr - 2026.5) / 1.5)**2)
    # SiC shock: linear ramp 2024→2028, plateau
    sic = 0.12 * min(1.0, (yr - 2024) / 4.0)
    # Per-segment impact:  backgrind, blade, laser
    return np.array([hbm + 0.05,   # backgrind: HBM thin-wafer dominant
                     sic  + 0.04,  # blade: SiC dominant
                     hbm  + 0.06]) # laser: HBM scribing dominant


# ── LV ODE (3 segments, 5 companies) ─────────────────────────────────────────

def lv_ode(t, S_flat, R, K):
    S = S_flat.reshape(3, N_CO)   # (seg, company)
    dS = np.zeros_like(S)
    shock = hbm_shock(t)
    for s in range(3):
        S_s = S[s]
        S_total = S_s.sum()
        for i in range(N_CO):
            r_eff = R[i, s] * (1 + shock[s])
            comp  = 0.08 * S_total   # mild competition — incumbent advantages persist
            dS[s, i] = r_eff * S_s[i] * (1 - S_total / K) - comp * S_s[i]
    return dS.ravel()


def simulate(years_ahead: int = 6):
    S0   = SHARE_2024.T.copy()   # (3 seg, 5 co)
    sol  = solve_ivp(lv_ode, [0, years_ahead], S0.ravel(),
                     args=(R_BASE, K_SAT), method="RK45",
                     dense_output=True, max_step=0.05)
    t_ev = np.linspace(0, years_ahead, 200)
    S_all = np.clip(sol.sol(t_ev).T.reshape(200, 3, N_CO), 0, 1)
    return t_ev + 2024, S_all


# ── DISCO revenue breakdown forecast ─────────────────────────────────────────

def disco_revenue_forecast(years: np.ndarray, S_all: np.ndarray,
                           n_mc: int = 500):
    N_yr = len(years)
    mc   = np.zeros((n_mc, N_yr, 3))
    for m in range(n_mc):
        for yi, yr in enumerate(years):
            dt   = yr - 2024
            tam  = DISCO_TAM * (1 + DISCO_CAGR)**dt
            share= S_all[yi, :, 0]   # DISCO shares across 3 segs
            noise= 1 + np.random.normal(0, 0.04, 3)
            mc[m, yi] = tam * share * noise
    return mc.mean(0), mc.std(0)   # (N_yr, 3)


# ── Technology radar for mid-end ──────────────────────────────────────────────

TECH_AXES = [
    "SiC\ndicing",
    "Thin wafer\n(<50µm)",
    "Laser\nscribing",
    "Grinding\nprecision",
    "Recipe\nAI/opt.",
    "Service\n& support",
]
N_AX = len(TECH_AXES)

TECH_SCORES = np.array([
    #SiC  Thin  Laser Grind  AI    Svc
    [9.5, 9.0,  7.5,  8.5,  6.5,  9.5],   # DISCO
    [6.5, 6.0,  4.0,  7.0,  4.0,  8.0],   # Accretech
    [4.5, 4.5,  9.0,  3.5,  4.0,  5.0],   # Han's Laser
    [4.0, 7.5,  2.0,  9.0,  3.0,  7.5],   # OKAMOTO
])
TECH_COS = ["DISCO", "Accretech", "Han's Laser", "OKAMOTO"]


# ── Market evolution summary ──────────────────────────────────────────────────

def tam_forecast(year: float) -> dict:
    dt = year - 2024
    disco_tam   = DISCO_TAM * (1 + DISCO_CAGR)**dt
    adj_tam     = ADJACENT_TAM * (1 + ADJACENT_CAGR)**dt
    disco_rev   = disco_tam * DISCO_SHARE_2024
    return {
        "disco_tam": disco_tam,
        "adj_tam": adj_tam,
        "disco_rev": disco_rev,
        "total_mid": disco_tam.sum() + adj_tam.sum(),
        "disco_total": disco_rev.sum(),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all(years, S_all, rev_mean, rev_std, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 15))
    gs  = fig.add_gridspec(3, 4, hspace=0.52, wspace=0.42)

    # ── Panel 1: Middle-end TAM by segment ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    x    = np.arange(len(ALL_SEGS))
    cols = ["#003087"]*3 + ["#aaccee"]*3
    bars = ax1.bar(x, ALL_TAM, color=cols, edgecolor="k", linewidth=0.7, alpha=0.88)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace("\n", " ") for s in ALL_SEGS],
                        rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("TAM 2024 [B USD]")
    ax1.set_title(f"(a) Middle-End Equipment TAM\n"
                  f"DISCO-addressable (dark): ${DISCO_TAM.sum():.1f}B  "
                  f"| Adjacent (light): ${ADJACENT_TAM.sum():.1f}B  "
                  f"| Total: ${ALL_TAM.sum():.1f}B")
    ax1.grid(alpha=0.25, axis="y")

    # CAGR annotations
    for xi, (tam, cagr) in enumerate(zip(ALL_TAM,
                                          np.hstack([DISCO_CAGR, ADJACENT_CAGR]))):
        ax1.text(xi, tam + 0.03, f"+{cagr*100:.0f}%/yr",
                 ha="center", fontsize=9, color="#003087" if xi < 3 else "gray",
                 fontweight="bold" if xi < 3 else "normal")

    p1 = mpatches.Patch(color="#003087", label="DISCO-addressable")
    p2 = mpatches.Patch(color="#aaccee", label="Adjacent (not yet)")
    ax1.legend(handles=[p1, p2], fontsize=9)

    # ── Panel 2: DISCO revenue breakdown forecast ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    seg_colors = ["#003087", "#cc3300", "#ff8c00"]
    seg_labels = ["Backgrinding", "Blade Dicing", "Laser/Stealth"]
    bottom = np.zeros(len(years))
    for s, (col, lab) in enumerate(zip(seg_colors, seg_labels)):
        mu  = rev_mean[:, s]
        ax2.fill_between(years, bottom, bottom + mu,
                         alpha=0.75, color=col, label=lab)
        bottom = bottom + mu

    ax2.plot(years, rev_mean.sum(1), "k-", lw=2.0, label="Total DISCO mid-end")
    ax2.fill_between(years,
                     rev_mean.sum(1) - rev_std.sum(1),
                     rev_mean.sum(1) + rev_std.sum(1),
                     alpha=0.15, color="k")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Revenue [B USD]")
    ax2.set_title("(b) DISCO Middle-End Revenue Forecast\n(stacked by segment, ±1σ MC)")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(alpha=0.25)

    # ── Panel 3: HBM/SiC demand shock magnitude ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    yrs_plot = np.linspace(2024, 2030, 100)
    shock_bg, shock_bl, shock_la = [], [], []
    for yr in yrs_plot:
        sh = hbm_shock(yr - 2024)
        shock_bg.append(sh[0]*100)
        shock_bl.append(sh[1]*100)
        shock_la.append(sh[2]*100)

    ax3.fill_between(yrs_plot, shock_bg, alpha=0.45, color="#003087",
                     label="Backgrinding (HBM thin-wafer)")
    ax3.fill_between(yrs_plot, shock_la, alpha=0.45, color="#ff8c00",
                     label="Laser dicing (HBM scribing)")
    ax3.fill_between(yrs_plot, shock_bl, alpha=0.45, color="#cc3300",
                     label="Blade dicing (SiC EV)")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Demand shock [% extra CAGR]")
    ax3.set_title("(c) HBM + SiC Demand Shock per Segment\n"
                  "HBM peaks 2026–27; SiC sustained through 2030")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.25)

    # ── Panel 4: Market share evolution (blade dicing) ────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:])
    seg_idx = 1   # Blade dicing
    for i, co in enumerate(COMPANIES):
        ax4.plot(years, S_all[:, seg_idx, i] * 100,
                 color=COLORS[co], lw=2.5 if co == "DISCO" else 1.5,
                 ls="-" if co in ("DISCO", "Han's Laser") else "--",
                 label=co)
    ax4.axvline(2024, color="gray", ls=":", lw=1.0, alpha=0.7)

    # Show laser dicing Han's threat
    seg_idx2 = 2   # Laser
    ax4b = ax4.twinx()
    ax4b.plot(years, S_all[:, seg_idx2, 2] * 100,  # Han's laser share
              color="#ff8c00", lw=1.5, ls="-.", alpha=0.7,
              label="Han's (laser)")
    ax4b.set_ylabel("Han's Laser dicing share [%]", color="#ff8c00", fontsize=10)
    ax4b.tick_params(axis="y", labelcolor="#ff8c00")
    ax4b.set_ylim(0, 60)

    ax4.set_xlabel("Year")
    ax4.set_ylabel("Blade Dicing Market Share [%]")
    ax4.set_title("(d) Blade Dicing Share + Han's Laser Threat\n"
                  "(dash-dot = Han's laser dicing share, right axis)")
    ax4.legend(loc="lower left", fontsize=9)
    ax4.grid(alpha=0.25)
    ax4.set_ylim(0, 75)

    # ── Panel 5: Technology radar ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2], projection="polar")
    angles = np.linspace(0, 2*np.pi, N_AX, endpoint=False).tolist()
    angles += angles[:1]

    for i, co in enumerate(TECH_COS):
        vals = TECH_SCORES[i].tolist() + [TECH_SCORES[i][0]]
        ax5.plot(angles, vals, color=COLORS[co],
                 lw=2.5 if co == "DISCO" else 1.5,
                 ls="-" if co == "DISCO" else "--",
                 label=co)
        ax5.fill(angles, vals, color=COLORS[co], alpha=0.06)

    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(TECH_AXES, fontsize=9)
    ax5.set_ylim(0, 10)
    ax5.set_yticks([2, 4, 6, 8, 10])
    ax5.set_yticklabels(["2","4","6","8","10"], fontsize=8)
    ax5.set_title("(e) Middle-End Technology Capability\n(DISCO vs rivals)", pad=18)
    ax5.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    # ── Panel 6: DISCO addressable market map ─────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.axis("off")

    rows = [
        ["Segment",          "TAM '24", "CAGR '30", "DISCO\nShare", "DISCO\nRev '24", "Threat"],
        ["Backgrinding",     "$1.5B",   "+22%",     "55%",          "$0.83B",         "OKAMOTO"],
        ["Blade dicing",     "$1.8B",   "+18%",     "62%",          "$1.12B",         "Accretech"],
        ["Laser/Stealth",    "$0.9B",   "+28%",     "45%",          "$0.41B",         "Han's Laser"],
        ["── total core ──", "$4.2B",   "+22%avg",  "–",            "$2.35B",         "–"],
        ["Wafer probing",    "$2.0B",   "+12%",     "0%",           "–",              "Advantest"],
        ["TSV/Hybrid bond",  "$1.5B",   "+38%",     "0%",           "–",              "ASMPT/BESI"],
        ["Dicing tape",      "$0.8B",   "+10%",     "0%",           "–",              "Furukawa/Lintec"],
        ["── total mid ──",  "$8.5B",   "–",        "–",            "~$2.35B",        "–"],
    ]

    tbl = ax6.table(
        cellText=[r[1:] for r in rows[1:]],
        colLabels=rows[0][1:],
        rowLabels=[r[0] for r in rows[1:]],
        cellLoc="center", loc="center",
        colWidths=[0.16]*5,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.42)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#003087")
            cell.set_text_props(color="white", fontweight="bold")
        elif r in (4, 9):   # total rows
            cell.set_facecolor("#dddddd")
            cell.set_text_props(fontweight="bold")
        elif 1 <= r <= 3:   # DISCO core
            cell.set_facecolor("#fff0cc")
        elif 5 <= r <= 7:   # adjacent
            cell.set_facecolor("#f0f4ff")
        if r == 0 and c < 0:
            cell.set_facecolor("#003087")

    ax6.set_title("(f) DISCO Middle-End Market Map", pad=10)

    fig.suptitle(
        "Middle-End Equipment — DISCO's Core Market & Adjacent Opportunities\n"
        r"Core addressable: \$4.2B (DISCO ~56% avg share)  |  "
        r"Total mid-end: \$8.5B  |  HBM + SiC dual demand shock",
        fontsize=13, fontweight="bold",
    )

    out_path = os.path.join(out_dir, "midend_equipment_model.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Figure → {out_path}")
    plt.close()
    return out_path


def print_summary(years, S_all, rev_mean):
    yr_2024 = np.argmin(np.abs(years - 2024))
    yr_2028 = np.argmin(np.abs(years - 2028))
    yr_2030 = np.argmin(np.abs(years - 2030))

    print("\n[DISCO revenue forecast — middle-end core]")
    seg_labels = ["Backgrinding", "Blade Dicing ", "Laser/Stealth"]
    for s, lab in enumerate(seg_labels):
        r24 = rev_mean[yr_2024, s]
        r28 = rev_mean[yr_2028, s]
        r30 = rev_mean[yr_2030, s]
        print(f"  {lab}: ${r24:.2f}B → ${r28:.2f}B → ${r30:.2f}B")
    print(f"  {'Total':12}: ${rev_mean[yr_2024].sum():.2f}B → "
          f"${rev_mean[yr_2028].sum():.2f}B → ${rev_mean[yr_2030].sum():.2f}B")

    print("\n[Middle-end TAM forecast]")
    for yr in [2024, 2027, 2030]:
        fc = tam_forecast(yr)
        print(f"  {yr}: DISCO-core ${fc['disco_tam'].sum():.1f}B  "
              f"adjacent ${fc['adj_tam'].sum():.1f}B  "
              f"DISCO rev ${fc['disco_total']:.2f}B")

    print("\n[Blade dicing share 2024 → 2030]")
    for i, co in enumerate(COMPANIES):
        s24 = S_all[yr_2024, 1, i]*100
        s30 = S_all[yr_2030, 1, i]*100
        print(f"  {co:<14}: {s24:.1f}% → {s30:.1f}%  ({s30-s24:+.1f}pp)")


def main():
    print("[Middle-End Equipment — DISCO-Centric Model]")

    print("\n[1] Simulating LV market share dynamics ...")
    years, S_all = simulate(years_ahead=6)

    print("[2] Monte Carlo revenue forecast (n=500) ...")
    rev_mean, rev_std = disco_revenue_forecast(years, S_all, n_mc=500)

    print_summary(years, S_all, rev_mean)

    print("\n[3] Plotting ...")
    plot_all(years, S_all, rev_mean, rev_std)
    print("Done.")


if __name__ == "__main__":
    main()
