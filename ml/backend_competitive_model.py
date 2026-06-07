"""
Backend Process Equipment — Competitive Landscape Simulation.

後工程装置セグメント:
  1. Blade dicing           ブレードダイシング
  2. Laser / stealth dicing レーザー/ステルスダイシング
  3. Wafer grinding         ウェーハ薄化研削
  4. Advanced packaging     先進パッケージ (flip-chip / hybrid bonding)
  5. Die bonding            ダイボンディング
  6. Wire bonding           ワイヤーボンディング

競合動態モデル:
  Lotka-Volterra 型市場シェア競争:
    dS_i/dt = r_i S_i (1 - S_total/K) - Σ_j α_ij S_i S_j
  に外生的需要ショック (SiC/EV ブースト 2024–2028) を重畳。

技術ロードマップ差分:
  6次元能力スコア行列 → 企業間 Euclidean 距離 → 脅威度指数。

Usage:
    python ml/backend_competitive_model.py          # full simulation + plots
    python ml/backend_competitive_model.py --years 5  # 2025–2030 only
"""

import argparse
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

np.random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Segments ──────────────────────────────────────────────────────────────────

SEGMENTS = [
    "Blade\nDicing",
    "Laser/Stealth\nDicing",
    "Wafer\nGrinding",
    "Advanced\nPackaging",
    "Die\nBonding",
    "Wire\nBonding",
]
N_SEG = len(SEGMENTS)

# Total addressable market 2024 [billion USD] per segment
TAM_2024 = np.array([1.8, 0.9, 0.7, 3.2, 1.1, 2.4])
# CAGR 2024–2030 [%/yr] — SiC/EV and advanced packaging drive growth
TAM_CAGR = np.array([0.18, 0.28, 0.15, 0.32, 0.12, 0.08])

# ── Companies ─────────────────────────────────────────────────────────────────

COMPANIES = ["DISCO", "Accretech", "Han's Laser", "ASMPT", "BESI", "K&S"]
N_CO = len(COMPANIES)

COLORS = {
    "DISCO":      "#003087",
    "Accretech":  "#cc3300",
    "Han's Laser":"#ff8c00",
    "ASMPT":      "#228b22",
    "BESI":       "#9932cc",
    "K&S":        "#666666",
}

# ── Market share matrix 2024 (company × segment) [fraction 0–1] ───────────────
# Rows: DISCO, Accretech, Han's Laser, ASMPT, BESI, K&S
# Cols: Blade, Laser, Grinding, AdvPkg, DieBond, WireBond
SHARE_2024 = np.array([
    # Blade  Laser  Grind  AdvPkg DieBond WireBond
    [0.62,  0.45,  0.55,  0.05,  0.02,   0.01],  # DISCO
    [0.22,  0.10,  0.25,  0.02,  0.05,   0.02],  # Accretech
    [0.08,  0.30,  0.05,  0.01,  0.01,   0.01],  # Han's Laser
    [0.04,  0.08,  0.06,  0.35,  0.20,   0.40],  # ASMPT
    [0.01,  0.02,  0.02,  0.18,  0.45,   0.08],  # BESI
    [0.01,  0.02,  0.02,  0.05,  0.08,   0.38],  # K&S
])
# Others make up the remainder (not modelled explicitly)

# ── Lotka-Volterra parameters ─────────────────────────────────────────────────

# Base growth rates r_i [/yr] — competitive aggressiveness
R_BASE = np.array([
    # DISCO: steady leader, moderate growth appetite
    [0.06, 0.10, 0.05, 0.12, 0.08, 0.04],
    # Accretech: similar but weaker R&D
    [0.04, 0.06, 0.04, 0.05, 0.03, 0.02],
    # Han's Laser: aggressive in laser, price competition
    [0.12, 0.18, 0.04, 0.03, 0.02, 0.02],
    # ASMPT: dominant in advanced pkg, growing
    [0.02, 0.03, 0.02, 0.15, 0.10, 0.08],
    # BESI: niche die bonding
    [0.02, 0.02, 0.02, 0.12, 0.06, 0.03],
    # K&S: wire bonding specialist, slower growth
    [0.01, 0.01, 0.01, 0.04, 0.03, 0.02],
])

# Competition coefficients α_ij: how strongly company j suppresses company i
# Higher α = stronger competitive pressure from j on i
def build_alpha():
    """Build N_CO × N_CO competition matrix per segment."""
    # Default: moderate competition
    alpha = np.ones((N_SEG, N_CO, N_CO)) * 0.3
    np.einsum("sii->si", alpha)[:] = 0.0  # no self-competition

    # Han's Laser strongly competes with DISCO in laser (seg 1)
    alpha[1, 0, 2] = 0.9   # Han's → DISCO in laser
    alpha[1, 1, 2] = 0.7   # Han's → Accretech in laser

    # ASMPT dominates advanced packaging
    alpha[3, :, 3] = 0.6   # ASMPT competes hard in AdvPkg
    alpha[3, 3, :] = 0.2   # others less threat to ASMPT

    # BESI in die bonding
    alpha[4, :, 4] = 0.65
    alpha[4, 4, :] = 0.2

    return alpha


ALPHA = build_alpha()
K_SATURATION = 0.95   # total modelled share can reach 95% (5% = others)


# ── SiC demand shock ──────────────────────────────────────────────────────────

def sic_shock(t_yr: float) -> np.ndarray:
    """
    Exogenous demand multiplier for each segment due to SiC/EV boom.
    Peaks around 2026–2027, decays as market matures.
    Returns per-segment multiplier on growth rate.
    """
    # Gaussian shock centred at 2026.5 (relative to t=0 = 2020)
    t_peak = 6.5
    sigma  = 2.0
    magnitude = np.array([0.30, 0.20, 0.25, 0.15, 0.05, 0.02])
    return magnitude * np.exp(-0.5 * ((t_yr - t_peak) / sigma)**2)


# ── ODE system ────────────────────────────────────────────────────────────────

def lv_ode(t, S_flat, r, alpha, K):
    """
    Lotka-Volterra ODE for all companies in all segments.
    S_flat: (N_SEG × N_CO,) flattened state
    Returns dS/dt flattened.
    """
    S = S_flat.reshape(N_SEG, N_CO)
    dS = np.zeros_like(S)
    shock = sic_shock(t)

    for s in range(N_SEG):
        S_s = S[s]
        S_total = S_s.sum()
        for i in range(N_CO):
            competition = np.sum(alpha[s, i, :] * S_s)
            r_eff = r[i, s] * (1.0 + shock[s])
            dS[s, i] = (r_eff * S_s[i] * (1 - S_total / K)
                        - competition * S_s[i])
    return dS.ravel()


def simulate(t_span=(0, 10), n_pts=200):
    """
    Integrate LV ODE from 2020 (t=0) to 2020+t_span[1].
    Returns times [yr], share array (N_pts × N_SEG × N_CO).
    """
    S0 = SHARE_2024.T.copy()   # (N_SEG, N_CO) — ODE expects seg-first layout
    t0, tf = t_span

    sol = solve_ivp(
        lv_ode,
        t_span,
        S0.ravel(),
        args=(R_BASE, ALPHA, K_SATURATION),
        method="RK45",
        dense_output=True,
        max_step=0.05,
    )
    t_eval = np.linspace(t0, tf, n_pts)
    S_all  = sol.sol(t_eval).T.reshape(n_pts, N_SEG, N_CO)
    S_all  = np.clip(S_all, 0, 1)
    return t_eval + 2020, S_all   # absolute years


# ── Technology capability matrix ──────────────────────────────────────────────

TECH_AXES = [
    "SiC/GaN\ncapability",
    "AI / Recipe\noptimisation",
    "Thin wafer\n(<50 µm)",
    "Advanced\npackaging",
    "Price\ncompetitiveness",
    "Service &\nreliability",
]
N_AX = len(TECH_AXES)

# Score 0–10 per axis (10 = world-leading)
TECH_SCORES = np.array([
    # SiC  AI   Thin  AdvPkg Price Svc
    [9.0,  6.0, 8.5,  3.0,   6.0,  9.5],  # DISCO
    [6.5,  4.0, 6.0,  2.5,   7.0,  8.0],  # Accretech
    [5.0,  3.5, 4.0,  2.0,   9.5,  5.0],  # Han's Laser
    [4.0,  5.5, 5.5,  9.5,   6.5,  7.5],  # ASMPT
    [3.5,  4.0, 4.0,  8.0,   7.0,  7.0],  # BESI
    [3.0,  3.5, 3.5,  5.0,   7.5,  7.5],  # K&S
])

# Projected 2028 scores (R&D investment + market trends)
TECH_SCORES_2028 = np.array([
    [9.5,  8.5, 9.0,  5.0,   6.0,  9.5],  # DISCO: AI gap closing fast
    [7.0,  5.0, 6.5,  3.0,   7.0,  8.0],  # Accretech
    [6.0,  5.0, 4.5,  2.5,   9.5,  5.5],  # Han's: laser improving
    [5.0,  7.0, 6.0,  9.8,   6.5,  8.0],  # ASMPT: AdvPkg dominant
    [4.0,  5.0, 4.5,  8.5,   7.0,  7.5],  # BESI
    [3.5,  4.5, 4.0,  6.0,   7.5,  7.5],  # K&S
])


# ── Revenue forecast ──────────────────────────────────────────────────────────

def revenue_forecast(years, S_all, n_mc=500, noise_sigma=0.03):
    """
    Monte Carlo revenue forecast: TAM(yr) × share(yr) + Gaussian noise.
    Returns (N_yr, N_CO) mean and (N_yr, N_CO) std.
    """
    N_yr = len(years)
    rev_mc = np.zeros((n_mc, N_yr, N_CO))

    for mc in range(n_mc):
        for yi, yr in enumerate(years):
            dt = yr - 2024
            tam = TAM_2024 * (1 + TAM_CAGR) ** dt   # (N_SEG,)
            share = S_all[yi]                          # (N_SEG, N_CO)
            noise = 1 + np.random.normal(0, noise_sigma, (N_SEG, N_CO))
            rev   = (tam[:, None] * share * noise).sum(axis=0)  # (N_CO,)
            rev_mc[mc, yi] = rev

    return rev_mc.mean(0), rev_mc.std(0)


# ── Threat index ──────────────────────────────────────────────────────────────

def threat_index(S_all, years):
    """
    Threat index for DISCO (company 0): rate of share loss to each competitor.
    Returns (N_CO-1 × N_SEG) matrix at final year vs initial.
    """
    S_init = S_all[0]    # (N_SEG, N_CO) at start
    S_fin  = S_all[-1]
    # Gain of each competitor vs DISCO loss
    disco_loss = np.clip(S_init[:, 0] - S_fin[:, 0], 0, 1)   # (N_SEG,)
    comp_gain  = np.clip(S_fin[:, 1:] - S_init[:, 1:], 0, 1) # (N_SEG, N_CO-1)
    # Threat = geometric mean of DISCO loss and competitor gain
    threat = np.sqrt(disco_loss[:, None] * comp_gain + 1e-6)  # (N_SEG, N_CO-1)
    return threat.T   # (N_CO-1, N_SEG)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all(years, S_all, rev_mean, rev_std, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 14))
    gs  = fig.add_gridspec(3, 4, hspace=0.50, wspace=0.40)

    # ── Panel 1: Market share evolution — blade dicing (DISCO's core) ─────────
    ax1 = fig.add_subplot(gs[0, :2])
    seg_idx = 0   # Blade dicing
    for i, co in enumerate(COMPANIES):
        ax1.plot(years, S_all[:, seg_idx, i] * 100,
                 color=COLORS[co], lw=2.0 if co == "DISCO" else 1.3,
                 ls="-" if co == "DISCO" else "--",
                 label=co)
    ax1.axvline(2024, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax1.text(2024.1, 5, "2024\n(base)", fontsize=9, color="gray")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Market Share [%]")
    ax1.set_title("(a) Blade Dicing Market Share 2024–2030")
    ax1.legend(ncol=3, fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.set_ylim(0, 80)

    # ── Panel 2: Market share — laser dicing (Han's threat) ───────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    seg_idx = 1   # Laser/Stealth
    for i, co in enumerate(COMPANIES):
        ax2.plot(years, S_all[:, seg_idx, i] * 100,
                 color=COLORS[co], lw=2.0 if co in ("DISCO","Han's Laser") else 1.3,
                 ls="-" if co in ("DISCO","Han's Laser") else "--",
                 label=co)
    ax2.axvline(2024, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Market Share [%]")
    ax2.set_title("(b) Laser/Stealth Dicing Market Share\n(Han's Laser threat)")
    ax2.legend(ncol=3, fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.set_ylim(0, 65)

    # ── Panel 3: Technology radar — 2024 ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2], projection="polar")
    angles = np.linspace(0, 2 * np.pi, N_AX, endpoint=False).tolist()
    angles += angles[:1]

    plot_cos = ["DISCO", "Accretech", "Han's Laser", "ASMPT"]
    for co in plot_cos:
        idx = COMPANIES.index(co)
        vals = TECH_SCORES[idx].tolist() + [TECH_SCORES[idx][0]]
        ax3.plot(angles, vals, color=COLORS[co],
                 lw=2.0 if co == "DISCO" else 1.3,
                 ls="-" if co == "DISCO" else "--",
                 label=co)
        ax3.fill(angles, vals, color=COLORS[co], alpha=0.05)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(TECH_AXES, fontsize=9)
    ax3.set_ylim(0, 10)
    ax3.set_yticks([2, 4, 6, 8, 10])
    ax3.set_yticklabels(["2","4","6","8","10"], fontsize=8)
    ax3.set_title("(c) Technology Capability 2024", pad=18)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    # ── Panel 4: Technology radar — 2028 ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:], projection="polar")
    for co in plot_cos:
        idx = COMPANIES.index(co)
        vals = TECH_SCORES_2028[idx].tolist() + [TECH_SCORES_2028[idx][0]]
        ax4.plot(angles, vals, color=COLORS[co],
                 lw=2.0 if co == "DISCO" else 1.3,
                 ls="-" if co == "DISCO" else "--",
                 label=co)
        ax4.fill(angles, vals, color=COLORS[co], alpha=0.05)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(TECH_AXES, fontsize=9)
    ax4.set_ylim(0, 10)
    ax4.set_yticks([2, 4, 6, 8, 10])
    ax4.set_yticklabels(["2","4","6","8","10"], fontsize=8)
    ax4.set_title("(d) Technology Capability 2028\n(projected)", pad=18)
    ax4.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    # ── Panel 5: Revenue forecast — DISCO ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    disco_idx = 0
    mu  = rev_mean[:, disco_idx]
    sig = rev_std[:, disco_idx]
    ax5.plot(years, mu, color=COLORS["DISCO"], lw=2.5, label="DISCO (mean)")
    ax5.fill_between(years, mu - sig, mu + sig,
                     alpha=0.20, color=COLORS["DISCO"], label="±1σ MC")
    ax5.fill_between(years, mu - 2*sig, mu + 2*sig,
                     alpha=0.08, color=COLORS["DISCO"], label="±2σ MC")

    for co in ["Accretech", "Han's Laser", "ASMPT"]:
        idx = COMPANIES.index(co)
        ax5.plot(years, rev_mean[:, idx], color=COLORS[co],
                 lw=1.3, ls="--", label=co)

    ax5.axvline(2024, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax5.set_xlabel("Year")
    ax5.set_ylabel("Revenue [B USD]")
    ax5.set_title("(e) Revenue Forecast 2024–2030\n(Monte Carlo, n=500)")
    ax5.legend(ncol=2, fontsize=9)
    ax5.grid(alpha=0.25)

    # ── Panel 6: Threat heatmap ────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2:])
    threat = threat_index(S_all, years)   # (N_CO-1, N_SEG)
    im = ax6.imshow(threat, cmap="YlOrRd", aspect="auto",
                    vmin=0, vmax=threat.max())
    plt.colorbar(im, ax=ax6, label="Threat index")
    ax6.set_xticks(range(N_SEG))
    ax6.set_xticklabels([s.replace("\n", " ") for s in SEGMENTS], rotation=30,
                        ha="right", fontsize=9)
    ax6.set_yticks(range(N_CO - 1))
    ax6.set_yticklabels(COMPANIES[1:], fontsize=10)
    ax6.set_title("(f) Competitor Threat Heatmap\n(DISCO share erosion risk per segment)")
    for i in range(N_CO - 1):
        for j in range(N_SEG):
            ax6.text(j, i, f"{threat[i,j]:.2f}", ha="center", va="center",
                     fontsize=9, color="black" if threat[i,j] < 0.08 else "white")

    fig.suptitle(
        "Backend Process Equipment — Competitive Simulation: DISCO vs Rivals 2024–2030\n"
        "Lotka-Volterra market-share dynamics + SiC/EV demand shock",
        fontsize=13, fontweight="bold",
    )

    out_path = os.path.join(out_dir, "backend_competitive_model.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Figure → {out_path}")
    plt.close()
    return out_path


def print_summary(years, S_all, rev_mean):
    yr_2024 = np.argmin(np.abs(years - 2024))
    yr_2028 = np.argmin(np.abs(years - 2028))
    yr_2030 = np.argmin(np.abs(years - 2030))

    print("\n[Market share summary — Blade Dicing]")
    print(f"  {'Company':<14} {'2024':>8} {'2028':>8} {'2030':>8}  {'Δ2024→2030':>12}")
    for i, co in enumerate(COMPANIES):
        s24 = S_all[yr_2024, 0, i] * 100
        s28 = S_all[yr_2028, 0, i] * 100
        s30 = S_all[yr_2030, 0, i] * 100
        print(f"  {co:<14} {s24:>7.1f}% {s28:>7.1f}% {s30:>7.1f}%  {s30-s24:>+10.1f}pp")

    print("\n[Revenue forecast — all segments combined] [B USD]")
    print(f"  {'Company':<14} {'2024':>8} {'2028':>8} {'2030':>8}")
    for i, co in enumerate(COMPANIES):
        print(f"  {co:<14} {rev_mean[yr_2024,i]:>8.2f} "
              f"{rev_mean[yr_2028,i]:>8.2f} {rev_mean[yr_2030,i]:>8.2f}")

    print("\n[Technology gap — DISCO vs competitors (2024 → 2028)]")
    disco_24 = TECH_SCORES[0]
    disco_28 = TECH_SCORES_2028[0]
    print(f"  DISCO AI score: {disco_24[1]:.1f} → {disco_28[1]:.1f}  "
          f"(+{disco_28[1]-disco_24[1]:.1f} pts, largest gain)")
    for i, co in enumerate(COMPANIES[1:], 1):
        gap_24 = np.linalg.norm(TECH_SCORES[0] - TECH_SCORES[i])
        gap_28 = np.linalg.norm(TECH_SCORES_2028[0] - TECH_SCORES_2028[i])
        print(f"  DISCO vs {co:<12}: gap {gap_24:.2f} → {gap_28:.2f}  "
              f"({'widening' if gap_28 > gap_24 else 'narrowing'})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=10,
                        help="Simulation horizon from 2020 (default: 10 → 2030)")
    args = parser.parse_args()

    print("[後工程競合シミュレーション]")
    print(f"  Segments : {N_SEG}")
    print(f"  Companies: {N_CO}  ({', '.join(COMPANIES)})")
    print(f"  Horizon  : 2020–{2020+args.years}")

    print("\n[1] Running Lotka-Volterra simulation …")
    years, S_all = simulate(t_span=(4, 4 + args.years), n_pts=300)
    # t=4 → 2024 (base year)

    print("\n[2] Monte Carlo revenue forecast (n=500) …")
    rev_mean, rev_std = revenue_forecast(years, S_all, n_mc=500)

    print_summary(years, S_all, rev_mean)

    print("\n[3] Plotting …")
    plot_all(years, S_all, rev_mean, rev_std)
    print("\nDone.")


if __name__ == "__main__":
    main()
