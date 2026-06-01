"""
Sumco — Si ウェーハ製造モデル (Shin-Etsu との複占比較)
==========================================================
300mm/200mm CZ シリコン単結晶引き上げ → スライシング → 研磨 → 出荷。

Key physics:
  - CZ 引き上げ: Burton-Prim-Slichter 酸素濃度モデル
  - 酸素析出: Ostwald 熟成 → OISF (Oxidation Induced Stacking Fault)
  - ウェーハ反り/bow: 熱応力 × 厚さ × 直径
  - 研磨 CMP: Preston 則 → TTV (Total Thickness Variation)
  - 生産コスト: エネルギー × 材料 × 歩留まり
  - Sumco vs Shin-Etsu 市場シェア

実行:
    python fem/sumco_model.py
    python fem/sumco_model.py --crystal --diameter 300
    python fem/sumco_model.py --cost

参考文献:
    Burton, Prim & Slichter (1953) J. Chem. Phys. — CZ 不純物分配
    Tan & Tice (1976) Phil. Mag. — Si 中の酸素析出
    Timoshenko (1925) Phil. Mag. — ウェーハ反り
    Sumco / Shin-Etsu Annual Reports (2024)
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

WAFER_SPECS = {
    "300mm_P": {"D_mm": 300, "t_um": 775, "price_usd": 125, "segment": "Logic/DRAM",
                "color": "#2166ac", "ref": "Sumco 300mm prime (2024)"},
    "300mm_EE": {"D_mm": 300, "t_um": 775, "price_usd": 175, "segment": "EUV/Advanced",
                 "color": "#d62728", "ref": "Sumco 300mm EE (2024)"},
    "200mm_P": {"D_mm": 200, "t_um": 525, "price_usd": 50,  "segment": "Power/Analog",
                "color": "#2ca02c", "ref": "Sumco 200mm prime (2024)"},
    "150mm": {"D_mm": 150, "t_um": 400, "price_usd": 20, "segment": "SiC/GaN epi-ready",
              "color": "#ff7f0e", "ref": "Sumco 150mm (2023)"},
}

def bps_oxygen(pull_rate_mm_min: float = 1.5, rotation_rpm: float = 15.0,
               C_melt_ppma: float = 24.0) -> dict:
    """
    Burton-Prim-Slichter: CZ 結晶中の酸素濃度。
    C_solid = k_eff × C_melt;  k_eff = k0 / (k0 + (1-k0)×exp(-v·δ/D))
    k0(O) ≈ 1.0 (segregation coefficient Si-O), δ = 境界層厚さ
    δ ∝ 1/sqrt(rotation_rpm)
    参考: Burton et al. (1953)
    """
    k0 = 1.25   # Si-O effective segregation (slightly > 1)
    D_O_cm2s = 1e-4  # O diffusivity in Si melt [cm²/s]
    pull_cm_s = pull_rate_mm_min / 60.0 / 10.0
    delta_cm  = 0.5 / math.sqrt(max(rotation_rpm, 1.0))  # boundary layer
    exp_term  = math.exp(-pull_cm_s * delta_cm / max(D_O_cm2s, 1e-15))
    k_eff     = k0 / (k0 + (1.0 - k0) * exp_term)
    C_solid_ppma = k_eff * C_melt_ppma
    # ASTM: typical spec 10-18 ppma
    in_spec = 10.0 <= C_solid_ppma <= 18.0
    return {
        "pull_rate_mm_min": pull_rate_mm_min,
        "rotation_rpm": rotation_rpm,
        "k_eff": round(k_eff, 4),
        "O_concentration_ppma": round(C_solid_ppma, 2),
        "ASTM_spec_10_18_ppma": in_spec,
    }


def oxygen_precipitation(C_O_ppma: float = 14.0, anneal_T_C: float = 1000.0,
                          anneal_t_hr: float = 4.0) -> dict:
    """
    酸素析出 (Ostwald ripening):
    ΔC = C_O - C_eq;  C_eq = C0 × exp(-Ea / kT)
    析出密度 N_ppt ∝ ΔC × sqrt(t) × exp(-Ea2/kT)
    """
    k_B_eV = 8.617e-5; Ea_eq = 1.0; Ea_nuc = 2.5
    C_eq   = 5.0 * math.exp(-Ea_eq / (k_B_eV * (anneal_T_C + 273.15)))
    dC     = max(C_O_ppma - C_eq, 0.0)
    T_K    = anneal_T_C + 273.15
    N_ppt  = 1e10 * dC * math.sqrt(anneal_t_hr) * math.exp(-Ea_nuc / (k_B_eV * T_K))
    OISF_risk = N_ppt > 1e8
    return {
        "C_O_ppma": C_O_ppma, "T_C": anneal_T_C, "t_hr": anneal_t_hr,
        "C_eq_ppma": round(C_eq, 3),
        "N_precipitate_cm3": round(N_ppt, 3),
        "OISF_risk": OISF_risk,
    }


def wafer_warp(D_mm: float = 300.0, t_um: float = 775.0,
               delta_T_C: float = 50.0) -> dict:
    """
    熱応力 → ウェーハ反り (bow):
    bow = CTE × delta_T × D² / (8 × t)  (Timoshenko bimetal analogy)
    """
    CTE_Si = 2.6e-6   # [/K]
    bow_um = CTE_Si * delta_T_C * (D_mm * 1000)**2 / (8.0 * t_um * 1000.0) * 1e6
    TTV_nm = 50.0 + 0.1 * bow_um   # empirical TTV from bow
    spec_bow_um = 30.0 if D_mm >= 300 else 20.0
    return {
        "D_mm": D_mm, "t_um": t_um, "delta_T_C": delta_T_C,
        "bow_um": round(bow_um, 2),
        "TTV_nm": round(TTV_nm, 1),
        "spec_bow_um": spec_bow_um,
        "within_spec": bow_um <= spec_bow_um,
    }


def wafer_cost(spec: str = "300mm_P", yield_pct: float = 92.0) -> dict:
    p = WAFER_SPECS.get(spec, WAFER_SPECS["300mm_P"])
    energy_cost  = 15.0   # USD per wafer (CZ growth + polish)
    material_cost = 8.0
    overhead_cost = 20.0
    total_mfg = (energy_cost + material_cost + overhead_cost) / (yield_pct / 100.0)
    margin_pct = (p["price_usd"] - total_mfg) / p["price_usd"] * 100
    return {
        "spec": spec, "yield_pct": yield_pct,
        "mfg_cost_usd": round(total_mfg, 2),
        "ASP_usd": p["price_usd"],
        "gross_margin_pct": round(margin_pct, 1),
    }


def market_share() -> dict:
    return {
        "Sumco":       {"share_pct": 28.0, "rev_B": 3.1, "color": "#2166ac",
                        "HQ": "Japan"},
        "Shin-Etsu":   {"share_pct": 30.0, "rev_B": 3.3, "color": "#d62728",
                        "HQ": "Japan"},
        "Siltronic":   {"share_pct": 14.0, "rev_B": 1.8, "color": "#2ca02c",
                        "HQ": "Germany"},
        "SK Siltron":  {"share_pct": 13.0, "rev_B": 1.5, "color": "#ff7f0e",
                        "HQ": "Korea"},
        "GlobalWafers":{"share_pct": 15.0, "rev_B": 1.6, "color": "#9467bd",
                        "HQ": "Taiwan"},
    }


def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # Panel 0: CZ 酸素濃度 vs 引き上げ速度
    pull_arr = np.linspace(0.3, 3.0, 100)
    for rpm, col in [(10, "#2166ac"), (15, "#d62728"), (20, "#2ca02c"), (25, "#ff7f0e")]:
        O_arr = [bps_oxygen(v, rpm)["O_concentration_ppma"] for v in pull_arr]
        ax[0].plot(pull_arr, O_arr, color=col, lw=2.2, label=f"{rpm} rpm")
    ax[0].axhspan(10, 18, alpha=0.15, color="#2166ac", label="ASTM spec 10-18 ppma")
    ax[0].set_xlabel("引き上げ速度 [mm/min]"); ax[0].set_ylabel("O 濃度 [ppma]")
    ax[0].set_title("CZ Si 中の酸素濃度\n(Burton-Prim-Slichter モデル)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

    # Panel 1: 酸素析出密度 vs 焼鈍温度
    T_arr = np.linspace(700, 1200, 100)
    for C_O, col in [(12.0, "#2166ac"), (16.0, "#d62728"), (20.0, "#ff7f0e")]:
        N_arr = [max(oxygen_precipitation(C_O, T, 4.0)["N_precipitate_cm3"], 1e4)
                 for T in T_arr]
        ax[1].semilogy(T_arr, N_arr, color=col, lw=2.2, label=f"[O]={C_O} ppma")
    ax[1].axhline(1e8, color="k", ls="--", lw=1.5, label="OISF リスク閾値")
    ax[1].set_xlabel("焼鈍温度 [°C]"); ax[1].set_ylabel("析出密度 [cm⁻³]")
    ax[1].set_title("Si 酸素析出 vs 焼鈍温度\n(Ostwald 熟成, 4h)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

    # Panel 2: ウェーハ反り vs 直径
    D_arr = np.linspace(100, 450, 100)
    for dT, col in [(20, "#2166ac"), (50, "#d62728"), (100, "#ff7f0e")]:
        bow = [wafer_warp(D, 775 if D >= 300 else 525, dT)["bow_um"] for D in D_arr]
        ax[2].plot(D_arr, bow, color=col, lw=2.2, label=f"ΔT={dT}°C")
    ax[2].axhline(30.0, color="k", ls="--", lw=1.5, label="300mm spec 30µm")
    ax[2].set_xlabel("ウェーハ直径 [mm]"); ax[2].set_ylabel("Bow [µm]")
    ax[2].set_title("ウェーハ反り vs 直径\n(Timoshenko 熱応力モデル)")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.25)

    # Panel 3: ウェーハコスト vs 歩留まり
    yield_arr = np.linspace(70, 99, 100)
    for sp, col in [("300mm_P", "#2166ac"), ("300mm_EE", "#d62728"),
                     ("200mm_P", "#2ca02c")]:
        cost = [wafer_cost(sp, y)["mfg_cost_usd"] for y in yield_arr]
        ax[3].plot(yield_arr, cost, color=col, lw=2.2,
                   label=sp.replace("_", " "))
        ax[3].axhline(WAFER_SPECS[sp]["price_usd"], color=col, ls=":", lw=1.0)
    ax[3].set_xlabel("製造歩留まり [%]"); ax[3].set_ylabel("製造コスト [USD]")
    ax[3].set_title("ウェーハ製造コスト vs 歩留まり\n(点線: ASP)")
    ax[3].legend(fontsize=9); ax[3].grid(alpha=0.25)

    # Panel 4: Siウェーハ市場シェア
    ms = market_share()
    labels_m = list(ms.keys())
    shares_m  = [ms[k]["share_pct"] for k in labels_m]
    colors_m  = [ms[k]["color"] for k in labels_m]
    wedges, texts, autotexts = ax[4].pie(shares_m, labels=labels_m, colors=colors_m,
                                          autopct="%1.1f%%", startangle=90, pctdistance=0.75)
    for at in autotexts: at.set_fontsize(9)
    ax[4].set_title("Si ウェーハ市場シェア (2024)\nSumco vs Shin-Etsu 複占")

    # Panel 5: ウェーハ仕様比較テーブル
    ax[5].axis("off")
    col_labels = ["仕様", "直径", "厚さ\n[µm]", "ASP\n[USD]", "粗利\n[%]", "用途"]
    tdata = []
    for sp, wp in WAFER_SPECS.items():
        wc = wafer_cost(sp, 92.0)
        tdata.append([sp.replace("_", " "), f"{wp['D_mm']}mm",
                      f"{wp['t_um']}", f"${wp['price_usd']}",
                      f"{wc['gross_margin_pct']:.0f}%", wp["segment"]])
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center", bbox=[0.0, 0.20, 1.0, 0.75])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#003080"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#e8f0ff")
    ax[5].set_title("Sumco Si ウェーハ仕様比較\n300mm · 200mm · 150mm")

    fig.suptitle("Sumco — Si ウェーハ製造 物理モデル\n"
                 "CZ 酸素濃度 · 酸素析出 · ウェーハ反り · コスト · 市場シェア",
                 fontsize=12, fontweight="bold")
    out = os.path.join(out_dir, "sumco_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Sumco model → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sumco Si wafer model")
    ap.add_argument("--crystal",   action="store_true")
    ap.add_argument("--cost",      action="store_true")
    ap.add_argument("--diameter",  type=int, default=300)
    ap.add_argument("--no-plot",   action="store_true")
    args = ap.parse_args()
    if args.crystal:
        for v in [0.5, 1.0, 1.5, 2.0, 3.0]:
            r = bps_oxygen(v)
            print(f"  v={v:.1f} mm/min  [O]={r['O_concentration_ppma']:.2f} ppma  "
                  f"spec={r['ASTM_spec_10_18_ppma']}")
    if args.cost:
        for sp in WAFER_SPECS:
            r = wafer_cost(sp)
            print(f"  {sp:15s}  mfg=${r['mfg_cost_usd']:.2f}  "
                  f"ASP=${r['ASP_usd']}  margin={r['gross_margin_pct']:.1f}%")
    if not args.no_plot:
        plot_all()

if __name__ == "__main__":
    main()
