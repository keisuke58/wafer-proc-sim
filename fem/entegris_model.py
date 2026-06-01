"""
Entegris — CMP スラリー / 特殊ガス / フィルタ サプライチェーン モデル
==========================================================
半導体製造材料・化学品の最大手。CMP スラリー、高純度特殊ガス、
フォトレジスト関連材料、フィルタ・精製システムをカバー。
AMAT CMP モデルへの材料入力として機能。

Key physics & economics:
  - CMP スラリー研磨粒子サイズ → MRR × 選択性 × ディフェクト
  - 高純度ガス不純物モデル: ppb レベル汚染 → デバイス特性劣化
  - フィルタ効率: Stokes 沈降 + 拡散 → 粒子捕集効率
  - 供給集中リスク: 特定材料の単一ソース依存度
  - 価格弾力性: 半導体サイクル × 在庫調整 → スラリー価格

実行:
    python fem/entegris_model.py
    python fem/entegris_model.py --slurry --material Cu
    python fem/entegris_model.py --gas --purity 6N

参考文献:
    Preston (1927) J. Soc. Glass Tech. — CMP 速度
    Iler (1979) The Chemistry of Silica — シリカ粒子
    Stokes (1851) Trans. Cambridge Phil. Soc. — 沈降速度
    Entegris Annual Report (2024)
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

SLURRY_PRODUCTS = {
    "iCue_Cu_ULSD": {
        "target": "Cu",
        "particle": "colloidal silica",
        "particle_d_nm": 70.0,
        "pH": 3.5,
        "H2O2_pct": 1.5,
        "MRR_base_nm_min": 580.0,
        "selectivity_Cu_SiO2": 30.0,
        "defect_density_cm2": 0.05,
        "price_USD_L": 45.0,
        "color": "#d62728",
        "ref": "Entegris iCue Cu ULSD slurry (2023)",
    },
    "iCue_W_Poly": {
        "target": "W",
        "particle": "alumina",
        "particle_d_nm": 120.0,
        "pH": 2.5,
        "H2O2_pct": 3.0,
        "MRR_base_nm_min": 400.0,
        "selectivity_Cu_SiO2": 25.0,
        "defect_density_cm2": 0.08,
        "price_USD_L": 35.0,
        "color": "#ff7f0e",
        "ref": "Entegris iCue W polishing slurry (2023)",
    },
    "Ceria_SiO2": {
        "target": "SiO2",
        "particle": "ceria",
        "particle_d_nm": 150.0,
        "pH": 7.0,
        "H2O2_pct": 0.0,
        "MRR_base_nm_min": 350.0,
        "selectivity_Cu_SiO2": 1.0,
        "defect_density_cm2": 0.12,
        "price_USD_L": 55.0,
        "color": "#2166ac",
        "ref": "Entegris ceria SiO2 slurry (2023)",
    },
}

SPECIALTY_GASES = {
    "WF6_5N5":  {"purity_N": 5.5, "impurity_ppb": 3.0,  "price_USD_kg": 450, "color": "#d62728"},
    "HCl_6N":   {"purity_N": 6.0, "impurity_ppb": 1.0,  "price_USD_kg": 180, "color": "#2166ac"},
    "NF3_6N":   {"purity_N": 6.0, "impurity_ppb": 1.0,  "price_USD_kg": 320, "color": "#2ca02c"},
    "SiH4_5N":  {"purity_N": 5.0, "impurity_ppb": 10.0, "price_USD_kg": 280, "color": "#ff7f0e"},
    "B2H6_5N":  {"purity_N": 5.0, "impurity_ppb": 10.0, "price_USD_kg": 1200,"color": "#9467bd"},
}

def slurry_mrr(product: str = "iCue_Cu_ULSD", pressure_kPa: float = 30.0,
               velocity_m_s: float = 0.5, dilution: float = 1.0) -> dict:
    """
    スラリー MRR: Preston + 粒子サイズ補正。
    MRR = Kp × P × V × (d_particle / d_ref)^0.3 × (1/dilution)^0.5
    """
    p = SLURRY_PRODUCTS.get(product, SLURRY_PRODUCTS["iCue_Cu_ULSD"])
    Kp = p["MRR_base_nm_min"] / (30.0 * 0.5 * 1.0)   # back-calculate from base
    d_ref = 100.0   # reference particle size [nm]
    particle_factor = (p["particle_d_nm"] / d_ref) ** 0.3
    dilution_factor = dilution ** (-0.5)
    MRR = Kp * pressure_kPa * velocity_m_s * particle_factor * dilution_factor
    defects_adj = p["defect_density_cm2"] * (p["particle_d_nm"] / d_ref) ** 0.5
    return {
        "product": product, "pressure_kPa": pressure_kPa,
        "MRR_nm_min": round(MRR, 1),
        "selectivity": p["selectivity_Cu_SiO2"],
        "defect_density_cm2": round(defects_adj, 4),
        "slurry_cost_per_wafer_USD": round(p["price_USD_L"] * 0.25 / dilution, 2),
    }


def gas_purity_impact(gas: str = "WF6_5N5", process: str = "CVD_W") -> dict:
    """
    ガス不純物 → デバイス影響。
    O2 impurity in WF6 → TiN oxidation → contact resistance 増加
    impurity_ppm → sheet resistance change ΔRs ∝ impurity_ppb^0.5
    """
    p = SPECIALTY_GASES.get(gas, SPECIALTY_GASES["WF6_5N5"])
    dRs_pct = 0.5 * math.sqrt(p["impurity_ppb"])   # 経験式
    process_sensitivity = {"CVD_W": 1.5, "ALD": 2.0, "Etch": 0.5, "PECVD": 1.0}
    sens = process_sensitivity.get(process, 1.0)
    impact = dRs_pct * sens
    return {
        "gas": gas, "process": process,
        "purity_N": p["purity_N"],
        "impurity_ppb": p["impurity_ppb"],
        "sheet_resistance_delta_pct": round(impact, 3),
        "yield_impact_ppm": round(impact * 10, 1),
        "acceptable": impact < 0.5,
    }


def filter_efficiency(particle_d_nm: float = 20.0,
                       filter_type: str = "UPE_membrane") -> dict:
    """
    フィルタ粒子捕集効率: Stokes + 拡散 複合モデル。
    大粒子: 慣性衝突 η_inertial ∝ (d/d_fiber)^2
    小粒子: 拡散捕集 η_diffusion ∝ 1/d^(2/3)
    MPPS (Most Penetrating Particle Size): d ≈ 100nm

    参考: Stokes (1851); Liu & Rubow (1990) Aerosol Sci.
    """
    filter_params = {
        "UPE_membrane":  {"d_fiber_nm": 200.0,  "base_eff": 99.9999},
        "PTFE_membrane": {"d_fiber_nm": 500.0,  "base_eff": 99.999},
        "Nylon_depth":   {"d_fiber_nm": 1000.0, "base_eff": 99.99},
    }
    fp = filter_params.get(filter_type, filter_params["UPE_membrane"])
    # MPPS at ~100nm: worst case efficiency
    MPPS = 100.0
    if particle_d_nm < MPPS:
        eta = fp["base_eff"] * (particle_d_nm / MPPS) ** (2.0/3.0)
    else:
        eta = fp["base_eff"]
    eta = min(eta, fp["base_eff"])
    return {
        "filter_type": filter_type,
        "particle_d_nm": particle_d_nm,
        "efficiency_pct": round(eta, 4),
        "LRV": round(math.log10(100.0 / max(100.0 - eta, 0.0001)), 2),  # log reduction value
    }


def supply_concentration_risk() -> dict:
    materials = {
        "EUV_resist_PAG":  {"suppliers": 2, "top_share_pct": 70, "alt_time_yr": 3},
        "CMP_ceria":        {"suppliers": 3, "top_share_pct": 50, "alt_time_yr": 2},
        "WF6_ultrahigh":    {"suppliers": 2, "top_share_pct": 65, "alt_time_yr": 4},
        "HF_ultrapure":     {"suppliers": 4, "top_share_pct": 35, "alt_time_yr": 1},
        "NF3_cleaning":     {"suppliers": 3, "top_share_pct": 45, "alt_time_yr": 1},
    }
    for k, v in materials.items():
        HHI = (v["top_share_pct"] / 100) ** 2 + (1 - v["top_share_pct"] / 100) ** 2 / max(v["suppliers"] - 1, 1)
        v["HHI"] = round(HHI, 3)
        v["risk_score"] = round(HHI * v["alt_time_yr"], 2)
    return materials


def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # Panel 0: CMP MRR vs 圧力
    P_arr = np.linspace(5, 60, 100)
    for prod, col in [(p, SLURRY_PRODUCTS[p]["color"]) for p in SLURRY_PRODUCTS]:
        mrr = [slurry_mrr(prod, P)["MRR_nm_min"] for P in P_arr]
        ax[0].plot(P_arr, mrr, color=col, lw=2.2,
                   label=f"{prod.replace('_',' ')}")
    ax[0].set_xlabel("圧力 [kPa]"); ax[0].set_ylabel("MRR [nm/min]")
    ax[0].set_title("Entegris スラリー CMP MRR vs 圧力\n(v=0.5 m/s, Preston モデル)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

    # Panel 1: スラリー粒径 → MRR × 欠陥
    d_arr = np.linspace(20, 300, 100)
    for prod, col in [(p, SLURRY_PRODUCTS[p]["color"]) for p in SLURRY_PRODUCTS]:
        orig_d = SLURRY_PRODUCTS[prod]["particle_d_nm"]
        mrr_arr = []
        for d in d_arr:
            SLURRY_PRODUCTS[prod]["particle_d_nm"] = d
            mrr_arr.append(slurry_mrr(prod)["MRR_nm_min"])
        SLURRY_PRODUCTS[prod]["particle_d_nm"] = orig_d
        ax[1].plot(d_arr, mrr_arr, color=col, lw=2.2, label=prod.replace("_", " "))
    ax[1].set_xlabel("粒子径 [nm]"); ax[1].set_ylabel("MRR [nm/min]")
    ax[1].set_title("スラリー粒子径 → MRR\n(粒径 補正係数 d^0.3)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

    # Panel 2: フィルタ効率 vs 粒子径 (MPPS)
    d_filter_arr = np.logspace(0, 3, 200)
    for ft, col in [("UPE_membrane", "#2166ac"), ("PTFE_membrane", "#d62728"),
                     ("Nylon_depth", "#2ca02c")]:
        eff = [filter_efficiency(d, ft)["efficiency_pct"] for d in d_filter_arr]
        ax[2].semilogx(d_filter_arr, eff, color=col, lw=2.2,
                       label=ft.replace("_", " "))
    ax[2].axvline(100, color="k", ls="--", lw=1.5, label="MPPS ~100nm")
    ax[2].set_xlabel("粒子径 [nm]"); ax[2].set_ylabel("捕集効率 [%]")
    ax[2].set_title("フィルタ粒子捕集効率\n(Stokes+拡散, MPPS=100nm)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.25)

    # Panel 3: ガス純度 → プロセス影響
    gases = list(SPECIALTY_GASES.keys())
    impacts = [gas_purity_impact(g)["sheet_resistance_delta_pct"] for g in gases]
    colors_g = [SPECIALTY_GASES[g]["color"] for g in gases]
    ax[3].bar(range(len(gases)), impacts, color=colors_g, width=0.6, alpha=0.85)
    ax[3].axhline(0.5, color="k", ls="--", lw=1.5, label="0.5% 許容上限")
    ax[3].set_xticks(range(len(gases)))
    ax[3].set_xticklabels([g.replace("_", "\n") for g in gases], fontsize=8)
    ax[3].set_ylabel("シート抵抗変化 [%]")
    ax[3].set_title("特殊ガス不純物 → デバイス影響\n(CVD/ALD プロセス)")
    ax[3].legend(fontsize=9); ax[3].grid(axis="y", alpha=0.25)

    # Panel 4: 供給集中リスク
    risk = supply_concentration_risk()
    materials_r = list(risk.keys())
    hhi_arr = [risk[m]["HHI"] for m in materials_r]
    rs_arr  = [risk[m]["risk_score"] for m in materials_r]
    x = np.arange(len(materials_r))
    ax4b = ax[4].twinx()
    ax[4].bar(x, hhi_arr, color="#d62728", alpha=0.7, width=0.5, label="HHI")
    ax4b.plot(x, rs_arr, "D--", color="#2166ac", ms=8, lw=1.8, label="リスクスコア")
    ax[4].set_xticks(x); ax[4].set_xticklabels([m.replace("_", "\n") for m in materials_r], fontsize=7.5)
    ax[4].set_ylabel("HHI", color="#d62728")
    ax4b.set_ylabel("リスクスコア (HHI×代替期間)", color="#2166ac")
    ax[4].set_title("半導体材料 供給集中リスク\n(HHI × 代替調達期間)")
    ax[4].grid(axis="y", alpha=0.25)

    # Panel 5: スラリー仕様比較テーブル
    ax[5].axis("off")
    col_labels = ["スラリー", "対象", "粒子", "MRR\n[nm/min]", "選択比", "価格\n[$/L]"]
    tdata = [[p.replace("_", " "), SLURRY_PRODUCTS[p]["target"],
              SLURRY_PRODUCTS[p]["particle"],
              f"{SLURRY_PRODUCTS[p]['MRR_base_nm_min']:.0f}",
              f"{SLURRY_PRODUCTS[p]['selectivity_Cu_SiO2']:.0f}×",
              f"${SLURRY_PRODUCTS[p]['price_USD_L']}"]
             for p in SLURRY_PRODUCTS]
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center", bbox=[0.0, 0.20, 1.0, 0.75])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a3a1a"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0fff0")
    ax[5].set_title("Entegris CMP スラリー 仕様\niCue Cu · W · Ceria SiO2")

    fig.suptitle("Entegris — CMP スラリー / 特殊ガス / フィルタ 材料モデル\n"
                 "スラリー MRR · ガス純度影響 · フィルタ効率 · 供給集中リスク",
                 fontsize=12, fontweight="bold")
    out = os.path.join(out_dir, "entegris_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Entegris model → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Entegris CMP slurry / specialty gas model")
    ap.add_argument("--slurry",   action="store_true")
    ap.add_argument("--gas",      action="store_true")
    ap.add_argument("--material", default="iCue_Cu_ULSD", choices=list(SLURRY_PRODUCTS.keys()))
    ap.add_argument("--purity",   default="WF6_5N5", choices=list(SPECIALTY_GASES.keys()))
    ap.add_argument("--no-plot",  action="store_true")
    args = ap.parse_args()
    if args.slurry:
        r = slurry_mrr(args.material)
        print(f"\n{args.material}: MRR={r['MRR_nm_min']} nm/min  "
              f"選択比={r['selectivity']}×  欠陥={r['defect_density_cm2']:.4f}/cm²")
    if args.gas:
        r = gas_purity_impact(args.purity)
        print(f"\n{args.purity}: {r['purity_N']}N  "
              f"ΔRs={r['sheet_resistance_delta_pct']:.3f}%  ok={r['acceptable']}")
    if not args.no_plot:
        plot_all()

if __name__ == "__main__":
    main()
