"""
TSMC — N2/N3E/A16 プロセスノード + CoWoS-S/L + SoIC 積層モデル
==========================================================
世界最大ファウンドリ。GAA ナノシート (N2)、FinFET 改良 (N3E)、
CoWoS 先進パッケージ、SoIC 3D 積層の物理・コストモデル。

Key physics:
  - N2 GAA 電流 (Virtual Source + DIBL)
  - N3E FinFET 電流 (改良型 Virtual Source)
  - CoWoS-S/L インターポーザー面積・帯域幅モデル
  - SoIC (System on Integrated Chip) 3D 熱モデル
  - TSMC ダイコスト: 300mm ウェーハ → CPGD
  - WLP (Wafer Level Package) fan-out コストモデル

実行:
    python fem/tsmc_model.py
    python fem/tsmc_model.py --node N2 --pkg CoWoS-S
    python fem/tsmc_model.py --cost --node N3E

参考文献:
    TSMC N2 Technology Overview (2023 IEDM)
    TSMC Advanced Packaging Technology Brief (2024)
    Liu et al. (2022) IEEE ECTC — SoIC thermal model
    Stow et al. (2017) DAC — CoWoS cost model
    TSMC Annual Report (2024)
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

eps0 = 8.854e-12

# ════════════════════════════════════════════════════════════════════════════
# 1. プロセスノード パラメータ
# ════════════════════════════════════════════════════════════════════════════

PROCESS_NODES = {
    "N16": {
        "type": "FinFET", "node_nm": 16.0, "L_g_nm": 34.0,
        "n_fin": 3, "Hfin_nm": 42.0, "Wfin_nm": 8.0,
        "t_ox_nm": 1.5, "Vdd_V": 0.9, "density_MT_mm2": 47.0,
        "D0": 0.06, "wafer_cost_usd": 4000,
        "color": "#ff7f0e", "ref": "TSMC N16FF+ (2016)"},
    "N7": {
        "type": "FinFET", "node_nm": 7.0, "L_g_nm": 18.0,
        "n_fin": 3, "Hfin_nm": 52.0, "Wfin_nm": 6.0,
        "t_ox_nm": 1.2, "Vdd_V": 0.75, "density_MT_mm2": 91.0,
        "D0": 0.07, "wafer_cost_usd": 9346,
        "color": "#2166ac", "ref": "TSMC N7 (2018)"},
    "N3E": {
        "type": "FinFET+", "node_nm": 3.0, "L_g_nm": 12.0,
        "n_fin": 4, "Hfin_nm": 62.0, "Wfin_nm": 5.0,
        "t_ox_nm": 0.95, "Vdd_V": 0.75, "density_MT_mm2": 167.0,
        "D0": 0.085, "wafer_cost_usd": 20000,
        "color": "#d62728", "ref": "TSMC N3E (2023)"},
    "N2": {
        "type": "GAAFET", "node_nm": 2.0, "L_g_nm": 10.0,
        "n_sheet": 3, "W_ns_nm": 6.0,
        "t_ox_nm": 0.8, "Vdd_V": 0.7, "density_MT_mm2": 200.0,
        "D0": 0.09, "wafer_cost_usd": 30000,
        "color": "#2ca02c", "ref": "TSMC N2 (2025)"},
    "A16": {
        "type": "GAAFET+BacksidePDN", "node_nm": 1.6, "L_g_nm": 8.0,
        "n_sheet": 3, "W_ns_nm": 5.5,
        "t_ox_nm": 0.75, "Vdd_V": 0.65, "density_MT_mm2": 250.0,
        "D0": 0.10, "wafer_cost_usd": 40000,
        "color": "#9467bd", "ref": "TSMC A16 roadmap (2026)"},
}

PACKAGE_TYPES = {
    "CoWoS-S": {
        "interposer": "Si",
        "rdl_layers": 3,
        "bump_pitch_um": 130.0,
        "max_area_mm2": 1200.0,
        "cost_per_mm2_usd": 0.35,
        "BW_per_mm_TBs": 0.012,
        "color": "#2166ac",
        "ref": "TSMC CoWoS-S brief (2024)"},
    "CoWoS-L": {
        "interposer": "RDL_Local",
        "rdl_layers": 5,
        "bump_pitch_um": 40.0,
        "max_area_mm2": 3000.0,
        "cost_per_mm2_usd": 0.55,
        "BW_per_mm_TBs": 0.025,
        "color": "#d62728",
        "ref": "TSMC CoWoS-L brief (2024)"},
    "SoIC_X": {
        "interposer": "SoIC_bond",
        "rdl_layers": 8,
        "bump_pitch_um": 9.0,
        "max_area_mm2": 400.0,
        "cost_per_mm2_usd": 1.20,
        "BW_per_mm_TBs": 0.15,
        "color": "#2ca02c",
        "ref": "TSMC SoIC-X brief (2024)"},
    "InFO_UHD": {
        "interposer": "RDL_only",
        "rdl_layers": 4,
        "bump_pitch_um": 20.0,
        "max_area_mm2": 2500.0,
        "cost_per_mm2_usd": 0.40,
        "BW_per_mm_TBs": 0.018,
        "color": "#9467bd",
        "ref": "TSMC InFO-UHD brief (2024)"},
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Core Physics Functions
# ════════════════════════════════════════════════════════════════════════════

def transistor_current(node: str = "N2", Vgs: float = 0.7,
                        Vds: float = 0.65) -> dict:
    """
    FinFET (N7/N3E) または GAAFET (N2/A16) のドレイン電流。
    Virtual Source Model + DIBL。
    """
    p = PROCESS_NODES.get(node, PROCESS_NODES["N2"])
    t_ox = p["t_ox_nm"] * 1e-9
    Cox  = 3.9 * eps0 / t_ox
    Vt   = 0.20

    if p["type"] in ("GAAFET", "GAAFET+BacksidePDN"):
        n_s  = p.get("n_sheet", 3)
        W_ch = p.get("W_ns_nm", 6.0) * 1e-9
        L_g  = p["L_g_nm"] * 1e-9
        DIBL = 0.025   # V/V for GAA
    else:  # FinFET
        n_s  = p.get("n_fin", 3)
        W_ch = (2 * p.get("Hfin_nm", 52.0) + p.get("Wfin_nm", 6.0)) * 1e-9  # effective width
        L_g  = p["L_g_nm"] * 1e-9
        DIBL = 0.04

    Vt_eff = Vt - DIBL * Vds
    Qinv   = Cox * max(Vgs - Vt_eff, 0.0)
    v_T    = 1.1e5   # thermal velocity [m/s]
    I_sat  = n_s * W_ch * Qinv * v_T * 0.5
    V_dsat = max(Vgs - Vt_eff, 0.01) * 0.5
    I_D    = I_sat * (1.0 - math.exp(-max(Vds, 0.0) / max(V_dsat, 0.01)))
    I_D_uA_um = I_D / max(n_s * W_ch, 1e-15) * 1e6 / 1e-6

    return {
        "node": node, "type": p["type"],
        "Vgs_V": Vgs, "Vds_V": Vds,
        "Id_uA_um": round(I_D_uA_um, 1),
        "Id_uA": round(I_D * 1e6, 3),
        "Vt_eff_V": round(Vt_eff, 4),
        "density_MT_mm2": p["density_MT_mm2"],
    }


def die_cost(node: str = "N3E", die_area_mm2: float = 100.0) -> dict:
    """
    TSMC ダイコストモデル:
    CPGD = wafer_cost / (dies_per_wafer × yield)
    yield = Murphy: [(1-exp(-D0·A))/(D0·A)]²

    参考: Stow (2017) DAC — wafer cost model
    """
    p  = PROCESS_NODES.get(node, PROCESS_NODES["N3E"])
    D0 = p["D0"]; A_cm2 = die_area_mm2 / 100.0
    val = (1.0 - math.exp(-D0 * A_cm2)) / max(D0 * A_cm2, 1e-12)
    Y   = val ** 2
    R   = 150.0  # 300mm
    n_dies = max(int((math.pi * R**2 - math.pi * R * math.sqrt(2*die_area_mm2)) / die_area_mm2), 1)
    good_dies = max(int(n_dies * Y), 1)
    cpgd = p["wafer_cost_usd"] / good_dies
    return {
        "node": node, "die_area_mm2": die_area_mm2,
        "wafer_cost_usd": p["wafer_cost_usd"],
        "dies_per_wafer": n_dies,
        "yield_pct": round(Y * 100, 1),
        "good_dies_per_wafer": good_dies,
        "CPGD_usd": round(cpgd, 2),
        "D0": D0,
    }


def cowos_bandwidth(pkg: str = "CoWoS-S", die_edge_mm: float = 25.0) -> dict:
    """
    CoWoS / SoIC ダイ間帯域幅: BW = BW_per_mm × 4辺 × die_edge
    面積コスト: cost = cost_per_mm2 × interposer_area
    """
    p = PACKAGE_TYPES.get(pkg, PACKAGE_TYPES["CoWoS-S"])
    BW_TBs   = p["BW_per_mm_TBs"] * die_edge_mm * 4
    area_mm2 = (die_edge_mm * 1.5) ** 2   # interposer ≈ 1.5× die edge square
    area_mm2 = min(area_mm2, p["max_area_mm2"])
    pkg_cost = p["cost_per_mm2_usd"] * area_mm2
    return {
        "pkg": pkg, "die_edge_mm": die_edge_mm,
        "BW_TBs": round(BW_TBs, 3),
        "interposer_area_mm2": round(area_mm2, 0),
        "pkg_cost_usd": round(pkg_cost, 0),
        "bump_pitch_um": p["bump_pitch_um"],
    }


def soic_thermal(P_die_W: float = 30.0, A_die_mm2: float = 100.0,
                 n_stacked: int = 2) -> dict:
    """
    SoIC 積層の熱抵抗モデル:
    各ダイを直列熱抵抗として積み上げ。
    R_total = n × R_bond;  R_bond = t_bond / (k_bond × A)
    k_bond ≈ 150 W/m·K (Cu-to-Cu direct bond), t_bond = 1µm

    参考: Liu et al. ECTC 2022
    """
    k_bond = 150.0  # Cu-Cu direct bond [W/m·K]
    t_bond_m = 1e-6
    A_m2 = A_die_mm2 * 1e-6
    R_bond = t_bond_m / (k_bond * A_m2)          # [°C/W] per bond
    R_total = n_stacked * R_bond
    T_amb = 35.0
    T_j   = T_amb + R_total * P_die_W * n_stacked
    return {
        "n_stacked": n_stacked, "P_die_W": P_die_W,
        "A_die_mm2": A_die_mm2,
        "R_bond_C_W": round(R_bond, 6),
        "R_total_C_W": round(R_total, 6),
        "T_junction_C": round(T_j, 1),
        "hotspot_ok": T_j < 100.0,
    }


def market_share_revenue() -> dict:
    """TSMC ノード別売上構成 (2024年推定)"""
    revenue_pct = {
        "3nm": 20.0, "5nm": 35.0, "7nm": 17.0,
        "16nm": 9.0, "28nm": 8.0, "older": 11.0,
    }
    total_rev_B = 90.0   # FY2024 ~$90B
    return {
        "FY2024_total_B": total_rev_B,
        "breakdown_pct": revenue_pct,
        "advanced_pct": revenue_pct["3nm"] + revenue_pct["5nm"],
        "revenue_B": {k: round(v/100*total_rev_B, 1) for k, v in revenue_pct.items()},
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    Vds_arr = np.linspace(0, 0.75, 120)

    # ── Panel 0: トランジスタ Id–Vd ──────────────────────────────────────
    for node, col in [(n, PROCESS_NODES[n]["color"]) for n in PROCESS_NODES]:
        Vdd = PROCESS_NODES[node]["Vdd_V"]
        for Vgs, ls in [(Vdd, "-"), (Vdd*0.7, "--"), (Vdd*0.4, ":")]:
            ids = [transistor_current(node, Vgs, Vd)["Id_uA_um"] for Vd in Vds_arr]
            lbl = f"TSMC {node}" if ls == "-" else None
            ax[0].plot(Vds_arr, ids, color=col, ls=ls, lw=2.0, label=lbl)
    ax[0].set_xlabel("Vds [V]"); ax[0].set_ylabel("Id [µA/µm]")
    ax[0].set_title("TSMC FinFET / GAAFET Id–Vd\nN16 · N7 · N3E · N2 · A16")
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=0.25)

    # ── Panel 1: ダイコスト vs ダイ面積 ──────────────────────────────────
    area_arr = np.linspace(10, 500, 200)
    for node, col in [(n, PROCESS_NODES[n]["color"]) for n in PROCESS_NODES]:
        cpgd = [die_cost(node, a)["CPGD_usd"] for a in area_arr]
        ax[1].semilogy(area_arr, cpgd, color=col, lw=2.2, label=f"TSMC {node}")
    ax[1].set_xlabel("ダイ面積 [mm²]"); ax[1].set_ylabel("CPGD [USD]")
    ax[1].set_title("TSMC ダイコスト vs ダイ面積\n(Murphy yield、300mm ウェーハ)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

    # ── Panel 2: CoWoS / SoIC 帯域幅比較 ────────────────────────────────
    edge_arr = np.linspace(5, 40, 100)
    for pkg, col in [(p, PACKAGE_TYPES[p]["color"]) for p in PACKAGE_TYPES]:
        bw = [cowos_bandwidth(pkg, e)["BW_TBs"] for e in edge_arr]
        ax[2].plot(edge_arr, bw, color=col, lw=2.2,
                   label=pkg.replace("-", " ").replace("_", " "))
    ax[2].set_xlabel("ダイエッジ長 [mm]"); ax[2].set_ylabel("帯域幅 [TB/s]")
    ax[2].set_title("TSMC CoWoS / SoIC 帯域幅\n(ダイエッジ長 × BW/mm × 4辺)")
    ax[2].legend(fontsize=8.5); ax[2].grid(alpha=0.25)

    # ── Panel 3: SoIC 熱モデル ───────────────────────────────────────────
    P_arr = np.linspace(5, 60, 100)
    for n_s, col in [(1, "#ff7f0e"), (2, "#d62728"), (3, "#2166ac"), (4, "#2ca02c")]:
        Tj = [soic_thermal(P, 100.0, n_s)["T_junction_C"] for P in P_arr]
        ax[3].plot(P_arr, Tj, color=col, lw=2.2, label=f"{n_s} ダイ積層")
    ax[3].axhline(100.0, color="k", ls="--", lw=1.5, label="100°C 限界")
    ax[3].set_xlabel("ダイ電力 [W]"); ax[3].set_ylabel("T_junction [°C]")
    ax[3].set_title("SoIC 積層 接合温度\n(Cu-Cu ダイレクトボンド, k=150 W/m·K)")
    ax[3].legend(fontsize=9); ax[3].grid(alpha=0.25)

    # ── Panel 4: 密度 / ウェーハコスト ロードマップ ──────────────────────
    nodes_nm2 = [PROCESS_NODES[n]["node_nm"] for n in PROCESS_NODES]
    densities2 = [PROCESS_NODES[n]["density_MT_mm2"] for n in PROCESS_NODES]
    costs2     = [PROCESS_NODES[n]["wafer_cost_usd"] for n in PROCESS_NODES]
    colors_n   = [PROCESS_NODES[n]["color"] for n in PROCESS_NODES]
    ax4b = ax[4].twinx()
    ax[4].semilogy(nodes_nm2, densities2, "s--", color="#2166ac", lw=2.2, ms=8, label="密度 [MT/mm²]")
    ax4b.plot(nodes_nm2, costs2, "D-", color="#d62728", lw=2.0, ms=8)
    ax4b.set_ylabel("ウェーハコスト [USD]", color="#d62728")
    ax[4].set_xlabel("ノード [nm]"); ax[4].set_ylabel("密度 [MT/mm²]", color="#2166ac")
    ax[4].set_title("TSMC 密度スケーリング vs ウェーハコスト\nN16 → A16 ロードマップ")
    ax[4].invert_xaxis(); ax[4].grid(alpha=0.25)
    ax[4].legend(loc="upper right", fontsize=8.5)

    # ── Panel 5: ノード比較サマリーテーブル ─────────────────────────────
    ax[5].axis("off")
    col_labels = ["Node", "Type", "密度\n[MT/mm²]", "Vdd\n[V]",
                  "CPGD\n100mm²", "CoWoS-L\nBW [TB/s]"]
    tdata = []
    for nn in PROCESS_NODES:
        dc = die_cost(nn, 100.0)
        bw = cowos_bandwidth("CoWoS-L", 25.0)
        tdata.append([
            f"TSMC {nn}",
            PROCESS_NODES[nn]["type"],
            f"{PROCESS_NODES[nn]['density_MT_mm2']:.0f}",
            f"{PROCESS_NODES[nn]['Vdd_V']:.2f}",
            f"${dc['CPGD_usd']:.0f}",
            f"{bw['BW_TBs']:.2f}",
        ])
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center", bbox=[0.0, 0.10, 1.0, 0.85])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#003366"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#e8f0ff")
    ax[5].set_title("TSMC ノード比較\nN16 · N7 · N3E · N2 · A16")

    fig.suptitle("TSMC — N2/N3E/A16 プロセス + CoWoS/SoIC 先進パッケージ\n"
                 "GAA電流 · ダイコスト · CoWoS帯域幅 · SoIC熱モデル",
                 fontsize=12, fontweight="bold")
    out = os.path.join(out_dir, "tsmc_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] TSMC model → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="TSMC N2/N3E/A16 + CoWoS/SoIC model")
    ap.add_argument("--node",    default="N2", choices=list(PROCESS_NODES.keys()))
    ap.add_argument("--pkg",     default="CoWoS-S", choices=list(PACKAGE_TYPES.keys()))
    ap.add_argument("--cost",    action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.cost:
        print(f"\nTSMC {args.node} ダイコスト")
        for A in [50, 100, 200, 400]:
            r = die_cost(args.node, A)
            print(f"  {A:4d}mm²  yield={r['yield_pct']:.1f}%  CPGD=${r['CPGD_usd']:.0f}")
    if not args.no_plot:
        plot_all()

if __name__ == "__main__":
    main()
