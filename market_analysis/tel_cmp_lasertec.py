"""
TEL CMP + Lasertec 検査 統合モデル
====================================
TEL CMP (Chemical Mechanical Planarization):
    Preston 方程式 → SiC 研磨速度 (Si の ~100×遅い)
    均一性モデル → center/edge 膜厚ばらつき
    Post-CMP 表面品質 → ALD/酸化への入力

Lasertec 検査 (2 モード):
    [A] EUV マスク検査 (ACTIS シリーズ) — 2nm ロジックチップ向け
        欠陥密度 → リソグラフィー歩留まり
        TSMC/Samsung の EUV プロセスを支える「唯一の装置」
    [B] SiC ウェーハ後検査 — ダイシング後品質
        Disco ダイシング後チッピング/PL マッピング
        (ml/lasertec_inspection.py の拡張統合版)

実行:
    python fem/tel_cmp_lasertec.py
    python fem/tel_cmp_lasertec.py --cmp
    python fem/tel_cmp_lasertec.py --euv
    python fem/tel_cmp_lasertec.py --sic
"""

import argparse
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── 物理定数 ────────────────────────────────────────────────────────────────
kB = 8.617e-5   # eV/K


# ════════════════════════════════════════════════════════════════════════════
# 1. TEL CMP — Preston 方程式 (SiC 対応)
# ════════════════════════════════════════════════════════════════════════════

# 研磨レート定数 [nm/min per kPa·m/s]  (Preston 係数 Kp)
CMP_PARAMS = {
    "4H-SiC":   {"Kp": 0.8e-3,  "hardness_GPa": 28.0, "color": "#2166ac",
                  "slurry": "diamond/colloidal SiO₂", "ref": "Ye et al. 2020"},
    "Si":        {"Kp": 80e-3,   "hardness_GPa":  9.5, "color": "#d62728",
                  "slurry": "fumed SiO₂ (KOH)", "ref": "Preston 1927"},
    "SiO2 (TEOS)": {"Kp": 50e-3, "hardness_GPa": 7.0, "color": "#2ca02c",
                  "slurry": "ceria", "ref": "Stein et al. 1999"},
    "Cu (BEOL)": {"Kp": 120e-3,  "hardness_GPa": 3.0, "color": "#ff7f0e",
                  "slurry": "H₂O₂/BTA", "ref": "Steigerwald 1997"},
}


def preston_mrr(pressure_kPa: float, velocity_m_s: float,
                material: str = "4H-SiC") -> float:
    """
    Preston 方程式: MRR = Kp × P × V  [nm/min]
    P: 研磨圧力 [kPa], V: 相対速度 [m/s]
    """
    Kp = CMP_PARAMS[material]["Kp"]
    return Kp * pressure_kPa * velocity_m_s


def cmp_uniformity(mrr_center: float, wafer_r_mm: float = 75.0,
                    platen_r_mm: float = 300.0,
                    omega_ratio: float = 1.05) -> np.ndarray:
    """
    ウェーハ内均一性モデル (相対速度の半径依存性):
    V(r) = |ω_platen × R_platen - ω_wafer × r|  (近似)
    研磨レートのウェーハ半径方向プロファイル。
    """
    rs = np.linspace(0, wafer_r_mm, 100)
    # プラテン中心からウェーハ中心が offset だけ離れている想定
    offset = wafer_r_mm * 0.3
    V_rel = np.abs(omega_ratio * platen_r_mm - (rs + offset))
    V_rel = V_rel / V_rel.mean()   # 正規化
    mrr   = mrr_center * V_rel
    return rs, mrr


def cmp_time(target_depth_nm: float, pressure_kPa: float = 30.0,
              velocity_m_s: float = 0.5, material: str = "4H-SiC") -> dict:
    """
    目標研磨深さに到達する時間 [min] と表面品質。
    SiC は Si の ~100× 遅い → 工業的に非常にコスト高。
    """
    mrr    = preston_mrr(pressure_kPa, velocity_m_s, material)
    t_min  = target_depth_nm / max(mrr, 1e-10)
    Ra_after = 0.3 * (1 + 100 / max(pressure_kPa, 1))   # 経験的

    return {
        "material":         material,
        "MRR_nm_min":       round(mrr, 3),
        "time_min":         round(t_min, 1),
        "Ra_after_nm":      round(Ra_after, 2),
        "cost_factor_vs_Si": round(CMP_PARAMS["Si"]["Kp"] / CMP_PARAMS[material]["Kp"], 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. Lasertec [A] EUV マスク検査 (ACTIS)
# ════════════════════════════════════════════════════════════════════════════

# EUV マスク欠陥タイプと検出感度 (ACTIS A150 相当)
EUV_DEFECT_TYPES = {
    "phase_defect":     {"size_min_nm": 1.5,  "printability": 0.95, "color": "#d62728"},
    "absorber_pinhole": {"size_min_nm": 3.0,  "printability": 0.80, "color": "#ff7f0e"},
    "particle":         {"size_min_nm": 2.0,  "printability": 0.70, "color": "#2ca02c"},
    "pit_bump":         {"size_min_nm": 2.5,  "printability": 0.60, "color": "#2166ac"},
}


def euv_mask_defect_density(node_nm: float, fab: str = "TSMC") -> dict:
    """
    EUV マスクの欠陥密度モデル [defects/cm²]:
    ノード微細化 → パターン密度増加 → 同一欠陥が"印刷される"確率増加。

    ACTIS A150 の検査感度: 1.5nm 以上の位相欠陥を検出可能。
    """
    # 欠陥密度: ノードが縮小するほど厳しくなる (指数的)
    D0    = 0.01   # defects/cm² (基準欠陥密度)
    alpha = 0.15   # ノード感度
    D     = D0 * math.exp(alpha * (28 - node_nm))   # 28nm 基準

    # 印刷欠陥率 (printable defect rate)
    p_print = 1 - math.exp(-D * 100)   # 100cm² マスク面積

    # 歩留まりへの影響 (Poisson 欠陥モデル)
    die_area_cm2 = {"TSMC": 0.8, "Samsung": 0.7, "Intel": 0.6}.get(fab, 0.7)
    yield_loss   = 1 - math.exp(-D * die_area_cm2)

    return {
        "node_nm":          node_nm,
        "fab":              fab,
        "defect_density":   round(D, 4),
        "print_rate_pct":   round(p_print * 100, 2),
        "yield_loss_pct":   round(yield_loss * 100, 3),
        "actis_required":   node_nm <= 7,   # 7nm 以下で ACTIS が必須
    }


def euv_inspection_roi(node_nm: float, wafer_cost_usd: float = 15000) -> dict:
    """
    Lasertec ACTIS の ROI 計算:
    欠陥1個が引き起こすウェーハ損失 vs 検査コスト。
    """
    res     = euv_mask_defect_density(node_nm)
    n_wafers_per_mask = 10000   # マスク1枚で処理するウェーハ数
    cost_per_defect   = res["yield_loss_pct"] / 100 * wafer_cost_usd * n_wafers_per_mask
    actis_cost_mask   = 500_000   # ACTIS 検査コスト [USD/マスク]
    roi               = cost_per_defect / max(actis_cost_mask, 1)

    return {
        "node_nm":             node_nm,
        "yield_loss_pct":      res["yield_loss_pct"],
        "cost_per_defect_M":   round(cost_per_defect / 1e6, 2),
        "actis_cost_k":        actis_cost_mask / 1000,
        "ROI":                 round(roi, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Lasertec [B] SiC 後検査 (ダイシング後)
# ════════════════════════════════════════════════════════════════════════════

def sic_postdice_inspection(process: str, n_dies: int = 100,
                             seed: int = 0) -> dict:
    """
    Disco ダイシング後の Lasertec SiC 検査。
    チッピング / サブサーフェスクラック / PL 欠陥を測定し
    Grade A/B/C/D に分類。
    """
    from ml.lasertec_inspection import grade_die, confocal_scan, chipping_measurement
    g = grade_die(process)
    c = confocal_scan(process)
    ch = chipping_measurement(process, n_sites=n_dies)

    # 検査後のフィードバック: Grade B以下なら Disco にアラート
    feedback = {
        "A": "合格 — 次工程へ",
        "B": "条件付き合格 — ブレード交換推奨",
        "C": "条件付き不合格 — プロセス確認要",
        "D": "廃棄 — Disco/TEL へ緊急フィードバック",
    }[g["grade"]]

    return {
        "process":    process,
        "grade":      g["grade"],
        "Ra_nm":      c["Ra_nm"],
        "chip_mean":  ch["mean_um"],
        "chip_max":   ch["max_um"],
        "feedback":   feedback,
        "scores":     g["scores"],
    }


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_cmp(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: MRR vs 材料比較
    ax = axes[0]
    materials = list(CMP_PARAMS.keys())
    P_arr = np.linspace(10, 60, 80)
    V = 0.5
    for mat in materials:
        mrrs = [preston_mrr(P, V, mat) for P in P_arr]
        ax.plot(P_arr, mrrs, color=CMP_PARAMS[mat]["color"],
                lw=2, label=mat)
    ax.set_xlabel("Pressure [kPa]", fontsize=11)
    ax.set_ylabel("MRR [nm/min]", fontsize=11)
    ax.set_title("Preston 方程式 — CMP 研磨レート\n(V=0.5m/s)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    ax.text(15, 2.5, "SiC は Si の\n~100× 遅い", fontsize=9,
            color=CMP_PARAMS["4H-SiC"]["color"],
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Panel 2: 均一性プロファイル
    ax2 = axes[1]
    for mat in ["4H-SiC", "Si", "SiO2 (TEOS)"]:
        mrr0 = preston_mrr(30, 0.5, mat)
        rs, mrr_r = cmp_uniformity(mrr0)
        ax2.plot(rs, mrr_r / mrr0 * 100, color=CMP_PARAMS[mat]["color"],
                 lw=2, label=mat)
    ax2.axhline(100, color="black", ls="--", lw=1, alpha=0.4)
    ax2.axhspan(98, 102, alpha=0.08, color="green", label="±2% 均一性目標")
    ax2.set_xlabel("Wafer Radius [mm]", fontsize=11)
    ax2.set_ylabel("Normalized MRR [%]", fontsize=11)
    ax2.set_title("CMP 均一性プロファイル\n(TEL CLEAN TRACK 相当)", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2)

    # Panel 3: SiC 研磨時間 vs 圧力
    ax3 = axes[2]
    target = 100   # nm
    Ps = np.linspace(10, 80, 60)
    for V_val, col, lbl in [(0.3,"#2166ac","V=0.3m/s"),
                             (0.5,"#2ca02c","V=0.5m/s"),
                             (0.8,"#d62728","V=0.8m/s")]:
        ts = [cmp_time(target, P, V_val, "4H-SiC")["time_min"] for P in Ps]
        ax3.semilogy(Ps, ts, color=col, lw=2, label=lbl)
    ax3.set_xlabel("Pressure [kPa]", fontsize=11)
    ax3.set_ylabel("CMP Time [min]", fontsize=11)
    ax3.set_title(f"SiC CMP 研磨時間 (目標 {target}nm)\nコスト問題の核心", fontsize=10)
    ax3.legend(fontsize=9); ax3.grid(alpha=0.2, which="both")
    ax3.axhline(60, color="red", ls="--", lw=1.5, label="60min 量産限界")

    fig.suptitle("TEL CMP モデル — Preston 方程式 × SiC 研磨\n"
                 "SiC の硬度 28GPa が製造コストを押し上げる主因", fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "tel_cmp.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] TEL CMP -> {out}")


def plot_euv(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    nodes = np.linspace(2, 28, 60)
    fabs  = ["TSMC", "Samsung", "Intel"]
    colors = {"TSMC": "#2166ac", "Samsung": "#d62728", "Intel": "#2ca02c"}

    # Panel 1: 欠陥密度 vs ノード
    ax = axes[0]
    for fab in fabs:
        dens = [euv_mask_defect_density(n, fab)["defect_density"] for n in nodes]
        ax.semilogy(nodes, dens, color=colors[fab], lw=2, label=fab)
    ax.axvline(2, color="purple", ls="--", lw=1.5, label="2nm ノード")
    ax.axvline(7, color="gray",   ls=":",  lw=1.2, label="ACTIS 必須ライン")
    ax.set_xlabel("Process Node [nm]", fontsize=11)
    ax.set_ylabel("Mask Defect Density [/cm²]", fontsize=11)
    ax.set_title("EUV マスク欠陥密度 vs ノード\n(Lasertec ACTIS 検出対象)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.2, which="both")
    ax.invert_xaxis()

    # Panel 2: Yield loss vs ノード
    ax2 = axes[1]
    for fab in fabs:
        yl = [euv_mask_defect_density(n, fab)["yield_loss_pct"] for n in nodes]
        ax2.plot(nodes, yl, color=colors[fab], lw=2, label=fab)
    ax2.axvline(2, color="purple", ls="--", lw=1.5)
    ax2.set_xlabel("Process Node [nm]", fontsize=11)
    ax2.set_ylabel("Yield Loss from Mask Defects [%]", fontsize=11)
    ax2.set_title("マスク欠陥による歩留まり損失\n2nm ノードでの臨界性", fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.2)
    ax2.invert_xaxis()

    # Panel 3: ROI テーブル
    ax3 = axes[2]
    ax3.axis("off")
    node_vals = [28, 14, 7, 5, 3, 2]
    rows = [["Node", "Yield Loss", "Cost at Risk", "ACTIS ROI"]]
    for n in node_vals:
        roi = euv_inspection_roi(n)
        rows.append([
            f"{n}nm",
            f"{roi['yield_loss_pct']:.2f}%",
            f"${roi['cost_per_defect_M']:.0f}M",
            f"{roi['ROI']:.0f}×",
        ])
    tbl = ax3.table(cellText=rows[1:], colLabels=rows[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1.2, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2166ac"); cell.set_text_props(color="white")
        elif r <= len(node_vals) and rows[r][0] in ["3nm", "2nm"]:
            cell.set_facecolor("#ffcccc")
    ax3.set_title("ACTIS 検査 ROI\n(マスク1枚で10,000枚ウェーハ処理)", fontsize=10)

    fig.suptitle("Lasertec ACTIS — EUV マスク検査\n"
                 "2nm ノードで世界唯一の必須装置 — TSMC/Samsung の歩留まりを守る",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "lasertec_euv.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Lasertec EUV -> {out}")


def plot_sic(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    procs  = ["blade", "laser_ps", "laser_fs", "stealth"]
    labels = {"blade":"ブレード","laser_ps":"レーザーps",
              "laser_fs":"レーザーfs","stealth":"ステルス"}
    colors = {"blade":"#d62728","laser_ps":"#ff7f0e",
              "laser_fs":"#2ca02c","stealth":"#2166ac"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: グレード比較
    ax = axes[0]
    grade_val = {"A":4,"B":3,"C":2,"D":1}
    grade_col = {"A":"#2ca02c","B":"#1f77b4","C":"#ff7f0e","D":"#d62728"}
    results = [sic_postdice_inspection(p) for p in procs]
    for k, r in enumerate(results):
        g = r["grade"]
        ax.bar(labels[r["process"]], grade_val[g],
               color=grade_col[g], alpha=0.85, edgecolor="k", lw=0.8)
        ax.text(k, grade_val[g]+0.05, f"Grade {g}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_yticks([1,2,3,4]); ax.set_yticklabels(["D","C","B","A"])
    ax.set_title("Lasertec SiC 後検査 グレード\n(ダイシング後 in-line)", fontsize=10)
    ax.grid(alpha=0.2, axis="y")

    # Panel 2: フィードバックチェーン
    ax2 = axes[1]
    ax2.axis("off")
    feedback_rows = [["プロセス","Grade","フィードバック"]]
    for r in results:
        feedback_rows.append([labels[r["process"]], r["grade"], r["feedback"][:20]])
    tbl = ax2.table(cellText=feedback_rows[1:], colLabels=feedback_rows[0],
                    loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.2, 2.2)
    grade_bg = {"A":"#d4f7d4","B":"#d4e8f7","C":"#fff4cc","D":"#f7d4d4"}
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#555"); cell.set_text_props(color="white")
        elif r <= len(results):
            g = feedback_rows[r][1]
            cell.set_facecolor(grade_bg.get(g, "white"))
    ax2.set_title("Disco へのフィードバックチェーン", fontsize=10)

    fig.suptitle("Lasertec SiC ウェーハ後検査 — Disco ダイシング後\n"
                 "チッピング / PL マッピング / Grade → Disco プロセスフィードバック",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "lasertec_sic.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Lasertec SiC -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmp", action="store_true")
    ap.add_argument("--euv", action="store_true")
    ap.add_argument("--sic", action="store_true")
    args = ap.parse_args()
    run_all = not any([args.cmp, args.euv, args.sic])
    if args.cmp or run_all: plot_cmp()
    if args.euv or run_all: plot_euv()
    if args.sic or run_all: plot_sic()


if __name__ == "__main__":
    main()
