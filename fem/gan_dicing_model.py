"""
GaN ダイシングモデル — RF/パワーデバイス用ウェーハ加工
=======================================================
Disco が対応する GaN (窒化ガリウム) ウェーハのダイシング解析。
SiC とは異なる破壊特性・基板構成を物理モデル化。

主なユースケース:
    - GaN-on-Si (RF HEMT, パワー GaN)  → Si 基板上 GaN エピ
    - GaN-on-SiC (高周波パワー)         → SiC 基板上 GaN エピ
    - Bulk GaN (次世代パワー)           → 単結晶 GaN 基板

SiC との主な違い:
    K_Ic: GaN = 0.9 MPa√m  vs  SiC = 2.8 MPa√m (GaN は脆い)
    硬度: GaN = HV1300      vs  SiC = HV2580
    → GaN は SiC より低フルエンスで切断可能だが「へき開」が起きやすい

実行:
    python fem/gan_dicing_model.py
    python fem/gan_dicing_model.py --substrate sic

参考文献:
    Levinshtein et al. (2001) Properties of Advanced Semiconductor Materials
    Tanaka et al. (2004) IEEE Trans Electron Devices — GaN dicing
    Disco Application Note: GaN-on-Si DAD / DFL
"""

import argparse
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.materials.material_properties import GaN, SiC, Si

# ── GaN-on-X 複合基板構成 ─────────────────────────────────────────────────────
SUBSTRATES = {
    "GaN_on_Si": {
        "epi_material":  GaN,
        "sub_material":  Si,
        "epi_um":        5.0,    # GaN エピ厚 [µm]
        "sub_um":        500.0,  # Si 基板厚 [µm]
        "color":         "#2166ac",
        "app":           "RF HEMT / パワー GaN",
        "challenge":     "Si/GaN 界面での剥離リスク (CTE ミスマッチ 54%)",
    },
    "GaN_on_SiC": {
        "epi_material":  GaN,
        "sub_material":  SiC,
        "epi_um":        4.0,
        "sub_um":        350.0,
        "color":         "#d62728",
        "app":           "高周波パワー (5G PA, EV)",
        "challenge":     "SiC の高硬度 → ブレード摩耗が激しい",
    },
    "Bulk_GaN": {
        "epi_material":  GaN,
        "sub_material":  GaN,
        "epi_um":        400.0,  # 基板全体が GaN
        "sub_um":        0.0,
        "color":         "#ff7f0e",
        "app":           "次世代 GaN パワー基板",
        "challenge":     "単結晶 GaN は高コスト・小口径",
    },
}

# ── プロセス別加工パラメータ推奨値 ────────────────────────────────────────────
PROCESS_PARAMS = {
    "blade": {
        "blade_W_um":    20,
        "cut_depth_um":  50,
        "feed_mm_s":     50,
        "spindle_krpm":  40,
        "coolant":       "DIW",
        "note":          "低送り速度 + 高回転が GaN に有効",
    },
    "laser_uv": {
        "wavelength_nm": 355,
        "pulse_regime":  "ps",
        "F_th_J_cm2":    0.35,
        "HAZ_factor":    0.05,
        "note":          "UV レーザー (355nm) は GaN に高吸収",
    },
    "stealth": {
        "wavelength_nm": 1064,
        "pulse_regime":  "ps",
        "focal_depth_um": 50,
        "note":          "GaN は厚みが薄いため浅い焦点位置が有効",
    },
}


# ════════════════════════════════════════════════════════════════════════════
# 1. GaN チッピング物理モデル
# ════════════════════════════════════════════════════════════════════════════

def chipping_model(substrate: str, process: str,
                   cut_depth_um: float, feed_mm_s: float) -> dict:
    """
    GaN-on-X ダイシングのチッピング幅推定 [µm]。
    SiC モデル (0.52 * depth^0.65) をベースに GaN 材料補正を適用。

    補正係数:
        K_Ic 補正: (K_Ic_SiC / K_Ic_mat)^0.5  (靱性が低いほどチッピング大)
        硬度補正:  (HV_mat / HV_SiC)^0.3       (硬いほど切削力大)
    """
    sub = SUBSTRATES[substrate]
    mat = sub["epi_material"]  # エピ層 (大部分が GaN)

    # SiC 基準のチッピングモデル
    base_chip = 0.52 * cut_depth_um ** 0.65

    # GaN の K_Ic は SiC より低い → チッピングが増える
    KIc_SiC = SiC["K_Ic"]   # Pa√m
    KIc_mat = mat["K_Ic"]   # Pa√m
    kic_factor = (KIc_SiC / max(KIc_mat, 0.1)) ** 0.5

    # 硬度補正 (Vickers)
    HV_SiC = 2580
    HV_GaN = 1300
    hv_factor = (HV_GaN / HV_SiC) ** 0.3  # 柔らかい → 加工しやすい

    # 送り速度補正 (高速送り → チッピング増)
    feed_factor = (feed_mm_s / 50.0) ** 0.4

    # 基板影響: GaN-on-Si では界面剥離リスクが追加
    delamination_risk = 0.0
    if substrate == "GaN_on_Si":
        # Si と GaN の CTE 差 (54%) → 熱による界面応力
        CTE_Si  = 2.6e-6   # /K
        CTE_GaN = 5.6e-6   # /K
        delta_cte = abs(CTE_GaN - CTE_Si)
        delamination_risk = delta_cte * 1e6 * cut_depth_um * 0.001

    chipping_um = base_chip * kic_factor * hv_factor * feed_factor
    chipping_um += delamination_risk

    return {
        "substrate":          substrate,
        "process":            process,
        "cut_depth_um":       cut_depth_um,
        "feed_mm_s":          feed_mm_s,
        "chipping_um":        round(chipping_um, 2),
        "kic_factor":         round(kic_factor, 3),
        "hv_factor":          round(hv_factor, 3),
        "delamination_risk":  round(delamination_risk, 3),
        "app":                sub["app"],
    }


def die_strength(substrate: str, chipping_um: float) -> float:
    """
    ダイ強度 [MPa] — チッピングによるノッチ効果。
    σ_f = K_Ic / (Y * √(π * a))  ここで a = チッピングサイズ
    """
    sub = SUBSTRATES[substrate]
    mat = sub["epi_material"]
    KIc = mat["K_Ic"]   # Pa√m
    Y   = 1.12                       # 形状係数 (edge crack)
    a_m = chipping_um * 1e-6         # µm → m
    if a_m <= 0:
        return 1000.0
    return KIc / (Y * math.sqrt(math.pi * a_m)) / 1e6  # Pa → MPa


def blade_wear_rate(substrate: str) -> float:
    """
    ブレード摩耗速度 [相対値, SiC=1.0]。
    硬度と破壊靱性に依存。
    """
    sub = SUBSTRATES[substrate]
    # 複合層の実効硬度 (エピ厚/基板厚で重み付け)
    h_epi = sub["epi_um"]
    h_sub = sub["sub_um"]
    h_tot = h_epi + h_sub
    HV_GaN = 1300; HV_SiC = 2580; HV_Si = 1100
    HV_sub = (HV_SiC if sub["sub_material"] is SiC
              else (HV_Si if sub["sub_material"] is Si else HV_GaN))
    HV_eff = (HV_GaN * h_epi + HV_sub * h_sub) / max(h_tot, 1)
    return (HV_eff / HV_SiC) ** 1.5


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    depths = np.linspace(20, 200, 80)
    feeds  = [20, 50, 100, 150]

    # ── Panel 1: チッピング vs 切込み深さ (基板別) ────────────────────────
    ax = axes[0, 0]
    for sub_name, sub_d in SUBSTRATES.items():
        chips = [chipping_model(sub_name, "blade", d, 50)["chipping_um"]
                 for d in depths]
        ax.plot(depths, chips, lw=2.5, color=sub_d["color"], label=sub_name)
    # SiC 比較
    sic_chips = [0.52 * d ** 0.65 for d in depths]
    ax.plot(depths, sic_chips, "k--", lw=1.5, label="4H-SiC (reference)")
    ax.set_xlabel("Cut depth [µm]", fontsize=10)
    ax.set_ylabel("Chipping [µm]", fontsize=10)
    ax.set_title("GaN Chipping vs Cut Depth\n(feed=50mm/s, blade dicing)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # ── Panel 2: チッピング vs 送り速度 (GaN-on-Si) ───────────────────────
    ax2 = axes[0, 1]
    feed_arr = np.linspace(10, 200, 80)
    for sub_name, sub_d in SUBSTRATES.items():
        chips = [chipping_model(sub_name, "blade", 100, f)["chipping_um"]
                 for f in feed_arr]
        ax2.plot(feed_arr, chips, lw=2.5, color=sub_d["color"], label=sub_name)
    ax2.set_xlabel("Feed speed [mm/s]", fontsize=10)
    ax2.set_ylabel("Chipping [µm]", fontsize=10)
    ax2.set_title("GaN Chipping vs Feed Speed\n(cut_depth=100µm)", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2)

    # ── Panel 3: ダイ強度 vs チッピング ──────────────────────────────────
    ax3 = axes[0, 2]
    chip_arr = np.linspace(0.5, 30, 100)
    for sub_name, sub_d in SUBSTRATES.items():
        strengths = [die_strength(sub_name, c) for c in chip_arr]
        ax3.plot(chip_arr, strengths, lw=2.5, color=sub_d["color"], label=sub_name)
    ax3.axhline(200, color="red", ls="--", lw=1.5, label="Min. strength 200 MPa")
    ax3.set_xlabel("Chipping size [µm]", fontsize=10)
    ax3.set_ylabel("Die strength [MPa]", fontsize=10)
    ax3.set_title("Die Strength vs Chipping\n(Griffith / K_Ic model)", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(alpha=0.2)

    # ── Panel 4: GaN vs SiC vs Si 材料比較レーダー ────────────────────────
    ax4 = axes[1, 0]
    ax4.axis("off")
    mat_data = [
        ["特性", "4H-SiC", "GaN (epi)", "Si"],
        ["K_Ic [MPa√m]",    "2.80", "0.90", "0.83"],
        ["硬度 HV",          "2580", "1300", "1100"],
        ["ヤング率 [GPa]",   "400",  "295",  "130"],
        ["バンドギャップ [eV]", "3.26", "3.43", "1.12"],
        ["電子移動度 [cm²/Vs]", "900", "1500", "1400"],
        ["熱伝導率 [W/mK]", "490",  "130",  "130"],
        ["ダイシング難易度",  "難",   "中",   "易"],
    ]
    tbl = ax4.table(cellText=mat_data[1:], colLabels=mat_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.3, 2.1)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#333"); cell.set_text_props(color="w")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f5f5")
    ax4.set_title("Material Comparison: SiC / GaN / Si", fontsize=10)

    # ── Panel 5: ブレード摩耗比較 ────────────────────────────────────────
    ax5 = axes[1, 1]
    sub_names = list(SUBSTRATES.keys()) + ["4H-SiC (ref)"]
    wear_rates = [blade_wear_rate(s) for s in list(SUBSTRATES.keys())] + [1.0]
    colors5 = [SUBSTRATES[s]["color"] for s in list(SUBSTRATES.keys())] + ["#333333"]
    bars = ax5.bar(sub_names, wear_rates, color=colors5, alpha=0.85, edgecolor="k", lw=0.8)
    ax5.axhline(1.0, color="red", ls="--", lw=1.5, label="SiC baseline")
    ax5.set_ylabel("Relative blade wear rate", fontsize=10)
    ax5.set_title("Blade Wear Rate vs Substrate\n(SiC = 1.0 baseline)", fontsize=10)
    ax5.legend(fontsize=9); ax5.grid(alpha=0.2, axis="y")
    for bar, val in zip(bars, wear_rates):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.2f}x", ha="center", fontsize=9, fontweight="bold")

    # ── Panel 6: プロセス推奨サマリー ────────────────────────────────────
    ax6 = axes[1, 2]
    ax6.axis("off")
    rows = [["基板", "推奨プロセス", "チッピング", "注意点"]]
    for sub_name, sub_d in SUBSTRATES.items():
        r = chipping_model(sub_name, "blade", 100, 50)
        rows.append([
            sub_name.replace("_", "\n"),
            "Blade + Stealth",
            f"{r['chipping_um']:.1f} µm",
            sub_d["challenge"][:20] + "…",
        ])
    tbl2 = ax6.table(cellText=rows[1:], colLabels=rows[0],
                     loc="center", cellLoc="center")
    tbl2.auto_set_font_size(False); tbl2.set_fontsize(8)
    tbl2.scale(1.15, 2.2)
    for (r, c), cell in tbl2.get_celld().items():
        if r == 0:
            cell.set_facecolor("#333"); cell.set_text_props(color="w")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f5f5")
    ax6.set_title("GaN Dicing Process Recommendations", fontsize=10)

    fig.suptitle("GaN ダイシングモデル (Disco DAD/DFL 対応)\n"
                 "GaN-on-Si / GaN-on-SiC / Bulk GaN × チッピング・ダイ強度・摩耗",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "gan_dicing.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print("\n  GaN ダイシングモデル サマリー (cut_depth=100µm, feed=50mm/s):")
    print(f"  {'基板':<16} {'チッピング':>10} {'ダイ強度':>10} {'摩耗レート':>10}")
    print("  " + "─" * 52)
    for sub_name in SUBSTRATES:
        r = chipping_model(sub_name, "blade", 100, 50)
        ds = die_strength(sub_name, r["chipping_um"])
        wr = blade_wear_rate(sub_name)
        print(f"  {sub_name:<16} {r['chipping_um']:>9.2f}µm  "
              f"{ds:>8.0f}MPa  {wr:>9.2f}x")
    print(f"\n[OK] GaN dicing model -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="all",
                    choices=["all", "GaN_on_Si", "GaN_on_SiC", "Bulk_GaN"])
    args = ap.parse_args()
    plot_all()


if __name__ == "__main__":
    main()
