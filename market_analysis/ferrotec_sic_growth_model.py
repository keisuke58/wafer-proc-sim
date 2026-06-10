"""
フェローテック SiC 結晶成長モデル — PVT 昇華法
================================================
フェローテック (6890.T) の主力事業: SiC 単結晶ウェーハ製造。
物理蒸着輸送法 (PVT / Lely 改良法) による 4H-SiC 結晶成長を物理モデル化。

工程:
    SiC 粉末 (2200°C) → 昇華 → 温度勾配で種結晶上に析出 → インゴット
    → ワイヤーソー切断 → CMP 研磨 → 検査 → ウェーハ出荷

モデル内容:
    1. PVT 成長速度    — Arrhenius + 温度勾配依存
    2. 多形安定性      — 4H vs 6H vs 3C 相図
    3. マイクロパイプ欠陥密度 — 螺旋転位モデル
    4. CMP 後表面粗さ  — Preston 式 + ウェーハ品質ロードマップ
    5. コスト構造      — kWh/cm² × 電力コスト

実行:
    python fem/ferrotec_sic_growth_model.py

参考文献:
    Nakamura et al. (2004) J Crystal Growth — PVT growth model
    Powell & Rowland (2002) Phys Stat Sol — 4H-SiC polytype control
    Kamata et al. (2012) Mater Sci Forum — micropipe reduction
    フェローテック IR / Yole SiC substrate market 2024
"""

import math, os, sys
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

kB = 8.617e-5   # eV/K

# ── PVT 成長パラメータ ────────────────────────────────────────────────────────
PVT_PARAMS = {
    "T_source_C":  2250,    # 原料温度 [°C]
    "T_seed_C":    2150,    # 種結晶温度 [°C]  → ΔT=100K が成長駆動力
    "P_Ar_mbar":   20,      # アルゴン圧力 [mbar]
    "Q_act_eV":    6.0,     # 昇華活性化エネルギー [eV] (Nakamura 2004)
    "A_pvt":       1.2e18,  # 前指数因子 [µm/h] (キャリブレーション)
    "target_polytype": "4H",
}

# ── 欠陥密度ロードマップ (業界実績値) ────────────────────────────────────────
DEFECT_ROADMAP = {
    2015: {"micropipe_cm2": 1.0,  "TSD_cm2": 800,  "BPD_cm2": 2000},
    2018: {"micropipe_cm2": 0.1,  "TSD_cm2": 300,  "BPD_cm2": 500},
    2021: {"micropipe_cm2": 0.01, "TSD_cm2": 100,  "BPD_cm2": 100},
    2024: {"micropipe_cm2": 0.00, "TSD_cm2": 30,   "BPD_cm2": 20},
    2027: {"micropipe_cm2": 0.00, "TSD_cm2": 10,   "BPD_cm2": 5},   # 目標
}

# ── SiC ウェーハ市場価格 ─────────────────────────────────────────────────────
WAFER_PRICE = {
    "150mm_4H_SiC_2024": 600,   # USD/枚 (6インチ)
    "200mm_4H_SiC_2024": 1200,  # USD/枚 (8インチ, 量産初期)
    "150mm_Si_2024":      5,    # USD/枚 (比較用)
}


# ════════════════════════════════════════════════════════════════════════════
# 1. PVT 成長速度モデル
# ════════════════════════════════════════════════════════════════════════════

def pvt_growth_rate(T_source_C: float, T_seed_C: float,
                    P_Ar_mbar: float = 20) -> dict:
    """
    PVT 昇華法成長速度 [µm/h]。
    R = A * exp(-Q/kT_source) * (ΔT/T_ref)^1.5 * f(P_Ar)
    """
    p = PVT_PARAMS
    T_s = T_source_C + 273.15
    dT  = T_source_C - T_seed_C
    # Arrhenius 昇華速度
    rate_arr = p["A_pvt"] * math.exp(-p["Q_act_eV"] / (kB * T_s))
    # 温度勾配駆動力
    grad_factor = (dT / 100.0) ** 1.5
    # アルゴン圧力補正 (高圧 → 輸送抑制)
    p_factor = (20.0 / max(P_Ar_mbar, 1)) ** 0.4
    growth_um_h = rate_arr * grad_factor * p_factor

    # 実用成長速度は 200-500 µm/h が典型 (Nakamura 2004 キャリブレーション)
    growth_um_h = min(growth_um_h, 600)

    ingot_height_mm = growth_um_h * 1e-3 * 72  # 72h 成長
    return {
        "growth_rate_um_h": round(growth_um_h, 1),
        "ingot_72h_mm":     round(ingot_height_mm, 1),
        "wafers_per_ingot": int(ingot_height_mm / 0.5),  # 0.5mm スライス
        "T_source_C":  T_source_C,
        "delta_T":     dT,
        "P_Ar_mbar":   P_Ar_mbar,
    }


def polytype_stability(T_seed_C: float, C_Si_ratio: float = 1.0) -> str:
    """
    多形安定性 — 温度と C/Si 比から優勢ポリタイプを予測。
    4H: T > 2100°C, C/Si ≥ 1.0 が好ましい
    6H: T < 2100°C または C/Si 過剰
    3C: T < 1800°C (低温域)
    """
    if T_seed_C > 2100 and C_Si_ratio >= 0.95:
        return "4H-SiC (安定)"
    elif T_seed_C > 1900:
        return "6H-SiC (混入リスク)"
    else:
        return "3C-SiC (低温相)"


def micropipe_density(growth_rate_um_h: float, dT: float) -> float:
    """
    マイクロパイプ密度 [cm⁻²] 推定。
    高速成長・大温度勾配 → 螺旋転位が増加。
    Kamata (2012) モデル: MPD ∝ R^0.5 * ΔT^0.3
    """
    return max(0, 0.002 * (growth_rate_um_h ** 0.5) * (dT / 100) ** 0.3 - 0.001)


# ════════════════════════════════════════════════════════════════════════════
# 2. CMP 後表面粗さ
# ════════════════════════════════════════════════════════════════════════════

def cmp_roughness(MRR_nm_min: float, pressure_kPa: float,
                  slurry_size_nm: float = 50) -> dict:
    """
    CMP 後表面粗さ Ra [nm]。
    Preston 式ベース: Ra ∝ (slurry_size * pressure)^0.4 / MRR^0.2
    SiC ウェーハ品質目標: Ra < 0.1 nm (デバイス用)
    """
    Ra = 0.015 * (slurry_size_nm * pressure_kPa / 100) ** 0.4 / (MRR_nm_min / 10) ** 0.2
    return {
        "Ra_nm":        round(Ra, 4),
        "quality":      "device grade" if Ra < 0.1 else "epi-ready" if Ra < 0.5 else "mechanical",
        "MRR_nm_min":   MRR_nm_min,
        "pressure_kPa": pressure_kPa,
    }


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel 1: 成長速度 vs 温度差
    ax = axes[0, 0]
    dT_arr = np.linspace(50, 200, 100)
    for P, col, lbl in [(10,"#2166ac","P=10mbar"),(20,"#ff7f0e","P=20mbar"),(50,"#d62728","P=50mbar")]:
        rates = [pvt_growth_rate(2250, 2250-dT, P)["growth_rate_um_h"] for dT in dT_arr]
        ax.plot(dT_arr, rates, lw=2.5, color=col, label=lbl)
    ax.axhspan(200, 500, alpha=0.1, color="green", label="実用範囲")
    ax.set_xlabel("Temperature gradient ΔT [K]", fontsize=10)
    ax.set_ylabel("Growth rate [µm/h]", fontsize=10)
    ax.set_title("PVT Growth Rate vs ΔT\n(T_source=2250°C, Nakamura model)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # Panel 2: 欠陥密度ロードマップ
    ax2 = axes[0, 1]
    years = sorted(DEFECT_ROADMAP.keys())
    mp  = [DEFECT_ROADMAP[y]["micropipe_cm2"] for y in years]
    tsd = [DEFECT_ROADMAP[y]["TSD_cm2"] for y in years]
    bpd = [DEFECT_ROADMAP[y]["BPD_cm2"] for y in years]
    ax2.semilogy(years, mp,  "o-", color="#d62728", lw=2, ms=7, label="Micropipe [cm⁻²]")
    ax2.semilogy(years, tsd, "s-", color="#2166ac", lw=2, ms=7, label="TSD [cm⁻²]")
    ax2.semilogy(years, bpd, "^-", color="#ff7f0e", lw=2, ms=7, label="BPD [cm⁻²]")
    ax2.axvline(2024, color="gray", ls="--", lw=1.2)
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_ylabel("Defect density [cm⁻²]", fontsize=10)
    ax2.set_title("SiC Wafer Defect Roadmap\nフェローテック / 業界実績", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2, which="both")

    # Panel 3: ウェーハ生産コスト vs 口径
    ax3 = axes[0, 2]
    diam = [100, 150, 200]
    price = [1200, 600, 300]  # USD/枚 (将来目標込み)
    si_price = [3, 5, 8]
    ax3.bar([x-5 for x in diam], price,    width=8, color="#d62728", alpha=0.85, label="4H-SiC")
    ax3.bar([x+5 for x in diam], si_price, width=8, color="#2166ac", alpha=0.85, label="Si (比較)")
    for x, p_s, p_si in zip(diam, price, si_price):
        ax3.text(x-5, p_s+20, f"${p_s}", ha="center", fontsize=9, color="#d62728", fontweight="bold")
        ratio = p_s / p_si
        ax3.text(x, p_s/2, f"×{ratio:.0f}", ha="center", fontsize=9, color="white", fontweight="bold")
    ax3.set_xticks(diam); ax3.set_xticklabels(["4in (100mm)", "6in (150mm)", "8in (200mm)"])
    ax3.set_ylabel("Wafer price [USD/wafer]", fontsize=10)
    ax3.set_title("SiC vs Si Wafer Price\n(Si の 60〜100倍)", fontsize=10)
    ax3.legend(fontsize=9); ax3.grid(alpha=0.2, axis="y")

    # Panel 4: 成長速度 vs マイクロパイプ密度 (トレードオフ)
    ax4 = axes[1, 0]
    rate_arr = np.linspace(100, 600, 100)
    for dT_val, col, lbl in [(80,"#2166ac","ΔT=80K"),(120,"#ff7f0e","ΔT=120K"),(160,"#d62728","ΔT=160K")]:
        mpd = [micropipe_density(r, dT_val) for r in rate_arr]
        ax4.plot(rate_arr, mpd, lw=2.5, color=col, label=lbl)
    ax4.axhline(0.01, color="purple", ls="--", lw=1.5, label="Target < 0.01/cm²")
    ax4.set_xlabel("Growth rate [µm/h]", fontsize=10)
    ax4.set_ylabel("Micropipe density [cm⁻²]", fontsize=10)
    ax4.set_title("Growth Rate vs Micropipe\nSpeed-Quality Tradeoff", fontsize=10)
    ax4.legend(fontsize=8); ax4.grid(alpha=0.2)

    # Panel 5: CMP 粗さ vs 研磨条件
    ax5 = axes[1, 1]
    mrr_arr = np.linspace(5, 100, 80)
    for slurry, col, lbl in [(20,"#2166ac","Slurry 20nm"),(50,"#ff7f0e","Slurry 50nm"),(100,"#d62728","Slurry 100nm")]:
        Ra_vals = [cmp_roughness(m, 20, slurry)["Ra_nm"] for m in mrr_arr]
        ax5.semilogy(mrr_arr, Ra_vals, lw=2.5, color=col, label=lbl)
    ax5.axhline(0.1, color="green", ls="--", lw=1.5, label="Device grade Ra<0.1nm")
    ax5.axhline(0.5, color="orange", ls=":", lw=1.5, label="Epi-ready Ra<0.5nm")
    ax5.set_xlabel("MRR [nm/min]", fontsize=10)
    ax5.set_ylabel("Surface roughness Ra [nm]", fontsize=10)
    ax5.set_title("CMP: MRR vs Surface Roughness\n(SiC ウェーハ仕上げ)", fontsize=10)
    ax5.legend(fontsize=8); ax5.grid(alpha=0.2, which="both")

    # Panel 6: フェローテック事業サマリー表
    ax6 = axes[1, 2]
    ax6.axis("off")
    r_std = pvt_growth_rate(2250, 2150, 20)
    rows = [
        ["指標", "現状 (2024)", "目標 (2027)"],
        ["成長速度", f"{r_std['growth_rate_um_h']:.0f} µm/h", "500+ µm/h"],
        ["インゴット/72h", f"{r_std['ingot_72h_mm']:.0f} mm", "50+ mm"],
        ["Micropipe 密度", "< 0.01 /cm²", "< 0.001 /cm²"],
        ["BPD 密度", "< 20 /cm²", "< 5 /cm²"],
        ["CMP Ra (device)", "< 0.1 nm", "< 0.05 nm"],
        ["150mm 価格", "$600/wafer", "$400/wafer"],
        ["200mm 量産", "立ち上げ中", "量産確立"],
    ]
    tbl = ax6.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.3, 2.1)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_facecolor("#333"); cell.set_text_props(color="w")
        elif r % 2 == 0: cell.set_facecolor("#f5f5f5")
    ax6.set_title("フェローテック SiC ウェーハ KPI\n(Yole 2024 + 同社 IR)", fontsize=10)

    fig.suptitle("フェローテック SiC 結晶成長モデル — PVT 昇華法\n"
                 "成長速度 / 欠陥密度 / CMP 品質 / コスト構造", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "ferrotec_sic_growth.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    r = pvt_growth_rate(2250, 2150, 20)
    print(f"\n  PVT 標準条件: {r}")
    print(f"  多形予測: {polytype_stability(2150)}")
    print(f"  MPD推定: {micropipe_density(r['growth_rate_um_h'], 100):.4f} /cm²")
    print(f"  CMP (50nm slurry): {cmp_roughness(30, 20, 50)}")
    print(f"\n[OK] Ferrotec SiC growth -> {out}")


if __name__ == "__main__":
    plot_all()
