"""
東京ガス パイプライン劣化・余寿命モデル
=========================================
都市ガス配管の物理モデル化。wafer-proc-sim の「物理 × ML」を
エネルギーインフラ領域に展開したポートフォリオ拡張。

1. 腐食モデル (Corrosion)
   - CO₂ 腐食 (de Waard & Milliams 1975 修正式)
   - 土壌腐食 (土壌比抵抗・水分依存)
   - 腐食速度 → 残肉厚 → 破裂圧力低下

2. 圧力サイクル疲労 (Paris 則)
   - ガス需要変動 → 内圧変動 → hoop 応力振幅
   - 亀裂進展: da/dN = C(ΔK)^m
   - 余寿命サイクル数

3. 余寿命予測 (RUL)
   - Weibull 確率分布
   - GP サロゲートによる不確かさ定量化

4. 水素混焼影響 (H₂ Blending)
   - 水素脆化による破壊靱性低下 (Nelson 曲線)
   - H₂ 混合率 vs 許容応力
   - 東京ガス 2030 目標 (H₂ 20% 混焼) への含意

実行:
    python fem/tokyogas_pipeline_model.py
    python fem/tokyogas_pipeline_model.py --h2-blend
"""

import argparse
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

kB = 8.617e-5  # eV/K

# ── 配管材料パラメータ ────────────────────────────────────────────────────────
PIPE_MATERIALS = {
    "API5L_X65": {           # 高圧幹線用鋼管（標準）
        "SMYS_MPa": 450,     # 降伏強度
        "UTS_MPa":  535,     # 引張強度
        "KIc_MPa_m05": 80,   # 破壊靱性
        "E_GPa": 207,
        "C_paris": 6.9e-12,  # Paris 則定数 [m/cycle / (MPa√m)^m]
        "m_paris": 3.0,      # Paris 則指数
        "color": "#2166ac",
        "label": "API 5L X65 (高圧幹線)",
    },
    "API5L_X52": {           # 中圧配管用
        "SMYS_MPa": 360,
        "UTS_MPa":  455,
        "KIc_MPa_m05": 65,
        "E_GPa": 207,
        "C_paris": 8.5e-12,
        "m_paris": 3.0,
        "color": "#d62728",
        "label": "API 5L X52 (中圧)",
    },
    "PE100":  {              # ポリエチレン低圧配管
        "SMYS_MPa": 10,
        "UTS_MPa":  25,
        "KIc_MPa_m05": 3.5,
        "E_GPa": 0.8,
        "C_paris": 1e-9,
        "m_paris": 2.5,
        "color": "#ff7f0e",
        "label": "PE100 (低圧)",
    },
}

# ── 標準配管仕様 ─────────────────────────────────────────────────────────────
PIPE_SPEC = {
    "D_mm":   508,     # 外径 [mm] (20インチ 幹線)
    "t_mm":   12.7,    # 肉厚 [mm]
    "P_MPa":  7.0,     # 設計圧力 [MPa]
    "P_min":  4.0,     # 最小運転圧 [MPa] (季節変動)
    "L_km":   100,     # 管路延長 [km]
}


# ════════════════════════════════════════════════════════════════════════════
# 1. 腐食モデル
# ════════════════════════════════════════════════════════════════════════════

def co2_corrosion_rate(T_C: float, pCO2_bar: float, pH: float = 6.5) -> float:
    """
    CO₂ 腐食速度 [mm/year] — de Waard & Milliams (1975) 修正式。
    T_C    : 温度 [°C]
    pCO2   : CO₂ 分圧 [bar]
    pH     : ガス中水分 pH (凝縮水)
    """
    T_K = T_C + 273.15
    # Arrhenius 型腐食速度
    log_CR = 5.8 - (1710 / T_K) + 0.67 * math.log10(pCO2_bar)
    CR = 10 ** log_CR  # mm/year (中性 pH 基準)
    # pH 補正: pH 上昇で腐食速度低下
    pH_factor = 10 ** (-(pH - 7.0) * 0.5)
    return CR * pH_factor


def soil_corrosion_rate(resistivity_ohm_m: float, moisture_pct: float) -> float:
    """
    土壌腐食速度 [mm/year]。
    土壌比抵抗と水分含量に依存（BS EN 12954 準拠）。
    """
    # 比抵抗が低い（電解質豊富）→ 腐食速度高い
    rho_factor = 1.0 / (1.0 + resistivity_ohm_m / 50)
    moist_factor = (moisture_pct / 20) ** 0.5
    return 0.05 * rho_factor * moist_factor  # mm/year


def wall_thickness_remaining(t0_mm: float, CR_mmyr: float, age_yr: float) -> float:
    """残肉厚 [mm]。"""
    return max(t0_mm - CR_mmyr * age_yr, 0.0)


def burst_pressure(t_mm: float, D_mm: float, SMYS_MPa: float) -> float:
    """
    破裂圧力 [MPa] — Barlow 式 (hoop stress)。
    P_burst = 2 * t * SMYS / D
    """
    return 2 * t_mm * SMYS_MPa / D_mm


# ════════════════════════════════════════════════════════════════════════════
# 2. 圧力サイクル疲労 (Paris 則)
# ════════════════════════════════════════════════════════════════════════════

def hoop_stress_amplitude(delta_P_MPa: float, D_mm: float, t_mm: float) -> float:
    """周方向応力振幅 [MPa]。δσ = δP * D / (2t)"""
    return delta_P_MPa * D_mm / (2 * t_mm)


def stress_intensity_factor(delta_sigma_MPa: float, a_mm: float) -> float:
    """
    応力拡大係数振幅 [MPa√m]。
    ΔK = Δσ * √(π*a) * F (形状係数 F≈1.12 for surface crack)
    """
    a_m = a_mm * 1e-3
    F = 1.12
    return delta_sigma_MPa * math.sqrt(math.pi * a_m) * F


def paris_crack_growth(material: str, a0_mm: float,
                       delta_P_MPa: float, n_cycles: int,
                       D_mm: float = 508, t_mm: float = 12.7) -> np.ndarray:
    """
    Paris 則による亀裂進展シミュレーション。
    a0_mm     : 初期欠陥サイズ [mm]
    delta_P   : 圧力振幅 [MPa]
    n_cycles  : シミュレーションサイクル数
    returns   : 亀裂サイズ履歴 [mm]
    """
    m = PIPE_MATERIALS[material]
    C = m["C_paris"]
    exp = m["m_paris"]
    a = a0_mm
    history = [a]
    for _ in range(n_cycles):
        ds = hoop_stress_amplitude(delta_P_MPa, D_mm, t_mm)
        dK = stress_intensity_factor(ds, a)
        da = C * (dK ** exp) * 1e3  # m → mm
        a = min(a + da, t_mm)
        history.append(a)
        if a >= t_mm:
            break
    return np.array(history)


def fatigue_life(material: str, a0_mm: float, delta_P_MPa: float,
                 D_mm: float = 508, t_mm: float = 12.7) -> int:
    """亀裂が貫通するまでのサイクル数（余寿命）。"""
    m = PIPE_MATERIALS[material]
    C = m["C_paris"]
    exp = m["m_paris"]
    a = a0_mm
    N = 0
    while a < t_mm:
        ds = hoop_stress_amplitude(delta_P_MPa, D_mm, t_mm)
        dK = stress_intensity_factor(ds, a)
        da = C * (dK ** exp) * 1e3
        if da <= 0:
            break
        a += da
        N += 1
        if N > 1e8:
            break
    return N


# ════════════════════════════════════════════════════════════════════════════
# 3. 水素混焼影響モデル (H₂ Blending)
# ════════════════════════════════════════════════════════════════════════════

def h2_fracture_toughness(KIc0: float, h2_pct: float) -> float:
    """
    H₂ 混焼率 → 破壊靱性低下。
    水素脆化 (Hydrogen Embrittlement): Nelson 曲線ベースの簡易モデル。
    KIc(H₂) = KIc0 * (1 - α * h2_pct/100)
    α ≈ 0.35 for carbon steel (Cialone & Holbrook 1985)
    """
    alpha = 0.35
    return KIc0 * max(1.0 - alpha * h2_pct / 100, 0.30)


def h2_allowable_stress(SMYS: float, h2_pct: float) -> float:
    """
    H₂ 混合率 → 許容応力低下。
    ASME B31.12 (水素配管規格) に基づく係数。
    """
    # H₂ 100% では設計係数が 0.5 倍（ASME B31.12 HB 条件）
    factor = 1.0 - 0.5 * (h2_pct / 100) ** 0.7
    return SMYS * factor


def h2_leakage_risk(h2_pct: float, P_MPa: float) -> float:
    """
    H₂ 比率 × 圧力 → 漏洩リスク指標。
    H₂ は分子が小さく浸透しやすい（拡散係数 4× CH₄）。
    """
    D_ratio = 1.0 + 3.0 * (h2_pct / 100)  # 拡散係数比
    return P_MPa * D_ratio * (h2_pct / 100 + 0.01)


# ════════════════════════════════════════════════════════════════════════════
# 4. 余寿命 Weibull 予測
# ════════════════════════════════════════════════════════════════════════════

def rul_weibull(age_yr: float, design_life_yr: float = 40,
                beta: float = 2.5) -> dict:
    """
    残余寿命 (RUL) の Weibull 確率。
    beta  : 形状パラメータ（摩耗故障 β>1, 老化 β≈3）
    """
    eta = design_life_yr / ((-math.log(0.5)) ** (1 / beta))  # 中央値基準
    survival = math.exp(-(age_yr / eta) ** beta)
    hazard = (beta / eta) * (age_yr / eta) ** (beta - 1)
    rul_median = eta * ((-math.log(0.5)) ** (1 / beta)) - age_yr
    return {
        "age_yr": age_yr,
        "survival_prob": round(survival, 4),
        "hazard_rate": round(hazard, 6),
        "rul_median_yr": round(max(rul_median, 0), 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    spec = PIPE_SPEC

    # ── Panel 1: CO₂ 腐食速度 × 温度 × pCO₂ ─────────────────────────────
    ax = axes[0, 0]
    T_arr = np.linspace(5, 80, 80)
    for pco2, col, lbl in [(0.001, "#2166ac", "pCO₂=0.001bar (低濃度)"),
                            (0.01,  "#ff7f0e", "pCO₂=0.01bar"),
                            (0.1,   "#d62728", "pCO₂=0.1bar (高濃度)")]:
        rates = [co2_corrosion_rate(T, pco2) for T in T_arr]
        ax.semilogy(T_arr, rates, lw=2.5, color=col, label=lbl)
    ax.set_xlabel("Temperature [°C]", fontsize=10)
    ax.set_ylabel("Corrosion rate [mm/year]", fontsize=10)
    ax.set_title("CO₂ Corrosion Rate\n(de Waard & Milliams)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2, which="both")
    ax.axhline(0.1, color="purple", ls="--", lw=1.5, label="Limit 0.1mm/yr")

    # ── Panel 2: 腐食進行 → 破裂圧力低下 (経年劣化) ────────────────────────
    ax2 = axes[0, 1]
    age_arr = np.linspace(0, 50, 100)
    for mat, col in [("API5L_X65", "#2166ac"), ("API5L_X52", "#d62728")]:
        m = PIPE_MATERIALS[mat]
        burst = []
        for age in age_arr:
            CR = co2_corrosion_rate(15, 0.005) + soil_corrosion_rate(30, 20)
            t_rem = wall_thickness_remaining(spec["t_mm"], CR, age)
            bp = burst_pressure(t_rem, spec["D_mm"], m["SMYS_MPa"])
            burst.append(bp)
        ax2.plot(age_arr, burst, lw=2.5, color=col, label=m["label"])
    ax2.axhline(spec["P_MPa"] * 1.5, color="red", ls="--", lw=1.5,
                label=f"MAOP×1.5 = {spec['P_MPa']*1.5:.1f} MPa")
    ax2.axhline(spec["P_MPa"],       color="orange", ls=":", lw=1.5,
                label=f"設計圧 = {spec['P_MPa']:.1f} MPa")
    ax2.set_xlabel("Pipeline age [years]", fontsize=10)
    ax2.set_ylabel("Burst pressure [MPa]", fontsize=10)
    ax2.set_title("Corrosion → Burst Pressure Degradation\n(D=508mm, t=12.7mm)", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2)

    # ── Panel 3: Paris 則 亀裂進展 ──────────────────────────────────────────
    ax3 = axes[0, 2]
    dP = spec["P_MPa"] - spec["P_min"]  # 圧力振幅
    for a0, col, lbl in [(0.5, "#2166ac", "初期欠陥 0.5mm"),
                          (1.0, "#ff7f0e", "初期欠陥 1.0mm"),
                          (2.0, "#d62728", "初期欠陥 2.0mm")]:
        hist = paris_crack_growth("API5L_X65", a0, dP, n_cycles=50000,
                                  D_mm=spec["D_mm"], t_mm=spec["t_mm"])
        ax3.semilogy(np.arange(len(hist)), hist, lw=2, color=col, label=lbl)
    ax3.axhline(spec["t_mm"], color="red", ls="--", lw=1.5, label=f"肉厚 {spec['t_mm']}mm (貫通)")
    ax3.set_xlabel("Pressure cycles", fontsize=10)
    ax3.set_ylabel("Crack size [mm]", fontsize=10)
    ax3.set_title("Paris Law Crack Growth\n(API 5L X65, ΔP=3.0 MPa)", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(alpha=0.2, which="both")

    # ── Panel 4: H₂ 混焼率 → 材料強度・リスク ──────────────────────────────
    ax4 = axes[1, 0]
    h2_arr = np.linspace(0, 100, 100)
    mat = PIPE_MATERIALS["API5L_X65"]
    KIc_vals = [h2_fracture_toughness(mat["KIc_MPa_m05"], h) for h in h2_arr]
    allow_vals = [h2_allowable_stress(mat["SMYS_MPa"], h) for h in h2_arr]
    ax4_twin = ax4.twinx()
    ax4.plot(h2_arr, KIc_vals,   color="#2166ac", lw=2.5, label="破壊靱性 KIc")
    ax4_twin.plot(h2_arr, allow_vals, color="#d62728", lw=2.5, ls="--",
                  label="許容応力 [MPa]")
    ax4.axvline(20, color="purple", ls=":", lw=2, label="東京ガス目標 20%")
    ax4.set_xlabel("H₂ blending ratio [%]", fontsize=10)
    ax4.set_ylabel("KIc [MPa√m]", fontsize=10, color="#2166ac")
    ax4_twin.set_ylabel("Allowable stress [MPa]", fontsize=10, color="#d62728")
    ax4.set_title("H₂ Blending → Material Strength\n(Nelson curve, ASME B31.12)", fontsize=10)
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax4.grid(alpha=0.2)

    # ── Panel 5: Weibull 余寿命予測 ──────────────────────────────────────────
    ax5 = axes[1, 1]
    age_arr2 = np.linspace(0, 50, 200)
    for beta, col, lbl in [(1.5, "#2166ac", "β=1.5 (初期欠陥型)"),
                            (2.5, "#ff7f0e", "β=2.5 (摩耗型)"),
                            (3.5, "#d62728", "β=3.5 (老化型)")]:
        surv = [rul_weibull(a, design_life_yr=40, beta=beta)["survival_prob"]
                for a in age_arr2]
        ax5.plot(age_arr2, surv, lw=2.5, color=col, label=lbl)
    ax5.axhline(0.90, color="red", ls="--", lw=1.5, label="点検基準 90%")
    ax5.axvline(40, color="gray", ls=":", lw=1.5, label="設計寿命 40年")
    ax5.set_xlabel("Pipeline age [years]", fontsize=10)
    ax5.set_ylabel("Survival probability", fontsize=10)
    ax5.set_title("RUL Prediction (Weibull)\nGas Pipeline Reliability", fontsize=10)
    ax5.legend(fontsize=8); ax5.grid(alpha=0.2)

    # ── Panel 6: 総合リスクマップ ─────────────────────────────────────────
    ax6 = axes[1, 2]
    ages = np.linspace(10, 50, 40)
    h2_levels = np.linspace(0, 30, 40)
    AA, HH = np.meshgrid(ages, h2_levels)
    risk = np.zeros_like(AA)
    for i in range(len(h2_levels)):
        for j in range(len(ages)):
            CR = co2_corrosion_rate(15, 0.005) + soil_corrosion_rate(30, 20)
            t_rem = wall_thickness_remaining(spec["t_mm"], CR, AA[i, j])
            KIc = h2_fracture_toughness(mat["KIc_MPa_m05"], HH[i, j])
            surv = rul_weibull(AA[i, j])["survival_prob"]
            # 複合リスク指標（正規化）
            risk[i, j] = (1 - surv) * (1 - KIc / mat["KIc_MPa_m05"]) * \
                         (1 - t_rem / spec["t_mm"]) * 100
    cp = ax6.contourf(AA, HH, risk, levels=15, cmap="YlOrRd")
    plt.colorbar(cp, ax=ax6, label="Composite risk index")
    ax6.axvline(40,  color="white", ls="--", lw=1.5, label="設計寿命")
    ax6.axhline(20,  color="cyan",  ls="--", lw=1.5, label="H₂ 20% 目標")
    ax6.set_xlabel("Pipeline age [years]", fontsize=10)
    ax6.set_ylabel("H₂ blending [%]", fontsize=10)
    ax6.set_title("Composite Risk Map\nAge × H₂ blending", fontsize=10)
    ax6.legend(fontsize=8)

    fig.suptitle("東京ガス パイプライン劣化・余寿命モデル\n"
                 "CO₂腐食 / Paris則疲労 / H₂混焼影響 / Weibull RUL",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "tokyogas_pipeline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    _print_summary()
    print(f"\n[OK] Tokyo Gas pipeline model -> {out}")


def _print_summary():
    spec = PIPE_SPEC
    mat = PIPE_MATERIALS["API5L_X65"]
    CR_total = co2_corrosion_rate(15, 0.005) + soil_corrosion_rate(30, 20)

    print("\n" + "=" * 60)
    print("  東京ガス パイプラインモデル サマリー")
    print("=" * 60)
    print(f"\n【配管仕様】 D={spec['D_mm']}mm, t={spec['t_mm']}mm, "
          f"P={spec['P_MPa']}MPa")
    print(f"【総腐食速度】 {CR_total:.4f} mm/year "
          f"(CO₂: {co2_corrosion_rate(15, 0.005):.4f} + "
          f"土壌: {soil_corrosion_rate(30, 20):.4f})")

    for age in [10, 20, 30, 40]:
        t_rem = wall_thickness_remaining(spec["t_mm"], CR_total, age)
        bp = burst_pressure(t_rem, spec["D_mm"], mat["SMYS_MPa"])
        rul = rul_weibull(age)
        print(f"  {age:2d}年後: 残肉厚 {t_rem:.2f}mm, "
              f"破裂圧 {bp:.1f}MPa, "
              f"生存確率 {rul['survival_prob']*100:.1f}%, "
              f"RUL {rul['rul_median_yr']:.0f}年")

    print(f"\n【H₂ 20% 混焼時の影響 (東京ガス 2030 目標)】")
    KIc_0   = h2_fracture_toughness(mat["KIc_MPa_m05"], 0)
    KIc_20  = h2_fracture_toughness(mat["KIc_MPa_m05"], 20)
    sig_0   = h2_allowable_stress(mat["SMYS_MPa"], 0)
    sig_20  = h2_allowable_stress(mat["SMYS_MPa"], 20)
    print(f"  KIc:      {KIc_0:.0f} → {KIc_20:.0f} MPa√m "
          f"({(KIc_20/KIc_0-1)*100:.1f}%)")
    print(f"  許容応力: {sig_0:.0f} → {sig_20:.0f} MPa "
          f"({(sig_20/sig_0-1)*100:.1f}%)")
    print(f"  → 既設 X65 配管は H₂ 20% に対して材料的に概ね許容範囲内")
    print(f"    ただし溶接部・欠陥箇所は個別評価が必要")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2-blend", action="store_true")
    args = ap.parse_args()
    plot_all()


if __name__ == "__main__":
    main()
