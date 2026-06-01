"""
信越化学 Si ウェーハ製造モデル — CZ 法 (チョクラルスキー法)
=============================================================
信越化学工業 (4063.T) の主力事業: Si 単結晶ウェーハ製造。
世界シェア約 30%、フェローテック SiC モデルとの直接比較が可能。

工程:
    多結晶 Si (ポリシリコン) → CZ 炉で単結晶引き上げ → インゴット
    → ワイヤーソー切断 → ラッピング → エッチング → CMP → 検査

モデル内容:
    1. CZ 成長速度       — von Mises + 引き上げ速度条件
    2. 酸素濃度制御      — Czochralski 溶融シリコン中の O 分布
    3. 結晶欠陥密度      — COP (Crystal-Originated Particle) モデル
    4. CMP 表面品質      — Preston 式 (フェローテックと同形式)
    5. Si vs SiC 総合比較 — コスト・品質・用途の定量比較
    6. 信越化学 DCF 簡易試算

実行:
    python fem/shinetsu_si_wafer_model.py

参考文献:
    Czochralski (1918) — 結晶引き上げ法 (100年越えの基礎技術)
    Abe et al. (1966) Jpn J Appl Phys — Si CZ 酸素制御
    Voronkov (1982) J Crystal Growth — V/G 則 (欠陥制御の基礎)
    信越化学 IR / SEMI SiC-to-Si コスト比較
"""

import math, os, sys
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

kB = 8.617e-5   # eV/K

# ── Si 物性 ─────────────────────────────────────────────────────────────────
SI_PROPS = {
    "T_melt_K":    1687,     # Si 融点
    "rho_liq":     2570,     # 溶融 Si 密度 [kg/m³]
    "rho_sol":     2329,     # 固体 Si 密度
    "lambda_sol":  140,      # 熱伝導率 [W/(m·K)]
    "L_fusion":    1.8e6,    # 融解潜熱 [J/kg]
    "alpha_th":    2.6e-6,   # 線熱膨張係数 [/K]
    "E_GPa":       130,
    "mu_bulk_cm2Vs": 1400,   # バルク電子移動度
}

# ── CZ 炉パラメータ ───────────────────────────────────────────────────────────
CZ_PARAMS = {
    "T_melt_K":      1687,
    "pull_rate_mm_min": 1.0,   # 標準引き上げ速度 [mm/min]
    "crucible_rpm":  10,        # ルツボ回転数 [rpm]
    "crystal_rpm":  -5,         # 結晶回転数 [rpm] (逆回転)
    "G_K_cm":       30,         # 固液界面温度勾配 [K/cm]
    "O_concentration_ppma": 12, # 酸素濃度 [ppma] (SEMI規格 10-20)
}

# ── ウェーハサイズ別スペック ──────────────────────────────────────────────────
WAFER_SPECS = {
    "200mm (8in)": {
        "D_mm": 200, "t_um": 725, "flat_zone_mm": 5,
        "price_USD": 80,   "yield_pct": 97,
        "color": "#9467bd", "era": "成熟 (主力)",
    },
    "300mm (12in)": {
        "D_mm": 300, "t_um": 775, "flat_zone_mm": 3,
        "price_USD": 150,  "yield_pct": 95,
        "color": "#2166ac", "era": "最先端 (主力)",
    },
    "450mm (18in)": {
        "D_mm": 450, "t_um": 925, "flat_zone_mm": 2,
        "price_USD": 500,  "yield_pct": 85,  # 開発中
        "color": "#d62728", "era": "開発中 (2030+?)",
    },
}


# ════════════════════════════════════════════════════════════════════════════
# 1. CZ 成長速度 / Voronkov V/G 則
# ════════════════════════════════════════════════════════════════════════════

def cz_critical_vg(T_K: float = 1687) -> float:
    """
    Voronkov (1982) 臨界 V/G 比 [cm²/(min·K)]。
    V/G > Vc/G: 空孔支配 (V-rich, COP 発生)
    V/G < Vc/G: 自己格子間支配 (I-rich, 転位クラスター)
    """
    # Voronkov 定数 (Si における実験値)
    return 1.34e-3   # cm²/(min·K)


def defect_type(pull_rate_mm_min: float, G_K_cm: float) -> dict:
    """
    V/G 則から欠陥タイプと密度を予測。
    """
    V = pull_rate_mm_min / 10   # mm/min → cm/min
    VG = V / G_K_cm
    Vc = cz_critical_vg()

    if VG > Vc * 1.2:
        defect = "V-rich (COP dominant)"
        cop_density = 1e5 * (VG / Vc - 1) ** 2   # cm⁻³ 概算
    elif VG < Vc * 0.8:
        defect = "I-rich (dislocation cluster)"
        cop_density = 0
    else:
        defect = "Perfect crystal zone"
        cop_density = 100   # 最小

    return {
        "V_cm_min":      round(V, 4),
        "G_K_cm":        G_K_cm,
        "VG_ratio":      round(VG, 5),
        "Vc_ratio":      round(Vc, 5),
        "defect_type":   defect,
        "COP_cm3":       round(cop_density, 0),
        "perfect_zone":  0.8 <= VG/Vc <= 1.2,
    }


def ingot_growth_time(D_mm: float, L_mm: float = 500,
                       pull_rate: float = 1.0) -> dict:
    """
    インゴット引き上げ時間と生産性。
    D_mm: 目標ウェーハ径, L_mm: インゴット長さ
    """
    # 実際の引き上げ速度はウェーハ径が大きいほど遅い
    rate_factor = (200 / D_mm) ** 0.5
    actual_rate = pull_rate * rate_factor   # mm/min

    time_h = L_mm / actual_rate / 60
    wafers = int(L_mm / WAFER_SPECS.get(
        f"{D_mm}mm", {"t_um": 750})["t_um"] * 1000 * 0.85)  # kerf + yield

    return {
        "D_mm":           D_mm,
        "actual_rate_mm_min": round(actual_rate, 3),
        "growth_time_h":  round(time_h, 1),
        "wafers_per_ingot": wafers,
        "energy_kWh":     round(time_h * 200, 0),  # ~200kW 炉
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. 酸素濃度 → デバイス影響
# ════════════════════════════════════════════════════════════════════════════

def oxygen_effect(O_ppma: float) -> dict:
    """
    酸素濃度 [ppma] → 移動度・ゲッタリング・機械強度への影響。
    SEMI M53 規格: 10-25 ppma
    """
    # 酸素散乱による移動度低下
    mu_reduction_pct = max(0, (O_ppma - 10) * 0.5)  # 10ppma以上で低下

    # 内部ゲッタリング効果 (重金属汚染のトラップ)
    gettering = "excellent" if 12 <= O_ppma <= 18 else \
                "good" if 10 <= O_ppma <= 20 else "poor"

    # 酸素析出物による機械強度向上
    strength_factor = 1.0 + (O_ppma / 15) * 0.15

    return {
        "O_ppma":            O_ppma,
        "mu_reduction_pct":  round(mu_reduction_pct, 1),
        "gettering":         gettering,
        "strength_factor":   round(strength_factor, 3),
        "SEMI_compliant":    10 <= O_ppma <= 25,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. CMP 表面品質 (フェローテックと同形式)
# ════════════════════════════════════════════════════════════════════════════

def si_cmp_roughness(MRR_nm_min: float, pressure_kPa: float,
                      slurry_size_nm: float = 30) -> dict:
    """Si CMP: SiC より軟らかく Ra が出やすい。"""
    # Si は SiC の 1/4 の硬さ → 同条件でより平滑
    Ra = 0.006 * (slurry_size_nm * pressure_kPa / 100) ** 0.35 / (MRR_nm_min / 10) ** 0.25
    return {
        "Ra_nm":    round(Ra, 4),
        "quality":  "device grade" if Ra < 0.1 else "polished" if Ra < 0.3 else "lapped",
        "MRR":      MRR_nm_min,
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. Si vs SiC 総合比較
# ════════════════════════════════════════════════════════════════════════════

def si_vs_sic_comparison() -> dict:
    """Si (信越化学) vs SiC (フェローテック) の定量比較。"""
    return {
        "Si (信越化学)": {
            "成長法":       "CZ法 (1415°C)",
            "成長速度":     "1.0 mm/min",
            "ウェーハ価格": "$80-150 /枚 (300mm)",
            "欠陥密度":     "COP ~10³ /cm³",
            "移動度":       "1400 cm²/Vs (電子)",
            "バンドギャップ": "1.12 eV",
            "熱伝導率":     "130 W/(m·K)",
            "市場規模":     "$12B (2024)",
            "成長率":       "+6%/y",
            "主用途":       "ロジック/メモリ/パワー",
        },
        "SiC (フェローテック)": {
            "成長法":       "PVT昇華法 (2250°C)",
            "成長速度":     "500 µm/h = 0.008 mm/min",
            "ウェーハ価格": "$600 /枚 (150mm)",
            "欠陥密度":     "MPD <0.01 /cm²",
            "移動度":       "900 cm²/Vs (電子)",
            "バンドギャップ": "3.26 eV",
            "熱伝導率":     "490 W/(m·K)",
            "市場規模":     "$0.8B (2024)",
            "成長率":       "+25%/y",
            "主用途":       "EV/産業パワー",
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. 信越化学 簡易 DCF
# ════════════════════════════════════════════════════════════════════════════

def shinetsu_dcf(scenario: str = "base") -> dict:
    """
    信越化学 (4063) 簡易 DCF。単位: 億円。
    FY2024 実績ベース。
    """
    scenarios = {
        "bear": {"rev_growth": 0.03, "op_margin": 0.28, "wacc": 0.09, "terminal_g": 0.02},
        "base": {"rev_growth": 0.06, "op_margin": 0.32, "wacc": 0.08, "terminal_g": 0.03},
        "bull": {"rev_growth": 0.10, "op_margin": 0.36, "wacc": 0.07, "terminal_g": 0.035},
    }
    s = scenarios[scenario]
    rev_fy24 = 23000   # 億円 (FY2024 実績)
    shares   = 4.1e8   # 発行済株式数

    fcf_total = 0
    rev = rev_fy24
    for yr in range(1, 11):
        rev *= (1 + s["rev_growth"])
        op   = rev * s["op_margin"]
        fcf  = op * 0.65   # 税後 FCF
        pv   = fcf / (1 + s["wacc"]) ** yr
        fcf_total += pv

    # ターミナルバリュー
    tv = fcf_total * (1 + s["terminal_g"]) / (s["wacc"] - s["terminal_g"]) * 0.3
    equity = (fcf_total + tv) * 1e8   # 億円 → 円
    fair_value = equity / shares

    return {
        "scenario":    scenario,
        "fair_value_jpy": round(fair_value),
        "rev_fy24_B":  round(rev_fy24 / 1e4, 1),
        "op_margin":   s["op_margin"],
        "wacc":        s["wacc"],
    }


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel 1: V/G 則 — 欠陥タイプマップ
    ax = axes[0, 0]
    pull_arr = np.linspace(0.3, 3.0, 200)
    G_arr    = np.linspace(15, 60, 200)
    PP, GG   = np.meshgrid(pull_arr, G_arr)
    VG_map   = (PP / 10) / GG
    Vc       = cz_critical_vg()
    ratio_map = VG_map / Vc
    cmap_custom = plt.cm.RdYlGn_r
    cp = ax.contourf(PP, GG, ratio_map, levels=20, cmap=cmap_custom, vmin=0.3, vmax=2.0)
    ax.contour(PP, GG, ratio_map, levels=[0.8, 1.0, 1.2],
               colors=["blue","black","red"], linewidths=[1.5, 2.5, 1.5])
    plt.colorbar(cp, ax=ax, label="V/G / (V/G)_critical")
    ax.axhline(CZ_PARAMS["G_K_cm"], color="white", ls="--", lw=1.5, label=f"標準 G={CZ_PARAMS['G_K_cm']}K/cm")
    ax.axvline(CZ_PARAMS["pull_rate_mm_min"], color="cyan", ls="--", lw=1.5, label="標準引き上げ速度")
    ax.set_xlabel("Pull rate [mm/min]", fontsize=10)
    ax.set_ylabel("Temperature gradient G [K/cm]", fontsize=10)
    ax.set_title("Voronkov V/G Defect Map\n(緑=Perfect zone, 赤=COP多発)", fontsize=10)
    ax.legend(fontsize=8)

    # Panel 2: 酸素濃度 → 移動度・強度
    ax2 = axes[0, 1]
    O_arr = np.linspace(5, 30, 100)
    mu_red = [oxygen_effect(O)["mu_reduction_pct"] for O in O_arr]
    str_fac = [oxygen_effect(O)["strength_factor"] for O in O_arr]
    ax2_twin = ax2.twinx()
    ax2.plot(O_arr, mu_red, "r-", lw=2.5, label="移動度低下 [%]")
    ax2_twin.plot(O_arr, str_fac, "b--", lw=2.5, label="機械強度係数")
    ax2.axvspan(10, 20, alpha=0.1, color="green", label="SEMI M53 規格範囲")
    ax2.axvline(CZ_PARAMS["O_concentration_ppma"], color="gray", ls=":", lw=2)
    ax2.set_xlabel("Oxygen concentration [ppma]", fontsize=10)
    ax2.set_ylabel("Mobility reduction [%]", color="red", fontsize=10)
    ax2_twin.set_ylabel("Mechanical strength factor", color="blue", fontsize=10)
    ax2.set_title("O Concentration Effects\nMobility vs Mechanical Strength Tradeoff", fontsize=10)
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labs1+labs2, fontsize=8)
    ax2.grid(alpha=0.2)

    # Panel 3: ウェーハ径スケーリング
    ax3 = axes[0, 2]
    D_arr = np.array([100, 150, 200, 300, 450])
    prices = [20, 40, 80, 150, 500]
    areas  = [math.pi*(D/2)**2 for D in D_arr]
    price_per_cm2 = [p/a*100 for p,a in zip(prices, areas)]
    times  = [ingot_growth_time(D, 500)["growth_time_h"] for D in D_arr]

    ax3_twin = ax3.twinx()
    ax3.bar(range(len(D_arr)), prices, color=["#9467bd","#ff7f0e","#2166ac","#d62728","#e74c3c"],
            alpha=0.8, edgecolor="k", lw=0.8)
    ax3_twin.plot(range(len(D_arr)), price_per_cm2, "ko-", lw=2, ms=7, label="$/cm²")
    ax3.set_xticks(range(len(D_arr)))
    ax3.set_xticklabels([f"{D}mm" for D in D_arr])
    ax3.set_ylabel("Wafer price [USD]", fontsize=10)
    ax3_twin.set_ylabel("Price per cm² [USD/cm²]", fontsize=10)
    ax3.set_title("Si Wafer Price Scaling\n(口径大 → $/cm² 低下)", fontsize=10)
    ax3_twin.legend(fontsize=9); ax3.grid(alpha=0.2, axis="y")
    # 開発中フラグ
    ax3.text(4, 430, "開発中", ha="center", fontsize=10, color="red", fontweight="bold")

    # Panel 4: Si vs SiC 比較バー
    ax4 = axes[1, 0]
    ax4.axis("off")
    comp = si_vs_sic_comparison()
    keys = list(list(comp.values())[0].keys())
    rows = [["指標", "Si (信越化学)", "SiC (フェローテック)"]]
    for k in keys:
        rows.append([k, comp["Si (信越化学)"][k], comp["SiC (フェローテック)"][k]])
    tbl = ax4.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.3, 1.85)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_facecolor("#333"); cell.set_text_props(color="w")
        elif c == 1: cell.set_facecolor("#e8f4f8")
        elif c == 2: cell.set_facecolor("#fef9e7")
        elif r % 2 == 0: cell.set_facecolor("#f8f8f8")
    ax4.set_title("Si vs SiC ウェーハ総合比較\n(信越化学 vs フェローテック)", fontsize=10)

    # Panel 5: COP 密度 vs 引き上げ速度
    ax5 = axes[1, 1]
    pull_rates = np.linspace(0.3, 2.5, 100)
    G_vals = [20, 30, 40]
    for G, col, lbl in zip(G_vals, ["#2166ac","#ff7f0e","#d62728"], [f"G={G}K/cm" for G in G_vals]):
        cops = [max(0, defect_type(p, G)["COP_cm3"]) for p in pull_rates]
        ax5.semilogy([p for p,c in zip(pull_rates,cops) if c > 0],
                     [c for c in cops if c > 0], lw=2.5, color=col, label=lbl)
    ax5.axvline(1.0, color="gray", ls="--", lw=1.5, label="標準引き上げ速度")
    ax5.axhline(1e4, color="purple", ls=":", lw=1.5, label="デバイス許容限界")
    ax5.set_xlabel("Pull rate [mm/min]", fontsize=10)
    ax5.set_ylabel("COP density [cm⁻³]", fontsize=10)
    ax5.set_title("COP Density vs Pull Rate\n(Voronkov V/G model)", fontsize=10)
    ax5.legend(fontsize=8); ax5.grid(alpha=0.2, which="both")

    # Panel 6: 信越化学 DCF サマリー
    ax6 = axes[1, 2]
    scenarios = ["bear", "base", "bull"]
    colors6   = ["#2166ac", "#ff7f0e", "#d62728"]
    dcf_results = {s: shinetsu_dcf(s) for s in scenarios}
    fvs = [dcf_results[s]["fair_value_jpy"] for s in scenarios]
    bars = ax6.bar(scenarios, fvs, color=colors6, alpha=0.85, edgecolor="k", lw=0.8)

    # 現在株価 (2024 年末時点概算)
    current_price = 5800 * 100  # 約 ¥5,800 (概算)
    ax6.axhline(current_price, color="black", ls="--", lw=2, label=f"現在値概算 ¥{current_price:,}")
    for bar, fv in zip(bars, fvs):
        ax6.text(bar.get_x()+bar.get_width()/2, fv+10000, f"¥{fv:,}", ha="center", fontsize=9, fontweight="bold")
    ax6.set_ylabel("Fair value per share [JPY]", fontsize=10)
    ax6.set_title("信越化学 簡易 DCF\n(3シナリオ / FY2024 ベース)", fontsize=10)
    ax6.legend(fontsize=9); ax6.grid(alpha=0.2, axis="y")
    ax6.set_xticklabels(["Bear\n(成長3%)", "Base\n(成長6%)", "Bull\n(成長10%)"])

    fig.suptitle("信越化学工業 (4063) — Si ウェーハ CZ 成長モデル\n"
                 "Voronkov V/G則 / 酸素制御 / Si vs SiC 比較 / DCF",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "shinetsu_si_wafer.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # サマリー出力
    print("\n" + "="*60)
    print("  信越化学 Si ウェーハモデル サマリー")
    print("="*60)

    r_base = defect_type(1.0, 30)
    print(f"\n【CZ 標準条件】 引き上げ 1.0mm/min, G=30K/cm")
    print(f"  V/G 比率: {r_base['VG_ratio']:.5f} (臨界値: {r_base['Vc_ratio']:.5f})")
    print(f"  欠陥タイプ: {r_base['defect_type']}")
    print(f"  COP 密度: {r_base['COP_cm3']:.0f} /cm³")

    for D in [200, 300]:
        ig = ingot_growth_time(D, 500)
        print(f"\n【{D}mm インゴット (500mm長)】")
        print(f"  引き上げ速度: {ig['actual_rate_mm_min']:.3f} mm/min")
        print(f"  成長時間: {ig['growth_time_h']:.0f} h")
        print(f"  ウェーハ数: {ig['wafers_per_ingot']} 枚/インゴット")
        print(f"  消費電力: {ig['energy_kWh']:.0f} kWh")

    print(f"\n【酸素濃度 12ppma】: {oxygen_effect(12)}")
    print(f"\n【CMP】: {si_cmp_roughness(30, 20, 30)}")

    print(f"\n【DCF フェアバリュー】")
    for s in scenarios:
        r = dcf_results[s]
        print(f"  {s:5s}: ¥{r['fair_value_jpy']:>8,} /株  (OP率{r['op_margin']*100:.0f}%, WACC{r['wacc']*100:.0f}%)")

    print(f"\n【Si vs SiC 成長速度比較】")
    print(f"  Si CZ:  1.0 mm/min = 60,000 µm/h")
    print(f"  SiC PVT: 0.008 mm/min = 500 µm/h  (120× 遅い)")
    print(f"  → SiC ウェーハが Si の 100× 高いのは当然")

    print(f"\n[OK] 信越化学 Si wafer -> {out}")


if __name__ == "__main__":
    plot_all()
