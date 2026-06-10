"""
東京ガス LNG タンク熱成層 + 水素製造モデル
==========================================
東京ガスの2大戦略技術を物理モデル化:
  1. LNG タンク熱成層 / ロールオーバーリスク
  2. PEM 水素電解槽 (Green H₂ コスト)

LNG タンク:
    熱成層 → ロールオーバー (急激なBOG発生) がインフラ最大リスク
    GIIGNL / NFPA 59A 準拠の安全管理

水素電解槽 (PEM):
    東京ガスの 2030 目標: H₂ 混焼 20% + 水素タウン実証
    Green H₂ コスト = f(電力コスト, 効率, 設備利用率)

実行:
    python fem/tokyogas_lng_hydrogen_model.py
"""

import math, os, sys
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── LNG 物性 ────────────────────────────────────────────────────────────────
LNG_PROPS = {
    "T_boil_K":    111.15,   # 沸点 @ 1atm
    "rho_liq":     430,      # kg/m³  液体密度
    "rho_vap":     1.8,      # kg/m³  蒸気密度
    "Cp_liq":      3500,     # J/(kg·K)
    "h_vap":       510e3,    # J/kg  蒸発潜熱
    "lambda_liq":  0.19,     # W/(m·K) 熱伝導率
    "mu_liq":      1.2e-4,   # Pa·s 粘性
    "beta":        3.5e-3,   # 1/K  体積膨張係数
}

# ── タンク仕様 ────────────────────────────────────────────────────────────────
TANK_SPEC = {
    "V_m3":         50000,   # 50,000 m³  地下タンク
    "D_m":          60,      # 内径 60m
    "H_m":          30,      # 液高さ 30m
    "U_wall":       0.03,    # W/(m²·K)  断熱壁熱通過率
    "T_ambient_K":  288,     # 外気温 15°C
}


# ════════════════════════════════════════════════════════════════════════════
# 1. LNG 熱成層 / ロールオーバー
# ════════════════════════════════════════════════════════════════════════════

def boil_off_rate(T_liq_K: float, spec: dict = TANK_SPEC) -> float:
    """BOG 発生量 [kg/h] — 壁面熱侵入。"""
    props = LNG_PROPS
    A_wall = math.pi * spec["D_m"] * spec["H_m"] + math.pi * (spec["D_m"]/2)**2
    Q_in = spec["U_wall"] * A_wall * (spec["T_ambient_K"] - T_liq_K)   # W
    return Q_in / props["h_vap"] * 3600  # kg/h


def stratification_index(T_bottom_K: float, T_top_K: float,
                         H_m: float = 30) -> dict:
    """
    熱成層強度指数。Δρ/H が大きいほどロールオーバーリスク高。
    Rayleigh 数で不安定性を評価。
    """
    props = LNG_PROPS
    dT  = T_top_K - T_bottom_K
    drho = props["rho_liq"] * props["beta"] * abs(dT)
    # 簡易 Rayleigh 数 (対流不安定性指標)
    g = 9.81
    nu = props["mu_liq"] / props["rho_liq"]
    alpha_th = props["lambda_liq"] / (props["rho_liq"] * props["Cp_liq"])
    Ra = g * props["beta"] * abs(dT) * H_m**3 / (nu * alpha_th)
    rollover_risk = "HIGH" if Ra > 1e12 else "MEDIUM" if Ra > 1e10 else "LOW"
    return {
        "dT_K": round(dT, 3),
        "drho_kg_m3": round(drho, 4),
        "Ra": round(Ra, 2),
        "rollover_risk": rollover_risk,
    }


def rollover_bog_spike(V_m3: float, dT_layer_K: float) -> float:
    """
    ロールオーバー発生時の瞬間 BOG スパイク [kg/s]。
    上下層が混合 → ΔH の潜熱が一気に放出。
    """
    props = LNG_PROPS
    m_layer = V_m3 / 2 * props["rho_liq"]   # 下層 LNG 質量 [kg]
    dH = props["Cp_liq"] * dT_layer_K       # J/kg 過熱分
    return m_layer * dH / props["h_vap"] / 60  # kg/s (1分で放出と仮定)


# ════════════════════════════════════════════════════════════════════════════
# 2. PEM 水素電解槽モデル
# ════════════════════════════════════════════════════════════════════════════

# PEM 電解槽パラメータ (東芝 H₂One / ITM Power 相当)
PEM_PARAMS = {
    "V_rev_V":      1.23,    # 可逆電圧 [V]
    "V_tn_V":       1.48,    # 熱中立電圧
    "alpha_a":      0.5,     # アノード移動係数
    "alpha_c":      0.5,     # カソード移動係数
    "i0_A_cm2":     1e-7,    # 交換電流密度
    "R_ohm_cm2":    0.18,    # オーム抵抗 [Ω·cm²]
    "eta_faradaic": 0.998,   # ファラデー効率
    "HHV_H2_kWh_kg": 39.4,  # H₂ 高位発熱量
}


def pem_cell_voltage(j_A_cm2: float) -> float:
    """
    PEM セル電圧 [V] — Butler-Volmer + オーム損失。
    V = V_rev + η_act + η_ohm
    """
    p = PEM_PARAMS
    F = 96485  # C/mol
    R = 8.314
    T = 353    # 80°C 動作温度 [K]
    # 活性化過電圧 (アノード支配)
    eta_act = (R * T / (p["alpha_a"] * F)) * math.asinh(j_A_cm2 / (2 * p["i0_A_cm2"]))
    # オーム損失
    eta_ohm = j_A_cm2 * p["R_ohm_cm2"]
    return p["V_rev_V"] + eta_act + eta_ohm


def pem_efficiency(j_A_cm2: float) -> float:
    """PEM 電解効率 [%] = V_tn / V_cell."""
    V = pem_cell_voltage(j_A_cm2)
    return PEM_PARAMS["V_tn_V"] / V * 100


def green_h2_cost(elec_price_jpy_kWh: float,
                  capacity_factor: float = 0.85,
                  capex_jpy_kW: float = 80000) -> dict:
    """
    Green H₂ 製造コスト [円/kg]。
    LCOH = (CAPEX_ann + OPEX) / H₂_production + electricity_cost
    """
    p = PEM_PARAMS
    # 代表電流密度での効率
    j_rep = 2.0   # A/cm²
    eta = pem_efficiency(j_rep) / 100
    kwh_per_kg = p["HHV_H2_kWh_kg"] / eta

    # 電力コスト
    elec_cost = kwh_per_kg * elec_price_jpy_kWh

    # 設備費 (20年, 割引率5%)
    ann_factor = 0.08   # 年金係数近似
    full_load_h = 8760 * capacity_factor
    capex_ann_per_kW = capex_jpy_kW * ann_factor
    # 1kW 電解槽の H₂ 生産量 [kg/h] ≈ 1kW / (kwh_per_kg)
    h2_per_kW_h = 1.0 / kwh_per_kg
    capex_per_kg = capex_ann_per_kW / (h2_per_kW_h * full_load_h)

    opex_per_kg = capex_per_kg * 0.03   # OPEX ≈ 3% CAPEX

    lcoh = elec_cost + capex_per_kg + opex_per_kg
    return {
        "elec_jpy_kWh":    elec_price_jpy_kWh,
        "kwh_per_kg_H2":   round(kwh_per_kg, 1),
        "elec_cost_jpy_kg": round(elec_cost, 0),
        "capex_cost_jpy_kg": round(capex_per_kg, 0),
        "LCOH_jpy_kg":     round(lcoh, 0),
        "LCOH_USD_kg":     round(lcoh / 150, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel 1: BOG 発生量 vs LNG 温度
    ax = axes[0, 0]
    T_arr = np.linspace(108, 120, 100)
    bog = [boil_off_rate(T) for T in T_arr]
    ax.plot(T_arr, bog, "b-", lw=2.5)
    ax.axvline(111.15, color="red", ls="--", lw=1.5, label="沸点 111.15K")
    ax.set_xlabel("LNG temperature [K]", fontsize=10)
    ax.set_ylabel("BOG rate [kg/h]", fontsize=10)
    ax.set_title("Boil-Off Gas Rate\n(50,000m³ LNG tank)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    # Panel 2: 成層強度 vs 上下温度差
    ax2 = axes[0, 1]
    dT_arr = np.linspace(0, 2.0, 100)
    Ra_arr = [stratification_index(111.15, 111.15+dT)["Ra"] for dT in dT_arr]
    ax2.semilogy(dT_arr, Ra_arr, "r-", lw=2.5)
    ax2.axhline(1e12, color="red",    ls="--", lw=1.5, label="ロールオーバー臨界 Ra=1e12")
    ax2.axhline(1e10, color="orange", ls=":",  lw=1.5, label="警戒 Ra=1e10")
    ax2.fill_between(dT_arr, 1e12, 1e15,
                     where=[r > 1e12 for r in Ra_arr], alpha=0.2, color="red")
    ax2.set_xlabel("ΔT (top - bottom) [K]", fontsize=10)
    ax2.set_ylabel("Rayleigh number", fontsize=10)
    ax2.set_title("LNG Stratification Index\nRollover Risk Assessment", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2, which="both")

    # Panel 3: ロールオーバー BOG スパイク
    ax3 = axes[0, 2]
    dT_ro = np.linspace(0.1, 3.0, 80)
    spikes = [rollover_bog_spike(TANK_SPEC["V_m3"], dT) for dT in dT_ro]
    normal_bog = boil_off_rate(111.15) / 3600
    ax3.fill_between(dT_ro, spikes, alpha=0.4, color="red")
    ax3.plot(dT_ro, spikes, "r-", lw=2.5, label="Rollover BOG spike")
    ax3.axhline(normal_bog, color="blue", ls="--", lw=2, label=f"Normal BOG {normal_bog:.1f} kg/s")
    ax3.set_xlabel("Layer ΔT [K]", fontsize=10)
    ax3.set_ylabel("BOG spike [kg/s]", fontsize=10)
    ax3.set_title("Rollover BOG Spike\n(50,000m³ tank)", fontsize=10)
    ax3.legend(fontsize=9); ax3.grid(alpha=0.2)

    # Panel 4: PEM 分極曲線
    ax4 = axes[1, 0]
    j_arr = np.linspace(0.01, 4.0, 200)
    V_arr = [pem_cell_voltage(j) for j in j_arr]
    eta_arr = [pem_efficiency(j) for j in j_arr]
    ax4_twin = ax4.twinx()
    ax4.plot(j_arr, V_arr,   "b-", lw=2.5, label="Cell voltage [V]")
    ax4_twin.plot(j_arr, eta_arr, "r--", lw=2.0, label="Efficiency [%]")
    ax4.axvline(2.0, color="gray", ls=":", lw=1.5, label="典型動作点 2A/cm²")
    ax4.set_xlabel("Current density [A/cm²]", fontsize=10)
    ax4.set_ylabel("Cell voltage [V]", color="blue", fontsize=10)
    ax4_twin.set_ylabel("Efficiency [%]", color="red", fontsize=10)
    ax4.set_title("PEM Electrolyzer\nPolarization Curve", fontsize=10)
    lines1, labs1 = ax4.get_legend_handles_labels()
    lines2, labs2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1+lines2, labs1+labs2, fontsize=8)
    ax4.grid(alpha=0.2)

    # Panel 5: Green H₂ LCOH vs 電力価格
    ax5 = axes[1, 1]
    elec_prices = np.linspace(5, 30, 80)
    for cf, col, lbl in [(0.95,"#2166ac","CF=95% (再エネ専用)"),(0.70,"#ff7f0e","CF=70%"),(0.40,"#d62728","CF=40% (余剰電力)")]:
        costs = [green_h2_cost(p, cf)["LCOH_jpy_kg"] for p in elec_prices]
        ax5.plot(elec_prices, costs, lw=2.5, color=col, label=lbl)
    ax5.axhline(300, color="green", ls="--", lw=1.5, label="目標 ¥300/kg (2030)")
    ax5.axhline(100, color="purple", ls=":", lw=1.5, label="天然ガス同等 ¥100/kg")
    ax5.set_xlabel("Electricity price [JPY/kWh]", fontsize=10)
    ax5.set_ylabel("Green H₂ LCOH [JPY/kg]", fontsize=10)
    ax5.set_title("Green H₂ Cost\nvs Electricity Price", fontsize=10)
    ax5.legend(fontsize=7); ax5.grid(alpha=0.2)

    # Panel 6: 東京ガス H₂ 戦略サマリー
    ax6 = axes[1, 2]
    ax6.axis("off")
    lcoh_now  = green_h2_cost(12, 0.70)
    lcoh_2030 = green_h2_cost(8,  0.85)
    rows = [
        ["指標", "現状 (2024)", "目標 (2030)"],
        ["H₂混焼率", "1% (実証)", "20%"],
        ["Green H₂ LCOH", f"¥{lcoh_now['LCOH_jpy_kg']}/kg", "¥300/kg"],
        ["PEM 効率", f"{pem_efficiency(2.0):.0f}%", "80%+"],
        ["電解槽コスト", "¥80,000/kW", "¥40,000/kW"],
        ["再エネ電力", "¥12/kWh", "¥8/kWh (洋上風力)"],
        ["LNG依存度", "~88%", "70% (H₂+再エネ)"],
        ["CO₂削減", "基準", "-20% (H₂混焼)"],
    ]
    tbl = ax6.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.3, 2.1)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_facecolor("#1a5276"); cell.set_text_props(color="w")
        elif r % 2 == 0: cell.set_facecolor("#f0f0f0")
    ax6.set_title("東京ガス 水素戦略 KPI\n(2030 中期経営計画)", fontsize=10)

    fig.suptitle("東京ガス LNG タンク × Green H₂ モデル\n"
                 "熱成層/ロールオーバー / PEM 電解槽 / LCOH", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "tokyogas_lng_hydrogen.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  BOG (通常): {boil_off_rate(111.15):.0f} kg/h")
    print(f"  成層指数 (ΔT=0.5K): {stratification_index(111.15, 111.65)}")
    print(f"  PEM効率 @ 2A/cm²: {pem_efficiency(2.0):.1f}%")
    print(f"  Green H₂ (¥12/kWh): {green_h2_cost(12)}")
    print(f"\n[OK] TokyoGas LNG+H₂ -> {out}")


if __name__ == "__main__":
    plot_all()
