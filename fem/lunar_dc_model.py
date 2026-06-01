"""
月面データセンター × 宇宙冷却モデル
=====================================
月の永久陰 (PSR) / 月夜面 / 深宇宙を冷却媒体として活用。

最大の発見: 量子コンピュータの稀有な活躍場所 — PSR なら希釈冷凍機不要!

References:
  - NASA LCROSS: PSR 温度実測 ~40K (2009)
  - ESA PROSPECT: 月極水氷探査 (2026)
  - IBM/Google 量子コンピュータ: 動作温度 10-15mK
  - NASA Kilopower/KRUSTY: 月面核分裂炉 1-10kW (2018実証)
  - International Lunar Research Station (ILRS) 中露計画 2030
"""

import math

SIGMA = 5.67e-8   # Stefan-Boltzmann 定数 W/m²K⁴
EPS   = 0.90      # 放射率

# ════════════════════════════════════════════════════════════════════════════
# 1. 各環境の背景温度
# ════════════════════════════════════════════════════════════════════════════

ENVIRONMENTS = {
    "Earth_DC":         {"T_bg_K": 300,  "label": "地上 DC (空冷)",       "can_solar": True},
    "LEO_orbit":        {"T_bg_K": 220,  "label": "LEO 軌道 (地球 IR込)", "can_solar": True},
    "Moon_dayside":     {"T_bg_K": 394,  "label": "月 日照面",             "can_solar": True},
    "Moon_night":       {"T_bg_K": 100,  "label": "月 夜面 (14日間)",      "can_solar": False},
    "Moon_PSR":         {"T_bg_K": 40,   "label": "月 PSR 永久陰",         "can_solar": False},
    "Moon_PSR_deep":    {"T_bg_K": 25,   "label": "月 PSR 深部 (水氷層)", "can_solar": False},
    "L2_point":         {"T_bg_K": 4,    "label": "太陽-地球 L2 点",       "can_solar": True},
}

# ════════════════════════════════════════════════════════════════════════════
# 2. 放熱モデル (Stefan-Boltzmann)
# ════════════════════════════════════════════════════════════════════════════

def radiator_area(P_kW: float, T_hot_K: float, T_bg_K: float) -> float:
    """必要放熱板面積 [m²]。T_hot > T_bg が必要。"""
    if T_bg_K >= T_hot_K:
        return float("inf")
    return P_kW * 1000 / (SIGMA * EPS * (T_hot_K**4 - T_bg_K**4))


def compare_environments(P_kW: float = 10.0, T_hot_K: float = 320.0) -> list[dict]:
    """全環境の放熱性能比較。"""
    leo_area = radiator_area(P_kW, T_hot_K, ENVIRONMENTS["LEO_orbit"]["T_bg_K"])
    results = []
    for key, env in ENVIRONMENTS.items():
        A = radiator_area(P_kW, T_hot_K, env["T_bg_K"])
        results.append({
            "env":         key,
            "label":       env["label"],
            "T_bg_K":      env["T_bg_K"],
            "area_m2":     round(A, 1) if A < 1e6 else None,
            "vs_leo":      round(A / leo_area, 3) if A < 1e6 else None,
            "can_solar":   env["can_solar"],
        })
    return results


# ════════════════════════════════════════════════════════════════════════════
# 3. ★ 量子コンピュータ × 月 PSR — 最重要発見
# ════════════════════════════════════════════════════════════════════════════

QUANTUM_COOLING = {
    "Earth_room":    {"T_base_K": 300,  "label": "地上 (室温)",              "cost_M$": 0},
    "Earth_cryo":    {"T_base_K": 4,    "label": "地上 希釈冷凍機",           "cost_M$": 3},
    "LEO_passive":   {"T_base_K": 4,    "label": "LEO 受動冷却",              "cost_M$": 0.5},
    "Moon_PSR":      {"T_base_K": 40,   "label": "月 PSR (受動)",             "cost_M$": 0},
    "Moon_PSR_cryo": {"T_base_K": 0.015,"label": "月 PSR + 小型冷凍機",       "cost_M$": 0.3},
}

# IBM Eagle / Google Sycamore: 動作温度 10-15mK
QUBIT_TARGET_MK = 15   # mK


def quantum_cooling_advantage(location: str = "Moon_PSR_cryo") -> dict:
    """
    量子コンピュータ冷却の各環境比較。
    地上では T = 300K → 0.015K まで 希釈冷凍機 (~$3M/台) が必要。
    月 PSR では T = 40K → 0.015K まで 小型冷凍機 (~$0.3M) で済む。

    冷却仕事量: W ∝ T_cold × ln(T_hot/T_cold) (Carnot)
    """
    env = QUANTUM_COOLING[location]
    T_base = env["T_base_K"]
    T_target = QUBIT_TARGET_MK / 1000   # K

    if T_base <= T_target:
        # すでに動作温度以下
        carnot_ratio = 1.0
        cooling_W_per_mW = 0
    else:
        # Carnot 効率: 最小冷却仕事 = Q × (T_hot/T_cold - 1)
        carnot_ratio = T_base / T_target
        cooling_W_per_mW = carnot_ratio * 0.01   # 1mW 熱負荷を除去するのに必要な電力

    # 地上希釈冷凍機との比較
    earth_ratio = 300 / T_target
    improvement = earth_ratio / max(carnot_ratio, 0.1)

    return {
        "location":         location,
        "label":            env["label"],
        "T_base_K":         T_base,
        "T_target_mK":      QUBIT_TARGET_MK,
        "carnot_ratio":     round(carnot_ratio, 0),
        "earth_ratio":      round(earth_ratio, 0),
        "improvement_x":    round(improvement, 1),
        "cooling_W_per_mW": round(cooling_W_per_mW, 1),
        "fridge_cost_M$":   env["cost_M$"],
        "feasible":         carnot_ratio < 5000,
    }


def lunar_quantum_dc_design(n_qubits: int = 1000,
                             location: str = "Moon_PSR") -> dict:
    """
    月 PSR 量子データセンター設計。
    n_qubits: 論理量子ビット数
    """
    # Surface Code: 1論理 qubit = ~1000 物理 qubit (p=0.1%)
    n_physical = n_qubits * 1000
    T_bg = ENVIRONMENTS[location]["T_bg_K"]

    # 熱負荷: 各物理 qubit ~1µW
    P_qubit_uW   = 1.0
    P_total_mW   = n_physical * P_qubit_uW / 1000

    # 冷凍機電力 (Carnot 限界 × 実効率 10倍)
    # 開始温度 T_bg から T_target=15mK まで冷やす
    T_target = QUBIT_TARGET_MK / 1000   # K
    T_earth  = 300.0                    # 地上は常温から
    carnot_psr   = T_bg   / T_target    # PSR 環境: T_bg/T_target
    carnot_earth = T_earth / T_target   # 地上:    300K/T_target
    P_fridge_W       = P_total_mW / 1000 * carnot_psr   * 10   # 実機効率 10× Carnot
    P_fridge_earth_W = P_total_mW / 1000 * carnot_earth * 10

    # 放熱板 (冷凍機の廃熱)
    A_rad = radiator_area(P_fridge_W / 1000, 200, T_bg)

    saving_pct = (1 - P_fridge_W / P_fridge_earth_W) * 100

    # 核電源 (太陽光なし → NASA Kilopower 想定)
    kilopower_units = math.ceil(P_fridge_W / 10000)   # 1台10kW

    return {
        "location":           location,
        "n_logical_qubits":   n_qubits,
        "n_physical_qubits":  n_physical,
        "T_bg_K":             T_bg,
        "P_qubit_total_mW":   round(P_total_mW, 2),
        "P_fridge_W":         round(P_fridge_W, 0),
        "P_fridge_earth_W":   round(P_fridge_earth_W, 0),
        "saving_pct":         round(saving_pct, 1),
        "radiator_m2":        round(A_rad, 2) if A_rad < 1e6 else 9999,
        "kilopower_units":    kilopower_units,
        "A_footprint_m2":     round(A_rad * 2, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. 月面 DC の現実的課題
# ════════════════════════════════════════════════════════════════════════════

LUNAR_DC_CHALLENGES = {
    "power_source": {
        "challenge":  "電源 — PSR は太陽光なし",
        "solution":   "NASA Kilopower 核分裂炉 (1-10kW, 2018年実証済)",
        "severity":   "medium",
        "timeline":   "2030年代 実用化可能",
    },
    "communication": {
        "challenge":  "通信遅延 — 月 = 2.56秒往復",
        "solution":   "リアルタイム処理には不向き。アーカイブ/量子演算/科学計算に特化",
        "severity":   "high",
        "timeline":   "レーザー通信 (NASA LLCD) で 622Mbps 実証済",
    },
    "construction": {
        "challenge":  "建設 — 月面への輸送コスト",
        "solution":   "Starship: $1000/kg → 1トン DC = $1M 打上コスト",
        "severity":   "medium",
        "timeline":   "2030年代以降 (Starship 量産後)",
    },
    "water_ice": {
        "challenge":  "冷媒 — 追加冷却が必要な場合",
        "solution":   "PSR に水氷あり (LCROSS 実証) → 冷媒・推進剤に利用可能",
        "severity":   "low",
        "timeline":   "NASA Artemis / ILRS で採掘計画あり",
    },
    "radiation": {
        "challenge":  "放射線 — 月には磁気圏なし",
        "solution":   "PSR は地形で遮蔽。SiC / GaN チップは耐放射線性高",
        "severity":   "medium",
        "timeline":   "既存の RAD-hard 技術で対応可能",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 5. SBSP + 月面 DC 統合構想
# ════════════════════════════════════════════════════════════════════════════

def integrated_lunar_system() -> dict:
    """
    月面統合システム: SBSP (発電) + PSR DC (計算) + 水氷 (冷媒/推進剤)
    SpaceX Starship で輸送前提。
    """
    # 月極 南極: 永久光峰 (ピーク・オブ・エタナル・ライト) から数 km の PSR
    # → 太陽光発電できる場所と PSR が隣接

    solar_W_m2      = 1361 * 0.35   # GaAs パネル 35% 効率
    panel_area_m2   = 1000          # 1000m² パネル (20m × 50m)
    P_solar_kW      = solar_W_m2 * panel_area_m2 / 1000

    # 電力配分
    P_compute_kW    = P_solar_kW * 0.50   # 50% 計算
    P_xmit_kW       = P_solar_kW * 0.20   # 20% 通信 (地球へ)
    P_thermal_kW    = P_solar_kW * 0.30   # 30% 熱管理・補機

    # PSR DC 設計 (量子 + 古典)
    classical_tflops = P_compute_kW * 100   # 100 TFLOPS/kW (rad-hard GPU)
    quantum_qubits   = 100                   # 論理 qubit (Surface Code)

    # 放熱 (PSR 40K)
    A_rad = radiator_area(P_compute_kW + P_thermal_kW, 320, 40)

    # 水氷採掘 (冷媒)
    ice_kg_yr = 1000   # 年間 1トン採掘 (Artemis 計画目標)

    # ビームフォーミング (地球への電力送信 or 通信)
    laser_power_W  = P_xmit_kW * 1000 * 0.5   # 50% 変換効率
    data_rate_Gbps = laser_power_W * 0.001     # 概算

    return {
        "solar_panel_m2":     panel_area_m2,
        "P_solar_kW":         round(P_solar_kW, 1),
        "P_compute_kW":       round(P_compute_kW, 1),
        "classical_tflops":   round(classical_tflops, 0),
        "quantum_qubits":     quantum_qubits,
        "radiator_m2":        round(A_rad, 1),
        "laser_power_W":      round(laser_power_W, 0),
        "data_rate_Gbps":     round(data_rate_Gbps, 2),
        "ice_cooling_kgyr":   ice_kg_yr,
        "latency_ms":         1280,    # 月→地球 片道
        "starship_cost_M$":   round(2000 * 1000 / 1e6, 1),  # 2トン @ $1k/kg
    }


# ════════════════════════════════════════════════════════════════════════════
# Quick demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  月面データセンター × 宇宙冷却 — 放熱比較")
    print("=" * 70)

    print("\n10kW を放熱するのに必要な放熱板面積 (T_hot=320K):")
    for r in compare_environments(10.0, 320.0):
        if r["area_m2"] is None:
            print(f"  {r['label']:<25}  ∞ (加熱される)  太陽光: {'○' if r['can_solar'] else '✗'}")
        else:
            print(f"  {r['label']:<25}  {r['area_m2']:>6.1f} m²  "
                  f"LEO比={r['vs_leo']:.2f}×  太陽光: {'○' if r['can_solar'] else '✗'}")

    print("\n" + "=" * 70)
    print("  ★ 量子コンピュータ冷却 — 月 PSR の本当の価値")
    print("=" * 70)
    for loc in QUANTUM_COOLING:
        q = quantum_cooling_advantage(loc)
        status = "✓" if q["feasible"] else "✗"
        print(f"  {q['label']:<28}  "
              f"T={q['T_base_K']:>6.3f}K  "
              f"Carnot比={q['carnot_ratio']:>8,.0f}×  "
              f"冷凍機コスト=${q['fridge_cost_M$']:.1f}M  {status}")

    print("\n月 PSR 量子 DC 設計 (1000論理 qubit):")
    for loc in ["Earth_DC_equiv", "Moon_PSR"]:
        if loc == "Earth_DC_equiv":
            loc = "Moon_PSR"
            r = lunar_quantum_dc_design(1000, "Moon_PSR")
            print(f"  月 PSR: 冷凍機電力={r['P_fridge_W']:.0f}W  "
                  f"地上比={r['P_fridge_earth_W']/r['P_fridge_W']:.0f}× 省電力  "
                  f"放熱板={r['radiator_m2']:.1f}m²")

    print("\n統合月面システム:")
    sys = integrated_lunar_system()
    print(f"  太陽光パネル: {sys['solar_panel_m2']}m²  →  {sys['P_solar_kW']:.0f}kW")
    print(f"  古典計算:   {sys['classical_tflops']:.0f} TFLOPS")
    print(f"  量子計算:   {sys['quantum_qubits']} 論理 qubit (PSR 冷却)")
    print(f"  放熱板:     {sys['radiator_m2']:.1f}m² (PSR 40K 利用)")
    print(f"  通信:       {sys['data_rate_Gbps']:.2f} Gbps レーザー (遅延 1.28秒)")
    print(f"  輸送コスト: ${sys['starship_cost_M$']:.1f}M (Starship $1k/kg 想定)")
