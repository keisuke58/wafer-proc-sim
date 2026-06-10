"""
LEO衛星 × 系統用BESS × 太陽光 パワーエレクトロニクスモデル
=============================================================
SiC が支配する高信頼性・高電圧領域の物理 + 市場モデル

References:
  - SpaceX Starlink v2 Mini power specs (2023)
  - IRENA "Utility-Scale Batteries" 2024
  - Wolfspeed "SiC for Utility Grid" AN (2023)
  - Fraunhofer ISE PV Report 2024
"""

import math

# ════════════════════════════════════════════════════════════════════════════
# 1. LEO 衛星 電力系モデル
# ════════════════════════════════════════════════════════════════════════════

SATELLITE_SPECS = {
    "Starlink_v2_Mini": {
        "label":             "Starlink v2 Mini",
        "orbit_km":          550,
        "mass_kg":           800,
        "solar_area_m2":     22,
        "solar_eff":         0.30,   # 30% 三接合 GaAs セル
        "power_bol_W":       1900,   # 寿命初期
        "power_eol_W":       1500,   # 5年後 (放射線劣化)
        "battery_kWh":       1.5,    # 日陰 (36分/周回)
        "bus_V":             100,
        "sic_pcu_modules":   4,      # PCU (電力調整ユニット)
        "tid_krad":          30,     # 5年 LEO TID
        "annual_launches":   500,
        "color":             "#001f3f",
    },
    "OneWeb_Gen2": {
        "label":             "OneWeb Gen2",
        "orbit_km":          1200,
        "mass_kg":           147,
        "solar_area_m2":     2.8,
        "solar_eff":         0.28,
        "power_bol_W":       320,
        "power_eol_W":       240,
        "battery_kWh":       0.15,
        "bus_V":             42,
        "sic_pcu_modules":   2,
        "tid_krad":          50,     # 1200km は TID 高い
        "annual_launches":   200,
        "color":             "#2166ac",
    },
    "GEO_telecom": {
        "label":             "GEO 通信衛星",
        "orbit_km":          35_786,
        "mass_kg":           6000,
        "solar_area_m2":     80,
        "solar_eff":         0.29,
        "power_bol_W":       20_000,
        "power_eol_W":       12_000,  # 15年後
        "battery_kWh":       25,
        "bus_V":             100,
        "sic_pcu_modules":   20,
        "tid_krad":          300,    # GEO 15年 = 高線量
        "annual_launches":   15,
        "color":             "#d73027",
    },
}

# SiC 放射線耐性 (TID効果モデル)
SIC_TID_MODEL = {
    "Vth_shift_mV_per_krad": 2.0,    # SiC: 小さい (Si: 10-30mV/krad)
    "leakage_increase_pct_per_krad": 0.05,
    "mobility_degradation_pct_per_krad": 0.02,
    "critical_tid_krad": 5000,       # SiC はほぼ無制限 (実用上)
}


def satellite_power_budget(sat: str = "Starlink_v2_Mini",
                            eclipse_frac: float = 0.35) -> dict:
    """
    衛星電力バジェット:
    - 日照時: 太陽電池 → 負荷 + バッテリー充電
    - 日陰時: バッテリー → 負荷
    """
    spec = SATELLITE_SPECS[sat]
    solar_irr = 1361   # W/m² (太陽定数)

    P_solar = spec["solar_area_m2"] * spec["solar_eff"] * solar_irr * (1 - eclipse_frac)
    P_load  = spec["power_bol_W"]

    P_sic_conversion_loss = P_load * 0.03   # SiC PCU: 97% 効率
    P_net   = P_solar - P_load - P_sic_conversion_loss

    # バッテリー充電 (日照時の余剰電力)
    bat_charge_W = max(0, P_net)
    # 日陰での放電レート
    bat_discharge_W = P_load / 0.97   # 日陰時に SiC DC/DC で供給

    # 放射線劣化 (5年後)
    tid5yr = spec["tid_krad"]
    dVth   = tid5yr * SIC_TID_MODEL["Vth_shift_mV_per_krad"]
    dLeak  = tid5yr * SIC_TID_MODEL["leakage_increase_pct_per_krad"]

    return {
        "sat":               sat,
        "label":             spec["label"],
        "P_solar_W":         round(P_solar, 0),
        "P_load_W":          P_load,
        "P_sic_loss_W":      round(P_sic_conversion_loss, 1),
        "P_net_W":           round(P_net, 1),
        "battery_eclipse_h": round(spec["battery_kWh"] * 1000 / bat_discharge_W / 60, 1),
        "tid_krad":          tid5yr,
        "dVth_mV":           round(dVth, 1),
        "dLeakage_pct":      round(dLeak, 2),
        "sic_modules":       spec["sic_pcu_modules"],
        "bus_V":             spec["bus_V"],
    }


def leo_constellation_sic_demand() -> dict:
    """全 LEO コンステレーション SiC ウェーハ需要。"""
    total_modules = 0
    breakdown = {}
    for key, spec in SATELLITE_SPECS.items():
        modules = spec["annual_launches"] * spec["sic_pcu_modules"]
        total_modules += modules
        breakdown[key] = {"modules": modules, "label": spec["label"]}

    # RAD-hard SiC: 特殊プロセス → 小ロット高単価
    die_area_mm2 = 50    # RAD-hard 小型 SiC ダイ
    dies_per_wafer = int(math.pi * 150 ** 2 / die_area_mm2 * 0.70)
    good_per_wafer = int(dies_per_wafer * 0.85)

    wafers_per_year = math.ceil(total_modules / good_per_wafer)

    return {
        "total_sic_modules":  total_modules,
        "wafers_per_year":    wafers_per_year,
        "wafers_per_day":     round(wafers_per_year / 365, 2),
        "revenue_M$":         round(total_modules * 2000 / 1e6, 1),  # $2000/module (space grade)
        "breakdown":          breakdown,
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. 系統用 BESS (Battery Energy Storage System)
# ════════════════════════════════════════════════════════════════════════════

BESS_CONFIGS = {
    "Frequency_Regulation": {
        "label":         "周波数調整用 BESS",
        "power_MW":      100,
        "duration_h":    0.5,
        "energy_MWh":    50,
        "cycles_year":   500,
        "chemistry":     "LFP",
        "dc_voltage_V":  1500,
        "ac_voltage_V":  33_000,
        "n_inverters":   10,      # 10MW × 10
        "color":         "#4CAF50",
    },
    "Solar_Firming": {
        "label":         "太陽光平滑化 BESS",
        "power_MW":      50,
        "duration_h":    4,
        "energy_MWh":    200,
        "cycles_year":   365,
        "chemistry":     "LFP",
        "dc_voltage_V":  1500,
        "ac_voltage_V":  11_000,
        "n_inverters":   5,
        "color":         "#FF9800",
    },
    "Grid_Black_Start": {
        "label":         "系統ブラックスタート",
        "power_MW":      20,
        "duration_h":    8,
        "energy_MWh":    160,
        "cycles_year":   10,
        "chemistry":     "NMC",
        "dc_voltage_V":  1000,
        "ac_voltage_V":  6_600,
        "n_inverters":   4,
        "color":         "#9C27B0",
    },
}

# SiC BESS インバータ効率 (IEC 62109 条件)
def bess_inverter_efficiency(P_frac: float = 1.0,
                              dc_V: float = 1500,
                              topology: str = "3L_NPC") -> float:
    """
    SiC 3レベル NPC インバータ効率モデル。
    P_frac: 定格比 (0-1)
    """
    # 部分負荷特性: 軽負荷時は効率低下
    if topology == "3L_NPC":
        eta_peak = 0.990   # SiC 3L NPC: 99.0%
        eta_10   = 0.970   # 10% 負荷時
    else:  # 2L
        eta_peak = 0.984
        eta_10   = 0.960

    # 2次補間
    eta = eta_10 + (eta_peak - eta_10) * min(P_frac / 0.5, 1.0)
    return round(eta, 4)


def bess_roundtrip_efficiency(config: str = "Frequency_Regulation") -> dict:
    """
    BESS 往復効率: 充電 → 蓄積 → 放電。
    SiC インバータ採用による改善を定量化。
    """
    c = BESS_CONFIGS[config]

    eta_inv_sic = bess_inverter_efficiency(0.80, c["dc_voltage_V"])
    eta_inv_si  = eta_inv_sic - 0.015   # Si IGBT は 1.5% 劣る

    eta_battery = 0.96   # LFP ラウンドトリップ (セル)
    eta_aux     = 0.99   # 補機 (冷却等)

    rte_sic = eta_inv_sic ** 2 * eta_battery * eta_aux   # 充放電で2回通る
    rte_si  = eta_inv_si  ** 2 * eta_battery * eta_aux

    # 年間損失差異
    energy_annual_MWh = c["energy_MWh"] * c["cycles_year"]
    loss_diff_MWh     = energy_annual_MWh * (rte_sic - rte_si)
    revenue_gain_musd   = loss_diff_MWh * 0.05 / 1e3  # $50/MWh 収入差

    # SiC モジュール需要
    power_per_inv = c["power_MW"] / c["n_inverters"]  # MW
    sic_per_inv   = math.ceil(power_per_inv * 1000 / 50)  # 50kW/module
    sic_total     = sic_per_inv * c["n_inverters"]

    return {
        "config":            config,
        "label":             c["label"],
        "eta_sic":           round(rte_sic * 100, 2),
        "eta_si":            round(rte_si  * 100, 2),
        "rte_gain_pp":       round((rte_sic - rte_si) * 100, 2),
        "annual_energy_MWh": energy_annual_MWh,
        "extra_revenue_M$":  round(revenue_gain_musd, 3),
        "sic_modules":       sic_total,
        "dc_voltage_V":      c["dc_voltage_V"],
        "power_MW":          c["power_MW"],
    }


def bess_market_forecast() -> list[dict]:
    """IEA / IRENA ベース 系統用 BESS 市場予測。"""
    # GWh 導入量 → SiC インバータ需要
    roadmap = [
        {"year": 2024, "install_GWh": 120,  "sic_pct": 25},
        {"year": 2025, "install_GWh": 200,  "sic_pct": 35},
        {"year": 2026, "install_GWh": 320,  "sic_pct": 50},
        {"year": 2027, "install_GWh": 500,  "sic_pct": 65},
        {"year": 2028, "install_GWh": 750,  "sic_pct": 78},
        {"year": 2030, "install_GWh": 1500, "sic_pct": 90},
    ]
    for r in roadmap:
        # 1GWh BESS ≈ 20MW × 4h → 20 MW SiC インバータ
        power_gw       = r["install_GWh"] / 4
        sic_power_gw   = power_gw * r["sic_pct"] / 100
        r["sic_mkt_B$"] = round(sic_power_gw * 0.02, 2)  # $20M/GW = $20k/MW (Yole 2024)
        r["wafers_day"] = round(sic_power_gw * 1000 / 50 / 400 * 1.2, 0)  # 需要換算
    return roadmap


# ════════════════════════════════════════════════════════════════════════════
# Quick demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== LEO 衛星 電力バジェット ===")
    for sat in SATELLITE_SPECS:
        r = satellite_power_budget(sat)
        print(f"  {r['label']}: 太陽電池={r['P_solar_W']:.0f}W  "
              f"SiC損失={r['P_sic_loss_W']:.1f}W  "
              f"TID 5yr dVth={r['dVth_mV']:.0f}mV")

    leo = leo_constellation_sic_demand()
    print(f"\n LEO 全コンステ SiC: {leo['total_sic_modules']} modules/年  "
          f"→ {leo['wafers_per_day']:.2f} wafers/day  "
          f"(${leo['revenue_M$']:.0f}M 宇宙グレード)")

    print("\n=== BESS 往復効率 ===")
    for config in BESS_CONFIGS:
        r = bess_roundtrip_efficiency(config)
        print(f"  {r['label']}: SiC={r['eta_sic']}% vs Si={r['eta_si']}%  "
              f"+{r['rte_gain_pp']}pp  追加収益=${r['extra_revenue_M$']*1000:.0f}k/年")

    print("\n=== BESS 市場予測 ===")
    for r in bess_market_forecast():
        print(f"  {r['year']}: {r['install_GWh']}GWh  "
              f"SiC={r['sic_pct']}%  市場=${r['sic_mkt_B$']:.1f}B")
