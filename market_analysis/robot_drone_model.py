"""
ヒューマノイドロボット × ドローン × eVTOL パワーエレクトロニクスモデル
=========================================================================
GaN DC/DC + モーター駆動 + 飛行力学 + バッテリー設計

References:
  - Tesla Optimus Gen-2 tech brief (2024)
  - Figure AI 02 specs (2024)
  - Goldman Sachs "Humanoid Robots" (2024): 300万台/年 by 2030
  - Betaflight/KISS ESC GaN application notes
  - DJI Matrice 350 power specs
  - Bell Nexus / Joby eVTOL specs
"""

import math

# ════════════════════════════════════════════════════════════════════════════
# 1. ヒューマノイドロボット
# ════════════════════════════════════════════════════════════════════════════

HUMANOID_SPECS = {
    "Tesla_Optimus_Gen2": {
        "mass_kg":        57,
        "dof":            28,       # 自由度
        "peak_torque_Nm": 85,       # 最大関節トルク
        "max_speed_ms":   1.5,      # 歩行速度
        "battery_Wh":     2300,
        "bus_voltage_V":  48,
        "compute_W":      200,      # オンボード AI チップ
        "peak_power_kW":  2.0,
        "price_usd":      20_000,   # 目標価格
        "color":          "#e45756",
    },
    "Figure_02": {
        "mass_kg":        60,
        "dof":            43,
        "peak_torque_Nm": 100,
        "max_speed_ms":   1.2,
        "battery_Wh":     2100,
        "bus_voltage_V":  48,
        "compute_W":      300,
        "peak_power_kW":  2.5,
        "price_usd":      30_000,
        "color":          "#4c78a8",
    },
    "Boston_Dynamics_Atlas": {
        "mass_kg":        89,
        "dof":            28,
        "peak_torque_Nm": 200,
        "max_speed_ms":   2.5,
        "battery_Wh":     3000,
        "bus_voltage_V":  72,
        "compute_W":      400,
        "peak_power_kW":  5.0,
        "price_usd":      75_000,
        "color":          "#54A24B",
    },
}

# GaN ESC (Electronic Speed Controller) パラメータ — 48V, 100kHz
GAN_JOINT_DRIVER = {
    "bus_V":         48,
    "fsw_kHz":       100,       # GaN なら 100kHz 可能 (Si: 20kHz)
    "Eon_uJ":        1.5,       # ターンオン損失 @ 10A
    "Eoff_uJ":       0.8,
    "Rd_mohm":       3.0,       # 導通抵抗
    "efficiency_pct": 97.5,
    "die_area_mm2":  4.0,       # 小型 GaN チップ
}


def joint_power_model(robot: str = "Tesla_Optimus_Gen2",
                      activity: str = "walking") -> dict:
    """
    関節アクチュエータ電力モデル。
    activity: "walking" | "running" | "manipulation" | "standby"
    """
    spec = HUMANOID_SPECS[robot]

    # 活動別 平均トルク・速度係数
    activity_factor = {
        "walking":      0.25,   # 定格の25%
        "running":      0.65,
        "manipulation": 0.40,
        "standby":      0.05,
    }
    af = activity_factor.get(activity, 0.25)

    torque_avg = spec["peak_torque_Nm"] * af
    omega_avg  = spec["max_speed_ms"] / 0.3 * af   # rad/s (脚リンク0.3m)

    P_mechanical = spec["dof"] * torque_avg * omega_avg
    P_elec_joint = P_mechanical / (GAN_JOINT_DRIVER["efficiency_pct"] / 100)
    P_total      = P_elec_joint + spec["compute_W"]

    # バッテリー持続時間
    runtime_h = spec["battery_Wh"] / max(P_total, 1)

    # GaN スイッチング損失 (全関節合計)
    fsw = GAN_JOINT_DRIVER["fsw_kHz"] * 1e3
    P_sw_per_joint = (GAN_JOINT_DRIVER["Eon_uJ"] + GAN_JOINT_DRIVER["Eoff_uJ"]) * 1e-6 * fsw * 6
    P_sw_total = P_sw_per_joint * spec["dof"]

    return {
        "robot":          robot,
        "activity":       activity,
        "P_mechanical_W": round(P_mechanical, 1),
        "P_electrical_W": round(P_elec_joint, 1),
        "P_total_W":      round(P_total, 1),
        "P_sw_loss_W":    round(P_sw_total, 1),
        "runtime_h":      round(runtime_h, 2),
        "gan_chips":      spec["dof"] * 2,   # 1関節あたり2チップ (ハーフブリッジ)
    }


def humanoid_market_forecast() -> list[dict]:
    """Goldman Sachs 2024 ベース ヒューマノイド市場予測。"""
    # 普及曲線: S字型 (スマホを参考に10年で1Bに達した)
    roadmap = [
        {"year": 2024, "units_k": 1,      "price_k$": 75},
        {"year": 2025, "units_k": 10,     "price_k$": 50},
        {"year": 2026, "units_k": 50,     "price_k$": 35},
        {"year": 2027, "units_k": 200,    "price_k$": 25},
        {"year": 2028, "units_k": 600,    "price_k$": 20},
        {"year": 2029, "units_k": 1500,   "price_k$": 16},
        {"year": 2030, "units_k": 3000,   "price_k$": 13},
    ]
    for r in roadmap:
        r["market_B$"] = round(r["units_k"] * r["price_k$"] / 1e3, 2)
        r["gan_chips_M"] = round(r["units_k"] * 50 / 1e3, 2)  # ~50 GaN chips/robot
    return roadmap


# ════════════════════════════════════════════════════════════════════════════
# 2. ドローン
# ════════════════════════════════════════════════════════════════════════════

DRONE_SPECS = {
    "FPV_military": {
        "label":       "軍用 FPV (ウクライナ型)",
        "mass_kg":     0.5,
        "payload_kg":  0.3,
        "n_motors":    4,
        "motor_kV":    2300,    # KV (rpm/V)
        "voltage_V":   22.2,    # 6S LiPo
        "peak_power_W": 800,
        "hover_pct":   0.30,    # ホバリング = 定格の30%
        "battery_Wh":  25,
        "cost_usd":    400,
        "color":       "#FF4136",
    },
    "Commercial_delivery": {
        "label":       "配送ドローン (DJI Matrice 型)",
        "mass_kg":     6.0,
        "payload_kg":  2.5,
        "n_motors":    6,
        "motor_kV":    400,
        "voltage_V":   48,
        "peak_power_W": 3000,
        "hover_pct":   0.45,
        "battery_Wh":  300,
        "cost_usd":    8_000,
        "color":       "#4c78a8",
    },
    "MALE_surveillance": {
        "label":       "MALE 偵察 (Bayraktar 型)",
        "mass_kg":     650,
        "payload_kg":  150,
        "n_motors":    1,       # ターボプロップ
        "motor_kV":    None,
        "voltage_V":   270,
        "peak_power_W": 100_000,
        "hover_pct":   0.60,
        "battery_Wh":  0,       # 燃料
        "cost_usd":    5_000_000,
        "color":       "#9C27B0",
    },
}


def drone_power_model(drone: str = "FPV_military") -> dict:
    """
    ドローン電力モデル (アクチュエーターディスク理論)。
    ホバリング推力 T = mg → 電力 P = T^(3/2) / sqrt(2ρA)
    """
    d = DRONE_SPECS[drone]
    total_mass = d["mass_kg"] + d["payload_kg"] * 0.5   # 半積載
    g   = 9.81
    rho = 1.225   # 空気密度 kg/m³

    # プロペラ面積 (経験則: D_prop ≈ 0.5m for 6kg drone)
    D_prop  = 0.05 + 0.08 * total_mass ** 0.5   # m
    A_total = d["n_motors"] * math.pi * (D_prop / 2) ** 2

    T_total = total_mass * g   # 必要推力 [N]
    P_ideal = T_total ** 1.5 / math.sqrt(2 * rho * A_total)
    FM      = 0.75   # Figure of Merit (実機効率)
    P_hover = P_ideal / FM

    # GaN ESC 効率
    P_input = P_hover / 0.965    # ESC 効率 96.5% (GaN)

    # 飛行時間
    if d["battery_Wh"] > 0:
        flight_min = d["battery_Wh"] / (P_input * d["hover_pct"]) * 60
    else:
        flight_min = 1440   # 燃料機: 24h

    return {
        "drone":         drone,
        "label":         d["label"],
        "total_mass_kg": round(total_mass, 1),
        "T_total_N":     round(T_total, 1),
        "P_hover_W":     round(P_hover, 0),
        "P_input_W":     round(P_input, 0),
        "flight_min":    round(flight_min, 1),
        "D_prop_m":      round(D_prop, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. eVTOL (空飛ぶクルマ)
# ════════════════════════════════════════════════════════════════════════════

EVTOL_SPECS = {
    "Joby_S4": {
        "label":         "Joby S4 (Uber Elevate)",
        "mass_kg":       2177,
        "passengers":    4,
        "n_rotors":      6,
        "cruise_kW":     150,
        "hover_kW":      700,
        "cruise_speed_kmh": 321,
        "range_km":      241,
        "battery_kWh":   250,
        "bus_V":         800,
        "sic_inverters": 6,
        "color":         "#FFDC00",
    },
    "Archer_Midnight": {
        "label":         "Archer Midnight",
        "mass_kg":       1814,
        "passengers":    4,
        "n_rotors":      12,
        "cruise_kW":     120,
        "hover_kW":      550,
        "cruise_speed_kmh": 240,
        "range_km":      96,
        "battery_kWh":   150,
        "bus_V":         800,
        "sic_inverters": 12,
        "color":         "#f58518",
    },
    "Wisk_Cora": {
        "label":         "Wisk Cora (自動操縦)",
        "mass_kg":       1134,
        "passengers":    1,
        "n_rotors":      12,
        "cruise_kW":     60,
        "hover_kW":      300,
        "cruise_speed_kmh": 150,
        "range_km":      40,
        "battery_kWh":   80,
        "bus_V":         400,
        "sic_inverters": 12,
        "color":         "#72B7B2",
    },
}


def evtol_sic_demand(model: str = "Joby_S4",
                     annual_units: int = 1000) -> dict:
    """eVTOL 1機あたりの SiC 需要 + 年間市場。"""
    spec = EVTOL_SPECS[model]

    # SiC インバータ: 各ローター×1 + バッテリー BMS×1
    sic_per_vehicle   = spec["sic_inverters"] + 2   # ローター + BMS2台
    sic_die_area_mm2  = 150   # 高信頼性 1200V SiC (大型)
    wafer_300mm_area  = math.pi * 150 ** 2
    dies_per_wafer    = int(wafer_300mm_area / sic_die_area_mm2 * 0.75)
    good_per_wafer    = int(dies_per_wafer * 0.90)

    total_sic_chips   = annual_units * sic_per_vehicle
    wafers_per_year   = math.ceil(total_sic_chips / good_per_wafer)

    # TCO: バッテリー劣化 + 電力コスト
    energy_per_flight = spec["battery_kWh"] * 0.80   # 80% 使用
    flights_per_day   = 20
    elec_cost_day     = energy_per_flight * flights_per_day * 0.10   # $0.10/kWh
    battery_life_cy   = 1000
    battery_cost      = 200 * spec["battery_kWh"]    # $200/kWh
    battery_amort_day = battery_cost / battery_life_cy * flights_per_day

    return {
        "model":              model,
        "label":              spec["label"],
        "sic_per_vehicle":    sic_per_vehicle,
        "sic_die_area_mm2":   sic_die_area_mm2,
        "total_sic_chips":    total_sic_chips,
        "wafers_per_year":    wafers_per_year,
        "elec_cost_day_usd":  round(elec_cost_day, 1),
        "battery_amort_day":  round(battery_amort_day, 1),
        "cost_per_flight_usd": round((elec_cost_day + battery_amort_day) / flights_per_day, 2),
        "hover_kW":           spec["hover_kW"],
        "cruise_kW":          spec["cruise_kW"],
        "range_km":           spec["range_km"],
    }


# ════════════════════════════════════════════════════════════════════════════
# Quick demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== ヒューマノイドロボット ===")
    for robot in HUMANOID_SPECS:
        r = joint_power_model(robot, "walking")
        print(f"  {robot}: P={r['P_total_W']:.0f}W  "
              f"GaN chips={r['gan_chips']}  "
              f"運転時間={r['runtime_h']:.1f}h")

    print("\n=== ドローン 電力 ===")
    for drone in ["FPV_military", "Commercial_delivery"]:
        r = drone_power_model(drone)
        print(f"  {r['label']}: ホバー={r['P_hover_W']:.0f}W  "
              f"飛行={r['flight_min']:.0f}min")

    print("\n=== eVTOL SiC 需要 ===")
    for model in EVTOL_SPECS:
        r = evtol_sic_demand(model, annual_units=500)
        print(f"  {r['label']}: SiC/機={r['sic_per_vehicle']}  "
              f"ウェーハ/年={r['wafers_per_year']}  "
              f"フライトコスト=${r['cost_per_flight_usd']}")

    print("\n=== ヒューマノイド市場予測 ===")
    for r in humanoid_market_forecast():
        print(f"  {r['year']}: {r['units_k']:,}k台  "
              f"${r['market_B$']:.1f}B  "
              f"GaN {r['gan_chips_M']:.1f}M chips")
