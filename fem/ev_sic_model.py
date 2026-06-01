"""
EV × SiC パワーエレクトロニクスモデル
======================================
SiC MOSFET の EV インバータ向け特性:
  スイッチング損失 / 熱設計 / Si IGBT 比較 / メーカー別需要

References:
  - Kimoto & Cooper (2014) Fundamentals of SiC Technology
  - Infineon "SiC for EV" application note (2023)
  - BloombergNEF EV Outlook 2024
"""

import math

# ════════════════════════════════════════════════════════════════════════════
# EV システムスペック
# ════════════════════════════════════════════════════════════════════════════
EV_SYSTEMS = {
    "BEV_400V":  {"bus_V": 400,  "peak_kW": 150,  "Isw_A": 400,  "fsw_kHz": 10},
    "BEV_800V":  {"bus_V": 800,  "peak_kW": 300,  "Isw_A": 400,  "fsw_kHz": 20},
    "PHEV":      {"bus_V": 400,  "peak_kW":  80,  "Isw_A": 200,  "fsw_kHz": 10},
    "OBC_11kW":  {"bus_V": 800,  "peak_kW":  11,  "Isw_A":  20,  "fsw_kHz": 100},
}

# SiC vs Si IGBT スイッチング特性 (典型値)
SWITCH_PARAMS = {
    "SiC_MOSFET_1200V": {
        "Eon_mJ":   0.20,   # ターンオン損失 @400A
        "Eoff_mJ":  0.15,   # ターンオフ損失 @400A
        "Vce_sat":  None,
        "Rd_mohm":  6.0,    # 導通抵抗 [mΩ]
        "Tj_max_C": 200,
        "cost_usd": 25,     # チップ単価
    },
    "Si_IGBT_1200V": {
        "Eon_mJ":   1.20,
        "Eoff_mJ":  0.80,
        "Vce_sat":  2.2,    # [V]
        "Rd_mohm":  None,
        "Tj_max_C": 150,
        "cost_usd": 8,
    },
}


def switching_loss(device: str = "SiC_MOSFET_1200V",
                   system: str = "BEV_800V") -> dict:
    """スイッチング損失 + 導通損失 → 全体効率。"""
    p  = SWITCH_PARAMS[device]
    s  = EV_SYSTEMS[system]
    fsw = s["fsw_kHz"] * 1e3

    # スイッチング損失 (6スイッチ 3相インバータ)
    P_sw = 6 * (p["Eon_mJ"] + p["Eoff_mJ"]) * 1e-3 * fsw   # [W]

    # 導通損失
    I_rms = s["Isw_A"] / math.sqrt(2)
    if p["Rd_mohm"]:
        P_cond = 6 * (p["Rd_mohm"] * 1e-3) * I_rms ** 2
    else:
        P_cond = 6 * p["Vce_sat"] * I_rms * (1 / math.pi + 0.5 * 0.9)  # 力率0.9

    P_total = P_sw + P_cond
    eff_pct = (1 - P_total / (s["peak_kW"] * 1000)) * 100

    return {
        "device":      device,
        "system":      system,
        "P_sw_W":      round(P_sw, 1),
        "P_cond_W":    round(P_cond, 1),
        "P_total_W":   round(P_total, 1),
        "efficiency_%": round(eff_pct, 2),
        "fsw_kHz":     s["fsw_kHz"],
    }


def thermal_model(P_loss_W: float, Rth_jc: float = 0.08,
                  Rth_cs: float = 0.05, T_coolant_C: float = 65.0,
                  n_switches: int = 6) -> dict:
    """
    熱抵抗モデル: Tj = T_coolant + (P/n_switches) × (Rth_jc + Rth_cs)
    Rth_jc: ジャンクション-ケース間熱抵抗 [K/W] (スイッチ1素子あたり)
    Rth_cs: ケース-冷却剤間熱抵抗 [K/W]
    n_switches: インバータのスイッチ数 (3相ブリッジ=6)
    """
    P_per_switch = P_loss_W / n_switches
    dT = P_per_switch * (Rth_jc + Rth_cs)
    Tj = T_coolant_C + dT
    return {
        "P_loss_W":   P_loss_W,
        "Tj_C":       round(Tj, 1),
        "dT_K":       round(dT, 1),
        "margin_K":   round(200 - Tj, 1),   # SiC Tj_max=200°C
        "safe":       Tj < 180,
    }


EV_MAKER_VOLUME = {
    "Tesla":         2_000_000,   # 台/年 (2026)
    "BYD":           3_500_000,
    "Volkswagen_BEV":  800_000,
    "Toyota_BEV":      300_000,
    "GM_BEV":          400_000,
    "Hyundai_BEV":     500_000,
}

SIC_CHIPS_PER_EV = {
    "BEV":  24,   # メインインバータ(18) + OBC(6)
    "PHEV": 12,
}


def ev_sic_demand() -> dict:
    """EV メーカー別 SiC チップ需要 → ウェーハ需要。"""
    total_chips = 0
    breakdown   = {}
    for maker, vol in EV_MAKER_VOLUME.items():
        chips = vol * SIC_CHIPS_PER_EV["BEV"]
        total_chips += chips
        breakdown[maker] = chips

    die_area_mm2    = 20.0   # EV 向け SiC チップ (650V/1200V, 小型)
    dies_per_wafer  = int(math.pi * 150**2 / die_area_mm2 * 0.75)  # 75% wafer utilization
    wafer_yield     = 0.90
    good_per_wafer  = int(dies_per_wafer * wafer_yield)
    wafers_per_year = math.ceil(total_chips / good_per_wafer)

    return {
        "total_chips_M":     round(total_chips / 1e6, 1),
        "wafers_per_year":   wafers_per_year,
        "wafers_per_day":    round(wafers_per_year / 365, 0),
        "breakdown":         breakdown,
        "dies_per_wafer":    dies_per_wafer,
    }


if __name__ == "__main__":
    print("=== EV × SiC モデル ===\n")
    for dev in ["SiC_MOSFET_1200V", "Si_IGBT_1200V"]:
        r = switching_loss(dev, "BEV_800V")
        print(f"{dev}: 損失={r['P_total_W']:.0f}W  効率={r['efficiency_%']:.2f}%")

    r_sic = switching_loss("SiC_MOSFET_1200V", "BEV_800V")
    th    = thermal_model(r_sic["P_total_W"])
    print(f"\nSiC 熱設計: Tj={th['Tj_C']}°C  余裕={th['margin_K']}K  {'✓ Safe' if th['safe'] else '✗ Over'}")

    dem = ev_sic_demand()
    print(f"\nEV 向け SiC 需要: {dem['total_chips_M']:.0f}M チップ/年")
    print(f"  → {dem['wafers_per_day']:.0f} ウェーハ/日")
