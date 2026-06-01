"""
核融合 × パワー半導体モデル
============================
トカマク / 慣性閉じ込め / HTS マグネット の電力系に
SiC / Ga₂O₃ / GaN がどう使われるかを定量化。

核融合の電力チェーン:
  電力網 → [SiC 変換器] → 超伝導マグネット電源
  電力網 → [GaN/SiC]   → ジャイロトロン RF 加熱 (140 GHz)
  電力網 → [高耐圧]      → 中性粒子入射 NBI (1 MV)
  プラズマ → [放射線耐性 SiC] → 診断センサー

References:
  - ITER Organization technical specs (2023)
  - Commonwealth Fusion Systems SPARC design (2021)
  - Helion Energy Series E/F (2023)
  - NIF ignition results (2022, Science)
  - Ga2O3 review: Pearton et al. APL 2018
"""

import math

# ════════════════════════════════════════════════════════════════════════════
# 1. 核融合炉スペック
# ════════════════════════════════════════════════════════════════════════════

FUSION_REACTORS = {
    "ITER": {
        "label":          "ITER (国際熱核融合実験炉)",
        "location":       "フランス・カダラッシュ",
        "type":           "tokamak",
        "status":         "建設中 (2035年 first plasma)",
        "fusion_power_MW": 500,
        "Q_factor":        10,        # Q = 出力/入力
        "TF_coil_current_kA": 68,
        "TF_coil_voltage_kV": 0.9,
        "PF_coil_power_MVA": 500,     # ポロイダルコイル電源
        "NBI_power_MW":    33,         # 中性粒子入射 (1MeV)
        "ECRH_power_MW":   20,         # 電子サイクロトロン加熱 (170GHz)
        "pulse_s":         400,        # プラズマ持続時間
        "magnet_type":     "LTS",      # 低温超伝導 (NbTi)
        "cooling_T_K":     4.5,
        "neutron_flux_n_cm2s": 5e13,  # 第一壁での中性子束
        "capex_B$":        22,
        "color":           "#2166ac",
    },
    "SPARC": {
        "label":          "SPARC (CFS × MIT)",
        "location":       "米国・マサチューセッツ",
        "type":           "compact_tokamak",
        "status":         "設計完了 (2027年 first plasma 目標)",
        "fusion_power_MW": 140,
        "Q_factor":        2,
        "TF_coil_current_kA": 40,
        "TF_coil_voltage_kV": 10,     # HTS は高電圧対応
        "PF_coil_power_MVA": 200,
        "NBI_power_MW":    25,
        "ECRH_power_MW":   25,
        "pulse_s":         10,
        "magnet_type":     "HTS",      # 高温超伝導 REBCO (20K)
        "cooling_T_K":     20,
        "neutron_flux_n_cm2s": 1e13,
        "capex_B$":        2,
        "color":           "#d73027",
    },
    "Helion_Polaris": {
        "label":          "Helion Polaris (Microsoft 契約)",
        "location":       "米国・ワシントン州",
        "type":           "FRC",       # Field-Reversed Configuration
        "status":         "2028年 商業電力供給 (Microsoft と契約)",
        "fusion_power_MW": 50,
        "Q_factor":        1,          # まず break-even 目標
        "TF_coil_current_kA": 30,
        "TF_coil_voltage_kV": 5,
        "PF_coil_power_MVA": 100,
        "NBI_power_MW":    20,
        "ECRH_power_MW":   0,
        "pulse_s":         1,          # 高繰り返しパルス
        "magnet_type":     "HTS",
        "cooling_T_K":     20,
        "neutron_flux_n_cm2s": 5e12,
        "capex_B$":        1,
        "color":           "#4CAF50",
    },
    "NIF_IFE": {
        "label":          "NIF 慣性閉じ込め (レーザー核融合)",
        "location":       "米国・ローレンスリバモア",
        "type":           "ICF",
        "status":         "2022年 点火達成 → 商業化研究中",
        "fusion_power_MW": 3,           # 現在の単発出力
        "Q_factor":        1.5,
        "TF_coil_current_kA": 0,
        "TF_coil_voltage_kV": 0,
        "PF_coil_power_MVA": 0,
        "NBI_power_MW":    0,
        "ECRH_power_MW":   500,         # レーザー駆動電源 (実質 ECRH 相当)
        "pulse_s":         1e-9,        # ナノ秒パルス
        "magnet_type":     "None",
        "cooling_T_K":     300,
        "neutron_flux_n_cm2s": 1e15,   # ペレット近傍
        "capex_B$":        3.5,
        "color":           "#f58518",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. 各サブシステムの半導体要件
# ════════════════════════════════════════════════════════════════════════════

def magnet_power_supply(reactor: str = "ITER") -> dict:
    """
    超伝導マグネット電源モデル。
    TF コイル: 定常 DC 大電流 → SiC チョッパー
    PF コイル: パルス変動電流 → 高速応答 SiC / Ga₂O₃
    """
    r = FUSION_REACTORS[reactor]

    # TF コイル: P = V × I
    P_TF_MW = r["TF_coil_voltage_kV"] * r["TF_coil_current_kA"] / 1000

    # PF コイル電源規模
    P_PF_MW = r["PF_coil_power_MVA"] * 0.9   # 力率 0.9

    # 必要な SiC / Ga₂O₃ デバイス電圧
    # HTS マグネット: 高電圧 (10kV) → Ga₂O₃ が有利
    # LTS マグネット: 低電圧 (< 1kV) → SiC が主流
    if r["magnet_type"] == "HTS":
        primary_device = "Ga2O3"         # 10kV 対応
        device_voltage_kV = r["TF_coil_voltage_kV"]
    else:
        primary_device = "SiC"
        device_voltage_kV = max(r["TF_coil_voltage_kV"], 1.7)

    # 変換器あたり SiC/Ga₂O₃ チップ数
    chips_per_converter = math.ceil(device_voltage_kV / 1.7 * 6)  # 直列 × 3相

    # 必要変換器数 (10MW/ユニット 仮定)
    n_converters = math.ceil((P_TF_MW + P_PF_MW) / 10)
    total_chips  = n_converters * chips_per_converter

    return {
        "reactor":           reactor,
        "P_TF_MW":           round(P_TF_MW, 1),
        "P_PF_MW":           round(P_PF_MW, 1),
        "primary_device":    primary_device,
        "device_voltage_kV": device_voltage_kV,
        "n_converters":      n_converters,
        "chips_per_conv":    chips_per_converter,
        "total_chips":       total_chips,
        "magnet_type":       r["magnet_type"],
    }


def gyrotron_power_supply(reactor: str = "ITER") -> dict:
    """
    ジャイロトロン (高周波加熱 ECRH) 電源モデル。
    140-170 GHz のミリ波を発生 → プラズマを 1億°C に加熱。
    電源: 直流 80-100 kV → 高耐圧 SiC / Ga₂O₃ が必須。
    """
    r = FUSION_REACTORS[reactor]
    P_ECRH = r["ECRH_power_MW"]
    if P_ECRH == 0:
        return {"reactor": reactor, "P_ECRH_MW": 0, "note": "ECRH なし"}

    # ジャイロトロン 1台: 1-2 MW @ 80kV
    power_per_gyro_MW = 1.0
    n_gyros = math.ceil(P_ECRH / power_per_gyro_MW)

    # 電源: 80kV DC → 必要デバイス
    # Ga₂O₃: 最大 100kV 理論 → ここで最適
    # 現状: Si thyratron → SiC / Ga₂O₃ に置き換え中
    device_voltage_kV = 80.0
    chips_per_supply = math.ceil(device_voltage_kV / 1.7 * 2)  # 2倍安全係数
    total_chips = n_gyros * chips_per_supply

    return {
        "reactor":        reactor,
        "P_ECRH_MW":      P_ECRH,
        "n_gyros":        n_gyros,
        "voltage_kV":     device_voltage_kV,
        "primary_device": "Ga2O3",   # 80kV → Ga₂O₃ の独壇場
        "chips_per_supply": chips_per_supply,
        "total_chips":    total_chips,
    }


def radiation_hard_electronics(reactor: str = "ITER") -> dict:
    """
    放射線耐性エレクトロニクス。電力系機器は遮蔽外 (設備室) に設置。

    場所別中性子束:
      第一壁 (プラズマ直近):  ~1e13-1e15 n/cm²/s  → 診断センサーのみ
      設備室 (遮蔽外 2m):    ~1e6 n/cm²/s         → 電力変換器
      建屋外周:               ~1e3 n/cm²/s         → 制御システム

    Si の変位損傷限界: ~1e12 n/cm²  (フルエンス)
    SiC:               ~1e15 n/cm²  (Si の 1000倍)
    """
    r = FUSION_REACTORS[reactor]
    # 電力系機器は遮蔽された設備室に設置 → 実効中性子束は第一壁の 1/10^7
    flux_first_wall  = r["neutron_flux_n_cm2s"]
    flux_equipment   = flux_first_wall * 1e-7   # 生体遮蔽後 (ITER 設計値)

    # 10年運転での累積フルエンス (設備室位置)
    years = 10
    fluence_10yr = flux_equipment * 3.15e7 * years   # n/cm²
    flux = flux_equipment

    # 各材料の寿命推定 (fluence 限界 / 実際のフルエンス)
    limits = {
        "Si":     1e12,
        "SiC":    1e15,
        "GaN":    5e14,
        "Ga2O3":  1e14,    # まだ研究中
        "Diamond": 1e16,   # 理論最高
    }

    survivability = {}
    for mat, lim in limits.items():
        years_to_fail = lim / flux / 3.15e7
        survivability[mat] = {
            "limit_n_cm2":    lim,
            "years_to_fail":  round(min(years_to_fail, 100), 1),
            "survives_10yr":  lim > fluence_10yr,
        }

    return {
        "reactor":          reactor,
        "flux_n_cm2s":      flux,
        "fluence_10yr":     fluence_10yr,
        "survivability":    survivability,
        "winner":           max(limits, key=limits.get),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. 材料別 核融合適合度
# ════════════════════════════════════════════════════════════════════════════

FUSION_MATERIAL_SUITABILITY = {
    "SiC": {
        "voltage_range":  "1-3.3 kV",
        "fusion_apps":    ["LTS マグネット電源", "PF コイル変換器", "炉内診断センサー"],
        "advantage":      "高放射線耐性 (Si の 1000倍)、高温動作",
        "limitation":     "3.3kV 超は困難",
        "readiness":      "商用 (ITER に採用予定)",
        "score":          8,
        "color":          "#2166ac",
    },
    "Ga2O3": {
        "voltage_range":  "3.3-100 kV",
        "fusion_apps":    ["HTS マグネット電源 (10kV)", "ジャイロトロン電源 (80kV)",
                           "NBI 高電圧電源 (1MV → 将来)"],
        "advantage":      "圧倒的な絶縁破壊電界 (8 MV/cm)、超高電圧対応",
        "limitation":     "熱伝導率 低 (11 W/mK)、p 型ドーピング困難",
        "readiness":      "研究段階 → 2030年代実用化",
        "score":          9,    # 核融合での将来ポテンシャル最高
        "color":          "#7b2d8b",
    },
    "GaN_on_SiC": {
        "voltage_range":  "0.2-1 kV",
        "fusion_apps":    ["RF ドライバー", "制御システム電源", "ジャイロトロン前段 RF"],
        "advantage":      "高周波・高電力密度",
        "limitation":     "1kV 超は困難",
        "readiness":      "商用",
        "score":          6,
        "color":          "#f4a582",
    },
    "Diamond": {
        "voltage_range":  "10 kV-∞",
        "fusion_apps":    ["究極の炉内センサー", "超高電圧スイッチ"],
        "advantage":      "最高の放射線耐性 + 熱伝導率",
        "limitation":     "p 型のみ、n 型困難、量産不可",
        "readiness":      "研究 (2040年代以降)",
        "score":          7,   # 技術的には最高だが実現性低い
        "color":          "#aec7e8",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 4. 核融合 SiC/Ga₂O₃ 市場予測
# ════════════════════════════════════════════════════════════════════════════

FUSION_MARKET_FORECAST = [
    {"year": 2025, "reactors": 2,   "stage": "実験炉",   "sic_M$": 50,    "ga2o3_M$": 5},
    {"year": 2027, "reactors": 5,   "stage": "実験炉",   "sic_M$": 200,   "ga2o3_M$": 30},
    {"year": 2030, "reactors": 10,  "stage": "原型炉",   "sic_M$": 800,   "ga2o3_M$": 150},
    {"year": 2033, "reactors": 20,  "stage": "原型炉",   "sic_M$": 2000,  "ga2o3_M$": 500},
    {"year": 2035, "reactors": 5,   "stage": "商業炉",   "sic_M$": 5000,  "ga2o3_M$": 2000},
    {"year": 2040, "reactors": 50,  "stage": "商業炉",   "sic_M$": 30000, "ga2o3_M$": 15000},
]


def fusion_semiconductor_summary() -> dict:
    """全炉 × 全サブシステムの半導体需要サマリー。"""
    total = {"SiC_chips": 0, "Ga2O3_chips": 0, "GaN_chips": 0}
    breakdown = {}

    for reactor in FUSION_REACTORS:
        mag  = magnet_power_supply(reactor)
        gyro = gyrotron_power_supply(reactor)

        sic_chips  = mag["total_chips"] if mag["primary_device"] == "SiC" else 0
        ga2o3_chips = (mag["total_chips"] if mag["primary_device"] == "Ga2O3" else 0) + \
                      gyro.get("total_chips", 0)

        total["SiC_chips"]   += sic_chips
        total["Ga2O3_chips"] += ga2o3_chips
        breakdown[reactor] = {
            "label":       FUSION_REACTORS[reactor]["label"],
            "SiC_chips":   sic_chips,
            "Ga2O3_chips": ga2o3_chips,
            "magnet":      mag["magnet_type"],
        }

    return {"total": total, "breakdown": breakdown,
            "forecast": FUSION_MARKET_FORECAST,
            "material_guide": FUSION_MATERIAL_SUITABILITY}


# ════════════════════════════════════════════════════════════════════════════
# Quick demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  核融合 × パワー半導体モデル")
    print("=" * 60)

    for reactor in FUSION_REACTORS:
        r = FUSION_REACTORS[reactor]
        mag  = magnet_power_supply(reactor)
        gyro = gyrotron_power_supply(reactor)
        rad  = radiation_hard_electronics(reactor)

        print(f"\n【{r['label']}】 ({r['status']})")
        print(f"  マグネット電源: {mag['P_TF_MW']+mag['P_PF_MW']:.0f} MW  "
              f"デバイス: {mag['primary_device']} {mag['device_voltage_kV']:.1f}kV  "
              f"チップ: {mag['total_chips']}")
        if gyro.get("n_gyros"):
            print(f"  ジャイロトロン: {gyro['P_ECRH_MW']}MW / {gyro['n_gyros']}台  "
                  f"→ {gyro['primary_device']} {gyro['voltage_kV']:.0f}kV  "
                  f"チップ: {gyro['total_chips']}")
        print(f"  放射線耐性 (中性子束 {r['neutron_flux_n_cm2s']:.0e} n/cm²/s):")
        for mat, s in rad["survivability"].items():
            ok = "✓" if s["survives_10yr"] else "✗"
            print(f"    {mat:<8} {ok} {s['years_to_fail']:.1f}年")

    print("\n  材料適合度まとめ:")
    for mat, m in sorted(FUSION_MATERIAL_SUITABILITY.items(),
                          key=lambda x: x[1]["score"], reverse=True):
        print(f"  {mat:<12} スコア={m['score']}  {m['voltage_range']:<12}  {m['readiness']}")

    print("\n  核融合 SiC/Ga₂O₃ 市場予測:")
    for r in FUSION_MARKET_FORECAST:
        print(f"  {r['year']}  {r['stage']:<6}  SiC=${r['sic_M$']/1000:.1f}B  "
              f"Ga₂O₃=${r['ga2o3_M$']/1000:.2f}B")
