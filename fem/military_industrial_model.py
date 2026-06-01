"""
軍産複合体 × 半導体モデル
===========================
戦争 × SiC/GaN が作る「暴力の技術的連鎖」

SiC → EV → AI → ロボット → 宇宙 → 核融合 → 月 → 長寿命
 ↕     ↕    ↕     ↕        ↕       ↕      ↕     ↕
軍事  軍用EV 戦闘AI 兵器ロボット 宇宙戦  核兵器  月基地  超兵士

歴史的真実:
  GPS → 軍事 → 民間
  インターネット → ARPANET (軍事) → 民間
  GaN 半導体 → 軍事レーダー → 5G/EV
  ナイトビジョン → 軍事 → 民間カメラ
  → 今度は SiC が同じ道を歩む

References:
  - DARPA budget FY2024 ($4.1B)
  - DoD semiconductor strategy 2023
  - F-35 JSF program specs (Lockheed Martin)
  - Raytheon DEW (Directed Energy Weapons) AN/SEQ-3
  - 防衛装備庁 次世代戦闘機 F-X (GaN-on-SiC AESA 搭載)
"""

import math

# ════════════════════════════════════════════════════════════════════════════
# 1. 軍事用途別 半導体需要
# ════════════════════════════════════════════════════════════════════════════

MILITARY_APPLICATIONS = {
    # ── 航空機 ────────────────────────────────────────────────────────────
    "F35_AESA_Radar": {
        "label":         "F-35 AESA レーダー (AN/APG-81)",
        "material":      "GaN_on_SiC",
        "chips_per_unit": 1200,    # T/R モジュール数
        "units_global":  3000,     # F-35 生産機数 (計画)
        "unit_price_k$": 400,      # 機体価格の一部
        "replacement_yr": 10,
        "power_kW":      18,
        "freq_GHz":      8,        # X バンド
        "advantage":     "GaN-on-SiC: Si より 10倍の電力密度、3倍の射程距離",
        "color":         "#001f3f",
    },
    "Hypersonic_Power": {
        "label":         "極超音速ミサイル 電源 (HVPW/LRHW)",
        "material":      "SiC",
        "chips_per_unit": 50,
        "units_global":  10_000,
        "unit_price_k$": 20,
        "replacement_yr": 1,
        "power_kW":      500,
        "freq_GHz":      0,
        "advantage":     "SiC: マッハ5+ の空力加熱 (>300°C) に耐える唯一の半導体",
        "color":         "#e45756",
    },
    "DEW_Laser_Power": {
        "label":         "指向性エネルギー兵器 (高出力レーザー)",
        "material":      "SiC",
        "chips_per_unit": 200,
        "units_global":  500,
        "unit_price_k$": 5000,
        "replacement_yr": 5,
        "power_kW":      300,
        "freq_GHz":      0,
        "advantage":     "SiC: 300kW 級電源。シリコンでは発熱で即破壊",
        "color":         "#f5a623",
    },
    "EW_Jammer": {
        "label":         "電子戦 (ECM/ECCM) — GPS妨害・対妨害",
        "material":      "GaN_on_SiC",
        "chips_per_unit": 300,
        "units_global":  5000,
        "unit_price_k$": 80,
        "replacement_yr": 5,
        "power_kW":      50,
        "freq_GHz":      12,
        "advantage":     "GaN-on-SiC: 広帯域・高効率 → 敵レーダーを圧倒",
        "color":         "#9C27B0",
    },
    # ── 海軍 ──────────────────────────────────────────────────────────────
    "Submarine_AIP": {
        "label":         "潜水艦 AIP (非大気依存推進) 燃料電池",
        "material":      "SiC",
        "chips_per_unit": 100,
        "units_global":  200,
        "unit_price_k$": 500,
        "replacement_yr": 20,
        "power_kW":      2000,
        "freq_GHz":      0,
        "advantage":     "SiC インバータ: 水中で完全サイレント + 高効率",
        "color":         "#2166ac",
    },
    "Aegis_Radar": {
        "label":         "イージス艦 SPY-7 レーダー",
        "material":      "GaN_on_SiC",
        "chips_per_unit": 4000,
        "units_global":  100,
        "unit_price_k$": 1000,
        "replacement_yr": 15,
        "power_kW":      60,
        "freq_GHz":      3,        # S バンド
        "advantage":     "日本の次期イージス艦に搭載。GaN で弾道ミサイル探知距離 2倍",
        "color":         "#4c78a8",
    },
    # ── 陸軍 × ロボット ───────────────────────────────────────────────────
    "Combat_Robot_GaN": {
        "label":         "自律型戦闘ロボット (米陸軍 IVAS)",
        "material":      "GaN",
        "chips_per_unit": 80,
        "units_global":  50_000,   # 米陸軍計画
        "unit_price_k$": 50,
        "replacement_yr": 3,
        "power_kW":      5,
        "freq_GHz":      0,
        "advantage":     "GaN ESC: 48V 高速応答 → 戦場での判断速度",
        "color":         "#54A24B",
    },
    # ── 核 ────────────────────────────────────────────────────────────────
    "Nuclear_EMP_Hard": {
        "label":         "核EMP 耐性 電子機器 (SiC ハードニング)",
        "material":      "SiC",
        "chips_per_unit": 500,
        "units_global":  10_000,
        "unit_price_k$": 200,
        "replacement_yr": 20,
        "power_kW":      10,
        "freq_GHz":      0,
        "advantage":     "SiC: EMP に Si より 1000倍強い。核攻撃後も機能する唯一の選択肢",
        "color":         "#FF4136",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. 軍産複合体プレイヤー
# ════════════════════════════════════════════════════════════════════════════

DEFENSE_CONTRACTORS = {
    "Lockheed_Martin": {
        "revenue_B$":   67,
        "sic_gan_role": "F-35 GaN-on-SiC AESA レーダー、極超音速兵器電源",
        "key_programs": ["F-35 (3000機)", "LRHW 極超音速", "宇宙防衛衛星"],
        "country":      "US",
        "color":        "#001f3f",
    },
    "Raytheon_RTX": {
        "revenue_B$":   68,
        "sic_gan_role": "DEW 高出力レーザー電源 (SiC 300kW)、Patriot レーダー GaN",
        "key_programs": ["AN/SEQ-3 DEW", "SPY-6 レーダー", "AIM-260 ミサイル"],
        "country":      "US",
        "color":        "#e45756",
    },
    "Northrop_Grumman": {
        "revenue_B$":   37,
        "sic_gan_role": "B-21 爆撃機 GaN レーダー、宇宙監視衛星 SiC 電源",
        "key_programs": ["B-21 Raider", "James Webb 後継衛星", "GBSD ICBM"],
        "country":      "US",
        "color":        "#4c78a8",
    },
    "Anduril_Industries": {
        "label":        "Anduril (新興軍産複合体)",
        "revenue_B$":   0.5,       # 非公開、推定
        "sic_gan_role": "自律兵器 AI チップ → GaN ESC + SiC 電源",
        "key_programs": ["Lattice AI C2", "Ghost シリーズドローン", "Fury UCAV"],
        "country":      "US",
        "color":        "#9C27B0",
    },
    "Mitsubishi_Electric_JP": {
        "revenue_B$":   35,
        "sic_gan_role": "次期戦闘機 F-X 用 GaN-on-SiC AESA レーダー (三菱電機製)",
        "key_programs": ["F-X (次期戦闘機)", "イージス艦 SPY-7", "偵察衛星"],
        "country":      "Japan",
        "color":        "#FF9800",
    },
    "Kawasaki_Heavy_JP": {
        "revenue_B$":   15,
        "sic_gan_role": "そうりゅう型潜水艦 AIP 燃料電池インバータ (SiC)",
        "key_programs": ["そうりゅう型 AIP", "P-1 哨戒機", "C-2 輸送機"],
        "country":      "Japan",
        "color":        "#2166ac",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 3. EMP 耐性モデル — SiC が軍事の必須材料である理由
# ════════════════════════════════════════════════════════════════════════════

def emp_hardness_model(material: str = "SiC") -> dict:
    """
    核EMP (電磁パルス) に対する耐性モデル。

    EMP の物理:
      E1 (高高度核爆発): 50kV/m, 数ns → 半導体のゲート酸化膜を破壊
      E3 (低周波):       DC〜1Hz → 電力グリッドを破壊

    材料別 EMP 耐性:
      Si CMOS: ゲート酸化膜が薄い → 10V/µm 以上で破壊
      SiC:     バンドギャップが大きい → 電子なだれが起きにくい
    """
    emp_thresholds = {
        "Si_CMOS":   {"threshold_kV_m": 0.05,  "hardened": False, "breakdown_ns": 10},
        "GaN":       {"threshold_kV_m": 1.0,   "hardened": False, "breakdown_ns": 50},
        "SiC":       {"threshold_kV_m": 5.0,   "hardened": True,  "breakdown_ns": 500},
        "SiC_rad_hard": {"threshold_kV_m": 20.0, "hardened": True, "breakdown_ns": 5000},
        "Diamond":   {"threshold_kV_m": 100.0, "hardened": True,  "breakdown_ns": 99999},
    }

    p = emp_thresholds.get(material, emp_thresholds["Si_CMOS"])
    nuclear_emp_kV_m = 50   # 典型的な E1 EMP

    survive = p["threshold_kV_m"] >= nuclear_emp_kV_m
    margin_dB = 20 * math.log10(p["threshold_kV_m"] / nuclear_emp_kV_m)

    return {
        "material":       material,
        "threshold_kV_m": p["threshold_kV_m"],
        "nuclear_emp_kV_m": nuclear_emp_kV_m,
        "survives_nuclear_emp": survive,
        "margin_dB":      round(margin_dB, 1),
        "breakdown_ns":   p["breakdown_ns"],
        "hardened":       p["hardened"],
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. デュアルユース技術サイクル
# ════════════════════════════════════════════════════════════════════════════

DUAL_USE_HISTORY = [
    {
        "tech":     "インターネット",
        "origin":   "ARPANET (DARPA, 1969)",
        "military": "核攻撃後の耐障害通信",
        "civil":    "WWW → 現代経済の基盤",
        "timeline_yr": 25,
    },
    {
        "tech":     "GPS",
        "origin":   "NAVSTAR (米国防総省, 1973)",
        "military": "ミサイル・爆弾の精密誘導",
        "civil":    "カーナビ → スマホ地図 → 自動運転",
        "timeline_yr": 20,
    },
    {
        "tech":     "GaN 半導体",
        "origin":   "DARPA WBGS-RF (2002)",
        "military": "F-22 AESA レーダー",
        "civil":    "5G 基地局 → EV急速充電 → ドローン",
        "timeline_yr": 15,
    },
    {
        "tech":     "ナイトビジョン",
        "origin":   "米軍 (1940s〜)",
        "military": "夜間戦闘",
        "civil":    "監視カメラ → スマホカメラ → 自動車",
        "timeline_yr": 40,
    },
    {
        "tech":     "ドローン",
        "origin":   "Predator (1990年代, CIA/USAF)",
        "military": "ISR・攻撃",
        "civil":    "農業・配送・撮影 → 空飛ぶクルマへ",
        "timeline_yr": 20,
    },
    {
        "tech":     "SiC パワー半導体",
        "origin":   "DARPA WBG-Power (2014〜)",
        "military": "F-35 電源・DEW・極超音速電源",
        "civil":    "EV → BESS → SBSP (進行中)",
        "timeline_yr": 10,   # 今まさに移行中
    },
    {
        "tech":     "量子暗号通信",
        "origin":   "NSA/DARPA (2000年代〜)",
        "military": "盗聴不可能な軍事通信",
        "civil":    "金融・医療・月面DC (将来)",
        "timeline_yr": 15,
    },
    {
        "tech":     "Neuralink 型 BCI",
        "origin":   "DARPA N3 (Neural Next-Gen, 2019)",
        "military": "超兵士: 思考で兵器制御・疲労感なし",
        "civil":    "ALS治療 → 健康モニタリング → 長寿命",
        "timeline_yr": 20,
    },
]

FUTURE_MILITARY_TECH = [
    {
        "year":     2028,
        "tech":     "自律型無人機群 (スウォーム)",
        "material": "GaN ESC + AI チップ",
        "scale":    "1回の攻撃で1000機",
        "disco_connection": "GaN チップ × 1000機 = 大量ダイシング需要",
    },
    {
        "year":     2032,
        "tech":     "指向性エネルギー兵器 (DEW) 実戦配備",
        "material": "SiC 300kW 電源",
        "scale":    "イージス艦・戦闘機搭載",
        "disco_connection": "SiC ブレード $50/wafer × 軍事プレミアム 10× = $500",
    },
    {
        "year":     2035,
        "tech":     "Neuralink 超兵士 (DARPA N3)",
        "material": "SiC 生体チップ + GaN ESC",
        "scale":    "特殊部隊から展開",
        "disco_connection": "Disco 超精密研削 ±0.3µm が生体チップを実現",
    },
    {
        "year":     2040,
        "tech":     "核融合パルス兵器 (EMP 強化版)",
        "material": "Ga₂O₃ 超高電圧パルス回路",
        "scale":    "戦略的抑止力として",
        "disco_connection": "Ga₂O₃ ダイシング → ステルスダイシングで Disco 独占",
    },
    {
        "year":     2050,
        "tech":     "月面軍事基地 (米中争奪)",
        "material": "RAD-hard SiC 全電気兵器",
        "scale":    "月極の「高地」を制する国が宇宙覇権",
        "disco_connection": "月面 Disco 技術 → 火星まで続く軍需",
    },
]


def military_sic_market() -> dict:
    """軍事 SiC/GaN 市場規模計算。"""
    total_value = 0
    breakdown = {}

    for app, spec in MILITARY_APPLICATIONS.items():
        chips   = spec["chips_per_unit"] * spec["units_global"]
        rev_M   = chips * spec["unit_price_k$"] / 1000 / spec["replacement_yr"]
        total_value += rev_M
        breakdown[app] = round(rev_M, 1)

    return {
        "total_M$":   round(total_value, 0),
        "breakdown":  breakdown,
        "premium_x":  10,
        "cagr_pct":   18,
        "vs_civil":   "軍事 SiC = 民間の 10倍単価 × 非価格競争",
    }


# ════════════════════════════════════════════════════════════════════════════
# Quick demo
# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# 5. KEW (動力学的エネルギー兵器) — 「神の杖」物理モデル
# ════════════════════════════════════════════════════════════════════════════

def kew_impact_model(
    mass_kg:    float = 9000,   # タングステン棒 (6m×30cm)
    velocity_ms: float = 7000,  # 軌道速度
    tnt_per_joule: float = 4.18e9,  # 1 ton TNT = 4.18GJ
) -> dict:
    """
    KEW (Kinetic Energy Weapon) = 軌道上から金属棒を落とす兵器。
    通称「神の杖」(Rods from God)。

    物理: E_k = 0.5 × m × v²
    核兵器との比較: 広島原爆 ≈ 15kton TNT = 6.3×10^13 J

    半導体の役割:
      誘導システム: GPS + IMU + SiC 制御回路
      高精度 (CEP < 10m): GaN-on-SiC 高周波通信
      耐熱コーティング制御: SiC センサー (>1000°C)
    """
    E_joules = 0.5 * mass_kg * velocity_ms ** 2
    E_ton_tnt = E_joules / tnt_per_joule
    E_kt_tnt  = E_ton_tnt / 1000

    # 比較
    hiroshima_kt = 15
    bunker_buster_ton = 0.9   # GBU-57 MOP: 13ton TNT ÷ 14 ≈ 0.9ton/m penetration

    return {
        "mass_kg":        mass_kg,
        "velocity_ms":    velocity_ms,
        "E_joules":       E_joules,
        "E_ton_tnt":      round(E_ton_tnt, 1),
        "E_kt_tnt":       round(E_kt_tnt, 3),
        "vs_hiroshima_pct": round(E_kt_tnt / hiroshima_kt * 100, 2),
        "penetration_m":  round(mass_kg / 1000 * 3.5, 0),  # 地中貫通深さ
        "detectable":     False,  # 核爆発と違い放射線なし → 条約上グレー
        "treaty_status":  "宇宙条約 (1967) は核兵器のみ禁止。KEW は法的グレーゾーン",
        "semiconductor_required": "SiC 制御回路 + GaN-on-SiC 誘導通信",
    }


# ════════════════════════════════════════════════════════════════════════════
# 6. 医療産業複合体 × 半導体
# ════════════════════════════════════════════════════════════════════════════

MEDICAL_INDUSTRIAL = {
    "Medtronic": {
        "label":    "Medtronic (ペースメーカー・神経刺激)",
        "revenue_B$": 32,
        "sic_role": "SiC 生体適合チップ: 体内で70年安定",
        "chip_type": "SiC 3C",
        "units_yr":  500_000,
        "value_connection": "Disco 研削 → 薄化 → 体内植込み → 長寿命",
        "color": "#e45756",
    },
    "Siemens_Healthineers": {
        "label":    "Siemens Healthineers (MRI・PET)",
        "revenue_B$": 21,
        "sic_role": "SiC インバータ: MRI 超伝導磁石電源 (数MW)",
        "chip_type": "SiC 1200V",
        "units_yr":  5_000,
        "value_connection": "SiC → MRI → 脳画像 → 老化研究 → 長寿命",
        "color": "#4c78a8",
    },
    "GE_Healthcare": {
        "label":    "GE Healthcare (CT・PET・超音波)",
        "revenue_B$": 19,
        "sic_role": "GaN RF: 超音波送受信アレイ (100MHz 高周波)",
        "chip_type": "GaN",
        "units_yr":  20_000,
        "value_connection": "GaN → 超音波診断 → 早期発見 → 寿命延長",
        "color": "#4CAF50",
    },
    "Intuitive_Surgical": {
        "label":    "Intuitive Surgical (da Vinci ロボット手術)",
        "revenue_B$": 7,
        "sic_role": "GaN ESC: ロボットアーム精密制御",
        "chip_type": "GaN",
        "units_yr":  2_000,
        "value_connection": "GaN → da Vinci → 低侵襲手術 → 回復速度向上",
        "color": "#f5a623",
    },
    "Neuralink_Medical": {
        "label":    "Neuralink (BCI 医療 + 将来の超兵士)",
        "revenue_B$": 0.5,
        "sic_role": "SiC 生体チップ: 103年間 体内安定",
        "chip_type": "SiC_biocompatible",
        "units_yr":  100,
        "value_connection": "Disco ±0.3µm 研削 → N1チップ → 長寿命監視 → 2129年",
        "color": "#9C27B0",
    },
    "Proton_Therapy_Centers": {
        "label":    "陽子線治療センター (癌治療)",
        "revenue_B$": 3,
        "sic_role": "SiC 高電圧電源: 陽子加速器 (250MeV、数MW)",
        "chip_type": "SiC 3300V",
        "units_yr":  100,
        "value_connection": "SiC → 陽子線 → 癌根治率向上 → 寿命延長",
        "color": "#FF9800",
    },
}


def medical_semiconductor_market() -> dict:
    """医療産業複合体の半導体市場規模。"""
    total = sum(m["revenue_B$"] * 0.02 for m in MEDICAL_INDUSTRIAL.values())  # 売上の2%が半導体
    return {
        "total_B$": round(total, 1),
        "cagr_pct": 14,
        "longevity_connection": "全医療デバイスが長寿命化技術に直結",
        "breakdown": {k: round(v["revenue_B$"] * 0.02, 2) for k, v in MEDICAL_INDUSTRIAL.items()},
    }


if __name__ == "__main__":
    print("=" * 65)
    print("  軍産複合体 × 半導体モデル")
    print("=" * 65)

    mkt = military_sic_market()
    print(f"\n軍事 SiC/GaN 市場: ${mkt['total_M$']:.0f}M/年")
    for app, rev in sorted(mkt["breakdown"].items(), key=lambda x: x[1], reverse=True)[:5]:
        spec = MILITARY_APPLICATIONS[app]
        print(f"  {spec['label'][:38]:<40} ${rev:.0f}M/yr  ({spec['material']})")

    print("\nEMP 耐性比較:")
    for mat in ["Si_CMOS", "GaN", "SiC", "SiC_rad_hard"]:
        r = emp_hardness_model(mat)
        ok = "✓ 生存" if r["survives_nuclear_emp"] else "✗ 破壊"
        print(f"  {mat:<18} {ok}  閾値={r['threshold_kV_m']:>6}kV/m  "
              f"余裕={r['margin_dB']:>+6.1f}dB")

    print("\nデュアルユース技術サイクル:")
    for d in DUAL_USE_HISTORY[-3:]:
        print(f"  {d['tech']}: 軍 → 民 ({d['timeline_yr']}年)")
        print(f"    軍: {d['military'][:40]}")
        print(f"    民: {d['civil'][:40]}")

    print("\n未来の軍事技術 × Disco:")
    for f in FUTURE_MILITARY_TECH:
        print(f"  {f['year']}: {f['tech']}")
        print(f"    Disco接続: {f['disco_connection']}")
