"""
Ga₂O₃ (酸化ガリウム) 完全デバイス物理モデル
==============================================
β-Ga₂O₃ は次世代パワー半導体の本命。
バンドギャップ 4.8eV、絶縁破壊電界 8MV/cm で SiC・GaN を凌駕。

最大課題: 熱伝導率 11 W/mK (SiC の 1/45) → 解決策を定量化

References:
  - Pearton et al. APL 2018 "A review of Ga₂O₃ materials, processing, and devices"
  - Wong et al. IEEE EDL 2021 (breakdown measurements)
  - Higashiwaki et al. Semicond. Sci. Technol. 2016
  - Bae et al. APL 2022 (diamond bonding thermal management)
"""

import math

# ════════════════════════════════════════════════════════════════════════════
# 材料定数
# ════════════════════════════════════════════════════════════════════════════

GA2O3_PROPERTIES = {
    "bandgap_eV":             4.80,   # vs SiC 3.26, GaN 3.40
    "breakdown_field_MV_cm":  8.0,    # vs SiC 2.5, GaN 3.3
    "electron_mobility":      200,    # cm²/Vs (バルク; 2DEG は ~300)
    "thermal_conductivity":   11,     # W/mK ← 最大弱点 (SiC: 490, GaN: 130)
    "dielectric_const":       10,
    "electron_sat_velocity":  2e7,    # cm/s
    "density_g_cm3":          5.88,
    "lattice_a_nm":           1.223,
    "melting_point_C":        1740,
    "substrate_cost_6in_usd": 3000,   # SiC $800、GaN-on-Si $150 と比較
    "wafer_availability":     "6-inch (2024年商用化開始)",
    "p_type_doping":          False,  # n型のみ → 大きな制限
}

COMPARE_MATERIALS = {
    "Si":    {"Eg": 1.12, "Ec": 0.3,  "mu": 1500, "kth": 150, "Baliga": 1},
    "SiC":   {"Eg": 3.26, "Ec": 2.5,  "mu": 900,  "kth": 490, "Baliga": 340},
    "GaN":   {"Eg": 3.40, "Ec": 3.3,  "mu": 2000, "kth": 130, "Baliga": 870},
    "Ga2O3": {"Eg": 4.80, "Ec": 8.0,  "mu": 200,  "kth": 11,  "Baliga": 3444},
    "Diamond":{"Eg": 5.47,"Ec": 10.0, "mu": 4500, "kth": 2200,"Baliga": 24660},
}


def baliga_figure_of_merit(material: str = "Ga2O3") -> dict:
    """
    Baliga 品質指数 (FOM) = ε × μ × Ec³
    パワーデバイスの理論性能指数。Si = 1 で規格化。

    高いほど: 低い Ron、高い耐圧、小さいチップ面積
    """
    p = COMPARE_MATERIALS[material]
    p_si = COMPARE_MATERIALS["Si"]

    fom_abs = p["mu"] * p["Ec"] ** 3
    fom_si  = p_si["mu"] * p_si["Ec"] ** 3
    fom_rel = fom_abs / fom_si

    return {
        "material":  material,
        "FOM_rel":   round(fom_rel, 0),
        "Eg_eV":     p["Eg"],
        "Ec_MV_cm":  p["Ec"],
        "mu_cm2Vs":  p["mu"],
        "kth_W_mK":  p["kth"],
        "Baliga":    p.get("Baliga", round(fom_rel, 0)),
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. Ron × Vbr トレードオフ (Baliga トレードオフ曲線)
# ════════════════════════════════════════════════════════════════════════════

def ron_vbr_tradeoff(V_br: float, material: str = "Ga2O3") -> dict:
    """
    耐圧 V_br [V] → 最小 Ron [mΩ·cm²]。
    理論式: Ron_min = 4·V_br² / (ε·μ·Ec³)

    V_br が同じなら: Ron_Ga2O3 << Ron_SiC << Ron_GaN << Ron_Si
    """
    p = COMPARE_MATERIALS[material]
    eps = p.get("dielectric_const", 10) * 8.854e-14   # F/cm
    # 各材料の誘電率
    eps_map = {"Si": 11.7, "SiC": 9.7, "GaN": 9.0, "Ga2O3": 10.0, "Diamond": 5.7}
    eps_r = eps_map.get(material, 10.0)
    eps = eps_r * 8.854e-14   # F/cm

    mu_cm2Vs = p["mu"]
    Ec_V_cm  = p["Ec"] * 1e6   # V/cm

    q = 1.6e-19
    Ron_min_mohm_cm2 = 4 * V_br**2 / (eps * mu_cm2Vs * Ec_V_cm**3) * 1e3

    return {
        "material":       material,
        "V_br_V":         V_br,
        "Ron_min_mohm_cm2": round(Ron_min_mohm_cm2, 4),
        "vs_SiC":         round(
            Ron_min_mohm_cm2 / ron_vbr_tradeoff(V_br, "SiC")["Ron_min_mohm_cm2"], 2
        ) if material != "SiC" else 1.0,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. 熱問題 — Ga₂O₃ 最大の弱点と解決策
# ════════════════════════════════════════════════════════════════════════════

THERMAL_SOLUTIONS = {
    "standalone": {
        "label":     "単体 Ga₂O₃ (解決なし)",
        "kth_eff":   11,
        "cost_mult": 1.0,
        "feasible":  False,
        "note":      "Tj が急上昇 → 実用不可 (高電力)",
    },
    "SiC_substrate": {
        "label":     "SiC 基板ウェーハボンディング",
        "kth_eff":   180,   # Ga₂O₃薄膜 + SiC 基板の複合
        "cost_mult": 3.0,
        "feasible":  True,
        "note":      "Ga₂O₃ 薄膜 (1µm) + SiC 基板 → 熱は SiC が担う",
    },
    "diamond_bonding": {
        "label":     "ダイヤモンド基板直接接合",
        "kth_eff":   800,   # ダイヤモンド面内
        "cost_mult": 15.0,
        "feasible":  True,
        "note":      "Bae et al. 2022: Ron を保ちながら Tj を 1/70 に",
    },
    "flip_chip": {
        "label":     "フリップチップ + ダイヤモンドヒートスプレッダ",
        "kth_eff":   400,
        "cost_mult": 5.0,
        "feasible":  True,
        "note":      "現実的な中期解: 既存 FC プロセスを流用",
    },
    "liquid_metal": {
        "label":     "液体金属 (In-Ga) 冷却",
        "kth_eff":   600,
        "cost_mult": 4.0,
        "feasible":  True,
        "note":      "研究段階。実装が複雑だが効果大",
    },
}


def thermal_model_ga2o3(P_W: float, A_mm2: float,
                         solution: str = "SiC_substrate",
                         T_ambient_C: float = 25) -> dict:
    """
    Ga₂O₃ デバイスの接合温度計算。
    P: 損失電力 [W], A: チップ面積 [mm²]

    熱抵抗 Rth = t / (k × A)
    Tj = T_ambient + P × Rth
    """
    sol = THERMAL_SOLUTIONS[solution]
    kth = sol["kth_eff"]   # W/mK

    t_chip_m = 150e-6      # チップ厚さ 150µm
    A_m2     = A_mm2 * 1e-6

    Rth_chip = t_chip_m / (kth * A_m2)   # K/W
    Rth_pkg  = 0.5                         # パッケージ熱抵抗

    Rth_total = Rth_chip + Rth_pkg
    Tj        = T_ambient_C + P_W * Rth_total

    return {
        "solution":     solution,
        "label":        sol["label"],
        "kth_eff":      kth,
        "Rth_K_W":      round(Rth_total, 2),
        "Tj_C":         round(Tj, 1),
        "safe":         Tj < 250,   # Ga₂O₃ 最大動作温度 ~250°C (実用)
        "feasible":     sol["feasible"],
        "cost_mult":    sol["cost_mult"],
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. 市場タイムライン — SiC を超える時期
# ════════════════════════════════════════════════════════════════════════════

GA2O3_ROADMAP = [
    {"year": 2024, "stage": "学術研究",   "V_br_kV": 3,   "Ron": 5.0,   "wafer_in": 4,  "cost_mult_vs_sic": 8},
    {"year": 2026, "stage": "試作品",     "V_br_kV": 5,   "Ron": 2.0,   "wafer_in": 6,  "cost_mult_vs_sic": 5},
    {"year": 2028, "stage": "小量産",     "V_br_kV": 8,   "Ron": 1.0,   "wafer_in": 6,  "cost_mult_vs_sic": 3},
    {"year": 2030, "stage": "ニッチ商用", "V_br_kV": 10,  "Ron": 0.5,   "wafer_in": 6,  "cost_mult_vs_sic": 2},
    {"year": 2033, "stage": "核融合採用", "V_br_kV": 15,  "Ron": 0.3,   "wafer_in": 8,  "cost_mult_vs_sic": 1.5},
    {"year": 2037, "stage": "主流化",     "V_br_kV": 20,  "Ron": 0.15,  "wafer_in": 8,  "cost_mult_vs_sic": 1.2},
    {"year": 2042, "stage": "SiC を超える","V_br_kV": 30, "Ron": 0.08,  "wafer_in": 12, "cost_mult_vs_sic": 0.9},
]

# ════════════════════════════════════════════════════════════════════════════
# Quick demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  Ga₂O₃ 完全デバイス物理モデル")
    print("=" * 65)

    print("\n材料比較 — Baliga FOM (Si=1):")
    for mat in COMPARE_MATERIALS:
        r = baliga_figure_of_merit(mat)
        print(f"  {mat:<10} FOM={r['FOM_rel']:>8,.0f}×  "
              f"Eg={r['Eg_eV']:.1f}eV  Ec={r['Ec_MV_cm']:.1f}MV/cm  "
              f"kth={r['kth_W_mK']}W/mK")

    print("\nRon × Vbr トレードオフ @ 10kV:")
    for mat in ["Si", "SiC", "GaN", "Ga2O3"]:
        r = ron_vbr_tradeoff(10_000, mat)
        print(f"  {mat:<10} Ron={r['Ron_min_mohm_cm2']:.4f} mΩ·cm²")

    print("\n熱問題 解決策比較 (30W, 10mm²):")
    for sol in THERMAL_SOLUTIONS:
        r = thermal_model_ga2o3(30, 10, sol)
        ok = "✓" if r["safe"] else "✗"
        print(f"  {ok} {r['label']:<35} Tj={r['Tj_C']:>6.0f}°C  "
              f"kth={r['kth_eff']:>5}W/mK  コスト×{r['cost_mult']:.0f}")

    print("\n市場ロードマップ:")
    for r in GA2O3_ROADMAP:
        print(f"  {r['year']} [{r['stage']:<12}] "
              f"Vbr={r['V_br_kV']:>3}kV  Ron={r['Ron']:.2f}mΩ·cm²  "
              f"コスト SiC比×{r['cost_mult_vs_sic']}")
