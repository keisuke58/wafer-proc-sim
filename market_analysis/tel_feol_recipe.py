"""
TEL FEOL Recipe — forward model (recipe → SiC MOSFET 性能)
============================================================
魔改造 TEL の土台。既存の物理ブロック(tel_process_model / tel_cleaning_model /
quantum_defect_model)を 1 本の「レシピ → 性能指標 (Figures of Merit)」写像に統合する。

この forward model の上に以下が乗る:
    ml/tel_feol_surrogate.py        GP 代理モデル (recipe → FOM)
    optimization/tel_feol_bo.py     多目的ベイズ最適化 (Pareto)
    pipeline/tel_apc.py             EnKF run-to-run 制御 (Advanced Process Control)
    optimization/tel_tmcmc_calibration.py  TMCMC 物理パラメータ較正

完全 FEOL フロー (Disco → TEL):
    [1] 洗浄 (CELLESTA)        残留炭素 → Dit 前駆体
    [2] ダイシング誘起欠陥      HAZ / Ra → trap density
    [3] RIE 表面処理           機械損傷層除去 + 追加損傷
    [4] 熱酸化 (Deal-Grove)    SiO₂ 界面 + Dit
    [5] NO ポストアニール       Dit 低減
    [6] ALD ゲート誘電体        EOT / Cox
    [7] SiC MOSFET 性能         µ_ch / V_th / I_on / R_on,sp

実行:
    python fem/tel_feol_recipe.py              # default recipe を評価
    python fem/tel_feol_recipe.py --sample 8   # 8 レシピをサンプリングして評価

参考: Kimoto & Cooper (2014); Saks (1999); Deal & Grove (1965); George (2010)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from market_analysis.tel_process_model import (
    ald_film,
    deal_grove,
    dit_from_oxidation,
    rie_damage,
    channel_mobility_inv,
    mosfet_performance,
)
from market_analysis.tel_cleaning_model import run_sequence, SurfaceState

eps0 = 8.854e-12  # F/m
eV = 1.602e-19    # C

# ════════════════════════════════════════════════════════════════════════════
# レシピ空間 (最適化変数)
# ════════════════════════════════════════════════════════════════════════════

# 連続変数: name -> (lo, hi) [物理単位]
CONTINUOUS = {
    "ox_T_C":     (1050.0, 1300.0),   # 熱酸化温度 [°C]
    "ox_t_hr":    (0.5, 4.0),          # 熱酸化時間 [hr]
    "anneal_T_C": (800.0, 1100.0),     # NO ポストアニール温度 [°C]
    "ald_cycles": (40.0, 200.0),       # ALD サイクル数
}

# カテゴリ変数: name -> [選択肢]
CATEGORICAL = {
    "clean_seq":       ["pregate_si", "pregate_sic"],
    "dicing_process":  ["blade", "laser_ps", "stealth"],
    "rie_gas":         ["CF4/O2", "SF6", "Cl2"],
    "ald_material":    ["Al2O3", "HfO2"],
}

# GP / BO 用ベクトル順序 (連続 → カテゴリ index)
VECTOR_NAMES = list(CONTINUOUS.keys()) + list(CATEGORICAL.keys())
VECTOR_BOUNDS = (
    [CONTINUOUS[k] for k in CONTINUOUS]
    + [(0.0, len(CATEGORICAL[k]) - 1) for k in CATEGORICAL]
)

# ダイシングプロセス → (HAZ µm, Ra nm)  [tel_process_model.full_pipeline と整合]
_HAZ = {"blade": 2.5, "laser_ps": 0.5, "stealth": 0.05}
_RA = {"blade": 80.0, "laser_ps": 30.0, "stealth": 5.0}


def default_recipe() -> dict:
    """基準レシピ (Kimoto & Cooper 標準 SiC MOSFET 条件)。"""
    return {
        "ox_T_C": 1150.0,
        "ox_t_hr": 2.0,
        "anneal_T_C": 950.0,
        "ald_cycles": 100,
        "clean_seq": "pregate_sic",
        "dicing_process": "blade",
        "rie_gas": "CF4/O2",
        "ald_material": "Al2O3",
    }


# ════════════════════════════════════════════════════════════════════════════
# Forward model: recipe → Figures of Merit
# ════════════════════════════════════════════════════════════════════════════

def simulate_recipe(recipe: dict, tool_state: dict = None) -> dict:
    """
    レシピ → SiC MOSFET 性能指標 (FOM)。

    tool_state: APC/EnKF 用の装置ドリフト。None なら公称。
        {"ox_T_offset_C": float, "ald_gpc_scale": float}
        - ox_T_offset_C : 酸化炉の温度ドリフト [°C]
        - ald_gpc_scale : ALD 成膜速度のドリフト倍率 (1.0 = 公称)

    返り値 FOM:
        mu_ch       チャンネル移動度 [cm²/Vs]   (大きいほど良)
        Dit         界面トラップ密度 [cm⁻²eV⁻¹] (小さいほど良)
        EOT_nm      等価酸化膜厚 [nm]
        V_th        閾値電圧 [V]                (2–4V が manufacturable)
        I_on_mA     オン電流 [mA]               (大きいほど良)
        R_on_sp     比オン抵抗 [mΩ·mm²]         (小さいほど良)
        uniformity  ALD 膜厚均一性 [%]          (小さいほど良)
        thermal_budget  熱予算コスト (throughput proxy, 小さいほど良)
    """
    ts = tool_state or {}
    ox_T = recipe["ox_T_C"] + ts.get("ox_T_offset_C", 0.0)
    ox_t = recipe["ox_t_hr"]
    anneal_T = recipe["anneal_T_C"]
    cycles = int(round(recipe["ald_cycles"]))
    clean_seq = recipe["clean_seq"]
    dicing = recipe["dicing_process"]
    rie_gas = recipe["rie_gas"]
    material = recipe["ald_material"]

    # ── [1] 洗浄 → 残留炭素由来 Dit ─────────────────────────────
    clean = run_sequence(clean_seq, dicing)
    Dit_clean = clean["after"]["dit_contrib"]   # cm⁻²eV⁻¹

    # ── [2] ダイシング誘起欠陥 → 表面 Dit 寄与 ──────────────────
    HAZ = _HAZ.get(dicing, 2.5)
    Ra = _RA.get(dicing, 80.0)

    # ── [3] RIE 表面処理 (機械損傷層を部分除去) ─────────────────
    rie = rie_damage(2.0, rie_gas, power_W=100.0)
    HAZ_after = max(0.0, HAZ - rie["etch_nm"] * 1e-3 * 0.5)
    Dit_dicing = 1e11 * (max(HAZ_after, 1e-3) / 0.1) ** 0.5   # cm⁻²eV⁻¹

    # ── [4]+[5] 熱酸化 + NO アニール → Dit ──────────────────────
    ox = deal_grove(ox_t, ox_T, "4H-SiC")
    Dit_ox = dit_from_oxidation(ox_T, ox_t, surface="Si-face", anneal_T_C=anneal_T)

    # ── [6] ALD ゲート誘電体 ────────────────────────────────────
    ald = ald_film(cycles, 250.0, material)
    gpc_scale = ts.get("ald_gpc_scale", 1.0)
    thickness_nm = ald["thickness_nm"] * gpc_scale
    EOT_ald = thickness_nm * 3.9 / ald["k"]
    Cox = ald["k"] * eps0 / (thickness_nm * 1e-9) * 1e3   # fF/µm²

    # ── Dit 統合: high-k は酸化/ダイシング Dit を強くパッシベート ──
    passivation = 0.2 if material in ("Al2O3", "HfO2") else 0.8
    Dit_total = (Dit_ox + Dit_dicing + Dit_clean) * passivation

    # ── [7] MOSFET 性能 ────────────────────────────────────────
    mu_ch = channel_mobility_inv(Dit_total, Cox, anneal="none")

    EOT_nm = ox["x_ox_nm"] + EOT_ald
    Cox_Fcm2 = 3.9 * eps0 / max(EOT_nm, 0.1) * 1e9 * 1e-4   # F/cm²
    N_it = Dit_total * 0.1                                  # cm⁻²
    V_th = 3.0 + eV * N_it / max(Cox_Fcm2, 1e-9)

    perf = mosfet_performance(mu_ch, Cox, V_th)

    # ── 熱予算コスト (throughput / cost proxy, 正規化) ──────────
    thermal_budget = (
        (ox_T * ox_t) / 2500.0
        + (anneal_T * 0.5) / 2500.0
        + cycles / 200.0
    )

    return {
        "mu_ch": float(mu_ch),
        "Dit": float(Dit_total),
        "EOT_nm": float(EOT_nm),
        "V_th": float(V_th),
        "I_on_mA": float(perf["I_on_mA"]),
        "R_on_sp": float(perf["R_on_mohm_mm2"]),
        "uniformity": float(ald["uniformity_pct"]),
        "thermal_budget": float(thermal_budget),
    }


# ── 多目的 / スカラー目的関数 ───────────────────────────────────────────────

#: 多目的最適化の目的 (name, "max"|"min")。Pareto はこの3軸で張る。
OBJECTIVES = [
    ("mu_ch", "max"),          # デバイス品質 (反転層移動度)
    ("R_on_sp", "min"),        # デバイス性能 (比オン抵抗)
    ("thermal_budget", "min"), # 製造コスト / throughput
]


def objective_vector(recipe: dict, tool_state: dict = None) -> np.ndarray:
    """多目的ベクトル。全て「最小化」に符号統一 (max は負号)。"""
    fom = simulate_recipe(recipe, tool_state)
    out = []
    for name, sense in OBJECTIVES:
        v = fom[name]
        out.append(-v if sense == "max" else v)
    return np.array(out, dtype=float)


def scalar_fom(recipe: dict, tool_state: dict = None) -> float:
    """
    スカラー複合 FOM (大きいほど良)。BO・APC のターゲット。
        µ_ch を主指標とし、V_th が製造ウィンドウ [2,4]V を外れたら罰則。
    """
    fom = simulate_recipe(recipe, tool_state)
    penalty = 1.0
    if not (2.0 <= fom["V_th"] <= 4.0):
        penalty = 0.5
    return fom["mu_ch"] * penalty


# ════════════════════════════════════════════════════════════════════════════
# レシピ ⇄ 数値ベクトル (GP / BO 用)
# ════════════════════════════════════════════════════════════════════════════

def recipe_to_vector(recipe: dict) -> np.ndarray:
    """レシピ dict → 数値ベクトル (連続は物理単位、カテゴリは index)。"""
    vec = [float(recipe[k]) for k in CONTINUOUS]
    for k in CATEGORICAL:
        vec.append(float(CATEGORICAL[k].index(recipe[k])))
    return np.array(vec, dtype=float)


def vector_to_recipe(vec: np.ndarray) -> dict:
    """数値ベクトル → レシピ dict。カテゴリは最近傍 index に丸める。"""
    vec = np.asarray(vec, dtype=float)
    rec = {}
    for i, k in enumerate(CONTINUOUS):
        lo, hi = CONTINUOUS[k]
        rec[k] = float(np.clip(vec[i], lo, hi))
    off = len(CONTINUOUS)
    for j, k in enumerate(CATEGORICAL):
        idx = int(np.clip(round(vec[off + j]), 0, len(CATEGORICAL[k]) - 1))
        rec[k] = CATEGORICAL[k][idx]
    rec["ald_cycles"] = float(round(rec["ald_cycles"]))
    return rec


def sample_recipes(n: int, seed: int = 0) -> list:
    """レシピ空間を Latin-Hypercube 風に一様サンプリング (代理モデル学習用)。"""
    rng = np.random.default_rng(seed)
    recipes = []
    for _ in range(n):
        rec = {}
        for k, (lo, hi) in CONTINUOUS.items():
            rec[k] = float(rng.uniform(lo, hi))
        rec["ald_cycles"] = float(round(rec["ald_cycles"]))
        for k, choices in CATEGORICAL.items():
            rec[k] = choices[int(rng.integers(len(choices)))]
        recipes.append(rec)
    return recipes


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="TEL FEOL recipe forward model")
    ap.add_argument("--sample", type=int, default=0,
                    help="N 個のランダムレシピを評価")
    args = ap.parse_args()

    if args.sample > 0:
        recs = sample_recipes(args.sample, seed=1)
        print(f"{'µ_ch':>7} {'Dit':>10} {'V_th':>6} {'R_on_sp':>8} "
              f"{'EOT':>6} {'tbudget':>8}  recipe")
        for r in recs:
            f = simulate_recipe(r)
            print(f"{f['mu_ch']:7.1f} {f['Dit']:10.2e} {f['V_th']:6.2f} "
                  f"{f['R_on_sp']:8.3f} {f['EOT_nm']:6.2f} {f['thermal_budget']:8.3f}  "
                  f"{r['ald_material']:6} {r['dicing_process']:9} {r['clean_seq']}")
    else:
        r = default_recipe()
        f = simulate_recipe(r)
        print("Default recipe:", r)
        print("FOM:")
        for k, v in f.items():
            print(f"  {k:16} {v:.4g}")
        print(f"  scalar_fom       {scalar_fom(r):.4g}")


if __name__ == "__main__":
    main()
