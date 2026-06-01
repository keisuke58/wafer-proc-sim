"""
Advantest Model — SiC パワーデバイス ATE テスト × 歩留まり解析
==============================================================
Disco (dicing) → TEL (ALD/oxide) → Advantest (test) の完全パイプライン出口。

Advantest T2000/V93000/T5503 相当の SiC MOSFET テストフローを模擬。

1. ATE テストフロー
   SiC MOSFET の標準パラメトリックテスト:
   V_th, I_off, R_on, BV_DSS, C_iss, Qg

2. 統計的歩留まり予測
   プロセスばらつき → 正規分布 → Cp/Cpk → yield
   不良モード別分類 (parametric / catastrophic)

3. ウェーハマップ解析
   プロセス起源の空間分布パターン (center/edge/random)
   Disco ダイシング HAZ 起源の edge-die 歩留まり低下

4. テスト経済学
   Cost per wafer / Cost per good die
   テスト時間 × テスター稼働率 → テストコスト最適化

5. 完全統合パイプライン
   Disco HAZ → TEL Dit → µ_ch → R_on → ATE 合否 → yield → cost

実行:
    python fem/advantest_model.py
    python fem/advantest_model.py --test-flow
    python fem/advantest_model.py --yield-map
    python fem/advantest_model.py --economics
    python fem/advantest_model.py --full-pipeline

参考文献:
    Kimoto & Cooper (2014) SiC Technology — device specs
    Advantest T2000 SoC/Power Device Test Spec (公開情報)
    Yano et al. (2020) IEICE — SiC MOSFET ATE methodology
    Semiconductor Engineering — cost-of-test models
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── SiC MOSFET 仕様（1200V クラス、典型値）───────────────────────────────
SPEC = {
    # パラメータ: [下限, 目標中心値, 上限, 単位, 測定条件]
    "V_th":   [1.5,  3.5,  6.0,  "V",     "V_GS=V_DS, I_D=1mA"],
    "I_off":  [0,    1e-6, 1e-4, "A",     "V_DS=1200V, V_GS=0V"],
    "R_on":   [0,    5.0,  12.0, "mΩ·cm²","V_GS=15V, I_D=20A"],
    "BV_DSS": [1200, 1350, 9999, "V",     "V_GS=0V, I_D=1mA"],
    "C_iss":  [0,    800,  1500, "pF",    "V_DS=25V, f=1MHz"],
    "Qg":     [0,    45,   90,   "nC",    "V_GS=-5→15V"],
}

# ウェーハ・テスト装置パラメータ
WAFER_DIAM_MM   = 150.0      # 6インチ SiC ウェーハ
DIE_SIZE_MM     = 5.0        # mm (5×5mm² ダイ)
WAFER_COST_USD  = 2500.0     # $/wafer (6インチ SiC エピ)
TESTER_COST_USD_HR = 400.0   # $/hr (Advantest T2000 相当)
TEST_TIME_S_DIE = 0.8        # s/die (標準 SiC MOSFET テスト)
HANDLER_EFF     = 0.85       # ハンドラー稼働効率


# ════════════════════════════════════════════════════════════════════════════
# 1. ATE テストフロー
# ════════════════════════════════════════════════════════════════════════════

def simulate_device_params(mu_ch: float, Dit: float, seed: int = 42,
                            n_die: int = 500) -> dict:
    """
    プロセスシミュレーション結果から n_die 個のデバイスパラメータを生成。
    プロセスばらつきを正規分布で模擬。

    mu_ch: チャンネル移動度 [cm²/Vs]  (tel_process_model から)
    Dit:   界面トラップ密度 [cm⁻²eV⁻¹]
    """
    rng = np.random.default_rng(seed)

    # V_th: 中心値 3.5V, σ = Dit/5e12 * 0.3V
    sigma_Vth = 0.3 * (1 + Dit / 5e12)
    V_th = rng.normal(3.5, sigma_Vth, n_die)

    # R_on: µ_ch に反比例, σ = 0.5 mΩ·cm²
    R_on_center = 5.0 * (900.0 / max(mu_ch, 1.0))
    sigma_Ron   = 0.5 + 0.3 * (900 - mu_ch) / 900
    R_on = rng.normal(R_on_center, max(sigma_Ron, 0.1), n_die)

    # I_off: 対数正規分布, Dit が高いと漏れが増える
    log_Ioff_mean = -6 + 2 * Dit / 5e12   # log10(A)
    I_off = 10 ** rng.normal(log_Ioff_mean, 0.8, n_die)

    # BV_DSS: 中心 1350V, σ = 30V
    BV_DSS = rng.normal(1350, 30, n_die)

    # C_iss, Qg: ほぼ固定（プロセス依存少）
    C_iss = rng.normal(800, 30, n_die)
    Qg    = rng.normal(45,  3,  n_die)

    return {"V_th":  V_th, "I_off": I_off, "R_on":   R_on,
            "BV_DSS": BV_DSS, "C_iss": C_iss, "Qg": Qg}


def run_ate_test(params: dict) -> dict:
    """
    ATE テスト判定: 各パラメータを SPEC と比較して合否を決定。
    """
    n  = len(next(iter(params.values())))
    pass_mask = np.ones(n, dtype=bool)
    fail_modes = {k: np.zeros(n, dtype=bool) for k in SPEC}

    for param, (lo, _, hi, *_) in SPEC.items():
        if param not in params:
            continue
        fail = (params[param] < lo) | (params[param] > hi)
        fail_modes[param] = fail
        pass_mask &= ~fail

    yield_pct = pass_mask.mean() * 100
    fail_by_param = {k: v.mean() * 100 for k, v in fail_modes.items()}

    return {
        "pass_mask":     pass_mask,
        "yield_pct":     round(yield_pct, 2),
        "fail_by_param": fail_by_param,
        "n_good":        int(pass_mask.sum()),
        "n_fail":        int((~pass_mask).sum()),
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. Cp/Cpk 統計的能力指数
# ════════════════════════════════════════════════════════════════════════════

def process_capability(values: np.ndarray, lo: float, hi: float) -> dict:
    """
    Cp  = (USL - LSL) / (6σ)       プロセス能力指数
    Cpk = min((USL-µ)/3σ, (µ-LSL)/3σ)  片側余裕

    Cp > 1.33: 良好  /  Cp > 1.67: 優秀  /  Cpk ≥ 1.33: 量産可
    """
    mu    = values.mean()
    sigma = values.std()
    if sigma < 1e-15:
        return {"Cp": float("inf"), "Cpk": float("inf"), "yield_est": 100.0}

    Cp  = (hi - lo) / (6 * sigma)
    Cpk = min((hi - mu) / (3 * sigma), (mu - lo) / (3 * sigma))

    # Cpk → 不良率推定 (正規分布)
    p_fail = 2 * stats.norm.cdf(-3 * Cpk)
    return {
        "mean":      round(mu, 4),
        "sigma":     round(sigma, 4),
        "Cp":        round(Cp, 3),
        "Cpk":       round(Cpk, 3),
        "yield_est": round((1 - p_fail) * 100, 3),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. ウェーハマップ解析
# ════════════════════════════════════════════════════════════════════════════

def wafer_map(mu_ch: float, Dit: float, seed: int = 0) -> dict:
    """
    150mm ウェーハ上のダイ配置 + プロセス起源の空間不良分布。

    不良パターン:
      - エッジ不良: Disco ダイシングの HAZ 起源 (ダイエッジ)
      - 中心/周辺ムラ: TEL ALD の膜厚均一性
      - ランダム: 欠陥密度 (量子欠陥モデル由来)
    """
    rng = np.random.default_rng(seed)
    R   = WAFER_DIAM_MM / 2
    d   = DIE_SIZE_MM
    xs  = np.arange(-R + d/2, R, d)
    ys  = np.arange(-R + d/2, R, d)
    dies = [(x, y) for x in xs for y in ys if x**2 + y**2 < (R-3)**2]

    n_die = len(dies)
    params = simulate_device_params(mu_ch, Dit, seed=seed, n_die=n_die)
    test   = run_ate_test(params)
    pass_m = test["pass_mask"]

    # 空間不良パターンの追加重み
    for i, (x, y) in enumerate(dies):
        r_norm = math.sqrt(x**2 + y**2) / R
        # エッジ不良 (ダイシング HAZ): 外周 15% に集中
        edge_fail_prob = 0.3 * max(0, r_norm - 0.85) / 0.15
        # 中心ムラ (ALD 均一性): σ = 0.3%
        center_factor  = abs(1.0 - r_norm * 0.3)   # 外周で ALD 膜厚ずれ
        if rng.random() < edge_fail_prob * (1 + (900 - mu_ch) / 900):
            pass_m[i] = False

    yield_pct = pass_m.mean() * 100
    return {
        "dies":      dies,
        "pass_mask": pass_m,
        "yield_pct": round(yield_pct, 2),
        "n_die":     n_die,
        "params":    params,
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. テスト経済学
# ════════════════════════════════════════════════════════════════════════════

def test_economics(yield_pct: float, n_die: int) -> dict:
    """
    Cost per Good Die (CPGD) モデル:
        CPGD = (Wafer Cost + Test Cost) / (n_die × yield)

    テスト時間最適化:
        短縮テスト (75% 時間) → yield loss +0.5%  → CPGD 低下?
        完全テスト (100% 時間) → yield 最大       → CPGD?
    """
    n_good        = n_die * yield_pct / 100
    test_cost_die = (TEST_TIME_S_DIE / 3600 * TESTER_COST_USD_HR) / HANDLER_EFF
    test_cost_w   = test_cost_die * n_die
    total_cost_w  = WAFER_COST_USD + test_cost_w

    cpgd_full = total_cost_w / max(n_good, 1)

    # 短縮テスト (critical items only, 0.6× 時間)
    yield_short     = yield_pct - 0.8   # 短縮で見逃し +0.8%
    n_good_short    = n_die * yield_short / 100
    test_cost_short = test_cost_die * 0.6 * n_die
    cpgd_short = (WAFER_COST_USD + test_cost_short) / max(n_good_short, 1)

    return {
        "n_die":          n_die,
        "yield_pct":      round(yield_pct, 2),
        "n_good":         round(n_good, 1),
        "test_cost_usd":  round(test_cost_w, 2),
        "total_cost_usd": round(total_cost_w, 2),
        "cpgd_full_usd":  round(cpgd_full, 3),
        "cpgd_short_usd": round(cpgd_short, 3),
        "test_cost_pct":  round(test_cost_w / total_cost_w * 100, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. 完全パイプライン統合
# ════════════════════════════════════════════════════════════════════════════

def full_pipeline_with_test(dicing_processes: list) -> list:
    """
    Disco + TEL + Advantest の完全パイプライン比較。
    各ダイシングプロセスについて end-to-end コスト・歩留まりを計算。
    """
    try:
        from fem.tel_process_model import (full_pipeline as tel_pipeline,
                                            dit_from_oxidation)
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from fem.tel_process_model import (full_pipeline as tel_pipeline,
                                            dit_from_oxidation)

    results = []
    for proc in dicing_processes:
        tel = tel_pipeline(dicing_process=proc, ald_material="Al2O3",
                           ald_cycles=50, ox_T_C=1150, ox_t_hr=2.0,
                           anneal_T_C=950)
        # ウェーハマップ
        wm   = wafer_map(tel["mu_ch"], tel["Dit_cm2eV"])
        econ = test_economics(wm["yield_pct"], wm["n_die"])

        results.append({
            "process":      proc,
            "HAZ_um":       tel["HAZ_um"],
            "tau_us":       tel["tau_SRH_us"],
            "mu_ch":        tel["mu_ch"],
            "Dit":          tel["Dit_cm2eV"],
            "R_on":         tel["R_on_mohm_mm2"],
            "yield_pct":    wm["yield_pct"],
            "cpgd_usd":     econ["cpgd_full_usd"],
            "test_cost_pct":econ["test_cost_pct"],
            "n_die":        wm["n_die"],
        })
    return results


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_test_flow(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    processes = ["blade", "laser_ps", "stealth"]
    labels    = {"blade": "ブレード", "laser_ps": "レーザー(ps)", "stealth": "ステルス"}
    colors    = {"blade": "#d62728",  "laser_ps": "#2ca02c", "stealth": "#2166ac"}
    mu_map    = {"blade": 865, "laser_ps": 874, "stealth": 881}
    dit_map   = {"blade": 6e11, "laser_ps": 3e11, "stealth": 1e11}

    # Panel 1-3: V_th, R_on, BV_DSS 分布
    for ax, (param, unit) in zip(axes.flat[:3],
                                  [("V_th","V"), ("R_on","mΩ·cm²"), ("BV_DSS","V")]):
        lo, _, hi = SPEC[param][:3]
        for proc in processes:
            p = simulate_device_params(mu_map[proc], dit_map[proc], n_die=300)
            vals = p[param]
            ax.hist(vals, bins=30, alpha=0.5, color=colors[proc],
                    label=labels[proc], density=True)
        ax.axvline(lo, color="red",  ls="--", lw=1.5, label="LSL")
        ax.axvline(hi, color="red",  ls="--", lw=1.5, label="USL")
        ax.set_xlabel(f"{param} [{unit}]", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{param} 分布", fontsize=11)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    # Panel 4: Cpk 比較
    ax4 = axes[1, 0]
    params_cmp = ["V_th", "R_on", "BV_DSS"]
    x = np.arange(len(params_cmp))
    w = 0.25
    for k, proc in enumerate(processes):
        cpks = []
        p = simulate_device_params(mu_map[proc], dit_map[proc], n_die=500)
        for param in params_cmp:
            lo, _, hi = SPEC[param][:3]
            cap = process_capability(p[param], lo, hi)
            cpks.append(cap["Cpk"])
        ax4.bar(x + (k-1)*w, cpks, w, color=colors[proc],
                label=labels[proc], alpha=0.85, edgecolor="k", lw=0.5)
    ax4.axhline(1.33, color="green", ls="--", lw=1.5, label="Cpk=1.33 (量産可)")
    ax4.set_xticks(x); ax4.set_xticklabels(params_cmp)
    ax4.set_ylabel("Cpk")
    ax4.set_title("プロセス能力指数 Cpk", fontsize=11)
    ax4.legend(fontsize=7); ax4.grid(alpha=0.2, axis="y")

    # Panel 5: 不良モード内訳
    ax5 = axes[1, 1]
    for k, proc in enumerate(processes):
        p = simulate_device_params(mu_map[proc], dit_map[proc], n_die=500)
        test = run_ate_test(p)
        modes = list(test["fail_by_param"].keys())
        vals  = [test["fail_by_param"][m] for m in modes]
        bot   = 0
        for m, v in zip(modes, vals):
            ax5.bar(k, v, bottom=bot, label=m if k==0 else "",
                    alpha=0.8, edgecolor="k", lw=0.3)
            bot += v
    ax5.set_xticks(range(3))
    ax5.set_xticklabels([labels[p] for p in processes])
    ax5.set_ylabel("Fail Rate [%]")
    ax5.set_title("不良モード内訳", fontsize=11)
    ax5.legend(fontsize=7, loc="upper left"); ax5.grid(alpha=0.2, axis="y")

    # Panel 6: 歩留まり比較
    ax6 = axes[1, 2]
    yields = []
    for proc in processes:
        p    = simulate_device_params(mu_map[proc], dit_map[proc], n_die=500)
        test = run_ate_test(p)
        yields.append(test["yield_pct"])
    bars = ax6.bar([labels[p] for p in processes], yields,
                   color=[colors[p] for p in processes],
                   alpha=0.85, edgecolor="k", lw=0.8)
    for bar, y in zip(bars, yields):
        ax6.text(bar.get_x() + bar.get_width()/2, y + 0.2,
                 f"{y:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax6.set_ylabel("ATE 歩留まり [%]")
    ax6.set_title("プロセス別 ATE 歩留まり", fontsize=11)
    ax6.set_ylim(0, 105); ax6.grid(alpha=0.2, axis="y")

    fig.suptitle("Advantest ATE — SiC MOSFET パラメトリックテスト\n"
                 "Disco ダイシングプロセス × デバイス歩留まり", fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "advantest_test_flow.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] ATE test flow -> {out}")


def plot_yield_map(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    configs = [
        ("blade",    865, 6e11, "ブレード"),
        ("laser_ps", 874, 3e11, "レーザー(ps)"),
        ("stealth",  881, 1e11, "ステルス"),
    ]

    for ax, (proc, mu, dit, lbl) in zip(axes, configs):
        wm = wafer_map(mu, dit)
        dies     = wm["dies"]
        pass_m   = wm["pass_mask"]
        xs       = [d[0] for d in dies]
        ys       = [d[1] for d in dies]
        colors_m = ["#2ca02c" if p else "#d62728" for p in pass_m]

        ax.scatter(xs, ys, c=colors_m, s=8, alpha=0.8)
        # ウェーハ境界
        theta = np.linspace(0, 2*np.pi, 200)
        R = WAFER_DIAM_MM / 2
        ax.plot(R*np.cos(theta), R*np.sin(theta), "k-", lw=1.5)
        ax.plot((R-3)*np.cos(theta), (R-3)*np.sin(theta),
                "gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{lbl}\n歩留まり {wm['yield_pct']:.1f}%  (n={wm['n_die']})",
                     fontsize=10)
        ax.set_xlabel("x [mm]", fontsize=9)
        ax.set_ylabel("y [mm]", fontsize=9)
        good_p = mpatches.Patch(color="#2ca02c", label="PASS")
        fail_p = mpatches.Patch(color="#d62728", label="FAIL")
        ax.legend(handles=[good_p, fail_p], fontsize=8, loc="lower right")

    fig.suptitle("ウェーハマップ — プロセス起源の空間不良分布\n"
                 "エッジ不良 (Disco HAZ) + ランダム不良 (量子欠陥)\n"
                 "150mm 6インチ SiC ウェーハ", fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "advantest_wafer_map.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Wafer map -> {out}")


def plot_economics(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    procs  = ["blade", "laser_ps", "stealth"]
    labels = ["ブレード", "レーザー(ps)", "ステルス"]
    colors = ["#d62728", "#2ca02c", "#2166ac"]
    mu_map = {"blade": 865, "laser_ps": 874, "stealth": 881}
    dit_map= {"blade": 6e11, "laser_ps": 3e11, "stealth": 1e11}

    cpgds, yields, test_pcts = [], [], []
    for proc in procs:
        wm   = wafer_map(mu_map[proc], dit_map[proc])
        econ = test_economics(wm["yield_pct"], wm["n_die"])
        cpgds.append(econ["cpgd_full_usd"])
        yields.append(wm["yield_pct"])
        test_pcts.append(econ["test_cost_pct"])

    # Panel 1: CPGD 比較
    ax = axes[0]
    bars = ax.bar(labels, cpgds, color=colors, alpha=0.85,
                  edgecolor="k", lw=0.8)
    for bar, v in zip(bars, cpgds):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"${v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Cost per Good Die [USD]", fontsize=11)
    ax.set_title("Cost per Good Die (CPGD)\nDiscoプロセス別", fontsize=11)
    ax.grid(alpha=0.2, axis="y")

    # Panel 2: コスト内訳 × 歩留まり感度
    ax2 = axes[1]
    yield_range = np.linspace(70, 99, 100)
    n_die_est   = 300   # typical 150mm wafer
    for proc, col, lbl in zip(procs, colors, labels):
        cpgds_y = []
        for y in yield_range:
            econ = test_economics(y, n_die_est)
            cpgds_y.append(econ["cpgd_full_usd"])
        ax2.plot(yield_range, cpgds_y, color=col, lw=2, label=lbl)
        wm = wafer_map(mu_map[proc], dit_map[proc])
        ax2.axvline(wm["yield_pct"], color=col, ls=":", lw=1.2, alpha=0.7)

    ax2.set_xlabel("Wafer Yield [%]", fontsize=11)
    ax2.set_ylabel("CPGD [USD]", fontsize=11)
    ax2.set_title("CPGD vs 歩留まり感度\n(点線: 各プロセスの実歩留まり)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    fig.suptitle("テスト経済学 — Advantest T2000 相当\n"
                 "Wafer Cost $2500 + Test Time 0.8s/die @ $400/hr",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "advantest_economics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Test economics -> {out}")


def plot_full_pipeline(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    procs = ["blade", "laser_ps", "laser_fs", "stealth"]
    labels = {"blade": "ブレード", "laser_ps": "レーザーps",
              "laser_fs": "レーザーfs", "stealth": "ステルス"}
    colors = {"blade": "#d62728", "laser_ps": "#ff7f0e",
              "laser_fs": "#2ca02c", "stealth": "#2166ac"}

    results = full_pipeline_with_test(procs)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    metrics = [
        ("HAZ_um",    "HAZ 深さ [µm]"),
        ("tau_us",    "SRH 寿命 τ [µs]"),
        ("mu_ch",     "チャンネル移動度 [cm²/Vs]"),
        ("R_on",      "R_on [mΩ·mm²]"),
        ("yield_pct", "ATE 歩留まり [%]"),
        ("cpgd_usd",  "CPGD [USD]"),
    ]

    for ax, (key, ylabel) in zip(axes.flat, metrics):
        vals = [r[key] for r in results]
        lbls = [labels[r["process"]] for r in results]
        cols = [colors[r["process"]] for r in results]
        bars = ax.bar(lbls, vals, color=cols, alpha=0.85,
                      edgecolor="k", lw=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + max(vals)*0.01,
                    f"{v:.2f}" if v < 100 else f"{v:.1f}",
                    ha="center", fontsize=8, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel.split("[")[0].strip(), fontsize=11)
        ax.grid(alpha=0.2, axis="y")
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("完全プロセスパイプライン統合結果\n"
                 "Disco ダイシング → TEL ALD/酸化 → Advantest ATE → CPGD",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "full_pipeline_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # テキストサマリー
    print(f"\n  {'プロセス':>12}  {'HAZ':>6}  {'τ[µs]':>7}  {'µ_ch':>7}  "
          f"{'Yield%':>8}  {'CPGD[$]':>8}")
    print("  " + "─"*60)
    for r in results:
        print(f"  {labels[r['process']]:>12}  {r['HAZ_um']:>6.2f}  "
              f"{r['tau_us']:>7.1f}  {r['mu_ch']:>7.1f}  "
              f"{r['yield_pct']:>8.2f}  {r['cpgd_usd']:>8.2f}")
    print(f"[✓] Full pipeline summary -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-flow",     action="store_true")
    ap.add_argument("--yield-map",     action="store_true")
    ap.add_argument("--economics",     action="store_true")
    ap.add_argument("--full-pipeline", action="store_true")
    args = ap.parse_args()

    run_all = not any([args.test_flow, args.yield_map,
                       args.economics, args.full_pipeline])

    if args.test_flow     or run_all: plot_test_flow()
    if args.yield_map     or run_all: plot_yield_map()
    if args.economics     or run_all: plot_economics()
    if args.full_pipeline or run_all: plot_full_pipeline()


if __name__ == "__main__":
    main()
