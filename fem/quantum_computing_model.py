"""
量子コンピュータ プロセス物理モデル
=====================================
5〜10 年後を見据えた量子デバイス製造の物理シミュレーション。

2 種類の主要 qubit 方式をモデル化:
  (A) Si スピン qubit  — CMOSプロセス親和性が高く, Terafab と直結
  (B) 超伝導 transmon qubit — IBM/Google 現在主流

モデル内容:
  1. Si スピン qubit コヒーレンス
     T1/T2 vs 界面粗さ (Rms) / 不純物密度 / 磁場
  2. 超伝導 transmon — Josephson 接合
     EJ/EC 比 vs アンハーモニシティ / qubit 周波数
  3. 量子エラー訂正スケーリング (Surface Code)
     物理 qubit → 論理 qubit 変換 + 必要物理 qubit 数
  4. 5〜10 年ロードマップ
     NISQ → フォールトトレラント → ユーティリティスケール
  5. Si qubit × Terafab 製造可能性
     300mm ウェーハでの qubit 集積密度

実行:
    python fem/quantum_computing_model.py
    python fem/quantum_computing_model.py --spin
    python fem/quantum_computing_model.py --transmon
    python fem/quantum_computing_model.py --qec
    python fem/quantum_computing_model.py --roadmap

参考文献:
    Zwanenburg et al. (2013) Rev. Mod. Phys. — Silicon quantum electronics
    Koch et al. (2007) PRA — Transmon qubit design
    Fowler et al. (2012) PRA — Surface code thresholds
    Google Quantum AI (2023) Nature — Suppressing quantum errors
    IBM Quantum Roadmap (2025) — 100k+ qubit targets
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 物理定数
hbar = 1.055e-34    # J·s
k_B  = 1.381e-23    # J/K
q_e  = 1.602e-19    # C
Phi0 = 2.068e-15    # 磁束量子 [Wb]

# ════════════════════════════════════════════════════════════════════════════
# 1. Si スピン qubit — コヒーレンス時間モデル
# ════════════════════════════════════════════════════════════════════════════

def si_spin_qubit_coherence(rms_nm: float,
                             n_imp_cm2: float = 1e10,
                             B_mT: float = 1.0,
                             T_mK: float = 20.0) -> dict:
    """
    Si/SiO2 界面スピン qubit の T1・T2 コヒーレンス時間を推定。

    T1 (緩和): フォノン散乱 + 電荷ノイズが支配
        1/T1 ∝ (Δ/ωq)² × γ_ph × T + γ_charge × n_imp
    T2 (脱コヒーレンス): 核スピンノイズ (²⁹Si) + 電荷ノイズ
        1/T2 = 1/(2T1) + 1/T2*  (T2* = 核スピン起因)

    Args:
        rms_nm:    界面粗さ RMS [nm]
        n_imp_cm2: 界面不純物密度 [/cm²]
        B_mT:      外部磁場 [mT]
        T_mK:      動作温度 [mK]
    """
    T_K = T_mK * 1e-3

    # qubit 遷移周波数 (Zeeman splitting)
    g_factor  = 2.0        # Si 電子 g 因子
    mu_B      = 9.274e-24  # Bohr magneton [J/T]
    B_T       = B_mT * 1e-3
    omega_q   = g_factor * mu_B * B_T / hbar   # rad/s
    f_q_GHz   = omega_q / (2 * math.pi) / 1e9

    # T1: フォノン緩和 (spin-orbit + valley-orbit)
    # Si スピン qubit はチャージ qubit より電荷ノイズ免疫が高い
    # 実測: ²⁸Si 濃縮で T1 ~ 10ms〜100ms (Veldhorst 2014, Takeda 2022)
    gamma_ph   = 1e3        # フォノン結合 [rad/s/K]
    gamma_ch   = 1e2 * n_imp_cm2 / 1e10   # スピン-電荷結合 (弱い)
    gamma_rms  = 1e2 * (rms_nm / 0.1) ** 2  # 界面粗さ起因 (valley mixing)
    T1_us      = 1e6 / (gamma_ph * T_K + gamma_ch + gamma_rms)

    # T2*: ²⁹Si 核スピンノイズ (天然 Si: 4.7% ²⁹Si)
    # 同位体精製 (²⁸Si > 99.9%) で大幅改善
    nat_Si29_frac  = 0.047
    T2star_us_nat  = 3.0 / nat_Si29_frac         # 天然 Si
    T2star_us_enr  = 3.0 / 0.001                  # 濃縮 Si (99.9% ²⁸Si)

    # T2: Hahn echo
    T2_us = min(2 * T1_us, T2star_us_enr)

    return {
        "f_q_GHz":        f_q_GHz,
        "T1_us":          T1_us,
        "T2_us":          T2_us,
        "T2star_nat_us":  T2star_us_nat,
        "T2star_enr_us":  T2star_us_enr,
        "B_mT":           B_mT,
        "T_mK":           T_mK,
        "rms_nm":         rms_nm,
        "n_imp_cm2":      n_imp_cm2,
    }


def si_qubit_sweep_rms(rms_range: np.ndarray) -> dict:
    """界面粗さ vs T1/T2 スイープ。"""
    T1s, T2s = [], []
    for r in rms_range:
        res = si_spin_qubit_coherence(r)
        T1s.append(res["T1_us"])
        T2s.append(res["T2_us"])
    return {"rms_nm": rms_range, "T1_us": np.array(T1s), "T2_us": np.array(T2s)}


# ════════════════════════════════════════════════════════════════════════════
# 2. 超伝導 Transmon qubit — Josephson 接合モデル
# ════════════════════════════════════════════════════════════════════════════

def transmon_qubit(EJ_GHz: float, EC_GHz: float) -> dict:
    """
    Transmon (charge-insensitive qubit) エネルギー準位。

    EJ: Josephson エネルギー [GHz × h]
    EC: チャージエネルギー = e²/(2C) [GHz × h]
    qubit 周波数: ω_q ≈ √(8 EJ EC) - EC  [GHz]
    アンハーモニシティ: α = E12 - E01 ≈ -EC  [GHz]

    設計指針:
        EJ/EC > 50 → 電荷ノイズ免疫 (transmon regime)
        |α| > 100 MHz → 個別制御可能
    """
    ratio = EJ_GHz / EC_GHz
    # plasmon 周波数
    f_01_GHz = math.sqrt(8 * EJ_GHz * EC_GHz) - EC_GHz
    # アンハーモニシティ (1次補正)
    alpha_GHz = -EC_GHz
    # Hamiltonian展開 2次補正
    delta_f_GHz = -EC_GHz / 4.0 * (1 / ratio)

    # T1: 誘電損失 + 放射損失
    Q_cap = 1e6    # キャパシタ品質因子
    T1_us = Q_cap / (2 * math.pi * f_01_GHz * 1e9) * 1e6

    # T2 ~ 2T1 (charge noise suppressed in transmon)
    T2_us = min(2 * T1_us, 300.0)  # 超伝導回路の現実的上限

    return {
        "EJ_GHz":      EJ_GHz,
        "EC_GHz":      EC_GHz,
        "EJ_EC_ratio": ratio,
        "f_01_GHz":    f_01_GHz,
        "alpha_MHz":   alpha_GHz * 1e3,   # GHz → MHz
        "T1_us":       T1_us,
        "T2_us":       T2_us,
        "charge_noise_immune": ratio > 50,
    }


def transmon_sweep_ratio(ratios: np.ndarray, EC_GHz: float = 0.25) -> dict:
    """EJ/EC 比 vs qubit 特性スイープ。"""
    f01s, alphas, T1s = [], [], []
    for r in ratios:
        res = transmon_qubit(r * EC_GHz, EC_GHz)
        f01s.append(res["f_01_GHz"])
        alphas.append(abs(res["alpha_MHz"]))
        T1s.append(res["T1_us"])
    return {
        "ratios":      ratios,
        "f01_GHz":     np.array(f01s),
        "alpha_MHz":   np.array(alphas),
        "T1_us":       np.array(T1s),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. 量子エラー訂正 — Surface Code スケーリング
# ════════════════════════════════════════════════════════════════════════════

def surface_code_overhead(p_phys: float, p_th: float = 0.01) -> dict:
    """
    Surface code: 物理 qubit エラー率 → 論理エラー率 → 必要物理 qubit 数。

    論理エラー率: p_L ≈ A × (p/p_th)^((d+1)/2)
    距離 d のコードブロック: n_phys = 2d² 物理 qubit

    Args:
        p_phys: 物理 qubit エラー率 (目標: < 0.1%)
        p_th:   Surface code 閾値 (~1%)
    """
    if p_phys >= p_th:
        return {"error": f"p_phys={p_phys} >= threshold={p_th}. QEC 不可能"}

    A = 0.1   # プリファクター
    results = []
    for d in range(3, 31, 2):  # 奇数距離 d = 3,5,...,29
        exp = (d + 1) / 2
        p_L = A * (p_phys / p_th) ** exp
        n_phys = 2 * d ** 2
        results.append({
            "d":       d,
            "p_L":     p_L,
            "n_phys":  n_phys,
        })

    # 実用的な論理エラー率目標: 1e-6 /回路
    target_pL = 1e-6
    feasible = [r for r in results if r["p_L"] < target_pL]
    min_d_for_target = feasible[0] if feasible else None

    return {
        "p_phys":           p_phys,
        "p_th":             p_th,
        "results":          results,
        "target_pL":        target_pL,
        "min_config":       min_d_for_target,
        "overhead_factor":  (min_d_for_target["n_phys"] if min_d_for_target else None),
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. 5〜10 年 量子コンピュータ ロードマップ
# ════════════════════════════════════════════════════════════════════════════

QUANTUM_ROADMAP = {
    # year: (物理qubit数, 論理qubit数, 物理エラー率%, メモ)
    2023: (1121,   0,    0.5,  "IBM Eagle 127Q / Google 53Q → NISQ初期"),
    2024: (3000,   0,    0.3,  "IBM Heron / Google Willow"),
    2025: (10000,  10,   0.1,  "Google Willow 論理qubit 実証"),
    2026: (50000,  100,  0.05, "IBM Flamingo / Microsoft トポロジカル"),
    2027: (200000, 1000, 0.02, "フォールトトレラント初期フェーズ"),
    2028: (500000, 5000, 0.01, "化学分子シミュレーション実用化"),
    2030: (1e6,    20000,0.005,"RSA-2048 解読可能な規模に近づく"),
    2033: (1e7,    100000,0.001,"大規模量子優位性 — 創薬・材料開発"),
}

SI_QUBIT_ROADMAP = {
    # year: (qubit/chip, T2_us, note)
    2024: (6,     100,   "Intel Tunnel Falls 6Q"),
    2025: (12,    300,   "濃縮 ²⁸Si + 300mm プロセス"),
    2026: (50,    500,   "IMEC/Intel 18A Si qubit"),
    2027: (200,   1000,  "Terafab 製 Si qubit チップ"),
    2028: (1000,  2000,  "QEC 実証 (d=5 surface code)"),
    2030: (10000, 5000,  "フォールトトレラント Si qubit"),
    2033: (100000,10000, "ユーティリティスケール"),
}


# ════════════════════════════════════════════════════════════════════════════
# 5. Si qubit × 300mm ウェーハ 集積密度
# ════════════════════════════════════════════════════════════════════════════

def si_qubit_wafer_integration(qubit_pitch_um: float = 100.0,
                                wafer_diam_mm: float = 300.0,
                                yield_pct: float = 90.0) -> dict:
    """
    300mm ウェーハ上の Si spin qubit 集積密度計算。

    qubit は通常 100µm ピッチ程度 (制御配線・フィルタ含む)。
    将来的に 10µm ピッチ (マルチプレクシング) を目指す。
    """
    wafer_area_mm2 = math.pi * (wafer_diam_mm / 2) ** 2 * 0.85  # 端部除く
    wafer_area_um2 = wafer_area_mm2 * 1e6
    pitch_um2      = qubit_pitch_um ** 2
    n_qubits_raw   = wafer_area_um2 / pitch_um2
    n_qubits_yield = n_qubits_raw * (yield_pct / 100)

    return {
        "pitch_um":       qubit_pitch_um,
        "wafer_area_mm2": wafer_area_mm2,
        "n_qubits_raw":   int(n_qubits_raw),
        "n_qubits_yield": int(n_qubits_yield),
        "density_per_mm2": n_qubits_raw / wafer_area_mm2,
    }


# ════════════════════════════════════════════════════════════════════════════
# 6. Plotting
# ════════════════════════════════════════════════════════════════════════════

def plot_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0a0a0a")
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.48, wspace=0.38,
                           left=0.07, right=0.97,
                           top=0.92, bottom=0.07)

    title_kw = dict(fontsize=9, color="#aaaaaa", pad=6)
    spine_c = "#333333"

    def style_ax(ax):
        ax.set_facecolor("#111111")
        for sp in ax.spines.values():
            sp.set_color(spine_c)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")

    # ── Panel 1: Si qubit T1/T2 vs 界面粗さ ───────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1)
    rms_range = np.linspace(0.05, 1.0, 100)
    sweep = si_qubit_sweep_rms(rms_range)
    ax1.semilogy(rms_range, sweep["T1_us"], color="#4393c3", lw=2, label="T₁")
    ax1.semilogy(rms_range, sweep["T2_us"], color="#d73027", lw=2, label="T₂")
    ax1.axvline(0.1, color="#74c476", ls="--", lw=1.2,
                label="優良界面 0.1nm rms")
    ax1.axhline(1000, color="#fdae61", ls=":", lw=1, label="QEC 目標 1ms")
    ax1.set_xlabel("界面粗さ RMS [nm]", fontsize=8)
    ax1.set_ylabel("コヒーレンス時間 T [µs]", fontsize=8)
    ax1.legend(fontsize=6, facecolor="#222222", labelcolor="white")
    ax1.set_title("Si スピン qubit — T₁/T₂ vs 界面粗さ", **title_kw)

    # ── Panel 2: EJ/EC 比 vs Transmon 特性 ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2)
    ratios = np.linspace(10, 150, 200)
    ts = transmon_sweep_ratio(ratios)
    ax2_r = ax2.twinx()
    ax2_r.set_facecolor("#111111")
    ax2.plot(ratios, ts["f01_GHz"], color="#4393c3", lw=2, label="f₀₁ [GHz]")
    ax2_r.plot(ratios, ts["alpha_MHz"], color="#d73027", lw=2,
               linestyle="--", label="|α| [MHz]")
    ax2.axvline(50, color="#74c476", ls=":", lw=1.5, label="EJ/EC=50 閾値")
    ax2.set_xlabel("EJ / EC", fontsize=8)
    ax2.set_ylabel("qubit 周波数 f₀₁ [GHz]", fontsize=8, color="#4393c3")
    ax2_r.set_ylabel("|α| アンハーモニシティ [MHz]", fontsize=8, color="#d73027")
    ax2_r.tick_params(colors="#aaaaaa", labelsize=7)
    lines1, lbls1 = ax2.get_legend_handles_labels()
    lines2, lbls2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbls1 + lbls2,
               fontsize=6, facecolor="#222222", labelcolor="white")
    ax2.set_title("Transmon qubit — EJ/EC vs 周波数・アンハーモニシティ", **title_kw)

    # ── Panel 3: QEC Surface Code オーバーヘッド ──────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3)
    p_values = [0.001, 0.003, 0.005, 0.008]
    colors3  = ["#4393c3", "#74add1", "#fdae61", "#d73027"]
    for p, col in zip(p_values, colors3):
        qec = surface_code_overhead(p)
        if "error" in qec:
            continue
        ds   = [r["d"]      for r in qec["results"]]
        pLs  = [r["p_L"]    for r in qec["results"]]
        ax3.semilogy(ds, pLs, "o-", color=col, ms=4,
                     label=f"p={p*100:.1f}%", lw=1.5)
    ax3.axhline(1e-6, color="#74c476", ls="--", lw=1.2, label="目標 10⁻⁶")
    ax3.set_xlabel("Surface Code 距離 d", fontsize=8)
    ax3.set_ylabel("論理エラー率 p_L", fontsize=8)
    ax3.legend(fontsize=6, facecolor="#222222", labelcolor="white")
    ax3.set_title("Surface Code QEC — 距離 d vs 論理エラー率", **title_kw)

    # ── Panel 4: 量子コンピュータ ロードマップ (物理 qubit) ─────────
    ax4 = fig.add_subplot(gs[1, :2])
    style_ax(ax4)
    years    = list(QUANTUM_ROADMAP.keys())
    phys_q   = [v[0] for v in QUANTUM_ROADMAP.values()]
    logic_q  = [v[1] for v in QUANTUM_ROADMAP.values()]
    err_pct  = [v[2] for v in QUANTUM_ROADMAP.values()]
    notes    = [v[3] for v in QUANTUM_ROADMAP.values()]
    ax4.semilogy(years, phys_q, "o-", color="#4393c3", lw=2.5, ms=7,
                 label="物理 qubit 数")
    ax4.semilogy(years, [max(l, 0.5) for l in logic_q], "s--",
                 color="#d73027", lw=2, ms=7, label="論理 qubit 数")
    ax4_r = ax4.twinx()
    ax4_r.set_facecolor("#111111")
    ax4_r.plot(years, err_pct, "^:", color="#74c476", lw=1.5, ms=6,
               label="物理エラー率 [%]")
    ax4_r.set_ylabel("エラー率 [%]", fontsize=8, color="#74c476")
    ax4_r.tick_params(colors="#aaaaaa", labelsize=7)
    for yr, pq, note in zip(years, phys_q, notes):
        if yr in [2025, 2028, 2033]:
            ax4.annotate(note.split("→")[-1].strip(),
                         xy=(yr, pq), xytext=(yr, pq*3),
                         fontsize=5, color="#aaaaaa",
                         arrowprops=dict(arrowstyle="-", color="#555555"))
    ax4.axvline(2026, color="#fdae61", ls=":", lw=1.5, label="現在 (2026)")
    ax4.set_xlabel("年", fontsize=8)
    ax4.set_ylabel("qubit 数", fontsize=8)
    lines1, l1 = ax4.get_legend_handles_labels()
    lines2, l2 = ax4_r.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, l1 + l2,
               fontsize=6, facecolor="#222222", labelcolor="white", loc="upper left")
    ax4.set_title("量子コンピュータ ロードマップ 2023〜2033", **title_kw)

    # ── Panel 5: Si qubit ロードマップ (Terafab との交点) ───────────
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5)
    si_years = list(SI_QUBIT_ROADMAP.keys())
    si_nq    = [v[0] for v in SI_QUBIT_ROADMAP.values()]
    si_T2    = [v[1] for v in SI_QUBIT_ROADMAP.values()]
    ax5.semilogy(si_years, si_nq, "o-", color="#4393c3", lw=2, ms=7,
                 label="qubit/chip")
    ax5_r = ax5.twinx()
    ax5_r.set_facecolor("#111111")
    ax5_r.semilogy(si_years, si_T2, "s--", color="#d73027", lw=2, ms=6,
                   label="T₂ [µs]")
    ax5_r.set_ylabel("T₂ [µs]", fontsize=8, color="#d73027")
    ax5_r.tick_params(colors="#aaaaaa", labelsize=7)
    ax5.axvline(2027, color="#fdae61", ls=":", lw=1.5, label="Terafab HVM")
    ax5.set_xlabel("年", fontsize=8)
    ax5.set_ylabel("qubit/chip", fontsize=8)
    lines1, l1 = ax5.get_legend_handles_labels()
    lines2, l2 = ax5_r.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, l1 + l2,
               fontsize=6, facecolor="#222222", labelcolor="white")
    ax5.set_title("Si Spin Qubit × Terafab ロードマップ", **title_kw)

    # ── Panel 6: 300mm ウェーハ qubit 集積密度 ────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    style_ax(ax6)
    pitches = [10, 20, 50, 100, 200]
    qubit_counts = [si_qubit_wafer_integration(p)["n_qubits_yield"]
                    for p in pitches]
    cols6 = plt.cm.Blues(np.linspace(0.4, 0.9, len(pitches)))
    bars6 = ax6.bar([f"{p}µm\nピッチ" for p in pitches],
                    [q/1000 for q in qubit_counts], color=cols6, width=0.6)
    for bar, q in zip(bars6, qubit_counts):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{q/1e3:.0f}k" if q > 1000 else str(q),
                 ha="center", fontsize=7, color="white")
    ax6.set_ylabel("qubit 数 / 300mm ウェーハ [千]", fontsize=8)
    ax6.set_title("300mm ウェーハ Si Qubit 集積密度", **title_kw)

    # ── Panel 7: QEC 必要物理 qubit 数 ────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    style_ax(ax7)
    p_values7 = np.array([0.001, 0.002, 0.003, 0.005, 0.008])
    n_phys_list = []
    for p in p_values7:
        qec = surface_code_overhead(p)
        mc = qec.get("min_config")
        n_phys_list.append(mc["n_phys"] if mc else 9999)
    bars7 = ax7.bar([f"{p*100:.1f}%" for p in p_values7], n_phys_list,
                    color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(p_values7))),
                    width=0.6)
    for bar, n in zip(bars7, n_phys_list):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(n), ha="center", fontsize=7, color="white")
    ax7.set_xlabel("物理エラー率 [%]", fontsize=8)
    ax7.set_ylabel("1 論理 qubit あたり物理 qubit 数", fontsize=8)
    ax7.set_title("QEC: 物理エラー率 → 必要物理 qubit 数", **title_kw)

    # ── Panel 8: 技術比較マトリックス ─────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    style_ax(ax8)
    methods = ["Si\nスピン", "超伝導\nTransmon", "イオン\nトラップ", "光子\n(フォトニック)",
               "トポロジカル\n(MS)"]
    # [T2 スコア, 集積性, 動作温度スコア(低い方が良い → 反転), エラー率スコア]
    scores = {
        "T₂コヒーレンス":  [0.8, 0.6, 1.0, 0.5, 0.9],
        "集積密度":        [0.9, 0.7, 0.2, 0.6, 0.5],
        "動作温度 (高い=良)": [0.3, 0.2, 0.8, 1.0, 0.2],
        "Terafab 親和性":  [1.0, 0.6, 0.1, 0.4, 0.7],
    }
    categories = list(scores.keys())
    x = np.arange(len(methods))
    width = 0.18
    colors8 = ["#2166ac", "#4393c3", "#74add1", "#abd9e9"]
    for i, (cat, vals) in enumerate(scores.items()):
        offset = (i - 1.5) * width
        ax8.bar(x + offset, vals, width, label=cat, color=colors8[i], alpha=0.85)
    ax8.set_xticks(x)
    ax8.set_xticklabels(methods, fontsize=6)
    ax8.set_ylabel("スコア (0〜1)", fontsize=8)
    ax8.set_ylim(0, 1.2)
    ax8.legend(fontsize=5, facecolor="#222222", labelcolor="white", loc="upper right")
    ax8.set_title("量子 qubit 方式 比較 (Terafab視点)", **title_kw)

    fig.suptitle(
        "量子コンピュータ プロセス物理モデル — Si Qubit × Terafab × 5〜10年ロードマップ",
        fontsize=13, color="white", fontweight="bold", y=0.97,
    )

    out = os.path.join(OUT_DIR, "quantum_computing_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → {out}")
    return out


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _print_summary():
    si = si_spin_qubit_coherence(rms_nm=0.1, n_imp_cm2=1e10, B_mT=1.0)
    tm = transmon_qubit(EJ_GHz=25.0, EC_GHz=0.25)
    qec = surface_code_overhead(0.001)
    integ = si_qubit_wafer_integration(qubit_pitch_um=50.0)

    print("\n=== Si スピン qubit ===")
    print(f"  qubit 周波数:   {si['f_q_GHz']:.3f} GHz")
    print(f"  T1:             {si['T1_us']:.0f} µs")
    print(f"  T2 (²⁸Si):     {si['T2_us']:.0f} µs")

    print("\n=== 超伝導 Transmon ===")
    print(f"  f₀₁:            {tm['f_01_GHz']:.2f} GHz")
    print(f"  アンハーモニシティ: {tm['alpha_MHz']:.0f} MHz")
    print(f"  T1:             {tm['T1_us']:.0f} µs")

    print("\n=== QEC Surface Code (p=0.1%) ===")
    if qec.get("min_config"):
        mc = qec["min_config"]
        print(f"  最小距離 d:     {mc['d']}")
        print(f"  必要物理 qubit: {mc['n_phys']} / 論理 qubit")
        print(f"  論理エラー率:   {mc['p_L']:.2e}")

    print("\n=== 300mm ウェーハ集積 (50µm ピッチ) ===")
    print(f"  理論 qubit 数:  {integ['n_qubits_raw']:,}")
    print(f"  実効 qubit 数:  {integ['n_qubits_yield']:,} (歩留まり 90%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="量子コンピュータ プロセスモデル")
    parser.add_argument("--spin",    action="store_true", help="Si spin qubit")
    parser.add_argument("--transmon",action="store_true", help="超伝導 transmon")
    parser.add_argument("--qec",     action="store_true", help="QEC スケーリング")
    parser.add_argument("--roadmap", action="store_true", help="ロードマップ")
    args = parser.parse_args()

    _print_summary()
    out = plot_all()
    print(f"\n全パネル出力: {out}")
