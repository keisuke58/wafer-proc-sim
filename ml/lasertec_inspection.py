"""
Lasertec Inspection Model — ダイシング後 in-line 欠陥検査
=========================================================
Lasertec SICA88 / MAGICS M7360 / DCI 相当の検査シミュレーション。

検査モード:
1. 共焦点レーザー走査 — 表面粗さ・チッピング寸法マッピング
2. PL (Photoluminescence) マッピング — V_Si/V_C 欠陥密度の空間分布
3. OCT 相当 深さ解析 — サブサーフェスクラック深さ
4. 欠陥分類 (Grade A/B/C/D) → Disco/TEL へのフィードバック

実行:
    python ml/lasertec_inspection.py
    python ml/lasertec_inspection.py --confocal
    python ml/lasertec_inspection.py --pl-map
    python ml/lasertec_inspection.py --grading

参考文献:
    Lasertec SICA88 spec (公開情報)
    Fujiwara et al. (2020) Jpn J Appl Phys — SiC surface inspection
    Kimoto & Cooper (2014) — SiC defect characterization
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── 検査仕様 ────────────────────────────────────────────────────────────────
INSPECTION_SPEC = {
    "Ra_nm_max":    50.0,   # 表面粗さ上限 [nm]
    "chip_um_max":   5.0,   # チッピング上限 [µm]  (2nm ノード: 0.5µm)
    "crack_um_max":  2.0,   # サブサーフェスクラック上限 [µm]
    "defect_density_max": 1e12,  # PL 検査で検出する欠陥密度上限 [cm⁻³]
}

# プロセス別の期待値 (quantum_defect_model 結果と整合)
PROCESS_PARAMS = {
    "blade":     {"HAZ": 2.5, "Ra": 80.0, "chip": 8.0, "crack": 3.0},
    "laser_ns":  {"HAZ": 2.0, "Ra": 60.0, "chip": 4.0, "crack": 2.0},
    "laser_ps":  {"HAZ": 0.5, "Ra": 30.0, "chip": 1.5, "crack": 0.5},
    "laser_fs":  {"HAZ": 0.2, "Ra": 15.0, "chip": 0.8, "crack": 0.2},
    "stealth":   {"HAZ": 0.05,"Ra":  5.0, "chip": 0.3, "crack": 0.05},
    "plasma":    {"HAZ": 0.8, "Ra": 20.0, "chip": 0.5, "crack": 0.3},
}


# ════════════════════════════════════════════════════════════════════════════
# 1. 共焦点レーザー走査
# ════════════════════════════════════════════════════════════════════════════

def confocal_scan(process: str, n_points: int = 200,
                   seed: int = 0) -> dict:
    """
    表面高さプロファイル h(x) をシミュレート。
    Ra, Rz, Rq を算出して SPEC 判定。
    """
    rng = np.random.default_rng(seed)
    p   = PROCESS_PARAMS.get(process, PROCESS_PARAMS["blade"])
    Ra  = p["Ra"]

    # フラクタル表面モデル (Hurst 指数 H ≈ 0.8 for polished SiC)
    H     = 0.8
    N     = n_points
    freqs = np.fft.rfftfreq(N)
    freqs[0] = 1e-10
    psd   = freqs ** (-(2*H + 1))
    phase = rng.uniform(0, 2*np.pi, len(freqs))
    spectrum = np.sqrt(psd) * np.exp(1j*phase)
    h    = np.fft.irfft(spectrum, N).real
    h    = h / (h.std() + 1e-15) * Ra / math.sqrt(2)   # Ra ≈ σ/√2

    Ra_meas = np.mean(np.abs(h - h.mean()))
    Rq_meas = np.sqrt(np.mean((h - h.mean())**2))
    Rz_meas = h.max() - h.min()

    pass_Ra = Ra_meas < INSPECTION_SPEC["Ra_nm_max"]

    return {
        "process":   process,
        "x_um":      np.linspace(0, 100, N),
        "h_nm":      h,
        "Ra_nm":     round(Ra_meas, 2),
        "Rq_nm":     round(Rq_meas, 2),
        "Rz_nm":     round(Rz_meas, 2),
        "pass_Ra":   pass_Ra,
    }


def chipping_measurement(process: str, n_sites: int = 50,
                          seed: int = 0) -> dict:
    """
    ダイエッジのチッピング寸法計測 (n_sites 箇所)。
    Weibull 分布でチッピングサイズをモデル化。
    """
    rng = np.random.default_rng(seed)
    p   = PROCESS_PARAMS.get(process, PROCESS_PARAMS["blade"])
    m_W = 2.5   # Weibull 形状
    eta = p["chip"] / (math.log(1/(1-0.5))) ** (1/m_W)   # 中央値から η を推定
    chips = eta * (-np.log(1 - rng.uniform(0.01, 0.99, n_sites))) ** (1/m_W)

    pass_chip = chips.max() < INSPECTION_SPEC["chip_um_max"]
    return {
        "chips_um":   chips,
        "max_um":     round(chips.max(), 3),
        "mean_um":    round(chips.mean(), 3),
        "p99_um":     round(np.percentile(chips, 99), 3),
        "pass":       pass_chip,
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. PL マッピング
# ════════════════════════════════════════════════════════════════════════════

def pl_map(process: str, grid_n: int = 40, seed: int = 0) -> dict:
    """
    PL (Photoluminescence) 強度マップ。
    V_Si の ZPL (1352nm) 強度 ∝ V_Si 密度。
    プロセス誘起 V_Si がエッジに集中する傾向を空間モデル化。
    """
    from ml.quantum_defect_model import process_induced_trap_density
    rng = np.random.default_rng(seed)
    p   = PROCESS_PARAMS[process]

    xs = np.linspace(0, 500, grid_n)   # µm (ダイ幅)
    ys = np.linspace(0, 500, grid_n)
    X, Y = np.meshgrid(xs, ys)

    # エッジからの距離 (ダイエッジ効果)
    edge_dist = np.minimum(np.minimum(X, 500-X), np.minimum(Y, 500-Y))
    edge_factor = np.exp(-edge_dist / (p["HAZ"] * 50 + 1))

    base_dens = process_induced_trap_density(p["HAZ"], p["Ra"], process)
    N_VSi = base_dens["V_Si (spin qubit)"]
    noise = rng.lognormal(0, 0.3, (grid_n, grid_n))
    density_map = N_VSi * (1 + 3 * edge_factor) * noise

    # PL 強度 ∝ N_VSi (ただし自己吸収で高密度では飽和)
    pl_intensity = density_map / (density_map + N_VSi) * 1000   # arb. units

    return {
        "X": X, "Y": Y,
        "density_map": density_map,
        "pl_intensity": pl_intensity,
        "N_VSi_center": float(density_map[grid_n//2, grid_n//2]),
        "N_VSi_edge":   float(density_map[0, grid_n//2]),
        "edge_ratio":   float(density_map[0, grid_n//2] / density_map[grid_n//2, grid_n//2]),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. 欠陥グレーディング
# ════════════════════════════════════════════════════════════════════════════

def grade_die(process: str, seed: int = 0) -> dict:
    """
    Lasertec の総合グレード判定:
    Grade A: 全項目 SPEC 内
    Grade B: チッピング軽微逸脱 → 追加プロセスで回復可
    Grade C: HAZ/欠陥密度逸脱 → デバイス性能低下
    Grade D: 致命的欠陥 → 廃棄
    """
    confocal = confocal_scan(process, seed=seed)
    chipping = chipping_measurement(process, seed=seed)
    pl       = pl_map(process, seed=seed)
    p        = PROCESS_PARAMS[process]

    scores = {
        "Ra":     confocal["Ra_nm"] / INSPECTION_SPEC["Ra_nm_max"],
        "chip":   chipping["mean_um"] / INSPECTION_SPEC["chip_um_max"],
        "crack":  p["crack"] / INSPECTION_SPEC["crack_um_max"],
        "defect": pl["N_VSi_center"] / INSPECTION_SPEC["defect_density_max"],
    }
    max_score = max(scores.values())

    if max_score < 0.5:
        grade = "A"
    elif max_score < 1.0:
        grade = "B"
    elif max_score < 2.0:
        grade = "C"
    else:
        grade = "D"

    return {"grade": grade, "scores": scores,
            "max_score": round(max_score, 3), "process": process}


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    procs  = ["blade", "laser_ps", "laser_fs", "stealth"]
    labels = {"blade":"ブレード","laser_ps":"レーザーps",
              "laser_fs":"レーザーfs","stealth":"ステルス"}
    colors = {"blade":"#d62728","laser_ps":"#ff7f0e",
              "laser_fs":"#2ca02c","stealth":"#2166ac"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # ── 表面プロファイル ─────────────────────────────────────────────
    ax = axes[0, 0]
    for proc in procs:
        r = confocal_scan(proc)
        ax.plot(r["x_um"], r["h_nm"], color=colors[proc], lw=1.0,
                alpha=0.8, label=f"{labels[proc]} Ra={r['Ra_nm']:.0f}nm")
    ax.axhline(INSPECTION_SPEC["Ra_nm_max"], color="red", ls="--",
               lw=1.5, label="Ra 上限")
    ax.set_xlabel("Position [µm]", fontsize=10)
    ax.set_ylabel("Height [nm]", fontsize=10)
    ax.set_title("共焦点レーザー 表面プロファイル", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # ── チッピング Weibull ────────────────────────────────────────────
    ax2 = axes[0, 1]
    for proc in procs:
        r = chipping_measurement(proc, n_sites=200)
        chips_sorted = np.sort(r["chips_um"])
        F = np.arange(1, len(chips_sorted)+1) / (len(chips_sorted)+1)
        ax2.semilogy(chips_sorted, 1-F, color=colors[proc], lw=2,
                     label=f"{labels[proc]} max={r['max_um']:.1f}µm")
    ax2.axvline(INSPECTION_SPEC["chip_um_max"], color="red", ls="--",
                lw=1.5, label="上限")
    ax2.axvline(0.5, color="purple", ls=":", lw=1.2, label="2nm目標0.5µm")
    ax2.set_xlabel("Chipping Size [µm]", fontsize=10)
    ax2.set_ylabel("Exceedance Prob.", fontsize=10)
    ax2.set_title("チッピング Weibull 分布", fontsize=10)
    ax2.legend(fontsize=7); ax2.grid(alpha=0.2, which="both")

    # ── PL マップ (ステルス vs ブレード) ─────────────────────────────
    for ax_pl, proc in [(axes[0, 2], "stealth"), (axes[1, 0], "blade")]:
        r = pl_map(proc)
        im = ax_pl.contourf(r["X"], r["Y"], np.log10(r["density_map"]+1),
                            levels=20, cmap="hot_r")
        plt.colorbar(im, ax=ax_pl, label="log₁₀(N_V_Si [cm⁻³])")
        ax_pl.set_title(f"PL マップ: {labels[proc]}\n"
                        f"中心 {r['N_VSi_center']:.1e} cm⁻³  "
                        f"エッジ比 ×{r['edge_ratio']:.1f}", fontsize=9)
        ax_pl.set_xlabel("x [µm]", fontsize=9)
        ax_pl.set_ylabel("y [µm]", fontsize=9)

    # ── グレード比較 ─────────────────────────────────────────────────
    ax5 = axes[1, 1]
    grade_map = {"A": 4, "B": 3, "C": 2, "D": 1}
    grade_colors = {"A":"#2ca02c","B":"#1f77b4","C":"#ff7f0e","D":"#d62728"}
    for k, proc in enumerate(procs):
        r = grade_die(proc)
        g = r["grade"]
        ax5.bar(labels[proc], grade_map[g], color=grade_colors[g],
                alpha=0.85, edgecolor="k", lw=0.8)
        ax5.text(k, grade_map[g]+0.05, f"Grade {g}", ha="center",
                 fontsize=11, fontweight="bold")
    ax5.set_yticks([1,2,3,4]); ax5.set_yticklabels(["D","C","B","A"])
    ax5.set_ylabel("Lasertec Grade", fontsize=10)
    ax5.set_title("総合グレード判定", fontsize=10)
    ax5.grid(alpha=0.2, axis="y")

    # ── スコア spider ─────────────────────────────────────────────────
    ax6 = axes[1, 2]
    items = ["Ra", "chip", "crack", "defect"]
    for proc in procs:
        r = grade_die(proc)
        vals = [r["scores"][i] for i in items]
        vals += [vals[0]]
        angles = [n/len(items)*2*math.pi for n in range(len(items))]
        angles += [angles[0]]
        ax6.plot(angles, vals, color=colors[proc], lw=2,
                 label=labels[proc])
        ax6.fill(angles, vals, color=colors[proc], alpha=0.06)
    ax6.axhline(1.0, color="red", ls="--", lw=1.2, alpha=0.7)
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(["Ra", "チップ", "クラック", "欠陥密度"], fontsize=9)
    ax6.set_title("検査スコア (1=上限)", fontsize=10)
    ax6.legend(fontsize=7); ax6.grid(alpha=0.2)

    fig.suptitle("Lasertec 検査シミュレーション — ダイシング後 in-line 品質評価\n"
                 "共焦点 / PL マッピング / グレーディング", fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "lasertec_inspection.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print("\n  プロセス別 Lasertec グレード:")
    print(f"  {'プロセス':>12}  {'Grade':>6}  {'Ra[nm]':>8}  "
          f"{'Chip[µm]':>10}  {'Crack[µm]':>10}")
    print("  " + "─"*55)
    for proc in list(PROCESS_PARAMS.keys()):
        c  = confocal_scan(proc)
        ch = chipping_measurement(proc)
        g  = grade_die(proc)
        pp = PROCESS_PARAMS[proc]
        print(f"  {proc:>12}  {g['grade']:>6}  {c['Ra_nm']:>8.1f}  "
              f"{ch['mean_um']:>10.3f}  {pp['crack']:>10.2f}")
    print(f"[✓] Lasertec inspection -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confocal", action="store_true")
    ap.add_argument("--pl-map",   action="store_true")
    ap.add_argument("--grading",  action="store_true")
    args = ap.parse_args()
    plot_all()


if __name__ == "__main__":
    main()
