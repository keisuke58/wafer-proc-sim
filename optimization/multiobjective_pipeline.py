"""
マルチ目的最適化 — 完全パイプライン統合
==========================================
NSGA-II (Non-dominated Sorting Genetic Algorithm II) で
Disco → TEL → Advantest パイプライン全体を最適化。

決定変数:
    x[0] HAZ_um       [0.05, 3.0]   ダイシング深度損傷 (プロセス選択に対応)
    x[1] Ra_nm        [5,   100]    表面粗さ
    x[2] ox_T_C       [950, 1250]   熱酸化温度 [°C]
    x[3] anneal_T_C   [900, 1050]   NO アニール温度 [°C]
    x[4] ald_cycles   [30,  150]    ALD サイクル数 (膜厚)
    x[5] pressure_kPa [20,   60]    CMP 圧力

目的関数 (全て最小化 → 符号反転で最大化):
    f1 = -τ_SRH       キャリア寿命を最大化
    f2 = -µ_ch        チャンネル移動度を最大化
    f3 = -yield_pct   ATE 歩留まりを最大化
    f4 = CPGD         コストを最小化
    f5 = chip_mean    チッピングを最小化
    f6 = R_on         オン抵抗を最小化

実行:
    python optimization/multiobjective_pipeline.py
    python optimization/multiobjective_pipeline.py --pop 100 --gen 50
    python optimization/multiobjective_pipeline.py --pareto2d  # 2D Pareto プロット
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT_DIR = os.path.join(ROOT, "results")


# ════════════════════════════════════════════════════════════════════════════
# 目的関数 — パイプライン評価
# ════════════════════════════════════════════════════════════════════════════

def pipeline_objectives(x: np.ndarray) -> np.ndarray:
    """
    決定変数 x → 目的関数ベクトル f (全て最小化方向)。
    パイプラインの主要ステップを呼び出す。
    """
    HAZ_um      = float(np.clip(x[0], 0.05, 3.0))
    Ra_nm       = float(np.clip(x[1], 5.0,  100.0))
    ox_T_C      = float(np.clip(x[2], 950,  1250))
    anneal_T_C  = float(np.clip(x[3], 900,  1050))
    ald_cycles  = int(np.clip(x[4],   30,   150))
    pressure    = float(np.clip(x[5], 20,    60))

    # ── Step 1: 欠陥密度 → キャリア寿命 ────────────────────────────────
    from ml.quantum_defect_model import (process_induced_trap_density,
                                          shockley_read_hall_lifetime)
    # HAZ → プロセス推定（連続近似）
    proc = ("stealth" if HAZ_um < 0.1 else
            "laser_fs" if HAZ_um < 0.3 else
            "laser_ps" if HAZ_um < 0.8 else
            "laser_ns" if HAZ_um < 1.5 else "blade")
    traps = process_induced_trap_density(HAZ_um, Ra_nm, proc)
    tau   = shockley_read_hall_lifetime(traps["V_C (Z1/2)"])

    # ── Step 2: TEL ALD + 熱酸化 ───────────────────────────────────────
    from fem.tel_process_model import (ald_film, deal_grove, dit_from_oxidation,
                                        channel_mobility, mosfet_performance)
    ald = ald_film(ald_cycles, 250.0, "HfO2")
    ox  = deal_grove(2.0, ox_T_C, "4H-SiC")
    Dit_ox = dit_from_oxidation(ox_T_C, 2.0, anneal_T_C=anneal_T_C)
    Dit_dicing = 1e11 * (HAZ_um / 0.1) ** 0.5
    Dit_ald = (Dit_ox + Dit_dicing) * 0.2

    mu_ch = channel_mobility(Dit_ald, ald["Cox_fF_um2"])

    eV = 1.602e-19; eps0 = 8.854e-12
    EOT_nm   = ox["x_ox_nm"] + ald["EOT_nm"]
    Cox_Fcm2 = 3.9 * eps0 / max(EOT_nm, 0.1) * 1e9 * 1e-4
    N_it  = Dit_ald * 0.1
    V_th  = 3.0 + eV * N_it / max(Cox_Fcm2, 1e-9)
    perf  = mosfet_performance(mu_ch, ald["Cox_fF_um2"], V_th)

    # ── Step 3: CMP コスト ─────────────────────────────────────────────
    from fem.tel_cmp_lasertec import cmp_time
    cmp = cmp_time(50, pressure, 0.5, "4H-SiC")

    # ── Step 4: Lasertec 後検査 → チッピング ────────────────────────────
    from ml.lasertec_inspection import chipping_measurement
    chip = chipping_measurement(proc, n_sites=50, seed=42)

    # ── Step 5: Advantest ATE ──────────────────────────────────────────
    from fem.advantest_model import wafer_map, test_economics
    wm   = wafer_map(mu_ch, Dit_ald, seed=0)
    econ = test_economics(wm["yield_pct"], wm["n_die"])

    # ── 目的関数ベクトル (全て最小化) ──────────────────────────────────
    f = np.array([
        -tau,                      # f1: 寿命を最大化
        -mu_ch,                    # f2: 移動度を最大化
        -wm["yield_pct"],          # f3: 歩留まりを最大化
        econ["cpgd_full_usd"],     # f4: CPGD を最小化
        chip["mean_um"],           # f5: チッピングを最小化
        perf["R_on_mohm_mm2"],     # f6: R_on を最小化
    ])
    return f


# ════════════════════════════════════════════════════════════════════════════
# NSGA-II 実装
# ════════════════════════════════════════════════════════════════════════════

def dominates(f_a: np.ndarray, f_b: np.ndarray) -> bool:
    """f_a が f_b を支配: 全目的で a ≤ b かつ少なくとも1つで a < b。"""
    return bool(np.all(f_a <= f_b) and np.any(f_a < f_b))


def fast_nondominated_sort(F: np.ndarray) -> list:
    """高速非支配ソート → フロントのリストを返す。"""
    n = len(F)
    dom_count = np.zeros(n, dtype=int)
    dominated_by = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i+1, n):
            if dominates(F[i], F[j]):
                dominated_by[i].append(j)
                dom_count[j] += 1
            elif dominates(F[j], F[i]):
                dominated_by[j].append(i)
                dom_count[i] += 1
        if dom_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)
    return fronts[:-1]


def crowding_distance(F: np.ndarray, front: list) -> np.ndarray:
    """混雑距離の計算。"""
    l   = len(front)
    dist = np.zeros(l)
    if l <= 2:
        dist[:] = np.inf
        return dist
    F_f = F[front]
    for m in range(F_f.shape[1]):
        idx    = np.argsort(F_f[:, m])
        fmin, fmax = F_f[idx[0], m], F_f[idx[-1], m]
        dist[idx[0]] = dist[idx[-1]] = np.inf
        if fmax - fmin < 1e-15:
            continue
        for i in range(1, l-1):
            dist[idx[i]] += (F_f[idx[i+1], m] - F_f[idx[i-1], m]) / (fmax - fmin)
    return dist


def tournament_select(pop: np.ndarray, F: np.ndarray,
                       fronts: list, crowd: dict, rng) -> int:
    """トーナメント選択: ランク < or 混雑距離 > で優劣判定。"""
    rank = np.zeros(len(pop), dtype=int)
    for r, front in enumerate(fronts):
        for i in front:
            rank[i] = r
    a, b = rng.integers(0, len(pop), 2)
    if rank[a] < rank[b]:
        return int(a)
    elif rank[b] < rank[a]:
        return int(b)
    else:
        ca = crowd.get(int(a), 0)
        cb = crowd.get(int(b), 0)
        return int(a) if ca >= cb else int(b)


def nsga2(n_var: int = 6, n_obj: int = 6,
          bounds: list = None,
          pop_size: int = 60, n_gen: int = 30,
          seed: int = 42,
          log_interval: int = 5) -> dict:
    """
    NSGA-II メインループ。

    Returns:
        pareto_X: Pareto フロントの決定変数
        pareto_F: Pareto フロントの目的関数値
        history:  世代ごとの Pareto フロント数
    """
    rng = np.random.default_rng(seed)

    if bounds is None:
        bounds = [(0.05, 3.0), (5.0, 100.0), (950, 1250),
                  (900, 1050), (30, 150), (20, 60)]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # ── 初期集団 ─────────────────────────────────────────────────────
    pop = rng.uniform(lo, hi, (pop_size, n_var))
    print(f"  初期集団 {pop_size} 個を評価中...")
    F   = np.array([pipeline_objectives(x) for x in pop])

    history = []

    for gen in range(n_gen):
        t0 = time.time()

        # ── 非支配ソート + 混雑距離 ──────────────────────────────────
        fronts  = fast_nondominated_sort(F)
        crowd_d = {}
        for front in fronts:
            cd = crowding_distance(F, front)
            for i, idx in enumerate(front):
                crowd_d[idx] = cd[i]

        # ── 子孫生成 (SBX + polynomial mutation) ─────────────────────
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_select(pop, F, fronts, crowd_d, rng)
            p2 = tournament_select(pop, F, fronts, crowd_d, rng)
            # SBX crossover η_c=20
            eta_c = 20.0
            u     = rng.uniform(0, 1, n_var)
            beta  = np.where(u < 0.5,
                             (2*u)**(1/(eta_c+1)),
                             (1/(2*(1-u)))**(1/(eta_c+1)))
            child = 0.5 * ((1+beta)*pop[p1] + (1-beta)*pop[p2])
            # Polynomial mutation η_m=20
            eta_m = 20.0
            delta = hi - lo
            u_m   = rng.uniform(0, 1, n_var)
            delta_q = np.where(u_m < 0.5,
                                (2*u_m)**(1/(eta_m+1)) - 1,
                                1 - (2*(1-u_m))**(1/(eta_m+1)))
            child += delta_q * delta * (rng.random(n_var) < 1/n_var)
            child  = np.clip(child, lo, hi)
            offspring.append(child)

        offspring = np.array(offspring[:pop_size])
        F_off = np.array([pipeline_objectives(x) for x in offspring])

        # ── エリート選択 (次世代集団) ─────────────────────────────────
        pop_comb = np.vstack([pop, offspring])
        F_comb   = np.vstack([F, F_off])
        fronts_c = fast_nondominated_sort(F_comb)
        crowd_c  = {}
        for front in fronts_c:
            cd = crowding_distance(F_comb, front)
            for i, idx in enumerate(front):
                crowd_c[idx] = cd[i]

        selected = []
        for front in fronts_c:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                needed = pop_size - len(selected)
                sorted_front = sorted(front, key=lambda i: -crowd_c.get(i, 0))
                selected.extend(sorted_front[:needed])
                break

        pop = pop_comb[selected]
        F   = F_comb[selected]
        dt  = time.time() - t0

        pareto_size = len(fronts_c[0]) if fronts_c else 0
        history.append(pareto_size)
        if (gen+1) % log_interval == 0 or gen == n_gen-1:
            print(f"  Gen {gen+1:3d}/{n_gen}  "
                  f"Pareto={pareto_size:3d}  "
                  f"({dt:.1f}s)")

    # Pareto フロントのみ返す
    final_fronts = fast_nondominated_sort(F)
    pareto_idx   = final_fronts[0]
    return {
        "pareto_X":   pop[pareto_idx],
        "pareto_F":   F[pareto_idx],
        "history":    history,
        "pop":        pop,
        "F":          F,
    }


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

OBJ_LABELS = [
    "キャリア寿命 τ [µs]",
    "移動度 µ_ch [cm²/Vs]",
    "Yield [%]",
    "CPGD [$]",
    "チッピング [µm]",
    "R_on [mΩ·mm²]",
]
OBJ_SIGN = [-1, -1, -1, 1, 1, 1]   # -1: 最大化方向を元に戻す


def plot_results(result: dict, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    pF   = result["pareto_F"].copy()
    pX   = result["pareto_X"].copy()
    allF = result["F"].copy()

    # 符号を元に戻す (最大化目標は正に)
    for j, s in enumerate(OBJ_SIGN):
        pF[:, j]   *= s
        allF[:, j] *= s

    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.38)

    PAIR_AXES = [
        (gs[0, 0], 0, 3, "寿命 vs CPGD"),
        (gs[0, 1], 1, 3, "移動度 vs CPGD"),
        (gs[0, 2], 2, 4, "Yield vs チッピング"),
        (gs[0, 3], 0, 5, "寿命 vs R_on"),
        (gs[1, 0], 0, 2, "寿命 vs Yield"),
        (gs[1, 1], 1, 4, "移動度 vs チッピング"),
    ]
    for ax_spec, i, j, title in PAIR_AXES:
        ax = fig.add_subplot(ax_spec)
        ax.scatter(allF[:, i], allF[:, j], c="lightgray", s=15, alpha=0.4,
                   zorder=1, label="全解")
        sc = ax.scatter(pF[:, i], pF[:, j],
                        c=range(len(pF)), cmap="viridis",
                        s=60, zorder=3, label="Pareto 解", edgecolors="k", lw=0.5)
        ax.set_xlabel(OBJ_LABELS[i], fontsize=8)
        ax.set_ylabel(OBJ_LABELS[j], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # 収束履歴
    ax_hist = fig.add_subplot(gs[1, 2:])
    ax_hist.plot(range(1, len(result["history"])+1), result["history"],
                 "#2166ac", lw=2.5, marker="o", ms=4)
    ax_hist.set_xlabel("世代", fontsize=10)
    ax_hist.set_ylabel("Pareto フロント解数", fontsize=10)
    ax_hist.set_title("NSGA-II 収束履歴", fontsize=10)
    ax_hist.grid(alpha=0.2)

    # 決定変数の分布 (Pareto解)
    var_labels = ["HAZ [µm]", "Ra [nm]", "酸化T [°C]",
                  "アニールT [°C]", "ALD cycles", "CMP P [kPa]"]
    for k in range(6):
        ax = fig.add_subplot(gs[2, k % 4] if k < 4 else
                             gs[2, k-2] if k == 4 else gs[2, 3])
        if k >= 4:
            break
        ax.hist(pX[:, k], bins=12, color="#2ca02c", alpha=0.8, edgecolor="k", lw=0.5)
        ax.set_xlabel(var_labels[k], fontsize=8)
        ax.set_ylabel("Pareto 解数", fontsize=8)
        ax.set_title(f"最適 {var_labels[k]}", fontsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle("NSGA-II マルチ目的最適化 — 完全パイプライン\n"
                 "決定変数: HAZ/Ra/酸化T/アニールT/ALD/CMP  "
                 f"目的: τ/µ_ch/Yield/CPGD/Chip/R_on  "
                 f"Pareto 解数: {len(pF)}",
                 fontsize=11)
    out = os.path.join(out_dir, "nsga2_pareto.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Pareto front -> {out}")

    # テキストサマリー
    print(f"\n  Pareto 解 サマリー ({len(pF)} 解)")
    print(f"  {'τ[µs]':>8} {'µ_ch':>7} {'Yield%':>7} "
          f"{'CPGD[$]':>8} {'Chip[µm]':>9} {'R_on':>7}"
          f"  |  {'HAZ':>6} {'Ra':>6} {'ALD':>5}")
    print("  " + "─"*72)
    idx_sorted = np.argsort(pF[:, 0])[::-1]   # 寿命降順
    for i in idx_sorted[:10]:
        print(f"  {pF[i,0]:>8.1f} {pF[i,1]:>7.1f} {pF[i,2]:>7.2f} "
              f"{pF[i,3]:>8.2f} {pF[i,4]:>9.3f} {pF[i,5]:>7.3f}"
              f"  |  {pX[i,0]:>6.2f} {pX[i,1]:>6.1f} {pX[i,4]:>5.0f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop",  type=int, default=40, help="集団サイズ")
    ap.add_argument("--gen",  type=int, default=20, help="世代数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"NSGA-II 開始: pop={args.pop}, gen={args.gen}")
    t0 = time.time()
    result = nsga2(pop_size=args.pop, n_gen=args.gen, seed=args.seed)
    print(f"\n最適化完了: {time.time()-t0:.1f}s")
    plot_results(result)


if __name__ == "__main__":
    main()
