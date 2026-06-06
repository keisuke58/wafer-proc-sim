"""
TEL FEOL 多目的ベイズ最適化 (ParEGO) — SiC MOSFET レシピの Pareto 探索
========================================================================
forward model (fem/tel_feol_recipe.py) を直接呼びながら、3 目的

    OBJECTIVES = [("mu_ch","max"), ("R_on_sp","min"), ("thermal_budget","min")]

の Pareto フロントを ParEGO 方式で探索する。ParEGO (Knowles 2006) は
多目的問題を反復ごとにランダムな Tchebycheff 重みでスカラー化し、その
スカラー目的に対して単一目的 BO (GP + Expected Improvement) を回すことで
Pareto フロント全体を少評価で覆う手法。

アルゴリズム:
    [1] 初期計画: sample_recipes(N_INIT) を真の forward model で評価。
    [2] 反復 (N_ITER 回):
        a. 評価済み全点の 3 目的を [0,1] に正規化。
        b. ランダムな重み w (単体上) を引き、拡張 Tchebycheff
           スカラー化 g(x) = max_j w_j f_j(x) + rho * sum_j w_j f_j(x)
           を計算 (全て最小化、小さいほど良)。
        c. 標準化した recipe ベクトルを入力、g を出力として GP を学習。
        d. ランダム候補プール (sample_recipes) 上で Expected Improvement
           (最小化版) を最大化し、最良点を真の forward model で評価。
        e. 評価アーカイブに追加。
    [3] 全評価点に対し非劣解ソート (all-min objective_vector) で Pareto
        集合を抽出。
    [4] 固定参照点に対する dominated hypervolume を、初期計画のみ
        (baseline) と全評価後で比較し、改善量を報告。

出力:
    results/tel_feol_pareto.png   全評価点 + Pareto 強調 (mu_ch vs R_on_sp,
                                  色 = thermal_budget) の散布図
    results/tel_feol_pareto.json  代表 Pareto レシピ (best mu_ch / R_on / tb)

実行:
    python optimization/tel_feol_bo.py                       # default run
    python optimization/tel_feol_bo.py --n-init 20 --n-iter 40 --seed 0

参考:
    Knowles (2006) IEEE TEC — ParEGO: scalarizing MO via Tchebycheff + EGO
    Jones, Schonlau, Welch (1998) — Efficient Global Optimization (EI)
    Zitzler & Thiele (1999) — dominated hypervolume indicator
    Kimoto & Cooper (2014) — SiC MOSFET technology (forward model 物理)
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, "/home/nishioka/git/wafer-proc-sim")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

from fem.tel_feol_recipe import (
    simulate_recipe,
    objective_vector,
    sample_recipes,
    recipe_to_vector,
    OBJECTIVES,
    VECTOR_BOUNDS,
)

RESULTS_DIR = "/home/nishioka/git/wafer-proc-sim/results"
FIG_PATH = os.path.join(RESULTS_DIR, "tel_feol_pareto.png")
JSON_PATH = os.path.join(RESULTS_DIR, "tel_feol_pareto.json")

_OBJ_NAMES = [name for name, _ in OBJECTIVES]   # ["mu_ch","R_on_sp","thermal_budget"]
_RHO = 0.05    # 拡張 Tchebycheff の重み (ParEGO 標準値)


# ════════════════════════════════════════════════════════════════════════════
# 非劣解ソート / hypervolume (all-min objective space)
# ════════════════════════════════════════════════════════════════════════════

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """a が b を支配 (全 min) : a <= b 全成分かつ少なくとも 1 成分で a < b。"""
    return np.all(a <= b) and np.any(a < b)


def pareto_mask(Y: np.ndarray) -> np.ndarray:
    """目的行列 Y (all-min, shape [n,m]) の非劣 (Pareto) 点を True で返す。"""
    n = Y.shape[0]
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if _dominates(Y[j], Y[i]):
                nd[i] = False
                break
    return nd


def hypervolume(Y: np.ndarray, ref: np.ndarray) -> float:
    """
    all-min 3 目的空間における dominated hypervolume (exact, 包除なしの
    スライス法)。ref は参照点 (全 min なので各目的の "最悪" 上限)。
    点を第 3 軸 (z) で昇順ソートし、z スラブごとに (x,y) の支配領域面積を
    逐次更新して厚み dz を掛けて積算する標準アルゴリズム。
    """
    Y = np.atleast_2d(Y)
    nd = Y[pareto_mask(Y)]
    if nd.size == 0:
        return 0.0
    nd = nd[np.all(nd < ref[None, :], axis=1)]
    if nd.shape[0] == 0:
        return 0.0
    if ref.shape[0] != 3:
        raise ValueError("exact HV is implemented for 3 objectives")

    # z (第3目的) 小さい順 = 良い順
    order = np.argsort(nd[:, 2])
    nd = nd[order]
    active = []                # (x, y) of points contributing to current 2D front
    vol = 0.0
    z_prev = nd[0, 2]
    for k in range(nd.shape[0] + 1):
        if k < nd.shape[0]:
            z_cur = nd[k, 2]
        else:
            z_cur = ref[2]
        dz = z_cur - z_prev
        if dz > 0 and active:
            vol += _area_2d(np.array(active), ref[:2]) * dz
        if k < nd.shape[0]:
            active.append((nd[k, 0], nd[k, 1]))
            z_prev = z_cur
    return float(vol)


def _area_2d(pts: np.ndarray, ref2: np.ndarray) -> float:
    """
    all-min 2D 点集合が ref2 まで支配する領域面積 (staircase)。
    Pareto 点を x 昇順 (良い順) に並べると y は単調増加 (悪化) になる。
    各点 i は x 区間 [x_i, x_{i+1}] で y を支配し、その高さは (ref_y - y_i)。
    """
    nd = pts[pareto_mask(pts)]
    nd = nd[np.argsort(nd[:, 0])]      # x 昇順
    area = 0.0
    for i in range(nd.shape[0]):
        x_i, y_i = nd[i]
        x_next = nd[i + 1, 0] if i + 1 < nd.shape[0] else ref2[0]
        if x_next > x_i and ref2[1] > y_i:
            area += (x_next - x_i) * (ref2[1] - y_i)
    return max(area, 0.0)


# ════════════════════════════════════════════════════════════════════════════
# ParEGO スカラー化 + GP + EI
# ════════════════════════════════════════════════════════════════════════════

def _normalize(Y: np.ndarray) -> np.ndarray:
    """all-min 目的を [0,1] に正規化 (列ごと、レンジ 0 を回避)。"""
    lo = Y.min(axis=0)
    hi = Y.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (Y - lo) / span


def _tchebycheff(Yn: np.ndarray, w: np.ndarray, rho: float = _RHO) -> np.ndarray:
    """拡張 Tchebycheff スカラー化 (全て最小化、小さいほど良)。"""
    wf = w[None, :] * Yn
    return wf.max(axis=1) + rho * wf.sum(axis=1)


def _simplex_weights(s: int, m: int) -> np.ndarray:
    """単体 {sum=1} 上の規則格子重みベクトル (Das-Dennis, s 分割)。"""
    from itertools import combinations
    pts = []
    for c in combinations(range(s + m - 1), m - 1):
        prev = -1
        coords = []
        for ci in c:
            coords.append(ci - prev - 1)
            prev = ci
        coords.append(s + m - 1 - prev - 1)
        pts.append([x / s for x in coords])
    return np.array(pts, dtype=float)


def _build_gp() -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=np.ones(len(VECTOR_BOUNDS)),
                 length_scale_bounds=(1e-1, 1e2), nu=2.5)
        + WhiteKernel(1e-3, (1e-6, 1e0))
    )
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=0,
        alpha=1e-8, random_state=0,
    )


def _expected_improvement_min(mu, sigma, y_best, xi=0.01):
    """最小化版 EI: 現状最良 y_best より小さくなる期待改善。"""
    sigma = np.maximum(sigma, 1e-12)
    imp = y_best - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    return np.maximum(ei, 0.0)


# ════════════════════════════════════════════════════════════════════════════
# ParEGO 本体
# ════════════════════════════════════════════════════════════════════════════

def run_parego(n_init=20, n_iter=40, pool=300, seed=0, verbose=True):
    """ParEGO 多目的 BO を実行し、評価アーカイブと診断値を返す。"""
    rng = np.random.default_rng(seed)
    m = len(OBJECTIVES)

    # ── [1] 初期計画 ────────────────────────────────────────────────
    recipes = sample_recipes(n_init, seed=seed)
    X = [recipe_to_vector(r) for r in recipes]
    Y = [objective_vector(r) for r in recipes]   # all-min
    X = list(X)
    Y = list(Y)

    n_baseline = len(Y)

    # ── 規則格子の Tchebycheff 重み (Das-Dennis) をシャッフルして巡回。
    #    各反復で異なる重み = Pareto フロントを満遍なく狙い、多様性を確保。
    W = _simplex_weights(6, m)              # m=3, s=6 -> 28 重みベクトル
    W = W[rng.permutation(W.shape[0])]

    # ── [2] BO 反復 ─────────────────────────────────────────────────
    for it in range(n_iter):
        Xa = np.array(X)
        Ya = np.array(Y)

        # a-b. 正規化 + 規則格子重みによる Tchebycheff スカラー化
        Yn = _normalize(Ya)
        w = W[it % W.shape[0]]
        g = _tchebycheff(Yn, w)     # 小さいほど良

        # c. GP を (標準化入力 -> g) で学習
        x_mu = Xa.mean(axis=0)
        x_sd = Xa.std(axis=0)
        x_sd = np.where(x_sd > 1e-12, x_sd, 1.0)
        Xs = (Xa - x_mu) / x_sd
        gp = _build_gp()
        gp.fit(Xs, g)

        # d. 候補プール上で EI 最大化
        cand_recipes = sample_recipes(pool, seed=int(rng.integers(1, 1_000_000)))
        Xc = np.array([recipe_to_vector(r) for r in cand_recipes])
        Xcs = (Xc - x_mu) / x_sd
        mu_c, sd_c = gp.predict(Xcs, return_std=True)
        y_best = g.min()
        ei = _expected_improvement_min(mu_c, sd_c, y_best)
        best_idx = int(np.argmax(ei))
        next_recipe = cand_recipes[best_idx]

        # e. 真の forward model を評価して追加
        X.append(recipe_to_vector(next_recipe))
        Y.append(objective_vector(next_recipe))
        recipes.append(next_recipe)

        if verbose and (it + 1) % 10 == 0:
            cur = np.array(Y)
            n_pf = int(pareto_mask(cur).sum())
            print(f"  iter {it+1:3d}/{n_iter}  |evals|={len(Y):3d}  "
                  f"|Pareto|={n_pf:2d}  EI*={ei[best_idx]:.3e}")

    Ya = np.array(Y)
    return {
        "recipes": recipes,
        "X": np.array(X),
        "Y": Ya,
        "n_baseline": n_baseline,
    }


# ════════════════════════════════════════════════════════════════════════════
# レポート / 図 / JSON
# ════════════════════════════════════════════════════════════════════════════

def _fom_row(recipe):
    f = simulate_recipe(recipe)
    return {k: float(f[k]) for k in
            ("mu_ch", "R_on_sp", "thermal_budget", "V_th", "Dit", "EOT_nm",
             "I_on_mA", "uniformity")}


def representative_recipes(recipes, Y):
    """Pareto 集合から best-mu_ch / best-R_on / best-thermal_budget を抽出。"""
    nd = pareto_mask(Y)
    idx = np.where(nd)[0]
    Ynd = Y[idx]
    # objective_vector 順: [-mu_ch, R_on_sp, thermal_budget]
    reps = {
        "best_mu_ch":         idx[int(np.argmin(Ynd[:, 0]))],   # 最小の -mu_ch = 最大 mu_ch
        "best_R_on_sp":       idx[int(np.argmin(Ynd[:, 1]))],
        "best_thermal_budget": idx[int(np.argmin(Ynd[:, 2]))],
    }
    out = {}
    for label, i in reps.items():
        r = recipes[i]
        out[label] = {"recipe": r, "fom": _fom_row(r)}
    return out


def make_figure(Y, recipes, out_path=FIG_PATH):
    """全評価点 + Pareto front を (mu_ch vs R_on_sp), 色=thermal_budget で描画。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 各点の物理 FOM
    foms = [simulate_recipe(r) for r in recipes]
    mu = np.array([f["mu_ch"] for f in foms])
    ron = np.array([f["R_on_sp"] for f in foms])
    tb = np.array([f["thermal_budget"] for f in foms])

    nd = pareto_mask(Y)
    pf_idx = np.where(nd)[0]
    # Pareto を mu_ch 昇順に並べて折れ線にする
    order = pf_idx[np.argsort(mu[pf_idx])]

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    sc = ax.scatter(mu, ron, c=tb, cmap="viridis", s=42,
                    edgecolor="k", linewidth=0.3, alpha=0.85,
                    label="全評価点 (all evaluations)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("thermal_budget (小さいほど良 / lower is better)")

    ax.scatter(mu[pf_idx], ron[pf_idx], s=150, facecolors="none",
               edgecolors="crimson", linewidths=2.0, zorder=5,
               label=f"Pareto front (n={len(pf_idx)})")
    ax.plot(mu[order], ron[order], "--", color="crimson",
            linewidth=1.2, alpha=0.7, zorder=4)

    ax.set_xlabel(r"$\mu_{ch}$  [cm$^2$/Vs]  (大きいほど良 →)")
    ax.set_ylabel(r"$R_{on,sp}$  [m$\Omega\cdot$mm$^2$]  (← 小さいほど良)")
    ax.set_title("TEL FEOL 多目的 BO (ParEGO) — SiC MOSFET レシピ Pareto front")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="TEL FEOL 多目的ベイズ最適化 (ParEGO, 3 目的 Pareto)")
    ap.add_argument("--n-init", type=int, default=20, help="初期計画点数")
    ap.add_argument("--n-iter", type=int, default=40, help="BO 反復回数")
    ap.add_argument("--pool", type=int, default=300, help="EI 候補プール数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[ParEGO] n_init={args.n_init} n_iter={args.n_iter} "
          f"pool={args.pool} seed={args.seed}")
    res = run_parego(n_init=args.n_init, n_iter=args.n_iter,
                     pool=args.pool, seed=args.seed)

    Y = res["Y"]
    recipes = res["recipes"]
    nb = res["n_baseline"]

    # ── 参照点 (all-min 空間, 全評価の最悪値より少し悪く固定) ──────────
    worst = Y.max(axis=0)
    ref = worst + 0.05 * np.abs(worst) + 1e-6

    hv_baseline = hypervolume(Y[:nb], ref)
    hv_final = hypervolume(Y, ref)
    hv_gain = hv_final - hv_baseline
    hv_gain_pct = 100.0 * hv_gain / hv_baseline if hv_baseline > 0 else float("nan")

    nd = pareto_mask(Y)
    n_pf = int(nd.sum())

    # ── レポート ──────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  総評価数            : {len(Y)}  (init {nb} + BO {len(Y)-nb})")
    print(f"  Pareto 集合サイズ   : {n_pf}")
    print(f"  参照点 (all-min)    : {np.array2string(ref, precision=3)}")
    print(f"  HV (baseline init)  : {hv_baseline:.4f}")
    print(f"  HV (ParEGO final)   : {hv_final:.4f}")
    print(f"  HV 改善             : +{hv_gain:.4f}  ({hv_gain_pct:+.1f}%)")
    print("=" * 64)

    reps = representative_recipes(recipes, Y)
    print("\n代表 Pareto レシピ:")
    for label, d in reps.items():
        f = d["fom"]
        r = d["recipe"]
        print(f"  [{label}]")
        print(f"      mu_ch={f['mu_ch']:.1f}  R_on_sp={f['R_on_sp']:.3f}  "
              f"tb={f['thermal_budget']:.3f}  V_th={f['V_th']:.2f}")
        print(f"      ald_material={r['ald_material']}  dicing={r['dicing_process']}  "
              f"clean={r['clean_seq']}  rie={r['rie_gas']}  "
              f"ox_T={r['ox_T_C']:.0f}C  cycles={int(r['ald_cycles'])}")

    # ── 図 + JSON ─────────────────────────────────────────────────────
    fig_path = make_figure(Y, recipes)
    print(f"\n[OK] saved -> {fig_path}")

    pf_idx = np.where(nd)[0]
    pareto_list = []
    for i in pf_idx:
        pareto_list.append({"recipe": recipes[i], "fom": _fom_row(recipes[i])})
    out = {
        "settings": {"n_init": args.n_init, "n_iter": args.n_iter,
                     "pool": args.pool, "seed": args.seed},
        "objectives": [{"name": n, "sense": s} for n, s in OBJECTIVES],
        "n_evaluations": int(len(Y)),
        "pareto_size": n_pf,
        "reference_point_allmin": ref.tolist(),
        "hypervolume_baseline": hv_baseline,
        "hypervolume_final": hv_final,
        "hypervolume_gain": hv_gain,
        "hypervolume_gain_pct": hv_gain_pct,
        "representatives": reps,
        "pareto_front": pareto_list,
    }
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, "w") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f"[OK] saved -> {JSON_PATH}")


if __name__ == "__main__":
    main()
