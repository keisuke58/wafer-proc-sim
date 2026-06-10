"""
TEL FEOL — TMCMC ベイズ較正 (チャンネル移動度モデルの物理パラメータ推定)
===========================================================================
SiC MOSFET 反転層移動度モデル

    1/µ = 1/µ_phonon + Dit/B_coul + 1/µ_sr        (Matthiessen 則)

の物理パラメータ (µ_phonon, B_coul, µ_sr) を、文献アンカー点に対して
TMCMC (Transitional MCMC; Ching & Chen 2007) でベイズ較正する。

これは market_analysis.tel_process_model.channel_mobility_inv と等価な順モデルである
(同関数では mu_coulomb = B_coul/Dit, すなわち 1/mu_coulomb = Dit/B_coul)。
順モデル本体は改変せず、尤度評価のためにこの移動度関係式のみをローカル再実装する。

アンカー点 (Kimoto & Cooper 2014, Table 3.2 / Chung 2001 IEEE EDL):
    Dit = 1e13 cm⁻²eV⁻¹  →  µ_ch ≈  5  cm²/Vs   (無処理界面)
    Dit = 1e12 cm⁻²eV⁻¹  →  µ_ch ≈ 36  cm²/Vs   (標準)
    Dit = 1e11 cm⁻²eV⁻¹  →  µ_ch ≈ 85  cm²/Vs   (NO アニール後 最良界面)

TMCMC の構成 (Ching & Chen 2007 に忠実):
    Stage 0 : broad uniform prior からサンプリング
    各 Stage:
      [a] tempering 指数 β_j を、重要度重みの CoV ≈ 1 になるよう
          二分探索で適応決定 (β: 0 → 1)
      [b] plausibility weight を計算し、stage 正規化定数を蓄積 (→ model evidence)
      [c] 多項分布リサンプリング
      [d] サンプル共分散を β²≈0.04 でスケールした Gaussian proposal で
          Metropolis ランダムウォーク 1 step
    収束 (β=1) または最大 stage 数で停止。

出力:
    各パラメータの事後平均 ± 標準偏差、log model evidence、
    事後予測 µ_ch(Dit) の 95% 信用区間バンド。

図 (results/tel_tmcmc_posterior.png):
    3 パラメータの事後周辺ヒストグラム (corner 風)
    + µ_ch(Dit) アンカー点と事後予測 95% バンド。

実行:
    python optimization/tel_tmcmc_calibration.py
    python optimization/tel_tmcmc_calibration.py --n-samples 800 --seed 7

参考: Ching & Chen (2007) J Eng Mech; Kimoto & Cooper (2014); Saks et al. (1999)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/home/nishioka/git/wafer-proc-sim")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "/home/nishioka/git/wafer-proc-sim/results"
FIG_PATH = os.path.join(RESULTS_DIR, "tel_tmcmc_posterior.png")

# ════════════════════════════════════════════════════════════════════════════
# 文献アンカー点 (Kimoto & Cooper 2014)
# ════════════════════════════════════════════════════════════════════════════
ANCHOR_DIT = np.array([1.0e13, 1.0e12, 1.0e11])      # cm⁻²eV⁻¹
ANCHOR_MU  = np.array([5.0,    36.0,   85.0])         # cm²/Vs
# 観測ノイズ (移動度抽出の不確かさ; 低 µ ほど絶対誤差小)。Gaussian 尤度の σ。
ANCHOR_SIGMA = np.array([1.0,  3.0,    6.0])          # cm²/Vs

# ── 較正パラメータ: name -> (lo, hi) broad uniform prior ─────────────────────
# market_analysis.tel_process_model の公称値 _MU_SP=155, _B_COUL=5.5e13, _MU_SR=320 を内包。
PARAM_NAMES = ["mu_phonon", "B_coul", "mu_sr"]
PARAM_BOUNDS = np.array([
    [80.0,   400.0],     # µ_phonon [cm²/Vs]  表面フォノン散乱限界
    [1.0e13, 1.5e14],    # B_coul   [cm⁻²eV⁻¹·cm²/Vs] Coulomb 散乱係数
    [150.0,  800.0],     # µ_sr     [cm²/Vs]  表面粗さ散乱限界
])
N_PARAMS = len(PARAM_NAMES)


# ════════════════════════════════════════════════════════════════════════════
# 移動度 順モデル (尤度用ローカル再実装)
#   channel_mobility_inv とビット等価: 1/mu = 1/mu_phonon + Dit/B_coul + 1/mu_sr
# ════════════════════════════════════════════════════════════════════════════
def mobility_model(theta: np.ndarray, Dit: np.ndarray) -> np.ndarray:
    """物理パラメータ theta=[mu_phonon, B_coul, mu_sr] と Dit から µ_ch を予測。"""
    mu_phonon, B_coul, mu_sr = theta
    Dit = np.asarray(Dit, dtype=float)
    inv_mu = 1.0 / mu_phonon + Dit / B_coul + 1.0 / mu_sr
    return 1.0 / inv_mu


# ── log-prior: broad uniform ──────────────────────────────────────────────────
def log_prior(theta: np.ndarray) -> float:
    if np.all(theta >= PARAM_BOUNDS[:, 0]) and np.all(theta <= PARAM_BOUNDS[:, 1]):
        return 0.0
    return -np.inf


# ── log-likelihood: 3 アンカー残差の Gaussian 尤度 ───────────────────────────
def log_likelihood(theta: np.ndarray) -> float:
    if not np.isfinite(log_prior(theta)):
        return -np.inf
    mu_pred = mobility_model(theta, ANCHOR_DIT)
    resid = (ANCHOR_MU - mu_pred) / ANCHOR_SIGMA
    return float(-0.5 * np.sum(resid ** 2) - np.sum(np.log(ANCHOR_SIGMA)))


# ════════════════════════════════════════════════════════════════════════════
# TMCMC コア (Ching & Chen 2007)
# ════════════════════════════════════════════════════════════════════════════
def tmcmc(log_like_fn, log_prior_fn, bounds,
          n_samples: int = 800, max_stages: int = 30,
          target_cov: float = 1.0, beta_scale: float = 0.2,
          seed: int = 42, verbose: bool = True) -> dict:
    """
    Transitional MCMC。

    Args:
        target_cov : 重要度重みの目標変動係数 (CoV)。Ching & Chen は ~1 を推奨。
        beta_scale : Metropolis proposal 共分散のスケール。共分散に beta_scale²
                     (≈0.04, Ching-Chen 推奨) を掛ける。
    Returns:
        samples (n,p), weights (n,), log_evidence, beta_final, stages, beta_history
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    n_p = bounds.shape[0]

    # ── Stage 0: prior サンプリング ───────────────────────────────────────────
    samples = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, n_p))
    log_likes = np.array([log_like_fn(s) for s in samples])

    p = 0.0                # tempering 指数 (= β)
    log_evidence = 0.0
    beta_hist = [0.0]
    scale2 = beta_scale ** 2

    for stage in range(max_stages):
        p_prev = p

        # ── [a] 次の β を CoV(weights) ≈ target_cov になるよう二分探索 ───────
        lo, hi = p_prev, 1.0
        for _ in range(60):
            dp = 0.5 * (lo + hi) - p_prev
            log_w = dp * log_likes
            log_w -= log_w.max()
            w = np.exp(log_w)
            cov_w = w.std() / w.mean()          # 重みの変動係数
            if cov_w > target_cov:
                hi = 0.5 * (lo + hi)
            else:
                lo = 0.5 * (lo + hi)
            if hi - lo < 1e-8:
                break
        p = min(1.0, lo)
        # CoV が target を超えない (=分布が緩い) なら一気に β=1 へ
        dp_full = 1.0 - p_prev
        log_w_full = dp_full * log_likes
        log_w_full -= log_w_full.max()
        w_full = np.exp(log_w_full)
        if w_full.std() / w_full.mean() <= target_cov:
            p = 1.0

        dp = p - p_prev

        # ── [b] plausibility weights + stage 正規化定数 (→ evidence) ─────────
        log_w = dp * log_likes
        c = log_w.max()
        w = np.exp(log_w - c)
        # stage の log 正規化定数 S_j = log mean_i exp(dp * L_i)
        log_evidence += c + np.log(w.mean())
        w_norm = w / w.sum()
        ess = 1.0 / np.sum(w_norm ** 2)

        if verbose:
            print("  Stage %2d: beta=%.4f (dbeta=%.4f)  CoV=%.2f  ESS=%.0f"
                  % (stage + 1, p, dp, w.std() / w.mean(), ess))

        beta_hist.append(p)

        if p >= 1.0:
            samples_final = samples
            weights_final = w_norm
            break

        # ── proposal 共分散: plausibility weight 付きサンプル共分散 ──────────
        # Ching & Chen (2007) Eq.17-18: 次の tempered target の重み付き共分散を
        # beta_scale²(≈0.04) でスケール。これがランダムウォークのステップ幅。
        mean_w = np.average(samples, weights=w_norm, axis=0)
        dev = samples - mean_w
        wcov = (w_norm[:, None] * dev).T @ dev      # 重み付き共分散 (Σw=1)
        prop_cov = scale2 * wcov
        # 数値安定化 (退化防止): パラメータごとにスケールが大きく異なるため
        # 各次元の自己分散に比例した相対 jitter を対角へ加える (絶対 jitter は不可)。
        prop_cov = prop_cov + np.diag(np.maximum(np.diag(prop_cov), 1e-30) * 1e-9)

        # ── [c] 多項分布リサンプリング ───────────────────────────────────────
        idx = rng.choice(n_samples, size=n_samples, p=w_norm)
        samples = samples[idx].copy()
        log_likes = log_likes[idx].copy()

        # ── [d] Metropolis ランダムウォーク (β²スケール 共分散) ──────────────
        try:
            L = np.linalg.cholesky(prop_cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(
                prop_cov + np.diag(np.maximum(np.diag(prop_cov), 1e-30) * 1e-6))

        n_acc = 0
        for i in range(n_samples):
            proposal = samples[i] + L @ rng.standard_normal(n_p)
            lp_prop = log_prior_fn(proposal)
            if not np.isfinite(lp_prop):
                continue
            ll_prop = log_like_fn(proposal)
            log_alpha = (p * ll_prop + lp_prop) - (p * log_likes[i] + log_prior_fn(samples[i]))
            if np.log(rng.random()) < log_alpha:
                samples[i] = proposal
                log_likes[i] = ll_prop
                n_acc += 1
        if verbose:
            print("            MCMC accept rate = %.2f" % (n_acc / n_samples))
    else:
        # max_stages 到達: 最終 β での重み
        samples_final = samples
        log_w = (1.0 - p) * log_likes
        log_w -= log_w.max()
        wf = np.exp(log_w)
        weights_final = wf / wf.sum()

    return {
        "samples": samples_final,
        "weights": weights_final,
        "log_evidence": float(log_evidence),
        "beta_final": float(p),
        "stages": len(beta_hist) - 1,
        "beta_history": np.array(beta_hist),
    }


# ════════════════════════════════════════════════════════════════════════════
# 事後予測バンド
# ════════════════════════════════════════════════════════════════════════════
def posterior_predictive_band(samples, weights, dit_grid, q=(2.5, 50.0, 97.5)):
    """事後サンプルから µ_ch(Dit) の重み付き分位点バンドを計算。"""
    # 重みに比例する回数だけリサンプルして等重みサンプルにする
    rng = np.random.default_rng(0)
    n = len(samples)
    idx = rng.choice(n, size=max(n, 2000), p=weights)
    eq = samples[idx]
    mu_curves = np.array([mobility_model(s, dit_grid) for s in eq])  # (m, len(grid))
    lo = np.percentile(mu_curves, q[0], axis=0)
    med = np.percentile(mu_curves, q[1], axis=0)
    hi = np.percentile(mu_curves, q[2], axis=0)
    return lo, med, hi


# ════════════════════════════════════════════════════════════════════════════
# 図
# ════════════════════════════════════════════════════════════════════════════
def make_figure(samples, weights, summary, out_path=FIG_PATH):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(samples), size=max(len(samples), 2000), p=weights)
    eq = samples[idx]   # 等重みサンプル

    colors = ["#2166ac", "#2ca02c", "#d62728"]
    units = ["[cm²/Vs]", "[cm⁻²eV⁻¹·cm²/Vs]", "[cm²/Vs]"]

    fig = plt.figure(figsize=(15, 5))

    # ── Panel 1-3: 各パラメータの事後周辺 ─────────────────────────────────────
    for k in range(N_PARAMS):
        ax = fig.add_subplot(1, 4, k + 1)
        ax.hist(eq[:, k], bins=35, density=True, color=colors[k],
                alpha=0.75, edgecolor="k", linewidth=0.3)
        m = summary["mean"][k]
        s = summary["std"][k]
        ax.axvline(m, color="k", lw=2, label="mean")
        ax.axvspan(m - s, m + s, color="k", alpha=0.10, label="±1σ")
        nm = PARAM_NAMES[k]
        ax.set_xlabel("%s %s" % (nm, units[k]), fontsize=10)
        ax.set_ylabel("posterior density", fontsize=9)
        if nm == "B_coul":
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.set_title("%s\n%.3g ± %.2g" % (nm, m, s), fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    # ── Panel 4: µ_ch(Dit) + 事後予測 95% バンド ─────────────────────────────
    ax = fig.add_subplot(1, 4, 4)
    dit_grid = np.logspace(10.7, 13.3, 120)
    lo, med, hi = posterior_predictive_band(samples, weights, dit_grid)
    ax.fill_between(dit_grid, lo, hi, color="#2166ac", alpha=0.25,
                    label="95% post. predictive")
    ax.plot(dit_grid, med, color="#2166ac", lw=2, label="posterior median")
    ax.errorbar(ANCHOR_DIT, ANCHOR_MU, yerr=ANCHOR_SIGMA, fmt="o",
                color="#d62728", ms=9, capsize=4, zorder=5,
                label="anchors (Kimoto 2014)")
    ax.set_xscale("log")
    ax.set_xlabel("Dit [cm⁻²eV⁻¹]", fontsize=10)
    ax.set_ylabel("µ_ch [cm²/Vs]", fontsize=10)
    ax.set_title("Posterior predictive\nµ_ch vs Dit", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, which="both")

    fig.suptitle("TEL FEOL - TMCMC Bayesian calibration (SiC MOSFET mobility model)\n"
                 "log evidence = %.2f   stages = %d   beta_final = %.2f"
                 % (summary["log_evidence"], summary["stages"],
                    summary["beta_final"]),
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] saved -> %s" % out_path)


# ════════════════════════════════════════════════════════════════════════════
# ドライバ
# ════════════════════════════════════════════════════════════════════════════
def run(n_samples=800, max_stages=30, seed=42, beta_scale=0.2):
    print("[*] TMCMC Bayesian calibration of SiC channel-mobility model")
    print("    forward: 1/mu = 1/mu_phonon + Dit/B_coul + 1/mu_sr")
    print("    anchors (Kimoto & Cooper 2014):")
    for d, m, s in zip(ANCHOR_DIT, ANCHOR_MU, ANCHOR_SIGMA):
        print("      Dit=%.0e  ->  mu=%.0f +/- %.0f cm2/Vs" % (d, m, s))
    print()

    res = tmcmc(log_likelihood, log_prior, PARAM_BOUNDS,
                n_samples=n_samples, max_stages=max_stages,
                beta_scale=beta_scale, seed=seed)

    samples = res["samples"]
    weights = res["weights"]
    mean = np.average(samples, weights=weights, axis=0)
    var = np.average((samples - mean) ** 2, weights=weights, axis=0)
    std = np.sqrt(var)
    map_idx = int(np.argmax(weights))

    summary = {
        "mean": mean, "std": std, "map": samples[map_idx],
        "log_evidence": res["log_evidence"],
        "beta_final": res["beta_final"], "stages": res["stages"],
    }

    print("\n[Results] Posterior mean +/- std:")
    for k, nm in enumerate(PARAM_NAMES):
        print("  %-10s = %12.4g +/- %-12.4g  (MAP %.4g)"
              % (nm, mean[k], std[k], samples[map_idx, k]))
    print("  log model evidence = %.3f" % res["log_evidence"])
    print("  stages = %d   beta_final = %.3f" % (res["stages"], res["beta_final"]))

    # 較正後アンカー再現チェック
    mu_fit = mobility_model(mean, ANCHOR_DIT)
    print("\n[Check] posterior-mean fit vs anchors:")
    for d, m, mf in zip(ANCHOR_DIT, ANCHOR_MU, mu_fit):
        print("  Dit=%.0e  data=%.1f  fit=%.1f  (resid=%+.2f)"
              % (d, m, mf, mf - m))

    make_figure(samples, weights, summary)
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="TMCMC Bayesian calibration of SiC channel-mobility params")
    ap.add_argument("--n-samples", type=int, default=800,
                    help="TMCMC サンプル数 (default 800)")
    ap.add_argument("--max-stages", type=int, default=30,
                    help="最大 tempering stage 数")
    ap.add_argument("--beta-scale", type=float, default=0.2,
                    help="Metropolis proposal 共分散スケール (β, 推奨 0.2 → β²≈0.04)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(n_samples=args.n_samples, max_stages=args.max_stages,
        seed=args.seed, beta_scale=args.beta_scale)


if __name__ == "__main__":
    main()
