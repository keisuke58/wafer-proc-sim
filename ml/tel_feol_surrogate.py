"""
TEL FEOL Surrogate — GP 代理モデル (recipe → FOM)
====================================================
fem/tel_feol_recipe.py の forward model (洗浄→ダイシング→RIE→熱酸化→NO アニール
→ALD→MOSFET 性能) は物理ブロックの連鎖でそこそこ重い。BO / APC が同じ写像を
何百回も呼ぶと無駄なので、Gaussian Process でその場を安価に近似する。

方法:
    入力     recipe_to_vector で 8 次元ベクトル化 → StandardScaler で標準化
    出力     µ_ch [cm²/Vs] (大きいほど良) と R_on_sp [mΩ·mm²] (小さいほど良) の 2 ターゲット
    モデル   各ターゲットに独立な GP
             kernel = ConstantKernel × RBF + WhiteKernel, normalize_y=True
    検証     80/20 hold-out + k-fold CV で RMSE / R² を両ターゲットについて報告
    保存     joblib → results/tel_feol_surrogate.pkl

狙い: 物理 forward model を 1 回も呼ばずに µ_ch / R_on_sp を予測できる代理を作り、
      BO/APC の繰り返し評価コストを潰す。両ターゲットの達成 R² を報告する。

実行:
    python ml/tel_feol_surrogate.py                 # N=120, 80/20 + 2-fold, 図保存
    python ml/tel_feol_surrogate.py --n 200 --folds 5
    python ml/tel_feol_surrogate.py --no-plot

参考: Rasmussen & Williams (2006) GPML; Kimoto & Cooper (2014);
      Frazier (2018) "A Tutorial on Bayesian Optimization".
"""

import argparse
import os
import sys
import warnings

import numpy as np
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, "/home/nishioka/git/wafer-proc-sim")

from fem.tel_feol_recipe import (
    simulate_recipe,
    sample_recipes,
    recipe_to_vector,
    default_recipe,
    VECTOR_NAMES,
)

RESULTS    = "/home/nishioka/git/wafer-proc-sim/results"
MODEL_PATH = os.path.join(RESULTS, "tel_feol_surrogate.pkl")
FIG_PATH   = os.path.join(RESULTS, "tel_feol_surrogate.png")

#: 代理対象 FOM (name, sense)。GP は生の物理単位を学習する (符号反転しない)。
TARGETS = [("mu_ch", "max"), ("R_on_sp", "min")]


# ════════════════════════════════════════════════════════════════════════════
# データ生成
# ════════════════════════════════════════════════════════════════════════════

def build_dataset(n: int, seed: int = 0):
    """recipe 空間を一様サンプリングし forward model を評価して学習データを作る。

    返り値:
        X        (n, 8)  recipe_to_vector のスタック (物理単位 + カテゴリ index)
        Y        (n, 2)  [mu_ch, R_on_sp] の物理値
        recipes  list[dict]
    """
    recipes = sample_recipes(n, seed=seed)
    X, Y = [], []
    for rec in recipes:
        fom = simulate_recipe(rec)
        X.append(recipe_to_vector(rec))
        Y.append([fom[name] for name, _ in TARGETS])
    return np.asarray(X, float), np.asarray(Y, float), recipes


# ════════════════════════════════════════════════════════════════════════════
# 代理モデル
# ════════════════════════════════════════════════════════════════════════════

def _make_gp() -> GaussianProcessRegressor:
    """RBF + WhiteKernel + ConstantKernel の単一ターゲット GP。

    入力は StandardScaler 済み (各次元 ~標準正規) なので length_scale=1.0 起点、
    ARD (次元ごとの length_scale) で連続変数とカテゴリ index の効きを分離する。
    等方 RBF だと R² が大きく落ちる (anneal_T / ald_material index の効きが
    支配的で、次元ごとに最適スケールが 2 桁違う) ので ARD を採用。
    n_restarts_optimizer は 60s 制約のため低めに抑える。
    normalize_y=True でターゲットの平均/分散を内部処理する。
    """
    n_dim = len(VECTOR_NAMES)  # 8
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=np.ones(n_dim),
              length_scale_bounds=(1e-1, 1e2))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=2,
        normalize_y=True,
        alpha=1e-10,
        random_state=0,
    )


class FeolSurrogate:
    """FEOL forward model の GP 代理。

    µ_ch と R_on_sp に独立な GP を 1 本ずつ持つ。入力は recipe_to_vector を
    共通の StandardScaler で標準化したもの。

    API:
        fit(recipes)                       -> self
        fit_xy(X, Y)                       -> self   (ベクトル直接)
        predict(recipe, return_std=False)  -> (mu_hat, R_on_hat[, std(2,)])
        predict_batch(X, return_std)       -> (mu(n,), R_on(n,)[, std(n,2)])
    """

    def __init__(self):
        self.scaler_x = StandardScaler()
        self.gp_mu  = _make_gp()   # µ_ch
        self.gp_ron = _make_gp()   # R_on_sp
        self.target_names = [name for name, _ in TARGETS]

    # ── 学習 ────────────────────────────────────────────────────────────────
    def fit_xy(self, X: np.ndarray, Y: np.ndarray):
        Xs = self.scaler_x.fit_transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp_mu.fit(Xs, Y[:, 0])
            self.gp_ron.fit(Xs, Y[:, 1])
        return self

    def fit(self, recipes: list):
        X = np.asarray([recipe_to_vector(r) for r in recipes], float)
        Y = np.asarray(
            [[simulate_recipe(r)[n] for n in self.target_names] for r in recipes],
            float,
        )
        return self.fit_xy(X, Y)

    # ── 予測 ────────────────────────────────────────────────────────────────
    def predict_batch(self, X: np.ndarray, return_std: bool = False):
        Xs = self.scaler_x.transform(np.atleast_2d(X))
        if return_std:
            mu, s_mu   = self.gp_mu.predict(Xs, return_std=True)
            ron, s_ron = self.gp_ron.predict(Xs, return_std=True)
            std = np.column_stack([s_mu, s_ron])
            return mu, ron, std
        mu  = self.gp_mu.predict(Xs)
        ron = self.gp_ron.predict(Xs)
        return mu, ron

    def predict(self, recipe: dict, return_std: bool = False):
        """単一 recipe → (mu_hat, R_on_hat[, std(2,)])。"""
        x = recipe_to_vector(recipe).reshape(1, -1)
        if return_std:
            mu, ron, std = self.predict_batch(x, return_std=True)
            return float(mu[0]), float(ron[0]), std[0]
        mu, ron = self.predict_batch(x)
        return float(mu[0]), float(ron[0])

    # ── 永続化 ──────────────────────────────────────────────────────────────
    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[OK] saved -> {path}")

    @staticmethod
    def load(path: str = MODEL_PATH):
        return joblib.load(path)


# ════════════════════════════════════════════════════════════════════════════
# 検証 (hold-out + k-fold)
# ════════════════════════════════════════════════════════════════════════════

def _metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return rmse, r2


def holdout_eval(X, Y, test_size=0.2, seed=0):
    """80/20 hold-out。テスト集合の真値と予測を両ターゲットについて返す。"""
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=test_size,
                                           random_state=seed)
    sur = FeolSurrogate().fit_xy(Xtr, Ytr)
    mu_hat, ron_hat = sur.predict_batch(Xte)
    res = {}
    res["mu_ch"]   = (Yte[:, 0], mu_hat,  *_metrics(Yte[:, 0], mu_hat))
    res["R_on_sp"] = (Yte[:, 1], ron_hat, *_metrics(Yte[:, 1], ron_hat))
    return res


def kfold_eval(X, Y, n_splits=5, seed=0):
    """k-fold CV。out-of-fold 予測を集計して RMSE / R² を報告する。"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros_like(Y)
    for tr, te in kf.split(X):
        sur = FeolSurrogate().fit_xy(X[tr], Y[tr])
        mu_hat, ron_hat = sur.predict_batch(X[te])
        oof[te, 0] = mu_hat
        oof[te, 1] = ron_hat
    out = {}
    out["mu_ch"]   = _metrics(Y[:, 0], oof[:, 0])
    out["R_on_sp"] = _metrics(Y[:, 1], oof[:, 1])
    return out, oof


# ════════════════════════════════════════════════════════════════════════════
# 図: parity plots
# ════════════════════════════════════════════════════════════════════════════

def plot_parity(holdout, kfold_r2, out_path=FIG_PATH):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    panels = [
        ("mu_ch",   "µ_ch  [cm²/Vs]",   "#2166ac"),
        ("R_on_sp", "R_on,sp  [mΩ·mm²]", "#d62728"),
    ]
    for ax, (key, label, color) in zip(axes, panels):
        y_true, y_pred, rmse, r2 = holdout[key]
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        pad = 0.05 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.7, label="ideal")
        ax.scatter(y_true, y_pred, s=55, color=color, edgecolors="k",
                   lw=0.6, alpha=0.85, zorder=5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", "box")
        ax.set_xlabel(f"true {label}", fontsize=11)
        ax.set_ylabel(f"predicted {label}", fontsize=11)
        ax.set_title(
            f"{key}  (hold-out 20%)\n"
            f"R² = {r2:.3f}   RMSE = {rmse:.3g}   "
            f"CV R² = {kfold_r2[key]:.3f}",
            fontsize=10,
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        "TEL FEOL GP Surrogate — Parity (predicted vs true)\n"
        "recipe → MOSFET FOM | cheap stand-in for full-physics BO/APC calls",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved -> {out_path}")
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="TEL FEOL GP surrogate (recipe -> mu_ch & R_on_sp)")
    ap.add_argument("--n", type=int, default=120,
                    help="サンプリングする recipe 数 (default 120)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=2,
                    help="k-fold CV の分割数 (default 2; 60s 制約のため。"
                         "精度を上げたければ --folds 5)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print(f"[*] サンプリング N={args.n} (seed={args.seed}) ...")
    X, Y, recipes = build_dataset(args.n, seed=args.seed)
    print(f"    mu_ch   range: {Y[:,0].min():.1f} .. {Y[:,0].max():.1f} cm²/Vs")
    print(f"    R_on_sp range: {Y[:,1].min():.3f} .. {Y[:,1].max():.3f} mΩ·mm²")

    # ── 検証 ────────────────────────────────────────────────────────────────
    print(f"\n[*] hold-out 80/20 検証 ...")
    holdout = holdout_eval(X, Y, test_size=0.2, seed=args.seed)
    for key in ("mu_ch", "R_on_sp"):
        _, _, rmse, r2 = holdout[key]
        print(f"    {key:9}  hold-out  R²={r2:6.3f}  RMSE={rmse:.4g}")

    print(f"\n[*] {args.folds}-fold CV ...")
    kfold, _ = kfold_eval(X, Y, n_splits=args.folds, seed=args.seed)
    kfold_r2 = {}
    for key in ("mu_ch", "R_on_sp"):
        rmse, r2 = kfold[key]
        kfold_r2[key] = r2
        print(f"    {key:9}  {args.folds}-fold  R²={r2:6.3f}  RMSE={rmse:.4g}")

    # ── 全データで最終モデルを学習し保存 ─────────────────────────────────────
    print(f"\n[*] 全 {args.n} 点で最終代理を学習 → 保存 ...")
    surrogate = FeolSurrogate().fit_xy(X, Y)
    surrogate.X_train_ = X       # 参考: 学習データを同梱 (BO の warm-start 用)
    surrogate.Y_train_ = Y
    surrogate.save(MODEL_PATH)
    print(f"    learned kernel (mu_ch) : {surrogate.gp_mu.kernel_}")

    # ── 健全性チェック: default recipe を予測 ─────────────────────────────────
    rec = default_recipe()
    fom = simulate_recipe(rec)
    mu_hat, ron_hat, std = surrogate.predict(rec, return_std=True)
    print(f"\n[*] default recipe 健全性チェック (true vs surrogate):")
    print(f"    mu_ch   true={fom['mu_ch']:7.2f}  pred={mu_hat:7.2f} ±{std[0]:.2f}")
    print(f"    R_on_sp true={fom['R_on_sp']:7.4f}  pred={ron_hat:7.4f} ±{std[1]:.4f}")

    # ── 図 ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_parity(holdout, kfold_r2, FIG_PATH)

    print(f"\n[=] 達成 R²  "
          f"mu_ch: hold-out {holdout['mu_ch'][3]:.3f} / CV {kfold_r2['mu_ch']:.3f}   |   "
          f"R_on_sp: hold-out {holdout['R_on_sp'][3]:.3f} / CV {kfold_r2['R_on_sp']:.3f}")


if __name__ == "__main__":
    main()
