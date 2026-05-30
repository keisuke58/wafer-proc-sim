"""
動的レシピ補正モジュール

「現在の加工条件 × 観測された出力」から、目標値を達成するための
最小限のパラメータ修正量を計算する。

アプローチ:
  目標: minimize ||x_new - x_current||²
  制約: GP_pred(x_new) ≤ target_values
  手法: 制約付き最適化 (scipy.optimize.minimize + GP)

Usage:
    from optimization.recipe_correction import RecipeCorrector

    corrector = RecipeCorrector()
    corrector.load_model()   # 既存GP モデルを読み込む

    correction = corrector.correct(
        x_current = np.array([30.0, 25.0]),   # [cut_depth_um, blade_W_um]
        y_observed = np.array([0.08, 2.1e9]), # [chipping, stress]
        targets    = {"deletion_fraction": 0.03, "max_principal_stress_Pa": 1.5e9},
    )
    print(correction.x_new)       # 補正後の条件
    print(correction.delta)       # 変化量
    print(correction.confidence)  # 信頼度
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import minimize, differential_evolution

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.surrogate_gp import DicingGPSurrogate, FEATURE_COLS, TARGET_COLS, MODEL_PATH


# ── データクラス ───────────────────────────────────────────────────────────────

@dataclass
class CorrectionResult:
    x_current:  np.ndarray          # 現在の条件
    x_new:      np.ndarray          # 補正後の条件
    delta:      np.ndarray          # x_new - x_current
    delta_pct:  np.ndarray          # 変化率 [%]
    y_pred_new: np.ndarray          # 補正後の予測出力
    y_pred_std: np.ndarray          # 予測不確かさ
    confidence: float               # 0–1（補正の信頼度）
    converged:  bool
    message:    str
    suggestions: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# RecipeCorrector
# ═══════════════════════════════════════════════════════════════════════════════

class RecipeCorrector:
    """
    GPサロゲートを使った最小変化量レシピ補正。

    目的関数:
        f(x) = ||x - x_current||² / ||x_scale||²   (正規化 L2 距離)

    制約:
        GP_mean(x)[i] ≤ target[i]  for each target

    パラメータ範囲は BOUNDS (surrogate_gp.py と同じ) を使用。
    """

    BOUNDS = {
        "cut_depth_um": (20.0, 360.0),  # 実データ範囲に合わせる
        "blade_W_um":   (20.0, 40.0),
    }

    def __init__(self):
        self.gp: Optional[DicingGPSurrogate] = None
        self.bounds_list = [self.BOUNDS[k] for k in FEATURE_COLS]
        self.x_scale = np.array([b[1] - b[0] for b in self.bounds_list])

    def load_model(self, path: str = MODEL_PATH):
        if os.path.exists(path):
            self.gp = DicingGPSurrogate.load(path)
        else:
            raise FileNotFoundError(
                f"GP model not found: {path}\n"
                "surrogate_gp.py で先にモデルを学習してください。"
            )

    def correct(self,
                x_current:  np.ndarray,
                y_observed: Optional[np.ndarray] = None,
                targets:    Optional[dict] = None,
                max_delta_pct: float = 30.0,
                method: str = "local") -> CorrectionResult:
        """
        x_current: (n_features,) 現在の加工条件
        y_observed: (n_targets,) 実測出力（ログ用、最適化には使用しない）
        targets: {"deletion_fraction": 0.03, "max_principal_stress_Pa": 1.5e9}
        max_delta_pct: 各パラメータの最大変化率 [%]
        method: "local" (高速) または "global" (より確実だが遅い)
        """
        if self.gp is None:
            self.load_model()

        x0 = np.asarray(x_current, dtype=float)

        # デフォルトターゲット: 全出力を現在予測値の 80% に
        if targets is None:
            mu0, _ = self.gp.predict(x0.reshape(1, -1), return_std=True)
            targets = {col: float(mu0[0, i]) * 0.8
                       for i, col in enumerate(TARGET_COLS)}

        target_vec = np.array([targets.get(col, np.inf) for col in TARGET_COLS])

        # パラメータ探索範囲を x_current ± max_delta_pct% に制限
        search_bounds = []
        for i, (lo, hi) in enumerate(self.bounds_list):
            delta_abs = x0[i] * max_delta_pct / 100.0
            sb_lo = max(lo, x0[i] - delta_abs)
            sb_hi = min(hi, x0[i] + delta_abs)
            search_bounds.append((sb_lo, sb_hi))

        # 目的関数: x_current からの正規化 L2 距離
        def objective(x):
            diff = (x - x0) / self.x_scale
            return float(np.dot(diff, diff))

        # 制約: GP予測 ≤ target
        def constraint_i(x, i):
            mu, _ = self.gp.predict(x.reshape(1, -1), return_std=True)
            return float(target_vec[i] - mu[0, i])   # ≥ 0 → feasible

        constraints = [
            {"type": "ineq", "fun": constraint_i, "args": (i,)}
            for i in range(len(TARGET_COLS))
            if not np.isinf(target_vec[i])
        ]

        # 最適化
        if method == "global":
            # Differential evolution (グローバル探索、制約なし→ペナルティ法)
            def penalized(x):
                obj = objective(x)
                for i in range(len(TARGET_COLS)):
                    if not np.isinf(target_vec[i]):
                        mu, _ = self.gp.predict(x.reshape(1, -1), return_std=True)
                        viol = mu[0, i] - target_vec[i]
                        if viol > 0:
                            obj += 100.0 * viol ** 2
                return obj
            res = differential_evolution(
                penalized, bounds=search_bounds,
                seed=42, maxiter=200, tol=1e-6, workers=1, popsize=10
            )
        else:
            res = minimize(
                objective, x0=x0,
                method="SLSQP",
                bounds=search_bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-8},
            )

        x_new = np.clip(res.x, [b[0] for b in self.bounds_list],
                                [b[1] for b in self.bounds_list])

        # 補正後の予測
        mu_new, sigma_new = self.gp.predict(x_new.reshape(1, -1), return_std=True)
        mu_new    = mu_new[0]
        sigma_new = sigma_new[0]

        delta     = x_new - x0
        delta_pct = delta / np.maximum(np.abs(x0), 1e-9) * 100.0

        # 信頼度: 制約を満たしているか + 予測不確かさが小さいか
        feasible = all(
            mu_new[i] <= target_vec[i] * 1.05 or np.isinf(target_vec[i])
            for i in range(len(TARGET_COLS))
        )
        uncertainty_ok = all(sigma_new / np.maximum(np.abs(mu_new), 1e-9) < 0.3)
        confidence = (0.7 if feasible else 0.3) + (0.3 if uncertainty_ok else 0.0)

        # 提案文
        suggestions = []
        for i, col in enumerate(FEATURE_COLS):
            d = delta_pct[i]
            if abs(d) > 1.0:
                direction = "増加" if d > 0 else "減少"
                suggestions.append(
                    f"{col}: {x0[i]:.1f} → {x_new[i]:.1f} ({d:+.1f}%{direction})"
                )
        if not suggestions:
            suggestions.append("現在条件は目標を既に満たしています（補正不要）")

        return CorrectionResult(
            x_current  = x0,
            x_new      = x_new,
            delta      = delta,
            delta_pct  = delta_pct,
            y_pred_new = mu_new,
            y_pred_std = sigma_new,
            confidence = confidence,
            converged  = bool(res.success),
            message    = res.message,
            suggestions = suggestions,
        )

    def batch_correct(self, df: pd.DataFrame,
                      targets: dict) -> pd.DataFrame:
        """
        DataFrame の各行を補正。
        df 列: FEATURE_COLS + TARGET_COLS
        """
        records = []
        for _, row in df.iterrows():
            x_cur = row[FEATURE_COLS].values.astype(float)
            y_obs = row[TARGET_COLS].values.astype(float)
            result = self.correct(x_cur, y_obs, targets)
            rec = {f"x_orig_{c}":  v for c, v in zip(FEATURE_COLS, x_cur)}
            rec.update({f"x_new_{c}":   v for c, v in zip(FEATURE_COLS, result.x_new)})
            rec.update({f"delta_pct_{c}": v for c, v in zip(FEATURE_COLS, result.delta_pct)})
            rec.update({f"y_pred_{c}":  v for c, v in zip(TARGET_COLS, result.y_pred_new)})
            rec["confidence"] = result.confidence
            rec["converged"]  = result.converged
            records.append(rec)
        return pd.DataFrame(records)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="動的レシピ補正")
    parser.add_argument("--csv",   required=True,
                        help="parametric_summary.csv のパス")
    parser.add_argument("--x",     nargs="+", type=float,
                        help=f"現在の加工条件 ({', '.join(FEATURE_COLS)})")
    parser.add_argument("--target-chipping", type=float, default=0.03,
                        help="目標チッピング率 (default: 0.03)")
    parser.add_argument("--method", choices=["local", "global"],
                        default="local")
    args = parser.parse_args()

    corrector = RecipeCorrector()
    corrector.load_model()

    if args.x:
        x_cur = np.array(args.x)
        targets = {"deletion_fraction": args.target_chipping}
        result = corrector.correct(x_cur, targets=targets, method=args.method)

        print(f"\n=== レシピ補正結果 ===")
        print(f"  収束: {'✓' if result.converged else '✗'}")
        print(f"  信頼度: {result.confidence:.2f}")
        print(f"\n  補正提案:")
        for s in result.suggestions:
            print(f"    {s}")
        print(f"\n  補正後予測:")
        for col, mu, sig in zip(TARGET_COLS, result.y_pred_new, result.y_pred_std):
            print(f"    {col}: {mu:.4g} ± {sig:.4g}")
    else:
        # バッチ補正
        df = pd.read_csv(args.csv)
        targets = {"deletion_fraction": args.target_chipping}
        result_df = corrector.batch_correct(df, targets)
        out_path = args.csv.replace(".csv", "_corrected.csv")
        result_df.to_csv(out_path, index=False)
        print(f"[✓] バッチ補正結果 → {out_path}")
        print(f"  平均信頼度: {result_df['confidence'].mean():.2f}")
        print(f"  収束率: {result_df['converged'].mean():.1%}")


if __name__ == "__main__":
    main()
