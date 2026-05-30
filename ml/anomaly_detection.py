"""
不良検出・異常予兆モジュール

3層の検出アプローチ:
  Layer 1 — GP不確かさ:  |actual - GP_mean| / GP_std > threshold
  Layer 2 — Isolation Forest: FEMデータで学習した密度ベース検出
  Layer 3 — 管理図 (Shewhart): プロセスドリフトの時系列監視

Usage:
    from ml.anomaly_detection import AnomalyDetector, ProcessMonitor

    # 学習
    detector = AnomalyDetector()
    detector.fit(X_train, Y_train)          # X: params, Y: [chipping, stress]

    # 新規点の検査
    result = detector.check(x_new, y_observed)
    if result.is_anomaly:
        print(result.cause, result.severity)

    # プロセス監視（時系列）
    monitor = ProcessMonitor(target_col="deletion_fraction", usl=0.05)
    for y in production_stream:
        alert = monitor.update(y)
"""

import os
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# GP surrogate をインポート（既存モジュール）
import sys
sys.path.insert(0, os.path.dirname(__file__))
from surrogate_gp import DicingGPSurrogate, FEATURE_COLS, TARGET_COLS, MODEL_PATH

ANOMALY_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "anomaly_model.pkl"
)


# ── データクラス ───────────────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    is_anomaly:  bool
    severity:    float          # 0.0（正常）〜 1.0（重大異常）
    layer_flags: dict           # {'gp': bool, 'iforest': bool, 'spc': bool}
    cause:       str            # 原因の説明文
    suggestions: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: GP 不確かさベース検出
# ═══════════════════════════════════════════════════════════════════════════════

class GPAnomalyLayer:
    """
    GPサロゲートの予測値と実測値の乖離を z スコアで評価する。
    |y_obs - mu| / sigma > z_thresh → 異常
    """

    def __init__(self, z_thresh: float = 3.0):
        self.z_thresh = z_thresh
        self.gp: Optional[DicingGPSurrogate] = None

    def load_gp(self):
        if os.path.exists(MODEL_PATH):
            self.gp = DicingGPSurrogate.load()
        else:
            raise FileNotFoundError(
                f"GP model not found: {MODEL_PATH}\n"
                "先に surrogate_gp.py を実行してモデルを学習してください。"
            )

    def check(self, x: np.ndarray, y_obs: np.ndarray) -> tuple[bool, float, str]:
        """
        x: (1, n_features), y_obs: (n_targets,)
        Returns (is_anomaly, max_z, description)
        """
        if self.gp is None:
            self.load_gp()

        mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
        mu     = mu[0]
        sigma  = sigma[0]

        z_scores = np.abs(y_obs - mu) / np.maximum(sigma, 1e-9)
        max_z    = float(z_scores.max())
        worst_i  = int(z_scores.argmax())
        is_anom  = max_z > self.z_thresh

        col_name = TARGET_COLS[worst_i] if worst_i < len(TARGET_COLS) else f"target_{worst_i}"
        desc = (
            f"GP z-score {max_z:.2f} > {self.z_thresh} on {col_name}: "
            f"observed={y_obs[worst_i]:.4g}, predicted={mu[worst_i]:.4g}±{sigma[worst_i]:.4g}"
            if is_anom else "GP: 正常範囲内"
        )
        return is_anom, max_z / self.z_thresh, desc


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2: Isolation Forest（密度ベース外れ値検出）
# ═══════════════════════════════════════════════════════════════════════════════

class IForestLayer:
    """
    FEMパラメトリックスタディ結果で学習した Isolation Forest。
    入力空間での外れ値を検出する（物理的に有り得ない条件を弾く）。
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.iforest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        )
        self.scaler  = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """X: (N, n_features)"""
        Xs = self.scaler.fit_transform(X)
        self.iforest.fit(Xs)
        self.is_fitted = True
        return self

    def check(self, x: np.ndarray) -> tuple[bool, float, str]:
        """x: (1, n_features)"""
        if not self.is_fitted:
            return False, 0.0, "IForest: 未学習（スキップ）"
        Xs    = self.scaler.transform(x.reshape(1, -1))
        pred  = self.iforest.predict(Xs)[0]          # 1=normal, -1=outlier
        score = -self.iforest.score_samples(Xs)[0]   # 高いほど異常
        is_anom = (pred == -1)
        desc = (
            f"IForest: 外れ値検出（anomaly score={score:.3f}）"
            if is_anom else f"IForest: 正常（score={score:.3f}）"
        )
        return is_anom, float(np.clip(score, 0, 1)), desc


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3: 管理図（Shewhart X-bar / Individual）
# ═══════════════════════════════════════════════════════════════════════════════

class ShewhartChart:
    """
    個値管理図 (Individuals Chart, I-chart)。
    ウォームアップ期間のデータから UCL/LCL を自動推定。
    ルール: 1点が3σ外, 8点連続して中心線の片側。
    """

    def __init__(self, warmup: int = 20, sigma_mult: float = 3.0):
        self.warmup     = warmup
        self.sigma_mult = sigma_mult
        self.history: list[float] = []
        self.ucl: Optional[float] = None
        self.lcl: Optional[float] = None
        self.cl:  Optional[float] = None

    def update(self, value: float) -> tuple[bool, str]:
        self.history.append(value)

        if len(self.history) == self.warmup:
            arr        = np.array(self.history)
            self.cl    = float(arr.mean())
            # MR (Moving Range) ベースの sigma 推定
            mr         = np.abs(np.diff(arr))
            sigma_est  = float(mr.mean()) / 1.128
            self.ucl   = self.cl + self.sigma_mult * sigma_est
            self.lcl   = self.cl - self.sigma_mult * sigma_est

        if self.ucl is None:
            return False, "管理図: ウォームアップ中"

        # ルール1: 1点が UCL/LCL 外
        if value > self.ucl:
            return True, f"管理図: UCL={self.ucl:.4g} 超過（{value:.4g}）"
        if value < self.lcl:
            return True, f"管理図: LCL={self.lcl:.4g} 下回り（{value:.4g}）"

        # ルール2: 8点連続して中心線の片側
        if len(self.history) >= 8:
            last8 = self.history[-8:]
            if all(v > self.cl for v in last8):
                return True, "管理図: 8点連続して中心線上側（ドリフト）"
            if all(v < self.cl for v in last8):
                return True, "管理図: 8点連続して中心線下側（ドリフト）"

        return False, "管理図: 管理状態"


# ═══════════════════════════════════════════════════════════════════════════════
# 統合 AnomalyDetector
# ═══════════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """3層の検出器を統合したクラス"""

    def __init__(self,
                 z_thresh: float = 3.0,
                 contamination: float = 0.05):
        self.gp_layer  = GPAnomalyLayer(z_thresh=z_thresh)
        self.if_layer  = IForestLayer(contamination=contamination)
        self.is_fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        X: (N, n_features) — process parameters
        Y: (N, n_targets)  — FEM outputs
        """
        # IForest は X（入力パラメータ空間）で学習
        self.if_layer.fit(X)
        # GP は surrogate_gp.py で別途学習済みを前提（load で取得）
        try:
            self.gp_layer.load_gp()
        except FileNotFoundError:
            print("[Warn] GPモデル未発見。Layer1はスキップされます。")
        self.is_fitted = True
        return self

    def check(self, x: np.ndarray,
              y_obs: Optional[np.ndarray] = None) -> AnomalyResult:
        """
        x     : (n_features,) 処理条件
        y_obs : (n_targets,)  実測結果（任意）
        """
        x = np.asarray(x, dtype=float).reshape(1, -1)
        flags = {}
        scores = []
        descs  = []

        # Layer 2: IForest（y_obs 不要）
        anom2, score2, desc2 = self.if_layer.check(x)
        flags["iforest"] = anom2
        scores.append(score2)
        descs.append(desc2)

        # Layer 1: GP（y_obs が必要）
        if y_obs is not None:
            try:
                y_obs_arr = np.asarray(y_obs, dtype=float).reshape(-1)
                anom1, score1, desc1 = self.gp_layer.check(x, y_obs_arr)
                flags["gp"] = anom1
                scores.append(score1)
                descs.append(desc1)
            except Exception as e:
                flags["gp"] = False
                descs.append(f"GP: スキップ ({e})")
        else:
            flags["gp"] = False

        flags["spc"] = False  # SPC は ProcessMonitor で管理

        # 総合判定：いずれかの層で異常 → 異常
        is_anom   = any(flags.values())
        severity  = float(np.max(scores)) if scores else 0.0
        cause     = " | ".join(descs)

        # 対処提案
        suggestions = []
        if flags.get("gp"):
            suggestions.append("測定値とモデル予測の乖離が大きい — キャリブレーション実施を推奨")
        if flags.get("iforest"):
            suggestions.append("加工条件が通常の範囲外 — run_config.json のパラメータを確認")
        if not suggestions and is_anom:
            suggestions.append("プロセス状態を確認し、直近の加工条件変更を記録してください")

        return AnomalyResult(
            is_anomaly  = is_anom,
            severity    = severity,
            layer_flags = flags,
            cause       = cause,
            suggestions = suggestions,
        )

    def save(self, path: str = ANOMALY_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[✓] AnomalyDetector saved → {path}")

    @staticmethod
    def load(path: str = ANOMALY_MODEL_PATH) -> "AnomalyDetector":
        model = joblib.load(path)
        print(f"[✓] AnomalyDetector loaded ← {path}")
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# ProcessMonitor: 時系列監視（プロダクション用）
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessMonitor:
    """
    プロダクション投入時の連続監視。
    各 lot の結果を受け取り、管理図で異常を検知する。
    """

    def __init__(self, target_col: str = "deletion_fraction",
                 usl: float = 0.05, lsl: float = 0.0,
                 warmup: int = 20):
        self.target_col = target_col
        self.usl        = usl
        self.lsl        = lsl
        self.chart      = ShewhartChart(warmup=warmup)
        self.log: list[dict] = []

    def update(self, value: float, lot_id: str = "") -> Optional[AnomalyResult]:
        """
        新しい lot 結果を投入。異常があれば AnomalyResult を返す。
        """
        # USL/LSL ハードチェック
        if value > self.usl:
            result = AnomalyResult(
                is_anomaly  = True,
                severity    = 1.0,
                layer_flags = {"spc": True, "gp": False, "iforest": False},
                cause       = f"USL={self.usl} 超過: {self.target_col}={value:.4g}",
                suggestions = ["即時停止・刃の交換またはパラメータ再調整を推奨"],
            )
        else:
            is_anom, desc = self.chart.update(value)
            result = AnomalyResult(
                is_anomaly  = is_anom,
                severity    = 0.5 if is_anom else 0.0,
                layer_flags = {"spc": is_anom, "gp": False, "iforest": False},
                cause       = desc,
                suggestions = ["管理図ドリフト — プロセス安定化の確認"] if is_anom else [],
            ) if is_anom else None

        self.log.append({
            "lot_id": lot_id,
            "value":  value,
            "anomaly": result.is_anomaly if result else False,
        })
        return result

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="不良検出・異常予兆")
    parser.add_argument("--csv",     required=True,
                        help="parametric_summary.csv のパス")
    parser.add_argument("--check",   nargs="+", type=float,
                        help="単一条件を検査: cut_depth_um blade_W_um [chipping stress]")
    parser.add_argument("--monitor", action="store_true",
                        help="全 CSV 行を時系列監視として流す")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    from surrogate_gp import FEATURE_COLS, TARGET_COLS

    # CSVに存在する列のみ使用（列名が異なる場合の互換処理）
    available_targets = [c for c in TARGET_COLS if c in df.columns]
    if not available_targets:
        # フォールバック: 数値列でFEATURE_COLS以外の最初の列を使用
        num_cols = df.select_dtypes(include="number").columns.tolist()
        available_targets = [c for c in num_cols
                             if c not in FEATURE_COLS][:2]
        print(f"[Warn] TARGET_COLS {TARGET_COLS} が見つからない。"
              f"代替: {available_targets}")
    X = df[FEATURE_COLS].values.astype(float)
    Y = df[available_targets].values.astype(float)

    detector = AnomalyDetector()
    detector.fit(X, Y)
    detector.save()
    print(f"[✓] AnomalyDetector 学習完了 ({len(X)} サンプル)")

    if args.check:
        vals = args.check
        x_new = np.array(vals[:len(FEATURE_COLS)])
        y_obs = np.array(vals[len(FEATURE_COLS):]) if len(vals) > len(FEATURE_COLS) else None
        result = detector.check(x_new, y_obs)
        print(f"\n=== 異常検査結果 ===")
        print(f"  異常: {'⚠ YES' if result.is_anomaly else '✓ NO'}")
        print(f"  深刻度: {result.severity:.2f}")
        print(f"  原因: {result.cause}")
        for s in result.suggestions:
            print(f"  → {s}")

    if args.monitor:
        monitor = ProcessMonitor(target_col=TARGET_COLS[0], usl=0.10, warmup=10)
        print(f"\n=== 時系列監視 ({TARGET_COLS[0]}) ===")
        for i, val in enumerate(Y[:, 0]):
            alert = monitor.update(val, lot_id=f"lot_{i:03d}")
            if alert:
                print(f"  lot_{i:03d}: ⚠ {alert.cause}")
        summary = monitor.summary()
        anomaly_rate = summary["anomaly"].mean()
        print(f"\n  異常率: {anomaly_rate:.1%} ({summary['anomaly'].sum()}/{len(summary)}件)")


if __name__ == "__main__":
    main()
