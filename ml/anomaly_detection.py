"""
不良検出・異常予兆モジュール

3層の検出アプローチ:
  Layer 1 — GP不確かさ:   |actual_chip - GP_mean| / GP_std > z_thresh
             実験GP (gp_experimental.pkl) でチッピング[µm]を予測
  Layer 2 — Isolation Forest: 入力パラメータ空間での外れ値検出
  Layer 3 — Shewhart管理図:   プロセスドリフトの時系列監視

Usage:
    # 単一点の検査 (depth bw feed spindle [observed_chip_um])
    python ml/anomaly_detection.py --csv results/parametric_summary_extended.csv \\
        --check 80 23 1.0 30000 2.5

    # プロセス監視 (CSV全行を時系列として流す)
    python ml/anomaly_detection.py --csv results/parametric_summary_extended.csv \\
        --monitor

    # デモ: 正常→異常→ドリフトのシナリオ
    python ml/anomaly_detection.py --demo

    # 実データ検証: SECOM (UCI 半導体工場センサ) で Layer2/3 を評価
    python ml/anomaly_detection.py --secom
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.train_from_experimental import (
    ExperimentalGPSurrogate, FEATURE_COLS, MODEL_PATH as EXP_GP_PATH,
)
from ml.surrogate_gp import resolve_target_cols, FEATURE_COLS as FEM_FEATURES

ANOMALY_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "anomaly_model.pkl"
)
CHIP_USL = 15.0   # µm — 生産閾値


# ── データクラス ───────────────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    is_anomaly:  bool
    severity:    float          # 0.0（正常）〜 1.0（重大）
    layer_flags: dict
    cause:       str
    suggestions: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: 実験GP不確かさベース検出（チッピング[µm]）
# ═══════════════════════════════════════════════════════════════════════════════

class GPAnomalyLayer:
    """
    実験GP (chipping_um) の予測値と観測値の z スコアで異常判定。
    入力: 4特徴量 [cut_depth_um, blade_W_um, feed_mm_s, spindle_rpm]
    観測: observed_chip_um [µm]
    """

    def __init__(self, z_thresh: float = 3.0):
        self.z_thresh = z_thresh
        self.gp: Optional[ExperimentalGPSurrogate] = None

    def load(self):
        if os.path.exists(EXP_GP_PATH):
            self.gp = ExperimentalGPSurrogate.load(EXP_GP_PATH)
        else:
            raise FileNotFoundError(
                f"実験GPモデル未発見: {EXP_GP_PATH}\n"
                "python ml/train_from_experimental.py を先に実行してください。"
            )

    def check(self, x4: np.ndarray,
              observed_chip_um: float) -> tuple[bool, float, str]:
        """
        x4: (4,) [cut_depth_um, blade_W_um, feed_mm_s, spindle_rpm]
        observed_chip_um: 実測チッピング [µm]
        """
        if self.gp is None:
            self.load()

        mu, sigma = self.gp.predict(x4.reshape(1, -1), return_std=True)
        mu, sigma = float(mu[0]), float(sigma[0])
        z = abs(observed_chip_um - mu) / max(sigma, 0.1)
        is_anom = z > self.z_thresh

        desc = (
            f"GP z={z:.2f} > {self.z_thresh}: "
            f"observed={observed_chip_um:.1f}µm, predicted={mu:.1f}±{sigma:.1f}µm"
            if is_anom else
            f"GP 正常 (z={z:.2f}, pred={mu:.1f}±{sigma:.1f}µm)"
        )
        return is_anom, min(z / self.z_thresh, 5.0), desc


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2: Isolation Forest（入力パラメータ空間）
# ═══════════════════════════════════════════════════════════════════════════════

class IForestLayer:
    """
    FEMスタディのパラメータ範囲で学習した Isolation Forest。
    物理的に有り得ない条件（範囲外）を弾く。
    """

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.iforest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        )
        self.scaler    = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        self.iforest.fit(Xs)
        self.is_fitted = True
        return self

    def check(self, x: np.ndarray) -> tuple[bool, float, str]:
        if not self.is_fitted:
            return False, 0.0, "IForest: 未学習（スキップ）"
        Xs    = self.scaler.transform(x.reshape(1, -1))
        pred  = self.iforest.predict(Xs)[0]      # 1=正常, -1=外れ値
        score = float(-self.iforest.score_samples(Xs)[0])
        is_anom = (pred == -1)
        desc = (
            f"IForest 外れ値 (score={score:.3f})"
            if is_anom else
            f"IForest 正常 (score={score:.3f})"
        )
        return is_anom, float(np.clip(score, 0, 1)), desc


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3: Shewhart 管理図（時系列監視）
# ═══════════════════════════════════════════════════════════════════════════════

class ShewhartChart:
    """個値管理図 (I-chart): ウォームアップ後に UCL/LCL を自動設定"""

    def __init__(self, warmup: int = 20, sigma_mult: float = 3.0):
        self.warmup     = warmup
        self.sigma_mult = sigma_mult
        self.history: list = []
        self.ucl = self.lcl = self.cl = None

    def update(self, value: float) -> tuple[bool, str]:
        self.history.append(value)

        if len(self.history) == self.warmup:
            arr       = np.array(self.history)
            self.cl   = float(arr.mean())
            mr        = np.abs(np.diff(arr))
            sigma_est = float(mr.mean()) / 1.128
            self.ucl  = self.cl + self.sigma_mult * sigma_est
            self.lcl  = self.cl - self.sigma_mult * sigma_est

        if self.ucl is None:
            return False, f"管理図: ウォームアップ中 ({len(self.history)}/{self.warmup})"

        if value > self.ucl:
            return True, f"管理図 UCL={self.ucl:.2f} 超過 ({value:.2f})"
        if value < self.lcl:
            return True, f"管理図 LCL={self.lcl:.2f} 下回り ({value:.2f})"
        if len(self.history) >= 8:
            last8 = self.history[-8:]
            if all(v > self.cl for v in last8):
                return True, "管理図: 8点連続して中心線上側（上昇ドリフト）"
            if all(v < self.cl for v in last8):
                return True, "管理図: 8点連続して中心線下側（下降ドリフト）"

        return False, f"管理図: 管理状態 (CL={self.cl:.2f})"


# ═══════════════════════════════════════════════════════════════════════════════
# 統合 AnomalyDetector
# ═══════════════════════════════════════════════════════════════════════════════

class AnomalyDetector:

    def __init__(self, z_thresh: float = 3.0, contamination: float = 0.1):
        self.gp_layer  = GPAnomalyLayer(z_thresh=z_thresh)
        self.if_layer  = IForestLayer(contamination=contamination)
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """X: (N, 2) [cut_depth_um, blade_W_um] — FEM パラメータ範囲"""
        self.if_layer.fit(X)
        try:
            self.gp_layer.load()
        except FileNotFoundError as e:
            print(f"[Warn] {e}")
        self.is_fitted = True
        return self

    def check(self, x4: np.ndarray,
              observed_chip_um: Optional[float] = None) -> AnomalyResult:
        """
        x4             : [cut_depth_um, blade_W_um, feed_mm_s, spindle_rpm]
        observed_chip_um: 実測チッピング [µm]（任意）
        """
        x4 = np.asarray(x4, dtype=float).ravel()
        x2 = x4[:2]   # IForest は depth + blade_W のみ
        flags, scores, descs = {}, [], []

        # Layer 2: IForest
        a2, s2, d2 = self.if_layer.check(x2)
        flags["iforest"] = a2; scores.append(s2); descs.append(d2)

        # Layer 1: GP
        if observed_chip_um is not None and self.gp_layer.gp is not None:
            a1, s1, d1 = self.gp_layer.check(x4, observed_chip_um)
            flags["gp"] = a1; scores.append(s1); descs.append(d1)

            # 硬判定: USL 超過
            if observed_chip_um > CHIP_USL:
                flags["usl"] = True
                descs.append(f"USL={CHIP_USL}µm 超過: {observed_chip_um:.1f}µm")
                scores.append(1.0)
        else:
            flags["gp"] = False

        flags["spc"] = False

        is_anom  = any(flags.values())
        severity = float(np.max(scores)) if scores else 0.0
        cause    = " | ".join(descs)

        suggestions = []
        if flags.get("usl"):
            suggestions.append("即時停止 — チッピングが生産閾値超過。刃の交換を推奨")
        if flags.get("gp") and not flags.get("usl"):
            suggestions.append("GPモデルとの乖離大 — センサー校正またはGP再学習を検討")
        if flags.get("iforest"):
            suggestions.append("パラメータが学習範囲外 — run_config.json を確認")

        return AnomalyResult(
            is_anomaly=is_anom, severity=severity,
            layer_flags=flags, cause=cause, suggestions=suggestions,
        )

    def save(self, path=ANOMALY_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[✓] AnomalyDetector → {path}")

    @staticmethod
    def load(path=ANOMALY_MODEL_PATH):
        return joblib.load(path)


# ═══════════════════════════════════════════════════════════════════════════════
# ProcessMonitor: プロダクション時系列監視
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessMonitor:

    def __init__(self, usl=CHIP_USL, warmup=15):
        self.usl   = usl
        self.chart = ShewhartChart(warmup=warmup)
        self.log: list = []

    def update(self, chip_um: float, lot_id: str = "") -> Optional[AnomalyResult]:
        if chip_um > self.usl:
            result = AnomalyResult(
                is_anomaly=True, severity=1.0,
                layer_flags={"spc": True},
                cause=f"USL={self.usl}µm 超過: chip={chip_um:.1f}µm",
                suggestions=["即時停止 — 刃の交換またはパラメータ再調整を推奨"],
            )
        else:
            is_anom, desc = self.chart.update(chip_um)
            result = AnomalyResult(
                is_anomaly=is_anom, severity=0.5 if is_anom else 0.0,
                layer_flags={"spc": is_anom},
                cause=desc,
                suggestions=["管理図ドリフト — プロセス安定化の確認"] if is_anom else [],
            ) if is_anom else None

        self.log.append({"lot_id": lot_id, "chip_um": chip_um,
                         "anomaly": result.is_anomaly if result else False})
        return result

    def summary(self):
        return pd.DataFrame(self.log)


# ── デモ ──────────────────────────────────────────────────────────────────────

def run_demo():
    """正常 → 徐々に悪化 → USL超過 のシナリオデモ"""
    import numpy as np
    rng = np.random.default_rng(42)

    detector = AnomalyDetector()
    detector.fit(np.array([[80., 23.], [150., 23.], [220., 23.],
                            [290., 23.], [360., 23.]]))

    monitor = ProcessMonitor(usl=CHIP_USL, warmup=10)
    print("\n=== デモ: 正常運転 → 刃摩耗 → 異常 ===")
    print(f"  {'Lot':>5}  {'条件(depth,bw,feed,rpm)':>26}  {'chip':>6}  {'状態':}")

    scenarios = (
        [(200., 23., 1.0, 30000., rng.uniform(3, 5))] * 15 +  # 正常
        [(200., 23., 1.0, 30000., v) for v in np.linspace(5, 14, 8)] +  # ドリフト
        [(300., 23., 2.0, 30000., rng.uniform(16, 20))] * 3    # 異常
    )
    alert_count = 0
    for i, (d, bw, f, rpm, chip) in enumerate(scenarios):
        x4 = np.array([d, bw, f, rpm])
        result = detector.check(x4, observed_chip_um=chip)
        mon    = monitor.update(chip, lot_id=f"lot_{i:03d}")

        is_alert = result.is_anomaly or (mon is not None)
        if is_alert:
            alert_count += 1
        status = "⚠ ALERT" if is_alert else "✓ OK"
        print(f"  {i:>4}   ({d:.0f}µm,{bw:.0f}µm,{f:.1f},{rpm:.0f})  "
              f"{chip:>5.1f}µm  {status}")
        if is_alert and result.suggestions:
            print(f"         → {result.suggestions[0]}")

    print(f"\n  合計アラート: {alert_count}/{len(scenarios)}件")
    df = monitor.summary()
    print(f"  管理図異常率: {df['anomaly'].mean():.0%}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECOM 実データ検証: Layer2(IForest) + Layer3(Shewhart) を実工場データで評価
# ═══════════════════════════════════════════════════════════════════════════════

def run_secom(contamination: float = 0.07):
    """
    UCI SECOM（半導体工場センサ 590ch・pass/fail）で、合成データ用に作った
    検出プリミティブが実データでも効くかを検証する。
      Layer 2: 正常(pass)ロットのみで IForest を学習 → 全ロットを採点し、
               fail を「外れ値」として検出できるか（教師なし不良検出）。
      Layer 3: 不良率を時系列(時刻順)で Shewhart 管理図にかけ、歩留り
               ドリフトを検知できるか。
    多指標で評価（recall=104不良の検出率, FPR, precision, ROC-AUC）。
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.load_secom import load_secom
    from sklearn.metrics import roc_auc_score, precision_score, recall_score

    X, y, ts = load_secom()
    n_fail = int(y.sum())
    print(f"\n=== SECOM 実データ検証 ===")
    print(f"  {X.shape[0]} ロット × {X.shape[1]} センサ, "
          f"不良 {n_fail}/{len(y)} ({y.mean():.1%}), {ts.min()} → {ts.max()}")

    # Layer 2: 正常ロットのみで学習（半教師あり外れ値検出）
    pass_idx = np.where(y == 0)[0]
    layer = IForestLayer(contamination=contamination, n_estimators=200)
    layer.fit(X[pass_idx])
    Xs = layer.scaler.transform(X)
    pred = (layer.iforest.predict(Xs) == -1).astype(int)   # 1 = 外れ値=不良予測
    score = -layer.iforest.score_samples(Xs)               # 高いほど異常

    auc = roc_auc_score(y, score)
    rec = recall_score(y, pred, zero_division=0)
    fpr = float(pred[y == 0].mean())
    prec = precision_score(y, pred, zero_division=0)
    print(f"\n  [Layer2 IsolationForest — 教師なし不良検出]")
    print(f"    ROC-AUC            : {auc:.3f}")
    print(f"    検出 recall (不良)  : {rec:.3f}  ({int(pred[y==1].sum())}/{n_fail})")
    print(f"    false-positive rate: {fpr:.3f}")
    print(f"    precision          : {prec:.3f}")
    print(f"    → 高次元ノイジー実データでは AUC が単純しきい値より重要"
          f"（合成データの想定より難）")

    # Layer 3: 不良率の時系列ドリフト監視（50ロット移動窓）
    win = 50
    rate = np.array([y[max(0, i - win):i + 1].mean()
                     for i in range(len(y))])
    chart = ShewhartChart(warmup=min(100, len(rate) // 3), sigma_mult=3.0)
    drift_alerts = sum(chart.update(float(r))[0] for r in rate)
    print(f"\n  [Layer3 Shewhart — 歩留りドリフト監視 (窓={win})]")
    print(f"    管理図アラート: {drift_alerts} 区間"
          f"（不良率の上昇/下降ドリフト検知）")
    return {"auc": auc, "recall": rec, "fpr": fpr, "precision": prec,
            "drift_alerts": drift_alerts}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="不良検出・異常予兆")
    parser.add_argument("--csv",     help="parametric_summary_extended.csv のパス")
    parser.add_argument("--check",   nargs="+", type=float,
                        help="depth bw feed spindle [chip_um]")
    parser.add_argument("--monitor", action="store_true",
                        help="CSV全行を時系列監視（chip列を使用）")
    parser.add_argument("--demo",    action="store_true",
                        help="正常→ドリフト→異常のシナリオデモ")
    parser.add_argument("--secom",   action="store_true",
                        help="UCI SECOM 実工場データで Layer2/3 を検証")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if args.secom:
        run_secom()
        return

    if not args.csv:
        parser.error("--csv が必要です（--demo を除く）")

    df = pd.read_csv(args.csv)
    X  = df[FEM_FEATURES].values.astype(float)

    detector = AnomalyDetector()
    detector.fit(X)
    detector.save()
    print(f"[✓] AnomalyDetector 学習完了 ({len(X)} サンプル)")

    if args.check:
        vals = args.check
        x4   = np.array(vals[:4] if len(vals) >= 4 else vals + [1.0, 30000.][:4-len(vals)])
        chip = vals[4] if len(vals) > 4 else None
        result = detector.check(x4, observed_chip_um=chip)
        print(f"\n=== 異常検査結果 ===")
        print(f"  異常: {'⚠ YES' if result.is_anomaly else '✓ NO'}")
        print(f"  深刻度: {result.severity:.2f}")
        print(f"  原因: {result.cause}")
        for s in result.suggestions:
            print(f"  → {s}")

    if args.monitor:
        # deletion_fraction を簡易チッピングプロキシとして使用
        target_col = resolve_target_cols(df)[0]
        vals = df[target_col].values
        monitor = ProcessMonitor(usl=0.10, warmup=min(10, len(vals)//2))
        print(f"\n=== 時系列監視 ({target_col}) ===")
        for i, val in enumerate(vals):
            alert = monitor.update(float(val), lot_id=f"lot_{i:03d}")
            if alert:
                print(f"  lot_{i:03d}: ⚠ {alert.cause}")
        s = monitor.summary()
        print(f"  異常率: {s['anomaly'].mean():.1%} "
              f"({s['anomaly'].sum()}/{len(s)}件)")


if __name__ == "__main__":
    main()
