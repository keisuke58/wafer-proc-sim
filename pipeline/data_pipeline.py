"""
装置データ収集・解析基盤 — WaferProcPipeline

Disco AIアルゴリズムエンジニアの仕事内容を一本化したパイプライン:
  Step 1. データ収集     — FEM結果 / 合成センサー / 実機ログ を統合
  Step 2. モデル更新     — GP サロゲートをオンライン更新
  Step 3. 異常検知       — 新規データ点の即時チェック
  Step 4. レシピ補正     — 異常検知 → 自動補正提案
  Step 5. ログ保存       — 全結果を JSON / CSV に記録

Usage:
    # 初回学習
    python pipeline/data_pipeline.py --init --csv results/parametric_summary.csv

    # 新規データ点の評価（1件）
    python pipeline/data_pipeline.py \\
        --check --x 30 25 --y 0.07 2.1e9 \\
        --target-chipping 0.03

    # 継続モニタリング（CSV の全行を順次処理）
    python pipeline/data_pipeline.py --monitor --csv results/parametric_summary.csv

    # センサーデータ付きの総合評価
    python pipeline/data_pipeline.py --full --csv results/parametric_summary.csv
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ml.surrogate_gp        import DicingGPSurrogate, FEATURE_COLS, TARGET_COLS, MODEL_PATH
from ml.anomaly_detection   import AnomalyDetector, ProcessMonitor, ANOMALY_MODEL_PATH
from optimization.recipe_correction import RecipeCorrector
from ml.sensor_simulation   import SensorSimulator, SensorConfig

LOG_PATH = ROOT / "results" / "pipeline_log.jsonl"


# ═══════════════════════════════════════════════════════════════════════════════
# ログ
# ═══════════════════════════════════════════════════════════════════════════════

def log_event(event: dict):
    """JSONL 形式でイベントを追記"""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now().isoformat(),
            **event
        }, ensure_ascii=False, default=str) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# WaferProcPipeline
# ═══════════════════════════════════════════════════════════════════════════════

class WaferProcPipeline:
    """
    Disco AIアルゴリズムエンジニアの業務を再現したパイプライン。

    コンポーネント:
      gp        : GP サロゲートモデル（予測・不確かさ定量化）
      detector  : 異常検知器（3層）
      corrector : レシピ補正器
      monitor   : 管理図ベース時系列監視
    """

    def __init__(self):
        self.gp        : DicingGPSurrogate = None
        self.detector  : AnomalyDetector   = None
        self.corrector : RecipeCorrector   = None
        self.monitor   : ProcessMonitor    = None
        self.is_ready  = False

    # ── 初期化 ─────────────────────────────────────────────────────────────

    def initialize(self, csv_path: str,
                   target_col: str = "deletion_fraction",
                   usl: float = 0.10):
        """
        FEM 結果 CSV から全コンポーネントを学習・初期化する。
        """
        print(f"[Pipeline] 初期化中 ← {csv_path}")
        df = pd.read_csv(csv_path)
        from ml.surrogate_gp import resolve_target_cols
        active_targets = resolve_target_cols(df)
        X  = df[FEATURE_COLS].values.astype(float)
        Y  = df[active_targets].values.astype(float)

        # 1. GP サロゲート
        self.gp = DicingGPSurrogate()
        self.gp.fit(X, Y)
        self.gp.save()
        print(f"  [1/4] GPサロゲート学習完了 ({len(X)} サンプル)")

        # 2. 異常検知器
        self.detector = AnomalyDetector()
        self.detector.fit(X, Y)
        self.detector.save()
        print(f"  [2/4] AnomalyDetector 学習完了")

        # 3. レシピ補正器
        self.corrector = RecipeCorrector()
        self.corrector.load_model()
        print(f"  [3/4] RecipeCorrector 準備完了")

        # 4. プロセス監視（管理図）
        target_idx = TARGET_COLS.index(target_col) if target_col in TARGET_COLS else 0
        self.monitor = ProcessMonitor(
            target_col=TARGET_COLS[target_idx],
            usl=usl, warmup=min(20, len(df)//3)
        )
        # ウォームアップ
        for val in Y[:, target_idx]:
            self.monitor.update(float(val))
        print(f"  [4/4] ProcessMonitor ウォームアップ完了 "
              f"(UCL={self.monitor.chart.ucl:.4g} "
              f"LCL={self.monitor.chart.lcl:.4g})")

        self.is_ready = True
        log_event({"event": "initialize", "n_samples": len(X), "csv": csv_path})
        print(f"[Pipeline] 初期化完了\n")

    def load(self):
        """既存の学習済みモデルを読み込む"""
        self.gp       = DicingGPSurrogate.load()
        self.detector = AnomalyDetector.load()
        self.corrector = RecipeCorrector()
        self.corrector.load_model()
        self.monitor  = ProcessMonitor()
        self.is_ready = True

    # ── 単一点評価 ─────────────────────────────────────────────────────────

    def evaluate(self, x: np.ndarray,
                 y_obs: np.ndarray = None,
                 targets: dict = None,
                 verbose: bool = True) -> dict:
        """
        1つの加工条件を評価し、結果を返す。

        x      : (n_features,) 加工条件
        y_obs  : (n_targets,) 実測出力（なければ GP で予測のみ）
        targets: {"deletion_fraction": 0.03}
        """
        if not self.is_ready:
            raise RuntimeError("pipeline.initialize() を先に呼んでください")

        result = {"timestamp": datetime.now().isoformat()}

        # Step 1: GP 予測
        x_arr = np.asarray(x, dtype=float)
        mu, sigma = self.gp.predict(x_arr.reshape(1, -1), return_std=True)
        result["gp_mean"]  = mu[0].tolist()
        result["gp_sigma"] = sigma[0].tolist()
        result["features"] = {c: float(v) for c, v in zip(FEATURE_COLS, x_arr)}

        if y_obs is not None:
            result["observed"] = np.asarray(y_obs).tolist()

        # Step 2: 異常検知
        anom = self.detector.check(x_arr, y_obs)
        result["anomaly"]  = anom.is_anomaly
        result["severity"] = anom.severity
        result["cause"]    = anom.cause

        # Step 3: 管理図 (y_obs[0] を監視)
        if y_obs is not None:
            spc_alert = self.monitor.update(float(y_obs[0]))
            if spc_alert:
                anom.is_anomaly = True
                result["spc_alert"] = spc_alert.cause

        # Step 4: 異常時はレシピ補正
        if anom.is_anomaly and targets is not None:
            correction = self.corrector.correct(x_arr, y_obs, targets)
            result["correction"] = {
                "x_new":      correction.x_new.tolist(),
                "delta_pct":  correction.delta_pct.tolist(),
                "confidence": correction.confidence,
                "suggestions": correction.suggestions,
            }

        if verbose:
            self._print_result(result, anom)

        log_event({"event": "evaluate", **result})
        return result

    def _print_result(self, result: dict, anom):
        status = "⚠ ANOMALY" if result["anomaly"] else "✓ NORMAL"
        print(f"\n  [{status}] severity={result['severity']:.2f}")
        print(f"  GP予測: " + ", ".join(
            f"{c}={v:.4g}±{s:.4g}"
            for c, v, s in zip(TARGET_COLS, result["gp_mean"], result["gp_sigma"])
        ))
        if result["anomaly"]:
            print(f"  原因: {result['cause']}")
            for s in anom.suggestions:
                print(f"  → {s}")
        if "correction" in result:
            print(f"  補正提案 (信頼度 {result['correction']['confidence']:.2f}):")
            for s in result["correction"]["suggestions"]:
                print(f"    {s}")

    # ── バッチ監視 ─────────────────────────────────────────────────────────

    def monitor_csv(self, csv_path: str,
                    targets: dict = None,
                    max_rows: int = None) -> pd.DataFrame:
        """
        CSV の全行を時系列として監視し、異常・補正提案を付加して返す。
        """
        df = pd.read_csv(csv_path)
        if max_rows:
            df = df.head(max_rows)

        records = []
        n_anomaly = 0

        print(f"[Pipeline] 監視開始 ({len(df)} 件) ...")
        for i, row in df.iterrows():
            x   = row[FEATURE_COLS].values.astype(float)
            y   = row[TARGET_COLS].values.astype(float) \
                  if all(c in row for c in TARGET_COLS) else None
            res = self.evaluate(x, y, targets, verbose=False)

            rec = {c: row[c] for c in FEATURE_COLS}
            if y is not None:
                rec.update({c: row[c] for c in TARGET_COLS})
            rec["anomaly"]  = res["anomaly"]
            rec["severity"] = res["severity"]
            rec["cause"]    = res.get("cause", "")
            if "correction" in res:
                rec["correction_confidence"] = res["correction"]["confidence"]
                for j, fc in enumerate(FEATURE_COLS):
                    rec[f"x_new_{fc}"] = res["correction"]["x_new"][j]
            records.append(rec)
            if res["anomaly"]:
                n_anomaly += 1

        result_df = pd.DataFrame(records)
        anomaly_rate = n_anomaly / len(df)
        print(f"[Pipeline] 完了: 異常率 {anomaly_rate:.1%} "
              f"({n_anomaly}/{len(df)}件)")

        # 保存
        out = str(csv_path).replace(".csv", "_monitored.csv")
        result_df.to_csv(out, index=False)
        print(f"[✓] 監視結果 → {out}")
        log_event({"event": "monitor_batch",
                   "n_total": len(df), "n_anomaly": n_anomaly,
                   "anomaly_rate": anomaly_rate})
        return result_df

    # ── 総合デモ ───────────────────────────────────────────────────────────

    def run_full_demo(self, csv_path: str):
        """
        パイプライン全体のデモ:
        1. 初期化
        2. 単一点評価（正常・異常）
        3. センサーデータ生成 + 特徴量抽出
        4. バッチ監視
        """
        print("=" * 60)
        print("  wafer-proc-sim Pipeline Demo")
        print("  Disco AIアルゴリズムエンジニア 業務全体をシミュレート")
        print("=" * 60)

        # Step 1: 初期化
        self.initialize(csv_path)

        # Step 2: 正常点の評価（実績値: cut=80µm, kerf=23µm → 削除率66%）
        # GP予測と実測が近ければ正常、遠ければ異常フラグ
        print("--- [Demo 2] 正常点の評価 ---")
        self.evaluate(
            x=np.array([80.0, 23.0]),
            y_obs=np.array([0.665, 2.3e6]),
            targets={"deletion_fraction": 0.30},
        )

        # Step 3: 過加工 → レシピ補正（目標: 削除率10%以下に）
        print("\n--- [Demo 3] 異常点 → レシピ補正 ---")
        self.evaluate(
            x=np.array([290.0, 23.0]),
            y_obs=np.array([0.45, 1.5e6]),
            targets={"deletion_fraction": 0.10},
        )

        # Step 4: センサーデータ生成
        print("\n--- [Demo 4] センサーデータ生成 ---")
        conditions = [
            {"cut_depth_um": 30, "wear_factor": 1.0,
             "ref_chipping": 0.02, "label": "normal"},
            {"cut_depth_um": 60, "wear_factor": 2.0,
             "ref_chipping": 0.10, "label": "worn"},
            {"cut_depth_um": 65, "wear_factor": 1.8,
             "ref_chipping": 0.15, "anomaly_at_s": 0.03,
             "label": "overcut"},
        ]
        sim = SensorSimulator()
        sensor_df = sim.generate_dataset(conditions, duration_s=0.05, fs=5000)
        feat_df   = pd.concat([
            sim.extract_features(grp)
            for _, grp in sensor_df.groupby("condition_id")
        ], ignore_index=True)

        out_dir = ROOT / "results"
        out_dir.mkdir(exist_ok=True)
        sensor_df.to_csv(out_dir / "demo_sensor.csv",   index=False)
        feat_df.to_csv  (out_dir / "demo_features.csv", index=False)
        print(f"  センサー: {len(sensor_df)} 点 → results/demo_sensor.csv")
        print(f"  特徴量:   {len(feat_df)} 行  → results/demo_features.csv")

        # Step 5: バッチ監視
        print("\n--- [Demo 5] バッチ監視 ---")
        monitored = self.monitor_csv(
            csv_path,
            targets={"deletion_fraction": 0.04},
            max_rows=20,
        )

        print("\n" + "=" * 60)
        print("  Pipeline Demo 完了")
        print(f"  ログ: {LOG_PATH}")
        print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="wafer-proc-sim データパイプライン")
    parser.add_argument("--csv",   default=str(ROOT / "results" / "parametric_summary.csv"))
    parser.add_argument("--init",  action="store_true", help="初期化（モデル学習）")
    parser.add_argument("--check", action="store_true", help="単一点評価")
    parser.add_argument("--monitor", action="store_true", help="CSV全行を監視")
    parser.add_argument("--full",  action="store_true", help="総合デモ実行")
    parser.add_argument("--x",     nargs="+", type=float,
                        help=f"条件 ({', '.join(FEATURE_COLS)})")
    parser.add_argument("--y",     nargs="+", type=float,
                        help=f"実測 ({', '.join(TARGET_COLS)})")
    parser.add_argument("--target-chipping", type=float, default=0.04)
    args = parser.parse_args()

    pipe = WaferProcPipeline()
    targets = {"deletion_fraction": args.target_chipping}

    if args.full:
        pipe.run_full_demo(args.csv)
        return

    if args.init or args.check or args.monitor:
        pipe.initialize(args.csv)

    if args.check and args.x:
        x = np.array(args.x)
        y = np.array(args.y) if args.y else None
        pipe.evaluate(x, y, targets)

    if args.monitor:
        pipe.monitor_csv(args.csv, targets)


if __name__ == "__main__":
    main()
