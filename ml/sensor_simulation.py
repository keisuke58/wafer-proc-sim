"""
合成センサーデータ生成モジュール

FEMの静的解析結果（力・変位・応力）から、実機装置が出力するような
時系列センサーデータを合成する。

生成するセンサー信号:
  1. 切削力 Fx, Fy [N]       — FEM の RF1/RF2 + 現実的なノイズ
  2. スピンドル振動 [g]       — 刃の回転周期 + 高調波 + 不均衡成分
  3. 音響放射 (AE) [V]       — チッピング発生時にスパイク
  4. 加工深さフィードバック [µm] — 実際の切り込み量の時系列

用途:
  - 機械学習の異常検知モデルの学習データ
  - 「正常加工 vs 刃の摩耗 vs 過加工」の時系列分類
  - Disco の装置ソフトウェアが受け取るデータの模擬

Usage:
    from ml.sensor_simulation import SensorSimulator

    sim = SensorSimulator(
        spindle_rpm   = 30000,
        feed_mm_s     = 15.0,
        cut_depth_um  = 100.0,
        blade_width_um = 23.0,
    )
    df = sim.generate(duration_s=0.5, fs=50000)  # 0.5秒 @ 50kHz
    df.to_csv("synthetic_sensor.csv", index=False)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class SensorConfig:
    """センサーシミュレーション設定"""
    # 加工条件
    spindle_rpm:    float = 30000.0   # スピンドル回転数 [rpm]
    feed_mm_s:      float = 15.0      # 送り速度 [mm/s]
    cut_depth_um:   float = 100.0     # 切り込み深さ [µm]
    blade_width_um: float = 23.0      # ブレード幅 [µm]
    material:       str   = "Si"      # 材料 (Si, SiC, GaN)

    # FEM 由来の参照値（FEMを実行して得た結果を入れる）
    ref_Fx_N:       float = 0.0       # FEM 切削力 Fx [N] (参照)
    ref_Fy_N:       float = 0.0       # FEM 推力 Fy [N]   (参照)
    ref_chipping:   float = 0.02      # チッピング率（削除率）
    ref_stress_GPa: float = 1.0       # 最大主応力 [GPa]

    # センサーノイズレベル
    noise_force_N:   float = 0.5      # 力センサーノイズ RMS [N]
    noise_vib_g:     float = 0.01     # 振動センサーノイズ RMS [g]
    noise_ae_V:      float = 0.05     # AEセンサーノイズ RMS [V]

    # 模擬する異常モード
    wear_factor:     float = 1.0      # 刃の摩耗係数 (1.0=正常, 2.0=磨耗)
    anomaly_at_s:    Optional[float] = None  # 異常発生時刻 [s]


class SensorSimulator:
    """
    FEM 静的結果 + 物理モデルから 時系列センサーデータを生成する。
    """

    # 材料別の AE (音響放射) スケーリング
    AE_SCALE = {"Si": 1.0, "SiC": 2.5, "GaN": 1.8}

    def __init__(self, config: SensorConfig = None, **kwargs):
        if config is None:
            config = SensorConfig(**{k: v for k, v in kwargs.items()
                                     if k in SensorConfig.__dataclass_fields__})
        self.cfg = config
        self._rng = np.random.default_rng(42)

    # ── 物理モデル ─────────────────────────────────────────────────────────

    def _spindle_freq(self) -> float:
        """スピンドル基本周波数 [Hz]"""
        return self.cfg.spindle_rpm / 60.0

    def _blade_mesh_freq(self) -> float:
        """刃の噛み合い周波数 ≈ 刃枚数 × スピンドル周波数 (概算)"""
        return self._spindle_freq() * 8   # 8: 典型的な刃枚数

    def _base_force(self) -> tuple[float, float]:
        """
        切削力 (Fx, Fy) のベース値を FEM 参照値 or 経験式から計算。
        Fx ≈ 切削力, Fy ≈ 背分力
        """
        if self.cfg.ref_Fx_N != 0 or self.cfg.ref_Fy_N != 0:
            return self.cfg.ref_Fx_N, self.cfg.ref_Fy_N
        # 経験式: 切り込み深さ・送り速度のべき乗則
        d  = self.cfg.cut_depth_um * 1e-3   # mm
        vf = self.cfg.feed_mm_s
        Fx = 0.5 * d**0.8 * vf**0.3 * self.cfg.wear_factor
        Fy = 0.3 * Fx
        return Fx, Fy

    # ── 信号生成 ───────────────────────────────────────────────────────────

    def _force_signal(self, t: np.ndarray,
                      base_Fx: float, base_Fy: float) -> tuple[np.ndarray, np.ndarray]:
        """
        時系列切削力 [N]
        = ベース値 + スピンドル周期変動 + ノイズ
        """
        fs_hz  = self._spindle_freq()
        fm_hz  = self._blade_mesh_freq()

        # スピンドル周期変動: 偏心による sin 成分
        Fx = (base_Fx
              + base_Fx * 0.05 * np.sin(2 * np.pi * fs_hz * t)        # 基本
              + base_Fx * 0.02 * np.sin(2 * np.pi * fm_hz * t)        # 噛み合い
              + self._rng.normal(0, self.cfg.noise_force_N, len(t)))   # ノイズ

        Fy = (base_Fy
              + base_Fy * 0.05 * np.cos(2 * np.pi * fs_hz * t)
              + self._rng.normal(0, self.cfg.noise_force_N * 0.7, len(t)))

        return Fx, Fy

    def _vibration_signal(self, t: np.ndarray,
                          Fx: np.ndarray, Fy: np.ndarray) -> np.ndarray:
        """
        スピンドル振動 [g]
        切削力の変動成分を振動として投影。
        """
        fs_hz = self._spindle_freq()
        # 高調波成分
        vib = (
            0.001 * np.sin(2 * np.pi * 2 * fs_hz * t)   # 2× スピンドル
          + 0.0005 * np.sin(2 * np.pi * 3 * fs_hz * t)  # 3×
          + 0.0002 * np.abs(Fx - Fx.mean()) / max(Fx.std(), 1e-9)  # 力変動
          + self._rng.normal(0, self.cfg.noise_vib_g, len(t))
        )
        # 摩耗増加で振動増加
        vib *= self.cfg.wear_factor
        return vib

    def _ae_signal(self, t: np.ndarray, dt: float) -> np.ndarray:
        """
        音響放射 (Acoustic Emission) [V]
        - 背景ノイズ
        - チッピング発生確率に応じたランダムスパイク
        - 異常時刻以降は大スパイク
        """
        ae_scale = self.AE_SCALE.get(self.cfg.material, 1.0)
        ae = self._rng.normal(0, self.cfg.noise_ae_V, len(t))

        # チッピングスパイク: 確率 ∝ chipping_rate
        spike_rate_hz = self.cfg.ref_chipping * 1000 * ae_scale
        n_expected = int(spike_rate_hz * (t[-1] - t[0]))
        if n_expected > 0:
            spike_idx = self._rng.choice(len(t), size=n_expected, replace=False)
            ae[spike_idx] += self._rng.exponential(0.5 * ae_scale, n_expected)

        # 異常時刻以降: 大スパイク
        if self.cfg.anomaly_at_s is not None:
            anom_mask = t >= self.cfg.anomaly_at_s
            ae[anom_mask] += self._rng.normal(0, 2.0 * ae_scale, anom_mask.sum())
            # 突発的な大スパイク
            anom_idx = np.where(anom_mask)[0]
            if len(anom_idx) > 0:
                spike_idx2 = self._rng.choice(anom_idx,
                                               size=max(1, len(anom_idx)//20),
                                               replace=False)
                ae[spike_idx2] += 5.0 * ae_scale

        return ae

    def _depth_feedback(self, t: np.ndarray, base_Fx: np.ndarray) -> np.ndarray:
        """
        加工深さフィードバック [µm]
        設定値 + サーボ遅延 + 力による弾性変形
        """
        target = self.cfg.cut_depth_um
        # 低域通過フィルタ（サーボ応答）
        alpha = 0.05   # 応答速度
        d = np.zeros(len(t))
        d[0] = target
        compliance = 0.01  # µm/N
        for i in range(1, len(t)):
            d[i] = (1 - alpha) * d[i-1] + alpha * (target - compliance * base_Fx[i])
        d += self._rng.normal(0, 0.1, len(t))   # 深さセンサーノイズ [µm]
        return d

    # ── 公開 API ───────────────────────────────────────────────────────────

    def generate(self, duration_s: float = 0.1, fs: int = 50000,
                 label: str = "normal") -> pd.DataFrame:
        """
        センサーデータを生成して DataFrame で返す。

        duration_s : 時系列長 [秒]
        fs         : サンプリング周波数 [Hz]
        label      : 異常ラベル ("normal", "worn", "overcut", etc.)
        """
        dt = 1.0 / fs
        t  = np.arange(0, duration_s, dt)

        base_Fx, base_Fy = self._base_force()

        Fx, Fy = self._force_signal(t, base_Fx, base_Fy)
        vib    = self._vibration_signal(t, Fx, Fy)
        ae     = self._ae_signal(t, dt)
        depth  = self._depth_feedback(t, Fx)

        df = pd.DataFrame({
            "time_s":       t,
            "Fx_N":         Fx,
            "Fy_N":         Fy,
            "F_total_N":    np.sqrt(Fx**2 + Fy**2),
            "vibration_g":  vib,
            "ae_V":         ae,
            "depth_um":     depth,
            "label":        label,
            # 加工条件（メタデータ）
            "spindle_rpm":   self.cfg.spindle_rpm,
            "feed_mm_s":     self.cfg.feed_mm_s,
            "cut_depth_um":  self.cfg.cut_depth_um,
            "wear_factor":   self.cfg.wear_factor,
        })

        return df

    def generate_dataset(self, conditions: list[dict],
                         duration_s: float = 0.05,
                         fs: int = 10000) -> pd.DataFrame:
        """
        複数条件のデータセットを生成（機械学習用）。
        conditions: [{"cut_depth_um": 30, "wear_factor": 1.0, "label": "normal"}, ...]
        """
        dfs = []
        for cond in conditions:
            label = cond.pop("label", "normal")
            cfg = SensorConfig(**{**self.cfg.__dict__, **cond})
            sim = SensorSimulator(config=cfg)
            df  = sim.generate(duration_s=duration_s, fs=fs, label=label)
            df["condition_id"] = len(dfs)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        時系列 DataFrame から機械学習用特徴量を抽出（1行 = 1加工）。
        """
        feat = {}
        for col in ["Fx_N", "Fy_N", "vibration_g", "ae_V", "depth_um"]:
            if col in df.columns:
                arr = df[col].values
                feat[f"{col}_mean"]  = arr.mean()
                feat[f"{col}_std"]   = arr.std()
                feat[f"{col}_max"]   = arr.max()
                feat[f"{col}_p95"]   = np.percentile(arr, 95)
                feat[f"{col}_rms"]   = np.sqrt(np.mean(arr**2))
        # AE スパイク回数
        ae_threshold = df["ae_V"].mean() + 3 * df["ae_V"].std()
        feat["ae_spike_count"] = int((df["ae_V"] > ae_threshold).sum())
        # 加工条件
        for col in ["spindle_rpm", "feed_mm_s", "cut_depth_um", "wear_factor"]:
            if col in df.columns:
                feat[col] = df[col].iloc[0]
        if "label" in df.columns:
            feat["label"] = df["label"].iloc[0]
        return pd.DataFrame([feat])


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="合成センサーデータ生成")
    parser.add_argument("--mode", choices=["single", "dataset"],
                        default="dataset")
    parser.add_argument("--duration", type=float, default=0.1,
                        help="時系列長 [秒]")
    parser.add_argument("--fs", type=int, default=10000,
                        help="サンプリング周波数 [Hz]")
    parser.add_argument("--output", default="synthetic_sensor.csv")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.output)

    if args.mode == "single":
        sim = SensorSimulator(
            spindle_rpm=30000, feed_mm_s=15.0,
            cut_depth_um=100.0, ref_chipping=0.02
        )
        df = sim.generate(duration_s=args.duration, fs=args.fs)
        df.to_csv(out_path, index=False)
        print(f"[✓] {len(df)} 点 → {out_path}")

    else:
        # 正常・摩耗・過加工の3クラスデータセット生成
        conditions = [
            # 正常 (wear=1.0, chipping=2%)
            *[{"cut_depth_um": d, "wear_factor": 1.0,
               "ref_chipping": 0.02, "label": "normal"}
              for d in [50, 80, 120, 150, 200]],
            # 刃の摩耗 (wear=2.0, chipping=8%)
            *[{"cut_depth_um": d, "wear_factor": 2.0,
               "ref_chipping": 0.08, "label": "worn"}
              for d in [50, 80, 120, 150, 200]],
            # 過加工 (deep cut, anomaly at 0.05s)
            *[{"cut_depth_um": d, "wear_factor": 1.5,
               "ref_chipping": 0.15, "anomaly_at_s": 0.05,
               "label": "overcut"}
              for d in [200, 250, 300]],
        ]
        sim = SensorSimulator()
        df  = sim.generate_dataset(conditions, duration_s=args.duration, fs=args.fs)
        df.to_csv(out_path, index=False)

        feat_df = pd.concat([
            sim.extract_features(grp)
            for _, grp in df.groupby("condition_id")
        ], ignore_index=True)
        feat_path = out_path.replace(".csv", "_features.csv")
        feat_df.to_csv(feat_path, index=False)

        print(f"[✓] {len(df)} 点 × {df['condition_id'].nunique()} 条件 → {out_path}")
        print(f"[✓] 特徴量 {len(feat_df)} 行 → {feat_path}")
        print(f"\n  クラス分布:")
        print(feat_df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
