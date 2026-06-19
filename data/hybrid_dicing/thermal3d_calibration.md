# 3-D熱ソルバ ↔ 解析HAZモデル 校正メモ

`fem/calibrate_haz_3d.py` による、解析モデル `analytical_groove`（Stage-Aで使用の
2D per-pulse Beer–Lambert推定）と 3-D過渡熱ソルバ（真値）の突合。

## 結果（SiC laser, n=12, dx=5µm, CPU）

| 量 | 関係 | R² | 備考 |
|---|---|---|---|
| excess HAZ | 3D ≈ **5.8×** analytic（平均比） | 0.00 | 解析は ~1.8µm 一定、3Dは 5–20µm |
| groove depth | 相関なし | 0.05 | 機構が別（fluence-ablation vs thermal-ablation） |

## 読み筋
- **解析モデルは熱HAZを約6倍過小評価**する（3Dが真値）。Stage-AのHAZを使う判定では
  `haz_um → ~6×` の補正を当てると3Dスケールに合う。
- **点対点の相関が弱いのは解像度由来**: dx=5µmで3D HAZが5µm刻みに量子化され、
  解析側もps領域でほぼ一定のため。信頼できる傾き（slope）には **dx≈2µm（GPU）** が必要。
- groove depthは機構が異なる（解析=フルエンス焼蝕、3D=熱焼蝕）ため直接比較は不適。
  個片化判定にはHAZ（両者とも熱量）を使うのが妥当。

## 次（GPU）
`python fem/calibrate_haz_3d.py --material SiC --n 60 --plot`（dx=2µm版）で
trustworthyな slope を取得 → `sic/hybrid/adapters.py` の LaserGrooveStage に補正係数を注入。
図: `results/thermal3d_haz_calibration.png`
