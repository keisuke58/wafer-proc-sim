# 3-D熱ソルバ ↔ 解析HAZモデル 校正メモ

`fem/calibrate_haz_3d.py` による、解析モデル `analytical_groove`（Stage-Aで使用の
2D per-pulse Beer–Lambert推定）と 3-D過渡熱ソルバ（真値）の突合。

## 結果

| 実行 | excess HAZ 平均比 (3D/解析) | R² | groove R² |
|---|---|---|---|
| CPU dx=5µm, n=12 | **5.8×** | 0.00 | 0.05 |
| **GPU dx=2µm, n=40** (vancouver RTX4090) | **3.56×** | 0.00 | 0.03 |

dxを5→2µmに細かくすると平均比は5.8→3.56へ収束（量子化が緩む）。だが
**R²は両方とも0のまま**。これは解像度の問題ではなく構造的:

> **解析モデルの excess HAZ は ~1.8µm の"定数"**（`L_diff × HAZ_factor` でレシピに
> ほぼ依存しない）。一方 3D は power/speed で 0–12µm に変動する。
> → **定数 vs 変動なので、点対点の相関は原理的に出ない。補正は倍率(~3.5×)止まり。**

図: `results/thermal3d_haz_calibration.png`（GPU版 `..._gpu_dx2.png`）

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
