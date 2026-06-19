# ハイブリッド2nmダイシング — 材料スイープ・データセット (Data Card)

- 生成: `sic/hybrid/dataset.py` / `pipeline/hybrid_dicing_pipeline.py --dataset`
- 行数: **4800**  ( 5 材料 × 3 ルート × 4 ジオメトリ × 80 レシピ )
- ジオメトリ (street µm / thickness µm): 20/80, 25/110, 30/140, 40/200
- 強度しきい値: 材料相対 (0.6·σ_flex) → 脆性材も可行勾配を持つ
- CSV: [`hybrid_2nm_dataset.csv`](hybrid_2nm_dataset.csv)

![material sweep](../../results/hybrid_2nm_materials.png)

## スキーマ

**キー**: `material`, `route`
**特徴量 (レシピ)**: `groove_power_W`, `groove_speed_mm_s`, `groove_passes`, `stealth_power_W`, `stealth_speed_mm_s`, `stealth_focal_depth_um`, `stealth_layers`, `plasma_pressure_mTorr`, `plasma_etch_rate_um_min`, `blade_feed_mm_s`, `blade_W_um`
**Stage出力**: `a_*`(レーザ溝), `b_*`(個片化)  例: `a_lowk_cleared_um`, `b_residual_MPa`, `b_ssd_um`
**ラベル/ターゲット**: `feasible`(0/1), `die_strength_MPa`, `total_chipping_um`, `throughput_wph`, `residual_stress_MPa`, `governing_crack_um`, `n_violations`

## 推奨タスク

- 分類: `feasible` (2nm制約充足) を特徴量から予測
- 回帰: `die_strength_MPa` / `total_chipping_um` / `residual_stress_MPa`
- 材料汎化: 4材料で学習→hold-out材料 (例 Ga2O3) でOODテスト

## 材料×ルート サマリ

| material | route | n | feasible% | σ_die中央 | chip中央[µm] | σ_res中央 | WPH中央 |
|---|---|--:|--:|--:|--:|--:|--:|
| Diamond | laser+blade | 320 | 5% | 1000 | 1.21 | 16.6 | 3.79 |
| Diamond | laser+plasma | 320 | 5% | 1000 | 1.04 | 17.0 | 2.04 |
| Diamond | laser+stealth | 320 | 5% | 1000 | 1.02 | 17.2 | 4.81 |
| Ga2O3 | laser+blade | 320 | 2% | 150 | 2.92 | 143.4 | 3.70 |
| Ga2O3 | laser+plasma | 320 | 5% | 150 | 1.04 | 20.1 | 2.07 |
| Ga2O3 | laser+stealth | 320 | 6% | 150 | 1.02 | 21.4 | 4.76 |
| GaN | laser+blade | 320 | 5% | 280 | 1.27 | 122.6 | 3.71 |
| GaN | laser+plasma | 320 | 4% | 280 | 1.04 | 24.3 | 2.06 |
| GaN | laser+stealth | 320 | 5% | 280 | 1.02 | 27.1 | 4.77 |
| Si | laser+blade | 320 | 5% | 200 | 1.02 | 22.5 | 3.73 |
| Si | laser+plasma | 320 | 5% | 200 | 1.04 | 8.4 | 2.08 |
| Si | laser+stealth | 320 | 6% | 200 | 1.02 | 5.5 | 4.77 |
| SiC | laser+blade | 320 | 4% | 500 | 1.02 | 63.9 | 3.64 |
| SiC | laser+plasma | 320 | 6% | 500 | 1.04 | 27.1 | 2.04 |
| SiC | laser+stealth | 320 | 6% | 500 | 1.02 | 30.9 | 4.63 |

## 既知の限界（重要）

- **レーザ光学/溝モデルは SiC 校正**（屈折率・アブレーション閾値・溝の熱定数）。したがって `feasible`/`a_haz_um`/`a_lowk_cleared_um` の材料依存は弱い。
- **材料固有シグナルは破壊/熱KPI**（`die_strength_MPa`, `*_chip_um`, `b_ssd_um`, `*_residual_MPa`）に出る。各材料の K_Ic/H/E・熱物性で駆動。
- 物性は literature-typical の order-of-magnitude プロトタイプ。実fab/DISCOデータ校正前提。
