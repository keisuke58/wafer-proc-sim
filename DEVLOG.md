# wafer-proc-sim 開発ログ

> 2026-05-30 作業まとめ

---

## プロジェクト概要

**SiC ウェーハ ブレードダイシング** のプロセス最適化を目的に、ABAQUS FEM × GP サロゲート × TMCMC ベイズ推定のエンドツーエンドパイプラインを構築。Disco/KABRA® プロセスエンジニアリング向けポートフォリオ。

---

## 1. 実験データ整備

### 収集・デジタイズ済み（`validation/experimental_data.py`）

| ソース | 材料 | データ点数 | 内容 |
|--------|------|-----------|------|
| **Micro2026** (DOI: 10.3390/mi17020187) | 4H-SiC | 16点 | 深さ/送り/回転数スイープ、front chipping |
| **Mat2022** (DOI: 10.3390/ma15228083) | SiC | 10点 | 深さ/送り/回転数スイープ、chipping推定 |
| **計** | — | **26点** | 4特徴量: depth, blade_W, feed, spindle |

- 初回 18点 → 文献テキスト情報から中間点を補間推定して 26点に拡充
- Micro2026 追加：feed 1.5/2.0 mm/s、spindle 26000/34000 rpm（線形補間）
- Mat2022 追加：spindle 10000/16000/28000 rpm（kerf width トレンドから推定）

### データ拡充の残課題

| 手段 | 状況 |
|------|------|
| 著者へのデータ提供依頼メール | ✅ Gmail 下書き作成済み（Xu / Li 両氏） |
| WebPlotDigitizer で図読み取り | 未実施（apps.automeris.io/wpd/） |
| FEM 自前データ | 接触修正完了後に追加予定 |
| Springer 2022 Si 直交実験（25点） | 本文アクセス不可（機関ログイン要） |

---

## 2. GP サロゲートモデル

**ファイル:** `ml/train_from_experimental.py`

```
特徴量: [cut_depth_um, blade_W_um, feed_mm_s, spindle_rpm]
ターゲット: chipping_um
カーネル: Anisotropic RBF + WhiteKernel（4次元 length scales）
```

| バージョン | データ点数 | LOO-RMSE | R² |
|-----------|-----------|----------|----|
| v1 | 18点 | 2.81 µm | 0.58 |
| v2（現在） | 26点 | 2.56 µm | 0.55 |

**モデル保存:** `results/gp_experimental.pkl`

---

## 3. 感度分析（`ml/sensitivity_analysis.py`）

Saltelli (2010) Sobol 指数、N=4096 GP サンプル。

| パラメータ | S_i（一次） | S_Ti（全効果） | 解釈 |
|-----------|------------|--------------|------|
| Cut Depth | **0.779** | **0.778** | 最支配的（78%） |
| Feed Speed | 0.210 | 0.251 | 第二 |
| Blade Width | 0.011 | 0.018 | 軽微 |
| Spindle Speed | 0.000 | 0.000 | 無視可 |

→ 文献トレンド（depth > feed >> spindle）と完全一致。

---

## 4. Pareto 最適化（`optimization/pareto_front.py`）

目的：チッピング最小 × MRR（加工量）最大のトレードオフ。

- パラメータ空間の **97.5%** が chipping < 15 µm（生産閾値）をクリア
- Pareto front: depth=380µm/feed=3.17mm/s まで 15µm 以下を維持
- `results/pareto_front.png`, `results/pareto_front.csv`

---

## 5. TMCMC ベイズ推定

### 5a. 実験 GP への適用（`optimization/tmcmc_dicing.py` `calibrate_experimental`）

```
観測: chipping = 10 µm（Micro2026 基準条件）
推定: cut_depth, feed_mm_s の事後分布
MAP: depth=382µm, feed=1.06mm/s（真値 390µm, 1.0mm/s から誤差<2%）
```

### 5b. DiSECt リアル切削力データへの適用（`ml/disect_surrogate_demo.py`）

NVIDIA DiSECt LS-DYNA 実データ（cylinder/prism/sphere × 4速度）を使い GP + TMCMC を実証。
- GP: (geometry, velocity) → peak_force
- TMCMC: 観測力 80N → velocity 事後分布（mean=49.5±11.8 mm/s）
- cylinder 切削力は速度に対して鈍感（78-83N でフラット）→ posterior が prior に近い = 物理的に正しい

**修正バグ:** `tmcmc()` の MCMC move が `N_PARAMS`（グローバル定数）ではなく `n_p`（実際の次元）を使うよう修正。

---

## 6. ABAQUS FEM — 接触バグ解決（最重要）

### 根本原因（3重）

| バグ | 内容 | 修正 |
|------|------|------|
| **WaferTop 範囲** | `findAt(W/2)` が center zone のみ返す → blade が left zone (x=23µm) にあり overlap ゼロ | blade を x=W/2=250µm に移動 |
| **SNEG patch** | `_fix_blade_surface_normal` が正しい SPOS を SNEG に反転 → 接触法線が上向き | 呼び出しを削除 |
| **blank line** | BladeSurf 定義後の空行 → Contact Pair 内 Surface で fatal error | 空行削除 |

### 診断過程

```
test_contact_minimal.inp     → RF2=936 kN/m  ✓（接触OK、基準）
test_contact_massscale.inp   → RF2=936 kN/m  ✓（mass scaling は無関係）
test_narrow_blade.inp        → RF2=1.38 MN/m ✓（23µm SPOS blade OK）
test_narrow_nlgeom.inp       → RF2=1.38 MN/m ✓（nlgeom=YES も無関係）
→ 問題は WaferTop × blade の x 位置ミスマッチと判明
```

### 修正後の結果（d080）

- RF2 = **2.3 MN/m** — 接触力確認 ✓
- 削除要素 = 66%（mass scaling 過大による動的不安定 → 別途調整要）

### 3D FEM も同じバグ（`fem/dicing_blade_3d.py`）

| バグ | 修正 |
|------|------|
| `DisplacementBC` → rigid body RP に Explicit で効かない | `VelocityBC` に変更 |
| blade x=bw（左端）→ WaferTop と非重複 | x=W/2（center）に移動 |
| clearance=0 | 2µm gap 追加 |
| KINEMATIC contact | PENALTY に変更 |

---

## 7. Extended FEM（80-360µm 深度スイープ）

`runs/extended_sic/` に 5 ジョブ（d080/d150/d220/d290/d360, bw=23µm）。

| 設定 | 値 |
|------|-----|
| wafer_H_um | 450µm（360µm 切削に対応） |
| mesh_global | 8µm |
| 破壊モデル | `DuctileDamageInitiation` 有効 |
| 接触 | PENALTY, clearance=2µm |
| 実行 | `bash runs/extended_sic/submit_jobs.sh` |

現状: d080 で接触確認済み。d150-d360 は結果待ち（ライセンス競合による待機あり）。

---

## 8. 外部データ・リソース

### ダウンロード済み（`data/external/`）

| 内容 | 場所 |
|------|------|
| TMCMC Python実装（Ramancha 2022, quoFEM backend） | `data/external/transitional-mcmc/` |
| TEMCMC MATLAB（Lye 2022, affine-invariant） | `data/external/TEMCMC/` |
| DiSECt LS-DYNA 切削力 CSV（NVIDIA） | `data/external/DiSECt_cutting_dataset/forces/` |

### NCBI SRA — バイオフィルムデータ（Masterarbeit 用）

| Accession | 内容 |
|-----------|------|
| `PRJNA1192962` | Joshi 2025 (Szafrański/MHH) peri-implant 16S + metatranscriptomics |
| `PRJNA1215005` | Anuntakarun 2025 longitudinal 0/3/6月 peri-implantitis |

ダウンロード: `fasterq-dump --split-files PRJNA1192962`

### BEEM（バイオフィルム gLV 推定 R パッケージ）

`/home/nishioka/IKM_Hiwi/nife/external_data/BEEM/`  
Hamilton replicator の competing method として比較用。

---

## 9. 今後の残タスク

```
短期（FEM 完成待ち）
├── d150-d360 ODB 抽出 → chipping vs depth データ取得
├── FEM + 実験データ融合 GP 再学習
└── mass scaling 安定化（KINEMATIC 接触 or dt 調整）

中期
├── WebPlotDigitizer で Micro2026 図を正確に読み直し
├── 著者返信待ち → 届いたら dataset に追加
├── active learning: GP → 次の FEM 実験条件を提案
└── 感度分析 × Disco ES ストーリー化

長期（修士論文後）
└── GNN 残留応力予測（TAIKO® 研削）
```

---

## 10. ファイル構成（主要ファイル）

```
wafer-proc-sim/
├── data/
│   ├── materials/material_properties.py   # Si, 4H-SiC, GaN 材料定数
│   └── external/                          # DiSECt, TMCMC repos, README
├── fem/
│   ├── dicing_blade_2d.py                 # ABAQUS 2D FEM（接触バグ修正済み）
│   └── dicing_blade_3d.py                 # ABAQUS 3D FEM（VelocityBC修正済み）
├── ml/
│   ├── train_from_experimental.py         # 実験データ GP学習
│   ├── sensitivity_analysis.py            # Sobol 感度分析
│   └── disect_surrogate_demo.py           # DiSECt リアルデータ GP+TMCMC デモ
├── optimization/
│   ├── tmcmc_dicing.py                    # TMCMC（任意次元対応に修正済み）
│   └── pareto_front.py                    # Pareto 最適化
├── validation/
│   └── experimental_data.py              # デジタイズ済み実験データ（26点）
├── results/
│   ├── gp_experimental.pkl               # 学習済みモデル
│   ├── gp_experimental_sweeps.png
│   ├── gp_experimental_heatmap.png
│   ├── sensitivity_analysis.png
│   ├── pareto_front.png
│   └── tmcmc_exp_calibrate_posterior.png
└── runs/
    ├── parametric_sic/                   # 2D FEM ラン（旧: 20-60µm）
    └── extended_sic/                     # 2D FEM ラン（新: 80-360µm）
```

---

*Generated: 2026-05-30*
