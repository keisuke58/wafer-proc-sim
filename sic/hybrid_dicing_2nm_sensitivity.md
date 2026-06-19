# ハイブリッド2nmダイシング — 重要変数ランキング (感度分析)

> 各ルートでレシピ空間をLHSサンプリング→フルモデル実行し、各KPIへの影響度を **符号付きSpearman順位相関 ρ** で評価。

- 材料: **SiC**, ストリート 25µm, 厚さ 110µm, low-k 2µm
- 指標: |ρ|=0 影響なし … |ρ|→1 強い単調影響。符号 + は「変数↑でKPI↑」。
- KPI: `feasible`(2nm制約), `σ_die`(破壊強度), `WPH`(スループット), `σ_res`(プロセス残留応力)

![variable importance](../results/hybrid_2nm_sensitivity.png)

## ヘッドライン — 各KPIの最重要ドライバ

| route | feasible | σ_die | WPH | σ_res |
|---|---|---|---|---|
| laser+stealth | groove speed [mm/s] (-0.35) | groove power [W] (+0.00) | groove speed [mm/s] (+0.81) | stealth speed [mm/s] (+0.25) |
| laser+plasma | groove speed [mm/s] (-0.37) | groove power [W] (+0.00) | plasma etch rate [µm/min] (+0.79) | plasma etch rate [µm/min] (-0.30) |
| laser+blade | groove speed [mm/s] (-0.33) | groove power [W] (+0.00) | groove speed [mm/s] (+0.67) | blade width [µm] (+0.22) |

## laser+stealth  (feasible 28/400)

### 総合ランキング (mean|ρ| over 4 KPI)

| 順位 | 変数 | 総合重要度 mean\|ρ\| |
|---:|---|---:|
| 1 | groove speed [mm/s] | 0.312 |
| 2 | groove passes | 0.194 |
| 3 | stealth speed [mm/s] | 0.166 |
| 4 | stealth layers | 0.107 |
| 5 | groove power [W] | 0.094 |
| 6 | stealth focal depth [µm] | 0.039 |
| 7 | stealth power [W] | 0.034 |

### KPI別 符号付きρ (と標準化β)

| 変数 | feasible | σ_die | WPH | σ_res |
|---|---:|---:|---:|---:|
| groove speed [mm/s] | -0.35 (-0.36) | +0.00 (+0.00) | +0.81 (+1.20) | -0.08 (+0.00) |
| groove passes | +0.33 (+0.35) | +0.00 (+0.00) | +0.44 (-0.48) | +0.00 (-0.00) |
| stealth speed [mm/s] | -0.05 (-0.06) | +0.00 (+0.00) | +0.36 (+0.24) | +0.25 (+0.00) |
| stealth layers | -0.10 (-0.06) | +0.00 (+0.00) | -0.29 (-0.24) | +0.04 (+0.00) |
| groove power [W] | +0.05 (+0.05) | +0.00 (+0.00) | +0.18 (+0.01) | -0.14 (-0.00) |
| stealth focal depth [µm] | +0.05 (+0.07) | +0.00 (+0.00) | +0.02 (+0.01) | -0.08 (-0.00) |
| stealth power [W] | +0.03 (+0.01) | +0.00 (+0.00) | -0.03 (+0.03) | -0.07 (-0.00) |

## laser+plasma  (feasible 26/400)

### 総合ランキング (mean|ρ| over 4 KPI)

| 順位 | 変数 | 総合重要度 mean\|ρ\| |
|---:|---|---:|
| 1 | plasma etch rate [µm/min] | 0.294 |
| 2 | groove speed [mm/s] | 0.268 |
| 3 | groove passes | 0.170 |
| 4 | plasma pressure [mTorr] | 0.097 |
| 5 | groove power [W] | 0.077 |

### KPI別 符号付きρ (と標準化β)

| 変数 | feasible | σ_die | WPH | σ_res |
|---|---:|---:|---:|---:|
| plasma etch rate [µm/min] | -0.09 (-0.10) | +0.00 (+0.00) | +0.79 (+0.71) | -0.30 (-0.00) |
| groove speed [mm/s] | -0.37 (-0.35) | +0.00 (+0.00) | +0.48 (+0.62) | +0.22 (+0.00) |
| groove passes | +0.36 (+0.34) | +0.00 (+0.00) | -0.16 (-0.21) | +0.17 (+0.00) |
| plasma pressure [mTorr] | +0.03 (+0.01) | +0.00 (+0.00) | +0.16 (+0.09) | -0.20 (-0.00) |
| groove power [W] | -0.09 (-0.02) | +0.00 (+0.00) | +0.18 (+0.03) | +0.04 (+0.00) |

## laser+blade  (feasible 19/400)

### 総合ランキング (mean|ρ| over 4 KPI)

| 順位 | 変数 | 総合重要度 mean\|ρ\| |
|---:|---|---:|
| 1 | groove speed [mm/s] | 0.265 |
| 2 | blade width [µm] | 0.134 |
| 3 | groove power [W] | 0.132 |
| 4 | blade feed [mm/s] | 0.110 |
| 5 | groove passes | 0.090 |

### KPI別 符号付きρ (と標準化β)

| 変数 | feasible | σ_die | WPH | σ_res |
|---|---:|---:|---:|---:|
| groove speed [mm/s] | -0.33 (-0.32) | +0.00 (+0.00) | +0.67 (+1.04) | +0.06 (+0.00) |
| blade width [µm] | -0.20 (-0.22) | +0.00 (+0.00) | +0.11 (-0.01) | +0.22 (+0.00) |
| groove power [W] | -0.04 (+0.00) | +0.00 (+0.00) | +0.34 (-0.00) | +0.16 (+0.00) |
| blade feed [mm/s] | -0.07 (-0.04) | +0.00 (+0.00) | +0.27 (+0.45) | -0.09 (-0.00) |
| groove passes | +0.26 (+0.24) | +0.00 (+0.00) | -0.07 (-0.51) | -0.04 (-0.00) |

## 読み方メモ

- **feasible列が実質的な設計ノブ**: 2nmで効くのは「low-kを切り切る溝深さ」と「HAZを抑える」変数。`groove_passes`/`groove_speed`が支配的なら、歩留りはレーザグルービング条件で決まる。
- **σ_dieがほぼ無相関 (|ρ|≈0)**: SiCはµm級欠陥を許容し強度が母材律速(pristine cap)になるため。強度は2nm SiCの律速KPIではない。
- **σ_res / WPH が実トレード**: 非接触ルート(stealth/plasma)はσ_resが低く、WPHはStage Aグルービングのpass数/速度で律速される。
