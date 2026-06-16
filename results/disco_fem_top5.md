# DISCO FEM — Top-5 ランキング（業務直結度順）

> wafer-proc-sim 既存FEMを **ビジネス重要度** で序列化した統合ビュー。
> 物理は各 owner モジュールが所有（本ファイルは再実装しない）。

| # | 解析 | DISCO製品 | Tier | Solver | Owner |
|---|------|-----------|------|--------|-------|
| 1 | 裏面研削の残留応力 → ウェーハ反り | TAIKO® / 薄ウェーハ研削 (HBM 25→10 µm) | Tier 1 — 売上直結 | ABAQUS/Standard | `fem/grinding_warpage_2d.py (+_3d, taiko/deflection.py)` |
| 2 | ダイシングのチッピング / き裂進展 | ブレード + ステルスダイシング / カーフ品質 | Tier 1 — 競争力の核 | ABAQUS/Explicit | `fem/dicing_blade_2d.py (+_3d, stealth_dicing_crack_model.py)` |
| 3 | ダイ強度 / 薄ダイ破壊確率 | 車載品質保証 · 3D積層ハンドリング | Tier 2 — 品質保証 | analytic | `fem/die_strength_model.py (+ ml/weibull_analysis.py)` |
| 4 | 装置の構造剛性 · モーダル解析 | スピンドル / ステージ剛性 · 共振回避 | Tier 2 — 装置設計の土台 | ABAQUS/Standard | `fem/stage_vibration_modal.py (+ C++ _frame_fem)` |
| 5 | レーザ / 研削の熱解析 (HAZ) | ステルス · KABRA® · 熱変形 | Tier 2 — レーザ工程 | ABAQUS/Standard | `fem/laser_groove_thermal_2d.py (+ kabra_thermal_2d.py)` |

## 解析プロキシ実行結果（ABAQUS不要の数値チェック）

### #1 裏面研削の残留応力 → ウェーハ反り
- **物理**: 研削力→残留応力→チャック解放で反り。TAIKOリングで拘束。
- **full model**: `fem/grinding_warpage_2d.py (+_3d, taiko/deflection.py)` (ABAQUS/Standard)
- **run**: `cd fem && abaqus cae noGUI=grinding_warpage_2d.py   # reads run_config.json`
- **proxy**:

```
case: h_thin=50 µm, ring=3 mm, 300 mm wafer (1 g)
taiko_w_max_um: 6167.12
uniform_vs_taiko_sag_x: 4.12
note: ring acts ≈ built-in clamp → ~4× less sag (HBM thin-wafer handling)
```

### #2 ダイシングのチッピング / き裂進展
- **物理**: Drucker-Prager脆性破壊 + G_c軟化。送り/切込みでチッピング量。
- **full model**: `fem/dicing_blade_2d.py (+_3d, stealth_dicing_crack_model.py)` (ABAQUS/Explicit)
- **run**: `cd fem && abaqus cae noGUI=dicing_blade_2d.py   # reads run_config.json`
- **proxy**:

```
sweep_depth80um_30krpm: {'feed_2mm_s': {'chipping_um': 6.26, 'fracture_MPa': np.float64(503.2)}, 'feed_3.5mm_s': {'chipping_um': 7.84, 'fracture_MPa': np.float64(449.9)}, 'feed_5mm_s': {'chipping_um': 9.04, 'fracture_MPa': np.float64(418.9)}}
strength_loss_MPa_per_um_at_5um: -56.7
real_data_fit_MAE_um: 0.87
note: feed↑ → chipping↑ → strength↓; full chipping field needs the FEM
```

### #3 ダイ強度 / 薄ダイ破壊確率
- **物理**: Griffith σ_f=K_Ic/(Y√πa) → Weibull実装時破壊率(ppm)。
- **full model**: `fem/die_strength_model.py (+ ml/weibull_analysis.py)` (analytic)
- **run**: `python -m fem.die_strength_model`
- **proxy**:

```
chip_2um: {'fracture_MPa': np.float64(890.5), 'P_fail_ppm': 0.0, 'pass': np.True_}
chip_5um: {'fracture_MPa': np.float64(563.2), 'P_fail_ppm': 1.0, 'pass': np.True_}
chip_10um: {'fracture_MPa': np.float64(398.2), 'P_fail_ppm': 15.8, 'pass': np.True_}
```

### #4 装置の構造剛性 · モーダル解析
- **物理**: 切断荷重で静たわみ + Lanczos固有値で f1 >> f_blade を保証。
- **full model**: `fem/stage_vibration_modal.py (+ C++ _frame_fem)` (ABAQUS/Standard)
- **run**: `cd fem && abaqus cae noGUI=stage_vibration_modal.py -- --spindle_rpm 30000`
- **proxy**:

```
chuck: SiC clamped disk a=150 mm, h=20 mm (idealized)
f1_Hz: 4845.7
f_blade_Hz_at_30krpm: 500.0
separation_ratio: 9.69
design_rule: f1 >> f_blade to avoid resonance (want ratio ≳ 3)
```

### #5 レーザ / 研削の熱解析 (HAZ)
- **物理**: 移動ガウス熱源→温度場→アブレーション/HAZ要素除去。多パス重畳。
- **full model**: `fem/laser_groove_thermal_2d.py (+ kabra_thermal_2d.py)` (ABAQUS/Standard)
- **run**: `cd fem && abaqus cae noGUI=laser_groove_thermal_2d.py`
- **proxy**:

```
beam: 15 W, 0.7 absorptivity, 10 µm radius on SiC
dT_peak_K: 3491.0
T_ablation_K: 3100.0
ablates: True
note: steady proxy; pulsed peak fluence & HAZ width need the FEM
```

![priority](disco_fem_top5.png)
