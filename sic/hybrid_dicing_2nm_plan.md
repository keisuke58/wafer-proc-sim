# 最強2nm ハイブリッドダイシング — 作戦 & 実装プラン

> wafer-proc-sim の既存ダイシング資産を **1本のハイブリッド・プロセスチェーン**に統合する計画。
> 背景は [dicing_2nm_review.md](dicing_2nm_review.md)。2026-06-20 起案。

---

## 0. 作戦（なぜハイブリッドか）

2nm級は単独方式では成立しない（[review](dicing_2nm_review.md) 参照）。**"全ダイシングの合致" = 各方式を工程の役割で分担させ、共通の最適化器で統べる**こと。

```
       ┌─ Stage A ─────────┐   ┌─ Stage B ──────────┐   ┌─ Stage C ───────┐
ウェハ →│ レーザーグルービング │ → │ 個片化 (stealth/    │ → │ 検査 / 欠陥分類   │→ ダイ
       │ low-k/金属/TEG除去  │   │  plasma/blade 選択) │   │  (CHDL/GAN)      │
       └────────┬───────────┘   └─────────┬──────────┘   └────────┬────────┘
                │ HAZ・low-k損傷            │ 割断応力・チッピング      │ 歩留り
                └──────────── 共通サロゲート + 多目的BO で同時最適化 ───────┘
                              制約: low-k非剥離 / die strength / throughput
```

**KPI（2nm制約）**: ① low-k デラミネーション無し（HAZ < 損傷閾値）② die break strength ≥ 目標 ③ チッピング ≤ ストリート裕度 ④ throughput（UPH）最大。

---

## 1. 既存資産マップ（再利用）

| Stage | 既存モジュール | 役割 |
|---|---|---|
| A レーザグルーブ | `fem/laser_groove_thermal_2d.py`, `fem/laser_groove_vectorized.py` | HAZ・溝形状の熱解析 |
| B-blade | `fem/dicing_heat_sim.py`, `pipeline/sic_dicing_pipeline.py`, `sic/sic_chipping_kernel.cpp` | ブレード熱・チッピングGP |
| B-stealth | `run_full_pipeline.py --process stealth`, `Micromachines2022_SiC_stealth_layered`(文献) | ステルス改質層 |
| B-plasma | `2104.02763`, `Micromachines2018_DRIE_levelset`(文献) | プラズマ個片化（モデルは新規） |
| 破壊判定 | `semiconductor_fracture_methods.py`, `semiconductor_cleavage_anisotropy.py`, `sic/physical_limits.py` | 延性–脆性遷移・SSD・劈開 |
| 最適化 | `ml/surrogate_gp.py`, `ml/surrogate_fno.py`, `sic/sic_kabra_gp.py`, `pipeline/recipe_optimizer_kernel.cpp` | サロゲート + レシピ最適化 |
| 統合実行 | `pipeline/run_full_pipeline.py`, `pipeline/disco_sw_stack.py` | フルチェーン実行 |

**結論: 不足は「方式横断の統一インターフェース」と「2stage連結＋同時最適化」だけ。** 物理ソルバは概ね揃っている。

---

## 2. ギャップ（新規実装）

1. **統一プロセス契約 `DicingStage` ABC** — 各方式を `simulate(recipe) -> StageOutcome{HAZ, kerf, chipping, residual_stress, die_strength, UPH}` の共通I/Fに揃える。
2. **プラズマ個片化サロゲート** — DRIE etch-lag を簡易レベルセット or 文献回帰で `DicingStage` 実装（B-plasma）。
3. **ハイブリッド・オーケストレータ** — Stage A 出力（溝深さ・残low-k・HAZ）を Stage B 入力へ受け渡す連結器。
4. **2nm制約モデル** — low-k 損傷閾値（HAZ温度×時間）・薄ダイ break strength（SSD亀裂→Weibull）を `constraints_2nm.py` に集約。
5. **多目的BO 統合最適化** — Stage A+B のレシピを結合変数として、`2511.23141` 流の constrained multi-objective BO（既存GP流用）。

---

## 3. 実装フェーズ（各フェーズ＝独立にコミット可能）

### Phase 1 — 統一I/F & アダプタ（足場）
- `sic/hybrid/base.py`: `DicingStage` ABC, `Recipe`, `StageOutcome` dataclass。
- 既存3ソルバ（laser_groove / dicing_heat_sim+chipping / stealth）を薄いアダプタで包む。
- `tests/test_hybrid_contract.py`: 各アダプタが契約を満たすかの smoke test。
- **完了条件**: 3方式が同一I/Fで `simulate()` でき、StageOutcome を返す。

### Phase 2 — 2nm制約モデル
- `sic/hybrid/constraints_2nm.py`: low-k HAZ閾値・die strength（`semiconductor_fracture_methods` のSSD→Weibull）・チッピング裕度。
- `tests/test_constraints_2nm.py`。
- **完了条件**: 任意 StageOutcome に対し feasible/violation を判定。

### Phase 3 — ハイブリッド・オーケストレータ
- `sic/hybrid/orchestrator.py`: A→B 連結、Bの方式選択（stealth/plasma/blade）。
- プラズマ最小実装 `sic/hybrid/plasma_stage.py`（文献回帰でOK、後で精緻化）。
- `pipeline/hybrid_dicing_pipeline.py`: 1コマンド実行（`run_full_pipeline` に寄せる）。
- **完了条件**: `python pipeline/hybrid_dicing_pipeline.py --route laser+stealth` が end-to-end で KPI を出す。

### Phase 4 — 統合最適化（最強レシピ探索）
- `sic/hybrid/optimize.py`: constrained multi-objective BO（既存 `surrogate_gp` 流用、`2511.23141` の two-level fidelity を参考）。
- 出力: パレート前線（die strength × UPH、low-k制約下）＋推奨ルート（A+B方式の最適組合せ）。
- `results/hybrid_2nm_pareto.png`。
- **完了条件**: 3ルート（laser+stealth / laser+plasma / laser+blade）を同一土俵で比較し「最強」を定量提示。

### Phase 5 — 検証 & 図
- `2411.18093`(SiC ps slicing 分離応力低減) と Stage-B stealth の割断応力を突合。
- `2308.02352`(super stealth kerf下限) で kerf 物理を sanity check。
- 4-panel まとめ図（`sic_dicing_pipeline.py` の図スタイル踏襲）。

---

## 4. ディレクトリ構成（新規）
```
sic/hybrid/
  __init__.py
  base.py            # DicingStage ABC, Recipe, StageOutcome
  adapters.py        # 既存ソルバ → 契約 への薄ラッパ
  plasma_stage.py    # 新規: DRIEサロゲート
  constraints_2nm.py # low-k / die strength / chipping 制約
  orchestrator.py    # A→B 連結 + ルート選択
  optimize.py        # constrained multi-objective BO
pipeline/hybrid_dicing_pipeline.py  # 1コマンド実行
tests/test_hybrid_*.py
```

## 5. 設計原則
- 既存ソルバは**書き換えない**（アダプタで包むだけ）。回帰リスク最小化。
- 物理パラメータは literature-typical の order-of-magnitude プロトタイプ。DISCO/fab 実データで校正する前提（`semiconductor_fracture_methods` の方針踏襲）。
- 各フェーズ独立コミット。Phase 1→2→3 が最小で end-to-end 動作する MVP ライン。

---

---

## 6. 実装ステータス（2026-06-20 完了）

Phase 1–4 実装・テスト済み（20 tests green）。Phase 5 の文献突合は未着手。

| 成果物 | パス |
|---|---|
| 統一I/F (Recipe/StageOutcome/DicingStage) | `sic/hybrid/base.py` |
| アダプタ (laser groove / stealth / blade) | `sic/hybrid/adapters.py` |
| プラズマ段 (ARDE) | `sic/hybrid/plasma_stage.py` |
| 2nm制約 + die strength | `sic/hybrid/constraints_2nm.py` |
| オーケストレータ + ルート | `sic/hybrid/orchestrator.py` |
| 多目的最適化 (3目的: σ_die↑/WPH↑/σ_res↓) | `sic/hybrid/optimize.py` |
| 感度分析 (変数重要度) | `sic/hybrid/sensitivity.py` |
| 図 + md出力 | `sic/hybrid/viz.py` |
| CLI | `pipeline/hybrid_dicing_pipeline.py` |
| テスト | `tests/test_hybrid_{contract,constraints,pipeline}.py` |
| 変数重要度ランキング(md) | `sic/hybrid_dicing_2nm_sensitivity.md` |
| 材料スイープ・データセット生成 | `sic/hybrid/dataset.py` |
| データセット(CSV)+データカード | `data/hybrid_dicing/{hybrid_2nm_dataset.csv,hybrid_2nm_summary.csv,README.md}` |
| OOD材料汎化デモ (RF/GP) | `ml/hybrid_ood_demo.py` |
| 図 | `results/hybrid_2nm_{pareto,sensitivity,materials,ood}.png` |

実行:
```
python pipeline/hybrid_dicing_pipeline.py                    # 3ルートKPI比較
python pipeline/hybrid_dicing_pipeline.py --optimize --plots # 多目的最適化+パレート図
python pipeline/hybrid_dicing_pipeline.py --sensitivity      # 変数重要度+図+md
```

### 主要知見（SiC, street 25µm, t 110µm, low-k 2µm）
1. **ns グルービングは2nmで不可**（HAZ>15µm→low-k剥離）。**ps/fs 冷間アブレーションが必須**。モデルが物理的に正しくns棄却。
2. **die strength は SiC では律速KPIにならない**（µm級欠陥を許容し母材律速500MPa＝flat）。感度ρ≈0で定量確認。**強度が効くのは脆性材(Ga2O3等)・粗ブレード時のみ**。
3. **実トレードは σ_res × WPH**。残留応力: **plasma < stealth < blade**（非接触が低応力）。最強ルート = **laser+stealth**（低σ_res×最高WPH）。
4. **歩留り(feasible)を支配するのは Stage A グルービング**（`groove_speed` −, `groove_passes` +）＝low-k切り切り。Stage B 個片化条件の寄与は二次。

### 残タスク（Phase 5）
`2411.18093`(SiC ps分離応力低減) / `2308.02352`(super stealth kerf下限) と Stage-B stealth 出力の突合検証図。
