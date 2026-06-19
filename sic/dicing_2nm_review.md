# 先端ノード（2nm級）チップ・ダイシング方式レビュー

> wafer-proc-sim における SiC/Si ダイシング・モデリングの文献的背景。
> 2026-06-20 追加。対応PDFは `references/papers/`。

## TL;DR
- **2nm級ロジックでは単独方式は成立しない。** BEOL の ULK（ultra low-k）絶縁膜が極めて脆く、薄化ダイ（〜数十µm）と相まって機械ブレード単独はデラミネーション／チッピングで論外。
- 現場解は2系統:
  1. **レーザーグルービング → プラズマ/ブレード** のハイブリッド（ストリートの low-k/金属/TEG をレーザで除去後に個片化）
  2. **ステルスダイシング（SD / SDBG）**（レーザ内部集光で改質層 → エキスパンドで割断。非接触・kerfless・低応力）
- **KABRA®／SiCインゴット縦スライス（`2411.18093`）は土俵が別** — これは SiC ボウル/インゴットのウェハ化（スライス）であって、2nmロジックの個片化（dicing/singulation）ではない。混同注意。

## 方式比較（2nm文脈）

| 方式 | 2nm適性 | 物理ボトルネック | 備考 |
|---|---|---|---|
| ブレード単独 | ✗ | low-k 剥離・チッピング・薄ダイ割れ | 先端ノードでは不可 |
| レーザーグルービング + ブレード/プラズマ | ◎ | HAZ・デブリ管理 | advanced node の事実上の標準。ストリートの金属/low-k除去が主目的 |
| ステルスダイシング (SD/SDBG) | ◎ | 改質層制御・割断応力の均一性 | 非接触・kerfless。SDBG=研削前ステルスで極薄対応。取り都数最大 |
| プラズマダイシング単独 (DRIE/Bosch) | △→○ | 金属/low-kは plasma で切れない → レーザ前加工要 | 無応力・任意形状・高歩留り。マスク必要 |

## wafer-proc-sim 各文献の位置づけ

| PDF | 軸 | 本リポでの使いどころ |
|---|---|---|
| `2511.23141_bayesian_laser_dicing` | レーザプロセスの多目的BO自動探索（実機） | `sic_kabra_gp.py` の GP/BOサロゲートと同型。プロセス窓探索の外部ベンチ |
| `2308.02352_super_stealth_dicing` | 数十nm幅・AR 10³〜10⁴ のステルス | kerf下限・割断物理の参照 |
| `2411.18093_multifocal_sic_ps_slicing` | 6"4H-SiC ps縦スライス・低kerf loss | **SiC異方性・割断応力**（`visualize_sic_anisotropy.py` / `sic_chipping_kernel`）の実験対照 |
| `Micromachines2022_..._SiC_stealth_layered` | SiC積層ステルス | SiC専用ステルスの先行 |
| `2104.02763_drie_reduced_etch_lag` / `Micromachines2018_..._DRIE_levelset_montecarlo` | プラズマ/DRIE モデリング | プラズマ個片化のレベルセット/MC モデル |
| `Materials2024_..._HBM_thinning_singulation` | 薄化＋個片化（HBM） | 薄ダイ・SDBG文脈 |
| `2507.06738_diffuma_chip_dicing_dataset` | ダイシング工程の時系列画像・拡散予測 | データ駆動の欠陥/工程モニタ。CHDLデータセット |
| `2407.20268_gan_dicing_defect_augmentation` | GANによる欠陥データ拡張・分類 | 欠陥分類のデータ拡張 |

## オープン論点（wafer-proc-sim への取り込み候補）
- ステルス改質層の割断応力を `sic_chipping_kernel` の破壊靱性モデルで予測 → `2411.18093` の SiC実測（分離tensile strength低減）と突合。
- レーザーグルービングの HAZ をheatモジュールでモデル化し、low-k 損傷閾値とプロセス窓を `sic_kabra_gp` の GP で探索。
