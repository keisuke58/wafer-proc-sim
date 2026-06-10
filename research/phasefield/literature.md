# Phase-Field Fracture × Bayesian Calibration — 文献リスト
# Keio / 村松研 修論準備用 (2026-2027)

## Layer 0: 絶対に読むべき3本（最初の2週間）

| # | 著者 | 年 | タイトル | 雑誌 | 重要度 | 何を得るか |
|---|---|---|---|---|---|---|
| 1 | Bourdin, Francfort, Marigo | 2000 | Numerical experiments in revisited brittle fracture | J. Mech. Phys. Solids 48(4) | ★★★★★ | AT2の原論文。Γ収束の物理的意味 |
| 2 | Miehe, Welschinger, Hofacker | 2010 | Thermodynamically consistent phase-field models of fracture | Int. J. Numer. Methods Eng. 83 | ★★★★★ | 実装標準（史上最引用PF論文）。SENTベンチマーク |
| 3 | Miehe, Hofacker, Welschinger | 2010 | A phase field model for rate-independent crack propagation | Comput. Methods Appl. Mech. Eng. 199 | ★★★★★ | Spectral split（引張/圧縮分離）の定式化 |

## Layer 1: コア理論（1ヶ月目）

| # | 著者 | 年 | タイトル | 雑誌 |
|---|---|---|---|---|
| 4 | Ambati, Gerasimov, De Lorenzis | 2015 | A review on phase-field models of brittle fracture and their application to laminated glass | Comput. Mech. 57 |
| 5 | Amor, Marigo, Maurini | 2009 | Regularized formulation of the variational brittle fracture | J. Mech. Phys. Solids 57(8) |
| 6 | Wu | 2017 | A unified phase-field theory for the mechanics of damage and quasi-brittle failure | J. Mech. Phys. Solids 103 |
| 7 | Borden et al. | 2012 | A phase-field description of dynamic brittle fracture | Comput. Methods Appl. Mech. Eng. 217 |

## Layer 2: 異方性フェーズフィールド（SiC / Ga₂O₃ 接続）

| # | 著者 | 年 | タイトル | 雑誌 |
|---|---|---|---|---|
| 8 | Teichtmeister, Kienle, Aldakheel, Keip | 2017 | Phase field modeling of fracture in anisotropic media | Int. J. Solids Struct. 97-98 |
| 9 | Li, Maurini | 2019 | Nucleation and propagation of cracks in anisotropic media | J. Mech. Phys. Solids 125 |
| 10 | Clayton, Knap | 2014 | A geometrically nonlinear phase field theory of brittle fracture | Int. J. Fract. 189 |
| 11 | Nagaraja et al. | 2023 | Phase-field modeling of brittle fracture with anisotropic surface energy | Comput. Mech. 72 |

**SiC特有の文献:**
| 12 | Sato et al. | 2021 | Fracture toughness anisotropy in 4H-SiC single crystals | J. Eur. Ceram. Soc. 41 |
| 13 | Yan et al. | 2020 | Cleavage fracture characteristics of β-Ga₂O₃ wafers | J. Phys. D: Appl. Phys. 53 |

## Layer 3: ベイズ較正 × フェーズフィールド（最重要・論文のコア）

| # | 著者 | 年 | タイトル | 雑誌 |
|---|---|---|---|---|
| 14 | **Noii et al.** | **2021** | **Bayesian inversion for unified ductile fracture phase field model** | **Comput. Methods Appl. Mech. Eng. 383** |
| 15 | **Noii, Khodadadian, Aldakheel, Wick** | **2022** | **Bayesian inversion for anisotropic hydraulic phase-field fracture** | **Comput. Methods Appl. Mech. Eng. 399** |
| 16 | Rappel, Beex, Noels, Bordas | 2018 | A tutorial on Bayesian inference to identify material parameters | Arch. Comput. Methods Eng. 27 |
| 17 | Noii, Aldakheel, Wick | 2023 | Probabilistic failure mechanisms via Monte Carlo simulations | Comput. Methods Appl. Mech. Eng. 414 |
| 18 | Wu, McAuliffe, Waisman, Deodatis | 2017 | Stochastic analysis of polymer composites: phase field fracture | Comput. Methods Appl. Mech. Eng. 312 |

> 注意: 文献14・15がこの修論の最直接のベンチマーク。必ずフルリード。
> Noii (2022) はまさに「異方性 × Bayes」= 本論文のターゲット形式。

## Layer 4: TMCMC（LUH修論からの橋渡し）

| # | 著者 | 年 | タイトル | 雑誌 |
|---|---|---|---|---|
| 19 | Ching, Chen | 2007 | Transitional Markov chain Monte Carlo method for Bayesian model updating | J. Eng. Mech. 133 |
| 20 | Beck, Zuev | 2013 | Rare-event simulation using TMCMC / BUS | Struct. Saf. 44 |
| 21 | Betz, Papaioannou, Straub | 2016 | Transitional Markov Chain Monte Carlo | Struct. Saf. 62 |

## Layer 5: FEniCSx 実装参考

| # | 著者 | 年 | タイトル | 備考 |
|---|---|---|---|---|
| 22 | Zolesi, Carrara, De Lorenzis | 2024 | On the efficiency of the implementation of phase-field fracture in FEniCSx | GitHub: `FEniCSx-phasefield` |
| 23 | Bleyer et al. | 2022 | Numerical tours of computational mechanics with FEniCSx | `comet-fenics.readthedocs.io` |

## 読む順番ロードマップ

```
Week 1-2:  #1, #2, #3        ← AT2の基礎を理解
Week 3-4:  #4, #5, #22, #23  ← FEniCSxで動かす
Month 2:   #8, #9, #12, #13  ← 異方性理論 + SiC/Ga₂O₃実験値取得
Month 3:   #14, #15, #16     ← Bayesian calibration の先行研究完全把握
Month 4:   #19, #20, #21     ← TMCMC理論の復習（LUH修論コード見直し）
Month 5+:  独自実装開始
```

## 慶應修論のポジショニング

**既存 (文献14・15):** Bayes PF fracture → isotropic材料 / 等方性表面エネルギー

**本論文の新規性:**
1. **異方性破壊エネルギー** G_c(θ) の TMCMC 推定（β-Ga₂O₃・4H-SiC）
2. **半導体ダイシング**への適用（プロセス工学との接続）
3. **劈開面角度の最適化**設計指針への還元

**投稿先候補:**
- 第一志望: Comput. Methods Appl. Mech. Eng. (CMAME, IF 7.2)
- 第二志望: Comput. Mech. (IF 4.1)
- 第三志望: Int. J. Fract. (IF 2.4)
- 国際会議: ECCOMAS 2028 / WCCM 2028
