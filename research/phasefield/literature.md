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

## Layer 4b: Bayes以外の推論・シミュレーション手法（候補比較）

> ユーザー指示: TMCMCは手段。Bayes以外のMLやSimulation手法も候補として把握しておく。

### A. Simulation-Based Inference (SBI) / 尤度なし推論

| 手法 | 略称 | 主要論文 | 特徴 | Bayes? |
|---|---|---|---|---|
| Sequential Neural Posterior Estimation | SNPE-C | Cranmer, Brehmer, Louppe (2020) PNAS | シミュレーターから直接posterior学習。尤度不要 | ○ amortized |
| Neural Likelihood Estimation | NLE | Papamakarios et al. (2019) NeurIPS | L(θ|x)をネットで推定 | ○ |
| Approximate Bayesian Computation | ABC | Sunnaker et al. (2013) PLOS Comp. Biol. | summary statistics比較。古典的 | ○ |

**wafer-proc-simとの接続**: SNPE-C は FEniCSx シミュレーターを black-box 扱いでき、sign-only (binary) observation にも適用可。probit 尤度を陽示的に書かなくてよい → Novel A の代替実装として有力。

---

### B. Physics-Informed Neural Network (PINN) 逆問題

| アプローチ | 論文 | 内容 |
|---|---|---|
| PINN forward | Raissi, Perdikaris, Karniadakis (2019) J. Comput. Phys. 378 | PDE残差をlossに組み込む |
| PINN inverse | Raissi et al. (2020) Science 367 | 観測データ + PDE → G_c, E の同定 |
| HP-VPINN | Kharazmi et al. (2021) Comput. Methods Appl. Mech. Eng. | variational formulation |
| PINN PF fracture | Goswami, Anitescu, Chatzi, Rabczuk (2020) Eng. Fract. Mech. | PFモデルをPINNで解く |

**特徴**: G_c(θ) を neural network parameterize → end-to-end 勾配降下で MAP 推定。  
**限界**: 不確かさ定量化 (UQ) が難しい。点推定のみ。Bayesian PINN (B-PINN) で拡張可能。  
**Lalain 2021**: B-PINN — DropOut + Laplace approximation で posterior 近似。

---

### C. Neural Operator (DeepONet / FNO) + 逆問題

| 手法 | 論文 |
|---|---|
| DeepONet | Lu, Jin, Pang, Zhang, Karniadakis (2021) Nature Machine Intelligence |
| FNO (Fourier Neural Operator) | Li et al. (2021) ICLR |
| UNet-FNO hybrid | Wen et al. (2022) Water Resources Research |

**wafer-proc-sim接続**: FNO で (Gc, β, ℓ) → d(x) を学習したら、逆問題は  
  `argmin_{θ} || d_FNO(θ) - y_obs ||`  
これは Bayesian posterior の MAP 推定に使える (Novel C のコア)。  
UQ が必要ならこの surrogate を SMC/NUTS のコールで使う。

---

### D. 微分可能シミュレーション (Differentiable Simulation)

| 手法 | ライブラリ | 特徴 |
|---|---|---|
| JAX + FEM | JAX-FEM (Xue et al. 2023), ngsolve+JAX | auto-diff through FEM solver |
| adjoint method | FEniCSx/dolfin-adjoint | adjoint PDE → ∂J/∂θ analytically |
| implicit differentiation | jaxopt | 制約付き最適化の微分 |

**特徴**: dolfin-adjoint を使えば FEniCSx solver を通じた勾配が計算できる → L-BFGS で Gc(θ) を直接最適化。  
**限界**: 確率的フレームワークでない → UQ なし。ただし MAP 推定として速い。  
**Bayesian との組み合わせ**: adjoint で log-posterior の gradient → HMC/NUTS に渡す。これが最も効率的な Bayes 推論になる可能性。

---

### E. 変分ベイズ / 近似推論

| 手法 | 論文 | 特徴 |
|---|---|---|
| Mean-field VI | Blei, Kucukelbir, McAuliffe (2017) JASA | 最速・精度↓ |
| Normalizing Flows | Rezende & Mohamed (2015) ICML | 柔軟な posterior |
| Stein Variational GD | Liu & Wang (2016) NeurIPS | 粒子ベース、勾配必要 |
| Laplace Approximation | MacKay (1992) | 2次近似、実装簡単 |

---

### 手法選択マトリクス（修論戦略）

| 手法 | UQ | 実装コスト | 速度 | 推奨シナリオ |
|---|---|---|---|---|
| TMCMC / SMC | ○ full posterior | 中 (既存コード流用) | 遅 | surrogate併用で ◎ |
| HMC/NUTS (NumPyro) | ○ full posterior | 低 (JAX) | 中-速 | adjoint勾配が取れる場合 ◎ |
| SNPE-C (SBI) | ○ amortized | 中-高 | 速 (amortized) | black-box simulator, 多数実験 ◎ |
| PINN inverse | △ MAP only | 中 | 速 | 初期MAP推定・warm start ◎ |
| FNO surrogate + MCMC | ○ full | 高 | 速 (surrogate後) | スケールアップ時 ◎ |
| dolfin-adjoint + L-BFGS | △ MAP only | 低 | 最速 | 検証・baseline ◎ |

**推奨戦略 (修論)**:  
1. `dolfin-adjoint + L-BFGS` → baseline MAP (実装1-2週間)  
2. `TMCMC or NUTS` + PINN surrogate → full posterior (メイン貢献)  
3. `SNPE-C` → 比較実験 (時間があれば)

---

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
