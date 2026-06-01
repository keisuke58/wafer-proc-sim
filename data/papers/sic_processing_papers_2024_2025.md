# SiC 加工・パッケージ 注目論文まとめ (2024–2025)

> wafer-proc-sim 関連テーマの最新研究。実装への示唆付き。

---

## 1. SiC/SiO₂ 界面 Dit 低減・µ_ch 向上

### ★★★ Pre-oxidation treatment → Dit -33%, Qeff -64.5%
**Performance improvement of silicon carbide gate oxide interface by pre-oxidation**  
ScienceDirect, January 2025  
<https://www.sciencedirect.com/science/article/abs/pii/S0169433225000728>

- 犠牲酸化膜（ドライ酸化）+ NO ポストアニールで炭素クラスターを除去
- Dit を 33.3% 低減、酸化膜中有効電荷 Qeff を 64.5% 低減
- **実装示唆**: `tel_cleaning_model.py` の Pre-Gate シーケンスに犠牲酸化ステップを追加可能

---

### ★★★ NO アニール → µ_inv ≈ 35 cm²/Vs（1桁向上）
**Improved inversion channel mobility for 4H-SiC MOSFETs following high temperature anneals in nitric oxide**  
IEEE Electron Device Letters, 2001 (基礎文献)  
<https://ieeexplore.ieee.org/document/915604/>

- 1175°C / 2h NO アニールで Dit が conduction band 近傍で大幅低減
- µ_ch がアニール前の ~3 cm²/Vs → **~35 cm²/Vs** に向上（1桁）
- **モデルへの示唆**: 現在の `tel_process_model.py` は bulk µ ≈ 880 cm²/Vs を返しているが、
  実測反転層は **20–60 cm²/Vs** が正しい値。表面フォノン散乱 + Coulomb 散乱の両方が必要。

---

### ★★ POCl₃ + NO 組み合わせアニールで Dit 最小化
**Carrier Trap Density Reduction at SiO₂/4H-SiC Interface with Annealing in POCl₃ and NO**  
MDPI Materials 16(12):4381, 2023  
<https://www.mdpi.com/1996-1944/16/12/4381>

- POCl₃ アニール単独: Dit 低いが絶縁破壊電圧が低下
- NO アニール単独: 絶縁破壊電圧は高いが Dit やや高め
- **組み合わせ**: Dit 低減 + 高絶縁破壊電圧を両立
- **実装示唆**: `SEQUENCES["pregate_sic"]` に `"POCl3_anneal"` ステップ追加候補

---

### ★ NO アニール時間 vs Near-Interface-Oxide Traps
**Impact of the NO annealing duration on the SiO₂/4H-SiC interface properties**  
Solid-State Electronics, 2023  
<https://www.sciencedirect.com/science/article/abs/pii/S1369800123005590>

- NO アニール時間を延ばすと NIOTs（近界面酸化物トラップ）が減少
- 0.13–0.23 eV above Ec の欠陥は残留する → 完全な Dit = 0 は困難

---

## 2. SiC ダイシング・レーザー加工

### ★★★ フェムト秒 Bessel ビームによるステルスダイシング
**Stealth dicing of SiC using femtosecond laser Bessel beam**  
IEEE CPMT Symposium Japan 2024  
<https://ieeexplore.ieee.org/iel7/10491594/10491891/10492365.pdf>

- Bessel ビーム（非回折ビーム）で深さ方向に均一な改質層を形成
- 従来 ps ガウシアンビームより改質層が均一 → チッピングほぼゼロ
- **実装示唆**: `laser_groove_thermal_2d.py` に Bessel モード（軸方向強度プロファイル）追加

---

### ★★ SiC 結晶異方性 × レーザー加工品質
**Impact of material anisotropy on ultrafast laser dicing of SiC wafers**  
Optics & Laser Technology, 2024  
<https://www.sciencedirect.com/science/article/abs/pii/S0030399224018164>

- {10-10} 面（m面）は {11-20} 面（a面）より断面粗さ **20% 低い**
- 結晶軸方向に合わせたダイシング方向最適化で品質向上
- **実装示唆**: `dicing_blade_2d.py` / `laser_groove_thermal_2d.py` に `crystal_direction` パラメータ追加

---

### ★ 精密積層型ステルスダイシング (PLSD)
**Precision Layered Stealth Dicing of SiC Wafers by Ultrafast Lasers**  
PMC / Micromachines, 2022  
<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9315561/>

- レーザーパワーを層ごとに 100% → 62% に線形減衰させて均一改質
- 断面粗さ Ra ≈ 1 µm を達成
- 半絶縁性 4H-SiC（RF デバイス用）への適用実証

---

## 3. ワイヤーボンディング信頼性

### ★★ Cu クラッド Al ワイヤー → 電力サイクル寿命 +26%
**Influence of Al/CucorAl wire bonding on reliability of SiC devices**  
IEEE ISPSD 2021  
<https://ieeexplore.ieee.org/document/9655999/>

- Al ワイヤーの代替として Cu コア + Al クラッドワイヤーを評価
- 電力サイクル寿命が純 Al 比 **+26%** 向上
- コストは Au ワイヤーの 1/60 → SiC EV インバーターへの採用加速
- **実装示唆**: `backend_model.py` に `"CucoreAl"` ワイヤー種別を追加

---

### ★★ Cu クリップボンディング → 寄生インダクタンス低減
**Cu Clip-Bonding Method With Optimized Source Inductance for SiC MOSFET Power Module**  
IEEE Transactions on Power Electronics, 2022  
<https://ieeexplore.ieee.org/document/9674776/>

- ワイヤーボンドをフラットな Cu クリップに置き換え
- ソースインダクタンスを最適化してマルチチップ並列時の電流アンバランスを低減
- SiC 高周波スイッチング（100 kHz+）に適合
- **実装示唆**: `backend_model.py` に `clip_bonding` モードを追加（L ≈ 0.3–0.5 nH）

---

### ★ SiC 高温パッケージ ダイアタッチ材料レビュー
**Review of Die-Attach Materials for SiC High-Temperature Packaging**  
IEEE Transactions on Components, Packaging and Manufacturing Technology, 2024  
<https://ieeexplore.ieee.org/iel8/63/4359240/10568422.pdf>

- Au-Sn / Ag 焼結 / 低温ガラス / Cu 焼結を体系比較
- SiC の 200°C+ 動作には Ag 焼結または Au-Sn が必須
- SAC305 はんだは SiC パワーデバイスには長期信頼性で劣る

---

## 4. パッケージ応力・反り

### ★★★ Ag ナノ焼結プロセスの残留応力・反り解析
**Residual Stress and Warping Analysis of the Nano-Silver Pressureless Sintering Process in SiC Power Device Packaging**  
MDPI Micromachines 15(9):1087, August 2024  
<https://www.mdpi.com/2072-666X/15/9/1087>

- 加圧不要（pressureless）Ag 焼結の収縮 → 残留応力 → 反りを FEM で解析
- 焼結温度・時間・基板材料の影響を定量化
- **実装示唆**: `backend_model.py` の `Ag_sinter` ダイアタッチを焼結収縮モデルに拡張

---

### ★★ 樹脂封止プロセス（EMC 硬化）の残留応力・永久反り
**Modeling of Residual Stress and Permanent Warpage Induced by Resin Molding in SiC-Based Power Modules**  
MDPI Energies 18(20):5364, 2025  
<https://www.mdpi.com/1996-1073/18/20/5364>

- EMC 硬化収縮（化学収縮 + CTE ミスマッチ）を弾塑性構成則でモデル化
- 硬化後の「永久反り」を定量予測
- **実装示唆**: `backend_model.py` の `timoshenko_warpage` に EMC 硬化収縮項を追加

---

### ★ SiC パワーモジュール 熱構造連成モデル
**Thermal-Structural Modeling of a SiC-Based Power Module Subjected to Spatial Temperature Gradients**  
MDPI Engineering Proceedings 131(1):5, 2025  
<https://www.mdpi.com/2673-4591/131/1/5>

- 空間的温度勾配（ホットスポット）下での応力・歪み発展を連成解析
- 均一温度仮定の限界を指摘 → 局所応力集中が実際の破壊起点

---

### ★ SiC パワーデバイス 電力サイクル信頼性レビュー
**Review on Power Cycling Reliability of SiC Power Device**  
MDPI Electronic Materials 5(2):7, June 2024  
<https://www.mdpi.com/2673-3978/5/2/7>

- 電力サイクル試験手法・破壊モード・加速因子を体系整理
- プレーナー / プレスパック / 3D / ハイブリッドパッケージの比較
- Coffin-Manson / Norris-Landzberg パラメータ最新値を整理

---

## wafer-proc-sim への実装ロードマップ

| 優先度 | 実装内容 | 対象ファイル | 根拠論文 |
|---|---|---|---|
| ★★★ | µ_inv モデル修正（表面フォノン散乱追加） | `fem/tel_process_model.py` | IEEE 2001 NO anneal |
| ★★★ | Pre-oxidation ステップ追加 | `fem/tel_cleaning_model.py` | ScienceDirect 2025 |
| ★★ | Bessel ビームモード追加 | `fem/laser_groove_thermal_2d.py` | IEEE 2024 CPMT |
| ★★ | 結晶方向パラメータ追加 | `fem/dicing_blade_2d.py` | ScienceDirect 2024 |
| ★★ | CucoreAl ワイヤー追加 | `fem/backend_model.py` | IEEE ISPSD 2021 |
| ★★ | Cu クリップボンディングモード | `fem/backend_model.py` | IEEE Trans. PE 2022 |
| ★ | Ag 焼結収縮モデル | `fem/backend_model.py` | MDPI Micromachines 2024 |
| ★ | EMC 硬化収縮 + 永久反り | `fem/backend_model.py` | MDPI Energies 2025 |

---

*最終更新: 2026-06-01*
