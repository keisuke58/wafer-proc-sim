"""
wafer-proc-sim Dashboard — SiC 半導体プロセス × 物理 ML 統合シミュレーター
===========================================================================
半導体バリューチェーン全体を物理モデル化したポートフォリオ。

フロントエンド (Disco):
  1. GP Surrogate       — チッピング予測 (Micro2026 実験データ)
  2. Recipe Correction  — TMCMC 逆推定 → レシピ最適化
  3. Process Capability — Cpk / リアルタイム品質管理
  4. Blade Wear         — ブレード摩耗予測
  5. Cost per Die       — 経済最適化

ファブプロセス (TEL):
  6. TEL Cleaning       — 洗浄 → Dit → µ_inv インタラクティブ計算

結晶・材料 (Disco / フェローテック):
  7. Crystal Anisotropy — ダイシング方向最適化

後工程 (K&S / Besi):
  8. Wire Bonding       — 信頼性予測 (Weibull / Coffin-Manson)

市場分析:
  9. Market Overview    — SiC 市場 × 装置メーカーシェア

  10. Anomaly Detection  — 3-layer アラートシステム

Run:
    streamlit run app.py
"""

import os
import sys
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="wafer-proc-sim | SiC Process Simulator",
    page_icon="💎",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("💎 wafer-proc-sim")
st.sidebar.markdown(
    "**Physics-informed ML** for SiC semiconductor processes.\n\n"
    "Disco dicing → TEL cleaning → Package reliability"
)
st.sidebar.divider()
st.sidebar.markdown("**Frontend (Disco)**")
page = st.sidebar.radio(
    "Module:",
    ["🏠 Overview",
     "🔮 GP Surrogate",
     "🔧 Recipe Correction",
     "📊 Process Capability",
     "🔪 Blade Wear",
     "💴 Cost per Die",
     "🧹 TEL Cleaning → Dit",
     "💎 Crystal Anisotropy",
     "🔗 Wire Bonding",
     "📈 Market Analysis",
     "🚨 Anomaly Detection"],
)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_gp():
    from ml.train_from_experimental import ExperimentalGPSurrogate, MODEL_PATH
    return ExperimentalGPSurrogate.load(MODEL_PATH)

@st.cache_resource
def load_anomaly():
    from ml.anomaly_detection import AnomalyDetector
    import numpy as np
    det = AnomalyDetector()
    det.fit(np.array([[80.,23.],[150.,23.],[220.,23.],[290.,23.],[360.,23.]]))
    return det


# ═══════════════════════════════════════════════════════════════════════════════
# Page 0: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("💎 wafer-proc-sim")
    st.markdown("### Physics-informed ML for SiC Semiconductor Process Simulation")
    st.markdown(
        "完全な半導体バリューチェーンを物理モデル × 機械学習で実装したポートフォリオ。  \n"
        "Disco / TEL / ASML / Advantest / K&S / フェローテック / 東京ガスに対応。"
    )
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("物理モデル数", "25+", "フロントエンド〜後工程")
    col2.metric("対応企業数", "12社", "バリューチェーン全体")
    col3.metric("GP LOO R²", "0.64", "Fusion GP (26+5 pts)")
    col4.metric("バリデーション", "5項目 PASS", "文献データ対比")

    st.divider()
    st.markdown("#### 📐 Pipeline Overview")
    st.code("""
信越化学 (Si CZ) / フェローテック (SiC PVT)   ← 材料上流
        ↓
ASML EUV 露光 → TEL 洗浄 → ALD/酸化           ← Fab プロセス
        ↓
Disco ダイシング (Blade / Stealth / Bessel)    ← Disco コア
    + 結晶異方性 {10-10} vs {11-20}
        ↓
Lasertec 検査 → Advantest ATE               ← 品質保証
        ↓
K&S/Besi Wire Bond / Hybrid bonding          ← 後工程
        ↓
東京ガス パイプライン / LNG / Green H₂         ← エネルギー応用
    """, language="text")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🎯 対象企業 (就活 / 投資)")
        st.markdown("""
| 企業 | モデル | 適用 |
|---|---|---|
| **Disco** (6146) | Dicing, Crystal, Bessel | 就活第一志望 |
| **東京ガス** | Pipeline, LNG, H₂ | 就活第一志望 |
| **TEL** | ALD, Cleaning, CMP | プロセス上流 |
| **フェローテック** | SiC CZ Growth | 投資候補 |
| **信越化学** | Si CZ, V/G則 | 投資対象 |
| **Lasertec** | SiC/EUV 検査 | 装置関連 |
        """)
    with col_b:
        st.markdown("#### 📊 バリデーション結果")
        st.markdown("""
| モデル | MAPE | R² | |
|---|---|---|---|
| Blade Chipping | 14.3% | 0.978 | ✅ |
| Au-Al IMC 成長 | 2.7% | 0.9997 | ✅ |
| Package Warpage | 6.1% | 0.9998 | ✅ |
| Wire Bond Weibull | η誤差 5.4% | — | ✅ |
| SiC µ_ch vs Dit | — | 0.64 | ⚠️ |
        """)

    st.divider()
    st.info("👈 左のメニューから各モジュールを選択してください。")


# ═══════════════════════════════════════════════════════════════════════════════
# Page 1: GP Surrogate
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 GP Surrogate":
    st.title("🔮 GP Chipping Surrogate")
    st.markdown(
        "Predict front chipping from process parameters.  \n"
        "Trained on Micro2026 + Mat2022 experimental data (26 pts, LOO RMSE=2.38µm)."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        depth   = st.slider("Cut Depth [µm]",   60,  420, 200, 10)
        blade_w = st.slider("Blade Width [µm]",  20,   55,  23,  1)
        feed    = st.slider("Feed Speed [mm/s]", 0.3,  3.5, 1.0, 0.1)
        spindle = st.slider("Spindle Speed [krpm]", 18, 42, 30, 1)
        usl     = st.number_input("USL [µm]", value=15.0, step=0.5)

    model = load_gp()
    x = np.array([[depth, blade_w, feed, spindle * 1000.0]])
    mu, sigma = model.predict(x, return_std=True)
    chip, unc = float(mu[0]), float(sigma[0])
    cpu = (usl - chip) / (3 * max(unc, 0.1))

    with col1:
        st.metric("Predicted Chipping", f"{chip:.2f} µm", f"±{unc:.2f} µm")
        color = "normal" if chip < usl * 0.8 else ("off" if chip < usl else "inverse")
        st.metric("Cpk (one-sided)", f"{cpu:.2f}",
                  "✅ OK" if cpu >= 1.33 else ("⚠️ Marginal" if cpu >= 1.0 else "❌ OOC"))
        st.metric("Margin to USL", f"{usl - chip:.2f} µm")

    # Sweep plot
    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        d_arr = np.linspace(60, 420, 80)
        X_d = np.column_stack([d_arr, np.full(80, blade_w),
                                np.full(80, feed), np.full(80, spindle*1e3)])
        mu_d, sig_d = model.predict(X_d, return_std=True)
        axes[0].plot(d_arr, mu_d, "#2166ac", lw=2, label="GP mean")
        axes[0].fill_between(d_arr, mu_d-2*sig_d, mu_d+2*sig_d, alpha=0.2, color="#2166ac")
        axes[0].axhline(usl, color="#d62728", ls="--", lw=1.5, label=f"USL={usl}µm")
        axes[0].axvline(depth, color="k", ls=":", lw=1.2, label=f"Current={depth}µm")
        axes[0].set_xlabel("Cut Depth [µm]"); axes[0].set_ylabel("Chipping [µm]")
        axes[0].set_title("Depth Sweep"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

        f_arr = np.linspace(0.3, 3.5, 80)
        X_f = np.column_stack([np.full(80, depth), np.full(80, blade_w),
                                f_arr, np.full(80, spindle*1e3)])
        mu_f, sig_f = model.predict(X_f, return_std=True)
        axes[1].plot(f_arr, mu_f, "#d62728", lw=2, label="GP mean")
        axes[1].fill_between(f_arr, mu_f-2*sig_f, mu_f+2*sig_f, alpha=0.2, color="#d62728")
        axes[1].axhline(usl, color="#d62728", ls="--", lw=1.5, label=f"USL={usl}µm")
        axes[1].axvline(feed, color="k", ls=":", lw=1.2, label=f"Current={feed}mm/s")
        axes[1].set_xlabel("Feed Speed [mm/s]"); axes[1].set_ylabel("Chipping [µm]")
        axes[1].set_title("Feed Sweep"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 2: Real-time Recipe Correction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Recipe Correction":
    st.title("🔧 Real-time Recipe Correction")
    st.markdown(
        "**Digital Twin closed loop**:  \n"
        "Sensor chipping measurement → TMCMC inverse inference → GP optimal recipe."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        obs_chip = st.slider("Observed Chipping [µm]", 1.0, 20.0, 8.5, 0.5)
        n_samples = st.select_slider("TMCMC Samples", [200, 400, 600, 1000], 400)
        run_btn = st.button("🚀 Run Inference", type="primary")

    if run_btn:
        with st.spinner("Running TMCMC inference..."):
            from optimization.realtime_recipe import (
                infer_recipe_from_chip, find_optimal_recipe,
            )
            model = load_gp()
            post = infer_recipe_from_chip(obs_chip, model, n_samples=n_samples)
            opt  = find_optimal_recipe(model)

        with col1:
            st.success("Inference complete!")
            st.metric("Inferred Depth",  f"{post['depth_mean']:.0f} ± {post['depth_std']:.0f} µm")
            st.metric("Inferred Feed",   f"{post['feed_mean']:.2f} ± {post['feed_std']:.2f} mm/s")
            st.divider()
            st.metric("Optimal Depth",   f"{opt['opt_depth_um']:.0f} µm")
            st.metric("Optimal Feed",    f"{opt['opt_feed_mm_s']:.2f} mm/s")
            st.metric("Predicted Chip",  f"{opt['opt_chip_pred']:.1f} ± {opt['opt_chip_std']:.1f} µm")
            st.metric("Safe Zone",       f"{opt['safe_fraction']*100:.0f}%")

        with col2:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            samples = post["samples"]
            axes[0].hist2d(samples[:,0], samples[:,1], bins=25, cmap="Blues", density=True)
            axes[0].axvline(post["depth_mean"], color="#d62728", lw=2, ls="--")
            axes[0].axhline(post["feed_mean"],  color="#2166ac", lw=2, ls="--")
            axes[0].set_xlabel("Cut Depth [µm]"); axes[0].set_ylabel("Feed [mm/s]")
            axes[0].set_title(f"TMCMC Posterior (obs={obs_chip}µm)")

            depths = np.linspace(80, 390, 50); feeds = np.linspace(0.5, 3.0, 50)
            D, F = np.meshgrid(depths, feeds)
            X = np.column_stack([D.ravel(), np.full(D.size,23), F.ravel(), np.full(D.size,30000)])
            mu_g, _ = model.predict(X, return_std=True)
            im = axes[1].contourf(D, F, mu_g.reshape(D.shape), levels=15, cmap="YlOrRd")
            plt.colorbar(im, ax=axes[1], label="Chipping [µm]")
            axes[1].contour(D, F, mu_g.reshape(D.shape), levels=[15], colors="white", lw=2)
            axes[1].scatter(opt["opt_depth_um"], opt["opt_feed_mm_s"],
                            color="lime", s=200, marker="*", zorder=5, edgecolors="k")
            axes[1].scatter(post["depth_mean"], post["feed_mean"],
                            color="cyan", s=100, marker="D", zorder=5, edgecolors="k")
            axes[1].set_xlabel("Cut Depth [µm]"); axes[1].set_ylabel("Feed [mm/s]")
            axes[1].set_title("GP Response + Optimal Recipe (★)")

            plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 3: Process Capability
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Process Capability":
    st.title("📊 Process Capability Analysis")
    st.markdown("Live Cp/Cpk/Cpm from simulated production data. Semiconductor standard: **Cpk ≥ 1.67**.")

    col1, col2 = st.columns([1, 2])
    with col1:
        depth2   = st.slider("Cut Depth [µm]",   60, 420, 200, 10, key="cpk_depth")
        feed2    = st.slider("Feed Speed [mm/s]", 0.3, 3.5, 1.0, 0.1, key="cpk_feed")
        usl2     = st.number_input("USL [µm]", value=15.0, step=0.5, key="cpk_usl")
        n_sim    = st.slider("Simulated wafers", 50, 500, 200, 50)
        noise    = st.slider("Measurement noise σ [µm]", 0.5, 3.0, 1.5, 0.1)

    from optimization.process_capability import simulate_production_cpk, plot_capability
    model = load_gp()
    recipe = [depth2, 23.0, feed2, 30000.0]
    result = simulate_production_cpk(model, recipe, n_wafers=n_sim,
                                      noise_std=noise, usl=usl2)

    with col1:
        color_cpk = "normal" if result.cpk >= 1.33 else ("off" if result.cpk >= 1.0 else "inverse")
        st.metric("Cpk", f"{result.cpk:.3f}", result.status)
        st.metric("Cp",  f"{result.cp:.3f}")
        st.metric("Cpm", f"{result.cpm:.3f}")
        st.metric("Mean Chipping", f"{result.mean:.2f} µm")
        st.metric("Std Dev",       f"{result.std:.2f} µm")
        st.metric("Est. PPM",      f"{result.ppm_estimated:.0f}")

    with col2:
        # Simulate actual data
        x = np.array([recipe])
        mu_v, sig_v = model.predict(x, return_std=True)
        rng = np.random.default_rng(0)
        total_s = np.sqrt(float(sig_v[0])**2 + noise**2)
        data = np.maximum(rng.normal(float(mu_v[0]), total_s, n_sim), 0.0)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(data, "o-", color="#2166ac", ms=3, lw=1, alpha=0.8)
        axes[0].axhline(result.mean, color="k", lw=1.5, label=f"μ={result.mean:.2f}")
        axes[0].axhline(usl2, color="#d62728", ls="--", lw=1.5, label=f"USL={usl2}")
        axes[0].axhline(result.mean+3*result.std, color="orange", ls="-.", lw=1, label="UCL")
        axes[0].set_xlabel("Wafer #"); axes[0].set_ylabel("Chipping [µm]")
        axes[0].set_title("I-Chart"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.25)

        from scipy.stats import norm as scipy_norm
        x_r = np.linspace(max(0, result.mean-4*result.std), result.mean+4*result.std, 200)
        axes[1].hist(data, bins=25, density=True, color="#2166ac", alpha=0.6, edgecolor="k", lw=0.3)
        axes[1].plot(x_r, scipy_norm.pdf(x_r, result.mean, result.std), "k-", lw=2)
        axes[1].axvline(usl2, color="#d62728", ls="--", lw=2, label="USL")
        axes[1].axvline(result.mean, color="k", lw=1.5, label="μ")
        axes[1].set_xlabel("Chipping [µm]"); axes[1].set_ylabel("Density")
        axes[1].set_title(f"Histogram  Cpk={result.cpk:.2f}"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

        plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 4: Blade Wear
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔪 Blade Wear":
    st.title("🔪 Blade Wear Monitor")
    st.markdown("Exponential degradation model: `chip(n) = chip₀ × exp(α × n / N_life)`")

    from optimization.blade_wear import BladeWearModel

    col1, col2 = st.columns([1, 2])
    with col1:
        chip0  = st.slider("Initial chipping chip₀ [µm]", 1.0, 8.0, 4.5, 0.1)
        alpha  = st.slider("Wear coefficient α", 0.5, 5.0, 2.0, 0.1)
        n_life = st.slider("Blade life N_life [wafers]", 50, 400, 200, 10)
        usl_w  = st.number_input("USL [µm]", value=15.0, step=0.5, key="wear_usl")
        n_cur  = st.slider("Current wafer count n", 0, n_life, 50, 5)

    model_w = BladeWearModel(chip_0=chip0, n_life=n_life, alpha=alpha, usl=usl_w)
    status  = model_w.status(n_cur)

    with col1:
        st.metric("Current Chipping", f"{status['chip_now']:.2f} µm")
        st.metric("Replace at",       f"wafer #{status['replace_at']}")
        st.metric("Wafers to USL",    f"{status['n_to_usl']:.0f}")
        st.info(status["alert"])

    with col2:
        n_arr = np.linspace(0, n_life * 1.1, 200)
        chip_arr = model_w.predict_array(n_arr)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(n_arr, chip_arr, "#d62728", lw=2.5, label="Wear curve")
        ax.axhline(usl_w, color="#d62728", ls="--", lw=1.5, label=f"USL={usl_w}µm")
        ax.axhline(model_w.warning, color="orange", ls=":", lw=1.2, label="Warning")
        ax.axvline(n_cur, color="k", ls="-", lw=1.5, label=f"Now (n={n_cur})")
        ax.axvline(status["replace_at"], color="#2ca02c", ls="--", lw=1.5,
                   label=f"Replace at #{status['replace_at']}")
        ax.scatter([n_cur], [status["chip_now"]], color="k", s=100, zorder=5)
        ax.set_xlabel("Wafers on Blade"); ax.set_ylabel("Chipping [µm]")
        ax.set_title("Blade Wear Curve"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
        ax.set_ylim(0, usl_w * 1.5)
        st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 5: Cost per Die
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💴 Cost per Die":
    st.title("💴 Cost-Per-Die Optimiser")
    st.markdown("Total cost = blade amortisation + machine time + yield loss penalty.")

    from optimization.cost_per_die import CostPerDieModel

    col1, col2 = st.columns([1, 2])
    with col1:
        depth3  = st.slider("Cut Depth [µm]",   80, 390, 200, 10, key="c_depth")
        feed3   = st.slider("Feed Speed [mm/s]", 0.5, 3.5, 1.0, 0.1, key="c_feed")
        blade_c = st.number_input("Blade Cost [¥]", value=8000, step=500)
        n_life3 = st.slider("Blade Life [wafers]", 50, 400, 200, 10, key="c_nlife")
        mach_c  = st.number_input("Machine Cost [¥/hr]", value=15000, step=1000)

    costs = {"blade_cost_jpy": blade_c, "blade_life_wafers": n_life3,
              "machine_cost_jpy_hr": mach_c}
    cm    = CostPerDieModel(costs)
    model = load_gp()
    x = np.array([[depth3, 23.0, feed3, 30000.0]])
    mu_c, sig_c = model.predict(x, return_std=True)
    r = cm.compute({"cut_depth_um": depth3, "blade_W_um": 23., "feed_mm_s": feed3,
                     "spindle_rpm": 30000.}, float(mu_c[0]), float(sig_c[0]))

    with col1:
        st.metric("Cost/Die",       f"¥{r['cost_total_jpy']:.2f}")
        st.metric("Blade share",    f"¥{r['cost_blade_jpy']:.3f}")
        st.metric("Machine share",  f"¥{r['cost_machine_jpy']:.3f}")
        st.metric("Yield loss",     f"¥{r['cost_yield_jpy']:.3f}")
        st.metric("Throughput",     f"{r['wph']:.0f} WPH")
        st.metric("Defect rate",    f"{r['p_defect']*1e6:.0f} PPM")

    with col2:
        feeds_p = np.linspace(0.5, 3.5, 60)
        costs_p, throughput_p = [], []
        for f_v in feeds_p:
            x_v = np.array([[depth3, 23., f_v, 30000.]])
            mu_v, sig_v = model.predict(x_v, return_std=True)
            rv = cm.compute({"cut_depth_um": depth3, "blade_W_um": 23.,
                              "feed_mm_s": f_v, "spindle_rpm": 30000.},
                             float(mu_v[0]), float(sig_v[0]))
            costs_p.append(rv["cost_total_jpy"])
            throughput_p.append(rv["wph"])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(feeds_p, costs_p, "#d62728", lw=2)
        axes[0].axvline(feed3, color="k", ls=":", lw=1.5, label=f"Current={feed3}")
        axes[0].axvline(feeds_p[np.argmin(costs_p)], color="#2ca02c", ls="--",
                         lw=1.5, label=f"Optimal={feeds_p[np.argmin(costs_p)]:.1f}")
        axes[0].set_xlabel("Feed Speed [mm/s]"); axes[0].set_ylabel("Cost/Die [¥]")
        axes[0].set_title("Cost vs Feed"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

        axes[1].plot(feeds_p, throughput_p, "#2166ac", lw=2)
        axes[1].axvline(feed3, color="k", ls=":", lw=1.5)
        axes[1].set_xlabel("Feed Speed [mm/s]"); axes[1].set_ylabel("Throughput [WPH]")
        axes[1].set_title("Throughput vs Feed"); axes[1].grid(alpha=0.25)
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 6: Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Detection — 3-Layer Monitor")
    st.markdown(
        "**Layer 1**: GP z-score  |  **Layer 2**: Isolation Forest  |  **Layer 3**: Shewhart chart"
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        depth4  = st.slider("Cut Depth [µm]",   80, 390, 200, 10, key="a_depth")
        blade_4 = st.slider("Blade Width [µm]",  20,  55,  23,  1, key="a_bw")
        feed4   = st.slider("Feed Speed [mm/s]", 0.5, 3.5, 1.0, 0.1, key="a_feed")
        spin4   = st.slider("Spindle [krpm]",    18,  42,  30,  1, key="a_spin")
        obs4    = st.slider("Observed Chip [µm]", 0.5, 20.0, 4.5, 0.5)
        check_btn = st.button("🔍 Check Anomaly", type="primary")

    if check_btn:
        detector = load_anomaly()
        x4 = np.array([depth4, blade_4, feed4, spin4 * 1000.])
        result = detector.check(x4, observed_chip_um=obs4)

        with col1:
            if result.is_anomaly:
                st.error(f"⚠️ ANOMALY DETECTED  (severity={result.severity:.2f})")
            else:
                st.success("✅ Normal")
            st.write("**Layers:**", result.layer_flags)
            st.write("**Cause:**", result.cause)
            for s in result.suggestions:
                st.warning(f"→ {s}")

    # Simulated production stream
    with col2:
        st.subheader("Production Stream Simulation")
        n_lots = st.slider("Number of lots", 20, 100, 50, 5)
        drift  = st.slider("Wear drift [µm/lot]", 0.0, 0.3, 0.1, 0.01)

        model = load_gp()
        x = np.array([[200., 23., 1.0, 30000.]])
        mu0 = float(model.predict(x)[0])
        rng = np.random.default_rng(42)
        chips = [max(0.5, rng.normal(mu0 + i*drift, 1.5)) for i in range(n_lots)]

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(chips, "o-", color="#2166ac", lw=1.5, ms=4)
        ax.axhline(15.0, color="#d62728", ls="--", lw=1.5, label="USL 15µm")
        ax.axhline(mu0, color="#2ca02c", ls=":", lw=1.2, label=f"Initial {mu0:.1f}µm")

        detector = load_anomaly()
        alerts = []
        for i, chip in enumerate(chips):
            r = detector.check(np.array([200., 23., 1.0, 30000.]), observed_chip_um=chip)
            if r.is_anomaly:
                alerts.append(i)
                ax.scatter(i, chip, color="#d62728", s=100, zorder=5, marker="x", lw=2)

        if alerts:
            ax.scatter([], [], color="#d62728", marker="x", s=80, label=f"Alerts ({len(alerts)})")

        ax.set_xlabel("Lot #"); ax.set_ylabel("Chipping [µm]")
        ax.set_title(f"Simulated Production: drift={drift}µm/lot  alerts={len(alerts)}/{n_lots}")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        st.pyplot(fig); plt.close()

        if alerts:
            st.warning(f"⚠️ {len(alerts)} anomalies detected at lots: {alerts}")
        else:
            st.success("✅ All lots within normal range")


# ═══════════════════════════════════════════════════════════════════════════════
# Page 7: TEL Cleaning → Dit
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧹 TEL Cleaning → Dit":
    st.title("🧹 TEL 洗浄モデル — Dit → µ_inv Calculator")
    st.markdown(
        "洗浄シーケンス × 薬液条件 → SiC/SiO₂ 界面トラップ密度 (Dit) → "
        "チャンネル移動度 µ_inv を定量計算。  \n"
        "**Matthiessen 則**: 1/µ = 1/µ_phonon + 1/µ_coulomb + 1/µ_roughness"
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("洗浄シーケンス選択")
        seq_options = {
            "Pre-Gate SiC 最適化 (Piranha+HF+SC2+O₃+HF)": "pregate_sic",
            "Pre-Gate Si 標準 RCA (SC1+HF+SC2+HF)":        "pregate_si",
            "Post-CMP (H₂水+Megasonic+HF+SC2)":            "post_cmp",
            "Post-Dicing (H₂水+Megasonic+SC2)":            "post_dicing",
        }
        seq_label = st.selectbox("洗浄シーケンス", list(seq_options.keys()))
        seq_name  = seq_options[seq_label]

        dicing_options = {"ブレード (blade)": "blade", "レーザーps (laser_ps)": "laser_ps",
                          "ステルス (stealth)": "stealth", "Post-CMP": "post_cmp"}
        dice_label = st.selectbox("ダイシングプロセス", list(dicing_options.keys()))
        dice_proc  = dicing_options[dice_label]

        anneal_opt = st.selectbox("ゲート酸化アニール", ["none", "NO (1175°C)", "N2O", "POCl3"])
        anneal_key = anneal_opt.split()[0] if " " in anneal_opt else anneal_opt

        T_ox  = st.slider("熱酸化温度 [°C]", 1050, 1200, 1150, 10)
        t_ox  = st.slider("熱酸化時間 [h]",  0.5,  5.0,  2.0,  0.5)

    from fem.tel_cleaning_model import run_sequence, SurfaceState
    from fem.tel_process_model import dit_from_oxidation, channel_mobility_inv, ald_film

    r_clean = run_sequence(seq_name, dice_proc)
    Dit_clean = r_clean["after"]["dit_contrib"]
    Dit_ox    = dit_from_oxidation(T_ox, t_ox, anneal_T_C=950)
    Dit_total = Dit_clean * 0.2 + Dit_ox
    ald       = ald_film(50, 250, "HfO2")
    mu_inv    = channel_mobility_inv(Dit_total, ald["Cox_fF_um2"], anneal=anneal_key)

    with col1:
        st.divider()
        st.metric("炭素除去率",   f"{r_clean['reduction']['carbon']:.1f}%")
        st.metric("金属除去率",   f"{r_clean['reduction']['metal']:.1f}%")
        st.metric("粒子除去率",   f"{r_clean['reduction']['particle']:.1f}%")
        st.divider()
        st.metric("Dit (洗浄起源)",  f"{Dit_clean:.2e} cm⁻²eV⁻¹")
        st.metric("Dit (酸化起源)",  f"{Dit_ox:.2e} cm⁻²eV⁻¹")
        st.metric("Dit 合計",        f"{Dit_total:.2e} cm⁻²eV⁻¹")
        st.divider()
        color = "normal" if mu_inv > 50 else ("off" if mu_inv > 20 else "inverse")
        st.metric("µ_inv [cm²/Vs]", f"{mu_inv:.1f}",
                  "Excellent" if mu_inv > 50 else ("Good" if mu_inv > 25 else "Poor"))
        st.caption("参考: NO アニール標準品 ~35 cm²/Vs (Chung et al. 2001)")

    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Dit vs µ_inv カーブ
        dit_arr = np.logspace(9, 14, 200)
        mu_arr  = [channel_mobility_inv(d, ald["Cox_fF_um2"]) for d in dit_arr]
        axes[0].semilogx(dit_arr, mu_arr, "#2166ac", lw=2.5)
        axes[0].scatter([Dit_total], [mu_inv], color="#d62728", s=150, zorder=5,
                        label=f"Current: {mu_inv:.1f} cm²/Vs")
        axes[0].axhline(35, color="green", ls="--", lw=1.5, label="NO anneal std ~35")
        axes[0].axhline(67, color="purple", ls=":",  lw=1.5, label="Best (Dit=1e11)")
        axes[0].set_xlabel("Dit [cm⁻²eV⁻¹]"); axes[0].set_ylabel("µ_inv [cm²/Vs]")
        axes[0].set_title("Dit → µ_inv (Matthiessen's rule)"); axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.2)

        # 洗浄ステップ別汚染推移
        steps  = ["初期"] + [h["step"] for h in r_clean["history"]]
        carbons = [r_clean["before"]["carbon"]] + [h["after"]["carbon"] for h in r_clean["history"]]
        axes[1].semilogy(range(len(steps)), carbons, "ro-", lw=2, ms=7)
        axes[1].set_xticks(range(len(steps)))
        axes[1].set_xticklabels(steps, rotation=20, fontsize=8)
        axes[1].set_ylabel("Carbon contamination [cm⁻²]"); axes[1].grid(alpha=0.2, which="both")
        axes[1].set_title("Carbon Removal per Step\n(Dit の前駆体)")

        plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 8: Crystal Anisotropy
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💎 Crystal Anisotropy":
    st.title("💎 SiC 結晶異方性 — ダイシング方向最適化")
    st.markdown(
        "4H-SiC の破壊靱性は結晶方向依存。  \n"
        "**m面 {10-10}** が最良、**a面 {11-20}** でチッピング +28%。  \n"
        "*(Optics & Laser Technology 2024)*"
    )

    from fem.crystal_anisotropy import (
        kic_vs_angle, chipping_factor_vs_angle,
        apply_crystal_correction, optimal_dicing_direction,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        theta = st.slider("ダイシング方向角 θ [°]", 0, 60, 0, 1,
                           help="0°=m面(最良), 30°=a面(最悪)")
        depth_c = st.slider("切込み深さ [µm]", 20, 300, 100, 10)
        process_c = st.selectbox("プロセス", ["blade", "stealth_laser"])
        base_chip = 0.52 * depth_c ** 0.65
        corrected = apply_crystal_correction(base_chip, theta)
        kic = kic_vs_angle(theta)
        cfac = chipping_factor_vs_angle(theta)

        st.divider()
        st.metric("K_Ic [MPa√m]", f"{kic:.3f}")
        st.metric("チッピング補正", f"×{cfac:.3f}")
        st.metric("予測チッピング (base)", f"{base_chip:.2f} µm")
        st.metric("予測チッピング (補正後)", f"{corrected:.2f} µm",
                  f"{(cfac-1)*100:+.1f}% vs m-face")

        rec = optimal_dicing_direction(process_c)
        st.divider()
        st.success(f"推奨: {rec['recommended']}")
        st.info(f"効果: {rec['benefit']}")

    with col2:
        theta_arr = np.linspace(0, 60, 300)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        kic_arr  = [kic_vs_angle(t) for t in theta_arr]
        chip_arr = [apply_crystal_correction(base_chip, t) for t in theta_arr]

        axes[0].plot(theta_arr, kic_arr, "#2166ac", lw=2.5)
        axes[0].axvline(theta, color="#d62728", ls="--", lw=2, label=f"θ={theta}°")
        axes[0].axvline(0,  color="green",  ls=":", lw=1.5, label="m-face (best)")
        axes[0].axvline(30, color="red",    ls=":", lw=1.5, label="a-face (worst)")
        axes[0].set_xlabel("Direction angle θ [°]"); axes[0].set_ylabel("K_Ic [MPa√m]")
        axes[0].set_title("Fracture Toughness Anisotropy"); axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.2)

        axes[1].plot(theta_arr, chip_arr, "#d62728", lw=2.5)
        axes[1].axvline(theta, color="k", ls="--", lw=2, label=f"θ={theta}°")
        axes[1].scatter([theta], [corrected], color="k", s=150, zorder=5)
        axes[1].set_xlabel("Direction angle θ [°]"); axes[1].set_ylabel("Chipping [µm]")
        axes[1].set_title(f"Chipping vs Direction (depth={depth_c}µm)")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.2)

        plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 9: Wire Bonding
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Wire Bonding":
    st.title("🔗 ワイヤーボンディング 信頼性予測")
    st.markdown("Au / Cu / Al ワイヤー材料別の信頼性を Weibull + Coffin-Manson で定量化。")

    from fem.backend_model import (pull_strength_samples, imc_thickness,
                                    heel_crack_life, wire_inductance, WIRE_MATERIALS)
    from scipy.stats import weibull_min

    col1, col2 = st.columns([1, 2])
    with col1:
        wire_sel = st.selectbox("ワイヤー材料", ["Au", "Cu", "Al"])
        dT_w     = st.slider("熱サイクル ΔT [K]", 20, 200, 100, 10)
        loop_h   = st.slider("ループ高さ [µm]", 100, 400, 200, 10)
        T_store  = st.slider("保存温度 [°C]", 50, 175, 125, 5)
        t_store  = st.slider("保存時間 [h]", 100, 3000, 1000, 100)
        wire_len = st.slider("ワイヤー長さ [µm]", 500, 4000, 2000, 100)

    m = WIRE_MATERIALS[wire_sel]
    samples  = pull_strength_samples(wire_sel, n=1000)
    imc_t    = imc_thickness(wire_sel, T_store, t_store)
    heel_N   = heel_crack_life(wire_sel, dT_w, loop_h)
    L_nH     = wire_inductance(loop_h, wire_len)

    with col1:
        st.divider()
        st.metric("プル強度 (平均)", f"{np.mean(samples):.2f} g")
        st.metric("プル強度 (B10)", f"{np.percentile(samples,10):.2f} g",
                  "MIL-STD-883 min: 3g ✅" if np.percentile(samples,10) > 3 else "⚠️ Below spec")
        st.metric("IMC 厚さ", f"{imc_t:.3f} µm",
                  "⚠️ > 1µm 注意" if imc_t > 1.0 else "OK")
        st.metric("ヒールクラック寿命", f"{heel_N:,.0f} cycles",
                  "✅" if heel_N > 50000 else "⚠️")
        st.metric("寄生インダクタンス", f"{L_nH:.3f} nH")

    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        x_w  = np.linspace(0, 16, 300)
        beta = m["beta"]; eta = m["eta_g"]
        axes[0].hist(samples, bins=30, density=True, color="#2166ac", alpha=0.6,
                     edgecolor="k", lw=0.3)
        axes[0].plot(x_w, weibull_min.pdf(x_w, beta, scale=eta), "#d62728", lw=2.5,
                     label=f"Weibull η={eta:.1f}g, β={beta:.1f}")
        axes[0].axvline(3.0, color="purple", ls="--", lw=1.5, label="MIL min 3g")
        axes[0].set_xlabel("Pull strength [g]"); axes[0].set_ylabel("PDF")
        axes[0].set_title(f"{wire_sel} Wire Pull Strength"); axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.2)

        dT_arr = np.linspace(20, 200, 100)
        lives  = [heel_crack_life(wire_sel, dT, loop_h) for dT in dT_arr]
        axes[1].semilogy(dT_arr, lives, "#2166ac", lw=2.5)
        axes[1].axhline(50000, color="green", ls="--", lw=1.5, label="Target 50k")
        axes[1].axvline(dT_w, color="#d62728", ls=":", lw=2, label=f"ΔT={dT_w}K")
        axes[1].scatter([dT_w], [heel_N], color="#d62728", s=150, zorder=5)
        axes[1].set_xlabel("ΔT [K]"); axes[1].set_ylabel("Fatigue life [cycles]")
        axes[1].set_title("Heel Crack Fatigue Life\n(Coffin-Manson)"); axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.2, which="both")

        plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 10: Market Analysis
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Market Analysis":
    st.title("📈 SiC 半導体装置 市場分析")
    st.markdown("SiC パワーデバイス市場 × 装置メーカーのバリューチェーン分析。")

    import matplotlib.ticker as mticker

    YEARS = np.arange(2022, 2031)
    SIC_TOTAL = np.array([2.0, 2.5, 3.0, 4.2, 5.8, 7.5, 9.2, 11.0, 13.0])
    SIC_EV    = np.array([1.1, 1.4, 1.7, 2.4, 3.3, 4.3, 5.3, 6.3,  7.5])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SiC デバイス市場 ($B)")
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        ax1.fill_between(YEARS, 0, SIC_EV, alpha=0.7, color="#2196f3", label="EV/HEV")
        ax1.fill_between(YEARS, SIC_EV, SIC_TOTAL, alpha=0.5, color="#4caf50", label="産業/その他")
        ax1.plot(YEARS, SIC_TOTAL, "ko-", lw=2, ms=5)
        cagr = (SIC_TOTAL[-1]/SIC_TOTAL[0])**(1/8)-1
        ax1.text(2026, 10, f"CAGR\n{cagr*100:.0f}%", fontsize=12, fontweight="bold", color="#d62728")
        ax1.set_ylabel("Market [$B]"); ax1.legend(fontsize=8); ax1.grid(alpha=0.2)
        plt.tight_layout(); st.pyplot(fig1); plt.close()

    with col2:
        st.subheader("装置メーカー SiC 依存度")
        companies = ["Disco", "K&S", "TEL", "Screen", "Advantest", "ASML"]
        sic_pct   = [45, 22, 18, 14, 12, 3]
        rev_B     = [2.9, 1.4, 17.5, 3.1, 4.2, 28.0]
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        colors2 = ["#ff7f0e","#9467bd","#d62728","#8c564b","#2ca02c","#2166ac"]
        for i, (co, sp, rv, col) in enumerate(zip(companies, sic_pct, rev_B, colors2)):
            ax2.scatter(sp, 18 if co == "Disco" else (15 if co in ["K&S","Advantest"] else 12),
                        s=rv*15, color=col, alpha=0.8, edgecolors="k", lw=1, zorder=5)
            ax2.annotate(co, xy=(sp, 18 if co == "Disco" else (15 if co in ["K&S","Advantest"] else 12)),
                         xytext=(sp+0.5, 18 if co == "Disco" else (15 if co in ["K&S","Advantest"] else 12)+0.3),
                         fontsize=9)
        ax2.set_xlabel("SiC exposure [%]"); ax2.set_ylabel("CAGR estimate [%]")
        ax2.set_title("(bubble size = 2023 revenue)"); ax2.grid(alpha=0.2)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.divider()
    st.markdown("#### Equipment Segment Context")
    col3, col4, col5 = st.columns(3)
    col3.metric("SiC Market CAGR", "~27%", "2023–2030 (Yole 2024)")
    col4.metric("Dicing/Grinding SiC exposure", "~45%", "highest process sensitivity")
    col5.metric("Cleaning SiC premium", "~2.5×", "vs Si (chemical stability)")
