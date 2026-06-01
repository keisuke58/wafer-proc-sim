"""
wafer-proc-sim — Physics-Informed ML Portfolio for SiC Semiconductor Processing
================================================================================
Academic report-style Streamlit dashboard covering the full SiC value chain:
material growth → wafer dicing → gate oxidation → packaging → market context.

Each section contains: objective, physical model, interactive simulation,
validation data, and literature references.

Streamlit Cloud entry point.  Run locally with:
    streamlit run app.py
"""

import os
import sys

# Streamlit Cloud clones to /mount/src/<repo> — ensure local imports work.
_root = os.path.abspath(os.path.dirname(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="wafer-proc-sim | SiC Process Physics Portfolio",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: tighten metrics, improve academic look ────────────────────────
st.markdown("""
<style>
[data-testid="metric-container"] {padding: 4px 0;}
.stExpander summary {font-weight: 600;}
h1 {border-bottom: 2px solid #4a90d9; padding-bottom: 0.3em;}
h2 {color: #2c5282;}
.caption-text {font-size: 0.82em; color: #666; font-style: italic;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("💎 wafer-proc-sim")
st.sidebar.markdown(
    "_Physics-Informed ML Portfolio_  \n"
    "SiC semiconductor process simulation  \n"
    "from crystal growth to packaging."
)
st.sidebar.divider()

_PAGES = [
    "🏠 Overview & Abstract",
    "─── Dicing Process (Disco) ───",
    "§1  GP Chipping Surrogate",
    "§2  TMCMC Recipe Correction",
    "§3  Process Capability (Cpk)",
    "§4  Blade Wear Model",
    "§5  Cost-per-Die Optimisation",
    "─── Fab / Materials ───",
    "§6  TEL Cleaning → Dit → µ",
    "§7  Crystal Anisotropy (4H-SiC)",
    "─── Packaging / Reliability ───",
    "§8  Wire Bond Reliability",
    "§9  Die Strength (Griffith/Weibull)",
    "─── Quality & Market ───",
    "§10 Anomaly Detection (3-Layer)",
    "§11 Market Analysis (SiC)",
]
_NAV = [p for p in _PAGES if not p.startswith("───")]

page = st.sidebar.radio("Section", _NAV, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption("Keisuke Nishioka · 2026  \nClaudeCode + Streamlit Cloud")


# ─────────────────────────────────────────────────────────────────────────────
# Cached model loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_gp():
    from ml.train_from_experimental import ExperimentalGPSurrogate, MODEL_PATH
    return ExperimentalGPSurrogate.load(MODEL_PATH)

@st.cache_resource
def load_anomaly():
    from ml.anomaly_detection import AnomalyDetector
    det = AnomalyDetector()
    det.fit(np.array([[80., 23.], [150., 23.], [220., 23.], [290., 23.], [360., 23.]]))
    return det


# ─────────────────────────────────────────────────────────────────────────────
def _ref(text: str):
    """Render a compact reference line."""
    st.markdown(f"<div class='caption-text'>📚 {text}</div>", unsafe_allow_html=True)


def _fig_caption(text: str):
    st.caption(f"**Figure** — {text}")


# ══════════════════════════════════════════════════════════════════════════════
# §0  Overview & Abstract
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview & Abstract":
    st.title("💎 wafer-proc-sim — Physics-Informed ML Portfolio")
    st.markdown(
        "**SiC Semiconductor Process Simulation: Crystal Growth to Packaging**  \n"
        "Keisuke Nishioka · 2026 · [GitHub](https://github.com/nishioka/wafer-proc-sim)"
    )

    with st.expander("📄 Abstract", expanded=True):
        st.markdown("""
We present **wafer-proc-sim**, a fully physics-based simulation portfolio covering the complete
SiC semiconductor value chain.  Eleven interactive modules implement validated physical models
spanning: (i) Gaussian Process surrogate modelling of blade-induced chipping
(LOO R² = 0.64, trained on Micro2026 + Mat2022 experimental data);
(ii) Bayesian inverse inference (TMCMC) for real-time recipe correction;
(iii) process capability analysis (Cpk/Cp/Cpm); (iv) TEL wet-cleaning physics
with Dit → µ_inv mapping via Matthiessen's rule (calibrated to Chung et al. 2001);
(v) 4H-SiC crystal anisotropy model (K_Ic(θ), validated against Opt. Laser Technol. 2024);
(vi) wire bond reliability via Weibull and Coffin-Manson fatigue;
(vii) die fracture strength via Griffith/Weibull statistics;
and (viii) a three-layer anomaly detection system.
All models are validated against published experimental data or manufacturer specifications.
        """)

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Physics Models", "25+", "frontend → packaging")
    col2.metric("Companies Covered", "12", "full value chain")
    col3.metric("GP LOO R²", "0.64", "Fusion GP (26+5 pts)")
    col4.metric("Validation Items", "5 PASS", "vs. literature data")

    st.divider()
    st.markdown("#### Pipeline Overview")
    st.code("""
Shin-Etsu (Si CZ) / Ferrotec (SiC PVT)          ← §M  Crystal Growth
        ↓
ASML EUV Lithography → TEL Cleaning → ALD/Oxide  ← §6  Fab Process
        ↓
Disco Dicing (Blade / Stealth / Bessel)           ← §1–§5  Core Dicing
    + Crystal Anisotropy {10-10} vs {11-20}       ← §7
        ↓
Lasertec Inspection → Advantest ATE               ← Quality Assurance
        ↓
K&S/Besi Wire Bond / Hybrid Bonding               ← §8–§9  Packaging
        ↓
Die Fracture Reliability (Griffith / Weibull)     ← §9
    """, language="text")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Industry Coverage")
        st.markdown("""
| Company / Process | Module | Physics Basis |
|---|---|---|
| **Disco** — dicing/grinding | §1–§5, §7 | FEM + GP surrogate |
| **TEL** — wet process | §6 | Arrhenius, Matthiessen |
| **ASML** — lithography | EUV model | Fourier optics |
| **Advantest** — ATE | ATE model | Weibull, Monte Carlo |
| **Lasertec** — inspection | Inspect model | Mie scatter, SNR |
| **K&S / Besi** — packaging | §8–§9 | Coffin-Manson, Griffith |
        """)
    with col_b:
        st.markdown("#### Validation Summary")
        try:
            from validation.quantitative_validation import validate_mosfet_mobility
            _r_mob = validate_mosfet_mobility()
            _mob_r2 = f"{_r_mob['R2']:.3f}"
            _mob_mape = f"{_r_mob['MAPE_%']:.1f}%"
            _mob_status = "✅" if _r_mob["R2"] >= 0.95 else "⚠️"
        except Exception:
            _mob_r2, _mob_mape, _mob_status = "N/A", "N/A", "⚠️"
        st.markdown(f"""
| Module | Metric | Result | Status |
|---|---|---|---|
| §1 Blade Chipping GP | LOO RMSE | 2.38 µm | ✅ |
| §1 Blade Chipping | MAPE vs exp. | 14.3% | ✅ |
| §8 Au-Al IMC growth | MAPE | 2.7% (R²=0.9997) | ✅ |
| §8 Package warpage | MAPE | 6.1% (R²=0.9998) | ✅ |
| §8 Weibull η (wire) | Error | 5.4% | ✅ |
| §6 µ_ch vs Dit | R² / MAPE | {_mob_r2} / {_mob_mape} | {_mob_status} |
        """)

    st.divider()
    st.info("👈 Select a section from the sidebar to explore the interactive simulations.")


# ══════════════════════════════════════════════════════════════════════════════
# §1  GP Chipping Surrogate
# ══════════════════════════════════════════════════════════════════════════════
elif page == "§1  GP Chipping Surrogate":
    st.title("§1  GP Chipping Surrogate")
    st.markdown(
        "**Objective**: Real-time prediction of front chipping from process parameters  \n"
        "using a Gaussian Process surrogate trained on experimental data."
    )
    with st.expander("📖 Physical Model & Methodology", expanded=False):
        st.markdown("""
**Gaussian Process (GP) surrogate** replaces expensive ABAQUS/FEM runs for real-time prediction.

**Model architecture**
- Kernel: Anisotropic RBF + WhiteKernel — captures length-scale heterogeneity across features
- Fusion GP: 26 experimental data points + 5 FEM-derived fracture proxy features → LOO R² = 0.64
- Input features: `[cut_depth_um, blade_W_um, feed_mm_s, spindle_rpm]`
- Output: predicted chipping µ [µm] ± σ (posterior standard deviation)

**Sobol sensitivity analysis** (Saltelli 2010):
| Parameter | S₁ (1st-order) | ST (total) |
|---|---|---|
| cut_depth_um | 0.78 | 0.82 |
| feed_mm_s | 0.25 | 0.29 |
| spindle_rpm | ~0.01 | ~0.02 |

The posterior variance (±2σ band) quantifies prediction uncertainty — wider at extrapolated conditions
or in data-sparse regions.

**Validation**: LOO RMSE = 2.38 µm against Micro2026 experimental dataset.
        """)
        _ref("Huang et al., Micromachines 17(2):187 (2026)  |  "
             "Rasmussen & Williams, Gaussian Processes for Machine Learning, MIT Press (2006)  |  "
             "Saltelli et al., Global Sensitivity Analysis, Wiley (2008)")

    col1, col2 = st.columns([1, 2])
    with col1:
        depth   = st.slider("Cut Depth [µm]",    60,  420, 200, 10)
        blade_w = st.slider("Blade Width [µm]",   20,   55,  23,  1)
        feed    = st.slider("Feed Speed [mm/s]",  0.3, 3.5, 1.0, 0.1)
        spindle = st.slider("Spindle Speed [krpm]", 18, 42, 30, 1)
        usl     = st.number_input("USL [µm]", value=15.0, step=0.5)

    model = load_gp()
    x = np.array([[depth, blade_w, feed, spindle * 1000.0]])
    mu, sigma = model.predict(x, return_std=True)
    chip, unc = float(mu[0]), float(sigma[0])
    cpu = (usl - chip) / (3 * max(unc, 0.1))

    with col1:
        st.divider()
        st.metric("Predicted Chipping", f"{chip:.2f} µm", f"±{unc:.2f} µm (1σ)")
        st.metric("Cpk (one-sided)", f"{cpu:.2f}",
                  "✅ Capable" if cpu >= 1.33 else ("⚠️ Marginal" if cpu >= 1.0 else "❌ Out-of-Control"))
        st.metric("Margin to USL", f"{usl - chip:.2f} µm")

    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        d_arr = np.linspace(60, 420, 80)
        X_d = np.column_stack([d_arr, np.full(80, blade_w),
                                np.full(80, feed), np.full(80, spindle * 1e3)])
        mu_d, sig_d = model.predict(X_d, return_std=True)
        axes[0].plot(d_arr, mu_d, "#2166ac", lw=2, label="GP mean")
        axes[0].fill_between(d_arr, mu_d - 2 * sig_d, mu_d + 2 * sig_d,
                              alpha=0.2, color="#2166ac", label="±2σ")
        axes[0].axhline(usl, color="#d62728", ls="--", lw=1.5, label=f"USL = {usl} µm")
        axes[0].axvline(depth, color="k", ls=":", lw=1.2, label=f"d = {depth} µm")
        axes[0].set_xlabel("Cut Depth [µm]"); axes[0].set_ylabel("Chipping [µm]")
        axes[0].set_title("(a) Depth Sweep"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

        f_arr = np.linspace(0.3, 3.5, 80)
        X_f = np.column_stack([np.full(80, depth), np.full(80, blade_w),
                                f_arr, np.full(80, spindle * 1e3)])
        mu_f, sig_f = model.predict(X_f, return_std=True)
        axes[1].plot(f_arr, mu_f, "#d62728", lw=2, label="GP mean")
        axes[1].fill_between(f_arr, mu_f - 2 * sig_f, mu_f + 2 * sig_f,
                              alpha=0.2, color="#d62728", label="±2σ")
        axes[1].axhline(usl, color="#d62728", ls="--", lw=1.5, label=f"USL = {usl} µm")
        axes[1].axvline(feed, color="k", ls=":", lw=1.2, label=f"f = {feed} mm/s")
        axes[1].set_xlabel("Feed Speed [mm/s]"); axes[1].set_ylabel("Chipping [µm]")
        axes[1].set_title("(b) Feed Sweep"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

        plt.tight_layout()
        st.pyplot(fig); plt.close()
        _fig_caption(
            "(a) GP posterior mean ± 2σ vs cut depth at fixed feed / spindle.  "
            "(b) Same vs feed speed.  Blue/red shading = predictive uncertainty."
        )


# ══════════════════════════════════════════════════════════════════════════════
# §2  TMCMC Recipe Correction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "§2  TMCMC Recipe Correction":
    st.title("§2  TMCMC Real-time Recipe Correction")
    st.markdown(
        "**Objective**: Bayesian inverse inference — given an observed chipping measurement,  \n"
        "infer the posterior distribution of process parameters and compute the optimal recipe."
    )
    with st.expander("📖 Physical Model & Methodology", expanded=False):
        st.markdown("""
**TMCMC (Transitional Markov Chain Monte Carlo)** solves the Bayesian inverse problem:

> Given observed chipping χ_obs, infer the posterior p(depth, feed | χ_obs)

**Likelihood** is defined via the GP surrogate:
$$p(\\chi_{obs} \\mid \\theta) = \\mathcal{N}(\\mu_{GP}(\\theta),\\; \\sigma^2_{GP}(\\theta) + \\sigma^2_{noise})$$

TMCMC transitions from prior to posterior via a sequence of tempered intermediate distributions:
$$p_j(\\theta) \\propto p(\\chi_{obs} \\mid \\theta)^{\\beta_j} \\cdot \\pi(\\theta), \\quad 0 = \\beta_0 < \\cdots < \\beta_N = 1$$

This avoids the direct sampling of the full posterior and is robust to multi-modal distributions.

**Optimal recipe** is found by minimising predicted chipping over the safe parameter space
(P(chipping < USL) ≥ 95%), subject to throughput constraints.
        """)
        _ref("Ching & Chen, J. Eng. Mech. 133(7):816 (2007)  |  "
             "Lye et al., Mech. Syst. Signal Process. 166 (2022)")

    col1, col2 = st.columns([1, 2])
    with col1:
        obs_chip  = st.slider("Observed Chipping [µm]", 1.0, 20.0, 8.5, 0.5)
        n_samples = st.select_slider("TMCMC Samples", [200, 400, 600, 1000], 400)
        run_btn   = st.button("🚀 Run Inference", type="primary")

    if run_btn:
        with st.spinner("Running TMCMC inference…"):
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
            st.metric("Optimal Depth",  f"{opt['opt_depth_um']:.0f} µm")
            st.metric("Optimal Feed",   f"{opt['opt_feed_mm_s']:.2f} mm/s")
            st.metric("Pred. Chipping", f"{opt['opt_chip_pred']:.1f} ± {opt['opt_chip_std']:.1f} µm")
            st.metric("Safe Zone",      f"{opt['safe_fraction']*100:.0f}%")

        with col2:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            samples = post["samples"]
            axes[0].hist2d(samples[:, 0], samples[:, 1], bins=25, cmap="Blues", density=True)
            axes[0].axvline(post["depth_mean"], color="#d62728", lw=2, ls="--",
                            label=f"depth = {post['depth_mean']:.0f} µm")
            axes[0].axhline(post["feed_mean"],  color="#2166ac", lw=2, ls="--",
                            label=f"feed = {post['feed_mean']:.2f} mm/s")
            axes[0].set_xlabel("Cut Depth [µm]"); axes[0].set_ylabel("Feed [mm/s]")
            axes[0].set_title(f"(a) TMCMC Posterior  (χ_obs = {obs_chip} µm)")
            axes[0].legend(fontsize=8)

            depths = np.linspace(80, 390, 50); feeds = np.linspace(0.5, 3.0, 50)
            D, F = np.meshgrid(depths, feeds)
            X = np.column_stack([D.ravel(), np.full(D.size, 23),
                                  F.ravel(), np.full(D.size, 30000)])
            mu_g, _ = model.predict(X, return_std=True)
            im = axes[1].contourf(D, F, mu_g.reshape(D.shape), levels=15, cmap="YlOrRd")
            plt.colorbar(im, ax=axes[1], label="Chipping [µm]")
            axes[1].contour(D, F, mu_g.reshape(D.shape), levels=[15],
                            colors="white", linewidths=2)
            axes[1].scatter(opt["opt_depth_um"], opt["opt_feed_mm_s"],
                            color="lime", s=200, marker="*", zorder=5, edgecolors="k",
                            label="Optimal ★")
            axes[1].scatter(post["depth_mean"], post["feed_mean"],
                            color="cyan", s=100, marker="D", zorder=5, edgecolors="k",
                            label="MAP ◆")
            axes[1].set_xlabel("Cut Depth [µm]"); axes[1].set_ylabel("Feed [mm/s]")
            axes[1].set_title("(b) GP Response Surface + Optimal Recipe")
            axes[1].legend(fontsize=8)

            plt.tight_layout(); st.pyplot(fig); plt.close()
            _fig_caption(
                "(a) Joint posterior p(depth, feed | χ_obs) from TMCMC sampling.  "
                "(b) GP mean chipping contour; ★ = cost-optimal recipe; ◆ = MAP estimate."
            )


# ══════════════════════════════════════════════════════════════════════════════
# §3  Process Capability (Cpk)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "§3  Process Capability (Cpk)":
    st.title("§3  Process Capability Analysis")
    st.markdown(
        "**Objective**: Compute Cp, Cpk, Cpm indices from GP-predicted chipping distribution  \n"
        "to assess whether the current recipe meets SPC quality requirements."
    )
    with st.expander("📖 Physical Model & Methodology", expanded=False):
        st.markdown("""
**Capability indices** (AIAG 4th ed.):

| Index | Formula | Interpretation |
|---|---|---|
| **Cp** | (USL − LSL) / (6σ) | Spread only — ignores mean offset |
| **Cpk** | min((USL − µ)/3σ, (µ − LSL)/3σ) | Accounts for process mean shift |
| **Cpm** | Cp / √(1 + (µ − T)²/σ²) | Taguchi index — penalises deviation from target T |

**Semiconductor convention**: Cpk ≥ 1.33 = capable; ≥ 1.67 = Six-Sigma production quality.

**Total process variance** combines GP prediction uncertainty and measurement noise:
$$\\sigma_{total} = \\sqrt{\\sigma^2_{GP} + \\sigma^2_{noise}}$$

This conservative estimate ensures the capability index accounts for both modelling
and instrumentation uncertainty sources.
        """)
        _ref("AIAG SPC Manual, 4th ed. (2005)  |  "
             "Montgomery, Introduction to Statistical Quality Control, 8th ed. (2019)")

    col1, col2 = st.columns([1, 2])
    with col1:
        depth2 = st.slider("Cut Depth [µm]",    60, 420, 200, 10, key="cpk_depth")
        feed2  = st.slider("Feed Speed [mm/s]", 0.3, 3.5, 1.0, 0.1, key="cpk_feed")
        usl2   = st.number_input("USL [µm]", value=15.0, step=0.5, key="cpk_usl")
        n_sim  = st.slider("Simulated wafers", 50, 500, 200, 50)
        noise  = st.slider("Measurement noise σ [µm]", 0.5, 3.0, 1.5, 0.1)

    from optimization.process_capability import simulate_production_cpk
    model = load_gp()
    recipe = [depth2, 23.0, feed2, 30000.0]
    result = simulate_production_cpk(model, recipe, n_wafers=n_sim,
                                      noise_std=noise, usl=usl2)

    with col1:
        st.divider()
        st.metric("Cpk", f"{result.cpk:.3f}", result.status)
        st.metric("Cp",  f"{result.cp:.3f}")
        st.metric("Cpm", f"{result.cpm:.3f}")
        st.metric("Process Mean µ", f"{result.mean:.2f} µm")
        st.metric("Process Std σ",  f"{result.std:.2f} µm")
        st.metric("Estimated PPM",  f"{result.ppm_estimated:.0f}")

    with col2:
        x = np.array([recipe])
        mu_v, sig_v = model.predict(x, return_std=True)
        rng = np.random.default_rng(0)
        total_s = np.sqrt(float(sig_v[0]) ** 2 + noise ** 2)
        data = np.maximum(rng.normal(float(mu_v[0]), total_s, n_sim), 0.0)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(data, "o-", color="#2166ac", ms=3, lw=1, alpha=0.8)
        axes[0].axhline(result.mean, color="k", lw=1.5, label=f"µ = {result.mean:.2f} µm")
        axes[0].axhline(usl2, color="#d62728", ls="--", lw=1.5, label=f"USL = {usl2} µm")
        axes[0].axhline(result.mean + 3 * result.std, color="orange", ls="-.",
                        lw=1, label="UCL = µ + 3σ")
        axes[0].set_xlabel("Wafer #"); axes[0].set_ylabel("Chipping [µm]")
        axes[0].set_title(f"(a) Individual-chart  (Cpk = {result.cpk:.2f})")
        axes[0].legend(fontsize=7); axes[0].grid(alpha=0.25)

        from scipy.stats import norm as scipy_norm
        x_r = np.linspace(max(0, result.mean - 4 * result.std),
                           result.mean + 4 * result.std, 200)
        axes[1].hist(data, bins=25, density=True, color="#2166ac",
                     alpha=0.6, edgecolor="k", lw=0.3)
        axes[1].plot(x_r, scipy_norm.pdf(x_r, result.mean, result.std), "k-", lw=2)
        axes[1].axvline(usl2, color="#d62728", ls="--", lw=2, label=f"USL = {usl2} µm")
        axes[1].axvline(result.mean, color="k", lw=1.5, label=f"µ = {result.mean:.2f} µm")
        axes[1].set_xlabel("Chipping [µm]"); axes[1].set_ylabel("Probability Density")
        axes[1].set_title(f"(b) Process Distribution  (Cpk = {result.cpk:.2f})")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

        plt.tight_layout(); st.pyplot(fig); plt.close()
        _fig_caption(
            "(a) Individual measurements vs. upper control limit (UCL = µ + 3σ) and USL.  "
            "(b) Empirical histogram with fitted normal distribution."
        )


# ══════════════════════════════════════════════════════════════════════════════
# §4  Blade Wear Model
# ══════════════════════════════════════════════════════════════════════════════
elif page == "§4  Blade Wear Model":
    st.title("§4  Blade Wear Model")
    st.markdown(
        "**Objective**: Predict progressive blade degradation and schedule preventive replacement  \n"
        "before chipping exceeds the USL."
    )
    with st.expander("📖 Physical Model & Methodology", expanded=False):
        st.markdown("""
Blade wear in SiC dicing follows an **exponential degradation** model driven by progressive
diamond grit dulling and bond erosion.

**Degradation equation**:
$$\\chi(n) = \\chi_0 \\cdot \\exp\\!\\left(\\alpha \\cdot \\frac{n}{N_{life}}\\right)$$

| Parameter | Description | Typical range |
|---|---|---|
| χ₀ | Initial chipping at blade start [µm] | 3–6 µm |
| α | Wear coefficient | 1–3 (SiC harder ≫ Si) |
| N_life | Nominal blade life [wafers] | 100–400 |

**Physical basis**: 4H-SiC (Vickers hardness HV 2580) wears the diamond grit ~2.4× faster
than Si (HV 1100). The exponential form captures accelerating wear as sharp grit edges are lost.

**Replacement strategy**:
- **Warning threshold** = USL × 0.8 (early alarm)
- **Hard replacement** at n where χ(n) = USL
        """)
        _ref("Pei et al., J. Mater. Process. Technol. 129:251 (2002)  |  "
             "Kim & Oh, Int. J. Mach. Tools Manuf. 48:681 (2008)")

    from optimization.blade_wear import BladeWearModel
    col1, col2 = st.columns([1, 2])
    with col1:
        chip0  = st.slider("Initial chipping χ₀ [µm]", 1.0, 8.0, 4.5, 0.1)
        alpha  = st.slider("Wear coefficient α", 0.5, 5.0, 2.0, 0.1)
        n_life = st.slider("Blade life N_life [wafers]", 50, 400, 200, 10)
        usl_w  = st.number_input("USL [µm]", value=15.0, step=0.5, key="wear_usl")
        n_cur  = st.slider("Current wafer count n", 0, n_life, 50, 5)

    model_w = BladeWearModel(chip_0=chip0, n_life=n_life, alpha=alpha, usl=usl_w)
    status  = model_w.status(n_cur)

    with col1:
        st.divider()
        st.metric("Current Chipping χ(n)", f"{status['chip_now']:.2f} µm")
        st.metric("Replace at",  f"wafer #{status['replace_at']}")
        st.metric("Wafers to USL", f"{status['n_to_usl']:.0f}")
        st.info(status["alert"])

    with col2:
        n_arr    = np.linspace(0, n_life * 1.1, 200)
        chip_arr = model_w.predict_array(n_arr)
        fig, ax  = plt.subplots(figsize=(9, 4))
        ax.plot(n_arr, chip_arr, "#d62728", lw=2.5, label="Wear curve χ(n)")
        ax.axhline(usl_w, color="#d62728", ls="--", lw=1.5, label=f"USL = {usl_w} µm")
        ax.axhline(model_w.warning, color="orange", ls=":", lw=1.2, label="Warning (0.8×USL)")
        ax.axvline(n_cur, color="k", ls="-", lw=1.5, label=f"Current (n = {n_cur})")
        ax.axvline(status["replace_at"], color="#2ca02c", ls="--", lw=1.5,
                   label=f"Replace at #{status['replace_at']}")
        ax.scatter([n_cur], [status["chip_now"]], color="k", s=100, zorder=5)
        ax.set_xlabel("Wafers on Blade"); ax.set_ylabel("Chipping [µm]")
        ax.set_title(f"Blade Wear Curve  (χ₀={chip0} µm, α={alpha}, N={n_life})")
        ax.legend(fontsize=8); ax.grid(alpha=0.25); ax.set_ylim(0, usl_w * 1.5)
        st.pyplot(fig); plt.close()
        _fig_caption(
            "Exponential wear curve χ(n) = χ₀·exp(α·n/N_life).  "
            "Orange dotted = warning threshold; green dashed = scheduled replacement."
        )


# ══════════════════════════════════════════════════════════════════════════════
# §5  Cost-per-Die Optimisation
# ══════════════════════════════════════════════════════════════════════════════
elif page == "§5  Cost-per-Die Optimisation":
    st.title("§5  Cost-per-Die Optimisation")
    st.markdown(
        "**Objective**: Minimise total cost per die by jointly optimising feed speed against  \n"
        "blade amortisation, machine time, and yield loss from chipping."
    )
    with st.expander("📖 Physical Model & Methodology", expanded=False):
        st.markdown("""
**Total cost per die** = C_blade + C_machine + C_yield_loss

| Term | Formula | Driver |
|---|---|---|
| C_blade | blade_cost / (N_life × dies_per_wafer) | Blade consumption rate |
| C_machine | (machine_cost/hr) / throughput [dies/hr] | Cycle time |
| C_yield_loss | die_value × P(defect) | GP-predicted chipping → yield |

**Throughput** = f(feed_speed, wafer_diameter, street_pitch).
**P(defect)** = Φ((χ_GP − USL) / σ_total) — normal CDF mapping of GP chipping to yield loss.

**Optimisation**: The model finds the feed speed that minimises C_total analytically.
Fast feed reduces machine time but raises chipping → trade-off drives a well-defined minimum.
        """)
        _ref("Gamberi et al., Int. J. Prod. Econ. 131:349 (2011)  |  "
             "Mach & Janda, Microelectron. Reliab. 44:747 (2004)")

    from optimization.cost_per_die import CostPerDieModel
    col1, col2 = st.columns([1, 2])
    with col1:
        depth3  = st.slider("Cut Depth [µm]",    80, 390, 200, 10, key="c_depth")
        feed3   = st.slider("Feed Speed [mm/s]", 0.5, 3.5, 1.0, 0.1, key="c_feed")
        blade_c = st.number_input("Blade Cost [¥]", value=8000, step=500)
        n_life3 = st.slider("Blade Life [wafers]", 50, 400, 200, 10, key="c_nlife")
        mach_c  = st.number_input("Machine Cost [¥/hr]", value=15000, step=1000)

    costs  = {"blade_cost_jpy": blade_c, "blade_life_wafers": n_life3,
               "machine_cost_jpy_hr": mach_c}
    cm     = CostPerDieModel(costs)
    model  = load_gp()
    x      = np.array([[depth3, 23.0, feed3, 30000.0]])
    mu_c, sig_c = model.predict(x, return_std=True)
    r = cm.compute({"cut_depth_um": depth3, "blade_W_um": 23., "feed_mm_s": feed3,
                     "spindle_rpm": 30000.}, float(mu_c[0]), float(sig_c[0]))

    with col1:
        st.divider()
        st.metric("Total Cost/Die", f"¥{r['cost_total_jpy']:.2f}")
        st.metric("  Blade share",  f"¥{r['cost_blade_jpy']:.3f}")
        st.metric("  Machine share",f"¥{r['cost_machine_jpy']:.3f}")
        st.metric("  Yield loss",   f"¥{r['cost_yield_jpy']:.3f}")
        st.metric("Throughput",     f"{r['wph']:.0f} WPH")
        st.metric("Defect Rate",    f"{r['p_defect']*1e6:.0f} PPM")

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

        opt_feed = feeds_p[int(np.argmin(costs_p))]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(feeds_p, costs_p, "#d62728", lw=2)
        axes[0].axvline(feed3, color="k", ls=":", lw=1.5, label=f"Current = {feed3} mm/s")
        axes[0].axvline(opt_feed, color="#2ca02c", ls="--", lw=1.5,
                         label=f"Optimal = {opt_feed:.1f} mm/s")
        axes[0].set_xlabel("Feed Speed [mm/s]"); axes[0].set_ylabel("Cost/Die [¥]")
        axes[0].set_title("(a) Cost vs Feed Speed"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

        axes[1].plot(feeds_p, throughput_p, "#2166ac", lw=2)
        axes[1].axvline(feed3, color="k", ls=":", lw=1.5, label=f"Current = {feed3} mm/s")
        axes[1].set_xlabel("Feed Speed [mm/s]"); axes[1].set_ylabel("Throughput [WPH]")
        axes[1].set_title("(b) Throughput vs Feed Speed"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

        plt.tight_layout(); st.pyplot(fig); plt.close()
        _fig_caption(
            "(a) Total cost per die vs feed speed; green dashed = cost-optimal feed.  "
            "(b) Corresponding throughput (wafers per hour)."
        )


# ══════════════════════════════════════════════════════════════════════════════
# §6  TEL Cleaning → Dit → µ_inv
# ══════════════════════════════════════════════════════════════════════════════
elif page == "§6  TEL Cleaning → Dit → µ":
    st.title("§6  TEL Cleaning → Interface Trap → Channel Mobility")
    st.markdown(
        "**Objective**: Model the full chain: cleaning sequence → SiC/SiO₂ interface trap density  \n"
        "(Dit) → inversion channel mobility µ_inv via Matthiessen's rule."
    )
    with st.expander("📖 Physical Model & Methodology", expanded=False):
        st.markdown("""
**Why SiC cleaning is harder than Si**:
SiC is chemically inert — standard RCA (SC-1/SC-2) removes organic/metal contamination
but leaves carbon clusters (C-clusters) at the surface. These C-clusters are the primary
precursor to interface traps at the SiO₂/SiC interface, limiting inversion layer mobility.

**Carbon cluster removal model**: Each cleaning step reduces carbon contamination via
a first-order reaction. Steps with oxidising species (H₂SO₄/H₂O₂, O₃) are most effective.

**Dit → µ_inv (Matthiessen's rule, calibrated to Chung et al. 2001)**:
$$\\frac{1}{\\mu_{inv}} = \\frac{1}{\\mu_{phonon}} + \\frac{B}{D_{it}} + \\frac{1}{\\mu_{roughness}}$$

| Scattering mechanism | µ_limit [cm²/Vs] | Physical origin |
|---|---|---|
| Surface phonon | ~120 | Acoustic phonon scattering |
| Coulomb (Dit) | B/Dit | Trapped charge Coulomb scattering |
| Surface roughness | ~200 | Interface roughness (Ra²) |

**Calibration**: Dit = 1×10¹² cm⁻²eV⁻¹ → µ_inv ≈ 35 cm²/Vs ✓ (NO anneal reference, Chung 2001).

*Note: Bulk Si mobility (1400 cm²/Vs) is irrelevant for SiC MOSFETs — inversion layer
mobility is limited to 20–80 cm²/Vs by interface scattering.*
        """)
        _ref("Chung et al., IEEE Electron Device Lett. 22(4):176 (2001)  |  "
             "Kimoto & Cooper, Fundamentals of Silicon Carbide Technology, Wiley (2014)  |  "
             "Sze & Ng, Physics of Semiconductor Devices, 3rd ed. (2007)")

    col1, col2 = st.columns([1, 2])
    with col1:
        seq_options = {
            "Pre-Gate SiC optimised (Piranha+HF+SC2+O₃+HF)": "pregate_sic",
            "Pre-Gate Si standard RCA (SC1+HF+SC2+HF)":       "pregate_si",
            "Post-CMP (H₂O+Megasonic+HF+SC2)":               "post_cmp",
            "Post-Dicing (H₂O+Megasonic+SC2)":                "post_dicing",
        }
        seq_label = st.selectbox("Cleaning Sequence", list(seq_options.keys()))
        seq_name  = seq_options[seq_label]

        dicing_options = {"Blade":       "blade",    "Laser ps":  "laser_ps",
                          "Stealth":     "stealth",  "Post-CMP":  "post_cmp"}
        dice_label = st.selectbox("Dicing Process", list(dicing_options.keys()))
        dice_proc  = dicing_options[dice_label]

        anneal_opt = st.selectbox("Gate Oxidation Anneal", ["none", "NO (1175°C)", "N2O", "POCl3"])
        anneal_key = anneal_opt.split()[0] if " " in anneal_opt else anneal_opt
        T_ox = st.slider("Thermal Oxide Temperature [°C]", 1050, 1200, 1150, 10)
        t_ox = st.slider("Thermal Oxide Time [h]", 0.5, 5.0, 2.0, 0.5)

    from fem.tel_cleaning_model import run_sequence
    from fem.tel_process_model import dit_from_oxidation, channel_mobility_inv, ald_film

    r_clean  = run_sequence(seq_name, dice_proc)
    Dit_clean = r_clean["after"]["dit_contrib"]
    Dit_ox    = dit_from_oxidation(T_ox, t_ox, anneal_T_C=950)
    Dit_total = Dit_clean * 0.2 + Dit_ox
    ald       = ald_film(50, 250, "HfO2")
    mu_inv    = channel_mobility_inv(Dit_total, ald["Cox_fF_um2"], anneal=anneal_key)

    with col1:
        st.divider()
        st.markdown("**Cleaning Efficiency**")
        st.metric("Carbon removal",   f"{r_clean['reduction']['carbon']:.1f} %")
        st.metric("Metal removal",    f"{r_clean['reduction']['metal']:.1f} %")
        st.metric("Particle removal", f"{r_clean['reduction']['particle']:.1f} %")
        st.divider()
        st.markdown("**Interface Trap Density**")
        st.metric("Dit (cleaning origin)", f"{Dit_clean:.2e} cm⁻²eV⁻¹")
        st.metric("Dit (oxidation origin)", f"{Dit_ox:.2e} cm⁻²eV⁻¹")
        st.metric("Dit total",             f"{Dit_total:.2e} cm⁻²eV⁻¹")
        st.divider()
        st.metric("µ_inv [cm²/Vs]", f"{mu_inv:.1f}",
                  "Excellent" if mu_inv > 50 else ("Good" if mu_inv > 25 else "Poor"))
        st.caption("Reference: NO anneal standard ≈ 35 cm²/Vs (Chung et al. 2001)")

    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        dit_arr = np.logspace(9, 14, 200)
        mu_arr  = [channel_mobility_inv(d, ald["Cox_fF_um2"]) for d in dit_arr]
        axes[0].semilogx(dit_arr, mu_arr, "#2166ac", lw=2.5)
        axes[0].scatter([Dit_total], [mu_inv], color="#d62728", s=150, zorder=5,
                         label=f"Current: {mu_inv:.1f} cm²/Vs")
        axes[0].axhline(35, color="green", ls="--", lw=1.5, label="NO anneal ref. ~35")
        axes[0].axhline(67, color="purple", ls=":",  lw=1.5, label="Best (Dit=1e11)")
        axes[0].set_xlabel("Dit [cm⁻²eV⁻¹]"); axes[0].set_ylabel("µ_inv [cm²/Vs]")
        axes[0].set_title("(a) Dit → µ_inv  (Matthiessen's rule)")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.2)

        steps   = ["Initial"] + [h["step"] for h in r_clean["history"]]
        carbons = [r_clean["before"]["carbon"]] + [h["after"]["carbon"] for h in r_clean["history"]]
        axes[1].semilogy(range(len(steps)), carbons, "ro-", lw=2, ms=7)
        axes[1].set_xticks(range(len(steps)))
        axes[1].set_xticklabels(steps, rotation=20, fontsize=8)
        axes[1].set_ylabel("Carbon contamination [cm⁻²]")
        axes[1].set_title("(b) Carbon Removal per Cleaning Step\n(Dit precursor)")
        axes[1].grid(alpha=0.2, which="both")

        plt.tight_layout(); st.pyplot(fig); plt.close()
        _fig_caption(
            "(a) Channel mobility µ_inv as a function of interface trap density Dit  "
            "via Matthiessen's rule; red dot = current condition.  "
            "(b) Carbon contamination evolution through the selected cleaning sequence."
        )


# ══════════════════════════════════════════════════════════════════════════════
# §7  Crystal Anisotropy
# ══════════════════════════════════════════════════════════════════════════════
elif page == "§7  Crystal Anisotropy (4H-SiC)":
    st.title("§7  4H-SiC Crystal Anisotropy — Dicing Direction Optimisation")
    st.markdown(
        "**Objective**: Quantify the direction-dependent fracture toughness K_Ic(θ) of 4H-SiC  \n"
        "and predict its effect on blade-induced chipping."
    )
    with st.expander("📖 Physical Model & Methodology", expanded=False):
        st.markdown("""
4H-SiC has hexagonal symmetry (space group P6₃mc, a = 3.08 Å, c = 10.08 Å).
Fracture toughness K_Ic varies with the crystallographic plane of the cut:

| Plane | Angle θ | K_Ic [MPa√m] | Chipping factor |
|---|---|---|---|
| {10-10} m-plane | 0° | 2.50 | 1.00 (best) |
| {11-20} a-plane | 30° | 2.10 | 1.28 (+28%) |

**Physical reason**: The a-plane {11-20} is a preferred cleavage plane in SiC (Schmid factor
alignment with basal glide), meaning crack propagation is easier → more lateral chipping.
The m-plane does not coincide with a natural cleavage direction.

**Chipping correction model**:
$$\\chi_{corrected}(\\theta) = \\chi_{GP} \\cdot \\left(\\frac{K_{Ic,max}}{K_{Ic}(\\theta)}\\right)^{1.5}$$

The 60° periodicity reflects the 6-fold hexagonal crystal symmetry.
        """)
        _ref("Impact of material anisotropy on ultrafast laser dicing of SiC, "
             "Opt. Laser Technol. (2024)  |  "
             "Strothoff et al., Semicond. Sci. Technol. 36:015007 (2021)")

    from fem.crystal_anisotropy import (
        kic_vs_angle, chipping_factor_vs_angle,
        apply_crystal_correction, optimal_dicing_direction,
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        theta    = st.slider("Dicing direction θ [°]", 0, 60, 0, 1,
                              help="0° = m-plane (best), 30° = a-plane (worst)")
        depth_c  = st.slider("Cut depth [µm]", 20, 300, 100, 10)
        process_c = st.selectbox("Process", ["blade", "stealth_laser"])
        base_chip = 0.52 * depth_c ** 0.65
        corrected = apply_crystal_correction(base_chip, theta)
        kic       = kic_vs_angle(theta)
        cfac      = chipping_factor_vs_angle(theta)

        st.divider()
        st.metric("K_Ic(θ) [MPa√m]", f"{kic:.3f}")
        st.metric("Chipping correction factor", f"×{cfac:.3f}")
        st.metric("Base chipping (isotropic)", f"{base_chip:.2f} µm")
        st.metric("Corrected chipping", f"{corrected:.2f} µm",
                  f"{(cfac - 1)*100:+.1f}% vs m-plane")

        rec = optimal_dicing_direction(process_c)
        st.divider()
        st.success(f"Recommended: {rec['recommended']}")
        st.info(f"Benefit: {rec['benefit']}")

    with col2:
        theta_arr = np.linspace(0, 60, 300)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        kic_arr  = [kic_vs_angle(t) for t in theta_arr]
        chip_arr = [apply_crystal_correction(base_chip, t) for t in theta_arr]

        axes[0].plot(theta_arr, kic_arr, "#2166ac", lw=2.5)
        axes[0].axvline(theta, color="#d62728", ls="--", lw=2, label=f"θ = {theta}°")
        axes[0].axvline(0,  color="green", ls=":", lw=1.5, label="{10-10} m-plane (best)")
        axes[0].axvline(30, color="red",   ls=":", lw=1.5, label="{11-20} a-plane (worst)")
        axes[0].set_xlabel("Dicing direction θ [°]")
        axes[0].set_ylabel("K_Ic [MPa√m]")
        axes[0].set_title("(a) Fracture Toughness Anisotropy")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.2)

        axes[1].plot(theta_arr, chip_arr, "#d62728", lw=2.5)
        axes[1].axvline(theta, color="k", ls="--", lw=2, label=f"θ = {theta}°")
        axes[1].scatter([theta], [corrected], color="k", s=150, zorder=5)
        axes[1].set_xlabel("Dicing direction θ [°]")
        axes[1].set_ylabel("Chipping [µm]")
        axes[1].set_title(f"(b) Chipping vs Direction  (depth = {depth_c} µm)")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.2)

        plt.tight_layout(); st.pyplot(fig); plt.close()
        _fig_caption(
            "(a) Direction-dependent fracture toughness K_Ic(θ) of 4H-SiC showing 60° periodicity.  "
            "(b) Predicted chipping after crystal correction; minimum at θ = 0° (m-plane)."
        )


# ══════════════════════════════════════════════════════════════════════════════
# §8  Wire Bond Reliability
# ══════════════════════════════════════════════════════════════════════════════
elif page == "§8  Wire Bond Reliability":
    st.title("§8  Wire Bond Reliability Prediction")
    st.markdown(
        "**Objective**: Predict wire bond pull strength (Weibull), IMC growth (Arrhenius),  \n"
        "and heel-crack fatigue life (Coffin-Manson) for Au, Cu, and Al wire."
    )
    with st.expander("📖 Physical Model & Methodology", expanded=False):
        st.markdown("""
**Pull strength — Weibull distribution:**
$$P(F \\le f) = 1 - \\exp\\!\\left(-\\left(\\frac{f}{\\eta}\\right)^\\beta\\right)$$

- η (scale): characteristic strength [g] — higher for Cu than Au
- β (shape): failure mode indicator — β > 1: wear-out; typical 6–10 for wire bonds
- **MIL-STD-883 Method 2023**: minimum 3 g for 25 µm wire

**Heel crack fatigue — Coffin-Manson:**
$$N_f = C \\cdot \\varepsilon_p^{-m}$$

Plastic strain amplitude ε_p = ΔCTE · E · ΔT captures thermomechanical cycling damage.

| Material | N_f (ΔT=100 K) | Comment |
|---|---|---|
| Au | ~50,000 cycles | Harman (1997) baseline |
| Cu | ~100,000 cycles | ~2× more fatigue-resistant |
| Al | ~30,000 cycles | Higher CTE mismatch |

**IMC growth — Arrhenius diffusion (Au-Al system):**
$$d(t,T) = \\sqrt{A \\cdot t \\cdot \\exp\\!\\left(-\\frac{Q}{k_B T}\\right)}$$

Activation energy Q = 0.60 eV (grain boundary diffusion, Breach et al. 2004).
Calibration: d = 0.29 µm at 125 °C / 1000 h ✓
        """)
        _ref("Harman, Wire Bonding in Microelectronics, McGraw-Hill (1997)  |  "
             "Breach et al., Microelectron. Reliab. 44:973 (2004)  |  "
             "MIL-STD-883J Method 2023 (2019)")

    from fem.backend_model import (pull_strength_samples, imc_thickness,
                                    heel_crack_life, wire_inductance, WIRE_MATERIALS)
    from scipy.stats import weibull_min

    col1, col2 = st.columns([1, 2])
    with col1:
        wire_sel = st.selectbox("Wire material", ["Au", "Cu", "Al"])
        dT_w     = st.slider("Thermal cycle ΔT [K]", 20, 200, 100, 10)
        loop_h   = st.slider("Loop height [µm]", 100, 400, 200, 10)
        T_store  = st.slider("Storage temperature [°C]", 50, 175, 125, 5)
        t_store  = st.slider("Storage time [h]", 100, 3000, 1000, 100)
        wire_len = st.slider("Wire length [µm]", 500, 4000, 2000, 100)

    m        = WIRE_MATERIALS[wire_sel]
    samples  = pull_strength_samples(wire_sel, n=1000)
    imc_t    = imc_thickness(wire_sel, T_store, t_store)
    heel_N   = heel_crack_life(wire_sel, dT_w, loop_h)
    L_nH     = wire_inductance(loop_h, wire_len)

    with col1:
        st.divider()
        st.metric("Pull strength (mean)", f"{np.mean(samples):.2f} g")
        st.metric("Pull strength (B10)", f"{np.percentile(samples, 10):.2f} g",
                  "✅ MIL-STD-883 ≥ 3 g" if np.percentile(samples, 10) > 3 else "⚠️ Below spec")
        st.metric("IMC thickness", f"{imc_t:.3f} µm",
                  "⚠️ > 1 µm — Kirkendall risk" if imc_t > 1.0 else "OK")
        st.metric("Heel crack fatigue life", f"{heel_N:,.0f} cycles",
                  "✅" if heel_N > 50000 else "⚠️ Below target")
        st.metric("Parasitic inductance", f"{L_nH:.3f} nH")

    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        x_w  = np.linspace(0, 16, 300)
        beta = m["beta"]; eta = m["eta_g"]
        axes[0].hist(samples, bins=30, density=True, color="#2166ac",
                     alpha=0.6, edgecolor="k", lw=0.3, label="Monte Carlo")
        axes[0].plot(x_w, weibull_min.pdf(x_w, beta, scale=eta), "#d62728", lw=2.5,
                     label=f"Weibull η={eta:.1f} g, β={beta:.1f}")
        axes[0].axvline(3.0, color="purple", ls="--", lw=1.5, label="MIL min 3 g")
        axes[0].set_xlabel("Pull strength [g]"); axes[0].set_ylabel("PDF")
        axes[0].set_title(f"(a) {wire_sel} Wire Pull Strength Distribution")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.2)

        dT_arr = np.linspace(20, 200, 100)
        lives  = [heel_crack_life(wire_sel, dT, loop_h) for dT in dT_arr]
        axes[1].semilogy(dT_arr, lives, "#2166ac", lw=2.5)
        axes[1].axhline(50000, color="green", ls="--", lw=1.5, label="Target 50,000 cycles")
        axes[1].axvline(dT_w, color="#d62728", ls=":", lw=2, label=f"ΔT = {dT_w} K")
        axes[1].scatter([dT_w], [heel_N], color="#d62728", s=150, zorder=5)
        axes[1].set_xlabel("ΔT [K]"); axes[1].set_ylabel("Fatigue life [cycles]")
        axes[1].set_title("(b) Heel Crack Fatigue Life  (Coffin-Manson)")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.2, which="both")

        plt.tight_layout(); st.pyplot(fig); plt.close()
        _fig_caption(
            f"(a) Weibull pull strength distribution for {wire_sel} wire; "
            "purple dashed = MIL-STD-883 minimum.  "
            "(b) Coffin-Manson heel crack fatigue life vs thermal cycle amplitude ΔT."
        )


# ══════════════════════════════════════════════════════════════════════════════
# §9  Die Strength (Griffith / Weibull)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "§9  Die Strength (Griffith/Weibull)":
    st.title("§9  Die Fracture Strength — Griffith / Weibull Analysis")
    st.markdown(
        "**Objective**: Link dicing-induced chipping to device-level fracture probability  \n"
        "via Griffith notch mechanics and Weibull statistical failure theory."
    )
    with st.expander("📖 Physical Model & Methodology", expanded=False):
        st.markdown("""
**Pipeline**:
```
GP surrogate → chipping size a [µm]
    ↓  Griffith notch model
Fracture strength σ_f [MPa]
    ↓  Weibull failure probability
P(fracture) under assembly / thermal stress
```

**1. Griffith notch model** (linear elastic fracture mechanics):
$$\\sigma_f = \\frac{K_{Ic}}{Y \\sqrt{\\pi a}}$$

where a = chipping size (edge crack half-length), Y = 1.12 (free-surface correction),
K_Ic(θ) from the crystal anisotropy model (§7).

**2. Weibull failure probability** (two-parameter):
$$P_f(\\sigma) = 1 - \\exp\\!\\left[-\\left(\\frac{\\sigma}{\\sigma_0}\\right)^m\\right]$$

| Material | Weibull m | σ_0 (reference) |
|---|---|---|
| 4H-SiC | 8.0 (tight) | Lawn (1993) |
| Si | 5.5 | Jaccodine (1963) |
| GaN | 4.5 | |

**3. Assembly thermal stress**:
$$\\sigma_{assembly} = \\frac{E \\cdot \\alpha_{CTE} \\cdot \\Delta T}{1 - \\nu}$$

Crystal anisotropy enters via K_Ic(θ) — dicing at a = 30° (a-plane) reduces σ_f by ~18%.
        """)
        _ref("Griffith, Phil. Trans. R. Soc. A 221:163 (1921)  |  "
             "Weibull, J. Appl. Mech. 18:293 (1951)  |  "
             "Lawn, Fracture of Brittle Solids, Cambridge Univ. Press (1993)  |  "
             "Jaccodine, J. Electrochem. Soc. 110:524 (1963)")

    from fem.die_strength_model import (
        fracture_strength, weibull_failure_prob, assembly_failure_rate, MATERIALS, WEIBULL_M,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        mat_sel   = st.selectbox("Die material", ["4H-SiC", "Si", "GaN"])
        chip_um   = st.slider("Chipping size a [µm]", 1.0, 50.0, 10.0, 0.5,
                               help="Edge crack half-length from GP surrogate")
        theta_die = st.slider("Crystal direction θ [°]", 0, 60, 0, 1,
                               help="0° = m-plane (best), 30° = a-plane (worst)")
        sigma_app = st.slider("Assembly stress σ [MPa]", 10, 800, 80, 10,
                               help="Thermal / mechanical stress at die attach / wire bond")

    sigma_f   = fracture_strength(chip_um, mat_sel, theta_die)
    mat_cfg   = MATERIALS[mat_sel]
    m_weibull = WEIBULL_M[mat_sel]
    sigma_0   = sigma_f * 0.8   # Weibull scale: distribution median
    pf        = weibull_failure_prob(sigma_app, sigma_0, m_weibull)
    assy      = assembly_failure_rate(chip_um, mat_sel, theta_die, sigma_app)

    with col1:
        st.divider()
        st.metric("Fracture strength σ_f", f"{sigma_f:.0f} MPa")
        st.metric("Failure probability P_f", f"{pf*100:.2f} %",
                  "✅ < 0.1%" if pf < 1e-3 else ("⚠️ 0.1–1%" if pf < 0.01 else "❌ > 1%"))
        st.metric("Weibull m", f"{m_weibull}")
        st.metric("Assembly failure rate", f"{assy['P_failure_ppm']:.0f} PPM")
        st.metric("Safety margin σ_f / σ_app", f"{assy['safety_margin']:.2f}×",
                  "✅ ≥ 1.5" if assy["safety_margin"] >= 1.5 else "⚠️ < 1.5")

    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        chip_arr   = np.linspace(1, 80, 200)
        sigma_arr  = [fracture_strength(c, mat_sel, theta_die) for c in chip_arr]
        sigma_arr2 = [fracture_strength(c, mat_sel, 30) for c in chip_arr]

        axes[0].plot(chip_arr, sigma_arr,  "#2166ac", lw=2.5, label=f"θ = {theta_die}° (selected)")
        axes[0].plot(chip_arr, sigma_arr2, "#d62728", lw=1.5, ls="--", label="θ = 30° (a-plane)")
        axes[0].axhline(sigma_app, color="orange", ls="-.", lw=1.5,
                         label=f"Applied σ = {sigma_app} MPa")
        axes[0].scatter([chip_um], [sigma_f], color="k", s=150, zorder=5)
        axes[0].set_xlabel("Chipping size a [µm]")
        axes[0].set_ylabel("Fracture strength σ_f [MPa]")
        axes[0].set_title(f"(a) Griffith Fracture Strength  ({mat_sel})")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.2)

        sigma_range = np.linspace(10, sigma_0 * 3, 300)
        pf_arr      = [weibull_failure_prob(s, sigma_0, m_weibull) for s in sigma_range]
        axes[1].semilogy(sigma_range, pf_arr, "#d62728", lw=2.5)
        axes[1].axvline(sigma_app, color="orange", ls="-.", lw=1.5,
                         label=f"Applied σ = {sigma_app} MPa")
        axes[1].axvline(sigma_f,   color="#2166ac", ls="--", lw=2,
                         label=f"σ_f = {sigma_f:.0f} MPa")
        axes[1].scatter([sigma_app], [pf], color="k", s=150, zorder=5)
        axes[1].set_xlabel("Applied stress σ [MPa]")
        axes[1].set_ylabel("Failure probability P_f")
        axes[1].set_title(f"(b) Weibull Failure CDF  (m = {m_weibull})")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.2, which="both")
        axes[1].set_ylim(1e-8, 1.0)

        plt.tight_layout(); st.pyplot(fig); plt.close()
        _fig_caption(
            f"(a) Griffith fracture strength vs chipping crack size for {mat_sel}; "
            "dashed = a-plane orientation (+28% chipping → −18% strength).  "
            "(b) Weibull failure CDF; dot = current operating point."
        )


# ══════════════════════════════════════════════════════════════════════════════
# §10  Anomaly Detection (3-Layer)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "§10 Anomaly Detection (3-Layer)":
    st.title("§10  Anomaly Detection — 3-Layer Monitor")
    st.markdown(
        "**Objective**: Detect process excursions in real-time using three complementary  \n"
        "anomaly detection methods with voting-based alarm logic."
    )
    with st.expander("📖 Physical Model & Methodology", expanded=False):
        st.markdown("""
Three complementary anomaly detection methods with different sensitivity/specificity tradeoffs:

| Layer | Method | Detects | False-positive risk |
|---|---|---|---|
| 1 | **GP z-score** | Statistical outlier vs surrogate prediction | Low (calibrated σ) |
| 2 | **Isolation Forest** | Density-based multivariate anomaly | Medium |
| 3 | **Shewhart chart** | Process shift / drift over time | Low (3σ rule) |

**Alarm policy**: flag only when **≥ 2 layers agree** — minimises false alarms while maintaining
high sensitivity to real excursions.

**Drift simulation**: blade wear introduces a linear trend in the chipping signal.
The 3-layer system is designed to detect this before the USL is breached (early warning at 0.8×USL).
The GP z-score layer is most sensitive to sudden jumps; the Shewhart chart detects gradual drift.
        """)
        _ref("Chandola et al., ACM Comput. Surv. 41(3):15 (2009)  |  "
             "Liu et al., ICDM: 413 (2008) — Isolation Forest  |  "
             "Shewhart, Economic Control of Quality, Van Nostrand (1931)")

    col1, col2 = st.columns([1, 2])
    with col1:
        depth4  = st.slider("Cut Depth [µm]",    80, 390, 200, 10, key="a_depth")
        blade_4 = st.slider("Blade Width [µm]",   20,  55,  23,  1, key="a_bw")
        feed4   = st.slider("Feed Speed [mm/s]", 0.5, 3.5, 1.0, 0.1, key="a_feed")
        spin4   = st.slider("Spindle [krpm]",     18,  42,  30,  1, key="a_spin")
        obs4    = st.slider("Observed Chipping [µm]", 0.5, 20.0, 4.5, 0.5)
        check_btn = st.button("🔍 Check Anomaly", type="primary")

    if check_btn:
        detector = load_anomaly()
        x4 = np.array([depth4, blade_4, feed4, spin4 * 1000.])
        result = detector.check(x4, observed_chip_um=obs4)
        with col1:
            if result.is_anomaly:
                st.error(f"⚠️ ANOMALY DETECTED  (severity = {result.severity:.2f})")
            else:
                st.success("✅ Normal — no anomaly detected")
            st.write("**Layer flags:**", result.layer_flags)
            st.write("**Likely cause:**", result.cause)
            for s in result.suggestions:
                st.warning(f"→ {s}")

    with col2:
        st.subheader("Production Stream Simulation")
        n_lots = st.slider("Number of lots", 20, 100, 50, 5)
        drift  = st.slider("Wear drift [µm/lot]", 0.0, 0.3, 0.1, 0.01)

        model = load_gp()
        x = np.array([[200., 23., 1.0, 30000.]])
        mu0 = float(model.predict(x)[0])
        rng = np.random.default_rng(42)
        chips = [max(0.5, rng.normal(mu0 + i * drift, 1.5)) for i in range(n_lots)]

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(chips, "o-", color="#2166ac", lw=1.5, ms=4, label="Chipping")
        ax.axhline(15.0, color="#d62728", ls="--", lw=1.5, label="USL 15 µm")
        ax.axhline(mu0,  color="#2ca02c", ls=":", lw=1.2, label=f"Initial {mu0:.1f} µm")

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
        ax.set_title(f"Simulated Production  (drift = {drift} µm/lot  |  alerts = {len(alerts)}/{n_lots})")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        st.pyplot(fig); plt.close()
        _fig_caption(
            f"Simulated {n_lots}-lot production stream with {drift} µm/lot blade wear drift.  "
            "Red × marks = lots flagged by ≥ 2 anomaly detection layers."
        )

        if alerts:
            st.warning(f"⚠️ {len(alerts)} anomalies detected at lots: {alerts}")
        else:
            st.success("✅ All lots within normal range")


# ══════════════════════════════════════════════════════════════════════════════
# §11  Market Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "§11 Market Analysis (SiC)":
    st.title("§11  Semiconductor Value Chain — Pipeline Coverage & Market Context")
    st.markdown(
        "**Objective**: Map every company in `run_full_pipeline.py` to its simulation step,  \n"
        "then overlay live stock data and SiC market projections for business context."
    )
    with st.expander("📖 Data Sources & Methodology", expanded=False):
        st.markdown("""
**Coverage**: All 13 companies appearing in `pipeline/run_full_pipeline.py` Steps 1–12.

**Estimated SiC process exposure** (analyst consensus + annual reports, ±5 pp):
- Disco 45%: world-monopoly on SiC dicing/grinding tools
- Lam Research 15%: etch + CVD liner used in SiC power device BEOL
- TEL 18%, Screen 14%: wet clean critical for SiC pre-gate
- AMAT 12%: CMP slurry + ion implant for SiC n-well
- K&S 22%: SiC power module wire bonding (thick Al)
- Lasertec 20%: SiC epi-wafer inspection + EUV mask inspection
- Advantest 12%: SiC MOSFET / SBD ATE testing
- ASML 4%: EUV mainly for leading-edge Si; SiC uses I-line/DUV
- Socionext 8%: fabless automotive SoC — customer of SiC power
- NVIDIA 3%: GPU server PSU uses SiC power modules
- Samsung 5%, SK Hynix 3%: HBM stacking on SiC interposer research

**SiC revenue proxy** = market cap × SiC exposure % / 10  (rough order-of-magnitude only).
*Not financial advice. All figures approximate.*
        """)
        _ref("Yole Développement, SiC Power Device Monitor 2024  |  "
             "SEMI World Fab Forecast 2024  |  Company IR / annual reports  |  "
             "Yahoo Finance via yfinance (15-min delay TSE / KRX, real-time US)")

    # ── Master company table ──────────────────────────────────────────────────
    # cat: Equipment / Package / Logic / Memory
    TICKERS = {
        "ASML":         {"ticker": "ASML",      "color": "#1f77b4", "sic_pct":  4,
                          "step": "§1  EUV Litho",      "cat": "Equipment"},
        "Lasertec":     {"ticker": "6920.T",    "color": "#9467bd", "sic_pct": 20,
                          "step": "§2/§5 Inspection",   "cat": "Equipment"},
        "TEL":          {"ticker": "8035.T",    "color": "#d62728", "sic_pct": 18,
                          "step": "§3/§6 CMP·ALD",      "cat": "Equipment"},
        "Disco":        {"ticker": "6146.T",    "color": "#ff7f0e", "sic_pct": 45,
                          "step": "§4  Dicing",          "cat": "Equipment"},
        "Advantest":    {"ticker": "6857.T",    "color": "#2ca02c", "sic_pct": 12,
                          "step": "§7  ATE Test",        "cat": "Equipment"},
        "Lam Research": {"ticker": "LRCX",      "color": "#17becf", "sic_pct": 15,
                          "step": "§10 Etch·ALD",        "cat": "Equipment"},
        "AMAT":         {"ticker": "AMAT",      "color": "#bcbd22", "sic_pct": 12,
                          "step": "§11 CMP·Implant",     "cat": "Equipment"},
        "Screen":       {"ticker": "7735.T",    "color": "#8c564b", "sic_pct": 14,
                          "step": "Wet Clean",            "cat": "Equipment"},
        "K&S":          {"ticker": "KLIC",      "color": "#7f7f7f", "sic_pct": 22,
                          "step": "§8  Wire Bond",        "cat": "Package"},
        "NVIDIA":       {"ticker": "NVDA",      "color": "#76b900", "sic_pct":  3,
                          "step": "§9  GPU BOM",          "cat": "Logic"},
        "Socionext":    {"ticker": "6526.T",    "color": "#e377c2", "sic_pct":  8,
                          "step": "§8  SoC Timing",       "cat": "Logic"},
        # Samsung (005930.KS) / SK Hynix (000660.KS) appear in Step 12 of the pipeline
        # but are excluded from live stock tracking (KRX-listed, out of scope).
        # Hyperscalers — AI data center customers of SiC power + TSMC wafer consumers
        "Microsoft":    {"ticker": "MSFT",      "color": "#00a4ef", "sic_pct":  3,
                          "step": "CapEx $80B  Maia 200",  "cat": "Hyperscaler"},
        "Alphabet":     {"ticker": "GOOGL",     "color": "#4285f4", "sic_pct":  4,
                          "step": "CapEx $75B  TPU v6e",   "cat": "Hyperscaler"},
        "Meta":         {"ticker": "META",      "color": "#1877f2", "sic_pct":  2,
                          "step": "CapEx $125B MTIA 500",  "cat": "Hyperscaler"},
        "Amazon":       {"ticker": "AMZN",      "color": "#ff9900", "sic_pct":  3,
                          "step": "CapEx $85B  Trainium3", "cat": "Hyperscaler"},
        "Tesla":        {"ticker": "TSLA",      "color": "#cc0000", "sic_pct": 18,
                          "step": "SiC inverter + AI5",    "cat": "Hyperscaler"},
    }

    def _currency(ticker: str) -> str:
        if ticker.endswith(".T"):  return "¥"
        if ticker.endswith(".KS"): return "₩"
        return "$"

    # ── Stock data fetch ──────────────────────────────────────────────────────
    period_opt = st.selectbox("Stock chart period", ["3mo", "6mo", "1y", "2y"], index=2)

    @st.cache_data(ttl=900)
    def fetch_stock_data(period, _tickers_key):
        try:
            import yfinance as yf
        except ImportError:
            return {}
        out = {}
        for name, cfg in TICKERS.items():
            try:
                t    = yf.Ticker(cfg["ticker"])
                hist = t.history(period=period)["Close"].dropna()
                info = t.info
                out[name] = {
                    "hist":    hist,
                    "price":   float(hist.iloc[-1]) if len(hist) > 0 else None,
                    "PE":      info.get("trailingPE"),
                    "mcap_B":  (info.get("marketCap") or 0) / 1e12,
                    "color":   cfg["color"],
                    "sic_pct": cfg["sic_pct"],
                    "cat":     cfg["cat"],
                }
            except Exception:
                out[name] = None
        return out

    with st.spinner("Fetching live data for 16 companies…"):
        stock_data = fetch_stock_data(period_opt, ",".join(TICKERS.keys()))

    if not stock_data:
        st.warning("yfinance not available — showing static coverage table and market projections.")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_cov, tab_price, tab_chart, tab_market, tab_hyper = st.tabs([
        "🗺️ Pipeline Coverage", "💹 Live Prices", "📊 Stock Charts",
        "🏭 SiC Market", "🤖 Hyperscalers",
    ])

    # ── Tab 1: Pipeline Coverage ──────────────────────────────────────────────
    with tab_cov:
        st.markdown("##### All companies in `pipeline/run_full_pipeline.py` mapped to simulation steps")
        import pandas as pd
        cov_rows = []
        for name, cfg in TICKERS.items():
            d = (stock_data or {}).get(name)
            mcap_str = f"${d['mcap_B']:.0f}B" if d and d.get("mcap_B") else "—"
            pe_str   = f"{d['PE']:.1f}×" if d and d.get("PE") else "—"
            cov_rows.append({
                "Company":       name,
                "Category":      cfg["cat"],
                "Pipeline Step": cfg["step"],
                "Ticker":        cfg["ticker"],
                "SiC Exp. %":    cfg["sic_pct"],
                "Mkt Cap":       mcap_str,
                "P/E":           pe_str,
            })
        cov_df = pd.DataFrame(cov_rows)
        # Color-highlight SiC exposure column
        st.dataframe(
            cov_df.style.background_gradient(
                subset=["SiC Exp. %"], cmap="YlOrRd", vmin=0, vmax=50
            ),
            use_container_width=True, hide_index=True,
        )
        _fig_caption(
            "SiC Exp. % = estimated fraction of revenue derived from SiC-related process tools.  "
            "Colour gradient: darker = higher SiC exposure.  "
            "Market cap and P/E from yfinance (15-min delay)."
        )

        # Pipeline flow graphic
        st.markdown("##### Process Flow")
        st.code("""
Step 1  ASML         → EUV aerial image / process window
Step 2  Lasertec     → EUV mask defect density (ACTIS)
Step 3  TEL          → SiC CMP (Preston MRR)
Step 4  Disco        → Dicing: blade / stealth / laser
Step 5  Lasertec     → Post-dicing SiC inspection grade
Step 6  TEL          → ALD HfO₂ + thermal oxidation → Dit → µ_ch
Step 7  Advantest    → ATE yield · CPGD
Step 8  Socionext    → SoC timing yield  |  K&S → wire bond
Step 9  NVIDIA       → GPU BOM equipment cost breakdown
Step 10 Lam Research → Plasma etch ARDE + ALD liner
Step 11 AMAT         → PECVD dep · CMP · ion implant activation
Step 12 Samsung / SK Hynix → HBM3E TSV yield · bandwidth  (KRX-listed, stock data excluded)
        """, language="text")

    # ── Tab 2: Live Prices ────────────────────────────────────────────────────
    with tab_price:
        if stock_data:
            for cat_label, cat_key in [
                ("⚙️ Equipment / Inspection / Test", "Equipment"),
                ("📦 Packaging", "Package"),
                ("💡 Logic & SoC", "Logic"),
                ("🤖 Hyperscalers (AI CapEx / SiC power customers)", "Hyperscaler"),
            ]:
                names_cat = [n for n, cfg in TICKERS.items() if cfg["cat"] == cat_key]
                if not names_cat:
                    continue
                st.markdown(f"**{cat_label}**")
                cols_cat = st.columns(len(names_cat))
                for col, name in zip(cols_cat, names_cat):
                    cfg = TICKERS[name]
                    d   = stock_data.get(name)
                    with col:
                        if d and d.get("price"):
                            cur = _currency(cfg["ticker"])
                            st.metric(
                                f"{name}",
                                f"{cur}{d['price']:,.0f}",
                                f"P/E {d['PE']:.1f}" if d.get("PE") else "P/E —",
                                help=cfg["step"],
                            )
                        else:
                            st.metric(name, "—", help=cfg["step"])
                st.divider()
        else:
            st.info("Stock data unavailable.")

    # ── Tab 3: Stock Charts ───────────────────────────────────────────────────
    with tab_chart:
        if stock_data:
            # Normalised performance — split by category (2×3 grid)
            cat_colors = {"Equipment": "#2166ac", "Package": "#7f7f7f",
                           "Logic": "#2ca02c",   "Hyperscaler": "#cc0000"}
            fig_n, axes_n = plt.subplots(2, 3, figsize=(15, 9), sharex=False)
            cat_order = [
                ("Equipment",   axes_n[0, 0]),
                ("Package",     axes_n[0, 1]),
                ("Logic",       axes_n[0, 2]),
                ("Hyperscaler", axes_n[1, 0]),
            ]
            # hide unused panels
            for ax_unused in [axes_n[1, 1], axes_n[1, 2]]:
                ax_unused.set_visible(False)
            for cat_key, ax in cat_order:
                plotted = 0
                for name, cfg in TICKERS.items():
                    if cfg["cat"] != cat_key:
                        continue
                    d = stock_data.get(name)
                    if d and d.get("hist") is not None and len(d["hist"]) > 1:
                        norm = d["hist"] / d["hist"].iloc[0] * 100
                        ax.plot(norm.index, norm.values, lw=1.8,
                                color=cfg["color"], label=name)
                        plotted += 1
                ax.axhline(100, color="gray", ls="--", lw=1, alpha=0.5)
                ax.set_title(f"{cat_key}  ({plotted} co.)", fontsize=10)
                ax.set_ylabel("Index (start=100)", fontsize=8)
                ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.2)
                ax.tick_params(axis="x", labelsize=7)
            fig_n.suptitle(f"Relative Stock Performance by Category  ({period_opt})",
                           fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_n); plt.close()
            _fig_caption(
                f"Normalised total return (start = 100) over {period_opt}, split by value-chain category.  "
                "Equipment = process tools; Package = wire bonders; Logic = GPU/SoC; Memory = DRAM/HBM."
            )

            st.divider()
            # P/E vs SiC Exposure bubble
            fig_pe, ax_pe = plt.subplots(figsize=(10, 5))
            for name, cfg in TICKERS.items():
                d = stock_data.get(name)
                if d and d.get("PE") and d.get("mcap_B", 0) > 0:
                    size = max(d["mcap_B"] * 1500, 30)
                    ax_pe.scatter(cfg["sic_pct"], d["PE"], s=size,
                                  color=cfg["color"], alpha=0.82,
                                  edgecolors="k", lw=0.8, zorder=5)
                    ax_pe.annotate(name, xy=(cfg["sic_pct"], d["PE"]),
                                   xytext=(cfg["sic_pct"] + 0.5, d["PE"] + 0.8),
                                   fontsize=9)
            ax_pe.set_xlabel("Estimated SiC process exposure [%]", fontsize=11)
            ax_pe.set_ylabel("Trailing P/E ratio", fontsize=11)
            ax_pe.set_title("P/E Ratio vs SiC Exposure  (bubble area ∝ market cap)",
                            fontsize=11)
            ax_pe.grid(alpha=0.2)
            plt.tight_layout(); st.pyplot(fig_pe); plt.close()
            _fig_caption(
                "Bubble area proportional to market capitalisation.  "
                "High SiC exposure + high P/E ≈ premium for SiC-pure-play positioning."
            )
        else:
            st.info("Stock data unavailable.")

    # ── Tab 4: SiC Market ────────────────────────────────────────────────────
    with tab_market:
        YEARS     = np.arange(2022, 2031)
        SIC_TOTAL = np.array([2.0, 2.5, 3.0, 4.2, 5.8, 7.5, 9.2, 11.0, 13.0])
        SIC_EV    = np.array([1.1, 1.4, 1.7, 2.4, 3.3, 4.3, 5.3, 6.3, 7.5])
        cagr      = (SIC_TOTAL[-1] / SIC_TOTAL[1]) ** (1 / 7) - 1

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.fill_between(YEARS, 0, SIC_EV, alpha=0.7, color="#2196f3", label="EV / HEV")
            ax3.fill_between(YEARS, SIC_EV, SIC_TOTAL, alpha=0.5,
                              color="#4caf50", label="Industrial / Other")
            ax3.plot(YEARS, SIC_TOTAL, "ko-", lw=2, ms=5, label="Total")
            ax3.text(2026.5, 10.5, f"CAGR ~{cagr*100:.0f}%", fontsize=12,
                      fontweight="bold", color="#d62728")
            ax3.set_ylabel("Market [$B]"); ax3.legend(fontsize=9)
            ax3.set_title("SiC Power Device Market 2022–2030  (Yole 2024)", fontsize=10)
            ax3.grid(alpha=0.2); plt.tight_layout()
            st.pyplot(fig3); plt.close()
            _fig_caption("SiC power device market 2022–2030; EV/HEV vs Industrial.")

        with col_m2:
            valid = {}
            if stock_data:
                valid = {n: d for n, d in stock_data.items()
                          if d and d.get("mcap_B", 0) > 0}
            if valid:
                # Sort by SiC revenue proxy descending
                sic_rev = {n: valid[n]["mcap_B"] * TICKERS[n]["sic_pct"] / 100 / 10
                            for n in valid}
                sorted_names = sorted(sic_rev, key=lambda n: sic_rev[n])
                vals_sorted  = [sic_rev[n] for n in sorted_names]
                cols_sorted  = [TICKERS[n]["color"] for n in sorted_names]

                fig4, ax4 = plt.subplots(figsize=(6, 4.8))
                bars = ax4.barh(sorted_names, vals_sorted,
                                color=cols_sorted, alpha=0.85, edgecolor="k", lw=0.6)
                for bar, v in zip(bars, vals_sorted):
                    ax4.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                              f"~{v:.1f}B", va="center", fontsize=8)
                ax4.set_xlabel("Est. SiC-related revenue proxy [$B]", fontsize=9)
                ax4.set_title("SiC Revenue Exposure by Company\n(market cap × SiC% ÷ 10)",
                              fontsize=10)
                ax4.grid(alpha=0.2, axis="x"); plt.tight_layout()
                st.pyplot(fig4); plt.close()
                _fig_caption("Sorted by SiC revenue proxy.  "
                              "Disco dominates due to high SiC exposure + large market cap.")
            else:
                st.info("Stock data unavailable — cannot compute revenue proxy chart.")

        st.divider()
        # Market cap comparison
        if stock_data:
            valid_mc = {n: d for n, d in stock_data.items()
                         if d and d.get("mcap_B", 0) > 0}
            if valid_mc:
                sorted_mc = sorted(valid_mc.items(), key=lambda x: x[1]["mcap_B"], reverse=True)
                names_mc  = [x[0] for x in sorted_mc]
                mcap_mc   = [x[1]["mcap_B"] for x in sorted_mc]
                cols_mc   = [TICKERS[n]["color"] for n in names_mc]

                fig5, ax5 = plt.subplots(figsize=(12, 3.5))
                bars5 = ax5.bar(names_mc, mcap_mc, color=cols_mc,
                                alpha=0.85, edgecolor="k", lw=0.6)
                for bar, v in zip(bars5, mcap_mc):
                    ax5.text(bar.get_x() + bar.get_width() / 2, v + max(mcap_mc) * 0.01,
                              f"${v:.0f}B", ha="center", fontsize=8, fontweight="bold")
                ax5.set_ylabel("Market Cap [$B]", fontsize=10)
                ax5.set_title("Market Capitalisation — All Pipeline Companies", fontsize=11)
                ax5.grid(alpha=0.2, axis="y"); plt.tight_layout()
                st.pyplot(fig5); plt.close()
                _fig_caption(
                    "Market cap in USD ($B) for all 16 tracked companies (pipeline + hyperscalers).  "
                    "Apple/Microsoft/Alphabet dominate by cap; Disco is largest Japan pure-play tool maker."
                )

    # ── Tab 5: Hyperscalers ──────────────────────────────────────────────────
    with tab_hyper:
        from fem.hyperscaler_model import (
            ASIC_SPECS, HYPERSCALER_CAPEX, NVIDIA_SHARE,
            tco_analysis, wafer_demand, tesla_fsd_inference_demand,
        )

        st.markdown("##### AI Data Center Custom Silicon — Specs, TCO, and TSMC Wafer Demand")
        st.markdown(
            "Source: `fem/hyperscaler_model.py`  |  "
            "CapEx figures: company IR / analyst consensus (2026E)"
        )

        # Live stock row for hyperscalers
        if stock_data:
            hyper_names = [n for n, cfg in TICKERS.items() if cfg["cat"] == "Hyperscaler"]
            hy_cols = st.columns(len(hyper_names))
            for col, name in zip(hy_cols, hyper_names):
                cfg = TICKERS[name]
                d   = stock_data.get(name)
                with col:
                    if d and d.get("price"):
                        st.metric(
                            f"{name}",
                            f"${d['price']:,.0f}",
                            f"P/E {d['PE']:.1f}" if d.get("PE") else "P/E —",
                            help=cfg["step"],
                        )
                    else:
                        st.metric(name, "—", help=cfg["step"])
            st.divider()

        # ── Chip specs comparison ─────────────────────────────────────────
        col_h1, col_h2 = st.columns([1, 2])
        with col_h1:
            chip_sel = st.selectbox(
                "Select chip for TCO analysis",
                [c for c in ASIC_SPECS if "H100" not in c],
                index=0,
            )
            nvidia_ref = st.selectbox(
                "NVIDIA reference",
                ["NVIDIA H100 SXM", "NVIDIA B200 SXM"],
                index=0,
            )
            tco = tco_analysis(chip_sel, {}, nvidia_ref)
            if tco:
                st.divider()
                s = ASIC_SPECS[chip_sel]
                st.metric("Peak BF16 TFlops",  f"{s['tflops_bf16']:,}")
                st.metric("TDP",               f"{s['tdp_W']} W")
                st.metric("TFlops/W",          f"{tco['flops_per_W']:.2f}")
                st.metric("Die cost (est.)",   f"${tco['die_cost_k']*1000:,.0f}")
                st.metric("Power cost (3 yr)", f"${tco['elec_cost_3yr_k']*1000:,.0f}")
                st.metric("TCO/TFlops vs NVIDIA",
                          f"{tco['tco_vs_h100_pct']:+.0f}%",
                          "Cheaper than ref." if tco["tco_vs_h100_pct"] > 0 else "More expensive")
                st.caption(f"Process: {s['process_nm']} nm  {s['foundry']}  "
                           f"Die: {s['die_area_mm2']} mm²  ({s['year']})")

        with col_h2:
            # TFLOPS/W bars
            chip_names = list(ASIC_SPECS.keys())
            short      = [c.split("\n")[0] for c in chip_names]
            fw         = [ASIC_SPECS[c]["tflops_bf16"] / ASIC_SPECS[c]["tdp_W"]
                          for c in chip_names]
            cols_chips = [ASIC_SPECS[c]["color"] for c in chip_names]

            fig_h1, ax_h1 = plt.subplots(figsize=(7, 4.2))
            bars_h = ax_h1.barh(short, fw, color=cols_chips,
                                height=0.65, edgecolor="k", lw=0.4)
            for bar, v in zip(bars_h, fw):
                ax_h1.text(v + 0.05, bar.get_y() + bar.get_height() / 2,
                           f"{v:.1f}", va="center", fontsize=8)
            ax_h1.set_xlabel("TFlops/W (BF16)", fontsize=10)
            ax_h1.set_title("(a) AI Chip Efficiency — TFlops per Watt", fontsize=10)
            ax_h1.grid(alpha=0.2, axis="x")
            plt.tight_layout(); st.pyplot(fig_h1); plt.close()
            _fig_caption(
                "(a) BF16 performance per watt for NVIDIA H100/B200 vs hyperscaler custom ASICs.  "
                "Google TPU v6e and Tesla AI5 lead on efficiency."
            )

        st.divider()

        # ── CapEx → wafer demand + NVIDIA share erosion ───────────────────
        col_h3, col_h4 = st.columns(2)
        with col_h3:
            demands   = {c: wafer_demand(c, HYPERSCALER_CAPEX[c]) for c in HYPERSCALER_CAPEX}
            hc_names  = list(HYPERSCALER_CAPEX.keys())
            hc_capex  = [HYPERSCALER_CAPEX[c]["capex_B"] for c in hc_names]
            hc_wpd    = [demands[c]["wafers_per_day"] for c in hc_names]
            hc_colors = [HYPERSCALER_CAPEX[c]["color"] for c in hc_names]

            fig_h2, ax_h2 = plt.subplots(figsize=(6, 4))
            x_pos = np.arange(len(hc_names))
            ax_h2b = ax_h2.twinx()
            bars_capex = ax_h2.bar(x_pos - 0.2, hc_capex, 0.4,
                                    color=hc_colors, alpha=0.85, edgecolor="k",
                                    lw=0.5, label="Total CapEx [$B]")
            bars_wpd   = ax_h2b.bar(x_pos + 0.2, hc_wpd, 0.4,
                                     color=hc_colors, alpha=0.45, edgecolor="k",
                                     lw=0.5, hatch="//", label="Wafers/day")
            ax_h2.set_xticks(x_pos)
            ax_h2.set_xticklabels(hc_names, rotation=30, ha="right", fontsize=8)
            ax_h2.set_ylabel("AI CapEx [$B]", fontsize=9)
            ax_h2b.set_ylabel("Est. TSMC wafers/day", fontsize=9, color="#555")
            total_capex = sum(hc_capex)
            ax_h2.set_title(f"(b) AI CapEx 2026E  (total ${total_capex:.0f}B)\n"
                             f"→ TSMC wafer demand (right axis)", fontsize=9)
            ax_h2.grid(alpha=0.2, axis="y")
            plt.tight_layout(); st.pyplot(fig_h2); plt.close()
            _fig_caption(
                f"(b) 2026E AI infrastructure CapEx (left) and implied TSMC wafer demand/day (right, hatched).  "
                f"Total tracked CapEx: ${total_capex:.0f}B."
            )

        with col_h4:
            yrs     = list(NVIDIA_SHARE.keys())
            gpu_sh  = [NVIDIA_SHARE[y][0] for y in yrs]
            asic_sh = [NVIDIA_SHARE[y][1] for y in yrs]
            mkt_B   = [NVIDIA_SHARE[y][2] for y in yrs]

            fig_h3, ax_h3 = plt.subplots(figsize=(6, 4))
            ax_h3.stackplot(yrs, gpu_sh, asic_sh,
                             labels=["NVIDIA GPU", "Custom ASIC (GAFAM + Tesla)"],
                             colors=["#76b900", "#2166ac"], alpha=0.85)
            ax_h3r = ax_h3.twinx()
            ax_h3r.plot(yrs, mkt_B, "o--", color="#d62728", lw=2, ms=6,
                        label="Market [$B]")
            for y, m in zip(yrs, mkt_B):
                ax_h3r.text(y, m + 15, f"${m}B", ha="center", fontsize=7,
                             color="#d62728")
            ax_h3.axvline(2026, color="gray", ls=":", lw=1.5, alpha=0.7,
                           label="2026 (now)")
            ax_h3.set_ylabel("Market share [%]", fontsize=9)
            ax_h3r.set_ylabel("AI chip market [$B]", fontsize=9, color="#d62728")
            ax_h3r.tick_params(colors="#d62728")
            lines1, lb1 = ax_h3.get_legend_handles_labels()
            lines2, lb2 = ax_h3r.get_legend_handles_labels()
            ax_h3.legend(lines1 + lines2, lb1 + lb2, fontsize=7, loc="upper left")
            ax_h3.set_title("(c) NVIDIA GPU vs Custom ASIC Market Share\n"
                             "(Tom's Hardware 2026 + analyst projections)", fontsize=9)
            ax_h3.grid(alpha=0.2)
            plt.tight_layout(); st.pyplot(fig_h3); plt.close()
            _fig_caption(
                "(c) AI accelerator market share evolution 2022–2030.  "
                "Custom ASIC share rising from 8% (2022) to ~60% (2030E).  "
                "Right axis: total market size [$B]."
            )

        # Tesla demand summary
        st.divider()
        tesla = tesla_fsd_inference_demand()
        st.markdown("##### Tesla AI5 Demand Estimate (2026)")
        t_cols = st.columns(4)
        t_cols[0].metric("FSD vehicle chips",    f"{tesla['n_chips_vehicle']:,}")
        t_cols[1].metric("Optimus robot chips",  f"{tesla['n_chips_optimus']:,}")
        t_cols[2].metric("Dojo 3 tiles",         f"{tesla['n_dojo_tiles']:,}")
        t_cols[3].metric("2nm wafers needed",
                          f"{tesla['wafers_needed']:,}",
                          f"~${tesla['wafer_cost_B']:.1f}B")
        _ref(
            "Tom's Hardware (2026) — Custom AI ASIC 70%/30% GPU/ASIC split  |  "
            "TSMC Q1 2026 Earnings — wafer revenue mix  |  "
            "Buildmvpfast.com — Hyperscaler CapEx $770B 2026"
        )

    st.caption(
        "Stock data: Yahoo Finance via yfinance (15-min delay TSE/KRX, real-time US NASDAQ).  "
        "SiC exposure estimates: analyst consensus ± 5 pp.  "
        "Market projections: Yole Développement 2024.  **Not financial advice.**"
    )
