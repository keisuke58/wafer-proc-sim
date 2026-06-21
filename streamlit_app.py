"""
wafer-proc-sim Streamlit demo — four-tab layout:
  Tab 1: SiC Dicing Live Monitor      (blade anomaly detection)
  Tab 2: KABRA® BO Optimizer          (interactive multi-objective BO)
  Tab 3: Physical Process Limits      (first-principles limits explorer)
  Tab 4: Cleavage Anisotropy          (SiC/GaN dicing orientation guide)
"""
import os
import sys

_root = os.path.abspath(os.path.dirname(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="wafer-proc-sim",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Tab 1 helpers ─────────────────────────────────────────────────────────────
SENSORS = {
    "spindle_current_A":    ("Spindle Current [A]",    2.5,  4.00),
    "vibration_rms_g":      ("Vibration RMS [g]",      0.15, 0.60),
    "acoustic_emission_dB": ("Acoustic Emission [dB]", 45.0, 55.0),
    "camera_chip_um":       ("Chipping [µm]",          4.5,  12.0),
}
CHART_H = 210


@st.cache_resource
def _start_sim():
    from dashboard.simulator import get_simulator
    sim = get_simulator()
    sim.start()
    return sim


def _sensor_chart(df, col, title, nominal, usl):
    anom = df["is_anomaly"].values
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["wafer_id"], y=df[col].astype(float),
        mode="lines+markers",
        line=dict(color="#2166ac", width=1.5),
        marker=dict(
            color=["#e31a1c" if a else "#2166ac" for a in anom],
            size=[9 if a else 4 for a in anom],
            symbol=["x" if a else "circle" for a in anom],
        ),
        name=title, showlegend=False,
    ))
    fig.add_hline(y=usl,     line_dash="dash", line_color="#f97316",
                  annotation_text="USL",  annotation_position="top right")
    fig.add_hline(y=nominal, line_dash="dot",  line_color="#aaa",
                  annotation_text="nom",  annotation_position="bottom right")
    fig.update_layout(height=CHART_H, title_text=title, title_font_size=11,
                      margin=dict(l=40, r=10, t=32, b=25),
                      xaxis_title="Wafer #", showlegend=False)
    return fig


def _fusion_chart(df):
    adf = df[df["is_anomaly"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["wafer_id"], y=df["fusion_z"].astype(float),
        mode="lines", fill="tozeroy",
        line=dict(color="#f97316", width=1.5),
        fillcolor="rgba(249,115,22,0.12)", name="fusion z",
    ))
    fig.add_hline(y=3.0, line_dash="dash", line_color="#e31a1c",
                  annotation_text="alert=3.0", annotation_position="top right")
    fig.add_trace(go.Scatter(
        x=adf["wafer_id"], y=adf["fusion_z"].astype(float),
        mode="markers",
        marker=dict(color="#e31a1c", size=10, symbol="x"),
        name="true anomaly", showlegend=True,
    ))
    fig.update_layout(height=CHART_H, title_text="Weighted fusion z-score",
                      title_font_size=11,
                      margin=dict(l=40, r=10, t=32, b=25),
                      xaxis_title="Wafer #", yaxis_title="z")
    return fig


def _f1(pred, truth):
    tp  = ((pred == 1) & (truth == 1)).sum()
    fp  = ((pred == 1) & (truth == 0)).sum()
    fn  = ((pred == 0) & (truth == 1)).sum()
    pr  = tp / (tp + fp + 1e-9)
    rc  = tp / (tp + fn + 1e-9)
    fpr = fp / (len(truth) - truth.sum() + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9), float(fpr)


# ── Tab 2 helpers: KABRA BO ───────────────────────────────────────────────────

def _kabra_response_surface(P_vals, v_vals, d_um):
    """Evaluate the synthetic KABRA model over a grid of (P, v) at fixed depth."""
    try:
        from sic.sic_kabra_gp import _kerf_quality, _rosenthal_haz, _throughput
        haz = np.array([[_rosenthal_haz(p, v) for v in v_vals] for p in P_vals])
        kq  = np.array([[_kerf_quality(p, v, d_um) for v in v_vals] for p in P_vals])
        tp  = np.array([[_throughput(v, p) for v in v_vals] for p in P_vals])
        return haz, kq, tp
    except Exception:
        return None, None, None


def _run_kabra_bo(n_iter, w_haz, w_kq, w_tp):
    """Run TuRBO BO with user-specified weights. Returns result dict."""
    try:
        from sic.sic_kabra_gp import (
            generate_doe, simulate_doe, KABRAGPSurrogate,
            run_bayesian_optimisation, BOUNDS
        )
        X = generate_doe(27)
        Y = simulate_doe(X)
        surrogate = KABRAGPSurrogate()
        surrogate.fit(X, Y)
        weights = {"w_haz": w_haz, "w_kq": w_kq, "w_tp": w_tp,
                   "label": "custom", "description": "User-defined"}
        result = run_bayesian_optimisation(surrogate, weights, n_iter=n_iter,
                                           bounds=list(BOUNDS.values()))
        return result
    except Exception as e:
        return {"error": str(e)}


# ── Tab 3 helpers: Physical Limits ────────────────────────────────────────────

def _get_physical_limits():
    """Run all physical limit models and return a summary DataFrame."""
    try:
        from sic.physical_limits import (
            blade_minimum_kerf,
            stealth_dicing_focal_limits,
            wafer_thinning_critical_thickness,
            plasma_aspect_ratio_limit,
            diamond_laser_ablation_limit,
        )
        rows = []

        r = blade_minimum_kerf()
        rows.append({
            "Process": "Blade Dicing",
            "Metric": "Min. kerf width",
            "Limit": f"{r['kerf_min_um']:.1f} µm",
            "Current": f"{r['current_kerf_um']:.0f} µm",
            "Headroom": f"{r['improvement_pct']:.0f}%",
            "Model": "Euler plate buckling",
        })

        r2 = stealth_dicing_focal_limits()
        rows.append({
            "Process": "Stealth Dicing",
            "Metric": "Min. modified-layer thickness",
            "Limit": f"{r2['modified_layer_min_um']:.1f} µm",
            "Current": f"{r2['current_layer_um']:.0f} µm",
            "Headroom": f"{r2['shrinkage_possible_x']:.1f}×",
            "Model": "Rayleigh length in Si",
        })

        r3 = wafer_thinning_critical_thickness()
        rows.append({
            "Process": "Wafer Thinning",
            "Metric": "Safety factor @ 25 µm",
            "Limit": f"SF = {r3['safety_at_25um']:.1f}",
            "Current": f"HBM4: {r3['current_hbm4_um']:.0f} µm",
            "Headroom": "Large",
            "Model": "Timoshenko plate bending",
        })

        r4 = plasma_aspect_ratio_limit(pressure_mTorr=10.0)
        rows.append({
            "Process": "Plasma DRIE",
            "Metric": "Max. aspect ratio @ 10 mTorr",
            "Limit": f"AR ≤ {r4['AR_max_practical']:.0f}",
            "Current": f"AR ≈ {r4['current_AR']:.0f}",
            "Headroom": f"{(r4['AR_max_practical']/r4['current_AR']-1)*100:.0f}%",
            "Model": "ARDE empirical (JVST 2017)",
        })

        r5 = diamond_laser_ablation_limit()
        rows.append({
            "Process": "Diamond Laser",
            "Metric": "Min. kerf @ 193 nm (ArF)",
            "Limit": f"{r5['min_kerf_193nm_um']:.2f} µm",
            "Current": "N/A",
            "Headroom": "N/A",
            "Model": "Photon energy threshold (E > 5.47 eV)",
        })

        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame([{"Process": "Error", "Metric": str(e),
                               "Limit": "", "Current": "", "Headroom": "", "Model": ""}])


def _arde_plot(pressure_range=(1, 50)):
    """Plot AR_crit(P) and AR_max(P) vs pressure."""
    try:
        from sic.physical_limits import plasma_aspect_ratio_limit
        P_arr = np.linspace(pressure_range[0], pressure_range[1], 100)
        arc_arr = []
        armax_arr = []
        for p in P_arr:
            r = plasma_aspect_ratio_limit(pressure_mTorr=float(p))
            arc_arr.append(r["AR_crit_50pct"])
            armax_arr.append(r["AR_max_practical"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=P_arr, y=armax_arr, mode="lines",
                                  name="AR_max (50% etch rate)",
                                  line=dict(color="#2166ac", width=2)))
        fig.add_trace(go.Scatter(x=P_arr, y=arc_arr, mode="lines",
                                  name="AR_crit (50% rate loss)",
                                  line=dict(color="#f97316", width=2,
                                            dash="dash")))
        fig.add_vline(x=10, line_dash="dot", line_color="gray",
                      annotation_text="10 mTorr ref", annotation_position="top right")
        fig.update_layout(
            title="ARDE: Aspect Ratio Limits vs Pressure",
            xaxis_title="Pressure [mTorr]",
            yaxis_title="Aspect Ratio",
            height=350,
            margin=dict(l=50, r=20, t=50, b=40),
        )
        return fig
    except Exception:
        return go.Figure()


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ wafer-proc-sim")
    st.caption("Physics + GP + BO for semiconductor processes")
    st.divider()
    st.markdown(
        "**Modules**\n"
        "- Blade dicing GP (heteroscedastic)\n"
        "- KABRA® laser BO (TuRBO)\n"
        "- Physical process limits\n"
        "- DRIE ARDE validation\n\n"
        "**Paper**: *Precision Engineering* (submitted)  \n"
        "**DOI**: [10.5281/zenodo.20495459](https://doi.org/10.5281/zenodo.20495459)"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Tabs
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 Live Dicing Monitor",
    "⚡ KABRA® BO Optimizer",
    "📐 Physical Limits",
    "🔬 Cleavage Anisotropy",
])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Live Monitor
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    with st.sidebar:
        st.divider()
        st.subheader("Monitor Config")
        anomaly_rate = st.slider("Anomaly rate",       0.00, 0.50, 0.15, 0.05)
        interval_s   = st.slider("Interval (s/wafer)",  0.5,  5.0,  1.0,  0.5)
        n_life       = st.slider("Blade life (wafers)",  50,  500,  200,   10)
        n_show       = st.slider("History window",        20,  200,   60,   10)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Apply", use_container_width=True, type="primary"):
                try:
                    sim = _start_sim()
                    sim.configure(anomaly_rate=anomaly_rate,
                                  interval_s=interval_s, n_life=n_life)
                    st.success("Applied")
                except Exception:
                    pass
        with c2:
            if st.button("Reset", use_container_width=True):
                try:
                    _start_sim().reset()
                    st.info("Reset")
                except Exception:
                    pass

    @st.fragment(run_every=1.0)
    def live_panel():
        try:
            sim  = _start_sim()
            rows = sim.get_history(n_show)
        except Exception as e:
            st.error(f"Simulator unavailable: {e}")
            return
        if not rows:
            st.info("Warming up — first wafer coming in ~1 s...")
            return

        df  = pd.DataFrame(rows)
        lat = df.iloc[-1]

        st.subheader("SiC Dicing — Live Sensor Monitor")
        k = st.columns(5)
        k[0].metric("Wafer #",        int(lat["wafer_id"]))
        k[1].metric("True chip [µm]", f"{lat['true_chip_um']:.1f}",
                    delta=f"{lat['true_chip_um'] - 4.5:+.1f} vs nominal")
        k[2].metric("Fusion z",       f"{lat['fusion_z']:.2f}",
                    delta="⚠ ALERT" if lat["fusion_flag"] else "OK",
                    delta_color="inverse" if lat["fusion_flag"] else "off")
        k[3].metric("Majority votes", f"{int(lat['majority_votes'])}/5")
        k[4].metric("Ground truth",
                    "ANOMALY" if lat["is_anomaly"] else "OK",
                    delta_color="inverse" if lat["is_anomaly"] else "off")
        st.divider()

        pairs = list(SENSORS.items())
        for row_pair in [pairs[:2], pairs[2:]]:
            cols = st.columns(2)
            for col, (key, (title, nom, usl)) in zip(cols, row_pair):
                col.plotly_chart(_sensor_chart(df, key, title, nom, usl),
                                  use_container_width=True)
        st.divider()

        left, right = st.columns([3, 1])
        with left:
            st.plotly_chart(_fusion_chart(df), use_container_width=True)
        with right:
            truth  = df["is_anomaly"].astype(int).values
            pred_A = df["fusion_flag"].astype(int).values
            pred_D = df["majority_flag"].astype(int).values
            f1_A, fpr_A = _f1(pred_A, truth)
            f1_D, fpr_D = _f1(pred_D, truth)
            st.markdown("**Running metrics**")
            st.metric("Weighted F1",    f"{f1_A:.3f}",
                      delta=f"FPR {fpr_A:.0%}", delta_color="inverse")
            st.metric("Majority F1",    f"{f1_D:.3f}",
                      delta=f"FPR {fpr_D:.0%}", delta_color="inverse")
            st.metric("True anomalies", int(truth.sum()))
            st.metric("Fusion alerts",  int(pred_A.sum()))
            miss = truth.sum() - int(((pred_A == 1) & (truth == 1)).sum())
            st.metric("Miss rate",      f"{miss / max(int(truth.sum()), 1):.0%}")

    live_panel()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: KABRA BO
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("KABRA® Laser-Slicing Bayesian Optimizer")
    st.caption(
        "Multi-objective TuRBO over (laser power, scan speed, focus depth). "
        "Adjust weights to prioritise HAZ reduction, kerf quality, or throughput."
    )

    col_ctrl, col_results = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Weight settings")
        st.markdown("*Weights are normalised internally (sum = 1).*")
        w_haz = st.slider("HAZ weight (minimise)",     0.0, 1.0, 0.40, 0.05,
                           key="w_haz")
        w_kq  = st.slider("Quality weight (maximise)", 0.0, 1.0, 0.40, 0.05,
                           key="w_kq")
        w_tp  = st.slider("Throughput weight (max.)",  0.0, 1.0, 0.20, 0.05,
                           key="w_tp")
        n_iter = st.slider("BO iterations", 5, 30, 15, 5, key="n_iter")

        preset = st.selectbox("Preset", ["Custom", "Quality-first (0.45/0.45/0.10)",
                                          "Balanced (0.40/0.40/0.20)",
                                          "Speed-first (0.10/0.10/0.80)"])
        if preset == "Quality-first (0.45/0.45/0.10)":
            w_haz, w_kq, w_tp = 0.45, 0.45, 0.10
        elif preset == "Balanced (0.40/0.40/0.20)":
            w_haz, w_kq, w_tp = 0.40, 0.40, 0.20
        elif preset == "Speed-first (0.10/0.10/0.80)":
            w_haz, w_kq, w_tp = 0.10, 0.10, 0.80

        run_bo = st.button("Run Optimizer", type="primary",
                            use_container_width=True)

    with col_results:
        # Static response surface (no BO needed)
        st.subheader("Process Map (Quality score at 100 µm focus)")
        P_grid = np.linspace(5, 30, 40)
        v_grid = np.linspace(100, 400, 40)
        focus_val = st.slider("Focus depth [µm]", 50, 150, 100, 10)
        _, kq_grid, _ = _kabra_response_surface(P_grid, v_grid, focus_val)

        if kq_grid is not None:
            fig_map = go.Figure(go.Contour(
                z=kq_grid, x=v_grid, y=P_grid,
                colorscale="RdYlGn", colorbar=dict(title="Quality"),
                contours=dict(showlabels=True),
            ))
            fig_map.update_layout(
                xaxis_title="Scan speed [mm/s]",
                yaxis_title="Laser power [W]",
                title=f"Kerf quality map (focus = {focus_val} µm)",
                height=320, margin=dict(l=50, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("sic_kabra_gp not available.")

    # BO results panel
    if run_bo:
        with st.spinner("Running TuRBO optimization..."):
            res = _run_kabra_bo(n_iter, w_haz, w_kq, w_tp)

        if "error" in res:
            st.error(f"BO error: {res['error']}")
        else:
            st.divider()
            st.subheader("Optimization Results")
            opt = res.get("optimal", {})
            baseline = res.get("baseline", {})

            cols = st.columns(3)
            P_opt = opt.get("P_W", 0)
            v_opt = opt.get("v_mm_s", 0)
            d_opt = opt.get("d_um", 0)
            tp_opt = opt.get("throughput_mm2_s", opt.get("throughput", 0))
            tp_base = baseline.get("throughput_mm2_s", baseline.get("throughput", 173.0))
            delta_tp = (tp_opt - tp_base) / max(tp_base, 1e-9) * 100

            cols[0].metric("Optimal Power", f"{P_opt:.1f} W",
                            delta=f"{P_opt - 15:.1f} W vs 15W")
            cols[1].metric("Optimal Speed", f"{v_opt:.0f} mm/s",
                            delta=f"{v_opt - 200:.0f} mm/s vs 200")
            cols[2].metric("Throughput gain", f"{delta_tp:+.1f}%",
                            delta_color="normal")

            st.markdown(
                f"**Optimal**: {P_opt:.1f} W / {v_opt:.0f} mm/s / {d_opt:.0f} µm focus  \n"
                f"**HAZ**: {opt.get('haz_width_um', '—'):.1f} µm  "
                f"**Quality**: {opt.get('kerf_quality', '—'):.3f}  "
                f"**Throughput**: {tp_opt:.1f} mm²/s  \n"
                f"*Reference (Leeftink 2024): +34% via GP+BO on DISCO LASER1205*"
            )

    # Leeftink comparison table (static)
    st.divider()
    st.subheader("Reference: Leeftink et al. (arXiv:2511.23141)")
    leeftink_df = pd.DataFrame([
        {"Scenario": "BOLD_A (balanced)",  "Throughput [wph]": 3.01, "vs. Baseline": "—"},
        {"Scenario": "BOLD_B (quality)",   "Throughput [wph]": 2.89, "vs. Baseline": "-4%"},
        {"Scenario": "BOLD_C (speed)",     "Throughput [wph]": 3.38, "vs. Baseline": "+12%"},
        {"Scenario": "BOLD+Expert (HitL)", "Throughput [wph]": 4.03, "vs. Baseline": "+34%"},
    ])
    st.dataframe(leeftink_df, use_container_width=True, hide_index=True)
    st.caption("Equipment: DISCO LASER1205, Si wafer 53 µm. HitL = human-in-the-loop.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Physical Limits
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Physical Process Limits")
    st.caption(
        "First-principles models for five DISCO semiconductor processes. "
        "Sources: `sic/physical_limits.py` — validated against HBM fracture data (Materials 2024)."
    )

    # Summary table
    st.subheader("Limit Summary")
    lim_df = _get_physical_limits()
    st.dataframe(lim_df, use_container_width=True, hide_index=True)

    st.divider()

    # Interactive: ARDE pressure dependence
    col_arde, col_hbm = st.columns(2)

    with col_arde:
        st.subheader("DRIE Plasma — ARDE Limits")
        p_max = st.slider("Pressure range max [mTorr]", 10, 100, 50, 5)
        st.plotly_chart(_arde_plot(pressure_range=(1, p_max)),
                         use_container_width=True)
        st.caption(
            "Model: R/R₀ = 1/(1+(AR/AR_crit)²),  AR_crit = 8√(P/10 mTorr)  "
            "(JVST A 35(5):05C301, 2017)"
        )

    with col_hbm:
        st.subheader("Fracture Strength vs Singulation Method")
        methods = ["Stealth Dicing", "Blade Dicing", "Laser Grooving"]
        thicknesses = [60, 90, 120]
        colors = ["#2196F3", "#FF9800", "#F44336"]

        # HBM data (Materials 2024, PMC11595813) — converted from gf units
        hbm_data = {
            "Stealth Dicing":  [375, 114, 66],
            "Blade Dicing":    [159,  49, 37],
            "Laser Grooving":  [ 49,  20,  9],
        }
        a_eff = {
            "Stealth Dicing": 1.6,
            "Blade Dicing":   8.6,
            "Laser Grooving": 91.0,
        }

        fig_hbm = go.Figure()
        for (method, strengths), color in zip(hbm_data.items(), colors):
            fig_hbm.add_trace(go.Scatter(
                x=thicknesses, y=strengths, mode="lines+markers",
                name=method, line=dict(color=color, width=2),
                marker=dict(size=8),
            ))
        fig_hbm.update_layout(
            xaxis_title="Wafer thickness [µm]",
            yaxis_title="Fracture strength [MPa]",
            height=320,
            margin=dict(l=50, r=20, t=20, b=40),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )
        st.plotly_chart(fig_hbm, use_container_width=True)

        st.markdown("**Griffith effective flaw depth** (K_Ic = 0.83 MPa√m):")
        flaw_df = pd.DataFrame([
            {"Method": m, "a_eff [µm]": a}
            for m, a in a_eff.items()
        ])
        st.dataframe(flaw_df, use_container_width=True, hide_index=True)
        st.caption(
            "Data: Kang et al., *Materials* 17(22):5529 (2024). "
            "3-point bending, Si(111) 12-inch, 4×8 mm chips."
        )

    st.divider()
    st.subheader("Blade Dicing: Minimum Kerf vs Cut Depth")
    try:
        from sic.physical_limits import blade_minimum_kerf
        depths  = np.linspace(10, 300, 50)
        kerfs   = [blade_minimum_kerf(cut_depth_um=d)["kerf_min_um"] for d in depths]
        fig_kerf = go.Figure()
        fig_kerf.add_trace(go.Scatter(x=depths, y=kerfs, mode="lines",
                                       line=dict(color="#2166ac", width=2),
                                       name="min. kerf (Euler buckling)"))
        fig_kerf.add_hline(y=23, line_dash="dash", line_color="#f97316",
                            annotation_text="Current: 23 µm",
                            annotation_position="top right")
        fig_kerf.update_layout(
            xaxis_title="Cut depth [µm]",
            yaxis_title="Minimum kerf width [µm]",
            height=280,
            margin=dict(l=50, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_kerf, use_container_width=True)
        st.caption("Model: t_min = a√(12(1-ν²)σ_yield/(π²E)). At 100 µm depth: kerf_min ≈ 7.2 µm.")
    except Exception as e:
        st.error(f"Could not load physical_limits: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Cleavage Anisotropy
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Cleavage Anisotropy — SiC / GaN Dicing Orientation Guide")
    st.caption(
        "Anisotropic AT2 model: Gc_eff(φ) = Gc · (1 + β sin²(φ − θ_cut)). "
        "β < 0 → cleavage is the soft direction. "
        "Cut **along** a cleavage plane (θ_cut → 0° or 90°) to null the "
        "deflection force and minimise chipping."
    )

    # material parameters (SiC β from wafer-proc-sim MAP fit; Si/GaN illustrative)
    _MAT = {
        "SiC": dict(beta=-0.51, color="#b8860b"),
        "GaN": dict(beta=-0.30, color="#c0392b"),
        "Si":  dict(beta=-0.12, color="#1f6f6f"),
    }

    def _gc_eff(phi, theta, beta):
        return 1.0 + beta * np.sin(phi - theta) ** 2

    def _kerf_toughness(theta, beta):
        return 1.0 + beta * np.sin(theta) ** 2

    def _deflection_force(theta, beta):
        return np.abs(beta * np.sin(2.0 * theta))

    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        theta_deg = st.slider("Cut angle θ_cut to cleavage plane [deg]", 0, 90, 45, 1)
        sel_mats  = st.multiselect("Materials", list(_MAT), default=list(_MAT))
        theta_rad = np.deg2rad(theta_deg)
        st.divider()
        st.markdown("**At selected θ_cut:**")
        for n in sel_mats:
            kt = _kerf_toughness(theta_rad, _MAT[n]["beta"])
            df = _deflection_force(theta_rad, _MAT[n]["beta"])
            st.markdown(f"- **{n}** — kerf Gc_eff/Gc = `{kt:.3f}`, deflection force = `{df:.3f}`")

    with col_info:
        th  = np.linspace(0, np.pi / 2, 181)
        phi = np.linspace(0, 2 * np.pi, 361)

        c1, c2 = st.columns(2)

        # (a) kerf toughness vs cut orientation
        fig_kt = go.Figure()
        for n in sel_mats:
            fig_kt.add_trace(go.Scatter(
                x=np.rad2deg(th),
                y=_kerf_toughness(th, _MAT[n]["beta"]),
                name=n, line=dict(color=_MAT[n]["color"], width=2),
            ))
        fig_kt.add_vline(x=theta_deg, line_dash="dash", line_color="gray",
                          annotation_text=f"θ={theta_deg}°",
                          annotation_position="top right")
        fig_kt.update_layout(
            title="Kerf cutting effort (Gc_eff/Gc)<br><sup>lower = easier split</sup>",
            xaxis_title="Cut angle θ_cut [deg]",
            yaxis_title="Gc_eff / Gc",
            height=300, margin=dict(l=50, r=20, t=60, b=40),
            legend=dict(orientation="h", y=-0.25),
        )
        c1.plotly_chart(fig_kt, use_container_width=True)

        # (b) deflection/chipping tendency vs cut orientation
        fig_df = go.Figure()
        for n in sel_mats:
            fig_df.add_trace(go.Scatter(
                x=np.rad2deg(th),
                y=_deflection_force(th, _MAT[n]["beta"]),
                name=n, line=dict(color=_MAT[n]["color"], width=2),
            ))
        fig_df.add_vline(x=theta_deg, line_dash="dash", line_color="gray",
                          annotation_text=f"θ={theta_deg}°",
                          annotation_position="top right")
        fig_df.update_layout(
            title="Kerf-deviation / chipping tendency<br><sup>ZERO at θ=0° or 90° → dice along cleavage</sup>",
            xaxis_title="Cut angle θ_cut [deg]",
            yaxis_title="|dGc_eff/dφ|",
            height=300, margin=dict(l=50, r=20, t=60, b=40),
            legend=dict(orientation="h", y=-0.25),
        )
        c2.plotly_chart(fig_df, use_container_width=True)

    # polar plot — full toughness landscape at selected θ_cut
    st.divider()
    st.subheader(f"Toughness anisotropy polar (θ_cut = {theta_deg}°)")
    fig_polar = go.Figure()
    for n in sel_mats:
        r = _gc_eff(phi, theta_rad, _MAT[n]["beta"])
        fig_polar.add_trace(go.Scatterpolar(
            r=r, theta=np.rad2deg(phi),
            mode="lines", name=f"{n} (β={_MAT[n]['beta']:+.2f})",
            line=dict(color=_MAT[n]["color"], width=2),
        ))
    fig_polar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1.1])),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", y=-0.1),
    )
    st.plotly_chart(fig_polar, use_container_width=True)
    st.caption(
        "β (SiC = −0.51) from wafer-proc-sim MAP fit to AT2 cleavage model. "
        "Si/GaN values are illustrative — calibrate to fab cleavage data."
    )
