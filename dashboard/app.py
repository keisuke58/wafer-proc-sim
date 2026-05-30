"""
Streamlit real-time dashboard — SiC Dicing Live Monitor.

Polls the FastAPI backend at http://localhost:8000/api/history every second
using st.fragment(run_every=1.0) for live chart updates without full reruns.

Run (FastAPI must be running first):
    streamlit run dashboard/app.py
"""
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

API = os.environ.get("DASHBOARD_API", "http://localhost:8000")

SENSORS = {
    "spindle_current_A":    ("Spindle Current [A]",    2.5,  4.00),
    "vibration_rms_g":      ("Vibration RMS [g]",      0.15, 0.60),
    "acoustic_emission_dB": ("Acoustic Emission [dB]", 45.0, 55.0),
    "camera_chip_um":       ("Chipping [µm]",          4.5,  12.0),
}

CHART_H = 210


def _sensor_chart(df: pd.DataFrame, col: str, title: str,
                  nominal: float, usl: float) -> go.Figure:
    anom = df["is_anomaly"].values
    vals = df[col].astype(float)
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["wafer_id"], y=vals,
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
    fig.update_layout(
        height=CHART_H, title_text=title, title_font_size=11,
        margin=dict(l=40, r=10, t=32, b=25),
        xaxis_title="Wafer #", showlegend=False,
    )
    return fig


def _fusion_chart(df: pd.DataFrame) -> go.Figure:
    adf = df[df["is_anomaly"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["wafer_id"], y=df["fusion_z"].astype(float),
        mode="lines", fill="tozeroy",
        line=dict(color="#f97316", width=1.5),
        fillcolor="rgba(249,115,22,0.12)",
        name="fusion z",
    ))
    fig.add_hline(y=3.0, line_dash="dash", line_color="#e31a1c",
                  annotation_text="alert=3.0", annotation_position="top right")
    fig.add_trace(go.Scatter(
        x=adf["wafer_id"], y=adf["fusion_z"].astype(float),
        mode="markers",
        marker=dict(color="#e31a1c", size=10, symbol="x"),
        name="true anomaly",
    ))
    fig.update_layout(
        height=CHART_H, title_text="Weighted fusion z-score",
        title_font_size=11,
        margin=dict(l=40, r=10, t=32, b=25),
        xaxis_title="Wafer #", yaxis_title="z",
    )
    return fig


def _f1(pred, truth):
    tp  = ((pred == 1) & (truth == 1)).sum()
    fp  = ((pred == 1) & (truth == 0)).sum()
    fn  = ((pred == 0) & (truth == 1)).sum()
    pr  = tp / (tp + fp + 1e-9)
    rc  = tp / (tp + fn + 1e-9)
    fpr = fp / (len(truth) - truth.sum() + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9), float(fpr)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SiC Dicing Monitor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Simulation Config")
    anomaly_rate = st.slider("Anomaly rate",      0.00, 0.50, 0.15, 0.05)
    interval_s   = st.slider("Interval (s/wafer)", 0.2,  3.0,  1.0,  0.1)
    n_life       = st.slider("Blade life (wafers)", 50, 500,  200,  10)
    n_show       = st.slider("History window",     20,  200,   60,  10)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Apply", use_container_width=True, type="primary"):
            try:
                requests.post(f"{API}/api/config", json={
                    "anomaly_rate": anomaly_rate,
                    "interval_s":   interval_s,
                    "n_life":       n_life,
                }, timeout=2)
                st.success("Applied")
            except Exception as e:
                st.error(str(e))
    with c2:
        if st.button("Reset", use_container_width=True):
            try:
                requests.post(f"{API}/api/reset", timeout=2)
                st.info("Reset")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.caption(f"API: `{API}`\nDISCO DAD3350 simulator\nwafer-proc-sim v2")


# ── Live fragment — reruns every 1 s without reloading sidebar ────────────────
@st.fragment(run_every=1.0)
def live_panel():
    try:
        resp = requests.get(f"{API}/api/history",
                            params={"n": n_show}, timeout=2)
        rows = resp.json()
    except Exception as e:
        st.error(
            f"**Cannot reach API** at `{API}`\n\n"
            f"Start with: `uvicorn dashboard.api:app --port 8000`\n\n{e}"
        )
        return

    if not rows:
        st.info("Waiting for first wafer reading...")
        return

    df  = pd.DataFrame(rows)
    lat = df.iloc[-1]

    # ── KPI strip ─────────────────────────────────────────────────────────────
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
                "🚨 ANOMALY" if lat["is_anomaly"] else "✅ OK",
                delta_color="inverse" if lat["is_anomaly"] else "off")

    st.divider()

    # ── 2×2 sensor charts ─────────────────────────────────────────────────────
    pairs = list(SENSORS.items())
    for row_pair in [pairs[:2], pairs[2:]]:
        cols = st.columns(2)
        for col, (key, (title, nom, usl)) in zip(cols, row_pair):
            col.plotly_chart(
                _sensor_chart(df, key, title, nom, usl),
                use_container_width=True,
            )

    st.divider()

    # ── Fusion timeline + running metrics ─────────────────────────────────────
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
        st.metric("Miss rate",
                  f"{(truth.sum() - int(((pred_A==1)&(truth==1)).sum())) / max(truth.sum(),1):.0%}")


live_panel()
