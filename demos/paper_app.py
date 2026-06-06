"""
Minimal Streamlit demo — core paper results only.

Run:
    streamlit run demos/paper_app.py
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

from ml.train_from_experimental import loo_heteroscedastic, load_data

st.set_page_config(page_title="SiC Dicing GP — Paper Results", layout="wide")
st.title("4H-SiC Blade Dicing — Heteroscedastic GP Surrogate")
st.caption("Core paper spine · wafer-proc-sim")

col1, col2 = st.columns(2)

with st.sidebar:
    st.header("Reproduce")
    st.code("./scripts/reproduce_paper.sh", language="bash")
    run_loo = st.button("Run LOO metrics (slow ~60s)")

FIG_DIR = os.path.join(ROOT, "paper", "sic_gp_quality", "figures")

with col1:
    st.subheader("Figure 1 — Data quality grades")
    fig1 = os.path.join(FIG_DIR, "fig1_quality_distribution.png")
    if os.path.isfile(fig1):
        st.image(fig1, use_container_width=True)
    else:
        st.info("Run `python paper/sic_gp_quality/gen_figures.py` to generate.")

with col2:
    st.subheader("Figure 2 — LOO scatter (A/B vs all)")
    fig2 = os.path.join(FIG_DIR, "fig2_loo_scatter.png")
    if os.path.isfile(fig2):
        st.image(fig2, use_container_width=True)
    else:
        st.info("Run `python paper/sic_gp_quality/gen_figures.py` to generate.")

st.subheader("Key results (README)")

if run_loo:
    with st.spinner("Running heteroscedastic LOO-CV …"):
        all_m = loo_heteroscedastic("all")
        ab_m = loo_heteroscedastic("AB")
    st.success("LOO complete")
    st.table({
        "Filter": ["all grades", "Grade A/B"],
        "n": [all_m["n"], ab_m["n"]],
        "LOO RMSE [µm]": [f"{all_m['rmse']:.2f}", f"{ab_m['rmse']:.2f}"],
        "LOO R²": [f"{all_m['r2']:.3f}", f"{ab_m['r2']:.3f}"],
    })
else:
    st.table({
        "Filter": ["all grades", "Grade A/B"],
        "n": [25, 11],
        "LOO RMSE [µm]": ["3.12", "1.62"],
        "LOO R²": ["0.34", "0.80"],
    })
    st.caption("Click **Run LOO metrics** in the sidebar to recompute live.")

X, y, df, _ = load_data("AB")
st.subheader(f"Training data preview (Grade A/B, n={len(df)})")
st.dataframe(df[["source", "quality", "cut_depth_um", "feed_mm_s", "spindle_rpm", "chipping_um"]].head(15))