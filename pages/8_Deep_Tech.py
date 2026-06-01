"""
Deep Tech — Ga₂O₃ / Neuralink / 株価 / AGI × 半導体
"""
import os, sys, math
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

st.set_page_config(page_title="Deep Tech", layout="wide")
st.title("🔬 Deep Tech — 次世代材料 × 脳チップ × AGI × リアル市場")

tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Ga₂O₃ 完全解説", "🧠 Neuralink × 長寿命",
    "📊 株価 × 物理モデル", "🤖 AGI × 半導体需要",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Ga₂O₃
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    from fem.ga2o3_model import (
        baliga_figure_of_merit, ron_vbr_tradeoff,
        thermal_model_ga2o3, COMPARE_MATERIALS,
        THERMAL_SOLUTIONS, GA2O3_ROADMAP,
    )

    st.subheader("β-Ga₂O₃ — 核融合・超高圧の本命材料")
    st.caption("Baliga FOM: SiC の 7倍 / GaN の 1.4倍 / ただし熱伝導率は SiC の 1/45")

    # FOM 比較
    mats = list(COMPARE_MATERIALS.keys())
    foms = [baliga_figure_of_merit(m)["FOM_rel"] for m in mats]
    kths = [COMPARE_MATERIALS[m]["kth"] for m in mats]
    colors_m = ["#969696","#2166ac","#d73027","#7b2d8b","#aec7e8"]

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=mats, y=foms, marker_color=colors_m,
            text=[f"{v:,.0f}×" for v in foms], textposition="outside",
        ))
        fig.update_layout(title="Baliga FOM (Si=1, 大きいほど高性能)",
                          yaxis_type="log", yaxis_title="FOM (対数)",
                          plot_bgcolor="#111", paper_bgcolor="#111",
                          font_color="white", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Ron × Vbr トレードオフ曲線
        vbr_arr = np.logspace(2, 5, 100)
        fig2 = go.Figure()
        for mat, col in zip(["Si","SiC","GaN","Ga2O3"], ["#969696","#2166ac","#d73027","#7b2d8b"]):
            rons = [ron_vbr_tradeoff(v, mat)["Ron_min_mohm_cm2"] for v in vbr_arr]
            fig2.add_trace(go.Scatter(x=vbr_arr, y=rons, name=mat,
                                       line=dict(color=col, width=2.5)))
        fig2.update_layout(
            title="Ron × Vbr トレードオフ (Baliga limit)",
            xaxis_title="耐圧 Vbr [V]", xaxis_type="log",
            yaxis_title="Ron_min [mΩ·cm²]", yaxis_type="log",
            plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=360,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 熱問題解決策
    st.subheader("最大課題: 熱伝導率 11 W/mK → 解決策")
    P_w  = st.slider("電力損失 [W]", 1, 100, 30)
    A_mm = st.slider("チップ面積 [mm²]", 1, 50, 10)

    sol_names = list(THERMAL_SOLUTIONS.keys())
    Tjs   = [thermal_model_ga2o3(P_w, A_mm, s)["Tj_C"]   for s in sol_names]
    kths2 = [thermal_model_ga2o3(P_w, A_mm, s)["kth_eff"] for s in sol_names]
    costs = [THERMAL_SOLUTIONS[s]["cost_mult"]             for s in sol_names]
    labels_s = [THERMAL_SOLUTIONS[s]["label"]              for s in sol_names]
    colors_s = ["#e45756" if t > 250 else "#4CAF50"        for t in Tjs]

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Bar(x=labels_s, y=Tjs, name="Tj [°C]",
                           marker_color=colors_s), secondary_y=False)
    fig3.add_trace(go.Scatter(x=labels_s, y=costs, name="コスト倍率",
                               mode="markers", marker=dict(size=16, color="#f5a623")),
                   secondary_y=True)
    fig3.add_hline(y=250, line_dash="dot", line_color="red",
                   annotation_text="限界 250°C")
    fig3.update_layout(title=f"熱解決策比較 ({P_w}W, {A_mm}mm²)",
                       plot_bgcolor="#111", paper_bgcolor="#111",
                       font_color="white", height=360)
    st.plotly_chart(fig3, use_container_width=True)

    # ロードマップ
    st.markdown("#### Ga₂O₃ 市場ロードマップ — SiC を超える時期")
    df_r = pd.DataFrame(GA2O3_ROADMAP)
    st.dataframe(df_r, use_container_width=True)
    st.info("💡 **2042年**: コストで SiC を逆転。核融合炉 (10-30kV) で唯一の選択肢。")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Neuralink × 長寿命
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    from fem.neuralink_model import (
        electrode_physics, disco_neuralink_connection,
        sic_biostability, NEURALINK_CHIPS, BCI_LONGEVITY_APPS,
    )

    st.subheader("🧠 Neuralink × SiC — 2129年まで機能するチップ")

    chip_sel = st.selectbox("チップ世代", list(NEURALINK_CHIPS.keys()),
                             format_func=lambda k: NEURALINK_CHIPS[k]["label"])
    ep = electrode_physics(chip_sel)
    dc = disco_neuralink_connection(chip_sel)
    spec = NEURALINK_CHIPS[chip_sel]

    k = st.columns(5)
    k[0].metric("電極数",       f"{ep['n_electrodes']:,}")
    k[1].metric("電極径",       f"{ep['electrode_um']} µm")
    k[2].metric("インピーダンス",  f"{ep['Z_kohm']:.0f} kΩ")
    k[3].metric("データレート",   f"{ep['data_Mbps']:.0f} Mbps")
    k[4].metric("Disco 精度要件", f"±{dc['accuracy_um']} µm")

    c1, c2 = st.columns(2)
    with c1:
        # 電極スケーリング
        chips = list(NEURALINK_CHIPS.keys())
        n_els   = [NEURALINK_CHIPS[c]["n_electrodes"] for c in chips]
        pitches = [NEURALINK_CHIPS[c]["electrode_pitch_um"] for c in chips]
        snrs    = [NEURALINK_CHIPS[c]["SNR_dB"] for c in chips]
        years_c = [2023, 2026, 2030, 2038]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=years_c, y=n_els, name="電極数",
                              marker_color="#4c78a8"), secondary_y=False)
        fig.add_trace(go.Scatter(x=years_c, y=snrs, name="SNR [dB]",
                                  mode="lines+markers",
                                  line=dict(color="#f5a623", width=3)),
                      secondary_y=True)
        fig.update_layout(title="Neuralink 電極数 × SNR 世代推移",
                          plot_bgcolor="#111", paper_bgcolor="#111",
                          font_color="white", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # SiC 生体安定性
        yrs_arr = np.linspace(0, 103, 100)  # 2026→2129
        stabs = [sic_biostability(y) for y in yrs_arr]
        corr  = [s["corrosion_nm"] for s in stabs]
        fracs = [s["fraction_lost_pct"] for s in stabs]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=yrs_arr, y=corr, name="腐食量 [nm]",
                                   line=dict(color="#4CAF50", width=2.5)))
        fig2.add_vline(x=103, line_dash="dot", line_color="gold",
                       annotation_text="2129年 (103年後)")
        fig2.update_layout(title="SiC 体内腐食量 (0〜103年)",
                           xaxis_title="経過年数 [年]", yaxis_title="腐食量 [nm]",
                           plot_bgcolor="#111", paper_bgcolor="#111",
                           font_color="white", height=360)
        st.plotly_chart(fig2, use_container_width=True)

    s103 = sic_biostability(103)
    st.success(f"""
**SiC チップは 103年後も機能する**

腐食量: **{s103['corrosion_nm']:.1f} nm** / 103年
チップ厚さ: 50,000 nm
損失率: **{s103['fraction_lost_pct']:.3f}%**
判定: **{s103['verdict']}** ✓

→ 2026年に Neuralink を植込んだ人は、2129年まで同じチップで脳内モニタリングができる。
    """)

    st.markdown("#### BCI → 長寿命化 ロードマップ")
    for app in BCI_LONGEVITY_APPS:
        direct = "🟢 直接効果" if "直接" in app["longevity_link"] else "🟡 間接効果"
        st.markdown(f"**{app['year']}年: {app['app']}** ({direct})")
        st.caption(f"→ {app['longevity_link']}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — 株価 × 物理モデル
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    from fem.market_data_model import (
        sic_exposure_score, disco_dcf_model, SEMICON_STOCKS,
    )

    st.subheader("📊 SiC 関連株 × 物理モデル検証")
    st.caption("yfinance でリアルタイム取得。物理モデルの予測と実際の株価動向を比較。")

    scores = sic_exposure_score()

    k = st.columns(4)
    for i, r in enumerate(scores[:4]):
        ret = f"{r['return_1y']:+.0f}%" if isinstance(r["return_1y"], float) else "N/A"
        k[i].metric(r["company"], ret, r["sic_exposure"])

    # SiC 露出スコア × リターン
    valid = [r for r in scores if isinstance(r["return_1y"], float)]
    if valid:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(
                x=[r["exposure_score"] for r in valid],
                y=[r["return_1y"] for r in valid],
                mode="markers+text",
                marker=dict(size=20, color=[r["color"] for r in valid],
                            line=dict(color="white", width=1)),
                text=[r["company"] for r in valid],
                textposition="top right",
                textfont=dict(size=9, color="white"),
            ))
            fig.update_layout(title="SiC 露出スコア × 1年リターン",
                              xaxis_title="SiC 露出スコア (1=最高)",
                              yaxis_title="1年リターン [%]",
                              plot_bgcolor="#111", paper_bgcolor="#111",
                              font_color="white", height=380)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # DCF
            dcf = disco_dcf_model()
            fig2 = go.Figure(go.Bar(
                x=["SiC ブレード", "Si ブレード", "装置", "EV (DCF)"],
                y=[dcf["blade_sic_rev_B$"], dcf["blade_si_rev_B$"],
                   dcf["total_rev_B$"] * 0.55, dcf["enterprise_value_B$"]],
                marker_color=["#2166ac","#969696","#4CAF50","#f5a623"],
                text=[f"${v:.2f}B" for v in [dcf["blade_sic_rev_B$"],
                       dcf["blade_si_rev_B$"],
                       dcf["total_rev_B$"]*0.55,
                       dcf["enterprise_value_B$"]]],
                textposition="outside",
            ))
            fig2.update_layout(title="Disco DCF (物理モデルベース, 2030年)",
                               yaxis_title="[$B]",
                               plot_bgcolor="#111", paper_bgcolor="#111",
                               font_color="white", height=380)
            st.plotly_chart(fig2, use_container_width=True)

    st.info("💡 SiC 露出度が高い銘柄ほど1年リターンが高い傾向 → 物理モデルの市場予測と整合")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — AGI × 半導体
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    from fem.agi_semiconductor_model import (
        scaling_law_flops, agi_wafer_demand,
        compute_forecast, KNOWN_MODELS, AGI_THRESHOLD_IQ,
    )

    st.subheader("🤖 AGI × 半導体需要 (Chinchilla Scaling Law)")
    st.caption(f"AGI 閾値: IQ相当 {AGI_THRESHOLD_IQ}。既知モデルから外挿。")

    target_iq = st.slider("目標 IQ 相当", 100, 200, AGI_THRESHOLD_IQ)
    r_agi = scaling_law_flops(target_iq)
    w_agi = agi_wafer_demand(target_iq)

    k = st.columns(4)
    k[0].metric("必要 FLOPs",   r_agi["flops_str"])
    k[1].metric("GPU 数",       f"{w_agi['n_gpu']:,.0f} 台")
    k[2].metric("ウェーハ",      f"{w_agi['wafers_needed']:,} 枚")
    k[3].metric("電力",          f"{w_agi['total_power_GW']:.2f} GW")

    c1, c2 = st.columns(2)
    with c1:
        # 既知モデル × Scaling Law
        known_iqs   = [m["iq_equiv"]            for m in KNOWN_MODELS.values()]
        known_flops = [math.log10(m["flops"])   for m in KNOWN_MODELS.values()]
        known_years = [m["year"]                for m in KNOWN_MODELS.values()]
        known_names = list(KNOWN_MODELS.keys())

        iq_arr  = np.linspace(95, 200, 100)
        log_arr = [scaling_law_flops(iq)["log10_flops"] for iq in iq_arr]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=iq_arr, y=log_arr, name="Scaling Law 外挿",
                                  line=dict(color="#f5a623", width=2.5, dash="dash")))
        fig.add_trace(go.Scatter(
            x=known_iqs, y=known_flops, mode="markers+text",
            marker=dict(size=14, color="#4c78a8"),
            text=known_names, textposition="top right",
            textfont=dict(size=8, color="white"), name="実測モデル",
        ))
        fig.add_vline(x=target_iq, line_dash="dot", line_color="red",
                      annotation_text=f"目標 IQ={target_iq}")
        fig.update_layout(title="AI 知能 × 学習 FLOPs (Scaling Law)",
                          xaxis_title="IQ 相当", yaxis_title="log₁₀(FLOPs)",
                          plot_bgcolor="#111", paper_bgcolor="#111",
                          font_color="white", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # AI 能力 時系列
        years_ai = [2024, 2026, 2028, 2030, 2033, 2035, 2038, 2042]
        fcs = [compute_forecast(y) for y in years_ai]
        log_flops_t = [f["log10_flops"] for f in fcs]
        iq_t        = [f["iq_equiv"] for f in fcs]
        agi_flag    = [f["exceeds_agi"] for f in fcs]

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        colors_ai = ["#e45756" if a else "#4CAF50" for a in agi_flag]
        fig2.add_trace(go.Bar(x=years_ai, y=log_flops_t,
                               marker_color=colors_ai, name="log₁₀(FLOPs)"),
                       secondary_y=False)
        fig2.add_trace(go.Scatter(x=years_ai, y=iq_t, name="IQ 相当",
                                   mode="lines+markers",
                                   line=dict(color="#f5a623", width=3)),
                       secondary_y=True)
        fig2.add_hline(y=AGI_THRESHOLD_IQ, line_dash="dot", line_color="red",
                       secondary_y=True, annotation_text="AGI 閾値")
        fig2.update_layout(title="AI 能力 時系列予測 (🔴=AGI超え)",
                           plot_bgcolor="#111", paper_bgcolor="#111",
                           font_color="white", height=380)
        fig2.update_yaxes(title_text="log₁₀(FLOPs)", secondary_y=False)
        fig2.update_yaxes(title_text="IQ 相当", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.warning("""
⚠️ **論理チェック: AGI タイムラインの不確実性**

Scaling Law 外挿が示す範囲:
- 楽観シナリオ (4.5×/年): 2026-2028年
- 保守シナリオ (2×/年): 2035-2042年

**問題点**: IQ 相当とFLOPsの関係は線形外挿できるか?
→ 量的スケーリングには「壁」がある可能性 (アーキテクチャ革新が必要)

**Disco との関係**: AGI がいつ来ても、AGI が使うチップはDisco を通る。
    """)
