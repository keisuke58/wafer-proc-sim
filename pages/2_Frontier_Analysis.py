"""
Frontier Analysis — Terafab / Quantum Computing / Hyperscaler
タブで3つのモデルを一覧表示。
"""
import os, sys, math
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

st.set_page_config(page_title="Frontier Analysis", layout="wide")
st.title("Frontier Analysis")
st.caption(
    "Terafab ($119B 垂直統合Fab) · 量子コンピュータ (Si Qubit / Surface Code) · "
    "Hyperscaler + Tesla カスタムシリコン · AI DC × SiC · EV × SiC"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏭 Terafab", "⚛️ Quantum Computing", "🖥️ Hyperscaler + Tesla",
    "🤖 AI DC × SiC", "🚗 EV × SiC",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TERAFAB
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    from fem.terafab_model import (
        production_target, yield_chain, capex_analysis,
        japan_impact_analysis, rad_hard_tid_model, TERAFAB,
    )

    prod  = production_target()
    capex = capex_analysis()
    imp   = japan_impact_analysis()

    k = st.columns(5)
    k[0].metric("目標", "1 TW compute/年")
    k[1].metric("総投資", f"${TERAFAB['total_capex_B']:.0f}B")
    k[2].metric("ウェーハ/日", f"{prod['wafers_per_day']:,}")
    k[3].metric("日本装置需要", f"${imp['total_demand_B']:.0f}B")
    k[4].metric("HVM 開始", TERAFAB["hvm_start"])

    c1, c2 = st.columns(2)

    with c1:
        # 宇宙 vs 地上 チップ数
        fig = go.Figure(go.Bar(
            x=["宇宙向け RAD-hard", "地上向け AI Accel"],
            y=[prod["space"]["n_chips"] / 1e6, prod["ground"]["n_chips"] / 1e6],
            marker_color=["#2166ac", "#d73027"],
            text=[f"{prod['space']['n_chips']/1e6:.0f}M", f"{prod['ground']['n_chips']/1e6:.1f}M"],
            textposition="outside",
        ))
        fig.update_layout(title="年間チップ生産数 [M枚] — CapEx $119B 換算",
                          yaxis_title="[M枚]",
                          plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=290)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
| | 仕様 | 枚数/年 |
|---|---|---|
| 宇宙向け | {prod["space"]["desc"]} | **{prod["space"]["n_chips"]/1e6:.0f}M** |
| 地上向け | {prod["ground"]["desc"]} | **{prod["ground"]["n_chips"]/1e6:.1f}M** |
""")

    with c2:
        # 歩留まり連鎖
        chain = yield_chain()
        steps = [s.split("(")[0].strip() for s in chain.keys()]
        cum_Y = [v["cumulative"] * 100 for v in chain.values()]
        step_Y= [v["step_yield"]  * 100 for v in chain.values()]
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=steps, y=step_Y, name="ステップ歩留まり",
                               marker_color="#4393c3", opacity=0.8), secondary_y=False)
        fig2.add_trace(go.Scatter(x=steps, y=cum_Y, name="累積歩留まり",
                                   mode="lines+markers",
                                   line=dict(color="#d73027", width=2.5)), secondary_y=True)
        fig2.update_layout(title=f"垂直統合 歩留まり連鎖 (総計 {cum_Y[-1]:.1f}%)",
                            plot_bgcolor="#111", paper_bgcolor="#111", font_color="white",
                            height=290, legend=dict(orientation="h", y=1.1),
                            xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # CapEx ドーナツ
        labels = list(capex["breakdown"].keys())
        values = [v[0] for v in capex["breakdown"].values()]
        colors = [v[1] for v in capex["breakdown"].values()]
        fig3 = go.Figure(go.Pie(labels=labels, values=values,
                                 marker=dict(colors=colors, line=dict(color="#111", width=1.5)),
                                 hole=0.4, textinfo="percent",
                                 hovertemplate="%{label}<br>$%{value}B<extra></extra>"))
        fig3.update_layout(
            annotations=[dict(text=f"${capex['total_B']:.0f}B", x=0.5, y=0.5,
                               font_size=16, showarrow=False, font_color="white")],
            title=f"CapEx 内訳 (日本装置 {capex['japan_share_pct']:.0f}%)",
            plot_bgcolor="#111", paper_bgcolor="#111", font_color="white",
            height=310, legend=dict(font=dict(size=8)))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # 日本装置波及
        companies = [k for k, _ in imp["items"]]
        demands   = [v["demand_B"] for _, v in imp["items"]]
        fig4 = go.Figure(go.Bar(
            y=companies, x=demands, orientation="h",
            marker_color=px.colors.sequential.Blues[2:],
            text=[f"${d:.1f}B" for d in demands], textposition="outside",
        ))
        fig4.update_layout(title=f"装置需要 合計 ${imp['total_demand_B']:.0f}B",
                            xaxis_title="[$B]", plot_bgcolor="#111", paper_bgcolor="#111",
                            font_color="white", height=310,
                            xaxis=dict(range=[0, max(demands)*1.3]))
        st.plotly_chart(fig4, use_container_width=True)

    # RAD-hard
    tid_range = np.linspace(0, 300, 300)
    rad = rad_hard_tid_model(tid_range)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=tid_range, y=rad["dVth_std_V"]*1e3, name="標準 CMOS",
                               line=dict(color="#d73027", width=2)))
    fig5.add_trace(go.Scatter(x=tid_range, y=rad["dVth_rhc_V"]*1e3, name="Intel 18A RAD-hard",
                               line=dict(color="#4393c3", width=2)))
    for env, val, col in [("LEO", rad["leo_tid_krad"], "green"),
                           ("GEO", rad["geo_tid_krad"], "orange"),
                           ("MEO", rad["meo_tid_krad"], "red")]:
        fig5.add_vline(x=val, line_dash="dot", line_color=col,
                       annotation_text=f"{env}", annotation_font_color=col)
    fig5.update_layout(title="宇宙向け RAD-hard TID 劣化", xaxis_title="累積 TID [krad]",
                        yaxis_title="ΔVth [mV]", plot_bgcolor="#111", paper_bgcolor="#111",
                        font_color="white", legend=dict(orientation="h"), height=260)
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QUANTUM COMPUTING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    from fem.quantum_computing_model import (
        si_spin_qubit_coherence, si_qubit_sweep_rms,
        transmon_qubit, transmon_sweep_ratio,
        surface_code_overhead, si_qubit_wafer_integration,
        QUANTUM_ROADMAP, SI_QUBIT_ROADMAP,
    )

    st.sidebar.markdown("---")
    st.sidebar.header("⚛️ Si Qubit パラメータ")
    rms_nm = st.sidebar.slider("界面粗さ RMS [nm]", 0.05, 1.0, 0.1, 0.05, key="q_rms")
    n_imp  = st.sidebar.select_slider("不純物密度 [/cm²]",
                                       options=[1e9, 5e9, 1e10, 5e10, 1e11, 1e12], value=1e10,
                                       format_func=lambda x: f"{x:.0e}", key="q_nimp")

    si_q  = si_spin_qubit_coherence(rms_nm=rms_nm, n_imp_cm2=n_imp)
    tm_q  = transmon_qubit(EJ_GHz=25.0, EC_GHz=0.25)
    qec   = surface_code_overhead(p_phys=0.001)
    integ = si_qubit_wafer_integration(qubit_pitch_um=50.0)
    mc    = qec.get("min_config", {})

    k = st.columns(5)
    k[0].metric("Si qubit T₁", f"{si_q['T1_us']:.0f} µs")
    k[1].metric("Si qubit T₂", f"{si_q['T2_us']:.0f} µs")
    k[2].metric("Transmon f₀₁", f"{tm_q['f_01_GHz']:.2f} GHz")
    k[3].metric("QEC d (p=0.1%)", f"d={mc.get('d','—')} → {mc.get('n_phys','—')} 物理/論理")
    k[4].metric("300mm Si qubit", f"{integ['n_qubits_yield']/1e6:.0f}M")

    c1, c2 = st.columns(2)
    with c1:
        rms_range = np.linspace(0.05, 1.0, 80)
        sw = si_qubit_sweep_rms(rms_range)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rms_range, y=sw["T1_us"], name="T₁",
                                  line=dict(color="#4393c3", width=2.5)))
        fig.add_trace(go.Scatter(x=rms_range, y=sw["T2_us"], name="T₂",
                                  line=dict(color="#d73027", width=2.5)))
        fig.add_hline(y=1000, line_dash="dash", line_color="orange",
                      annotation_text="QEC 目標 1ms")
        fig.add_vline(x=rms_nm, line_dash="dot", line_color="white")
        fig.update_layout(title="T₁/T₂ vs 界面粗さ RMS", xaxis_title="RMS [nm]",
                           yaxis_title="[µs]", yaxis_type="log",
                           plot_bgcolor="#111", paper_bgcolor="#111", font_color="white",
                           height=290, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        p_values = [0.001, 0.003, 0.005, 0.008]
        cols_qec = ["#4393c3", "#74add1", "#fdae61", "#d73027"]
        fig2 = go.Figure()
        for p, col in zip(p_values, cols_qec):
            q = surface_code_overhead(p)
            if "results" in q:
                ds  = [r["d"] for r in q["results"]]
                pLs = [r["p_L"] for r in q["results"]]
                fig2.add_trace(go.Scatter(x=ds, y=pLs, name=f"p={p*100:.1f}%",
                                           mode="lines+markers", line=dict(color=col)))
        fig2.add_hline(y=1e-6, line_dash="dash", line_color="green",
                       annotation_text="目標 10⁻⁶")
        fig2.update_layout(title="Surface Code QEC: d vs 論理エラー率",
                            xaxis_title="距離 d", yaxis_title="p_L", yaxis_type="log",
                            plot_bgcolor="#111", paper_bgcolor="#111", font_color="white",
                            height=290, legend=dict(orientation="h"))
        st.plotly_chart(fig2, use_container_width=True)

    # ロードマップ
    years   = list(QUANTUM_ROADMAP.keys())
    phys_q  = [QUANTUM_ROADMAP[y][0] for y in years]
    logic_q = [max(QUANTUM_ROADMAP[y][1], 0.5) for y in years]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=years, y=phys_q, name="物理 qubit",
                               mode="lines+markers", line=dict(color="#4393c3", width=2.5)))
    fig3.add_trace(go.Scatter(x=years, y=logic_q, name="論理 qubit",
                               mode="lines+markers", line=dict(color="#d73027", width=2, dash="dash")))
    fig3.add_vline(x=2027, line_dash="dot", line_color="orange",
                   annotation_text="Terafab HVM")
    fig3.add_vline(x=2026, line_dash="dot", line_color="white",
                   annotation_text="現在")
    fig3.update_layout(title="量子コンピュータ ロードマップ 2023〜2033",
                        xaxis_title="年", yaxis_type="log",
                        plot_bgcolor="#111", paper_bgcolor="#111", font_color="white",
                        height=280, legend=dict(orientation="h"))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HYPERSCALER + TESLA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    from fem.hyperscaler_model import (
        ASIC_SPECS, HYPERSCALER_CAPEX, NVIDIA_SHARE, TESLA_CHIPS,
        wafer_demand, tco_analysis, tesla_fsd_inference_demand,
    )

    total_capex = sum(v["capex_B"] for v in HYPERSCALER_CAPEX.values())
    total_wpd   = sum(wafer_demand(c, v)["wafers_per_day"] for c, v in HYPERSCALER_CAPEX.items())
    best_chip   = max(ASIC_SPECS, key=lambda n: ASIC_SPECS[n]["tflops_bf16"] / ASIC_SPECS[n]["tdp_W"])
    best_eff    = ASIC_SPECS[best_chip]["tflops_bf16"] / ASIC_SPECS[best_chip]["tdp_W"]
    td          = tesla_fsd_inference_demand()

    k = st.columns(5)
    k[0].metric("Hyperscaler CapEx", f"${total_capex:.0f}B")
    k[1].metric("TSMC 推定需要", f"{total_wpd:,} W/日")
    k[2].metric("最高効率", f"{best_chip.split(chr(10))[0]}", delta=f"{best_eff:.1f} TFlops/W")
    k[3].metric("NVIDIA AIシェア 2026", f"{NVIDIA_SHARE[2026][0]}%",
                delta=f"→ {NVIDIA_SHARE[2030][0]}% (2030)", delta_color="inverse")
    k[4].metric("Tesla FSD 年間需要", f"{td['n_chips_vehicle']/1e6:.0f}M 枚")

    c1, c2 = st.columns(2)
    chip_names  = list(ASIC_SPECS.keys())
    short_names = [c.split("\n")[0] for c in chip_names]
    chip_colors = [ASIC_SPECS[c]["color"] for c in chip_names]

    with c1:
        effs = [ASIC_SPECS[c]["tflops_bf16"] / ASIC_SPECS[c]["tdp_W"] for c in chip_names]
        fig = go.Figure(go.Bar(y=short_names, x=effs, orientation="h",
                                marker_color=chip_colors,
                                text=[f"{e:.1f}" for e in effs], textposition="outside"))
        fig.update_layout(title="演算効率 (TFlops/W)", xaxis_title="TFlops/W",
                           plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tco_data = []
        for name in chip_names:
            if "H100" in name:
                continue
            t = tco_analysis(name, {})
            if t:
                tco_data.append((name.split("\n")[0], t["tco_vs_h100_pct"],
                                 t["tco_total_k"]*1000, ASIC_SPECS[name]["color"]))
        names_t, advs, totals, cols_t = zip(*tco_data)
        fig2 = go.Figure(go.Bar(
            y=list(names_t), x=list(advs), orientation="h",
            marker_color=["#d73027" if a < 0 else c for a, c in zip(advs, cols_t)],
            text=[f"{a:+.0f}%" for a in advs], textposition="outside",
            hovertemplate="%{y}<br>%{x:.0f}%<br>$%{customdata:.0f}/chip 3yr<extra></extra>",
            customdata=list(totals),
        ))
        fig2.add_vline(x=0, line_color="white")
        fig2.update_layout(title="TCO 優位性 vs H100 [%]",
                            plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        yrs    = list(NVIDIA_SHARE.keys())
        gpu_sh = [NVIDIA_SHARE[y][0] for y in yrs]
        asc_sh = [NVIDIA_SHARE[y][1] for y in yrs]
        mkt_B  = [NVIDIA_SHARE[y][2] for y in yrs]
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(x=yrs, y=gpu_sh, name="NVIDIA GPU",
                                   fill="tozeroy", fillcolor="rgba(118,185,0,0.3)",
                                   line=dict(color="#76b900", width=2)), secondary_y=False)
        fig3.add_trace(go.Scatter(x=yrs, y=asc_sh, name="Custom ASIC",
                                   fill="tonexty", fillcolor="rgba(33,102,172,0.3)",
                                   line=dict(color="#2166ac", width=2)), secondary_y=False)
        fig3.add_trace(go.Scatter(x=yrs, y=mkt_B, name="市場[$B]",
                                   line=dict(color="#fdae61", width=1.5, dash="dot"),
                                   mode="lines+markers"), secondary_y=True)
        fig3.add_vline(x=2026, line_dash="dot", line_color="white")
        fig3.update_layout(title="NVIDIA vs ASIC シェア推移",
                            plot_bgcolor="#111", paper_bgcolor="#111", font_color="white",
                            height=300, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        gen_names = list(TESLA_CHIPS.keys())
        gen_tfl   = [TESLA_CHIPS[g]["tflops"] for g in gen_names]
        gen_cols  = [TESLA_CHIPS[g]["color"]   for g in gen_names]
        fig4 = go.Figure(go.Bar(x=gen_names, y=gen_tfl, marker_color=gen_cols,
                                 text=[f"{v:,}" for v in gen_tfl], textposition="outside"))
        fig4.update_layout(title="Tesla カスタムシリコン ロードマップ",
                            yaxis_title="Peak TFLOPS", yaxis_type="log",
                            plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=300)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown(f"""
#### Tesla AI5 年間需要試算
| 用途 | チップ数 | 備考 |
|---|---|---|
| FSD 車両向け | **{td['n_chips_vehicle']/1e6:.0f}M 枚** | 600万台/年 × 2チップ |
| Optimus ロボット | **{td['n_chips_optimus']/1e3:.0f}K 枚** | 50万台/年 |
| Dojo 3 学習 | **{td['n_dojo_tiles']} タイル** | 1 ExaFLOPS クラスタ |
| **2nm ウェーハ需要** | **{td['wafers_needed']:,} 枚** | **${td['wafer_cost_B']:.1f}B** @ $30k/wafer |
""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI DC × SiC
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    from fem.ai_datacenter_model import (
        server_power, cluster_power, sic_wafer_demand_from_ai,
        sic_vs_si_tco, ai_chip_yield_and_wafer_demand,
        ai_power_and_sic_market_forecast,
        AI_CHIP_POWER_W, AI_CLUSTER_GPUS,
    )

    st.subheader("AI データセンター × SiC パワーエレクトロニクス")

    chip_sel = st.selectbox("AI チップ", list(AI_CHIP_POWER_W.keys()), index=0, key="ai_chip")
    sv = server_power(chip_sel)
    wd = sic_wafer_demand_from_ai()

    k = st.columns(5)
    k[0].metric("サーバー IT 電力", f"{sv['p_server_it_W']/1000:.1f} kW")
    k[1].metric("PSU 損失", f"{sv['p_loss_W']:.0f} W")
    k[2].metric("全 AI DC 電力 (2026)", "~1,700 MW")
    k[3].metric("AI SiC ウェーハ/日", f"{wd['wafers_per_day']:.0f} 枚")
    k[4].metric("EV SiC ウェーハ/日", f"{wd['ev_wafers_per_day']:,} 枚")

    c1, c2 = st.columns(2)

    with c1:
        # 電力予測
        fc = ai_power_and_sic_market_forecast()
        years  = [r["year"] for r in fc]
        ai_gw  = [r["ai_avg_power_GW"] for r in fc]
        sic_b  = [r["sic_market_B$"] for r in fc]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=years, y=ai_gw, name="AI DC 平均電力 [GW]",
                             marker_color="#e45756"), secondary_y=False)
        fig.add_trace(go.Scatter(x=years, y=sic_b, name="SiC 市場規模 [$B]",
                                 mode="lines+markers", line=dict(color="#f5a623", width=3)),
                      secondary_y=True)
        fig.update_layout(title="AI データセンター電力 × SiC 市場予測",
                          plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=340)
        fig.update_yaxes(title_text="平均電力 [GW]", secondary_y=False)
        fig.update_yaxes(title_text="SiC 市場規模 [$B]", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # TCO 比較
        mw_sel = st.slider("クラスター規模 [MW]", 10, 500, 100, 10, key="dc_mw")
        tco = sic_vs_si_tco(mw_sel, years=5)

        fig2 = go.Figure(go.Bar(
            x=["電気代節約", "冷却費節約", "初期コスト増", "純節約"],
            y=[tco["elec_saving_M$"], tco["cooling_saving_M$"],
               -tco["capex_premium_M$"], tco["net_saving_M$"]],
            marker_color=["#4daf4a", "#4daf4a", "#e41a1c", "#2166ac"],
            text=[f"${v:+.1f}M" for v in [tco["elec_saving_M$"], tco["cooling_saving_M$"],
                                            -tco["capex_premium_M$"], tco["net_saving_M$"]]],
            textposition="outside",
        ))
        fig2.update_layout(title=f"SiC vs Si IGBT 5年 TCO ({mw_sel}MW クラスター)",
                           yaxis_title="コスト [M$]",
                           plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=340)
        st.plotly_chart(fig2, use_container_width=True)

    st.info(f"💡 回収期間 **{tco['payback_years']:.1f}年** / CO₂削減 **{tco['co2_reduction_tpa']:,.0f} t/年** "
            f"(PSU 効率: Si 94% → SiC 98.5%)")

    # AI チップ収率
    st.markdown("#### AI チップ収率 & ウェーハ需要 (Murphy モデル)")
    cols = st.columns(4)
    for i, chip in enumerate(["H100_SXM5", "B200_SXM", "TPU_v6e", "Tesla_AI5"]):
        yld = ai_chip_yield_and_wafer_demand(chip, 50_000)
        cols[i % 4].metric(
            chip,
            f"収率 {yld['murphy_yield_pct']}%",
            f"{yld['wafers_needed']:,} wafers / 50k chips",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EV × SiC
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    from fem.ev_sic_model import (
        switching_loss, thermal_model, ev_sic_demand,
        EV_SYSTEMS, SWITCH_PARAMS, EV_MAKER_VOLUME,
    )

    st.subheader("EV × SiC パワーエレクトロニクス")

    sys_sel = st.selectbox("EV システム", list(EV_SYSTEMS.keys()), index=1, key="ev_sys")
    r_sic = switching_loss("SiC_MOSFET_1200V", sys_sel)
    r_si  = switching_loss("Si_IGBT_1200V",    sys_sel)
    th    = thermal_model(r_sic["P_total_W"])
    dem   = ev_sic_demand()

    k = st.columns(5)
    k[0].metric("SiC 損失",    f"{r_sic['P_total_W']:.0f} W")
    k[1].metric("Si 損失",     f"{r_si['P_total_W']:.0f} W",
                f"-{r_si['P_total_W']-r_sic['P_total_W']:.0f}W SiC優位")
    k[2].metric("Tj (SiC)",   f"{th['Tj_C']:.0f} °C",
                f"余裕 {th['margin_K']:.0f}K {'✓' if th['safe'] else '✗'}")
    k[3].metric("EV SiC チップ/年", f"{dem['total_chips_M']:.0f}M 枚")
    k[4].metric("必要ウェーハ/日",  f"{dem['wafers_per_day']:.0f} 枚")

    c1, c2 = st.columns(2)

    with c1:
        # スイッチング損失比較
        devices = list(SWITCH_PARAMS.keys())
        p_sw   = [switching_loss(d, sys_sel)["P_sw_W"]   for d in devices]
        p_cond = [switching_loss(d, sys_sel)["P_cond_W"] for d in devices]

        fig = go.Figure()
        fig.add_bar(name="スイッチング損失", x=devices, y=p_sw,   marker_color="#e45756")
        fig.add_bar(name="導通損失",         x=devices, y=p_cond, marker_color="#4c78a8")
        fig.update_layout(barmode="stack", title="SiC vs Si IGBT 損失内訳",
                          yaxis_title="損失 [W]",
                          plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # EV メーカー別 SiC 需要
        makers = list(EV_MAKER_VOLUME.keys())
        vols   = [EV_MAKER_VOLUME[m] / 1e6 for m in makers]
        chips  = [dem["breakdown"].get(m, 0) / 1e6 for m in makers]

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=makers, y=vols, name="EV 台数 [M台/年]",
                              marker_color="#4c78a8"), secondary_y=False)
        fig2.add_trace(go.Scatter(x=makers, y=chips, name="SiC チップ [M個/年]",
                                  mode="lines+markers", line=dict(color="#f5a623", width=3)),
                       secondary_y=True)
        fig2.update_layout(title="EV メーカー別 SiC 需要",
                           plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=340)
        fig2.update_yaxes(title_text="EV 台数 [M台/年]", secondary_y=False)
        fig2.update_yaxes(title_text="SiC チップ [M個/年]", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)

    eff_gain = r_si["efficiency_%"] - r_sic["efficiency_%"]
    st.info(f"💡 SiC 採用で航続距離 **+3〜5%** 相当 (インバータ効率差 {r_sic['efficiency_%']:.2f}% vs {r_si['efficiency_%']:.2f}%)")
