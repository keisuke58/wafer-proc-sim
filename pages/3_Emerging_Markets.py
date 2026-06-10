"""
Emerging Markets — ロボット / ドローン / eVTOL / 衛星 / BESS
"""
import os, sys
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

st.set_page_config(page_title="Emerging Markets", layout="wide")
st.title("Emerging Markets — 次世代 SiC / GaN 応用")
st.caption("ヒューマノイドロボット · 軍用ドローン · eVTOL · LEO衛星 · 系統用BESS")

tab1, tab2, tab3, tab4 = st.tabs(["🤖 ロボット & ドローン", "🛰️ 衛星 & 宇宙", "⚡ BESS & グリッド", "⚛️ 核融合"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — ロボット & ドローン
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    from market_analysis.robot_drone_model import (
        joint_power_model, humanoid_market_forecast,
        drone_power_model, evtol_sic_demand,
        HUMANOID_SPECS, DRONE_SPECS, EVTOL_SPECS,
    )

    st.subheader("ヒューマノイドロボット × GaN パワーエレクトロニクス")

    col_r, col_a = st.columns([1, 2])
    with col_r:
        robot_sel = st.selectbox("ロボット機種", list(HUMANOID_SPECS.keys()))
        act_sel   = st.selectbox("活動モード",
                                  ["walking", "running", "manipulation", "standby"])

    r = joint_power_model(robot_sel, act_sel)
    spec = HUMANOID_SPECS[robot_sel]

    k = st.columns(5)
    k[0].metric("全身電力",    f"{r['P_total_W']:.0f} W")
    k[1].metric("機械出力",    f"{r['P_mechanical_W']:.0f} W")
    k[2].metric("GaN チップ数", f"{r['gan_chips']} 個")
    k[3].metric("バッテリー稼働", f"{r['runtime_h']:.1f} h")
    k[4].metric("GaN SW損失",  f"{r['P_sw_loss_W']:.0f} W")

    c1, c2 = st.columns(2)
    with c1:
        # ヒューマノイド市場予測
        fc = humanoid_market_forecast()
        years = [r["year"] for r in fc]
        units = [r["units_k"] for r in fc]
        mkt   = [r["market_B$"] for r in fc]
        chips = [r["gan_chips_M"] for r in fc]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=years, y=units, name="出荷台数 [k台]",
                             marker_color="#72B7B2"), secondary_y=False)
        fig.add_trace(go.Scatter(x=years, y=mkt, name="市場規模 [$B]",
                                  mode="lines+markers",
                                  line=dict(color="#e45756", width=3)), secondary_y=True)
        fig.update_layout(title="ヒューマノイドロボット 市場予測 (Goldman Sachs 2024)",
                          plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=340)
        fig.update_yaxes(title_text="出荷台数 [k台]", secondary_y=False)
        fig.update_yaxes(title_text="市場規模 [$B]", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # ロボット別スペック比較
        robots = list(HUMANOID_SPECS.keys())
        powers = [joint_power_model(rb, act_sel)["P_total_W"] for rb in robots]
        times  = [joint_power_model(rb, act_sel)["runtime_h"] for rb in robots]
        prices = [HUMANOID_SPECS[rb]["price_usd"] / 1000 for rb in robots]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="全身電力 [W]", x=robots, y=powers,
                               marker_color="#4c78a8"))
        fig2.add_trace(go.Bar(name="稼働時間×100 [min]", x=robots,
                               y=[t * 60 for t in times], marker_color="#72B7B2"))
        fig2.update_layout(barmode="group", title="ロボット性能比較",
                           plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=340)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("ドローン × eVTOL パワーモデル")

    c3, c4 = st.columns(2)
    with c3:
        # ドローン電力比較
        drone_results = {d: drone_power_model(d) for d in ["FPV_military", "Commercial_delivery"]}
        labels = [r["label"] for r in drone_results.values()]
        p_hover = [r["P_hover_W"] for r in drone_results.values()]
        t_fly   = [r["flight_min"] for r in drone_results.values()]

        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Bar(x=labels, y=p_hover, name="ホバー電力 [W]",
                               marker_color="#FF4136"), secondary_y=False)
        fig3.add_trace(go.Scatter(x=labels, y=t_fly, name="飛行時間 [min]",
                                   mode="markers", marker=dict(size=20, color="#FFDC00")),
                       secondary_y=True)
        fig3.update_layout(title="ドローン 電力 × 飛行時間",
                           plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # eVTOL コスト比較
        evtol_models = list(EVTOL_SPECS.keys())
        hover_kw  = [EVTOL_SPECS[m]["hover_kW"]  for m in evtol_models]
        cruise_kw = [EVTOL_SPECS[m]["cruise_kW"] for m in evtol_models]
        ranges    = [EVTOL_SPECS[m]["range_km"]  for m in evtol_models]
        ev_labels = [EVTOL_SPECS[m]["label"].split("(")[0] for m in evtol_models]

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name="ホバー [kW]",  x=ev_labels, y=hover_kw,
                               marker_color="#FFDC00"))
        fig4.add_trace(go.Bar(name="クルーズ [kW]", x=ev_labels, y=cruise_kw,
                               marker_color="#f58518"))
        fig4.update_layout(barmode="group", title="eVTOL 電力比較",
                           plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=320)
        st.plotly_chart(fig4, use_container_width=True)

    st.info("💡 **GaN の優位性**: ロボット関節は 48V / 100kHz → GaN が支配的。"
            "eVTOL は 800V / 高信頼性 → SiC 必須。電圧で棲み分け。")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — 衛星 & 宇宙
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    from market_analysis.space_grid_model import (
        satellite_power_budget, leo_constellation_sic_demand,
        SATELLITE_SPECS, SIC_TID_MODEL,
    )
    from market_analysis.terafab_model import rad_hard_tid_model

    st.subheader("LEO / GEO 衛星 × SiC 電力系モデル")

    sat_sel = st.selectbox("衛星タイプ", list(SATELLITE_SPECS.keys()),
                            format_func=lambda k: SATELLITE_SPECS[k]["label"])
    r = satellite_power_budget(sat_sel)
    leo = leo_constellation_sic_demand()

    k = st.columns(5)
    k[0].metric("太陽電池出力", f"{r['P_solar_W']:.0f} W")
    k[1].metric("SiC PCU 損失", f"{r['P_sic_loss_W']:.1f} W")
    k[2].metric("SiC モジュール/機", str(r["sic_modules"]))
    k[3].metric("TID 5年 ΔVth", f"{r['dVth_mV']:.0f} mV")
    k[4].metric("全コンステ SiC/日", f"{leo['wafers_per_day']:.2f} wafers")

    c1, c2 = st.columns(2)
    with c1:
        # TID 劣化曲線
        tid_arr = np.linspace(0, 500, 300)
        dVth_sic = tid_arr * SIC_TID_MODEL["Vth_shift_mV_per_krad"]
        dVth_si  = tid_arr * 15.0   # Si MOSFET: 15mV/krad

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tid_arr, y=dVth_sic, name="SiC MOSFET",
                                  line=dict(color="#2166ac", width=2.5)))
        fig.add_trace(go.Scatter(x=tid_arr, y=dVth_si,  name="Si MOSFET",
                                  line=dict(color="#d73027", width=2.5)))
        for env, val, col in [("LEO 5yr", 30, "cyan"), ("GEO 15yr", 300, "orange")]:
            fig.add_vline(x=val, line_dash="dot", line_color=col,
                          annotation_text=env, annotation_font_color=col)
        fig.update_layout(title="放射線 TID 劣化: SiC vs Si",
                          xaxis_title="累積 TID [krad]", yaxis_title="ΔVth [mV]",
                          plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # 衛星別電力バジェット比較
        sats = list(SATELLITE_SPECS.keys())
        sat_labels = [SATELLITE_SPECS[s]["label"] for s in sats]
        p_solar = [satellite_power_budget(s)["P_solar_W"] for s in sats]
        p_load  = [SATELLITE_SPECS[s]["power_bol_W"] for s in sats]
        p_loss  = [satellite_power_budget(s)["P_sic_loss_W"] for s in sats]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=sat_labels, y=p_solar, name="太陽電池",
                               marker_color="#f5a623"))
        fig2.add_trace(go.Bar(x=sat_labels, y=p_load,  name="負荷電力",
                               marker_color="#4c78a8"))
        fig2.add_trace(go.Bar(x=sat_labels, y=p_loss,  name="SiC 変換損失",
                               marker_color="#e45756"))
        fig2.update_layout(barmode="group", title="衛星別 電力バジェット [W]",
                           yaxis_type="log",
                           plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=360)
        st.plotly_chart(fig2, use_container_width=True)

    st.info(f"💡 SiC の TID 耐性: {SIC_TID_MODEL['Vth_shift_mV_per_krad']} mV/krad (Si の 1/7)。"
            f"GEO 衛星 15年運用 (300 krad) でも ΔVth={300*SIC_TID_MODEL['Vth_shift_mV_per_krad']:.0f}mV のみ。")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — BESS & グリッド
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    from market_analysis.space_grid_model import (
        bess_roundtrip_efficiency, bess_inverter_efficiency,
        bess_market_forecast, BESS_CONFIGS,
    )

    st.subheader("系統用 BESS × SiC 3レベル NPC インバータ")

    config_sel = st.selectbox("BESS 用途", list(BESS_CONFIGS.keys()),
                               format_func=lambda k: BESS_CONFIGS[k]["label"])
    r = bess_roundtrip_efficiency(config_sel)
    fc = bess_market_forecast()

    k = st.columns(5)
    k[0].metric("SiC 往復効率", f"{r['eta_sic']:.2f}%")
    k[1].metric("Si IGBT 往復効率", f"{r['eta_si']:.2f}%",
                f"+{r['rte_gain_pp']:.2f}pp SiC優位")
    k[2].metric("年間エネルギー", f"{r['annual_energy_MWh']:,} MWh")
    k[3].metric("追加収益 (電力差)", f"${r['extra_revenue_M$']*1000:.0f}k/年")
    k[4].metric("SiC モジュール数", f"{r['sic_modules']}")

    c1, c2 = st.columns(2)
    with c1:
        # 部分負荷効率曲線
        p_fracs = np.linspace(0.05, 1.0, 50)
        eta_sic = [bess_inverter_efficiency(p, 1500, "3L_NPC") * 100 for p in p_fracs]
        eta_si  = [(bess_inverter_efficiency(p, 1500, "3L_NPC") - 0.015) * 100 for p in p_fracs]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_fracs * 100, y=eta_sic, name="SiC 3L-NPC",
                                  line=dict(color="#4CAF50", width=2.5)))
        fig.add_trace(go.Scatter(x=p_fracs * 100, y=eta_si,  name="Si IGBT 2L",
                                  line=dict(color="#d73027", width=2.5)))
        fig.add_vline(x=80, line_dash="dot", line_color="gray",
                      annotation_text="定格80%", annotation_font_color="gray")
        fig.update_layout(title="BESS インバータ 部分負荷効率",
                          xaxis_title="負荷率 [%]", yaxis_title="効率 [%]",
                          plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # BESS 市場予測
        years   = [r["year"] for r in fc]
        gwh     = [r["install_GWh"] for r in fc]
        sic_mkt = [r["sic_mkt_B$"] for r in fc]

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=years, y=gwh, name="BESS 導入量 [GWh]",
                               marker_color="#4CAF50", opacity=0.8), secondary_y=False)
        fig2.add_trace(go.Scatter(x=years, y=sic_mkt, name="SiC 市場 [$B]",
                                   mode="lines+markers",
                                   line=dict(color="#f5a623", width=3)), secondary_y=True)
        fig2.update_layout(title="系統用 BESS 導入量 × SiC 市場 (IRENA 2024)",
                           plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=340)
        fig2.update_yaxes(title_text="BESS 導入量 [GWh]", secondary_y=False)
        fig2.update_yaxes(title_text="SiC 市場規模 [$B]", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)

    # BESS 用途別比較
    st.markdown("#### BESS 用途別 SiC 効果比較")
    cols = st.columns(4)
    for i, (cfg, label_key) in enumerate(BESS_CONFIGS.items()):
        rv = bess_roundtrip_efficiency(cfg)
        cols[i % 4].metric(
            rv["label"],
            f"往復効率 {rv['eta_sic']:.2f}%",
            f"Si比 +{rv['rte_gain_pp']:.2f}pp",
        )

    st.info("💡 **SiC 3L-NPC インバータ** は往復効率 93% (Si 90%)。"
            "100MW BESS で年間 **数億円** 相当の追加発電収益。2030年 SiC-BESS 市場 $6.8B 予測。")

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — 核融合
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    from market_analysis.fusion_model import (
        magnet_power_supply, gyrotron_power_supply,
        radiation_hard_electronics, fusion_semiconductor_summary,
        FUSION_REACTORS, FUSION_MATERIAL_SUITABILITY, FUSION_MARKET_FORECAST,
    )
    import pandas as pd

    st.subheader("核融合炉 × SiC / Ga₂O₃ パワーエレクトロニクス")
    st.caption("トカマク (ITER/SPARC) · 慣性閉じ込め (NIF) · FRC (Helion)")

    reactor_sel = st.selectbox("核融合炉", list(FUSION_REACTORS.keys()),
                                format_func=lambda k: FUSION_REACTORS[k]["label"])
    r_spec = FUSION_REACTORS[reactor_sel]
    mag   = magnet_power_supply(reactor_sel)
    gyro  = gyrotron_power_supply(reactor_sel)
    rad   = radiation_hard_electronics(reactor_sel)

    k = st.columns(5)
    k[0].metric("核融合出力",   f"{r_spec['fusion_power_MW']} MW")
    k[1].metric("Q 値",        f"{r_spec['Q_factor']}")
    k[2].metric("マグネット電源", f"{mag['P_TF_MW']+mag['P_PF_MW']:.0f} MW")
    k[3].metric("主デバイス",   mag["primary_device"])
    k[4].metric("状況",        r_spec["status"][:12])

    c1, c2 = st.columns(2)

    with c1:
        # 放射線耐性 (設備室位置)
        mats  = list(rad["survivability"].keys())
        years = [min(rad["survivability"][m]["years_to_fail"], 100) for m in mats]
        colors= ["#4CAF50" if rad["survivability"][m]["survives_10yr"]
                 else "#e45756" for m in mats]
        fig = go.Figure(go.Bar(
            x=mats, y=years,
            marker_color=colors,
            text=[f"{'✓' if rad['survivability'][m]['survives_10yr'] else '✗'} {min(rad['survivability'][m]['years_to_fail'],100):.0f}yr"
                  for m in mats],
            textposition="outside",
        ))
        fig.add_hline(y=10, line_dash="dot", line_color="yellow",
                      annotation_text="10年 運転目標")
        fig.update_layout(title=f"設備室 放射線耐性 ({reactor_sel})",
                          yaxis_title="推定寿命 [年]", yaxis_range=[0, 110],
                          plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # 核融合 SiC/Ga₂O₃ 市場予測
        years_f = [r["year"] for r in FUSION_MARKET_FORECAST]
        sic_b   = [r["sic_M$"] / 1000 for r in FUSION_MARKET_FORECAST]
        ga2o3_b = [r["ga2o3_M$"] / 1000 for r in FUSION_MARKET_FORECAST]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=years_f, y=sic_b, name="SiC [$B]",
                                   mode="lines+markers",
                                   line=dict(color="#2166ac", width=3)))
        fig2.add_trace(go.Scatter(x=years_f, y=ga2o3_b, name="Ga₂O₃ [$B]",
                                   mode="lines+markers",
                                   line=dict(color="#7b2d8b", width=3)))
        fig2.update_layout(title="核融合 半導体市場予測 (商業炉普及シナリオ)",
                           xaxis_title="年", yaxis_title="市場規模 [$B]",
                           plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=360)
        st.plotly_chart(fig2, use_container_width=True)

    # 材料適合度テーブル
    st.markdown("#### 材料別 核融合適合度")
    rows = []
    for mat, m in sorted(FUSION_MATERIAL_SUITABILITY.items(),
                          key=lambda x: x[1]["score"], reverse=True):
        rows.append({
            "材料": mat,
            "電圧範囲": m["voltage_range"],
            "核融合用途": " / ".join(m["fusion_apps"][:2]),
            "優位点": m["advantage"][:40],
            "成熟度": m["readiness"],
            "スコア": "⭐" * m["score"],
        })
    st.dataframe(pd.DataFrame(rows).set_index("材料"), use_container_width=True)

    st.markdown(f"""
#### サブシステム別 半導体要件 ({r_spec['label']})

| サブシステム | 電力 | デバイス | 電圧 | チップ数 |
|---|---|---|---|---|
| TF コイル電源 | {mag['P_TF_MW']:.0f} MW | **{mag['primary_device']}** | {mag['device_voltage_kV']:.1f} kV | {mag['total_chips']} |
| PF コイル電源 | {mag['P_PF_MW']:.0f} MW | **{mag['primary_device']}** | {mag['device_voltage_kV']:.1f} kV | — |
| ジャイロトロン | {gyro.get('P_ECRH_MW', 0)} MW | **Ga₂O₃** | 80 kV | {gyro.get('total_chips', 0)} |
    """)

    st.warning(
        "⚛️ **核融合 × Ga₂O₃ が最重要**\n\n"
        "ジャイロトロン電源 (80kV DC) と HTS マグネット電源 (10kV) は "
        "**Ga₂O₃ の絶縁破壊電界 8 MV/cm** がなければ実現困難。\n"
        "SiC は設備室の低放射線環境 (補助電源・制御系) で活躍。\n"
        "2040年に商業炉 50基なら SiC $30B + Ga₂O₃ $15B の巨大市場。"
    )
