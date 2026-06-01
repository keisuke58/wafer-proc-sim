"""
Space Compute — Starlink チップ / 宇宙 DC / 宇宙太陽光発電
"""
import os, sys
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

st.set_page_config(page_title="Space Compute", layout="wide")
st.title("🚀 Space Compute — イーロン・マスクの宇宙チップ戦略")
st.caption("Starlink カスタム ASIC · 宇宙データセンター · 宇宙太陽光発電 (SBSP)")

tab1, tab2, tab3 = st.tabs(["📡 Starlink チップ", "🖥️ 宇宙データセンター", "☀️ 宇宙太陽光発電"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Starlink チップ
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    from fem.space_compute_model import (
        starlink_chip_demand, starshield_military_value,
        STARLINK_CHIP_SPECS, STARLINK_CONSTELLATION,
    )

    st.subheader("Starlink / SpaceX カスタムチップ全体像")

    demand = starlink_chip_demand()
    shield = starshield_military_value()

    k = st.columns(5)
    k[0].metric("コンステ規模",    f"{demand['n_satellites']:,} 機")
    k[1].metric("年間新規打上",    f"{demand['annual_new_sats']:,} 機/年")
    k[2].metric("端末 ASIC/年",   f"{demand['user_terminal_chips_yr']/1e6:.1f}M 個")
    k[3].metric("年間チップ価値",   f"${demand['total_chip_value_M$']:.0f}M")
    k[4].metric("Starshield 契約", f"${shield['contract_B$']:.1f}B")

    c1, c2 = st.columns(2)
    with c1:
        # チップ別年間需要
        chip_names  = [STARLINK_CHIP_SPECS[k]["label"] for k in demand["annual_chips"]]
        chip_counts = list(demand["annual_chips"].values())
        chip_prices = [STARLINK_CHIP_SPECS[k]["price_usd"] for k in demand["annual_chips"]]
        chip_values = [n * p / 1e6 for n, p in zip(chip_counts, chip_prices)]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=chip_names, y=chip_counts, name="チップ数/年",
                             marker_color=["#001f3f","#2166ac","#4CAF50"]),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=chip_names, y=chip_values, name="価値 [$M/年]",
                                  mode="markers", marker=dict(size=20, color="#f5a623")),
                      secondary_y=True)
        fig.update_layout(title="Starlink 衛星チップ 年間需要",
                          plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # チップスペック比較テーブル
        rows = []
        for key, spec in STARLINK_CHIP_SPECS.items():
            rows.append({
                "チップ": spec["label"],
                "プロセス": spec["process_node"],
                "ダイ [mm²]": spec["die_area_mm2"],
                "電力 [W]": spec["power_W"],
                "耐放射線": "✓" if spec["rad_hard"] else "✗",
                "TID [krad]": spec["tid_krad"],
                "単価 [$]": spec["price_usd"],
            })
        st.dataframe(pd.DataFrame(rows).set_index("チップ"), use_container_width=True, height=360)

    # Starlink システム図
    st.markdown("#### Starlink 電力 × 通信チップ チェーン")
    st.code("""
太陽電池 (1.9kW)
    ↓ [SiC 電源管理 IC × 4]  ← SiC 1200V, 放射線耐性
バッテリー + バスライン (100V DC)
    ↓ [衛星メイン SoC × 2]   ← 12nm, 軌道制御・暗号化
    ↓ [ビームフォーミング ASIC × 4]  ← 7nm TSMC, 1024素子フェーズドアレイ
    ↓ 28GHz フェーズドアレイアンテナ
地上ユーザー端末 [フェーズドアレイ ASIC × 8]  ← 平面アンテナ 1280素子
    ↓
インターネット (最大 300Mbps / ユーザー)
""", language="text")

    st.info(f"💡 **Starshield (軍事版)**: {shield['n_starshield_sats']:,}機 × TID {shield['tid_spec_krad']}krad 耐性 "
            f"→ 追加 SiC チップ {shield['sic_chips_total']:,}個。"
            f" US DoD 契約 **${shield['contract_B$']:.1f}B**。")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — 宇宙データセンター
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    from fem.space_compute_model import (
        space_dc_physics, space_dc_vs_ground,
        SPACE_DC_CONCEPTS,
    )

    st.subheader("軌道上データセンター — 物理制約 × コスト分析")

    concept_sel = st.selectbox("コンセプト", list(SPACE_DC_CONCEPTS.keys()),
                                format_func=lambda k: SPACE_DC_CONCEPTS[k]["label"])
    r = space_dc_physics(concept_sel)

    k = st.columns(5)
    k[0].metric("計算電力",    f"{r['P_compute_kW']} kW")
    k[1].metric("放熱板面積",  f"{r['A_radiator_m2']:.1f} m²")
    k[2].metric("GPU 相当数",  f"{r['n_gpu_equiv']} 台")
    k[3].metric("計算性能",    f"{r['tflops']:,} TFLOPS")
    k[4].metric("通信遅延",    f"{r['latency_ms']} ms")

    c1, c2 = st.columns(2)
    with c1:
        # 放熱板面積 vs 電力 (Stefan-Boltzmann)
        T_vals = np.linspace(300, 500, 100)
        sigma = 5.67e-8; eps = 0.9
        A_vals = [10 * 1000 / (sigma * eps * (T**4 - 4**4)) for T in T_vals]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=T_vals, y=A_vals, name="10kW 放熱板面積",
                                  line=dict(color="#f5a623", width=2.5)))
        fig.add_vline(x=320, line_dash="dot", line_color="cyan",
                      annotation_text="GPU 冷却後 ~320K", annotation_font_color="cyan")
        fig.update_layout(title="宇宙 DC 放熱板 (Stefan-Boltzmann 則)",
                          xaxis_title="放射板温度 [K]", yaxis_title="必要面積 [m²]",
                          plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # 地上 vs 宇宙 コスト比較 (打上コスト別)
        launch_costs = [100, 500, 1000, 5000, 10000, 50000]  # $/kg
        space_totals = []
        ground_total = space_dc_vs_ground(10.0)["ground_total_M$"]
        for lc in launch_costs:
            mass = 10 * 50   # 10kW × 50kg/kW
            launch_M = mass * lc / 1e6
            opex_M = 0.1 * 5
            space_totals.append(launch_M + opex_M)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=launch_costs, y=space_totals, name="宇宙 DC (5年 TCO)",
                                   mode="lines+markers",
                                   line=dict(color="#e45756", width=2.5)))
        fig2.add_hline(y=ground_total, line_dash="dash", line_color="#4CAF50",
                       annotation_text=f"地上 DC ${ground_total:.0f}M", annotation_font_color="#4CAF50")
        fig2.add_vline(x=40000, line_dash="dot", line_color="gray",
                       annotation_text="現在 $40k/kg", annotation_font_color="gray")
        fig2.add_vline(x=1000, line_dash="dot", line_color="cyan",
                       annotation_text="Starship 目標 $1k/kg", annotation_font_color="cyan")
        fig2.update_xaxis(type="log")
        fig2.update_layout(title="宇宙 DC vs 地上 DC TCO (10kW, 5年)",
                           xaxis_title="打上コスト [$/kg] (対数)", yaxis_title="5年 TCO [$M]",
                           plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=360)
        st.plotly_chart(fig2, use_container_width=True)

    vs = space_dc_vs_ground(r["P_compute_kW"])
    st.warning(
        f"⚠️ **現時点 (打上 $40k/kg)**: 宇宙 DC は地上の {vs['space_total_M$']/vs['ground_total_M$']:.0f}倍 高コスト。\n\n"
        f"✅ **Starship 実現後 ($1k/kg)**: {r['label']} の TCO は地上と**同等**になる可能性。\n\n"
        f"損益分岐点: **${vs['breakeven_$/kg']:,.0f}/kg** 以下でスペースが有利。"
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — 宇宙太陽光発電 SBSP
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    from fem.space_compute_model import (
        sbsp_power_chain, sbsp_semiconductor_role,
        sbsp_market_forecast, SBSP_PROJECTS,
    )

    st.subheader("宇宙太陽光発電 (SBSP) — 電力チェーン × 半導体")

    proj_sel = st.selectbox("プロジェクト", list(SBSP_PROJECTS.keys()),
                             format_func=lambda k: SBSP_PROJECTS[k]["label"])
    r = sbsp_power_chain(proj_sel)

    k = st.columns(5)
    k[0].metric("目標出力",    f"{r['P_target_GW']*1000:.1f} MW")
    k[1].metric("総合効率",    f"{r['eta_total_pct']:.1f}%")
    k[2].metric("太陽電池面積", f"{r['A_solar_km2']:.4f} km²")
    k[3].metric("SiC ユニット数", f"{r['n_sic_units']:,}")
    k[4].metric("LCOE 現状",   f"${r['lcoe_sbsp_$/kWh']:.1f}/kWh")

    c1, c2 = st.columns(2)

    with c1:
        # 電力チェーン効率ウォーターフォール
        spec = SBSP_PROJECTS[proj_sel]
        stages = ["太陽電池\n(GaAs)", "DC/DC変換\n(SiC)", "RF送信\n(GaN-on-SiC)",
                  "大気透過", "整流アンテナ\n(GaAs)"]
        eff_each = [spec["solar_eff"], spec["dc_rf_eff"],
                    spec["transmission_eff"], spec["rectenna_eff"], 1.0]
        cumulative = [1.0]
        for e in eff_each[:-1]:
            cumulative.append(cumulative[-1] * e)
        cumulative_pct = [c * 100 for c in cumulative]
        loss_pct = [100 - c for c in cumulative_pct]

        fig = go.Figure(go.Bar(
            x=stages, y=[e * 100 for e in eff_each[:-1]] + [r["eta_total_pct"]],
            marker_color=["#f5a623","#4CAF50","#2166ac","#72B7B2","#e45756"],
            text=[f"{e*100:.0f}%" for e in eff_each[:-1]] + [f"総合\n{r['eta_total_pct']:.1f}%"],
            textposition="inside",
        ))
        fig.update_layout(title="SBSP 電力チェーン 各ステップ効率",
                          yaxis_title="効率 [%]", yaxis_range=[0, 110],
                          plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # SBSP 市場予測
        fc = sbsp_market_forecast()
        years_f  = [r["year"] for r in fc]
        sic_vals = [r["sic_M$"] / 1000 for r in fc]
        gan_vals = [r["gan_M$"] / 1000 for r in fc]
        sbsp_gw  = [r["sbsp_GW"] for r in fc]

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=years_f, y=sic_vals, name="SiC 市場 [$B]",
                               marker_color="#2166ac"), secondary_y=False)
        fig2.add_trace(go.Bar(x=years_f, y=gan_vals, name="GaN 市場 [$B]",
                               marker_color="#d73027"), secondary_y=False)
        fig2.add_trace(go.Scatter(x=years_f, y=sbsp_gw, name="SBSP 導入 [GW]",
                                   mode="lines+markers",
                                   line=dict(color="#f5a623", width=3)), secondary_y=True)
        fig2.update_layout(title="SBSP 関連半導体市場予測",
                           barmode="stack",
                           plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=360)
        fig2.update_yaxes(title_text="半導体市場 [$B]", secondary_y=False)
        fig2.update_yaxes(title_text="SBSP 導入量 [GW]", secondary_y=True, type="log")
        st.plotly_chart(fig2, use_container_width=True)

    # 半導体役割テーブル
    st.markdown("#### SBSP 各コンポーネント × 半導体材料")
    roles = sbsp_semiconductor_role()
    st.dataframe(pd.DataFrame(roles), use_container_width=True)

    # LCOE 比較
    lcoe_data = {
        "電源": ["SBSP (現在)", "SBSP (Starship後)", "地上太陽光", "洋上風力", "原子力"],
        "LCOE [$/kWh]": [r["lcoe_sbsp_$/kWh"], r["lcoe_sbsp_$/kWh"]/40,
                          0.04, 0.09, 0.10],
    }
    fig3 = go.Figure(go.Bar(
        x=lcoe_data["電源"], y=lcoe_data["LCOE [$/kWh]"],
        marker_color=["#e45756","#4CAF50","#f5a623","#72B7B2","#9C27B0"],
        text=[f"${v:.2f}" for v in lcoe_data["LCOE [$/kWh]"]],
        textposition="outside",
    ))
    fig3.update_layout(title="発電コスト (LCOE) 比較",
                       yaxis_title="$/kWh", yaxis_type="log",
                       plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=320)
    st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "💡 **SBSP のゲームチェンジャー条件**:\n\n"
        f"現在の LCOE: **${r['lcoe_sbsp_$/kWh']:.0f}/kWh** (地上太陽光の {r['lcoe_sbsp_$/kWh']/0.04:.0f}倍)\n\n"
        "Starship が $100/kg を達成すれば LCOE は **$10/kWh** 以下に。"
        "さらに GaN-on-SiC の RF 変換効率が 95% に達すれば **$1/kWh** が視野に入る。\n\n"
        "**2023年 Caltech が世界初の軌道上ワイヤレス電力伝送に成功** ← 概念実証済み ✓"
    )
