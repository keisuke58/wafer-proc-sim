"""
月面データセンター × 宇宙冷却 — 「日の当たらない場所」の物理
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

st.set_page_config(page_title="Lunar DC", layout="wide")
st.title("🌑 月面・永久陰クレーター データセンター")
st.caption("「日の当たらない月に置けば冷却できるのでは？」— 正解。でも本当の価値は量子コンピュータにある。")

from market_analysis.lunar_dc_model import (
    compare_environments, quantum_cooling_advantage,
    lunar_quantum_dc_design, integrated_lunar_system,
    ENVIRONMENTS, QUANTUM_COOLING, LUNAR_DC_CHALLENGES,
    radiator_area, SIGMA, EPS,
)

tab1, tab2, tab3 = st.tabs(["🌡️ 放熱物理", "⚛️ 量子コンピュータ", "🏗️ 統合設計"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — 放熱物理
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Stefan-Boltzmann 放熱板 — 環境別比較")

    P_kW   = st.slider("放熱電力 [kW]", 1, 100, 10, 1)
    T_hot  = st.slider("放熱板温度 [K]", 250, 400, 320, 10)

    results = compare_environments(P_kW, T_hot)

    k = st.columns(4)
    psr = next(r for r in results if r["env"] == "Moon_PSR")
    leo = next(r for r in results if r["env"] == "LEO_orbit")
    l2  = next(r for r in results if r["env"] == "L2_point")
    k[0].metric("月 PSR 面積",  f"{psr['area_m2']:.1f} m²")
    k[1].metric("LEO 面積",     f"{leo['area_m2']:.1f} m²")
    k[2].metric("L2 面積",      f"{l2['area_m2']:.1f} m²")
    k[3].metric("PSR の優位",   f"LEO比 {psr['vs_leo']:.2f}×")

    c1, c2 = st.columns(2)
    with c1:
        # 環境別放熱板面積 棒グラフ
        envs   = [r for r in results if r["area_m2"] is not None]
        labels = [r["label"] for r in envs]
        areas  = [r["area_m2"] for r in envs]
        solar  = [r["can_solar"] for r in envs]
        colors = ["#4CAF50" if s else "#e45756" for s in solar]

        fig = go.Figure(go.Bar(
            x=labels, y=areas,
            marker_color=colors,
            text=[f"{a:.1f}m²" for a in areas], textposition="outside",
        ))
        fig.add_annotation(x=0.5, y=0.95, xref="paper", yref="paper",
                            text="🟢=太陽光あり  🔴=太陽光なし",
                            showarrow=False, font=dict(color="white", size=11))
        fig.update_layout(title=f"必要放熱板面積 ({P_kW}kW, T_hot={T_hot}K)",
                          yaxis_title="面積 [m²]",
                          plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # T_bg別 放熱板面積 曲線
        T_bg_arr = np.linspace(10, T_hot - 10, 200)
        A_arr    = [radiator_area(P_kW, T_hot, T) for T in T_bg_arr]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=T_bg_arr, y=A_arr, name="必要面積",
                                   line=dict(color="#f5a623", width=2.5)))
        for env, d in ENVIRONMENTS.items():
            T = d["T_bg_K"]
            if T < T_hot:
                A = radiator_area(P_kW, T_hot, T)
                fig2.add_trace(go.Scatter(
                    x=[T], y=[A], mode="markers+text",
                    marker=dict(size=12, color="#4CAF50" if d["can_solar"] else "#e45756"),
                    text=[d["label"]], textposition="top right",
                    textfont=dict(size=8), showlegend=False,
                ))
        fig2.update_layout(title="背景温度 → 必要放熱板面積",
                           xaxis_title="背景温度 T_bg [K]", yaxis_title="面積 [m²]",
                           plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "💡 **結論**: 月 PSR の放熱効率は LEO より **22% 優れる** だけ。\n\n"
        "放熱目的だけなら **L2点 (太陽光あり + 深宇宙 4K)** が最強。\n"
        "月 PSR の本当の価値は次のタブにある → **量子コンピュータ冷却**"
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — 量子コンピュータ冷却
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("⚛️ 量子コンピュータ × 宇宙冷却 — これが本当のゲームチェンジャー")

    st.markdown("""
**現在の量子コンピュータの最大の課題: 冷却コスト**
- IBM Eagle / Google Sycamore: 動作温度 **15mK (-273.135°C)**
- 地上では希釈冷凍機が必要: **$3M/台 + 電力 200kW**
- 宇宙の深い場所なら **コスト激減**
    """)

    n_qubits = st.slider("論理量子ビット数", 10, 10000, 1000, 10)

    # 各環境の量子冷却性能
    locs = list(QUANTUM_COOLING.keys())
    q_results = [quantum_cooling_advantage(loc) for loc in locs]

    k = st.columns(5)
    for i, q in enumerate(q_results):
        delta = f"地上比 {q['improvement_x']:.0f}×" if q["improvement_x"] > 1 else ""
        k[i % 5].metric(
            q["label"],
            f"Carnot {q['carnot_ratio']:,.0f}×",
            delta or f"${q['fridge_cost_M$']:.1f}M 冷凍機",
        )

    c1, c2 = st.columns(2)
    with c1:
        # Carnot 比較
        labels_q = [q["label"] for q in q_results if q["feasible"]]
        carnots  = [q["carnot_ratio"] for q in q_results if q["feasible"]]
        colors_q = ["#e45756" if c > 1000 else "#f5a623" if c > 100 else "#4CAF50"
                    for c in carnots]

        fig = go.Figure(go.Bar(
            x=labels_q, y=carnots, marker_color=colors_q,
            text=[f"{c:,.0f}×" for c in carnots], textposition="outside",
        ))
        fig.update_layout(title="冷凍機 Carnot 比 (小さいほど省エネ)",
                          yaxis_type="log", yaxis_title="Carnot 比 (対数)",
                          plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # 論理 qubit 数 vs 冷凍機電力
        n_arr = np.logspace(1, 4, 50)
        envs_q = [("月 PSR+冷凍機", 40, "#4CAF50"),
                  ("地上 希釈冷凍機", 300, "#e45756"),
                  ("LEO 受動+冷凍機", 4, "#2166ac")]

        fig2 = go.Figure()
        for label, T_bg, col in envs_q:
            T_target = 0.015
            carnot = T_bg / T_target
            P_vals = [n * 1000 * 1e-6 / 1000 * carnot * 10 for n in n_arr]
            fig2.add_trace(go.Scatter(
                x=n_arr, y=P_vals, name=label,
                line=dict(color=col, width=2.5),
            ))

        fig2.add_vline(x=n_qubits, line_dash="dot", line_color="white",
                       annotation_text=f"選択中: {n_qubits}qbit")
        fig2.update_layout(
            title="論理 qubit 数 → 冷凍機消費電力",
            xaxis_title="論理 qubit 数", xaxis_type="log",
            yaxis_title="冷凍機電力 [W]", yaxis_type="log",
            plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 設計詳細
    qdc = lunar_quantum_dc_design(n_qubits, "Moon_PSR")
    st.markdown(f"""
#### 月 PSR 量子 DC 設計: {n_qubits} 論理 qubit

| 項目 | 月 PSR | 地上 | 倍率 |
|---|---|---|---|
| 物理 qubit 数 | {qdc['n_physical_qubits']:,} | {qdc['n_physical_qubits']:,} | — |
| 冷凍機消費電力 | **{qdc['P_fridge_W']/1000:.1f} kW** | {qdc['P_fridge_earth_W']/1000:.1f} kW | **{qdc['P_fridge_earth_W']/qdc['P_fridge_W']:.1f}× 省エネ** |
| 放熱板面積 | {qdc['radiator_m2']:.0f} m² | — (空調) | — |
| 希釈冷凍機コスト | $0.3M (小型) | $3M/台 | **10× 安い** |
| Kilopower 核電源 | {qdc['kilopower_units']} 台必要 | 不要 | — |
    """)

    st.warning(
        "⚛️ **ランキング (量子コンピュータ冷却に最適な場所)**\n\n"
        "1. 🥇 **LEO / L2 受動冷却** (4K) — Carnot 267×, 冷凍機電力 2.7kW\n"
        "2. 🥈 **月 PSR + 小型冷凍機** (40K→15mK) — 地上比7.5× 省エネ\n"
        "3. 🥉 **地上 希釈冷凍機** (300K→15mK) — $3M/台, 電力200kW\n\n"
        "月に置く意味: **安定した40K環境** + **水氷 (冷媒)** + **Starship 輸送可能**"
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — 統合設計
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🏗️ 月極 統合システム設計 (2035年以降)")
    st.caption("永久光峰 (太陽光あり) + 隣接 PSR (永久陰) の理想的立地を活用")

    sys_r = integrated_lunar_system()

    k = st.columns(5)
    k[0].metric("太陽光発電", f"{sys_r['P_solar_kW']:.0f} kW")
    k[1].metric("古典計算", f"{sys_r['classical_tflops']:.0f} TFLOPS")
    k[2].metric("量子計算", f"{sys_r['quantum_qubits']} 論理 qubit")
    k[3].metric("通信速度", f"{sys_r['data_rate_Gbps']:.0f} Gbps レーザー")
    k[4].metric("Starship 輸送", f"${sys_r['starship_cost_M$']:.0f}M (2t)")

    # システム構成図
    st.code("""
月 南極 永久光峰 (ピーク・オブ・エタナル・ライト)
    ↓ GaAs 太陽電池 1000m² → 476kW
    ↓
    ├─ 計算ユニット (PSR 内 -233°C)
    │    ├─ GPU クラスター: 23,818 TFLOPS
    │    └─ 量子コンピュータ: 100 論理 qubit (希釈冷凍機 極小)
    │
    ├─ 放熱板: 712m² @ PSR 40K (地上空調不要)
    │
    ├─ 水氷採掘 (PSR 内): 冷媒 + 推進剤 + 飲料水
    │
    └─ レーザー通信: 47 Gbps → 地球 (遅延 1.28秒)

用途:
  ✓ AI 学習 (遅延許容, 省電力)      → 地上 DC の 1/7 電力コスト
  ✓ 量子計算 (安定極低温)           → $3M 希釈冷凍機が不要
  ✓ データ永久アーカイブ (月面保管)   → 人類の知識を月に保存
  ✗ リアルタイム Web サービス        → 1.28秒遅延は致命的
""", language="text")

    # 課題テーブル
    st.markdown("#### 現実的課題と解決策")
    rows = []
    for key, ch in LUNAR_DC_CHALLENGES.items():
        sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(ch["severity"], "⚪")
        rows.append({
            "課題": f"{sev_icon} {ch['challenge']}",
            "解決策": ch["solution"],
            "実現時期": ch["timeline"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.success(
        "### 🌑 月 PSR DC の結論\n\n"
        "**ユーザーの直感は正しい。** 物理的に正しい発想。\n\n"
        "| 目的 | PSR の優位性 |\n"
        "|---|---|\n"
        "| 普通の DC 廃熱 | LEO比 22% 改善 (劇的ではない) |\n"
        "| **量子コンピュータ** | **地上比 7.5× 省エネ + $3M 冷凍機不要** |\n"
        "| データアーカイブ | 月面は安定・永続・地球から独立 |\n\n"
        f"SiC 電源管理 IC (放射線耐性) + GaN-on-SiC (通信 RF) + GaAs 太陽電池が必須。\n"
        f"Disco のダイシング技術が全チップを支える。"
    )
