"""
Disco — 全応用領域を繋ぐ最強ポジション
"""
import os, sys
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

st.set_page_config(page_title="Disco Moat", layout="wide")
st.title("🔪 Disco — 全産業を貫く最強ポジション")
st.caption("EV · AI · ロボット · ドローン · 衛星 · BESS · 核融合 · SBSP — 全部 Disco を通る")

from fem.disco_moat_model import (
    disco_total_tam, disco_revenue_from_application,
    moat_score, DISCO_BLADE_ECONOMICS, APPLICATION_TO_DISCO,
    DISCO_MOAT,
)

tam  = disco_total_tam()
moat = moat_score()
apps = tam["applications"]

# ── KPI ──────────────────────────────────────────────────────────────────
k = st.columns(5)
k[0].metric("モデル内 TAM",     f"${tam['total_M$']:.0f}M/年")
k[1].metric("消耗品比率",        f"{tam['blade_ratio']:.0f}%")
k[2].metric("SiC ブレード premium", f"Si 比 {moat['sic_premium_x']:.0f}×")
k[3].metric("モートスコア",      f"{moat['moat_score']}/10")
k[4].metric("特許数",           f"{moat['total_patents']}件+")

st.divider()

c1, c2 = st.columns(2)

with c1:
    # 全応用領域からの収益 — 横棒グラフ
    labels = [a["label"][:30] for a in apps if a["total_rev_M$"] > 0.05]
    revs   = [a["total_rev_M$"] for a in apps if a["total_rev_M$"] > 0.05]
    colors = [a["color"]        for a in apps if a["total_rev_M$"] > 0.05]
    trends = [a["trend"]        for a in apps if a["total_rev_M$"] > 0.05]

    trend_icons = {"explosive":"🚀","very_fast":"⚡","fast_growth":"📈",
                   "growth":"↗","stable":"→","future":"🔮","slow_growth":"↘"}
    label_disp = [f"{trend_icons.get(t,'')} {l}" for l, t in zip(labels, trends)]

    fig = go.Figure(go.Bar(
        y=label_disp, x=revs, orientation="h",
        marker_color=colors,
        text=[f"${v:.1f}M" for v in revs], textposition="outside",
    ))
    fig.update_layout(
        title="Disco 応用領域別 年間収益 (モデル内 TAM)",
        xaxis_title="Disco 収益 [$M/年]",
        plot_bgcolor="#111", paper_bgcolor="#111", font_color="white",
        height=520, margin=dict(l=270),
        xaxis=dict(range=[0, max(revs) * 1.3]),
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # 材料別 収益 × ブレードコスト (バブル)
    mats    = list(tam["by_material"].keys())
    revs_m  = [tam["by_material"][m] for m in mats]
    blade_c = [DISCO_BLADE_ECONOMICS[m]["blade_cost_per_wafer"] for m in mats]
    wpd_m   = [sum(a["wafers_day"] for a in apps if a["material"] == m) for m in mats]
    colors_m= [DISCO_BLADE_ECONOMICS[m]["color"] for m in mats]

    fig2 = go.Figure()
    for i, mat in enumerate(mats):
        fig2.add_trace(go.Scatter(
            x=[blade_c[i]], y=[revs_m[i]],
            mode="markers+text",
            marker=dict(size=max(wpd_m[i] ** 0.4 * 15, 15),
                        color=colors_m[i], opacity=0.85,
                        line=dict(color="white", width=1.5)),
            text=[mat], textposition="top center",
            textfont=dict(size=11, color="white"),
            name=mat, showlegend=False,
            hovertemplate=f"<b>{mat}</b><br>ブレード: ${blade_c[i]:.0f}/wafer<br>"
                          f"Disco収益: ${revs_m[i]:.1f}M<extra></extra>",
        ))
    fig2.update_layout(
        title="材料別: ブレード単価 × Disco 収益<br>(バブル大=ウェーハ枚数多)",
        xaxis_title="ブレードコスト [$/wafer] (対数)", xaxis_type="log",
        yaxis_title="Disco 収益 [$M/年]",
        plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=520,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── ブレードコスト比較 ─────────────────────────────────────────────────────
st.divider()
st.subheader("材料別 ブレードコスト — SiC は Si の 125倍")

blade_mats  = list(DISCO_BLADE_ECONOMICS.keys())
blade_costs = [DISCO_BLADE_ECONOMICS[m]["blade_cost_per_wafer"] for m in blade_mats]
blade_lives = [DISCO_BLADE_ECONOMICS[m]["blade_life_wafers"]    for m in blade_mats]
blade_clrs  = [DISCO_BLADE_ECONOMICS[m]["color"]                for m in blade_mats]
blade_methods=[DISCO_BLADE_ECONOMICS[m]["dicing_method"]        for m in blade_mats]

c3, c4 = st.columns(2)
with c3:
    fig3 = go.Figure(go.Bar(
        x=blade_mats, y=blade_costs,
        marker_color=blade_clrs,
        text=[f"${c:.1f}" for c in blade_costs], textposition="outside",
    ))
    fig3.add_hline(y=DISCO_BLADE_ECONOMICS["Si"]["blade_cost_per_wafer"],
                   line_dash="dot", line_color="gray",
                   annotation_text="Si 基準 $0.4/wafer")
    fig3.update_layout(
        title="材料別 ブレードコスト [$/wafer]", yaxis_type="log",
        yaxis_title="$/wafer (対数)", plot_bgcolor="#111",
        paper_bgcolor="#111", font_color="white", height=340,
    )
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    fig4 = go.Figure(go.Bar(
        x=blade_mats, y=blade_lives,
        marker_color=blade_clrs,
        text=[f"{l}枚" for l in blade_lives], textposition="outside",
    ))
    fig4.update_layout(
        title="ブレード寿命 [wafer/枚]  ← 短いほど消耗品売れる",
        yaxis_title="wafer/枚", yaxis_type="log",
        plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=340,
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── 供給チェーン フロー ────────────────────────────────────────────────────
st.divider()
st.subheader("全産業 → Disco → 最終製品 フロー")
st.code("""
AI チップ (NVIDIA H100)  ──┐
EV SiC インバータ         ──┤
ヒューマノイド GaN ESC    ──┤
軍用ドローン GaN          ──┤
eVTOL SiC                ──┤──▶ [Disco ダイシング装置] ──▶ 個別チップ
Starlink ビームフォーミング──┤      世界シェア 70%
衛星 SiC 電源            ──┤      SiC ブレード $50/wafer (Si の 125倍)
系統 BESS SiC            ──┤      ステルスダイシング特許 (競合なし)
核融合 SiC / Ga₂O₃       ──┤
宇宙太陽光 GaAs          ──┘

「半導体チップが存在する限り、Disco は永遠に必要とされる」
""", language="text")

# ── モート詳細 ────────────────────────────────────────────────────────────
st.subheader("Disco の経済的堀 (Warren Buffett 的分析)")
c5, c6 = st.columns(2)
with c5:
    for key, m in DISCO_MOAT.items():
        moat_color = {"very_high": "🟢", "high": "🟡", "medium": "🟠"}.get(m["moat"], "⚪")
        st.markdown(f"**{moat_color} {m['title']}**")
        st.caption(m["detail"])
        if m["patents"]:
            st.caption(f"特許 {m['patents']}件+")
        st.divider()

with c6:
    # Disco 収益モデル図
    fig5 = go.Figure(go.Waterfall(
        name="Disco 収益構造",
        orientation="v",
        measure=["relative","relative","total"],
        x=["消耗品 (ブレード)\nリカーリング", "装置販売\n(5年更新)", "合計 TAM"],
        y=[tam["blade_M$"], tam["saw_M$"], 0],
        text=[f"${tam['blade_M$']:.0f}M", f"${tam['saw_M$']:.0f}M",
              f"${tam['total_M$']:.0f}M"],
        connector=dict(line=dict(color="#444")),
        decreasing=dict(marker_color="#e45756"),
        increasing=dict(marker_color="#4CAF50"),
        totals=dict(marker_color="#2166ac"),
    ))
    fig5.update_layout(
        title="Disco 収益ウォーターフォール (モデル内)",
        yaxis_title="[$M/年]",
        plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=340,
    )
    st.plotly_chart(fig5, use_container_width=True)

# ── 結論 ──────────────────────────────────────────────────────────────────
st.divider()
st.success(f"""
### 🏆 結論: Disco は「半導体の関所」

**どんな新技術が登場しても Disco を通らずには商品にならない。**

| 新興市場 | 材料 | Disco へのインパクト |
|---|---|---|
| ヒューマノイドロボット (CAGR 150%) | GaN | ブレード需要 急拡大 |
| 軍用ドローン (CAGR 58%) | GaN | 量産 × 消耗品↑ |
| 系統 BESS (CAGR 47%) | SiC | **125倍**プレミアム継続 |
| 核融合 (2040年代) | SiC + **Ga₂O₃** | 超高単価 新市場 |
| SBSP 宇宙太陽光 (2040年) | GaAs + SiC | km² 規模の大量需要 |

**SiC ブレード = Si の 125倍の単価 × 再来需要 = 最強のリカーリングビジネス**

モートスコア: **{moat['moat_score']}/10** — ステルスダイシング特許により競合実質ゼロ
""")
