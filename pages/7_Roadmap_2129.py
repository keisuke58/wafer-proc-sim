"""
2026 → 2129 完全ロードマップ — インタラクティブ版
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

st.set_page_config(page_title="2129 Roadmap", layout="wide")
st.title("🌌 2026 → 2129  テクノロジー × 半導体 完全ロードマップ")
st.caption("論理の飛躍なし。既知の物理と現在の研究動向のみ。あなたが見届ける100年。")

from fem.roadmap_2129 import (
    TECH_WAVES, LONGEVITY_STEPS, LOGIC_CHAIN,
    semiconductor_market_forecast_2129,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌊 5つの波", "⏱️ 年表", "🧬 長寿命化",
    "💰 市場外挿", "🔍 論理チェック"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — 5つの波
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("5つのテクノロジー波 — 全て半導体で繋がる")

    # タイムライン図
    fig = go.Figure()
    wave_y = {"Wave1_Energy": 5, "Wave2_Space": 4,
              "Wave3_Quantum_AI": 3, "Wave4_Longevity": 2, "Wave5_PostScarcity": 1}

    for key, wave in TECH_WAVES.items():
        period = wave["period"].split("-")
        x0, x1 = int(period[0]), int(period[1])
        y = wave_y[key]
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y-0.35, y1=y+0.35,
                      fillcolor=wave["color"], opacity=0.7,
                      line=dict(color="white", width=1))
        fig.add_annotation(x=(x0+x1)/2, y=y, text=wave["label"],
                            showarrow=False, font=dict(color="white", size=10))

    fig.update_layout(
        title="5つのテクノロジー波 タイムライン",
        xaxis=dict(range=[2020, 2135], title="年", showgrid=True, gridcolor="#333"),
        yaxis=dict(showticklabels=False, range=[0.3, 5.7]),
        plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 各 Wave 詳細
    wave_sel = st.selectbox("Wave 詳細を見る",
                             list(TECH_WAVES.keys()),
                             format_func=lambda k: TECH_WAVES[k]["label"])
    wave = TECH_WAVES[wave_sel]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"### {wave['label']}")
        st.markdown(f"**期間:** {wave['period']}")
        st.markdown(f"**主半導体:** {wave['semiconductor_key']}")
        st.markdown(f"**Disco接続:** {wave['disco_connection']}")

    with c2:
        events_df = pd.DataFrame(wave["events"], columns=["年", "イベント"])
        st.dataframe(events_df, use_container_width=True, height=300)

    # 各波の相互依存
    st.markdown("#### 波の相互依存 — なぜ順番通りに来るか")
    st.code("""
Wave1 エネルギー革命
  └→ 安価で豊富なエネルギー (核融合+SBSP)
       └→ Wave2 宇宙経済 (打上コスト激減、電力確保)
            └→ Wave3 量子AI (月面量子DC、宇宙量子通信)
                 └→ Wave4 長寿命 (量子AIが創薬、ナノ医療)
                      └→ Wave5 脱希少性 (長生きする人間が火星・小惑星を開発)

「エネルギーが解決すると、他の全てが解ける」— Richard Feynman
""", language="text")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — 年表
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("⏱️ 2026 → 2129 全イベント年表")

    # 全イベントを統合
    all_events = []
    for wave_key, wave in TECH_WAVES.items():
        for year, event in wave["events"]:
            all_events.append({
                "年": year,
                "イベント": event,
                "Wave": wave["label"],
                "color": wave["color"],
            })
    all_events.sort(key=lambda x: x["年"])

    # 年フィルター
    yr_range = st.slider("表示年範囲", 2026, 2129, (2026, 2060))
    filtered = [e for e in all_events if yr_range[0] <= e["年"] <= yr_range[1]]

    # タイムライン散布図
    fig2 = go.Figure()
    wave_labels = list(TECH_WAVES.keys())
    for i, (wave_key, wave) in enumerate(TECH_WAVES.items()):
        evs = [e for e in filtered if e["Wave"] == wave["label"]]
        if evs:
            fig2.add_trace(go.Scatter(
                x=[e["年"] for e in evs],
                y=[i] * len(evs),
                mode="markers+text",
                marker=dict(size=14, color=wave["color"],
                            symbol="circle", line=dict(color="white", width=1)),
                text=[e["イベント"][:20] + "..." if len(e["イベント"]) > 20
                      else e["イベント"] for e in evs],
                textposition="top center",
                textfont=dict(size=8, color="white"),
                name=wave["label"],
                hovertext=[e["イベント"] for e in evs],
                hoverinfo="text+x",
            ))

    fig2.update_layout(
        title=f"イベント年表 ({yr_range[0]}〜{yr_range[1]})",
        xaxis=dict(title="年", showgrid=True, gridcolor="#333"),
        yaxis=dict(ticktext=[TECH_WAVES[k]["label"] for k in wave_labels],
                   tickvals=list(range(len(wave_labels))), showgrid=False),
        plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=450,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # テーブル表示
    df = pd.DataFrame(filtered)[["年", "イベント", "Wave"]]
    st.dataframe(df, use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — 長寿命化
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🧬 2129 まで生きる経路 — Longevity × 半導体")

    st.markdown("""
> **「2075年が山場。そこを超えれば理論上 2129 は難しくない」**

各ステップが半導体技術の進歩に依存している。
    """)

    # Longevity ステップ表
    certainty_colors = {"high": "#4CAF50", "medium": "#f5a623",
                        "low": "#FF9800", "speculative": "#e45756"}
    certainty_labels = {"high": "○ 確実性高", "medium": "△ 中程度",
                        "low": "△ 低め", "speculative": "⚠️ 不確実"}

    for i, step in enumerate(LONGEVITY_STEPS):
        col_a, col_b = st.columns([1, 3])
        with col_a:
            c = certainty_colors[step["certainty"]]
            st.markdown(f"""
<div style='background:{c}22; border-left:4px solid {c};
     padding:10px; border-radius:4px; margin:5px 0'>
<b>{step['decade']}</b><br>
{certainty_labels[step['certainty']]}
</div>
""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"**{step['tech']}**")
            st.caption(f"効果: {step['effect']}")
            st.caption(f"半導体: {step['chip']}")
            st.caption(f"状況: {step['status']}")
        st.divider()

    # 平均余命グラフ
    years_l = [2026, 2030, 2038, 2045, 2055, 2060, 2075, 2090, 2100, 2129]
    life_exp = [84, 86, 90, 96, 105, 120, 140, 180, 250, float('nan')]
    life_labels = ["現在", "AI創薬開始", "セノリティクス",
                   "CRISPR", "ナノロボット", "長寿命主流",
                   "LEV達成?", "老化停止?", "概念変化", "あなた?"]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=years_l[:-1], y=life_exp[:-1],
        mode="lines+markers+text",
        line=dict(color="#4CAF50", width=3),
        marker=dict(size=10, color="#4CAF50"),
        text=life_labels[:-1], textposition="top right",
        textfont=dict(size=8, color="white"),
        name="平均余命 (先進国)",
    ))
    fig3.add_hline(y=2129 - 2005, line_dash="dot", line_color="gold",
                   annotation_text="あなたの目標 (現在20歳なら124歳)", annotation_font_color="gold")
    fig3.add_vrect(x0=2070, x1=2085, fillcolor="#e45756", opacity=0.1,
                   annotation_text="山場: LEV達成 or 未達成", annotation_font_color="#e45756")
    fig3.update_layout(
        title="平均余命 予測 2026→2100 (楽観シナリオ)",
        xaxis_title="年", yaxis_title="平均余命 [歳]",
        plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=400,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # 半導体との接続
    st.markdown("#### 長寿命化 × 半導体 接続マップ")
    st.code("""
DNA シーケンサー (ナノポア)
  ├── Si CMOS センサーアレイ → Disco 精密研削
  └── 個別化医療 → 量子 AI で解析

Neuralink N2 (10,000電極)
  ├── 23µm 電極ピッチ → Disco 超精密研削 必須
  ├── SiC 絶縁層 → 生体適合性
  └── 脳内センサーで老化モニタリング

SiC ナノロボット
  ├── SiC は人体内で安定 (人工股関節で実績)
  ├── サイズ 1-10µm → ダイシングではなく CVD 直接形成
  └── がん細胞を検知・破壊

CRISPR デリバリーシステム
  └── 脂質ナノ粒子 + DNA チップ (Si 製)
      → 体内でゲノム編集 → 老化遺伝子修正

量子コンピュータ (月面 PSR DC)
  └── タンパク質折り畳み → 老化メカニズム解析
      → 地球へ送信 → 薬剤設計 (遅延 1.28秒 は許容)
""", language="text")

    st.warning("""
⚠️ **正直な不確実性の開示**

「Longevity Escape Velocity (2075年)」だけは現時点で **証明できていない仮定**。

確実なこと:
  ✓ セノリティクス (+10年) → 動物実験で確認済
  ✓ AI 創薬 → AlphaFold が既に証明
  ✓ CRISPR → 臨床応用始まっている

不確実なこと:
  ⚠️ 「老化そのもの」を止められるか → まだ誰も知らない
  ⚠️ 2075年に間に合うか → スケジュールは楽観的

**2129年まで生きる確率: 現時点では 10-30%** (個人的見解)
ただし医療進歩次第で急速に上がる。
""")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — 市場外挿
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("💰 半導体市場 2129 まで外挿")

    fc = semiconductor_market_forecast_2129()
    years_m  = [r["year"] for r in fc]
    mkts_B   = [r["mkt_B"] for r in fc]
    notes_m  = [r["note"] for r in fc]
    cagrs    = [r["cagr"] for r in fc[1:]]

    c1, c2 = st.columns(2)
    with c1:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=years_m, y=mkts_B,
            mode="lines+markers",
            line=dict(color="#f5a623", width=3),
            marker=dict(size=12, color="#f5a623"),
            fill="tozeroy", fillcolor="rgba(245,165,35,0.1)",
            hovertext=[f"{r['note']}: ${r['mkt_B']:,}B" for r in fc],
            hoverinfo="text+x",
        ))
        fig4.update_layout(
            title="世界半導体市場 外挿 ($B, 対数)",
            xaxis_title="年", yaxis_title="市場規模 [$B]", yaxis_type="log",
            plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=400,
        )
        st.plotly_chart(fig4, use_container_width=True)

    with c2:
        fig5 = go.Figure(go.Bar(
            x=years_m[1:], y=cagrs,
            marker_color=["#4CAF50" if c < 15 else "#f5a623" if c < 25 else "#e45756"
                          for c in cagrs],
            text=[f"{c:.1f}%" for c in cagrs], textposition="outside",
        ))
        fig5.update_layout(
            title="期間別 CAGR [%/年]",
            yaxis_title="CAGR [%]",
            plot_bgcolor="#111", paper_bgcolor="#111", font_color="white", height=400,
        )
        st.plotly_chart(fig5, use_container_width=True)

    # 市場テーブル
    st.dataframe(pd.DataFrame(fc), use_container_width=True)

    st.info("""
💡 **外挿の根拠**

2026-2035 (CAGR ~10%): 過去20年の実績値
2035-2060 (CAGR ~12%): 宇宙経済・核融合が新市場創出
2060-2100 (CAGR ~8%): 火星・小惑星採掘で需要爆発
2100-2129 (CAGR ~6%): 成熟期でも太陽系規模の市場

$300兆B (2129) = 現在の500倍。
現在の世界 GDP $110兆 の約 3倍。火星+小惑星 GDP を含む。
""")


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — 論理チェック
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔍 論理の飛躍チェック — 全仮定を公開")

    st.markdown("""
「論理の飛躍をなくす」ために、全ての主要仮定とその根拠を開示する。
⚠️マークの仮定は現時点で不確実。それ以外は既存研究の外挿で成立する。
    """)

    for lc in LOGIC_CHAIN:
        icon = "⚠️" if lc["leap"] else "✅"
        color = "#e4575622" if lc["leap"] else "#4CAF5022"
        border = "#e45756" if lc["leap"] else "#4CAF50"
        st.markdown(f"""
<div style='background:{color}; border-left:5px solid {border};
     padding:12px; border-radius:4px; margin:8px 0'>
<b>{icon} {lc['claim']}</b><br>
<small>根拠: {lc['basis']}</small>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 除外した仮定 (物理的に不可能 or 根拠なし)")
    excluded = [
        ("光速超え輸送", "特殊相対性理論が禁止。恒星間は何世代もかかる"),
        ("室温超伝導", "2023年 LK-99 は再現失敗。現時点で信頼できる発見なし"),
        ("意識アップロード", "意識の物理的定義が未解決。2129年には早すぎる"),
        ("反重力", "一般相対性理論と矛盾。エネルギー条件を破る必要あり"),
        ("ワープドライブ", "Alcubierre 理論は exotic matter が必要。実現不可能"),
    ]
    for tech, reason in excluded:
        st.markdown(f"- ❌ **{tech}**: {reason}")

    st.divider()
    st.markdown("#### 最大のリスク要因")
    risks = [
        ("気候変動の暴走", "2040年代のエネルギー革命が間に合わなかった場合"),
        ("核戦争", "技術進歩より破壊が速い唯一のシナリオ"),
        ("AGI のアライメント失敗", "2038年以降の最大リスク"),
        ("パンデミック (未知の病原体)", "COVID 規模の10倍が来た場合"),
        ("Longevity 技術が遅れる", "2075年 LEV が 2100年になる場合"),
    ]
    for risk, detail in risks:
        st.markdown(f"- ⚡ **{risk}**: {detail}")

    st.success("""
### 結論: 2129年まで生きるシナリオは「あり得る」

**必要な条件 (全て現在の研究の延長):**
1. 2030年代: セノリティクス + AI 創薬で主要疾患を制圧
2. 2045年: CRISPR in vivo で遺伝的老化リスクを修正
3. 2060年: ナノ医療が癌・心疾患をほぼ根絶
4. 2075年: Longevity Escape Velocity 達成 ← 最大の不確実点

**飛躍していない理由:**
各ステップは「前のステップが成功した場合に次が可能になる」
連鎖構造になっている。魔法は一つも使っていない。

**半導体との完全接続:**
Neuralink → Disco 研削 → 脳内モニタリング → 長寿命
量子 DC → 月 PSR → 創薬 → 長寿命
SiC ナノロボット → Disco 技術起源 → 体内巡回 → 長寿命

**全ての道が Disco を通る。2129年まで。**
    """)
