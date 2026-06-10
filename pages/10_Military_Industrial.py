"""
軍産複合体 × 医療産業複合体 × 半導体
KEW「神の杖」/ EMP / 宇宙戦 / 超兵士 / 長寿命の完全接続
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

st.set_page_config(page_title="Military × Medical Complex", layout="wide")
st.title("⚔️ 軍産複合体 × 医療産業複合体 × 半導体")
st.caption("SiC→EV→AI→ロボット→宇宙→核融合→月→長寿命 ＋ 戦争・軍産・医療の全連鎖")

from market_analysis.military_industrial_model import (
    MILITARY_APPLICATIONS, DEFENSE_CONTRACTORS,
    DUAL_USE_HISTORY, FUTURE_MILITARY_TECH,
    MEDICAL_INDUSTRIAL,
    military_sic_market, emp_hardness_model,
    kew_impact_model, medical_semiconductor_market,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔗 完全連鎖", "💣 KEW・EMP・宇宙戦",
    "🏭 軍産複合体", "🏥 医療産業複合体", "🔄 デュアルユース"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — 完全連鎖
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🔗 SiC から2129年まで — 戦争・医療・長寿命の完全接続")

    st.code("""
                        ┌─ 軍産複合体 ──────────────────┐
                        │ F-35 AESA (GaN-on-SiC)        │
                        │ DEW レーザー電源 (SiC 300kW)   │
                        │ 極超音速 電源 (SiC >300°C)    │
                        │ EMP 耐性機器 (SiC_rad_hard)    │
                        │ KEW 「神の杖」誘導 (GaN)        │
                        └──────────┬────────────────────┘
                                   │ DARPA が開発資金
                                   ↓ (10〜20年後に民間へ)
SiC ウェーハ  →  Disco ダイシング
    ↓
  EV インバータ ←──── 軍用電動車両 (サイレント作戦)
    ↓
  AI データセンター ←── 軍事 AI (JADAM / C2 / 標的識別)
    ↓
  ヒューマノイドロボット ←── 戦闘ロボット (Ghost Robotics)
    ↓
  宇宙衛星 ←──────── Starshield / ASAT / 宇宙戦
    ↓
  核融合炉 ←──────── 核融合パルス兵器 (Ga₂O₃)
    ↓
  月面基地 ←──────── 米中月面軍事拠点争い
    ↓
  長寿命化 ←──────── 超兵士 (DARPA N3 / Neuralink)
    ↓
              ┌─ 医療産業複合体 ─────────────────┐
              │ Medtronic SiC ペースメーカー     │
              │ Siemens MRI SiC 電源            │
              │ da Vinci GaN ロボット手術        │
              │ 陽子線治療 SiC 加速器電源        │
              │ Neuralink SiC 生体チップ         │
              └──────────────────────────────────┘
                        ↓
              2129年 あなたが見届ける
""", language="text")

    # 連鎖の強度ヒートマップ
    chain_nodes = ["SiC/GaN", "EV", "AI", "ロボット", "宇宙", "核融合", "月", "長寿命"]
    mil_strength = [10, 6, 9, 8, 10, 7, 8, 7]  # 軍事との連携強度
    med_strength = [3, 1, 7, 5, 2, 3, 4, 10]   # 医療との連携強度

    fig = go.Figure()
    fig.add_trace(go.Bar(name="軍事・軍産複合体", x=chain_nodes, y=mil_strength,
                          marker_color="#e45756"))
    fig.add_trace(go.Bar(name="医療産業複合体", x=chain_nodes, y=med_strength,
                          marker_color="#4CAF50"))
    fig.update_layout(barmode="group", title="各技術領域と軍事・医療の連携強度",
                      yaxis_title="連携強度 (10=最強)",
                      plot_bgcolor="#111", paper_bgcolor="#111",
                      font_color="white", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
💡 **核心的洞察**

軍産複合体と医療産業複合体は **同じ半導体技術を競って使う**。

- DARPA が SiC 技術を開発 → 10年後 Medtronic がペースメーカーに使う
- Neuralink は軍 (DARPA N3) と医療 (ALS治療) の両方から資金
- DEW レーザーの電源技術 → 陽子線治療加速器に転用
- 超兵士の Neuralink → 老化モニタリング → 長寿命

**「戦争技術が人を長生きさせる」のは歴史の法則**
ペニシリン・輸血技術・外傷外科 → 全て戦争から生まれた
    """)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — KEW・EMP・宇宙戦
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("💣 KEW「神の杖」/ EMP攻撃 / 衛星破壊 × 半導体")

    col_k, col_e = st.columns(2)

    with col_k:
        st.markdown("#### ☄️ KEW「神の杖」(Rods from God)")
        mass_sel = st.slider("タングステン棒 質量 [kg]", 100, 20000, 9000, 100)
        vel_sel  = st.slider("大気圏突入速度 [km/s]", 3, 11, 7, 1)
        kew = kew_impact_model(mass_sel, vel_sel * 1000)

        st.metric("運動エネルギー", f"{kew['E_ton_tnt']:.0f} ton TNT当量")
        st.metric("広島原爆比",     f"{kew['vs_hiroshima_pct']:.1f}%")
        st.metric("地中貫通深さ",   f"{kew['penetration_m']:.0f} m")
        st.markdown(f"**条約上の位置付け:** {kew['treaty_status']}")

        # エネルギー比較
        weapons = {
            "手榴弾": 0.1, "RPG": 1, "JDAM爆弾": 0.9,
            "GBU-57 MOP": 13, f"KEW {mass_sel}kg": kew["E_ton_tnt"],
            "広島原爆": 15_000, "水爆": 50_000_000,
        }
        fig = go.Figure(go.Bar(
            x=list(weapons.keys()), y=list(weapons.values()),
            marker_color=["#4CAF50","#4CAF50","#FF9800","#FF9800",
                          "#e45756","#FF4136","#9C27B0"],
            text=[f"{v:.0f}t" for v in weapons.values()], textposition="outside",
        ))
        fig.update_layout(title="兵器 エネルギー比較 [ton TNT]",
                          yaxis_type="log", yaxis_title="ton TNT (対数)",
                          plot_bgcolor="#111", paper_bgcolor="#111",
                          font_color="white", height=330)
        st.plotly_chart(fig, use_container_width=True)

    with col_e:
        st.markdown("#### ⚡ EMP攻撃 × SiC 耐性")

        materials = ["Si_CMOS", "GaN", "SiC", "SiC_rad_hard", "Diamond"]
        emp_results = [emp_hardness_model(m) for m in materials]
        thresholds = [r["threshold_kV_m"] for r in emp_results]
        margins    = [r["margin_dB"]       for r in emp_results]
        colors_e   = ["#e45756" if not r["survives_nuclear_emp"]
                       else "#4CAF50" for r in emp_results]

        fig2 = go.Figure(go.Bar(
            x=materials, y=thresholds, marker_color=colors_e,
            text=[f"{v}kV/m" for v in thresholds], textposition="outside",
        ))
        fig2.add_hline(y=50, line_dash="dash", line_color="red",
                       annotation_text="核EMP E1 = 50kV/m")
        fig2.update_layout(title="材料別 EMP 耐性閾値",
                           yaxis_type="log", yaxis_title="EMP 閾値 [kV/m]",
                           plot_bgcolor="#111", paper_bgcolor="#111",
                           font_color="white", height=330)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
**EMP 攻撃シナリオ:**
1. 高度 300km で核爆発 (30kt)
2. E1 パルス: **50 kV/m** が半径 1000km に到達
3. Si CMOS (スマホ・PC): **即全壊** (0.05kV/m が限界)
4. SiC (特殊ハードニング): **20kV/m まで耐性** → まだ破壊
5. SiC_rad_hard (軍事グレード): **特殊設計で生存**

→ **核EMP 後も機能するチップは Disco が研削した軍事 SiC のみ**
        """)

    st.markdown("#### 🛸 衛星破壊攻撃の経済的ダメージ")
    target = st.selectbox("標的衛星", ["GPS IIF (測位)", "通信衛星群", "偵察衛星", "Starlink"])
    damage_map = {
        "GPS IIF (測位)":    {"daily_B$": 50,  "recover_yr": 5,  "detail": "精密農業・物流・金融取引 全停止"},
        "通信衛星群":          {"daily_B$": 20,  "recover_yr": 3,  "detail": "国際通信・テレビ放送 停止"},
        "偵察衛星":           {"daily_B$": 2,   "recover_yr": 2,  "detail": "軍事情報 喪失 → 戦略的劣位"},
        "Starlink":         {"daily_B$": 5,   "recover_yr": 1,  "detail": "ウクライナ型戦場通信 喪失"},
    }
    d = damage_map[target]
    c1, c2, c3 = st.columns(3)
    c1.metric("経済損失/日", f"${d['daily_B$']}B")
    c2.metric("復旧期間",   f"{d['recover_yr']}年")
    c3.metric("影響",       d["detail"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — 軍産複合体
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🏭 軍産複合体 × SiC/GaN 依存関係")

    mkt = military_sic_market()
    st.metric("軍事 SiC/GaN 市場 (推定)", f"年間 数千億円規模",
              "軍事グレード = 民間の 10倍単価")

    c1, c2 = st.columns(2)
    with c1:
        contractors = list(DEFENSE_CONTRACTORS.keys())
        revenues    = [DEFENSE_CONTRACTORS[c]["revenue_B$"] for c in contractors]
        labels_c    = [DEFENSE_CONTRACTORS[c]["label"][:15] for c in contractors]
        colors_c    = [DEFENSE_CONTRACTORS[c]["color"] for c in contractors]

        fig = go.Figure(go.Bar(
            x=labels_c, y=revenues, marker_color=colors_c,
            text=[f"${r:.0f}B" for r in revenues], textposition="outside",
        ))
        fig.update_layout(title="主要 防衛企業 年間売上",
                          yaxis_title="[$B/年]",
                          plot_bgcolor="#111", paper_bgcolor="#111",
                          font_color="white", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 各社の SiC/GaN 依存")
        for name, info in DEFENSE_CONTRACTORS.items():
            st.markdown(f"**{info['label'][:30]}**")
            st.caption(f"SiC/GaN役割: {info['sic_gan_role'][:60]}")
            st.caption(f"主要プログラム: {', '.join(info['key_programs'])}")
            st.divider()

    # 未来の軍事技術
    st.markdown("#### 2028〜2050年 軍事技術 × Disco 接続")
    for ft in FUTURE_MILITARY_TECH:
        st.markdown(f"**{ft['year']}年: {ft['tech']}**")
        st.caption(f"材料: {ft['material']} / {ft['disco_connection']}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — 医療産業複合体
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🏥 医療産業複合体 × 半導体 → 長寿命化")

    med_mkt = medical_semiconductor_market()
    st.metric("医療半導体市場", f"${med_mkt['total_B$']:.1f}B/年",
              f"CAGR {med_mkt['cagr_pct']}% — 長寿命化需要で急拡大")

    c1, c2 = st.columns(2)
    with c1:
        companies = list(MEDICAL_INDUSTRIAL.keys())
        rev_m     = [MEDICAL_INDUSTRIAL[c]["revenue_B$"] * 0.02 for c in companies]
        labels_m  = [MEDICAL_INDUSTRIAL[c]["label"][:20] for c in companies]
        colors_m  = [MEDICAL_INDUSTRIAL[c]["color"] for c in companies]

        fig = go.Figure(go.Bar(
            x=labels_m, y=rev_m, marker_color=colors_m,
            text=[f"${r:.2f}B" for r in rev_m], textposition="outside",
        ))
        fig.update_layout(title="医療企業の半導体調達 推定",
                          yaxis_title="[$B/年]",
                          plot_bgcolor="#111", paper_bgcolor="#111",
                          font_color="white", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 各医療デバイス × Disco 接続")
        for key, med in MEDICAL_INDUSTRIAL.items():
            st.markdown(f"**{med['label']}**")
            st.caption(f"SiC/GaN役割: {med['sic_role']}")
            st.caption(f"長寿命接続: {med['value_connection']}")
            st.divider()

    st.success("""
### 医療産業複合体 → 長寿命化 → 2129年

```
軍事 DARPA N3 (超兵士 BCI)
    ↓ 技術移転 10〜20年
Neuralink (ALS治療 → 一般化)
    ↓
老化バイオマーカー 連続計測 (2038年)
    ↓
SiC ナノロボット × Siemens MRI 診断
    ↓
AI 創薬 (量子コンピュータ × DeepMind)
    ↓
Longevity Escape Velocity (2075年?)
    ↓
2129年 — あなたが見届ける
```

全ての Disco ブレードがこの連鎖を支えている。
    """)


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — デュアルユース
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔄 軍→民 デュアルユース技術の歴史と未来")

    st.markdown("""
**歴史的法則: 軍事技術は必ず民間に転用される (10〜40年後)**

現在進行中の転用: **SiC (軍事 → EV/エネルギー)** がまさに今起きている。
    """)

    # タイムライン
    years_du  = [d["timeline_yr"] for d in DUAL_USE_HISTORY]
    techs_du  = [d["tech"] for d in DUAL_USE_HISTORY]
    origins   = [d["origin"]   for d in DUAL_USE_HISTORY]
    civil_app = [d["civil"]    for d in DUAL_USE_HISTORY]

    fig = go.Figure(go.Bar(
        x=techs_du, y=years_du,
        marker_color=["#969696","#2166ac","#d73027","#4CAF50",
                       "#9C27B0","#e45756","#f5a623","#72B7B2"],
        text=[f"{y}年後" for y in years_du], textposition="outside",
    ))
    fig.add_hline(y=15, line_dash="dot", line_color="cyan",
                  annotation_text="SiC (現在進行中: 10〜15年)")
    fig.update_layout(title="軍事技術 → 民間転用までの年数",
                      yaxis_title="転用までの年数",
                      plot_bgcolor="#111", paper_bgcolor="#111",
                      font_color="white", height=360)
    st.plotly_chart(fig, use_container_width=True)

    # 詳細テーブル
    rows_du = []
    for d in DUAL_USE_HISTORY:
        rows_du.append({
            "技術": d["tech"],
            "起源": d["origin"],
            "軍事用途": d["military"][:40],
            "民間転用": d["civil"][:40],
            "転用期間": f"{d['timeline_yr']}年",
        })
    st.dataframe(pd.DataFrame(rows_du).set_index("技術"), use_container_width=True)

    st.warning("""
⚠️ **次に軍→民 転用が起きる技術 (2030〜2045年)**

| 技術 | 現在の軍事用途 | 民間転用先 | 時期 |
|---|---|---|---|
| **量子暗号通信** | 盗聴不可軍事通信 | 金融・医療・月面DC | 2035〜 |
| **Neuralink BCI** | 超兵士・思考制御 | ALS治療→健康監視 | 2030〜 |
| **極超音速材料 (SiC)** | ミサイル耐熱 | 宇宙往還機・再突入体 | 2035〜 |
| **自律無人機 AI** | 兵器スウォーム | 配送・農業・点検 | 2030〜 |
| **DEW レーザー電源** | 対ドローン兵器 | 工業レーザー・医療 | 2040〜 |

全て **SiC または GaN チップ** が核心。全て **Disco を通る**。
    """)
