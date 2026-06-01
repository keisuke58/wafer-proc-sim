"""
宇宙安全保障 × 半導体 — ASAT / ケスラー / Disco 地政学
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

st.set_page_config(page_title="Space Security", layout="wide")
st.title("🛡️ 宇宙安全保障 × 半導体")
st.caption(
    "ASAT攻撃 · ケスラー症候群 · 軍事衛星チップ · Disco の地政学的リスク · 日本の戦略的ポジション"
)

from fem.space_security_model import (
    kessler_cascade_model, military_chip_demand,
    disco_strategic_shutdown_impact,
    ORBITAL_ENVIRONMENT, MILITARY_SATELLITE_PROGRAMS,
    ASAT_THREAT_MATRIX, JAPAN_SPACE_SECURITY, DISCO_GEOPOLITICS,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "☄️ ケスラー症候群", "🛰️ 軍事衛星チップ",
    "🏭 Disco 地政学リスク", "🇯🇵 日本の戦略"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — ケスラー症候群
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("ケスラー症候群 — ASAT攻撃が引き起こすデブリ連鎖の物理")

    st.markdown("""
**ケスラー症候群とは?**
衛星破壊 → デブリ → 別の衛星に衝突 → さらにデブリ → 連鎖
→ 軌道が永久に使えなくなる

1978年に予言。2007年中国の実験で現実のリスクに。
    """)

    orbit_sel = st.selectbox("軌道", list(ORBITAL_ENVIRONMENT.keys()),
                              format_func=lambda k: ORBITAL_ENVIRONMENT[k]["label"])
    asat_n    = st.slider("ASAT攻撃 発数", 0, 50, 0)

    r = kessler_cascade_model(orbit_sel, asat_n)

    k = st.columns(5)
    k[0].metric("現在のデブリ数",   f"{ORBITAL_ENVIRONMENT[orbit_sel]['n_debris_10cm']:,}")
    k[1].metric("ASAT追加デブリ",   f"{asat_n * 2000:,}")
    k[2].metric("臨界密度比",       f"{r['critical_ratio']:.2f}")
    k[3].metric("連鎖まで",         f"{r['cascade_years']} 年")
    k[4].metric("リスク判定",       r["kessler_risk"])

    c1, c2 = st.columns(2)
    with c1:
        # ASAT発数 vs 臨界比
        asat_arr = list(range(0, 51, 5))
        ratios   = [kessler_cascade_model(orbit_sel, a)["critical_ratio"] for a in asat_arr]
        cascades = [kessler_cascade_model(orbit_sel, a)["cascade_years"]   for a in asat_arr]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=asat_arr, y=ratios, name="臨界密度比",
                                  line=dict(color="#e45756", width=3),
                                  fill="tozeroy", fillcolor="rgba(228,87,86,0.15)"))
        fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                      annotation_text="臨界点 (連鎖開始)")
        fig.add_hline(y=0.7, line_dash="dot", line_color="orange",
                      annotation_text="警戒域")
        fig.add_vline(x=asat_n, line_dash="dot", line_color="white",
                      annotation_text=f"現在: {asat_n}発")
        fig.update_layout(title="ASAT攻撃発数 → ケスラー臨界比",
                          xaxis_title="ASAT攻撃 発数", yaxis_title="臨界密度比",
                          plot_bgcolor="#111", paper_bgcolor="#111",
                          font_color="white", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # 実際の ASAT 事件
        events = [
            ("2007 中国\nFengyun-1C", 2800, "#e41a1c"),
            ("2019 インド\nMicrosat-R", 400, "#ff7f00"),
            ("2021 ロシア\nKosmos-1408", 1500, "#4daf4a"),
            ("2008 米国\nUSA-193", 0, "#377eb8"),
        ]
        ev_names  = [e[0] for e in events]
        ev_debris = [e[1] for e in events]
        ev_colors = [e[2] for e in events]

        fig2 = go.Figure(go.Bar(
            x=ev_names, y=ev_debris,
            marker_color=ev_colors,
            text=[f"{d:,}個" for d in ev_debris], textposition="outside",
        ))
        fig2.update_layout(title="実際の ASAT 実験と生成デブリ数",
                           yaxis_title="デブリ数 (10cm以上)",
                           plot_bgcolor="#111", paper_bgcolor="#111",
                           font_color="white", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    if r["critical_ratio"] >= 1.0:
        st.error(f"""
🔴 **ケスラー連鎖開始**: {asat_n}発の攻撃で臨界密度を超えた。

推定GDP損失: **${r['gdp_loss_yr_T$']:.1f}T/年** (GPS・通信衛星停止による)
→ 世界GDPの約10%が消える。これは核戦争に近い経済的打撃。
        """)
    else:
        st.info(f"現在の臨界比 {r['critical_ratio']:.2f} — まだ安全圏内。"
                f"ただし {asat_n}発の攻撃でデブリが急増。")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — 軍事衛星チップ
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🛰️ 軍事衛星 × RAD-hard 半導体需要")

    programs = list(MILITARY_SATELLITE_PROGRAMS.keys())
    results  = [military_chip_demand(p) for p in programs]

    k = st.columns(4)
    for i, r in enumerate(results):
        k[i].metric(r["label"][:15], f"SiC {r['total_sic']:,}個",
                    f"${r['chip_value_M$']:.0f}M")

    c1, c2 = st.columns(2)
    with c1:
        labels_p  = [r["label"][:20] for r in results]
        sic_vals  = [r["total_sic"]     for r in results]
        gan_vals  = [r["total_gan"]     for r in results]
        colors_p  = [MILITARY_SATELLITE_PROGRAMS[p]["color"] for p in programs]

        fig = go.Figure()
        fig.add_bar(name="RAD-hard SiC", x=labels_p, y=sic_vals,
                    marker_color="#2166ac")
        fig.add_bar(name="GaN-on-SiC RF", x=labels_p, y=gan_vals,
                    marker_color="#d73027")
        fig.update_layout(barmode="stack", title="軍事衛星プログラム別 半導体需要",
                          yaxis_title="チップ数",
                          plot_bgcolor="#111", paper_bgcolor="#111",
                          font_color="white", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # ASAT 脅威マトリクス
        st.markdown("#### ASAT 脅威 × 半導体ターゲット")
        rows = []
        for key, threat in ASAT_THREAT_MATRIX.items():
            rows.append({
                "脅威": key,
                "種類": threat["type"],
                "高度 [km]": threat["target_alt_km"],
                "実証済": "✓" if threat["demonstrated"] else "開発中",
                "デブリ": f"{threat['debris_generated']:,}個",
                "標的チップ": threat["semiconductor_target"],
            })
        st.dataframe(pd.DataFrame(rows).set_index("脅威"),
                     use_container_width=True, height=280)

    st.markdown("#### RAD-hard 半導体: なぜ SiC が軍事に選ばれるか")
    st.code("""
通常の Si チップ  → TID 10krad で誤動作
SiC チップ       → TID 100krad 以上耐性 (民間衛星の3〜10倍)

宇宙放射線の内訳:
  陽子線 (太陽フレア):  LEO で 1-10krad/年
  電子線 (Van Allen 帯): GEO で 10-100krad/年
  宇宙線:               0.01krad/年 (少ないが高エネルギー)

Starshield の SiC チップ (TID 100krad):
  = LEO で 10〜100年分の放射線に耐える
  = 軍事任務中に交換不要
  = Disco の超精密研削なしには作れない
""", language="text")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Disco 地政学リスク
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🏭 Disco — 「半導体の関所」の地政学的リスク")

    d = disco_strategic_shutdown_impact()

    k = st.columns(4)
    k[0].metric("Disco 世界シェア", "70%", "ダイシング装置")
    k[1].metric("SiC 影響率",      f"{d['sic_affected_pct']}%",
                "Disco なしでSiC切れない")
    k[2].metric("GDP損失/日",       f"${d['gdp_impact_B_day']:.2f}B",
                "輸出停止時")
    k[3].metric("年間損失",          f"${d['semiconductor_gdp_loss_yr_T$']:.1f}T",
                "半導体版TSMC")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### シナリオ別 インパクト")
        scenarios = [
            ("通常操業", 0, "#4CAF50"),
            ("輸出制限 (中国向け)", 20, "#FF9800"),
            ("工場火災 (2週間)", 60, "#f5a623"),
            ("全面操業停止 (3ヶ月)", 100, "#e45756"),
        ]
        s_labels = [s[0] for s in scenarios]
        s_impacts = [s[1] for s in scenarios]
        s_colors  = [s[2] for s in scenarios]

        fig = go.Figure(go.Bar(
            x=s_labels, y=s_impacts,
            marker_color=s_colors,
            text=[f"{v}%" for v in s_impacts], textposition="outside",
        ))
        fig.update_layout(title="Disco 操業停止シナリオ別 半導体生産への影響",
                          yaxis_title="半導体生産への影響 [%]",
                          plot_bgcolor="#111", paper_bgcolor="#111",
                          font_color="white", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 代替手段の現実")
        st.markdown("""
| 代替案 | 実現性 | 理由 |
|---|---|---|
| Accretech (日本) | △ 部分的 | Si のみ、SiC は不可 |
| 中国メーカー | ✗ 不可 | 精度 10× 劣る |
| 自社開発 | ✗ 10年以上 | ノウハウ構築が困難 |
| Disco 在庫備蓄 | △ 8週分 | 長期停止には対応不可 |

**結論: Disco は実質的に代替不可能**

これはTSMCリスクと同じ構造:
- TSMC → 先端ロジックチップ
- Disco → チップのダイシング (全種類)
        """)

    st.error("""
⚠️ **最大のリスク: Disco 本社は東京都品川区**

東京に大規模自然災害・有事が発生した場合:
→ 世界の半導体サプライチェーンが数週間で停止
→ EV生産、AI DC、軍事通信が全て影響
→ 代替供給先がない

日本政府の対策: 2023年「経済安全保障推進法」でDisco を重要経済基盤に指定
    """)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — 日本の戦略
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🇯🇵 日本の宇宙安全保障 × 半導体戦略")

    st.markdown("""
**日本の独自ポジション:**
攻撃的な宇宙軍事力は持てないが、
**半導体という経済的兵器を持っている**
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 日本の「半導体カード」")
        for k_jp, v_jp in JAPAN_SPACE_SECURITY["semiconductor_angle"].items():
            st.markdown(f"- **{k_jp}**: {v_jp}")

        st.divider()
        st.markdown("#### 2023年 宇宙安全保障構想 の要点")
        for point in JAPAN_SPACE_SECURITY["2023_strategy"]["key_points"]:
            st.markdown(f"- {point}")

    with c2:
        st.markdown("#### 日本の脆弱性")
        for k_v, v_v in JAPAN_SPACE_SECURITY["vulnerabilities"].items():
            st.markdown(f"- **{k_v}**: {v_v}")

        st.divider()
        st.markdown("#### Starlink マスクリスク")
        st.markdown("""
ウクライナ戦争でマスクが一時的に Starlink を制限したように:
- 民間インフラへの依存 = 地政学的脆弱性
- 自衛隊が Starlink に依存 → マスクが「友達」でなくなったら?
- 対策: 国産衛星 + みちびき強化 + EU IRIS² との連携
        """)

    # 日本の半導体 × 宇宙安全保障 接続図
    st.markdown("#### 日本の半導体が宇宙安全保障に与える影響")
    st.code("""
日本の半導体輸出規制 (経産省)
    ↓
  Disco 装置の対中輸出制限
    → 中国の軍事衛星 (国網) のSiCチップ製造が困難に
    → PLA の「宇宙軍」が弱体化

  SiC ウェーハ (信越化学・Sumco) の輸出制限
    → 中国のパワー半導体が5年遅延
    → J-20 ステルス戦闘機の電子戦装置に影響

日本の戦略的価値:
  「攻撃力はないが、半導体サプライチェーンの
   キーストーンを握る国」= 現代の重要な安全保障上の地位
    """, language="text")

    st.success("""
### 結論: 日本の宇宙安全保障における半導体の役割

**守り (防衛):**
- JAXA × 防衛省の連携で RAD-hard SiC 衛星を整備
- Disco 装置の国内在庫・施設分散でリスク低減
- みちびき GPS を SiC クロック回路で強化

**攻め (抑止):**
- Disco 輸出制限 = 中国の宇宙軍を間接的に弱体化させる経済兵器
- SiC/GaN 輸出管理 = 先端軍事衛星の製造阻害
- TSMC より目立たないが、同等に重要なチョークポイント

**2129年に向けて:**
- 月・火星の宇宙安全保障も「どの国がチップを制するか」で決まる
- Disco 技術を持つ国が宇宙覇権に最も近い
    """)
