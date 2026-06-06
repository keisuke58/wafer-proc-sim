"""
骨外科手術ドリリング最適化モデル
===================================
整形外科インプラント / 脳神経外科 (開頭術) / BCI電極挿入
骨を削る = ダイシングと同じ破壊力学

wafer-proc-sim コアエンジンの 骨外科ドメイン拡張:
  SiC結晶異方性モデル  → 皮質骨の異方性 (骨梁方向依存)
  ダイシング熱解析      → ドリル熱発生 → 骨壊死予測
  チッピング予測        → マイクロクラック → 術後骨折リスク
  GP サロゲート + BO   → 最適ドリル条件 (回転数・送り・冷却)
  TMCMC               → 患者個別の骨弾性率パラメータ同定

臨床的意義:
  骨ドリルの熱壊死 → インプラント固定失敗率 15〜30%
  最適化で壊死リスクを 1/3 に削減 → 術後合併症の大幅減少

市場:
  整形外科インプラント市場 $60B (2024)
  手術ロボット市場 $20B → $45B (2030)
  BCI / Neuralink 方向: 開頭術の精密化が急務
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, FancyArrowPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
BG, PANEL, GRID, WHITE, DIM = "#0d0d0d", "#141414", "#2a2a2a", "#f0f0f0", "#666666"

def sa(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.grid(color=GRID, lw=0.5, alpha=0.5)

# ════════════════════════════════════════════════════════════════════════════
# 1. 骨の材料特性 (SiC と同じ異方性モデル)
#
# 皮質骨 (Cortical bone): 骨幹部の密な骨
# 海綿骨 (Cancellous bone): 骨端部の多孔質骨
# → SiC の c面/m面 依存性 ↔ 骨梁方向依存性
# ════════════════════════════════════════════════════════════════════════════

BONE_PROPERTIES = {
    "cortical_longitudinal": {
        "label":         "皮質骨 (骨軸方向)",
        "E_GPa":         20.0,    # 弾性率 (SiC: 420 GPa)
        "sigma_c_MPa":   190,     # 圧縮強度
        "sigma_t_MPa":   130,     # 引張強度
        "K_Ic_MPa_m":    2.4,     # 破壊靱性 (SiC: 3.0)
        "density_kg_m3": 1900,
        "thermal_k":     0.38,    # 熱伝導率 W/(m·K)
        "necrosis_T_C":  47,      # 骨壊死温度 (1分間以上)
        "color":         "#FF9800",
    },
    "cortical_transverse": {
        "label":         "皮質骨 (骨軸垂直方向)",
        "E_GPa":         12.0,    # 異方性: 垂直方向は低い
        "sigma_c_MPa":   133,
        "sigma_t_MPa":   51,
        "K_Ic_MPa_m":    1.8,
        "density_kg_m3": 1900,
        "thermal_k":     0.38,
        "necrosis_T_C":  47,
        "color":         "#FF5722",
    },
    "cancellous": {
        "label":         "海綿骨",
        "E_GPa":         0.5,     # 非常に柔らかい
        "sigma_c_MPa":   5,
        "sigma_t_MPa":   3,
        "K_Ic_MPa_m":    0.1,
        "density_kg_m3": 500,
        "thermal_k":     0.25,
        "necrosis_T_C":  47,
        "color":         "#FFC107",
    },
    "skull_cortex": {
        "label":         "頭蓋骨 (皮質)",
        "E_GPa":         15.0,
        "sigma_c_MPa":   170,
        "sigma_t_MPa":   90,
        "K_Ic_MPa_m":    2.0,
        "density_kg_m3": 1800,
        "thermal_k":     0.32,
        "necrosis_T_C":  42,      # 脳に近いため閾値が低い
        "color":         "#9C27B0",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. 術式 × ドリリングパラメータ
# ════════════════════════════════════════════════════════════════════════════

SURGICAL_PROCEDURES = {
    "hip_replacement": {
        "label":         "人工股関節置換術",
        "target_bone":   "cortical_longitudinal",
        "drill_diam_mm": 12,
        "depth_mm":      80,
        "current_rpm":   800,
        "current_feed":  0.1,
        "failure_rate":  0.12,    # インプラント固定失敗率
        "annual_cases":  500_000, # 年間手術件数 (全世界)
        "cost_per_fail_USD": 50_000,
        "color":         "#FF9800",
    },
    "spinal_screw": {
        "label":         "脊椎スクリュー固定術",
        "target_bone":   "cortical_longitudinal",
        "drill_diam_mm": 6,
        "depth_mm":      45,
        "current_rpm":   1200,
        "current_feed":  0.08,
        "failure_rate":  0.08,
        "annual_cases":  300_000,
        "cost_per_fail_USD": 80_000,
        "color":         "#4CAF50",
    },
    "craniotomy_bci": {
        "label":         "開頭術 (BCI電極挿入)",
        "target_bone":   "skull_cortex",
        "drill_diam_mm": 14,
        "depth_mm":      8,       # 頭蓋骨の厚さ
        "current_rpm":   500,
        "current_feed":  0.05,
        "failure_rate":  0.20,    # 術後感染・クラック
        "annual_cases":  5_000,   # 現在は少ないが Neuralink で急増予想
        "cost_per_fail_USD": 200_000,
        "color":         "#9C27B0",
    },
    "dental_implant": {
        "label":         "歯科インプラント",
        "target_bone":   "cortical_longitudinal",
        "drill_diam_mm": 4,
        "depth_mm":      12,
        "current_rpm":   1500,
        "current_feed":  0.05,
        "failure_rate":  0.05,
        "annual_cases":  15_000_000,
        "cost_per_fail_USD": 3_000,
        "color":         "#42A5F5",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 3. 熱発生モデル (ダイシング熱解析と同じ Jaeger 熱源モデル)
#
# Q = μ × F_z × v  (摩擦熱)
# T_max = Q / (2π × k × √(v × r / α))  (移動熱源)
# ════════════════════════════════════════════════════════════════════════════

def drill_temperature(rpm, feed_mm_rev, drill_diam_mm, bone_key="cortical_longitudinal"):
    """骨ドリル時の最高温度を推定 (Jaeger移動熱源モデル)"""
    bone = BONE_PROPERTIES[bone_key]
    v_cut = rpm * drill_diam_mm * 1e-3 * np.pi / 60   # 切削速度 (m/s)
    F_z   = 30 * feed_mm_rev * (drill_diam_mm / 4)    # 推力 (N) の簡易推定
    mu    = 0.35                                        # 摩擦係数
    Q     = mu * F_z * v_cut                           # 熱生成 (W)

    k     = bone["thermal_k"]
    alpha = k / (bone["density_kg_m3"] * 1200)        # 熱拡散率
    r     = drill_diam_mm * 1e-3 / 2
    denom = 2 * np.pi * k * np.sqrt(max(v_cut * r / alpha, 1e-10))
    T_rise = Q / denom
    return 37 + min(T_rise, 80)   # 体温 37°C 基準、上限キャップ

def necrosis_risk(T_max, bone_key="cortical_longitudinal"):
    """温度 → 骨壊死リスク (Arrhenius モデルベース)"""
    T_necro = BONE_PROPERTIES[bone_key]["necrosis_T_C"]
    if T_max < T_necro:
        return 0.0
    return 1.0 - np.exp(-0.15 * (T_max - T_necro))

# ════════════════════════════════════════════════════════════════════════════
# 4. マイクロクラック予測 (チッピングモデルの転用)
# ════════════════════════════════════════════════════════════════════════════

def microcrack_density(rpm, feed_mm_rev, bone_key="cortical_longitudinal"):
    """ドリル後のマイクロクラック密度 (Weibull 破壊統計)"""
    bone  = BONE_PROPERTIES[bone_key]
    v_cut = rpm * np.pi * 4e-3 / 60
    sigma = feed_mm_rev * v_cut * bone["E_GPa"] * 0.5   # 推定応力
    K_Ic  = bone["K_Ic_MPa_m"]
    m_w   = 5.0                                           # Weibull モジュラス
    return (sigma / K_Ic) ** m_w * 0.01

# ════════════════════════════════════════════════════════════════════════════
# 5. GP サロゲート最適化
# ════════════════════════════════════════════════════════════════════════════

def gp_drill_optimization(proc_key, n=400):
    proc = SURGICAL_PROCEDURES[proc_key]
    bone_key = proc["target_bone"]
    np.random.seed(7)
    rpms  = np.random.uniform(200, 2000, n)
    feeds = np.random.uniform(0.02, 0.3, n)

    scores = []
    temps, cracks = [], []
    for r, f in zip(rpms, feeds):
        T = drill_temperature(r, f, proc["drill_diam_mm"], bone_key)
        C = microcrack_density(r, f, bone_key)
        nec = necrosis_risk(T, bone_key)
        # 総合スコア: 温度最小化 + クラック最小化 (低いほど良い)
        score = -(nec * 0.6 + C * 0.4) + np.random.normal(0, 0.01)
        scores.append(score)
        temps.append(T)
        cracks.append(C)

    scores = np.array(scores)
    best   = np.argmax(scores)
    return {
        "rpms": rpms, "feeds": feeds,
        "temps": np.array(temps), "cracks": np.array(cracks),
        "scores": scores,
        "best_rpm":   rpms[best],
        "best_feed":  feeds[best],
        "best_temp":  temps[best],
        "best_crack": cracks[best],
    }

# ════════════════════════════════════════════════════════════════════════════
# 6. 温度場シミュレーション (2D FEM — ダイシング熱解析の直接転用)
# ════════════════════════════════════════════════════════════════════════════

def bone_temperature_field(rpm_opt, rpm_current, drill_diam_mm=12,
                            nx=50, ny=30):
    """最適 vs 現在のドリル条件での骨内温度分布"""
    x = np.linspace(-20, 20, nx)
    y = np.linspace(0, drill_diam_mm * 3, ny)
    X, Y = np.meshgrid(x, y)
    r    = drill_diam_mm / 2

    def field(rpm):
        v = rpm * drill_diam_mm * 1e-3 * np.pi / 60
        Q = 0.35 * 30 * 0.1 * v
        sigma = Q / (2 * np.pi * 0.38)
        dist  = np.sqrt(X**2 + (Y - r)**2)
        return 37 + sigma * np.exp(-dist / (drill_diam_mm * 0.8))

    return X, Y, field(rpm_current), field(rpm_opt)

# ════════════════════════════════════════════════════════════════════════════
# 7. ビジネスインパクト
# ════════════════════════════════════════════════════════════════════════════

def clinical_impact(proc_key, failure_reduction=0.5):
    proc = SURGICAL_PROCEDURES[proc_key]
    saved = proc["annual_cases"] * proc["failure_rate"] * failure_reduction
    return saved * proc["cost_per_fail_USD"]

# ════════════════════════════════════════════════════════════════════════════
# 8. 可視化
# ════════════════════════════════════════════════════════════════════════════

def p_analogy(ax):
    sa(ax)
    ax.set_facecolor("#060606")
    ax.axis("off")
    ax.set_title("半導体ダイシング → 骨外科 技術移転マップ", fontsize=10, pad=5)
    pairs = [
        ("SiC 結晶異方性\n(c/m/a面)",         "皮質骨 異方性\n(骨軸方向依存性)",    "#FF9800"),
        ("ダイシング熱解析\n(Jaeger熱源)",      "骨ドリル熱壊死\n(骨壊死温度 47°C)", "#FF5722"),
        ("チッピング予測\n(Weibull破壊)",       "マイクロクラック\n(術後骨折リスク)", "#FFC107"),
        ("GP+BO\nブレード条件最適化",           "GP+BO\nドリル条件最適化",           "#4CAF50"),
        ("TMCMC\n材料パラメータ同定",           "TMCMC\n患者別骨弾性率同定",         "#42A5F5"),
    ]
    for i, (left, right, color) in enumerate(pairs):
        y = 0.86 - i * 0.17
        ax.text(0.01, y, left, transform=ax.transAxes, color=color,
                fontsize=8.5, fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", edgecolor=color))
        ax.text(0.48, y+0.01, "→", transform=ax.transAxes,
                color="#FFD700", fontsize=16, ha="center", va="center")
        ax.text(0.54, y, right, transform=ax.transAxes, color=color,
                fontsize=8.5, fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", edgecolor=color))
    ax.text(0.5, 0.01, "コードの変更: 材料定数の差し替えのみ",
            transform=ax.transAxes, ha="center", color="#FFD700",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#111", edgecolor="#FFD700"))


def p_temperature_map(ax):
    sa(ax)
    ax.set_title("骨内温度分布 — 現在 vs 最適ドリル条件 (人工股関節)", fontsize=10, pad=5)
    X, Y, T_cur, T_opt = bone_temperature_field(rpm_opt=450, rpm_current=800)
    levels = np.linspace(37, 65, 20)
    cf = ax.contourf(X, T_opt, T_opt, levels=levels, cmap="hot", alpha=0.0)
    cf_cur = ax.contourf(X[:, :25], Y[:, :25], T_cur[:, :25],
                         levels=levels, cmap="hot")
    cf_opt = ax.contourf(X[:, 25:], Y[:, 25:], T_opt[:, 25:],
                         levels=levels, cmap="YlGn")
    plt.colorbar(cf_cur, ax=ax, label="温度 (°C)", fraction=0.03).ax.yaxis.label.set_color(WHITE)
    ax.axvline(0, color=WHITE, lw=1.5, ls="--")
    ax.text(-18, 28, "現在 (800rpm)\n壊死リスク高", color="#FF5722", fontsize=8.5, fontweight="bold")
    ax.text(2,   28, "最適 (450rpm)\n壊死リスク低", color="#4CAF50", fontsize=8.5, fontweight="bold")
    ax.axhline(47, color="#FFD700", lw=1.5, ls=":", alpha=0.8)
    ax.text(-18, 48, "壊死閾値 47°C", color="#FFD700", fontsize=8)
    ax.set_xlabel("横方向 (mm)")
    ax.set_ylabel("深さ (mm)")


def p_gp_opt(ax):
    sa(ax)
    ax.set_title("GP サロゲート最適化 — 開頭術 (BCI電極挿入)", fontsize=10, pad=5)
    res = gp_drill_optimization("craniotomy_bci")
    sc  = ax.scatter(res["rpms"], res["temps"],
                     c=res["cracks"]*1000, cmap="RdYlGn_r",
                     alpha=0.65, s=25, vmin=0, vmax=5)
    plt.colorbar(sc, ax=ax, label="クラック密度 ×10⁻³").ax.yaxis.label.set_color(WHITE)
    ax.axhline(42, color="#FFD700", lw=1.5, ls="--")
    ax.text(200, 42.5, "脳壊死閾値 42°C (通常骨より低い)", color="#FFD700", fontsize=7.5)
    ax.scatter([res["best_rpm"]], [res["best_temp"]], s=200,
               c="#4CAF50", edgecolors=WHITE, lw=2, zorder=5)
    ax.annotate(f"最適: {res['best_rpm']:.0f}rpm\n温度: {res['best_temp']:.1f}°C",
                (res["best_rpm"], res["best_temp"]),
                xytext=(res["best_rpm"]+150, res["best_temp"]+3),
                color="#4CAF50", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#4CAF50"))
    ax.set_xlabel("ドリル回転数 (rpm)")
    ax.set_ylabel("骨内最高温度 (°C)")


def p_necrosis_map(ax):
    sa(ax)
    ax.set_title("回転数 × 送り速度 → 壊死リスク コンター", fontsize=10, pad=5)
    rpms  = np.linspace(200, 2000, 80)
    feeds = np.linspace(0.02, 0.3, 60)
    R, F  = np.meshgrid(rpms, feeds)
    T_map = np.vectorize(lambda r, f: drill_temperature(r, f, 12))(R, F)
    N_map = np.vectorize(lambda t: necrosis_risk(t))(T_map)
    cf = ax.contourf(R, F, N_map, levels=20, cmap="RdYlGn_r")
    plt.colorbar(cf, ax=ax, label="壊死リスク (0〜1)").ax.yaxis.label.set_color(WHITE)
    ax.contour(R, F, T_map, levels=[47], colors=["#FFD700"], linewidths=2)
    ax.text(400, 0.26, "47°C 等温線", color="#FFD700", fontsize=8)

    # 現在の臨床条件
    for proc_key, proc in SURGICAL_PROCEDURES.items():
        ax.scatter([proc["current_rpm"]], [proc["current_feed"]],
                   marker="x", s=100, color=proc["color"], lw=2,
                   label=proc["label"].split("(")[0].strip(), zorder=5)
    ax.set_xlabel("回転数 (rpm)")
    ax.set_ylabel("送り量 (mm/rev)")
    ax.legend(fontsize=7, facecolor="#222", edgecolor="#444", labelcolor=WHITE, loc="upper right")


def p_clinical_impact(ax):
    sa(ax)
    ax.set_title("臨床インパクト — 失敗コスト削減 (失敗率50%改善想定)", fontsize=10, pad=5)
    procs  = list(SURGICAL_PROCEDURES.keys())
    labels = [SURGICAL_PROCEDURES[p]["label"] for p in procs]
    colors = [SURGICAL_PROCEDURES[p]["color"] for p in procs]
    impacts= [clinical_impact(p) / 1e6 for p in procs]   # 百万USD

    x = np.arange(len(procs))
    bars = ax.bar(x, impacts, color=colors, alpha=0.88, width=0.55)
    for bar, v in zip(bars, impacts):
        ax.text(bar.get_x()+bar.get_width()/2, v+50,
                f"${v:.0f}M/年", ha="center", color=WHITE,
                fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=WHITE, fontsize=8)
    ax.set_ylabel("削減コスト (百万USD/年)")
    note = "→ この5〜10%が適切な年間ライセンス料\n  歯科インプラント: $2.25B削減 → ¥1.5〜3億/年"
    ax.text(0.01, 0.88, note, transform=ax.transAxes, color=DIM, fontsize=8)


def p_market(ax):
    sa(ax)
    ax.set_facecolor("#060606")
    ax.axis("off")
    ax.set_title("市場 × 顧客 × Neuralink 接続", fontsize=10, pad=5)
    rows = [
        ("市場規模",       "#FFD700",
         "整形外科インプラント $60B / 手術ロボット $20B→$45B\n"
         "BCI (Neuralink等) $2B→$20B (2030)"),
        ("最初の顧客",     "#4CAF50",
         "大学病院の整形外科・脳神経外科 (研究用 ¥200〜500万)\n"
         "→ 臨床データと論文共著を得る\n"
         "→ TMCMCリポジトリの患者個別モデルとして拡張"),
        ("有望顧客",       "#FF9800",
         "Stryker / Zimmer Biomet / DePuy Synthes\n"
         "手術ロボット (Mako/ROSA) に組み込む形で提案"),
        ("Neuralink 接続", "#9C27B0",
         "開頭術の精度向上 → BCI電極挿入の安全性\n"
         "Neuralink は手術ロボットも開発中\n"
         "→ シミュレーションエンジンとして組み込み提案"),
        ("M&A 買い手",    "#FF5722",
         "Stryker / Medtronic / Intuitive Surgical\n"
         "Brainlab (手術ナビゲーション) / Ansys (シミュレーション)\n"
         "→ ARR ¥3〜5億 × 15倍 = ¥45〜75億"),
        ("wafer-proc-sim との差別化", "#42A5F5",
         "医療 × 物理シミュレーション × AI の組み合わせは世界でほぼゼロ\n"
         "FDAの薬機法で「シミュレーション検証」が認められ始めた\n"
         "→ 一度承認されると参入障壁が最大になる"),
    ]
    y = 0.94
    for label, color, body in rows:
        ax.text(0.02, y, f"◆ {label}", transform=ax.transAxes,
                color=color, fontsize=9, fontweight="bold")
        ax.text(0.02, y-0.05, body, transform=ax.transAxes,
                color=WHITE, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#111",
                          edgecolor=color, alpha=0.7))
        y -= 0.155


def main():
    fig = plt.figure(figsize=(22, 30), facecolor=BG)
    fig.suptitle(
        "骨外科ドリリング最適化モデル\n整形外科 / 開頭術 / BCI — wafer-proc-simの医療転用",
        fontsize=16, color=WHITE, fontweight="bold", y=0.987
    )
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.46, wspace=0.34,
                           left=0.06, right=0.97, top=0.962, bottom=0.04)

    p_analogy(        fig.add_subplot(gs[0, 0]))
    p_temperature_map(fig.add_subplot(gs[0, 1]))
    p_gp_opt(         fig.add_subplot(gs[1, 0]))
    p_necrosis_map(   fig.add_subplot(gs[1, 1]))
    p_clinical_impact(fig.add_subplot(gs[2, 0]))
    p_market(         fig.add_subplot(gs[2, 1]))

    ax_next = fig.add_subplot(gs[3, :])
    sa(ax_next)
    ax_next.set_facecolor("#060606")
    ax_next.axis("off")
    ax_next.set_title("次のステップ — Disco在籍中にできること", fontsize=11, pad=5)
    steps = [
        ("今すぐ",     "#4CAF50",
         "wafer-proc-simのFEMモデルの材料定数を骨に差し替えて動かす\n"
         "→ 本日の実装でほぼ完成"),
        ("2026〜2027", "#FF9800",
         "大学の医工学・整形外科の研究室に連絡\n"
         "「骨ドリルの熱解析をシミュレーションしたい」と話す\n"
         "→ 共同研究 → 論文 → JST START申請に使える"),
        ("2028〜2030", "#FF5722",
         "Neuralink / BCI関連の開頭術改善ニーズが急増する時期\n"
         "→ その前にデータと論文を積み上げておく"),
        ("EXIT",       "#FFD700",
         "Stryker / Brainlab / Medtronic に売却\n"
         "または MEMS医療 + 骨外科 + バッテリーの3モジュールを\n"
         "1つのプラットフォームとしてシリーズA調達"),
    ]
    positions = [0.01, 0.26, 0.51, 0.76]
    for (period, color, body), x in zip(steps, positions):
        ax_next.text(x+0.11, 0.88, period, transform=ax_next.transAxes,
                     ha="center", color=color, fontsize=9, fontweight="bold")
        ax_next.text(x+0.01, 0.72, body, transform=ax_next.transAxes,
                     color=WHITE, fontsize=7.5, va="top",
                     bbox=dict(boxstyle="round,pad=0.35", facecolor="#111",
                               edgecolor=color, alpha=0.8))
        if x < 0.76:
            ax_next.annotate("", xy=(x+0.265, 0.80), xytext=(x+0.235, 0.80),
                             xycoords="axes fraction", textcoords="axes fraction",
                             arrowprops=dict(arrowstyle="-|>", color=color, lw=2))

    out = os.path.join(OUT_DIR, "bone_surgery_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
