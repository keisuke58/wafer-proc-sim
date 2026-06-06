"""
「資本側」とはどんな人か — 構造・経路・規模の完全可視化
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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
# 資本側への5つの入口
# ════════════════════════════════════════════════════════════════════════════

ENTRY_PATHS = {
    "inheritance": {
        "label":       "相続\n(生まれた家)",
        "color":       "#B71C1C",
        "prob":        0.01,    # 全人口の1%
        "time_years":  0,
        "initial_M":   200,
        "example":     "ロスチャイルド家\n森ビル創業家\n伊藤忠一族",
        "merit":       "時間ゼロ、リスクゼロ",
        "demerit":     "選べない",
        "global_pct":  0.1,
    },
    "founder": {
        "label":       "創業\n(会社を作る)",
        "color":       "#E65100",
        "prob":        0.001,   # 0.1% — 大成功する確率
        "time_years":  10,
        "initial_M":   0,
        "example":     "Zuckerberg / Gates\nホリエモン / 孫正義\n前澤友作",
        "merit":       "青天井、社会的インパクト",
        "demerit":     "成功確率<1%、10年無収入リスク",
        "global_pct":  0.01,
    },
    "early_employee": {
        "label":       "アーリー社員\n(ストックオプション)",
        "color":       "#F57F17",
        "prob":        0.005,
        "time_years":  5,
        "initial_M":   0,
        "example":     "Google初期社員\nNVIDIA 2000年代社員\nメルカリ上場前社員",
        "merit":       "給与+株で一気に資本側へ",
        "demerit":     "会社を見極める眼力が必要",
        "global_pct":  0.05,
    },
    "finance": {
        "label":       "金融\n(PE/HF/VC)",
        "color":       "#1A237E",
        "prob":        0.002,
        "time_years":  15,
        "initial_M":   0,
        "example":     "Goldman Sachs MD\nブラックストーン\nソフトバンクビジョン",
        "merit":       "他人の資本を動かして報酬を得る",
        "demerit":     "参入障壁が高い、長時間労働",
        "global_pct":  0.02,
    },
    "accumulation": {
        "label":       "積み上げ\n(長期投資×高収入)",
        "color":       "#1B5E20",
        "prob":        0.05,
        "time_years":  30,
        "initial_M":   0,
        "example":     "Disco→外資+投資率35%\n医師+不動産\nエンジニア+インデックス",
        "merit":       "誰でも入れる唯一の経路",
        "demerit":     "30年かかる、途中で諦めやすい",
        "global_pct":  1.0,
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 資本側の「自走」シミュレーション
# 資産が一定額を超えると労働収入なしで生活できる
# ════════════════════════════════════════════════════════════════════════════

FIRE_THRESHOLD_M = 100   # ¥1億 = 年¥350-500万の不労所得

SIM_Y = np.arange(0, 51)

def capital_snowball(initial_M, return_rate, annual_add_M):
    w = initial_M
    history = [w]
    fire_year = None
    for i, _ in enumerate(SIM_Y[1:], 1):
        w = w * (1 + return_rate) + annual_add_M
        history.append(w)
        if fire_year is None and w >= FIRE_THRESHOLD_M:
            fire_year = i
    return np.array(history), fire_year

SNOWBALL_CASES = {
    "start0_low":   (0,   0.05, 1.2,  "#546e7a", "初期¥0 / 年積立¥120万 / 利率5%\n(日本平均サラリーマン)"),
    "start0_disco": (0,   0.07, 2.5,  "#42A5F5", "初期¥0 / 年積立¥250万 / 利率7%\n(Disco 投資25%)"),
    "start0_nvidia":(0,   0.09, 5.0,  "#76FF03", "初期¥0 / 年積立¥500万 / 利率9%\n(NVIDIA積極投資)"),
    "start50_mid":  (50,  0.07, 2.5,  "#FF9800", "初期¥5000万 / 年積立¥250万\n(Disco→外資+相続少し)"),
    "start200_high":(200, 0.10, 0,    "#FF1744", "初期¥2億 / 積立なし / 利率10%\n(創業・相続・PE — 純粋資本側)"),
}

# ════════════════════════════════════════════════════════════════════════════
# 職業別 資本化されやすさ
# ════════════════════════════════════════════════════════════════════════════

JOBS = {
    # (ラベル, 年収M, 資本化スコア0-10, 資本側到達年数, color)
    "医師":         (15,  5, 20, "#FF9800"),
    "弁護士":       (12,  4, 25, "#FF7043"),
    "外資金融":     (20,  8, 12, "#1565C0"),
    "外資IT(NVIDIA)":(18, 9, 10, "#76FF03"),
    "Disco":        (9,   6, 28, "#42A5F5"),
    "日本平均":     (4.6, 2, 50, "#546e7a"),
    "起業家(成功)": (0,  10,  5, "#FF1744"),
    "YouTuber上位": (30, 7,  8, "#FF5722"),
    "PE/VC":        (25,  9,  8, "#9C27B0"),
    "不動産投資家": (8,   8, 15, "#795548"),
}

# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def p_entry_paths(ax):
    sa(ax)
    ax.set_facecolor("#060606")
    ax.axis("off")
    ax.set_title("資本側への5つの入口", fontsize=11, pad=5)

    positions = [(0.5, 0.88), (0.08, 0.55), (0.30, 0.55), (0.70, 0.55), (0.92, 0.55)]
    for i, (key, path) in enumerate(ENTRY_PATHS.items()):
        x, y = positions[i]
        ax.text(x, y, path["label"], transform=ax.transAxes,
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=WHITE,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=path["color"],
                          edgecolor=WHITE, alpha=0.85, linewidth=1.2))
        # 確率
        ax.text(x, y-0.09, f"到達確率 {path['prob']*100:.1f}%",
                transform=ax.transAxes, ha="center", color=path["color"], fontsize=7.5)
        # 例
        ax.text(x, y-0.17, path["example"],
                transform=ax.transAxes, ha="center", color=DIM, fontsize=6.5)

    # 矢印 (中央→各パス)
    for x_dst in [0.08, 0.30, 0.70, 0.92]:
        ax.annotate("", xy=(x_dst, 0.63), xytext=(0.5, 0.80),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))

    # 共通ゴール
    ax.text(0.5, 0.18,
            "「資本側」= 資産が年間生活費の25倍以上\n"
            "→ 不労所得だけで生きられる状態\n"
            "日本基準: ¥1億〜 / 世界基準: ¥7.5億〜 (Top0.1%)",
            transform=ax.transAxes, ha="center", va="center",
            color=WHITE, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#1a1a1a",
                      edgecolor="#FFD700", alpha=0.9))

    # 共通ゴールへの矢印
    for x_src in [0.08, 0.30, 0.50, 0.70, 0.92]:
        ax.annotate("", xy=(0.5, 0.27), xytext=(x_src, 0.47),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.0))


def p_snowball(ax):
    sa(ax)
    ax.set_title("資産スノーボール — ¥1億 (FIRE) 到達シミュレーション", fontsize=10, pad=5)

    ax.axhline(FIRE_THRESHOLD_M, color="#FFD700", lw=2, ls="--", alpha=0.8)
    ax.text(1, FIRE_THRESHOLD_M*1.04, "FIRE ライン ¥1億 (年収入¥350-500万)", color="#FFD700", fontsize=8)

    for key, (init, ret, add, color, label) in SNOWBALL_CASES.items():
        w, fire_yr = capital_snowball(init, ret, add)
        short_label = label.split("\n")[0]
        ax.plot(SIM_Y, w, color=color, lw=2.2, label=short_label)
        if fire_yr and fire_yr <= 50:
            ax.axvline(fire_yr, color=color, lw=0.8, ls=":", alpha=0.5)
            ax.text(fire_yr+0.3, w[fire_yr]*0.85,
                    f"{fire_yr}年後", color=color, fontsize=7)

    ax.set_xlabel("経過年数")
    ax.set_ylabel("資産 (百万円)")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 50000)
    ax.legend(fontsize=7.5, facecolor="#222", edgecolor="#444", labelcolor=WHITE,
              loc="upper left")
    note = "純粋資本側(赤): 初日から指数増殖。労働は不要になる"
    ax.text(0.01, 0.01, note, transform=ax.transAxes, color=DIM, fontsize=7.5)


def p_jobs(ax):
    sa(ax)
    ax.set_title("職業別 「資本側」到達スコア vs 年収", fontsize=10, pad=5)

    for label, (income, cap_score, years, color) in JOBS.items():
        size = max(30, (11 - years/5) * 40)
        ax.scatter(income, cap_score, s=size, c=color, alpha=0.88,
                   edgecolors=WHITE, lw=0.8, zorder=3)
        offset_x = 0.3
        offset_y = 0.2
        if label == "日本平均":      offset_y = -0.5
        if label == "不動産投資家":  offset_x = -2.5
        if label == "起業家(成功)":  offset_x = 0.3; offset_y = -0.5
        ax.annotate(f"{label}\n({years}年で到達)",
                    (income, cap_score),
                    xytext=(income+offset_x, cap_score+offset_y),
                    color=WHITE, fontsize=7,
                    arrowprops=dict(arrowstyle="-", color="#444", lw=0.8))

    ax.set_xlabel("年収 (百万円)")
    ax.set_ylabel("資本化スコア (0=労働依存, 10=完全資本側)")
    ax.set_xlim(-2, 35)
    ax.set_ylim(0, 11)
    ax.axhline(7, color="#FFD700", lw=1, ls="--", alpha=0.6)
    ax.text(0.5, 7.1, "← この上が「資本側」の入口", color="#FFD700", fontsize=8)
    note = "円のサイズ = 到達スピード (大=早い)"
    ax.text(0.01, 0.01, note, transform=ax.transAxes, color=DIM, fontsize=7.5)


def p_what_they_do(ax):
    sa(ax)
    ax.set_facecolor("#060606")
    ax.axis("off")
    ax.set_title("資本側の人は「何をしているか」", fontsize=11, pad=5)

    blocks = [
        # (x, y, タイトル, 内容, color)
        (0.02, 0.75, "朝起きると", "#FFD700",
         "昨夜の株価上昇で\n¥300万増えている\n(寝ているだけで)"),
        (0.35, 0.75, "会議の中身", "#4CAF50",
         "「このファンドに\n¥50億入れるか？」\n労働の話は出ない"),
        (0.68, 0.75, "税金対策", "#FF9800",
         "法人・信託・タックス\nヘイブンで実効税率\n<10%に抑える"),
        (0.02, 0.35, "リスク管理", "#2196F3",
         "資産を20以上の\nアセットに分散\n1つが消えても平気"),
        (0.35, 0.35, "次世代戦略", "#9C27B0",
         "子供に信託で移転\n相続税を最小化\n資本の永続化"),
        (0.68, 0.35, "労働者との違い", "#FF5722",
         "「解雇」も「倒産」も\n関係ない。時間を\n自分で支配している"),
    ]

    for x, y, title, color, body in blocks:
        ax.text(x+0.12, y+0.10, title, transform=ax.transAxes,
                ha="center", color=color, fontsize=9, fontweight="bold")
        ax.text(x+0.12, y, body, transform=ax.transAxes,
                ha="center", va="top", color=WHITE, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#181818",
                          edgecolor=color, alpha=0.8))

    ax.text(0.5, 0.05,
            "共通点: 「時間」を売っていない。資本・知識・ネットワークが働いている。",
            transform=ax.transAxes, ha="center",
            color="#FFD700", fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#111",
                      edgecolor="#FFD700", alpha=0.9))


def p_how_to_enter(ax):
    sa(ax)
    ax.set_facecolor("#060606")
    ax.axis("off")
    ax.set_title("西岡さんが資本側に近づく現実的ルート", fontsize=11, pad=5)

    steps = [
        ("Step 1\nDiscoで基礎", "#42A5F5",
         "半導体の専門知識 × 高給 (¥6-9M)\n投資元本を積み上げる最初の5年\n→ 毎月¥15-20万を全世界株へ"),
        ("Step 2\n外資転職", "#4CAF50",
         "Lam/AMAT→NVIDIAで年収 ¥15-25M\nRSU(株式報酬)で給与以外の資本収入\n→ ここで初めて「資本」が育ち始める"),
        ("Step 3\n複利の加速", "#FF9800",
         "資産¥5000万超えで利息>生活費の一部\n投資率を上げても生活水準は落ちない\n→ 雪だるまが転がり始める"),
        ("Step 4\nFIRE or 継続", "#FF5722",
         "¥1-2億で選択肢が生まれる\n「働く必要」→「働きたいから働く」へ\n→ ここが「資本側」の入口"),
    ]

    for i, (title, color, body) in enumerate(steps):
        x = 0.02 + i * 0.245
        ax.text(x+0.10, 0.80, title, transform=ax.transAxes,
                ha="center", color=color, fontsize=9, fontweight="bold")
        ax.text(x+0.10, 0.68, body, transform=ax.transAxes,
                ha="center", va="top", color=WHITE, fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#151515",
                          edgecolor=color, alpha=0.8))
        if i < 3:
            ax.annotate("", xy=(x+0.245+0.01, 0.80),
                        xytext=(x+0.21, 0.80),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2))

    ax.text(0.5, 0.18,
            "結論: 「資本側」は生まれ持った人だけではない\n"
            "しかし 労働だけでは絶対に到達できない\n"
            "→ 高収入 × 高投資率 × 長期 の3つが必要条件",
            transform=ax.transAxes, ha="center", va="center",
            color=WHITE, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#1a1a1a",
                      edgecolor="#FFD700", alpha=0.9))


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(22, 28), facecolor=BG)
    fig.suptitle(
        "「資本側」とはどんな人か — 構造・入口・到達経路",
        fontsize=18, color=WHITE, fontweight="bold", y=0.987
    )

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.46, wspace=0.32,
        left=0.05, right=0.97,
        top=0.962, bottom=0.04,
    )

    p_entry_paths(  fig.add_subplot(gs[0, 0]))
    p_snowball(     fig.add_subplot(gs[0, 1]))
    p_jobs(         fig.add_subplot(gs[1, 0]))
    p_what_they_do( fig.add_subplot(gs[1, 1]))
    p_how_to_enter( fig.add_subplot(gs[2, :]))

    out = os.path.join(OUT_DIR, "capital_class_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
