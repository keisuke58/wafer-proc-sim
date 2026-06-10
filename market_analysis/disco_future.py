"""
Disco 魔改造 #8 DNA物理切断 + #9 宇宙資源採掘
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
BG,PANEL,WHITE,DIM="#0d0d0d","#141414","#f0f0f0","#666666"
def sa(ax):
    ax.set_facecolor(PANEL); ax.tick_params(colors=WHITE,labelsize=8)
    ax.xaxis.label.set_color(WHITE); ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.grid(color="#2a2a2a",lw=0.5,alpha=0.5)

# ══════════════════════════════════════════
# #8 DNA物理的切断
# ダイシングの切込み深さ nm スケール → DNA の直径 2nm に対応できる可能性
DNA_SPECS = {
    "diameter_nm":    2.0,    # DNA二重鎖の直径
    "bp_per_nm":      3.0,    # 1nmあたりの塩基対数
    "crispr_precision_bp": 1, # CRISPRの精度 (1bp = 0.34nm)
    "disco_blade_kerf_nm": 50,# 現在の最小カーフ幅
    "laser_focus_nm": 200,    # レーザー最小焦点径
    "target_nm":      5,      # DNA切断に必要な精度
}
CRISPR_MARKET = {
    "years": [2024,2025,2026,2028,2030,2032,2035],
    "B_USD": [2.5, 3.2, 4.1, 6.8, 11, 18, 35],
}
PHYSICAL_CUTTING_ADVANTAGE = {
    "CRISPR化学":      {"off_target_pct": 5,  "speed_bp_s": 10,  "cost_per_cut": 1000, "color":"#FF5252"},
    "光ピンセット":    {"off_target_pct": 2,  "speed_bp_s": 1,   "cost_per_cut": 5000, "color":"#FF9800"},
    "AFMカンチレバー": {"off_target_pct": 1,  "speed_bp_s": 0.1, "cost_per_cut":10000, "color":"#FFC107"},
    "Disco物理切断(理想)":{"off_target_pct":0.1,"speed_bp_s":100, "cost_per_cut": 100, "color":"#B2FF59"},
}

# #9 宇宙資源採掘
MOON_MINERALS = {
    "Helium-3":     {"abundance_ppt": 10,  "value_per_kg_M$": 3,   "disco_role": "採掘岩石切断", "color":"#EA80FC"},
    "Titanium":     {"abundance_pct": 5,   "value_per_kg$": 50,    "disco_role": "岩石コア採取", "color":"#CE93D8"},
    "REE":          {"abundance_ppm": 200, "value_per_kg$": 2000,  "disco_role": "精密抽出切断", "color":"#B39DDB"},
    "Iron/Aluminum":{"abundance_pct": 20,  "value_per_kg$": 1,     "disco_role": "構造材料採掘", "color":"#9FA8DA"},
    "Water_ice":    {"abundance_pct": 8,   "value_per_kg$": 500,   "disco_role": "氷コア切出し", "color":"#90CAF9"},
}
SPACE_MARKET = {
    "years": [2025,2027,2030,2033,2035,2040],
    "lunar_mining_B$": [0.1, 0.5, 2, 8, 20, 100],
    "disco_cut_B$":    [0,   0.05,0.3,1.5,4,  20],
}

# ══════════════════════════════════════════
def p_precision_compare(ax):
    sa(ax)
    ax.set_title("DNA切断技術の精度・速度比較\nDiscoの物理切断が到達できるか", fontsize=9, pad=5)
    methods = list(PHYSICAL_CUTTING_ADVANTAGE.keys())
    off_t  = [PHYSICAL_CUTTING_ADVANTAGE[m]["off_target_pct"] for m in methods]
    speeds = [PHYSICAL_CUTTING_ADVANTAGE[m]["speed_bp_s"] for m in methods]
    costs  = [PHYSICAL_CUTTING_ADVANTAGE[m]["cost_per_cut"] for m in methods]
    colors = [PHYSICAL_CUTTING_ADVANTAGE[m]["color"] for m in methods]
    sc = ax.scatter(off_t, speeds, s=[c/20 for c in costs],
                    c=colors, alpha=0.85, edgecolors=WHITE, lw=1.2)
    for m, ot, sp in zip(methods, off_t, speeds):
        ax.annotate(m, (ot,sp), xytext=(ot+0.1, sp*1.5),
                    color=PHYSICAL_CUTTING_ADVANTAGE[m]["color"], fontsize=8)
    ax.set_xlabel("オフターゲット率 (%) ← 低いほど良い")
    ax.set_ylabel("切断速度 (bp/s) ↑ 高いほど良い")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.text(0.02,0.02,"円のサイズ = コスト (大=高い)",
            transform=ax.transAxes, color=DIM, fontsize=7.5)

def p_scale_gap(ax):
    sa(ax)
    ax.set_title("スケールギャップ — 現在のDiscoとDNA切断の距離", fontsize=9, pad=5)
    scales = {
        "SiCウェーハ\nカーフ幅": 30e-6,
        "MEMS構造": 1e-6,
        "最先端EUV\nパターン": 13e-9,
        "DNA直径": 2e-9,
        "目標切断\n精度": 0.5e-9,
    }
    labels = list(scales.keys())
    vals   = list(scales.values())
    colors_s = ["#546e7a","#FF9800","#FF5252","#B2FF59","#69F0AE"]
    y = np.arange(len(labels))
    bars = ax.barh(y, [np.log10(v)+12 for v in vals], color=colors_s, alpha=0.85, height=0.55)
    ax.set_yticks(y); ax.set_yticklabels(labels, color=WHITE, fontsize=8)
    for bar, v, label in zip(bars, vals, labels):
        ax.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                f"{v*1e9:.1f} nm", color=WHITE, va="center", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("スケール (log, 小←大)")
    ax.axvline(np.log10(30e-6)+12, color="#FF9800", lw=2, ls="--")
    ax.text(np.log10(30e-6)+12.05, 4.5, "現在のDiscoの限界\n→ここから100倍の\n精度向上が必要",
            color="#FF9800", fontsize=7.5)

def p_crispr_market(ax):
    sa(ax)
    ax.set_title("CRISPR/ゲノム編集市場 + Disco物理切断の可能性", fontsize=9, pad=5)
    ax.fill_between(CRISPR_MARKET["years"], CRISPR_MARKET["B_USD"], alpha=0.2, color="#B2FF59")
    ax.plot(CRISPR_MARKET["years"], CRISPR_MARKET["B_USD"], "o-", color="#B2FF59", lw=2.5, ms=6)
    ax.set_ylabel("市場規模 ($B)")
    ax.set_ylim(0, 40)
    ax.text(2026, 30, "2035年 $35B\n(CAGR 26%)", color="#B2FF59", fontsize=9)
    ax.text(0.02,0.08,
            "現実的な参入タイムライン:\n2030年代にナノスケール\nアクチュエータと組み合わせ",
            transform=ax.transAxes, color=DIM, fontsize=8,
            bbox=dict(boxstyle="round",facecolor="#111",edgecolor="#B2FF59",alpha=0.7))

def p_moon_minerals(ax):
    sa(ax)
    ax.set_title("月面資源マップ — Discoが関与できる採掘工程", fontsize=9, pad=5)
    minerals = list(MOON_MINERALS.keys())
    colors_m = [MOON_MINERALS[m]["color"] for m in minerals]
    roles    = [MOON_MINERALS[m]["disco_role"] for m in minerals]
    x = np.arange(len(minerals))
    # 経済価値インデックス (定性的)
    value_idx = [90, 60, 75, 30, 80]
    bars = ax.bar(x, value_idx, color=colors_m, alpha=0.88, width=0.55)
    ax.set_xticks(x); ax.set_xticklabels(minerals, color=WHITE, fontsize=8)
    ax.set_ylabel("経済的重要度スコア")
    for bar, role in zip(bars, roles):
        ax.text(bar.get_x()+bar.get_width()/2, 2, role,
                ha="center", va="bottom", color=WHITE, fontsize=6.5,
                rotation=0)
    ax.text(0.02,0.88,"He-3: 核融合燃料\n将来価値 $3M/kg",
            transform=ax.transAxes, color="#EA80FC", fontsize=8,
            bbox=dict(boxstyle="round",facecolor="#111",edgecolor="#EA80FC",alpha=0.7))

def p_space_market(ax):
    sa(ax)
    ax.set_title("月面採掘市場 + Disco装置需要予測", fontsize=9, pad=5)
    y = SPACE_MARKET["years"]
    ax.fill_between(y, SPACE_MARKET["lunar_mining_B$"], alpha=0.15, color="#EA80FC")
    ax.plot(y, SPACE_MARKET["lunar_mining_B$"], "o-", color="#EA80FC", lw=2.2, ms=5, label="月面採掘市場全体")
    ax.fill_between(y, SPACE_MARKET["disco_cut_B$"], alpha=0.3, color="#FFD700")
    ax.plot(y, SPACE_MARKET["disco_cut_B$"], "s-", color="#FFD700", lw=2.2, ms=5, label="Disco宇宙装置売上")
    ax.set_ylabel("市場規模 ($B)")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    ax.text(2040, 22, "2040年\n$20B", color="#FFD700", fontsize=9, fontweight="bold")
    ax.text(0.02,0.08,"JAXAとの共同開発から\n宇宙対応装置ラインを展開",
            transform=ax.transAxes, color=DIM, fontsize=8)

def p_disco_space_tech(ax):
    sa(ax); ax.set_facecolor("#060606"); ax.axis("off")
    ax.set_title("Discoの宇宙対応技術 — 既に持っているもの", fontsize=9, pad=5)
    items = [
        ("真空対応", "#EA80FC",
         "現在のダイシング装置は\n真空チャンバー対応済み\n月面(真空)でもそのまま動く"),
        ("極端環境耐性", "#CE93D8",
         "半導体製造の\nクリーンルーム基準を満たす\n月面の温度差(-180°C〜+130°C)\nへの改良は必要"),
        ("岩石切断", "#B39DDB",
         "SiC(硬度9.5)を切れるブレードは\n月面岩石(硬度6〜8)を楽に切る\n新しい刃の開発不要"),
        ("遠隔操作", "#9FA8DA",
         "既存装置のIoT化を進めれば\n地球からの遠隔操作に転用可\nNASA/JAXAの要求仕様に対応"),
    ]
    for i,(title,color,body) in enumerate(items):
        x=(i%2)*0.5+0.01; y=0.92-(i//2)*0.46
        ax.text(x+0.22,y,title,transform=ax.transAxes,
                ha="center",color=color,fontsize=9,fontweight="bold")
        ax.text(x+0.02,y-0.06,body,transform=ax.transAxes,
                color=WHITE,fontsize=8.5,va="top",
                bbox=dict(boxstyle="round,pad=0.4",facecolor="#111",edgecolor=color,alpha=0.8))

def main():
    fig = plt.figure(figsize=(22,26), facecolor=BG)
    fig.suptitle("Disco 魔改造 #8 DNA物理切断 + #9 宇宙資源採掘\n「ぶっ飛び」2案 — 15〜20年後の未来",
                 fontsize=14, color=WHITE, fontweight="bold", y=0.988)
    gs = gridspec.GridSpec(3,3,figure=fig,hspace=0.46,wspace=0.34,
                           left=0.06,right=0.97,top=0.960,bottom=0.04)
    p_precision_compare(fig.add_subplot(gs[0,0]))
    p_scale_gap(        fig.add_subplot(gs[0,1]))
    p_crispr_market(    fig.add_subplot(gs[0,2]))
    p_moon_minerals(    fig.add_subplot(gs[1,0]))
    p_space_market(     fig.add_subplot(gs[1,1]))
    p_disco_space_tech( fig.add_subplot(gs[1,2]))
    ax_s = fig.add_subplot(gs[2,:])
    sa(ax_s); ax_s.set_facecolor("#060606"); ax_s.axis("off")
    ax_s.set_title("2案の正直な評価", fontsize=10, pad=5)
    for i,(title,color,body) in enumerate([
        ("#8 DNA物理切断", "#B2FF59",
         "現在のDiscoの精度: 30μm カーフ\nDNA切断に必要: 0.5nm = 60,000倍の精度向上\n"
         "→ 現状では物理的に不可能\nただしナノスケールアクチュエータ + EUV光学系と組み合わせれば\n"
         "2035〜2040年代に可能性が開く\n市場: $35B (2035) — 取れれば巨大"),
        ("#9 宇宙資源採掘", "#EA80FC",
         "技術的には最も現実的な「ぶっ飛び案」\n真空対応・岩石切断 → 既に技術がある\n"
         "JAXA「月面資源探査」2030年代に本格化\n→ 今から共同研究を始めれば間に合う\n"
         "市場: 100B USD (2040), Disco装置シェア20%で 20B USD"),
        ("2案への投資戦略", "#FFD700",
         "DNA: 研究投資のみ (年¥1〜2億) — 東大・慶應と共同研究\n"
         "宇宙: 本格投資 (年¥5〜10億) — JAXAとMOU締結から\n\n"
         "「ぶっ飛び」こそがDiscoの時価総額に\n非線形のプレミアムをもたらす\n"
         "→ 投資家に「この会社は20年先を見ている」と思わせる"),
    ]):
        x=0.01+i*0.33
        ax_s.text(x+0.14,0.92,title,transform=ax_s.transAxes,
                  ha="center",color=color,fontsize=9.5,fontweight="bold")
        ax_s.text(x+0.01,0.78,body,transform=ax_s.transAxes,
                  color=WHITE,fontsize=8.5,va="top",
                  bbox=dict(boxstyle="round,pad=0.4",facecolor="#111",edgecolor=color,alpha=0.8))
    out = os.path.join(OUT_DIR,"disco_future.png")
    plt.savefig(out,dpi=150,bbox_inches="tight",facecolor=BG)
    plt.close(fig); print(f"Saved: {out}")

if __name__ == "__main__":
    main()
