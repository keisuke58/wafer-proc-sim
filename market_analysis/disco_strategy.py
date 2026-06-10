"""
Disco 魔改造 #5 Rapidus株主化 + #6 SiC自社生産 + #7 レーザー分社化IPO
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

# ══════════════════════════════════════════════
# #5 Rapidus株主化
RAPIDUS_SCENARIOS = {
    "失敗":   {"prob":0.50, "disco_return_x":0.0,  "color":"#FF5252"},
    "部分成功":{"prob":0.30, "disco_return_x":3.0,  "color":"#FF9800"},
    "成功":   {"prob":0.15, "disco_return_x":10.0, "color":"#4CAF50"},
    "大成功": {"prob":0.05, "disco_return_x":30.0, "color":"#FFD700"},
}
DISCO_INVEST_B_JPY = 20   # Disco が Rapidus に出資する金額

# #6 SiC自社生産
SIC_COST = {
    "外部調達": {"cost_per_wafer_USD":800,  "margin_pct":0.32, "color":"#FF5252"},
    "一部内製": {"cost_per_wafer_USD":400,  "margin_pct":0.55, "color":"#FF9800"},
    "完全内製": {"cost_per_wafer_USD":200,  "margin_pct":0.75, "color":"#4CAF50"},
}
SIC_WAFER_DEMAND = {  # Disco の SiC 消耗品に使う 年間ウェーハ数推計
    "2024": 50_000,
    "2027": 120_000,
    "2030": 300_000,
}

# #7 レーザー分社化
LASER_DIVISION = {
    "revenue_B_JPY":   393 * 0.25,   # レーザー部門推定売上 ¥98B
    "growth_rate":     0.18,          # 年18%成長
    "arr_multiple":    30,            # SaaS的バリュエーション (ARR×30)
    "pe_multiple":     15,            # 装置会社バリュエーション
    "disco_pe_current":30,
}

# ══════════════════════════════════════════════
def p_rapidus_ev(ax):
    sa(ax)
    ax.set_title(f"Rapidus株主化 — 期待値計算\n(Disco出資額 ¥{DISCO_INVEST_B_JPY}億想定)", fontsize=9, pad=5)
    scenarios = list(RAPIDUS_SCENARIOS.keys())
    probs  = [RAPIDUS_SCENARIOS[s]["prob"] for s in scenarios]
    returns= [RAPIDUS_SCENARIOS[s]["disco_return_x"] * DISCO_INVEST_B_JPY
              for s in scenarios]
    colors = [RAPIDUS_SCENARIOS[s]["color"] for s in scenarios]
    x = np.arange(len(scenarios))
    bars = ax.bar(x, returns, color=colors, alpha=0.88, width=0.55)
    for bar, p, r in zip(bars, probs, returns):
        ax.text(bar.get_x()+bar.get_width()/2, r+2,
                f"¥{r:.0f}B\n(確率{p*100:.0f}%)",
                ha="center", color=WHITE, fontsize=8, fontweight="bold")
    ev = sum(p*r for p,r in zip(probs,returns))
    ax.axhline(DISCO_INVEST_B_JPY, color="#FF5252", lw=1.5, ls="--")
    ax.text(0.1, DISCO_INVEST_B_JPY+2, f"投資額 ¥{DISCO_INVEST_B_JPY}B", color="#FF5252", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, color=WHITE, fontsize=9)
    ax.set_ylabel("Disco の回収額 (億円)")
    ax.text(0.98,0.95,f"期待リターン: ¥{ev:.0f}B\n投資対効果: {ev/DISCO_INVEST_B_JPY:.1f}倍",
            transform=ax.transAxes, ha="right", color="#FFD700", fontsize=9,
            bbox=dict(boxstyle="round",facecolor="#111",edgecolor="#FFD700"))

def p_rapidus_demand(ax):
    sa(ax)
    ax.set_title("Rapidus成功シナリオでの Disco 装置需要", fontsize=9, pad=5)
    years = np.arange(2024, 2035)
    base  = np.linspace(393, 450, len(years))
    rapid_bull = base + np.array([0,0,5,15,30,60,100,150,200,250,300])
    rapid_bear = base.copy()
    ax.fill_between(years, base, rapid_bull, alpha=0.3, color="#FF5252", label="Rapidus成功上乗せ")
    ax.plot(years, base,       "--", color="#546e7a", lw=2, label="Rapidusなし基本")
    ax.plot(years, rapid_bull, "o-", color="#FF5252", lw=2.5, ms=5, label="Rapidus成功シナリオ")
    ax.set_ylabel("Disco 売上 (億円/年)")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    ax.text(2033, rapid_bull[-1]+5, f"¥{rapid_bull[-1]:.0f}B", color="#FF5252", fontsize=9, fontweight="bold")
    ax.text(0.02,0.08,"株主として利益相反がなくなる\n→ 全力で Rapidus を支援できる",
            transform=ax.transAxes, color=DIM, fontsize=8)

def p_sic_margin(ax):
    sa(ax)
    ax.set_title("SiCウェーハ 自社生産 → 原価・粗利の変化", fontsize=9, pad=5)
    models = list(SIC_COST.keys())
    costs  = [SIC_COST[m]["cost_per_wafer_USD"] for m in models]
    margins= [SIC_COST[m]["margin_pct"]*100 for m in models]
    colors = [SIC_COST[m]["color"] for m in models]
    x = np.arange(len(models))
    ax2 = ax.twinx()
    bars = ax.bar(x-0.15, costs, 0.3, color=colors, alpha=0.85, label="ウェーハコスト ($/枚)")
    ax2.bar(x+0.15, margins, 0.3, color=colors, alpha=0.5, label="粗利率 (%)")
    for bar, c in zip(bars, costs):
        ax.text(bar.get_x()+bar.get_width()/2, c+10, f"${c}", ha="center", color=WHITE, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(models, color=WHITE, fontsize=9)
    ax.set_ylabel("ウェーハコスト ($/枚)", color=WHITE)
    ax2.set_ylabel("粗利率 (%)", color=DIM)
    ax2.tick_params(colors=DIM); ax2.set_facecolor(PANEL)
    for sp in ax2.spines.values(): sp.set_edgecolor("#333")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE, loc="upper right")
    ax2.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE, loc="upper left")

def p_sic_profit(ax):
    sa(ax)
    ax.set_title("SiC自社生産 — 年間利益増加額の試算", fontsize=9, pad=5)
    years = [2024, 2027, 2030]
    demands = [50_000, 120_000, 300_000]
    for model_key, demand_list in [("外部調達", [d*800*0 for d in demands]),
                                    ("完全内製", [d*(800-200) for d in demands])]:
        pass
    # 外部調達vs完全内製の利益差
    profit_diff = [d*(800-200)/1e6 for d in demands]   # $M
    bars = ax.bar([str(y) for y in years], profit_diff,
                  color=["#FF9800","#4CAF50","#FFD700"], alpha=0.88, width=0.5)
    for bar, p in zip(bars, profit_diff):
        ax.text(bar.get_x()+bar.get_width()/2, p+2,
                f"${p:.0f}M\n(¥{p*150/100:.0f}億)",
                ha="center", color=WHITE, fontsize=9, fontweight="bold")
    ax.set_ylabel("完全内製化による利益増加 ($M/年)")
    ax.set_ylim(0, max(profit_diff)*1.3)
    ax.text(0.02,0.85,"前提: ウェーハコスト\n外部$800→内製$200",
            transform=ax.transAxes, color=DIM, fontsize=8)

def p_laser_valuation(ax):
    sa(ax)
    ax.set_title("レーザー部門 分社化IPO — バリュエーション比較", fontsize=9, pad=5)
    years = np.arange(0, 8)
    rev   = LASER_DIVISION["revenue_B_JPY"] * (1+LASER_DIVISION["growth_rate"])**years
    val_pe    = rev * LASER_DIVISION["pe_multiple"]      # 装置PERベース
    val_saas  = rev * LASER_DIVISION["arr_multiple"]     # SaaSバリュエーション
    ax.fill_between(years, val_pe, val_saas, alpha=0.2, color="#40C4FF", label="分社化による価値増加ゾーン")
    ax.plot(years, val_pe,   "o-", color="#546e7a", lw=2.2, ms=5, label=f"装置PER({LASER_DIVISION['pe_multiple']}倍)での評価")
    ax.plot(years, val_saas, "s-", color="#40C4FF", lw=2.5, ms=5, label=f"SaaS評価(ARR×{LASER_DIVISION['arr_multiple']}倍)")
    ax.set_xlabel("分社化後 経過年数")
    ax.set_ylabel("レーザー部門 時価総額 (億円)")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    ax.text(6, val_saas[-1]+100, f"¥{val_saas[-1]:.0f}B", color="#40C4FF", fontsize=9, fontweight="bold")
    delta = val_saas[-1] - val_pe[-1]
    ax.text(0.02,0.08, f"分社化で生まれる追加価値:\n最大 ¥{delta:.0f}億 (7年後)",
            transform=ax.transAxes, color="#FFD700", fontsize=8,
            bbox=dict(boxstyle="round",facecolor="#111",edgecolor="#FFD700"))

def p_strategy_map(ax):
    sa(ax); ax.set_facecolor("#060606"); ax.axis("off")
    ax.set_title("3案の統合戦略マップ", fontsize=9, pad=5)
    items = [
        ("Rapidus株主化", "#FF5252", "2年以内に交渉\n出資額¥20億\n成功時リターン最大30倍\n装置受注の優先交渉権も獲得"),
        ("SiC自社生産", "#FF9800",  "5〜7年の中期戦略\nM&AまたはJV設立\n2030年需要300k枚で\n利益+¥270億/年"),
        ("レーザー分社化", "#40C4FF","5年以内に分社化\n装置PER→SaaS評価へ\n7年後追加時価総額\n最大¥4,000億"),
        ("3案の共通点", "#FFD700",   "いずれも「既存資産の\n再評価・再定義」\n新しい技術開発不要\n戦略判断だけで実現できる"),
    ]
    for i,(title,color,body) in enumerate(items):
        x = (i%2)*0.5+0.01; y = 0.92-(i//2)*0.48
        ax.text(x+0.22,y,title,transform=ax.transAxes,ha="center",
                color=color,fontsize=9,fontweight="bold")
        ax.text(x+0.02,y-0.06,body,transform=ax.transAxes,
                color=WHITE,fontsize=8.5,va="top",
                bbox=dict(boxstyle="round,pad=0.4",facecolor="#111",edgecolor=color,alpha=0.8))

def main():
    fig = plt.figure(figsize=(22,26), facecolor=BG)
    fig.suptitle("Disco 魔改造 #5 Rapidus株主化 + #6 SiC自社生産 + #7 レーザー分社化IPO",
                 fontsize=14, color=WHITE, fontweight="bold", y=0.988)
    gs = gridspec.GridSpec(3,3,figure=fig,hspace=0.46,wspace=0.34,
                           left=0.06,right=0.97,top=0.962,bottom=0.04)
    p_rapidus_ev(    fig.add_subplot(gs[0,0]))
    p_rapidus_demand(fig.add_subplot(gs[0,1]))
    p_strategy_map(  fig.add_subplot(gs[0,2]))
    p_sic_margin(    fig.add_subplot(gs[1,0]))
    p_sic_profit(    fig.add_subplot(gs[1,1]))
    p_laser_valuation(fig.add_subplot(gs[1,2]))
    ax_s = fig.add_subplot(gs[2,:])
    sa(ax_s); ax_s.set_facecolor("#060606"); ax_s.axis("off")
    ax_s.set_title("統合シナリオ: 3案全て実行した場合の Disco 時価総額", fontsize=10, pad=5)
    years = np.arange(0,11)
    base_mktcap = 1_200   # 現在の時価総額 ¥1.2兆 (概算)
    combined = base_mktcap + years**2 * 50 + years * 80
    ax_s.fill_between(years, base_mktcap, combined, alpha=0.25, color="#FFD700")
    ax_s.axhline(base_mktcap, color="#546e7a", lw=1.5, ls="--")
    ax_s.plot(years, combined, "o-", color="#FFD700", lw=2.5, ms=6)
    ax_s.text(10, combined[-1]+50, f"¥{combined[-1]/1000:.1f}兆", color="#FFD700", fontsize=12, fontweight="bold")
    ax_s.text(0, base_mktcap+30, f"現在 ¥{base_mktcap/1000:.1f}兆", color="#546e7a", fontsize=9)
    ax_s.set_xlabel("3案実行後 経過年数"); ax_s.set_ylabel("Disco 推定時価総額 (億円)")
    out = os.path.join(OUT_DIR,"disco_strategy.png")
    plt.savefig(out,dpi=150,bbox_inches="tight",facecolor=BG)
    plt.close(fig); print(f"Saved: {out}")

if __name__ == "__main__":
    main()
