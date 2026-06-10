"""
Disco 魔改造 #3 プロセスデータDB + #4 消耗品サブスク化
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

# ══════════════════════════════════
# #3 プロセスDB データ
N_MACHINES   = 30_000
DATA_PER_DAY_GB = 0.5   # 1台あたり
SENSORS = ["主軸電流","振動(X/Y/Z)","スピンドル温度","クーラント流量",
           "ブレード摩耗電流","切削音響","ウェーハ枚数","ブレード交換回数"]

def db_value_model(n_machines, years):
    """データ蓄積量 → 推定価値 (Metcalfe則: n^2に比例)"""
    data_TB  = n_machines * Data_per_yr_TB * years
    value_B  = (n_machines / 1000)**2 * 0.05 * np.log1p(years)
    return data_TB, value_B

Data_per_yr_TB = DATA_PER_DAY_GB * 365 / 1000

# サブスクリプション モデル
CONSUMABLE_CURRENT = {
    "revenue_B_JPY":  393 * 0.32,  # 現在の消耗品売上 ¥125.7B
    "avg_contract_M": 5,            # 平均顧客の年間消耗品購買 ¥5M
    "n_customers":    6000,         # 推定顧客数
    "churn_rate":     0.05,         # 年5%解約
}
SAAS_MODEL = {
    "price_per_1000cuts_JPY": 50_000,   # 1000カットあたり ¥5万
    "avg_cuts_per_customer_day": 500,
    "conversion_rate": 0.30,            # 既存顧客の30%が移行
    "nrr": 1.10,                        # Net Revenue Retention 110%
}

def saas_arr_growth(years):
    base_customers = CONSUMABLE_CURRENT["n_customers"] * SAAS_MODEL["conversion_rate"]
    arr = []
    customers = base_customers
    for y in years:
        if y == 0: arr.append(0); continue
        rev_per_cust = (SAAS_MODEL["price_per_1000cuts_JPY"] *
                        SAAS_MODEL["avg_cuts_per_customer_day"] * 365 / 1000 / 1e6)
        arr.append(customers * rev_per_cust)
        customers *= SAAS_MODEL["nrr"]
    return np.array(arr)

# ══════════════════════════════════
def p_data_volume(ax):
    sa(ax)
    ax.set_title(f"世界{N_MACHINES:,}台のデータ蓄積量\n今は全部捨てている", fontsize=9, pad=5)
    years = np.arange(0, 11)
    tb    = N_MACHINES * Data_per_yr_TB * years
    ax.fill_between(years, tb, alpha=0.2, color="#E040FB")
    ax.plot(years, tb, "o-", color="#E040FB", lw=2.5, ms=5)
    ax.set_xlabel("蓄積年数"); ax.set_ylabel("累積データ量 (TB)")
    for y, t in zip(years[::2], tb[::2]):
        ax.text(y, t+500, f"{t:.0f}TB", ha="center", color=WHITE, fontsize=7.5)
    ax.text(0.02, 0.85, f"センサー種別:\n" + "\n".join(SENSORS[:4]),
            transform=ax.transAxes, color=DIM, fontsize=7)

def p_data_value(ax):
    sa(ax)
    ax.set_title("データベース推定価値 — Metcalfe則\n(接続装置数の二乗に比例)", fontsize=9, pad=5)
    machines = np.logspace(2, 4.5, 100)
    years_list = [1, 3, 5, 10]
    colors_v = ["#90CAF9","#42A5F5","#1565C0","#E040FB"]
    for yr, col in zip(years_list, colors_v):
        vals = (machines/1000)**2 * 0.05 * np.log1p(yr)
        ax.plot(machines, vals, lw=2, color=col, label=f"{yr}年蓄積")
    ax.axvline(N_MACHINES, color="#FFD700", lw=2, ls="--")
    ax.text(N_MACHINES*1.05, 30, f"現在\n{N_MACHINES//1000}k台", color="#FFD700", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("接続装置数 (台)")
    ax.set_ylabel("推定DB価値 ($B)")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)

def p_data_products(ax):
    sa(ax); ax.set_facecolor("#060606"); ax.axis("off")
    ax.set_title("データから作れる製品", fontsize=9, pad=5)
    products = [
        ("異常検知SaaS", "#E040FB",
         "ブレード破損をリアルタイム予測\n→ 装置停止前に自動アラート\n→ 顧客の歩留まり向上に直結"),
        ("最適レシピDB", "#9C27B0",
         "材料×厚さ×目的別の最適条件\nをDBから自動提案\n→ 新製品立ち上げ期間を1/3に"),
        ("予知保全MRR", "#7B1FA2",
         "消耗品交換タイミングを\nAIで予測して自動発注\n→ 顧客の在庫コスト削減"),
        ("競合ベンチマーク", "#4A148C",
         "匿名化した業界平均と\n自社装置の比較レポート\n→ 顧客が「離れられなくなる」"),
    ]
    for i,(title,color,body) in enumerate(products):
        y = 0.88 - i*0.22
        ax.text(0.02,y,f"◆ {title}",transform=ax.transAxes,
                color=color,fontsize=9,fontweight="bold")
        ax.text(0.02,y-0.05,body,transform=ax.transAxes,
                color=WHITE,fontsize=8,va="top",
                bbox=dict(boxstyle="round,pad=0.25",facecolor="#111",edgecolor=color,alpha=0.7))

def p_saas_transition(ax):
    sa(ax)
    ax.set_title("消耗品売上 → サブスク(MRR)への移行モデル", fontsize=9, pad=5)
    years = np.arange(0, 11)
    current_blade = np.array([CONSUMABLE_CURRENT["revenue_B_JPY"] *
                               (1-SAAS_MODEL["conversion_rate"]*y/10) for y in years])
    saas = saas_arr_growth(years)
    ax.fill_between(years, current_blade, alpha=0.3, color="#546e7a", label="従来消耗品売上")
    ax.fill_between(years, saas, alpha=0.4, color="#69F0AE", label="サブスクARR")
    ax.plot(years, current_blade, "o-", color="#90A4AE", lw=2, ms=4)
    ax.plot(years, saas, "s-", color="#69F0AE", lw=2.5, ms=5)
    ax.plot(years, current_blade+saas, "^--", color="#FFD700", lw=2, ms=5, label="合計")
    ax.set_xlabel("移行後 経過年数"); ax.set_ylabel("売上 (億円/年)")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    ax.text(8, saas[8]+5, f"ARR\n¥{saas[8]:.0f}B", color="#69F0AE", fontsize=8.5, fontweight="bold")

def p_saas_metrics(ax):
    sa(ax)
    ax.set_title("SaaSメトリクス予測 — Disco消耗品のサブスク転換", fontsize=9, pad=5)
    metrics_years = np.arange(1,6)
    mrr    = saas_arr_growth(metrics_years) / 12
    churn  = [5, 4, 3, 2, 2]
    nrr    = [105, 108, 110, 112, 115]

    ax2 = ax.twinx()
    ax.bar(metrics_years-0.2, mrr, 0.35, color="#69F0AE", alpha=0.85, label="MRR (億円)")
    ax2.plot(metrics_years+0.1, nrr, "D-", color="#FFD700", lw=2, ms=7, label="NRR (%)")
    ax2.plot(metrics_years+0.1, [100-c*2 for c in churn], "v-",
             color="#FF5252", lw=2, ms=7, label="Gross Retention (%)")
    ax.set_xlabel("年目"); ax.set_ylabel("MRR (億円)", color="#69F0AE")
    ax2.set_ylabel("NRR / Retention (%)", color="#FFD700")
    ax2.set_ylim(85, 120); ax2.tick_params(colors=WHITE)
    ax2.set_facecolor(PANEL)
    for sp in ax2.spines.values(): sp.set_edgecolor("#333")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE, loc="upper left")
    ax2.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE, loc="upper right")

def p_rolls_royce(ax):
    sa(ax); ax.set_facecolor("#060606"); ax.axis("off")
    ax.set_title("Rolls-Royce モデルの Disco 版", fontsize=9, pad=5)
    pairs = [
        ("Rolls-Royce\n'Power by the Hour'",  "Disco\n'Cuts by 1,000'"),
        ("エンジンを売らない\nフライト時間で課金", "ブレードを売らない\nカット数で課金"),
        ("航空会社のリスク低下\n→解約しにくい",   "顧客の在庫管理不要\n→スイッチングコスト最大"),
        ("GE/P&Wが同様モデル\nで収益安定化",     "消耗品32%→ARR化で\nバリュエーション向上"),
    ]
    for i,((l,r)) in enumerate(pairs):
        y = 0.88-i*0.21
        ax.text(0.02,y,l,transform=ax.transAxes,color="#90A4AE",fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3",facecolor="#1a1a1a",edgecolor="#90A4AE"))
        ax.text(0.50,y,"→",transform=ax.transAxes,color="#FFD700",fontsize=14,ha="center",va="center")
        ax.text(0.55,y,r,transform=ax.transAxes,color="#69F0AE",fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3",facecolor="#1a1a1a",edgecolor="#69F0AE"))

def main():
    fig = plt.figure(figsize=(22,24), facecolor=BG)
    fig.suptitle("Disco 魔改造 #3 世界最大プロセスDB + #4 消耗品サブスク化",
                 fontsize=15, color=WHITE, fontweight="bold", y=0.988)
    gs = gridspec.GridSpec(3,3,figure=fig,hspace=0.48,wspace=0.34,
                           left=0.06,right=0.97,top=0.960,bottom=0.04)
    p_data_volume(  fig.add_subplot(gs[0,0]))
    p_data_value(   fig.add_subplot(gs[0,1]))
    p_data_products(fig.add_subplot(gs[0,2]))
    p_saas_transition(fig.add_subplot(gs[1,0]))
    p_saas_metrics( fig.add_subplot(gs[1,1]))
    p_rolls_royce(  fig.add_subplot(gs[1,2]))
    ax_s = fig.add_subplot(gs[2,:])
    sa(ax_s); ax_s.set_facecolor("#060606"); ax_s.axis("off")
    ax_s.set_title("2案の統合インパクト", fontsize=10, pad=5)
    for i,(title,color,body) in enumerate([
        ("#3 プロセスDB", "#E040FB",
         "今すぐ始められる (装置は存在する)\nデータ収集インフラ整備: ¥10〜30億\n"
         "3年後: 異常検知SaaS ARR ¥50億\n5年後: 業界標準DBとして ¥200億の価値\n"
         "M&A/IPO時の「データ資産」として別途バリュエーション"),
        ("#4 消耗品サブスク", "#69F0AE",
         "既存顧客6,000社の30%を転換するだけ\n移行コスト: システム開発 ¥5〜10億\n"
         "3年後ARR: ¥80億 / NRR 110%\n消耗品売上がSaaSバリュエーション(ARR×20倍)に変わる\n"
         "= 消耗品¥126Bが時価総額+¥1.6兆のインパクト"),
        ("投資家へのメッセージ", "#FFD700",
         "DiscoはSaaS企業になれる\n現在PER 30倍 → SaaS企業なら ARR×30倍\n"
         "消耗品¥126BをARR化 → 時価総額+¥1.5〜2兆\n「精密切断のRolls-Royce」として再定義"),
    ]):
        x = 0.01+i*0.33
        ax_s.text(x+0.14,0.92,title,transform=ax_s.transAxes,
                  ha="center",color=color,fontsize=9.5,fontweight="bold")
        ax_s.text(x+0.01,0.78,body,transform=ax_s.transAxes,
                  color=WHITE,fontsize=8.5,va="top",
                  bbox=dict(boxstyle="round,pad=0.4",facecolor="#111",edgecolor=color,alpha=0.8))
    out = os.path.join(OUT_DIR,"disco_data_saas.png")
    plt.savefig(out,dpi=150,bbox_inches="tight",facecolor=BG)
    plt.close(fig); print(f"Saved: {out}")

if __name__ == "__main__":
    main()
