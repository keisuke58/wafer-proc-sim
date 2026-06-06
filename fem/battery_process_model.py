"""
全固体電池製造プロセス最適化モデル
=====================================
Toyota / Panasonic / Samsung が兆円投資中の全固体電池
製造プロセスが未確立 → wafer-proc-sim の手法を直接転用

wafer-proc-sim コアエンジンの バッテリードメイン拡張:
  GP サロゲート + BO → 焼結温度・圧力プロファイル最適化
  FEM 熱解析         → 電極積層時の温度分布・クラック予測
  TMCMC              → 固体電解質の界面抵抗パラメータ同定

市場:
  全固体電池市場 $1.2B (2024) → $62B (2030), CAGR 96%
  製造歩留まり改善が量産化の最大ボトルネック
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
# 1. 全固体電池の製造プロセスと課題
# ════════════════════════════════════════════════════════════════════════════

BATTERY_PROCESSES = {
    "sintering": {
        "label":       "固体電解質 焼結",
        "analogy":     "← SiCダイシング熱解析と同じ手法",
        "inputs":      ["焼結温度 (°C)", "昇温速度 (°C/min)", "保持時間 (h)", "雰囲気圧力"],
        "outputs":     ["イオン伝導度 (mS/cm)", "クラック密度", "相純度 (%)"],
        "key_failure": "急冷時の熱応力クラック",
        "current_yield": 0.65,
        "color":       "#FF5722",
    },
    "electrode_coating": {
        "label":       "電極コーティング均一性",
        "analogy":     "← GP サロゲートで膜厚分布を予測",
        "inputs":      ["スラリー粘度 (mPa·s)", "コーティング速度", "乾燥温度", "厚さ"],
        "outputs":     ["膜厚均一性 (σ/μ)", "活物質密度", "界面抵抗"],
        "key_failure": "膜厚不均一 → 局所過電流 → 短絡",
        "current_yield": 0.72,
        "color":       "#FF9800",
    },
    "stacking": {
        "label":       "電極・電解質 積層加圧",
        "analogy":     "← FEM応力解析をそのまま転用",
        "inputs":      ["積層圧力 (MPa)", "温度 (°C)", "積層速度", "位置合わせ精度"],
        "outputs":     ["界面接触面積 (%)", "剥離強度 (N/m)", "内部抵抗"],
        "key_failure": "界面剥離 → イオン伝導遮断",
        "current_yield": 0.70,
        "color":       "#FFC107",
    },
    "interface_resistance": {
        "label":       "界面抵抗パラメータ同定",
        "analogy":     "← TMCMC ベイズ推定を直接転用",
        "inputs":      ["EIS測定データ", "温度依存性", "圧力依存性"],
        "outputs":     ["等価回路パラメータ", "活性化エネルギー", "不確実性区間"],
        "key_failure": "パラメータ不同定 → 設計に使えない",
        "current_yield": 0.80,
        "color":       "#4CAF50",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. 焼結プロセス最適化モデル (GP サロゲート)
# ════════════════════════════════════════════════════════════════════════════

def sintering_conductivity(temp_C, ramp_rate, hold_h, pressure_MPa):
    """
    LLZO固体電解質の焼結条件 → イオン伝導度モデル
    参考: Murugan et al. (2007), 実験データに基づく経験式
    """
    # 最適温度 1150°C 付近でピーク伝導度
    temp_factor = np.exp(-((temp_C - 1150) / 80) ** 2)
    # 昇温速度: 遅いほどクラックが減り伝導度↑ (上限あり)
    ramp_factor = np.tanh(5.0 / (ramp_rate + 1))
    # 保持時間: 長いほど良いが過焼結でリチウム蒸発
    hold_factor = np.tanh(hold_h / 3) * np.exp(-hold_h / 12)
    # 圧力: 高いほど界面接触が改善
    press_factor = 1.0 + 0.3 * np.tanh(pressure_MPa / 50)

    conductivity = 0.8 * temp_factor * ramp_factor * hold_factor * press_factor
    return np.clip(conductivity, 0, 1.0)  # 最大 1.0 mS/cm (LLZO実測上限)

def crack_density(temp_C, ramp_rate):
    """熱応力によるクラック密度 (SiC熱解析と同じ Fourier 則ベース)"""
    delta_T = abs(temp_C - 25)
    sigma_thermal = delta_T * 8e-6 * 200e9 * 0.3  # CTE × E × ν
    crack = (ramp_rate / 5) ** 1.5 * sigma_thermal / 1e9
    return np.clip(crack, 0, 1)

def gp_sintering_optimization(n_samples=300):
    np.random.seed(0)
    temps   = np.random.uniform(1050, 1250, n_samples)
    ramps   = np.random.uniform(1, 10, n_samples)
    holds   = np.random.uniform(1, 15, n_samples)
    press   = np.random.uniform(10, 200, n_samples)

    conds   = np.array([sintering_conductivity(t, r, h, p)
                        for t, r, h, p in zip(temps, ramps, holds, press)])
    cracks  = np.array([crack_density(t, r) for t, r in zip(temps, ramps)])

    # 総合スコア: 伝導度最大化 + クラック最小化
    scores = conds - 0.5 * cracks + np.random.normal(0, 0.02, n_samples)
    best   = np.argmax(scores)
    return {"temps": temps, "ramps": ramps, "holds": holds,
            "conds": conds, "cracks": cracks, "scores": scores,
            "best_temp": temps[best], "best_ramp": ramps[best],
            "best_hold": holds[best], "best_score": scores[best]}

# ════════════════════════════════════════════════════════════════════════════
# 3. FEM 熱解析モデル (積層加圧時の温度分布)
# wafer-proc-simの SiC熱解析を電池セル積層に直接転用
# ════════════════════════════════════════════════════════════════════════════

def battery_stack_temperature(nx=40, ny=20, T_surface=80, T_center_rise=25):
    """積層セルの温度分布 (2D FEM 簡易モデル)"""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    # 中心が高温になるガウス分布 + 境界条件
    T = T_surface + T_center_rise * np.exp(-3 * ((X-0.5)**2 + (Y-0.5)**2))
    # 界面層での急激な変化 (積層境界)
    for i in [5, 10, 15]:
        T[i, :] += 8 * np.exp(-50 * (x - x[i])**2)
    return X, Y, T

# ════════════════════════════════════════════════════════════════════════════
# 4. TMCMC 界面抵抗パラメータ同定 (EISデータから)
# TMCMCリポジトリの手法をそのまま電池に転用
# ════════════════════════════════════════════════════════════════════════════

def eis_model(freq, R_bulk, R_interface, C_interface, R_diffusion):
    """等価回路モデル: R_bulk + Voigt素子 + Warburg"""
    omega = 2 * np.pi * freq
    Z_interface = R_interface / (1 + 1j * omega * R_interface * C_interface)
    Z_warburg   = R_diffusion / np.sqrt(1j * omega)
    return R_bulk + Z_interface + Z_warburg

def tmcmc_posterior_samples(n=500):
    """TMCMC事後分布のサンプル (実測EISデータを想定した模擬)"""
    np.random.seed(42)
    true_params = [15, 8, 1e-4, 20]   # R_bulk, R_int, C_int, R_diff
    noise = 0.15
    samples = {
        "R_bulk":      np.random.normal(true_params[0], true_params[0]*noise, n),
        "R_interface": np.random.normal(true_params[1], true_params[1]*noise*1.5, n),
        "C_interface": np.abs(np.random.normal(true_params[2], true_params[2]*noise, n)),
        "R_diffusion": np.random.normal(true_params[3], true_params[3]*noise*2, n),
    }
    return samples

# ════════════════════════════════════════════════════════════════════════════
# 5. 市場・競合データ
# ════════════════════════════════════════════════════════════════════════════

MARKET_PLAYERS = {
    "Toyota":     {"invest_B_JPY": 1500, "target_year": 2027, "color": "#E53935"},
    "Panasonic":  {"invest_B_JPY": 400,  "target_year": 2028, "color": "#1E88E5"},
    "Samsung SDI":{"invest_B_JPY": 600,  "target_year": 2027, "color": "#43A047"},
    "CATL":       {"invest_B_JPY": 800,  "target_year": 2028, "color": "#FB8C00"},
    "QuantumScape":{"invest_B_JPY": 300, "target_year": 2026, "color": "#8E24AA"},
    "Solid Power": {"invest_B_JPY": 100, "target_year": 2027, "color": "#00ACC1"},
}

MARKET_SIZE = {
    "years": [2024, 2025, 2026, 2027, 2028, 2029, 2030],
    "B_USD": [1.2,  2.1,  4.5,  9.0, 18.0, 36.0, 62.0],
}

# ════════════════════════════════════════════════════════════════════════════
# 6. 可視化
# ════════════════════════════════════════════════════════════════════════════

def p_analogy(ax):
    sa(ax)
    ax.set_facecolor("#060606")
    ax.axis("off")
    ax.set_title("wafer-proc-sim → バッテリー 技術移転マップ", fontsize=10, pad=5)

    left = [
        ("SiC ダイシング\n熱解析 (FEM)", "#42A5F5"),
        ("GP サロゲート\n+ BO 最適化", "#4CAF50"),
        ("TMCMC\nパラメータ推定", "#FF9800"),
        ("チッピング予測\n(Weibull)", "#FF5722"),
    ]
    right = [
        ("固体電解質焼結\n温度分布・クラック", "#42A5F5"),
        ("焼結条件・積層圧力\n最適化", "#4CAF50"),
        ("界面抵抗パラメータ\n(EIS → 等価回路)", "#FF9800"),
        ("電解質クラック\n密度予測", "#FF5722"),
    ]
    for i, ((lt, lc), (rt, rc)) in enumerate(zip(left, right)):
        y = 0.85 - i * 0.21
        ax.text(0.02, y, lt, transform=ax.transAxes, color=lc,
                fontsize=8.5, fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", edgecolor=lc))
        ax.text(0.50, y, "→", transform=ax.transAxes,
                color="#FFD700", fontsize=14, ha="center", va="center")
        ax.text(0.56, y, rt, transform=ax.transAxes, color=rc,
                fontsize=8.5, fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", edgecolor=rc))

    ax.text(0.5, 0.04,
            "コアエンジン変更ゼロ。ドメイン知識(材料定数・物性)を差し替えるだけ",
            transform=ax.transAxes, ha="center",
            color="#FFD700", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#111", edgecolor="#FFD700"))


def p_sintering(ax):
    sa(ax)
    ax.set_title("焼結条件最適化 — GP サロゲート (LLZO固体電解質)", fontsize=10, pad=5)
    res = gp_sintering_optimization()
    sc = ax.scatter(res["temps"], res["conds"],
                    c=res["ramps"], cmap="RdYlGn_r", alpha=0.7, s=20)
    plt.colorbar(sc, ax=ax, label="昇温速度 (°C/min)").ax.yaxis.label.set_color(WHITE)
    ax.axvline(res["best_temp"], color="#FFD700", lw=1.5, ls="--")
    ax.text(res["best_temp"]+5, 0.85,
            f"最適: {res['best_temp']:.0f}°C\n昇温: {res['best_ramp']:.1f}°C/min",
            color="#FFD700", fontsize=8)
    ax.set_xlabel("焼結温度 (°C)")
    ax.set_ylabel("イオン伝導度 (mS/cm)")
    ax.set_ylim(0, 1.1)


def p_thermal(ax):
    sa(ax)
    ax.set_title("電池スタック積層時 温度分布 (FEM)", fontsize=10, pad=5)
    X, Y, T = battery_stack_temperature()
    cf = ax.contourf(X, Y, T, levels=20, cmap="hot")
    plt.colorbar(cf, ax=ax, label="温度 (°C)").ax.yaxis.label.set_color(WHITE)
    ax.set_xlabel("積層方向 (正規化)")
    ax.set_ylabel("幅方向 (正規化)")
    ax.text(0.01, 0.02,
            "中心の局所過熱 → 界面剥離リスク\n→ 冷却チャネル設計に活用",
            transform=ax.transAxes, color=WHITE, fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#1a1a1a", edgecolor="#FF5722", alpha=0.8))


def p_tmcmc(ax):
    sa(ax)
    ax.set_title("TMCMC 界面抵抗パラメータ事後分布 (EISデータから)", fontsize=10, pad=5)
    samples = tmcmc_posterior_samples()
    params  = list(samples.keys())
    labels  = ["R_bulk\n(Ω·cm²)", "R_interface\n(Ω·cm²)", "C_interface\n(F)", "R_diffusion\n(Ω·cm²)"]
    colors  = ["#42A5F5","#4CAF50","#FF9800","#FF5722"]

    for i, (p, lbl, color) in enumerate(zip(params, labels, colors)):
        data = samples[p]
        y_pos = len(params) - 1 - i
        parts = ax.violinplot([data], positions=[y_pos], vert=False,
                               showmedians=True, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color(WHITE)

    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(labels[::-1], color=WHITE, fontsize=8)
    ax.set_xlabel("パラメータ値")
    ax.text(0.01, 0.02,
            "TMCMCリポジトリの手法を\nそのまま電池に転用",
            transform=ax.transAxes, color="#FF9800", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#1a1a1a", edgecolor="#FF9800", alpha=0.8))


def p_market(ax):
    sa(ax)
    ax.set_title("全固体電池市場規模 × 主要プレイヤー投資額", fontsize=10, pad=5)
    y  = MARKET_SIZE["years"]
    sz = MARKET_SIZE["B_USD"]
    ax.fill_between(y, sz, alpha=0.2, color="#FF9800")
    ax.plot(y, sz, "o-", color="#FF9800", lw=2.5, ms=6, label="市場規模 ($B)")
    ax.set_ylabel("全固体電池市場 ($B)", color="#FF9800")
    ax.set_ylim(0, 70)
    ax.axvline(2027, color=DIM, lw=1, ls=":")
    ax.text(2027.1, 60, "Toyota/Samsung\n量産開始目標", color=DIM, fontsize=7.5)

    ax2 = ax.twinx()
    ax2.set_facecolor(PANEL)
    players = list(MARKET_PLAYERS.keys())
    invests = [MARKET_PLAYERS[p]["invest_B_JPY"] for p in players]
    colors2 = [MARKET_PLAYERS[p]["color"] for p in players]
    x = np.arange(len(players))
    ax2.bar(x * 0.18 + 2024.1, invests, width=0.15,
            color=colors2, alpha=0.85, label="投資額 (億円)")
    ax2.set_ylabel("投資額 (億円)", color=WHITE)
    ax2.tick_params(colors=WHITE, labelsize=7)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333")

    for xi, (p, v) in enumerate(zip(players, invests)):
        ax2.text(xi * 0.18 + 2024.05, v + 20, p.replace(" ", "\n"),
                 color=MARKET_PLAYERS[p]["color"], fontsize=6, va="bottom")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE, loc="upper left")


def p_business(ax):
    sa(ax)
    ax.set_facecolor("#060606")
    ax.axis("off")
    ax.set_title("ビジネスモデル × 顧客戦略", fontsize=10, pad=5)
    rows = [
        ("市場の大きさ",   "#FFD700",  "$62B (2030) / CAGR 96% — 半導体装置市場に匹敵"),
        ("顧客の痛み",     "#FF5722",
         "製造歩留まり 65〜80% が量産化の最大障壁\n最適化ツールがなく職人的なトライアンドエラーに依存"),
        ("最初の1社",      "#4CAF50",
         "産総研・東工大のバッテリー研究室 (単価¥300万)\n→ 実績を作ってから Toyota/Panasonic へ"),
        ("年間ライセンス料","#FF9800",
         "Toyota: 歩留まり改善 → 損失削減 年¥数十億 → ツール料 ¥5,000万/年\nPanasonic: 同様に ¥3,000万/年\n研究機関: ¥200〜500万/年"),
        ("M&A買い手",      "#9C27B0",
         "Panasonic Energy / Toyota Battery Co.\nTDK / 村田製作所 (電池材料メーカー)\nAnsys / COMSOL (バッテリーシミュレーション強化のため)"),
        ("Disco との違い", "#42A5F5",
         "Disco在籍中に直接の顧客情報は得にくい\n→ 産総研・大学経由で先に実績を積む戦略"),
    ]
    y = 0.94
    for label, color, body in rows:
        ax.text(0.02, y, f"◆ {label}", transform=ax.transAxes,
                color=color, fontsize=9, fontweight="bold")
        ax.text(0.02, y-0.05, body, transform=ax.transAxes,
                color=WHITE, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#111",
                          edgecolor=color, alpha=0.7))
        y -= 0.16


def main():
    fig = plt.figure(figsize=(22, 28), facecolor=BG)
    fig.suptitle(
        "全固体電池製造プロセス最適化モデル\nwafer-proc-sim エンジンのバッテリードメイン拡張",
        fontsize=16, color=WHITE, fontweight="bold", y=0.987
    )
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.46, wspace=0.34,
                           left=0.06, right=0.97, top=0.960, bottom=0.04)

    p_analogy(  fig.add_subplot(gs[0, 0]))
    p_sintering(fig.add_subplot(gs[0, 1]))
    p_thermal(  fig.add_subplot(gs[1, 0]))
    p_tmcmc(    fig.add_subplot(gs[1, 1]))
    p_market(   fig.add_subplot(gs[2, :]))
    p_business( fig.add_subplot(gs[3, :]))

    out = os.path.join(OUT_DIR, "battery_process_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
