"""
米中大戦 × 日本半導体サプライチェーン崩壊モデル
====================================================
シナリオ: 台湾海峡での軍事衝突 → 米中全面対立が日本の半導体・エネルギーに与える
定量的インパクトをシミュレーション

連鎖構造:
  台湾有事 → TSMC停止 → 先端ロジック全滅
       ↓
  中国禁輸 → レアアース/ガリウム/ゲルマニウム → 装置製造不能
       ↓
  シーレーン封鎖 → LNG/石油90日枯渇 → エネルギー危機
       ↓
  日本国内: Disco売上崩壊 / Rapidus計画消滅 / 円暴落

モデル内容:
  1. サプライチェーン依存度マップ
  2. シナリオ別GDPインパクト (短期紛争 / 台湾占領 / 全面戦争)
  3. Disco売上シミュレーション (有事シナリオ)
  4. NVIDIA Japan の地政学的ポジション変化
  5. 日本のレアアース/エネルギー90日ルール
  6. キャリア最適戦略ヒートマップ

References:
  - Cabinet Office Japan: GDP sensitivity to trade disruption (2023)
  - METI: Critical materials dependency report (2024)
  - JOGMEC: LNG strategic reserve status (2024)
  - Disco IR: FY2024 annual report (売上地域別)
  - NVIDIA: Export control impact assessment (2024)
  - RAND: War with China: Thinking Through the Unthinkable (2016)
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass, field
from typing import Dict, List

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ════════════════════════════════════════════════════════════════════════════
# 1. 日本の戦略物資依存度
# ════════════════════════════════════════════════════════════════════════════

STRATEGIC_DEPENDENCIES = {
    # ── レアアース / 半導体材料 ────────────────────────────────────────────
    "Gallium": {
        "label":          "ガリウム (GaN基板・化合物半導体)",
        "china_share":    0.98,   # 中国が世界生産の98%
        "japan_stock_days": 60,   # 戦略備蓄日数 (推定)
        "alt_source_years": 5.0,  # 代替調達確立まで
        "impact_sector":  ["GaN装置", "5G", "GaN-EV"],
        "severity":       "CRITICAL",
        "color":          "#d62728",
    },
    "Germanium": {
        "label":          "ゲルマニウム (光ファイバ/赤外線センサ)",
        "china_share":    0.60,
        "japan_stock_days": 90,
        "alt_source_years": 3.0,
        "impact_sector":  ["光通信", "衛星センサ", "夜視装置"],
        "severity":       "HIGH",
        "color":          "#ff7f0e",
    },
    "REE_Neodymium": {
        "label":          "ネオジム (モーター永久磁石)",
        "china_share":    0.85,
        "japan_stock_days": 180,  # 日本が備蓄改善済み
        "alt_source_years": 4.0,
        "impact_sector":  ["EV", "風力発電", "ロボット"],
        "severity":       "HIGH",
        "color":          "#ff7f0e",
    },
    "SiC_Substrate": {
        "label":          "SiC基板 (台湾Wolfspeed/STマイクロ)",
        "china_share":    0.05,   # 中国依存は低いが台湾経由
        "taiwan_share":   0.35,
        "japan_stock_days": 45,
        "alt_source_years": 2.0,
        "impact_sector":  ["EV-SiC", "Disco消耗品", "電力変換"],
        "severity":       "HIGH",
        "color":          "#ff7f0e",
    },
    "TSMC_Logic": {
        "label":          "TSMC先端ロジック (3/5nm)",
        "china_share":    0.0,
        "taiwan_share":   0.92,   # 先端ロジックの92%
        "japan_stock_days": 30,   # ウェーハ在庫
        "alt_source_years": 7.0,  # Samsung/Intelでは代替不可
        "impact_sector":  ["AI GPU", "スマホ", "HPC", "車載"],
        "severity":       "CRITICAL",
        "color":          "#d62728",
    },
    "LNG": {
        "label":          "LNG (液化天然ガス)",
        "china_share":    0.0,
        "middle_east_share": 0.65,  # 中東依存
        "japan_stock_days": 14,   # 実質備蓄 (日本は基本在庫回転モデル)
        "alt_source_years": 1.0,  # 豪州/米国からだが時間要
        "impact_sector":  ["発電", "製造業", "民生"],
        "severity":       "CRITICAL",
        "color":          "#d62728",
    },
    "Crude_Oil": {
        "label":          "原油",
        "china_share":    0.0,
        "middle_east_share": 0.92,
        "japan_stock_days": 200,  # 国家備蓄 (IEA基準90日+民間)
        "alt_source_years": 0.5,
        "impact_sector":  ["輸送", "石化", "プラスチック"],
        "severity":       "MEDIUM",
        "color":          "#2ca02c",
    },
    "HBM_Memory": {
        "label":          "HBMメモリ (SK Hynix/Samsung)",
        "china_share":    0.0,
        "korea_share":    0.95,
        "japan_stock_days": 60,
        "alt_source_years": 4.0,
        "impact_sector":  ["AI学習", "HPC", "NVIDIA GPU"],
        "severity":       "HIGH",
        "color":          "#ff7f0e",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. 戦争シナリオ定義
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class WarScenario:
    name: str
    label: str
    color: str
    duration_months: float        # 紛争継続期間
    taiwan_falls: bool            # 台湾陥落するか
    sealane_blocked_months: float # シーレーン封鎖期間
    # 日本GDPへのインパクト (% の変化)
    gdp_impact_yr1: float
    gdp_impact_yr3: float
    gdp_impact_yr5: float
    # Disco 売上インパクト (multiplicative factor)
    disco_revenue_factor_yr1: float
    disco_revenue_factor_yr3: float
    # NVIDIA JP 事業規模 (有事後, 現在比)
    nvidia_jp_factor_yr3: float
    # 日本の軍事支出 GDP比 → 半導体国防需要
    defense_gdp_pct_yr5: float    # 現在 1.1% → ?
    # レアアース代替確立率
    rare_earth_alt_yr3: float

SCENARIOS: Dict[str, WarScenario] = {
    "short_conflict": WarScenario(
        name="short_conflict",
        label="短期紛争\n(数週間・台湾維持)",
        color="#2ca02c",
        duration_months=2,
        taiwan_falls=False,
        sealane_blocked_months=1,
        gdp_impact_yr1=-4.0,
        gdp_impact_yr3=-1.0,
        gdp_impact_yr5=+1.5,   # 復興需要でプラス転換
        disco_revenue_factor_yr1=0.65,
        disco_revenue_factor_yr3=0.90,
        nvidia_jp_factor_yr3=1.8,   # 防衛AI需要急増
        defense_gdp_pct_yr5=2.5,
        rare_earth_alt_yr3=0.40,
    ),
    "taiwan_occupation": WarScenario(
        name="taiwan_occupation",
        label="台湾占領\n(中国支配・膠着)",
        color="#ff7f0e",
        duration_months=24,
        taiwan_falls=True,
        sealane_blocked_months=6,
        gdp_impact_yr1=-12.0,
        gdp_impact_yr3=-8.0,
        gdp_impact_yr5=-3.0,
        disco_revenue_factor_yr1=0.40,
        disco_revenue_factor_yr3=0.55,
        nvidia_jp_factor_yr3=2.5,   # 軍事AIで急拡大するが管理下に
        defense_gdp_pct_yr5=4.0,
        rare_earth_alt_yr3=0.25,
    ),
    "total_war": WarScenario(
        name="total_war",
        label="全面戦争\n(核使用リスク)",
        color="#d62728",
        duration_months=60,
        taiwan_falls=True,
        sealane_blocked_months=36,
        gdp_impact_yr1=-30.0,
        gdp_impact_yr3=-45.0,
        gdp_impact_yr5=-60.0,
        disco_revenue_factor_yr1=0.15,
        disco_revenue_factor_yr3=0.10,
        nvidia_jp_factor_yr3=0.3,   # 輸出管理で機能不全
        defense_gdp_pct_yr5=8.0,
        rare_earth_alt_yr3=0.05,
    ),
}

# ════════════════════════════════════════════════════════════════════════════
# 3. Disco 売上シミュレーション (有事)
# ════════════════════════════════════════════════════════════════════════════

DISCO_FY2024 = {
    "total_revenue_B_JPY":  393.3,
    "china_taiwan_ratio":   0.35,   # 中国+台湾 ≈ 35%
    "korea_ratio":          0.18,
    "japan_ratio":          0.22,
    "us_europe_ratio":      0.25,
    "consumable_ratio":     0.32,
    "equipment_ratio":      0.68,
}

def disco_revenue_path(scenario: WarScenario, years: np.ndarray) -> np.ndarray:
    """有事シナリオ別のDiscoの売上軌跡を返す (FY2024=1.0 で正規化)"""
    baseline = 1.0
    china_tw_loss = DISCO_FY2024["china_taiwan_ratio"]

    revenue = np.ones_like(years, dtype=float)
    for i, yr in enumerate(years):
        if yr <= 0:
            revenue[i] = baseline
        elif yr <= 1:
            # フェーズ1: 受注キャンセル・工場停止
            shock = baseline - china_tw_loss * min(yr, 1) * (1 - scenario.disco_revenue_factor_yr1)
            revenue[i] = max(shock, scenario.disco_revenue_factor_yr1 * baseline)
        elif yr <= 3:
            t = (yr - 1) / 2
            r1 = scenario.disco_revenue_factor_yr1
            r3 = scenario.disco_revenue_factor_yr3
            revenue[i] = r1 + (r3 - r1) * t
        else:
            t = (yr - 3) / 5
            r3 = scenario.disco_revenue_factor_yr3
            # 長期: 防衛・国内需要で部分回復、ただし台湾陥落なら天井が低い
            recover_ceiling = 1.1 if not scenario.taiwan_falls else 0.75
            revenue[i] = r3 + (recover_ceiling - r3) * min(t, 1.0)
    return revenue

# ════════════════════════════════════════════════════════════════════════════
# 4. GDP インパクト計算
# ════════════════════════════════════════════════════════════════════════════

JAPAN_GDP_TRILLION_JPY = 591   # 2024年名目GDP

def gdp_loss_trillion(scenario: WarScenario) -> Dict[str, float]:
    return {
        "yr1": JAPAN_GDP_TRILLION_JPY * abs(scenario.gdp_impact_yr1) / 100,
        "yr3": JAPAN_GDP_TRILLION_JPY * abs(scenario.gdp_impact_yr3) / 100,
        "yr5": JAPAN_GDP_TRILLION_JPY * abs(scenario.gdp_impact_yr5) / 100,
    }

# ════════════════════════════════════════════════════════════════════════════
# 5. レアアース × 禁輸タイムライン
# ════════════════════════════════════════════════════════════════════════════

def stockpile_depletion_days(material_key: str) -> Dict[str, float]:
    m = STRATEGIC_DEPENDENCIES[material_key]
    stock = m["japan_stock_days"]
    return {
        "days_to_zero": stock,
        "weeks_to_zero": stock / 7,
        "months_to_zero": stock / 30,
    }

# ════════════════════════════════════════════════════════════════════════════
# 6. キャリアシナリオ × 有事ポジション
# ════════════════════════════════════════════════════════════════════════════

CAREER_WAR_MATRIX = {
    # (会社, シナリオ) → (安定度 0-10, 成長性 0-10, 社会的意義 0-10)
    ("Disco",         "short_conflict"):    (7, 6, 7),
    ("Disco",         "taiwan_occupation"): (4, 4, 8),  # 国防装置として重要になる
    ("Disco",         "total_war"):         (2, 2, 9),  # 生存自体は重要だが経営危機
    ("Lam_AMAT",      "short_conflict"):    (8, 7, 7),
    ("Lam_AMAT",      "taiwan_occupation"): (6, 8, 8),  # 国産化需要で急増
    ("Lam_AMAT",      "total_war"):         (3, 4, 9),
    ("NVIDIA_JP",     "short_conflict"):    (7, 9, 8),
    ("NVIDIA_JP",     "taiwan_occupation"): (5, 9, 9),  # 軍事AI最前線
    ("NVIDIA_JP",     "total_war"):         (2, 3, 10), # 輸出管理で機能不全、でも意義は最大
    ("Rapidus",       "short_conflict"):    (6, 7, 9),
    ("Rapidus",       "taiwan_occupation"): (7, 9, 10), # 存在意義が爆発的に高まる
    ("Rapidus",       "total_war"):         (4, 6, 10),
    ("Tokyo_Gas",     "short_conflict"):    (8, 5, 7),
    ("Tokyo_Gas",     "taiwan_occupation"): (6, 6, 9),  # エネルギー安全保障の主役
    ("Tokyo_Gas",     "total_war"):         (5, 7, 10),
}

# ════════════════════════════════════════════════════════════════════════════
# 7. 可視化
# ════════════════════════════════════════════════════════════════════════════

def plot_us_china_war_model():
    fig = plt.figure(figsize=(22, 26), facecolor="#0a0a0a")
    fig.suptitle(
        "米中大戦 × 日本半導体サプライチェーン崩壊モデル",
        fontsize=20, color="white", fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.52,
        wspace=0.38,
        left=0.06, right=0.97,
        top=0.94, bottom=0.04,
    )

    ax_dep   = fig.add_subplot(gs[0, :2])   # 依存度バー
    ax_stock = fig.add_subplot(gs[0, 2])    # 備蓄日数
    ax_disco = fig.add_subplot(gs[1, :2])   # Disco売上軌跡
    ax_gdp   = fig.add_subplot(gs[1, 2])    # GDP損失
    ax_nx    = fig.add_subplot(gs[2, :])    # NVIDIA JP × 地政学
    ax_career = fig.add_subplot(gs[3, :2])  # キャリアヒートマップ
    ax_chain  = fig.add_subplot(gs[3, 2])   # 崩壊連鎖フロー

    _style_ax(ax_dep)
    _style_ax(ax_stock)
    _style_ax(ax_disco)
    _style_ax(ax_gdp)
    _style_ax(ax_nx)
    _style_ax(ax_career)
    _style_ax(ax_chain)

    _plot_dependencies(ax_dep)
    _plot_stockpile(ax_stock)
    _plot_disco(ax_disco)
    _plot_gdp(ax_gdp)
    _plot_nvidia_jp(ax_nx)
    _plot_career_heatmap(ax_career)
    _plot_chain(ax_chain)

    plt.savefig(
        os.path.join(OUT_DIR, "us_china_war_model.png"),
        dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)
    print("Saved: results/us_china_war_model.png")


def _style_ax(ax):
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")


def _plot_dependencies(ax):
    ax.set_title("日本の戦略物資依存度 (中国 + 台湾/朝鮮半島)", color="white", fontsize=10, pad=6)
    keys = list(STRATEGIC_DEPENDENCIES.keys())
    labels = [STRATEGIC_DEPENDENCIES[k]["label"] for k in keys]
    china  = [STRATEGIC_DEPENDENCIES[k]["china_share"] * 100 for k in keys]
    taiwan = [STRATEGIC_DEPENDENCIES[k].get("taiwan_share", 0) * 100 for k in keys]
    korea  = [STRATEGIC_DEPENDENCIES[k].get("korea_share", 0) * 100 for k in keys]
    me     = [STRATEGIC_DEPENDENCIES[k].get("middle_east_share", 0) * 100 for k in keys]

    y = np.arange(len(keys))
    h = 0.55
    ax.barh(y, china, h, color="#d62728", alpha=0.9, label="中国")
    ax.barh(y, taiwan, h, left=china, color="#e377c2", alpha=0.9, label="台湾")
    ax.barh(y, korea, h, left=[c+t for c,t in zip(china,taiwan)],
            color="#17becf", alpha=0.9, label="韓国")
    ax.barh(y, me, h,
            left=[c+t+k for c,t,k in zip(china,taiwan,korea)],
            color="#ff7f0e", alpha=0.9, label="中東")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="white", fontsize=7.5)
    ax.set_xlabel("依存度 (%)", color="white", fontsize=8)
    ax.axvline(80, color="#ff4444", lw=1.2, ls="--", alpha=0.7)
    ax.text(81, len(keys)-0.5, "危険域 80%", color="#ff4444", fontsize=7)
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right", fontsize=7, facecolor="#222", edgecolor="#555",
              labelcolor="white", ncol=2)


def _plot_stockpile(ax):
    ax.set_title("有事: 備蓄枯渇までの日数", color="white", fontsize=10, pad=6)
    keys = ["Gallium", "LNG", "SiC_Substrate", "HBM_Memory", "TSMC_Logic", "REE_Neodymium", "Crude_Oil"]
    labels = ["ガリウム", "LNG", "SiC基板", "HBMメモリ", "TSMC先端品", "ネオジム", "原油"]
    days = [STRATEGIC_DEPENDENCIES[k]["japan_stock_days"] for k in keys]
    colors = [STRATEGIC_DEPENDENCIES[k]["color"] for k in keys]

    bars = ax.barh(labels, days, color=colors, alpha=0.85)
    for bar, d in zip(bars, days):
        ax.text(d + 3, bar.get_y() + bar.get_height()/2,
                f"{d}日", color="white", va="center", fontsize=7.5)
    ax.axvline(30, color="#ff4444", lw=1, ls="--", alpha=0.7)
    ax.axvline(90, color="#ff8800", lw=1, ls="--", alpha=0.7)
    ax.text(31, 6.5, "30日\n(臨界)", color="#ff4444", fontsize=6.5)
    ax.text(91, 6.5, "90日\n(IEA基準)", color="#ff8800", fontsize=6.5)
    ax.set_xlabel("日数", color="white", fontsize=8)
    ax.set_xlim(0, 250)


def _plot_disco(ax):
    ax.set_title("Disco 売上シミュレーション — 有事シナリオ別 (FY2024=1.0)", color="white", fontsize=10, pad=6)
    years = np.linspace(-0.5, 8, 300)
    for sc in SCENARIOS.values():
        rev = disco_revenue_path(sc, years)
        ax.plot(years, rev * DISCO_FY2024["total_revenue_B_JPY"],
                color=sc.color, lw=2.2, label=sc.label.replace("\n", " "))

    ax.axhline(DISCO_FY2024["total_revenue_B_JPY"], color="#aaaaaa", lw=1, ls=":", alpha=0.6)
    ax.text(0.1, DISCO_FY2024["total_revenue_B_JPY"] + 4, "FY2024 ¥393B", color="#aaaaaa", fontsize=7)
    ax.axvline(0, color="#666666", lw=1, ls="--")
    ax.text(0.1, 30, "← 有事勃発", color="#888888", fontsize=7)
    ax.set_xlabel("有事後 経過年数", color="white", fontsize=8)
    ax.set_ylabel("Disco 売上 (¥B/yr)", color="white", fontsize=8)
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 430)
    ax.legend(loc="upper right", fontsize=7.5, facecolor="#222", edgecolor="#555",
              labelcolor="white")


def _plot_gdp(ax):
    ax.set_title("日本GDP損失 (兆円)", color="white", fontsize=10, pad=6)
    scenarios_list = list(SCENARIOS.values())
    x = np.arange(3)
    labels = ["1年後", "3年後", "5年後"]
    w = 0.22

    for i, sc in enumerate(scenarios_list):
        losses = gdp_loss_trillion(sc)
        vals = [losses["yr1"], losses["yr3"], losses["yr5"]]
        bars = ax.bar(x + i * w, vals, w, color=sc.color, alpha=0.85,
                      label=sc.label.replace("\n", " "))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                    f"¥{v:.0f}T", ha="center", va="bottom", color="white", fontsize=6.5)

    ax.set_xticks(x + w)
    ax.set_xticklabels(labels, color="white", fontsize=8)
    ax.set_ylabel("GDP損失 (兆円)", color="white", fontsize=8)
    ax.legend(fontsize=6.5, facecolor="#222", edgecolor="#555", labelcolor="white",
              loc="upper left")


def _plot_nvidia_jp(ax):
    ax.set_title(
        "NVIDIA Japan — 地政学的ポジション変化 (有事後3年)",
        color="white", fontsize=10, pad=6
    )

    categories = [
        "GPU輸出規制\nリスク",
        "軍事AI\n需要",
        "日本政府\n連携",
        "在日米軍\nインフラ",
        "民間AI\n事業継続",
    ]
    # 平時 vs 各シナリオ (0-10スケール)
    peacetime      = [2, 4, 3, 2, 9]
    short_conflict = [5, 7, 7, 6, 7]
    taiwan_occ     = [7, 9, 9, 9, 5]
    total_war      = [9, 6, 8, 10, 2]

    x = np.arange(len(categories))
    w = 0.18
    for i, (vals, label, color) in enumerate([
        (peacetime,      "平時",                      "#888888"),
        (short_conflict, "短期紛争",                  "#2ca02c"),
        (taiwan_occ,     "台湾占領",                  "#ff7f0e"),
        (total_war,      "全面戦争",                  "#d62728"),
    ]):
        ax.bar(x + i * w, vals, w, color=color, alpha=0.85, label=label)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(categories, color="white", fontsize=8)
    ax.set_ylabel("スコア (0-10)", color="white", fontsize=8)
    ax.set_ylim(0, 11.5)
    ax.legend(loc="upper right", fontsize=8, facecolor="#222", edgecolor="#555",
              labelcolor="white")

    note = (
        "注: 全面戦争では輸出規制でGPU供給停止、しかし在日米軍インフラとしての重要性は最大\n"
        "    → NVIDIA JPは「外資IT企業」から「米国軍事ロジスティクス拠点」に変貌する"
    )
    ax.text(0.01, 0.02, note, transform=ax.transAxes,
            color="#aaaaaa", fontsize=7, va="bottom")


def _plot_career_heatmap(ax):
    ax.set_title("キャリア × 有事シナリオ 総合スコアマトリクス", color="white", fontsize=10, pad=6)

    companies  = ["Disco", "Lam_AMAT", "NVIDIA_JP", "Rapidus", "Tokyo_Gas"]
    scenarios_keys = ["short_conflict", "taiwan_occupation", "total_war"]
    scenario_labels = ["短期紛争", "台湾占領", "全面戦争"]
    company_labels  = ["Disco", "Lam/AMAT", "NVIDIA JP", "Rapidus", "東京ガス"]

    # 総合スコア = 安定度×0.3 + 成長性×0.4 + 社会的意義×0.3
    matrix = np.zeros((len(companies), len(scenarios_keys)))
    for i, co in enumerate(companies):
        for j, sc in enumerate(scenarios_keys):
            stab, growth, sig = CAREER_WAR_MATRIX[(co, sc)]
            matrix[i, j] = stab * 0.3 + growth * 0.4 + sig * 0.3

    cmap = LinearSegmentedColormap.from_list("war", ["#330000", "#ff7f0e", "#00cc44"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=10)

    ax.set_xticks(range(len(scenarios_keys)))
    ax.set_xticklabels(scenario_labels, color="white", fontsize=9)
    ax.set_yticks(range(len(companies)))
    ax.set_yticklabels(company_labels, color="white", fontsize=9)

    for i in range(len(companies)):
        for j in range(len(scenarios_keys)):
            stab, growth, sig = CAREER_WAR_MATRIX[(companies[i], scenarios_keys[j])]
            score = matrix[i, j]
            ax.text(j, i, f"{score:.1f}\n(安{stab} 成{growth} 義{sig})",
                    ha="center", va="center", color="white", fontsize=7, fontweight="bold")

    plt.colorbar(im, ax=ax, label="総合スコア", fraction=0.03, pad=0.04).ax.yaxis.label.set_color("white")
    ax.set_xlabel("シナリオ", color="white", fontsize=8)
    ax.set_ylabel("転職先候補", color="white", fontsize=8)


def _plot_chain(ax):
    ax.set_title("崩壊連鎖フロー", color="white", fontsize=10, pad=6)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    chain = [
        (5, 9.2,  "台湾海峡\n軍事衝突",   "#d62728", 14),
        (5, 7.5,  "TSMC停止\n(先端ロジック消滅)", "#e45756", 11),
        (2, 5.8,  "中国禁輸\nGa/Ge/REE",  "#ff7f0e", 10),
        (8, 5.8,  "シーレーン封鎖\nLNG/原油",      "#ff8c00", 10),
        (5, 4.2,  "製造業停止\n& エネルギー危機",  "#ffbb00", 10),
        (2, 2.6,  "Disco\n受注崩壊",      "#aaaaaa", 9),
        (8, 2.6,  "NVIDIA JP\n輸出管理下",         "#9ecae1", 9),
        (5, 1.1,  "円暴落・\nGDP -12〜45%",        "#ff4444", 12),
    ]

    for i, (x, y, label, color, fs) in enumerate(chain):
        ax.text(x, y, label, ha="center", va="center", fontsize=fs * 0.6,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85, edgecolor="#ffffff44"))
        if i < len(chain) - 1:
            nx, ny = chain[i+1][0], chain[i+1][1]
            if i == 1:
                # 2股に分岐
                for bx in [2, 8]:
                    ax.annotate("", xy=(bx, 6.2), xytext=(x, y - 0.4),
                                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2))
                continue
            if i in [2, 3]:
                ax.annotate("", xy=(5, 4.6), xytext=(x, y - 0.4),
                            arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2))
                continue
            if i == 4:
                for bx in [2, 8]:
                    ax.annotate("", xy=(bx, 3.0), xytext=(x, y - 0.4),
                                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2))
                continue
            if i in [5, 6]:
                ax.annotate("", xy=(5, 1.5), xytext=(x, y - 0.4),
                            arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2))
                continue
            ax.annotate("", xy=(nx, ny + 0.5), xytext=(x, y - 0.4),
                        arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2))


# ════════════════════════════════════════════════════════════════════════════
# 8. サマリーレポート
# ════════════════════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "="*70)
    print("  米中大戦 × 日本半導体インパクト サマリー")
    print("="*70)

    print("\n【戦略物資 禁輸即効性ランキング】")
    items = [
        (k, STRATEGIC_DEPENDENCIES[k]["label"], STRATEGIC_DEPENDENCIES[k]["japan_stock_days"])
        for k in STRATEGIC_DEPENDENCIES
    ]
    for k, label, days in sorted(items, key=lambda x: x[2]):
        bar = "■" * int(days / 10) + "□" * (20 - int(days / 10))
        print(f"  {days:3d}日  {bar}  {label}")

    print("\n【シナリオ別 日本GDP損失試算】")
    for sc in SCENARIOS.values():
        losses = gdp_loss_trillion(sc)
        print(f"\n  [{sc.label.replace(chr(10), ' ')}]")
        print(f"    1年後: ▼¥{losses['yr1']:.0f}兆  "
              f"3年後: ▼¥{losses['yr3']:.0f}兆  "
              f"5年後: ▼¥{losses['yr5']:.0f}兆")
        print(f"    Disco売上 1年後: {sc.disco_revenue_factor_yr1*100:.0f}%  "
              f"3年後: {sc.disco_revenue_factor_yr3*100:.0f}%")
        print(f"    NVIDIA JP 3年後事業規模: {sc.nvidia_jp_factor_yr3:.1f}x  "
              f"防衛費GDP比→{sc.defense_gdp_pct_yr5}%")

    print("\n【キャリア推奨 (総合スコア上位)】")
    scored = []
    for (co, sc), (stab, growth, sig) in CAREER_WAR_MATRIX.items():
        score = stab * 0.3 + growth * 0.4 + sig * 0.3
        scored.append((score, co, sc, stab, growth, sig))
    for score, co, sc, stab, growth, sig in sorted(scored, reverse=True)[:6]:
        print(f"  {score:.1f}  {co:12s}  x {sc:20s}  "
              f"(安{stab} 成{growth} 義{sig})")
    print()


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_summary()
    plot_us_china_war_model()
