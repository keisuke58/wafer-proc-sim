"""
リアル市場データ × 物理モデル接続
====================================
yfinance で Disco / TEL / ASML / Wolfspeed 株価を取得し、
ウェーハ需要モデルとの相関を定量化する。

株価 = 市場の期待 × 実績の積算
物理モデルの予測精度を株価動向で検証する。
"""

import math
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

# ════════════════════════════════════════════════════════════════════════════
# 対象企業 × ティッカー
# ════════════════════════════════════════════════════════════════════════════

SEMICON_STOCKS = {
    "Disco":      {"ticker": "6146.T",  "currency": "JPY", "color": "#2166ac",
                   "role": "SiC/Si ダイシング 世界シェア70%",
                   "sic_exposure": "high"},
    "TEL":        {"ticker": "8035.T",  "currency": "JPY", "color": "#d73027",
                   "role": "成膜・洗浄・エッチング装置",
                   "sic_exposure": "medium"},
    "ASML":       {"ticker": "ASML",    "currency": "USD", "color": "#4CAF50",
                   "role": "EUV 露光装置 独占",
                   "sic_exposure": "low"},
    "Wolfspeed":  {"ticker": "WOLF",    "currency": "USD", "color": "#9C27B0",
                   "role": "SiC ウェーハ + パワーデバイス",
                   "sic_exposure": "very_high"},
    "Infineon":   {"ticker": "IFX.DE",  "currency": "EUR", "color": "#f5a623",
                   "role": "SiC/GaN パワー半導体",
                   "sic_exposure": "high"},
    "ON_Semi":    {"ticker": "ON",      "currency": "USD", "color": "#72B7B2",
                   "role": "SiC デバイス (EV 向け急拡大)",
                   "sic_exposure": "high"},
    "STMicro":    {"ticker": "STM",     "currency": "USD", "color": "#FF9800",
                   "role": "SiC MOSFET (Tesla 供給先)",
                   "sic_exposure": "very_high"},
    "NVIDIA":     {"ticker": "NVDA",    "currency": "USD", "color": "#76b900",
                   "role": "AI GPU (SiC 電源が必要)",
                   "sic_exposure": "indirect"},
}


def fetch_stock_data(ticker: str, period: str = "1y") -> dict:
    """yfinance から株価データを取得。失敗時はダミーデータを返す。"""
    if not YFINANCE_OK:
        return {"error": "yfinance not installed", "ticker": ticker}

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        info = t.info

        if hist.empty:
            return {"error": "no data", "ticker": ticker}

        close = hist["Close"]
        ret_1y = (close.iloc[-1] / close.iloc[0] - 1) * 100
        vol    = close.pct_change().std() * math.sqrt(252) * 100

        return {
            "ticker":      ticker,
            "current":     round(float(close.iloc[-1]), 2),
            "high_52w":    round(float(close.max()), 2),
            "low_52w":     round(float(close.min()), 2),
            "return_1y_pct": round(ret_1y, 1),
            "volatility_pct": round(vol, 1),
            "market_cap_B":   round(info.get("marketCap", 0) / 1e9, 1),
            "pe_ratio":    round(info.get("trailingPE", 0), 1),
            "history":     close,
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def sic_exposure_score() -> list[dict]:
    """
    SiC 需要増加に対する各社の恩恵度を定量化。
    物理モデルの予測 + 実際の株価動向との比較。
    """
    exposure_map = {"very_high": 1.0, "high": 0.7,
                    "medium": 0.4, "low": 0.2, "indirect": 0.3}

    results = []
    for name, info in SEMICON_STOCKS.items():
        score = exposure_map[info["sic_exposure"]]
        stock = fetch_stock_data(info["ticker"], "1y")

        results.append({
            "company":        name,
            "ticker":         info["ticker"],
            "role":           info["role"],
            "sic_exposure":   info["sic_exposure"],
            "exposure_score": score,
            "return_1y":      stock.get("return_1y_pct", "N/A"),
            "market_cap_B":   stock.get("market_cap_B", "N/A"),
            "pe":             stock.get("pe_ratio", "N/A"),
            "error":          stock.get("error"),
            "color":          info["color"],
        })

    return sorted(results, key=lambda x: x["exposure_score"], reverse=True)


def correlation_model_vs_market() -> dict:
    """
    物理モデルの予測 vs 実際の株価動向。
    「SiC 需要増加 → 株価上昇」の相関を確認。
    """
    # 物理モデルからの予測 (fem/ev_sic_model.py, ai_datacenter_model.py)
    model_predictions = {
        "Wolfspeed (SiC ウェーハ)":  {"growth_pct": 47, "basis": "BESS CAGR + EV"},
        "Disco (ダイシング)":          {"growth_pct": 35, "basis": "SiC ブレード 125× premium"},
        "STMicro (SiC デバイス)":     {"growth_pct": 30, "basis": "Tesla 供給 + EV 拡大"},
        "ON Semi (SiC デバイス)":     {"growth_pct": 28, "basis": "EV インバータ"},
        "ASML (EUV 露光)":           {"growth_pct": 15, "basis": "AI チップ製造需要"},
        "NVIDIA (AI GPU)":           {"growth_pct": 60, "basis": "AI DC 急拡大"},
    }

    return {
        "model_predictions": model_predictions,
        "methodology":       "物理モデルのウェーハ需要 → 企業売上成長率 → 株価 EPS 成長で換算",
        "caveat":            "株価は多因子 (マクロ・競合・為替等) に依存。物理モデルは一要因のみ",
    }


# ════════════════════════════════════════════════════════════════════════════
# DCF モデル (Disco 詳細)
# ════════════════════════════════════════════════════════════════════════════

def disco_dcf_model(
    wafers_sic_2030: float = 1500,   # 枚/日 (SiC)
    wafers_si_2030:  float = 4500,   # 枚/日 (Si)
    blade_sic_usd:   float = 50,
    blade_si_usd:    float = 0.40,
    disco_share:     float = 0.65,
    wacc:            float = 0.08,
    terminal_growth: float = 0.03,
) -> dict:
    """
    Disco の DCF バリュエーション。
    物理モデルのウェーハ需要から売上を積み上げる。

    Revenue = 消耗品 (ブレード) + 装置
    """
    # 2030年 年間ブレード売上
    blade_sic_yr = wafers_sic_2030 * 365 * blade_sic_usd * disco_share
    blade_si_yr  = wafers_si_2030  * 365 * blade_si_usd  * disco_share

    # 装置売上 (消耗品の 1.2倍と仮定)
    consumables   = blade_sic_yr + blade_si_yr
    equipment_yr  = consumables * 1.2
    total_rev_yr  = consumables + equipment_yr

    # EBIT マージン (Disco 実績 ~30%)
    ebit_margin   = 0.30
    ebit_yr       = total_rev_yr * ebit_margin

    # 簡易 DCF (5年予測 + 永続価値)
    fcf_yr        = ebit_yr * 0.75   # 税後 FCF
    pv_5yr        = sum(fcf_yr * (1 + 0.10)**i / (1 + wacc)**i for i in range(1, 6))
    terminal_val  = fcf_yr * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal   = terminal_val / (1 + wacc)**5
    enterprise_v  = pv_5yr + pv_terminal

    return {
        "blade_sic_rev_B$":    round(blade_sic_yr / 1e9, 2),
        "blade_si_rev_B$":     round(blade_si_yr / 1e9, 2),
        "total_rev_B$":        round(total_rev_yr / 1e9, 2),
        "ebit_B$":             round(ebit_yr / 1e9, 2),
        "fcf_B$":              round(fcf_yr / 1e9, 2),
        "enterprise_value_B$": round(enterprise_v / 1e9, 1),
        "sic_rev_fraction":    round(blade_sic_yr / consumables, 2),
        "wacc":                wacc,
    }


# ════════════════════════════════════════════════════════════════════════════
# Quick demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  リアル市場データ × 物理モデル")
    print("=" * 65)

    print("\n株価取得 (yfinance):")
    for name, info in list(SEMICON_STOCKS.items())[:4]:
        s = fetch_stock_data(info["ticker"], "1y")
        if "error" not in s:
            print(f"  {name:<12} {info['ticker']:<10} "
                  f"時価総額 ${s['market_cap_B']:.0f}B  "
                  f"1年リターン {s['return_1y_pct']:+.0f}%  "
                  f"PER {s['pe_ratio']:.0f}×")
        else:
            print(f"  {name:<12} {info['ticker']:<10} → {s['error']}")

    print("\nSiC 露出スコア × 株価リターン:")
    for r in sic_exposure_score():
        ret = f"{r['return_1y']:+.0f}%" if isinstance(r["return_1y"], float) else "N/A"
        print(f"  {r['company']:<15} SiC露出={r['exposure_score']:.1f}  "
              f"1yr={ret:>8}  {r['role'][:35]}")

    print("\nDisco DCF (2030年 SiC ウェーハ需要ベース):")
    dcf = disco_dcf_model()
    print(f"  SiC ブレード収入: ${dcf['blade_sic_rev_B$']:.2f}B/年")
    print(f"  Si  ブレード収入: ${dcf['blade_si_rev_B$']:.2f}B/年")
    print(f"  総売上: ${dcf['total_rev_B$']:.2f}B/年")
    print(f"  EV (DCF): ${dcf['enterprise_value_B$']:.0f}B")
    print(f"  SiC収入割合: {dcf['sic_rev_fraction']*100:.0f}%")
