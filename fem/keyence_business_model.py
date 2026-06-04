"""
Keyence Corporation — Business Model Simulation
================================================
Competitor/partner analysis within the wafer-proc-sim semiconductor
equipment portfolio.  All monetary values are stored in JPY internally;
USD display uses a fixed rate of 150 JPY/USD (FY2024 average).

Key public facts (FY2024, Keyence IR):
  Revenue:          ¥1,004 B  (CAGR ~12% last decade)
  Operating margin: 55.1%     (highest among Japanese manufacturers)
  Direct sales:     ~4,000 sales engineers, 0 distributors
  Product ASP:      ¥0.5 M – ¥30 M per unit
  R&D ratio:        ~6% of revenue
  SGA ratio:        ~17%
  Semi/electronics: ~35% of revenue
  Business model:   "Kaizen proposal" — sell productivity improvement, not specs

Modules
-------
1. KeyenceFinancials  — dataclass, FY2021–2024 annual data
2. direct_sales_model — margin comparison: direct vs distributor
3. value_pricing_model — cost + alpha × customer_ROI
4. product_margin_breakdown — per-SKU BOM/ASP/gross-margin
5. market_share_simulation — 2024→2034 semi metrology share
6. plot_main — 4-panel PNG saved to results/
7. main — argparse entry-point (--no-plot)

References
----------
Keyence Annual Report FY2024 (public IR)
Keyence VK-X3100 spec sheet (2024)
keyence_metrology_model.py — VK-X ROI: $40.6 M net 5yr benefit

Usage
-----
    python fem/keyence_business_model.py
    python fem/keyence_business_model.py --no-plot
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

USD_JPY = 150.0          # fixed display conversion rate (FY2024 avg)
B = 1e9                  # billion
M = 1e6                  # million

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ─────────────────────────────────────────────────────────────────────────────
# 1. KeyenceFinancials — annual data dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnnualRecord:
    """Single fiscal-year financial record (JPY)."""
    fy: int
    revenue_jpy: float          # total revenue [JPY]
    op_margin: float            # operating profit margin [0–1]
    fcf_margin: float           # free cash flow margin [0–1]
    rd_ratio: float             # R&D spend / revenue [0–1]
    sga_ratio: float            # SGA / revenue [0–1]
    semi_share: float           # semi+electronics as fraction of revenue [0–1]
    n_employees: int            # headcount (approx.)

    @property
    def op_profit_jpy(self) -> float:
        return self.revenue_jpy * self.op_margin

    @property
    def fcf_jpy(self) -> float:
        return self.revenue_jpy * self.fcf_margin

    @property
    def revenue_usd(self) -> float:
        return self.revenue_jpy / USD_JPY

    @property
    def op_profit_usd(self) -> float:
        return self.op_profit_jpy / USD_JPY


@dataclass
class KeyenceFinancials:
    """
    Keyence Corporation FY2021–2024 consolidated financials.

    Data sourced from Keyence annual reports and IR presentations.
    FY ends March 31 of the stated year.
    """
    records: List[AnnualRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.records:
            self.records = _default_financials()

    def get_fy(self, fy: int) -> AnnualRecord:
        for r in self.records:
            if r.fy == fy:
                return r
        raise KeyError(f"FY{fy} not found in KeyenceFinancials")

    def revenue_series(self) -> Tuple[List[int], List[float]]:
        fys = [r.fy for r in self.records]
        revs = [r.revenue_jpy / B for r in self.records]   # ¥B
        return fys, revs

    def op_margin_series(self) -> Tuple[List[int], List[float]]:
        fys = [r.fy for r in self.records]
        margins = [r.op_margin * 100 for r in self.records]  # %
        return fys, margins

    def cagr(self) -> float:
        """Revenue CAGR over the stored period."""
        r0 = self.records[0].revenue_jpy
        r1 = self.records[-1].revenue_jpy
        n = self.records[-1].fy - self.records[0].fy
        return (r1 / r0) ** (1.0 / n) - 1.0 if n > 0 else 0.0


def _default_financials() -> List[AnnualRecord]:
    """
    Keyence annual records FY2019–2024.
    Revenue and margins from public annual reports; employee count approximate.
    FY2019 included for trend chart baseline.
    """
    return [
        AnnualRecord(fy=2019, revenue_jpy=517.9 * B, op_margin=0.540,
                     fcf_margin=0.420, rd_ratio=0.060, sga_ratio=0.175,
                     semi_share=0.34, n_employees=8_500),
        AnnualRecord(fy=2020, revenue_jpy=551.2 * B, op_margin=0.542,
                     fcf_margin=0.422, rd_ratio=0.060, sga_ratio=0.173,
                     semi_share=0.34, n_employees=8_700),
        AnnualRecord(fy=2021, revenue_jpy=675.4 * B, op_margin=0.550,
                     fcf_margin=0.438, rd_ratio=0.059, sga_ratio=0.172,
                     semi_share=0.35, n_employees=9_100),
        AnnualRecord(fy=2022, revenue_jpy=902.0 * B, op_margin=0.556,
                     fcf_margin=0.448, rd_ratio=0.058, sga_ratio=0.170,
                     semi_share=0.36, n_employees=9_700),
        AnnualRecord(fy=2023, revenue_jpy=938.0 * B, op_margin=0.553,
                     fcf_margin=0.445, rd_ratio=0.059, sga_ratio=0.171,
                     semi_share=0.35, n_employees=10_100),
        AnnualRecord(fy=2024, revenue_jpy=1_004.0 * B, op_margin=0.551,
                     fcf_margin=0.440, rd_ratio=0.060, sga_ratio=0.170,
                     semi_share=0.35, n_employees=10_600),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. direct_sales_model
# ─────────────────────────────────────────────────────────────────────────────

def direct_sales_model(
    n_engineers: int = 4_000,
    revenue_per_eng_M_jpy: float = 251.0,  # ¥1,004 B / 4,000 ≈ ¥251 M/eng
    conversion_rate: float = 0.35,
) -> Dict[str, object]:
    """
    Simulate Keyence's direct-sales engine and compare with a distributor model.

    Keyence sales engineers make technical "Kaizen proposal" visits: they
    demonstrate ROI to the customer, which justifies a significant ASP premium
    over catalog prices.  Because engineers ARE salespeople, SGA/revenue stays
    low (~17%) despite the large direct force.

    Parameters
    ----------
    n_engineers : int
        Number of Keyence direct sales engineers globally.
    revenue_per_eng_M_jpy : float
        Revenue attributed per sales engineer [¥M].
        Default ≈ ¥1,004 B / 4,000 = ¥251 M per engineer.
    conversion_rate : float
        Fraction of customer visits that result in an order.

    Returns
    -------
    dict with keys:
        total_revenue_jpy, revenue_per_eng_jpy,
        keyence_op_margin, distributor_op_margin,
        asp_premium_pct, margin_diff_pp,
        waterfall : list of (label, value_jpy) for margin waterfall
    """
    total_revenue_jpy = n_engineers * revenue_per_eng_M_jpy * M

    # Keyence direct model cost structure (% of revenue, FY2024 actual)
    keyence_cogs_pct    = 0.240   # 24% COGS  (high-margin hw + sw)
    keyence_rd_pct      = 0.060   # 6%  R&D
    keyence_sga_pct     = 0.170   # 17% SGA (incl. engineers' salaries)
    keyence_op_margin   = 1.0 - keyence_cogs_pct - keyence_rd_pct - keyence_sga_pct

    # Typical Japanese industrial distributor model
    dist_cogs_pct       = 0.450   # 45% — distributor takes hardware at cost
    dist_channel_pct    = 0.150   # 15% — distributor margin
    dist_rd_pct         = 0.060   # 6%  — same R&D assumption
    dist_sga_pct        = 0.100   # 10% — lower internal SGA (no direct force)
    dist_op_margin      = 1.0 - dist_cogs_pct - dist_channel_pct - dist_rd_pct - dist_sga_pct

    # ASP premium: direct channel removes ~¥ distributor margin, so Keyence
    # can price higher and still be cost-competitive for the customer
    asp_premium_pct = (dist_channel_pct / (1 - dist_channel_pct)) * 100  # ~17.6%

    # Waterfall: direct model (revenue = 100%)
    waterfall = [
        ("Revenue",               total_revenue_jpy),
        ("- COGS",                -keyence_cogs_pct     * total_revenue_jpy),
        ("- R&D",                 -keyence_rd_pct       * total_revenue_jpy),
        ("- SGA (direct force)",  -keyence_sga_pct      * total_revenue_jpy),
        ("= Operating Profit",     keyence_op_margin    * total_revenue_jpy),
    ]

    waterfall_dist = [
        ("Revenue",               total_revenue_jpy),
        ("- COGS",                -dist_cogs_pct        * total_revenue_jpy),
        ("- Channel margin",      -dist_channel_pct     * total_revenue_jpy),
        ("- R&D",                 -dist_rd_pct          * total_revenue_jpy),
        ("- SGA (indirect)",      -dist_sga_pct         * total_revenue_jpy),
        ("= Operating Profit",     dist_op_margin       * total_revenue_jpy),
    ]

    return {
        "total_revenue_jpy":      total_revenue_jpy,
        "revenue_per_eng_jpy":    revenue_per_eng_M_jpy * M,
        "n_engineers":            n_engineers,
        "conversion_rate":        conversion_rate,
        "keyence_op_margin":      keyence_op_margin,
        "distributor_op_margin":  dist_op_margin,
        "asp_premium_pct":        asp_premium_pct,
        "margin_diff_pp":         (keyence_op_margin - dist_op_margin) * 100,
        "waterfall_direct":       waterfall,
        "waterfall_dist":         waterfall_dist,
        # cost structure breakdowns [fraction of revenue]
        "keyence_breakdown": {
            "COGS":  keyence_cogs_pct,
            "R&D":   keyence_rd_pct,
            "SGA":   keyence_sga_pct,
            "OP":    keyence_op_margin,
        },
        "dist_breakdown": {
            "COGS":    dist_cogs_pct,
            "Channel": dist_channel_pct,
            "R&D":     dist_rd_pct,
            "SGA":     dist_sga_pct,
            "OP":      dist_op_margin,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. value_pricing_model
# ─────────────────────────────────────────────────────────────────────────────

def value_pricing_model(
    cost_to_make_jpy: float,
    customer_roi_5yr_jpy: float,
    alpha: float = 0.30,
) -> float:
    """
    Value-based pricing: Price = cost_to_make + alpha * customer_ROI_5yr.

    Keyence captures a fraction (alpha) of the 5-year customer value it creates,
    rather than applying a simple cost-plus mark-up.  The remaining value
    (1 - alpha) is the customer's incentive to buy.

    Parameters
    ----------
    cost_to_make_jpy : float
        Bill-of-materials (BOM) + manufacturing cost [JPY].
    customer_roi_5yr_jpy : float
        Present value of total customer benefit over 5 years [JPY].
        For the VK-X, keyence_metrology_model.py computes $40.6 M net
        benefit ≈ ¥6,090 M (at 150 JPY/USD).
    alpha : float
        Fraction of customer value that Keyence captures in the price.
        Empirically ~0.25–0.35 for premium metrology tools.

    Returns
    -------
    float
        Justified product price [JPY].

    Examples
    --------
    >>> # VK-X: BOM ¥4M, customer ROI $40.6M → ¥6,090M
    >>> p = value_pricing_model(4e6, 40.6e6 * 150, alpha=0.30)
    >>> print(f"VK-X justified price: ¥{p/1e6:.1f}M")   # ≈ ¥1,831 M  (gross)
    """
    if alpha < 0 or alpha > 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    price = cost_to_make_jpy + alpha * customer_roi_5yr_jpy
    return price


def value_pricing_analysis(
    products: List[Dict] | None = None,
    alpha: float = 0.30,
) -> List[Dict]:
    """
    Run value_pricing_model for a standard product list and return results.

    Parameters
    ----------
    products : list of dict, optional
        Each dict must contain: name, cost_jpy, customer_roi_5yr_jpy, asp_actual_jpy.
        If None, defaults for VK-X, LJ-X, IV3 are used.
    alpha : float
        Value-capture fraction passed to value_pricing_model.

    Returns
    -------
    list of dict — one per product, with pricing breakdown.
    """
    if products is None:
        # VK-X customer ROI from keyence_metrology_model: net_5yr ≈ $40.6M
        vkx_roi_5yr_jpy = 40.6e6 * USD_JPY   # ¥6,090 M

        # LJ-X 8000 2D/3D laser profiler — semiconductor backgrinding inline
        ljx_roi_5yr_jpy = 18.0e6 * USD_JPY   # ¥2,700 M (estimated)

        # IV3 series vision system — die bond inspection
        iv3_roi_5yr_jpy = 8.0e6 * USD_JPY    # ¥1,200 M (estimated)

        products = [
            {
                "name":               "VK-X (confocal laser microscope)",
                "cost_jpy":           4.0 * M,   # BOM + mfg ≈ ¥4 M
                "customer_roi_5yr_jpy": vkx_roi_5yr_jpy,
                "asp_actual_jpy":     25.0 * M,  # ¥25 M list price
            },
            {
                "name":               "LJ-X (2D/3D laser profiler)",
                "cost_jpy":           1.5 * M,   # BOM ≈ ¥1.5 M
                "customer_roi_5yr_jpy": ljx_roi_5yr_jpy,
                "asp_actual_jpy":     8.0 * M,   # ¥8 M (range ¥5–12 M)
            },
            {
                "name":               "IV3 (smart vision system)",
                "cost_jpy":           0.3 * M,   # BOM ≈ ¥0.3 M
                "customer_roi_5yr_jpy": iv3_roi_5yr_jpy,
                "asp_actual_jpy":     2.5 * M,   # ¥2.5 M (range ¥1–4 M)
            },
        ]

    results = []
    for p in products:
        price_calc = value_pricing_model(
            cost_to_make_jpy=p["cost_jpy"],
            customer_roi_5yr_jpy=p["customer_roi_5yr_jpy"],
            alpha=alpha,
        )
        value_per_cost = p["customer_roi_5yr_jpy"] / p["cost_jpy"]
        customer_value_kept = (1 - alpha) * p["customer_roi_5yr_jpy"]

        results.append({
            "name":                   p["name"],
            "cost_jpy":               p["cost_jpy"],
            "asp_actual_jpy":         p["asp_actual_jpy"],
            "customer_roi_5yr_jpy":   p["customer_roi_5yr_jpy"],
            "price_value_based_jpy":  price_calc,
            "alpha":                  alpha,
            "value_per_cost_ratio":   value_per_cost,
            "customer_value_kept_jpy": customer_value_kept,
            "keyence_value_captured_jpy": alpha * p["customer_roi_5yr_jpy"],
            "gross_margin_actual":    1.0 - p["cost_jpy"] / p["asp_actual_jpy"],
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. product_margin_breakdown
# ─────────────────────────────────────────────────────────────────────────────

# Product database: name -> {bom_jpy, asp_jpy, description}
PRODUCT_DB: Dict[str, Dict] = {
    "VK-X": {
        "bom_jpy":        4.0 * M,
        "asp_jpy":       25.0 * M,
        "opex_alloc_jpy": 1.5 * M,   # allocated R&D + SGA per unit
        "description":    "Confocal laser scanning microscope (405 nm, VK-X3100)",
        "segment":        "semiconductor/electronics",
        "category":       "metrology",
    },
    "LJ-X": {
        "bom_jpy":        1.5 * M,
        "asp_jpy":        8.0 * M,
        "opex_alloc_jpy": 0.7 * M,
        "description":    "2D/3D laser profiler for inline process control",
        "segment":        "semiconductor/electronics",
        "category":       "sensing",
    },
    "IV3": {
        "bom_jpy":        0.3 * M,
        "asp_jpy":        2.5 * M,
        "opex_alloc_jpy": 0.2 * M,
        "description":    "Smart vision system for die-bond & packaging inspection",
        "segment":        "semiconductor/electronics",
        "category":       "vision",
    },
}


def product_margin_breakdown(product: str) -> Dict[str, float]:
    """
    Estimate per-unit economics for a Keyence product.

    Parameters
    ----------
    product : str
        One of 'VK-X', 'LJ-X', 'IV3'.

    Returns
    -------
    dict with keys:
        product, description,
        bom_jpy, asp_jpy,
        gross_profit_jpy, gross_margin,
        opex_alloc_jpy,
        unit_op_profit_jpy, unit_op_margin,
        asp_to_bom_multiple
    """
    if product not in PRODUCT_DB:
        raise KeyError(
            f"Unknown product '{product}'. Known: {list(PRODUCT_DB.keys())}"
        )
    db = PRODUCT_DB[product]
    bom = db["bom_jpy"]
    asp = db["asp_jpy"]
    opex = db["opex_alloc_jpy"]

    gross_profit = asp - bom
    gross_margin = gross_profit / asp
    unit_op_profit = asp - bom - opex
    unit_op_margin = unit_op_profit / asp

    return {
        "product":             product,
        "description":         db["description"],
        "segment":             db["segment"],
        "category":            db["category"],
        "bom_jpy":             bom,
        "asp_jpy":             asp,
        "gross_profit_jpy":    gross_profit,
        "gross_margin":        gross_margin,           # fraction
        "gross_margin_pct":    gross_margin * 100,     # percent
        "opex_alloc_jpy":      opex,
        "unit_op_profit_jpy":  unit_op_profit,
        "unit_op_margin":      unit_op_margin,         # fraction
        "unit_op_margin_pct":  unit_op_margin * 100,   # percent
        "asp_to_bom_multiple": asp / bom,
    }


def all_product_margins() -> List[Dict]:
    """Return product_margin_breakdown for all known SKUs."""
    return [product_margin_breakdown(p) for p in PRODUCT_DB]


# ─────────────────────────────────────────────────────────────────────────────
# 5. market_share_simulation
# ─────────────────────────────────────────────────────────────────────────────

def market_share_simulation(
    years: int = 10,
    cagr_base: float = 0.08,
    keyence_growth_premium: float = 0.04,
) -> Dict[str, object]:
    """
    Simulate Keyence's semiconductor metrology market share 2024 → 2034.

    The semiconductor metrology/inspection market (excl. EUV lithography
    tools) grows at cagr_base.  Keyence grows at cagr_base +
    keyence_growth_premium due to its kaizen-proposal model, R&D intensity,
    and direct sales penetration.

    Competitors modelled:
        KLA Corporation      — leading metrology (process control)
        Hitachi High-Tech    — electron-beam & optical inspection
        Onto Innovation      — advanced packaging metrology
        Others               — commoditised sensors, vision cameras

    Parameters
    ----------
    years : int
        Number of years to simulate (starting 2024).
    cagr_base : float
        Total addressable market CAGR (fraction per year).
    keyence_growth_premium : float
        Keyence's extra growth rate above market CAGR.

    Returns
    -------
    dict with keys:
        year_range : list[int]
        tam_B_jpy  : list[float]   total market [¥B]
        keyence_rev_B_jpy   : list[float]
        kla_rev_B_jpy       : list[float]
        hitachi_rev_B_jpy   : list[float]
        onto_rev_B_jpy      : list[float]
        others_rev_B_jpy    : list[float]
        keyence_share_pct   : list[float]
        cagr_base, keyence_growth_premium, years
    """
    # Semiconductor metrology sub-market 2024 (Keyence semi segment share)
    # Keyence FY2024 semi+electronics revenue ≈ ¥1,004 B × 35% ≈ ¥351 B
    # But only metrology/sensing portion; estimate ¥120 B for comparability
    # with pure-play metrology vendors.
    #
    # Rough 2024 semiconductor metrology TAM (~¥3,000 B / $20 B globally):
    tam_2024_jpy = 3_000.0 * B   # ¥3,000 B

    # Initial share estimates (2024)
    shares_2024 = {
        "keyence":  0.040,   # ~4%  — smaller but fast-growing in semi
        "kla":      0.350,   # ~35% — dominant process control
        "hitachi":  0.120,   # ~12% — electron-beam inspection
        "onto":     0.080,   # ~8%  — advanced packaging
        "others":   0.410,   # ~41% — rest
    }

    # Individual CAGRs (fraction/yr) — relative to TAM growth
    cagr = {
        "keyence":  cagr_base + keyence_growth_premium,
        "kla":      cagr_base + 0.02,   # slight premium: EUV, DRAM scaling
        "hitachi":  cagr_base + 0.01,   # moderate: e-beam demand
        "onto":     cagr_base + 0.03,   # strong: advanced packaging boom
        "others":   cagr_base - 0.01,   # below market: commoditisation
    }

    year_range = list(range(2024, 2024 + years + 1))
    n = len(year_range)

    # Simulate revenues
    revs = {k: np.zeros(n) for k in shares_2024}
    tam  = np.zeros(n)

    tam[0] = tam_2024_jpy
    for k, s0 in shares_2024.items():
        revs[k][0] = tam_2024_jpy * s0

    for i in range(1, n):
        tam[i] = tam[i - 1] * (1 + cagr_base)
        for k in shares_2024:
            revs[k][i] = revs[k][i - 1] * (1 + cagr[k])

    # Normalise to TAM at each step (re-distribute rounding)
    total_rev = sum(revs[k] for k in revs)
    for k in revs:
        revs[k] = revs[k] / total_rev * tam

    keyence_share_pct = (revs["keyence"] / tam * 100).tolist()

    return {
        "year_range":           year_range,
        "tam_B_jpy":            (tam / B).tolist(),
        "keyence_rev_B_jpy":    (revs["keyence"] / B).tolist(),
        "kla_rev_B_jpy":        (revs["kla"] / B).tolist(),
        "hitachi_rev_B_jpy":    (revs["hitachi"] / B).tolist(),
        "onto_rev_B_jpy":       (revs["onto"] / B).tolist(),
        "others_rev_B_jpy":     (revs["others"] / B).tolist(),
        "keyence_share_pct":    keyence_share_pct,
        "cagr_base":            cagr_base,
        "keyence_growth_premium": keyence_growth_premium,
        "years":                years,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. plot_main — 4-panel figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_main(save: bool = True) -> plt.Figure:
    """
    Generate a 4-panel business model figure.

    Panels
    ------
    (A) Revenue & operating margin trend FY2019–2024 (bar + line)
    (B) Direct sales vs distributor cost waterfall (stacked bar)
    (C) Value-pricing: customer 5yr ROI vs product price (scatter)
    (D) Market share forecast 2024–2034 (stacked area)

    Parameters
    ----------
    save : bool
        If True, save PNG to OUT_DIR/keyence_business_model.png.

    Returns
    -------
    matplotlib.figure.Figure
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Keyence Corporation — Business Model Analysis\n"
        "Semiconductor Equipment Portfolio: Competitor/Partner Perspective",
        fontsize=13, fontweight="bold", y=0.98,
    )
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    kf = KeyenceFinancials()

    # ── (A) Revenue & operating margin ──────────────────────────────────────
    fys, revs = kf.revenue_series()
    _, margins = kf.op_margin_series()

    colors_a = ["#1a5276" if fy >= 2021 else "#5d8aa8" for fy in fys]
    bars = ax_a.bar(fys, revs, color=colors_a, alpha=0.85, label="Revenue (¥B)")
    ax_a.set_ylabel("Revenue [¥ B]", fontsize=9)
    ax_a.set_ylim(0, 1_300)
    ax_a.set_xticks(fys)
    ax_a.set_xticklabels([f"FY{y}" for y in fys], fontsize=8)
    ax_a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"¥{x:.0f}B"))

    ax_a2 = ax_a.twinx()
    ax_a2.plot(fys, margins, "o-", color="#e74c3c", lw=2, ms=6, label="OP margin (%)")
    ax_a2.set_ylim(40, 70)
    ax_a2.set_ylabel("Operating Margin [%]", fontsize=9, color="#e74c3c")
    ax_a2.tick_params(axis="y", labelcolor="#e74c3c")
    ax_a2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Annotate last bar
    last_rev = revs[-1]
    ax_a.annotate(
        f"¥{last_rev:.0f}B\n(OP {margins[-1]:.1f}%)",
        xy=(fys[-1], last_rev), xytext=(fys[-1] - 0.5, last_rev + 80),
        fontsize=7.5, ha="center",
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    cagr_pct = kf.cagr() * 100
    ax_a.set_title(
        f"(A) Revenue & Operating Margin  (CAGR {cagr_pct:.1f}% FY19-24)",
        fontsize=9, fontweight="bold",
    )
    lines_a  = [plt.Rectangle((0, 0), 1, 1, fc="#1a5276", alpha=0.85),
                plt.Line2D([0], [0], color="#e74c3c", lw=2, marker="o")]
    labels_a = ["Revenue (¥B)", "OP margin"]
    ax_a.legend(lines_a, labels_a, fontsize=7.5, loc="upper left")

    # ── (B) Direct sales vs distributor waterfall ────────────────────────────
    ds = direct_sales_model()

    # Build waterfall data for side-by-side stacked bar
    # Normalise to revenue = 100 for visual clarity
    models = ["Keyence\n(Direct)", "Distributor\n(Indirect)"]
    bk = ds["keyence_breakdown"]
    bd = ds["dist_breakdown"]

    # Stacked bar: COGS, Channel, R&D, SGA, OP Profit
    cogs    = [bk["COGS"] * 100,  bd["COGS"] * 100]
    channel = [0.0,               bd["Channel"] * 100]
    rd      = [bk["R&D"] * 100,   bd["R&D"] * 100]
    sga     = [bk["SGA"] * 100,   bd["SGA"] * 100]
    op      = [bk["OP"] * 100,    bd["OP"] * 100]

    bar_x   = [0, 1]
    bar_kw  = dict(width=0.55, edgecolor="white", linewidth=0.8)
    colors_b = {
        "COGS":    "#e74c3c",
        "Channel": "#e67e22",
        "R&D":     "#f1c40f",
        "SGA":     "#2ecc71",
        "OP":      "#1a5276",
    }

    bot = np.zeros(2)
    for label, vals, clr in [
        ("COGS",    cogs,    colors_b["COGS"]),
        ("Channel", channel, colors_b["Channel"]),
        ("R&D",     rd,      colors_b["R&D"]),
        ("SGA",     sga,     colors_b["SGA"]),
        ("OP Profit", op,    colors_b["OP"]),
    ]:
        ax_b.bar(bar_x, vals, bottom=bot, color=clr, label=label, **bar_kw)
        # Annotate non-zero segments
        for xi, (v, b) in enumerate(zip(vals, bot)):
            if v > 2.0:
                ax_b.text(xi, b + v / 2, f"{v:.0f}%",
                          ha="center", va="center", fontsize=8,
                          color="white", fontweight="bold")
        bot += np.array(vals)

    ax_b.set_xticks(bar_x)
    ax_b.set_xticklabels(models, fontsize=9)
    ax_b.set_ylabel("% of Revenue", fontsize=9)
    ax_b.set_ylim(0, 115)
    ax_b.set_title(
        f"(B) Direct vs Distributor Margin\n"
        f"Keyence OP {bk['OP']*100:.0f}%  vs  Distributor OP {bd['OP']*100:.0f}%",
        fontsize=9, fontweight="bold",
    )
    ax_b.legend(fontsize=7.5, loc="upper right", ncol=2)

    # ── (C) Value-pricing scatter ────────────────────────────────────────────
    vp_results = value_pricing_analysis(alpha=0.30)
    colors_c   = ["#1a5276", "#2ecc71", "#e74c3c"]
    markers_c  = ["o", "s", "^"]

    for i, r in enumerate(vp_results):
        roi_m  = r["customer_roi_5yr_jpy"] / M    # ¥M
        asp_m  = r["asp_actual_jpy"] / M
        vbp_m  = r["price_value_based_jpy"] / M
        ax_c.scatter(roi_m, asp_m, color=colors_c[i], marker=markers_c[i],
                     s=120, zorder=5, label=r["name"].split("(")[0].strip())
        ax_c.scatter(roi_m, vbp_m, color=colors_c[i], marker=markers_c[i],
                     s=60, alpha=0.4, zorder=4)
        ax_c.annotate(
            f"ASP ¥{asp_m:.0f}M\ngross {r['gross_margin_actual']*100:.0f}%",
            xy=(roi_m, asp_m), xytext=(roi_m * 0.75, asp_m * 1.25),
            fontsize=7, color=colors_c[i],
            arrowprops=dict(arrowstyle="->", color=colors_c[i], lw=0.8),
        )

    # Draw alpha guideline
    roi_range = np.linspace(500, 7_000, 200)
    bom_vkx = PRODUCT_DB["VK-X"]["bom_jpy"] / M
    ax_c.plot(
        roi_range,
        bom_vkx + 0.30 * roi_range,
        "k--", lw=1.2, alpha=0.5,
        label=r"Value price ($\alpha$=0.30)",
    )

    ax_c.set_xlabel("Customer 5yr ROI [¥M]", fontsize=9)
    ax_c.set_ylabel("Product Price [¥M]", fontsize=9)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_title(
        "(C) Value-Based Pricing: Customer ROI vs Product ASP\n"
        r"Price = BOM + $\alpha$ × Customer ROI  ($\alpha$=0.30)",
        fontsize=9, fontweight="bold",
    )
    ax_c.legend(fontsize=7.5)
    ax_c.grid(True, alpha=0.3, which="both", ls="--")

    # ── (D) Market share forecast ────────────────────────────────────────────
    ms = market_share_simulation(years=10)
    yrs = ms["year_range"]

    stacks = {
        "Keyence":     np.array(ms["keyence_rev_B_jpy"]),
        "KLA":         np.array(ms["kla_rev_B_jpy"]),
        "Hitachi HT":  np.array(ms["hitachi_rev_B_jpy"]),
        "Onto":        np.array(ms["onto_rev_B_jpy"]),
        "Others":      np.array(ms["others_rev_B_jpy"]),
    }
    palette_d = ["#1a5276", "#c0392b", "#27ae60", "#8e44ad", "#95a5a6"]

    ax_d.stackplot(
        yrs,
        [stacks[k] for k in stacks],
        labels=list(stacks.keys()),
        colors=palette_d,
        alpha=0.85,
    )

    # Overlay Keyence share % on right axis
    ax_d2 = ax_d.twinx()
    ax_d2.plot(yrs, ms["keyence_share_pct"], "w--", lw=1.8,
               label="Keyence share (%)", zorder=10)
    ax_d2.set_ylim(0, 20)
    ax_d2.set_ylabel("Keyence Market Share [%]", fontsize=9, color="white")
    ax_d2.tick_params(axis="y", labelcolor="white")
    ax_d2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    ax_d.set_xlabel("Year", fontsize=9)
    ax_d.set_ylabel("Revenue [¥B]", fontsize=9)
    ax_d.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"¥{x:.0f}B"))
    ax_d.set_title(
        f"(D) Semiconductor Metrology Share Forecast 2024–2034\n"
        f"Market CAGR {ms['cagr_base']*100:.0f}%, "
        f"Keyence premium +{ms['keyence_growth_premium']*100:.0f}%",
        fontsize=9, fontweight="bold",
    )
    handles_d, labels_d = ax_d.get_legend_handles_labels()
    ax_d.legend(handles_d, labels_d, fontsize=7.5, loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        out_path = os.path.join(OUT_DIR, "keyence_business_model.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[plot_main] Saved → {out_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    width = 65
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keyence Corporation business model simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figure generation (faster; useful for CI or quick analysis)",
    )
    args = parser.parse_args()

    # ── 1. Financials summary ────────────────────────────────────────────────
    _print_section("1. Keyence Financials FY2019–2024")
    kf = KeyenceFinancials()
    print(f"  {'FY':<6} {'Revenue(¥B)':>12} {'OP Margin':>10} "
          f"{'FCF Margin':>11} {'Employees':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*11} {'-'*10}")
    for r in kf.records:
        print(f"  {r.fy:<6} {r.revenue_jpy/B:>11.1f}B "
              f"{r.op_margin*100:>9.1f}% "
              f"{r.fcf_margin*100:>10.1f}% "
              f"{r.n_employees:>10,}")
    print(f"\n  Revenue CAGR (FY2019→2024): {kf.cagr()*100:.1f}%")
    fy24 = kf.get_fy(2024)
    print(f"  FY2024 OP profit: ¥{fy24.op_profit_jpy/B:.1f}B "
          f"(${fy24.op_profit_usd/1e9:.1f}B USD)")

    # ── 2. Direct sales model ────────────────────────────────────────────────
    _print_section("2. Direct Sales Model vs Distributor")
    ds = direct_sales_model()
    print(f"  Engineers:           {ds['n_engineers']:,}")
    print(f"  Revenue/engineer:    ¥{ds['revenue_per_eng_jpy']/M:.0f}M")
    print(f"  Keyence OP margin:   {ds['keyence_op_margin']*100:.1f}%")
    print(f"  Distributor OP marg: {ds['distributor_op_margin']*100:.1f}%")
    print(f"  Margin advantage:    +{ds['margin_diff_pp']:.1f}pp (direct vs dist)")
    print(f"  ASP premium:         +{ds['asp_premium_pct']:.1f}% vs distributor list")

    # ── 3. Value-pricing analysis ────────────────────────────────────────────
    _print_section("3. Value-Based Pricing (alpha=0.30)")
    vp = value_pricing_analysis(alpha=0.30)
    print(f"  {'Product':<35} {'ASP(¥M)':>8} {'Gross%':>7} "
          f"{'ROI 5yr(¥M)':>12} {'VBP(¥M)':>9}")
    print(f"  {'-'*35} {'-'*8} {'-'*7} {'-'*12} {'-'*9}")
    for r in vp:
        name_short = r["name"].split("(")[0].strip()
        print(f"  {name_short:<35} {r['asp_actual_jpy']/M:>7.0f}M "
              f"{r['gross_margin_actual']*100:>6.0f}% "
              f"{r['customer_roi_5yr_jpy']/M:>11.0f}M "
              f"{r['price_value_based_jpy']/M:>8.0f}M")
    print()
    print("  VK-X detail (from keyence_metrology_model ROI = $40.6M net 5yr):")
    vkx = vp[0]
    print(f"    Customer 5yr ROI:    ¥{vkx['customer_roi_5yr_jpy']/M:.0f}M "
          f"(${vkx['customer_roi_5yr_jpy']/M/USD_JPY:.1f}M USD)")
    print(f"    Keyence captures:    ¥{vkx['keyence_value_captured_jpy']/M:.0f}M "
          f"(alpha={vkx['alpha']:.2f})")
    print(f"    Customer keeps:      ¥{vkx['customer_value_kept_jpy']/M:.0f}M")

    # ── 4. Product margin breakdown ──────────────────────────────────────────
    _print_section("4. Product Margin Breakdown")
    all_mb = all_product_margins()
    print(f"  {'SKU':<8} {'BOM(¥M)':>8} {'ASP(¥M)':>8} {'GrossM%':>8} "
          f"{'UnitOP%':>8} {'ASP/BOM':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for mb in all_mb:
        print(f"  {mb['product']:<8} {mb['bom_jpy']/M:>7.1f}M "
              f"{mb['asp_jpy']/M:>7.1f}M "
              f"{mb['gross_margin_pct']:>7.1f}% "
              f"{mb['unit_op_margin_pct']:>7.1f}% "
              f"{mb['asp_to_bom_multiple']:>7.1f}×")
    print()
    vkx_mb = product_margin_breakdown("VK-X")
    print(f"  VK-X: BOM ¥{vkx_mb['bom_jpy']/M:.0f}M → ASP ¥{vkx_mb['asp_jpy']/M:.0f}M "
          f"→ Gross margin {vkx_mb['gross_margin_pct']:.0f}%")

    # ── 5. Market share simulation ───────────────────────────────────────────
    _print_section("5. Market Share Simulation 2024→2034")
    ms = market_share_simulation(years=10)
    print(f"  {'Year':<6} {'TAM(¥B)':>9} {'Keyence(¥B)':>12} {'Share%':>8} "
          f"{'KLA(¥B)':>9}")
    print(f"  {'-'*6} {'-'*9} {'-'*12} {'-'*8} {'-'*9}")
    for i, yr in enumerate(ms["year_range"]):
        if yr in (2024, 2026, 2028, 2030, 2032, 2034):
            print(f"  {yr:<6} {ms['tam_B_jpy'][i]:>8.0f}B "
                  f"{ms['keyence_rev_B_jpy'][i]:>11.1f}B "
                  f"{ms['keyence_share_pct'][i]:>7.1f}% "
                  f"{ms['kla_rev_B_jpy'][i]:>8.0f}B")

    # ── 6. Plot ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        _print_section("6. Generating 4-panel figure")
        plot_main(save=True)
    else:
        print("\n[--no-plot] Figure generation skipped.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
