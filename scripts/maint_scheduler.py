"""
maint_scheduler.py  --  予防保全スケジューラー
wafer-proc-sim 統合モニタリング

全モジュールの SPC トレンドを集約して:
  - コンポーネントヘルスインデックス  H ∈ [0, 100]
  - 線形回帰による RUL (Remaining Useful Life) [SG]
  - メンテナンスアクション優先順位と推奨スケジュール

対象コンポーネント:
  SPINDLE_CURR : スピンドル電流 -> ダイシングブレード摩耗
  GRIND_TTV    : 研削後 TTV    -> 砥石目詰まり (glazing)
  KERF_WIDTH   : カーフ幅      -> ダイシングブレード刃先摩耗
"""

import math
import numpy as np

# =============================================================================
# SPC 凍結限界 (各モジュール Phase1 出力から)
# =============================================================================

# スピンドル電流 [A] (spindle_current_demo.py Phase1 結果)
SC_CL   = 1.9885;  SC_UCL  = 2.2407;  SC_SIG  = 0.1680
SC_SUBG = 4;       SC_SE   = SC_SIG / math.sqrt(SC_SUBG)

# 研削 TTV [μm] (grinding_spc.py Phase1 結果)
TTV_CL  = 0.8297;  TTV_UCL = 1.1065;  TTV_SIG = 0.0924
TTV_SUBG = 4;      TTV_SE  = TTV_SIG / math.sqrt(TTV_SUBG)

# カーフ幅 [μm] Phase1 凍結 (spc_kerf_demo.py Phase1 結果)
KW_CL   = 50.021;  KW_UCL  = 50.984;  KW_SIG  = 0.331
KW_SUBG = 4;       KW_SE   = KW_SIG / math.sqrt(KW_SUBG)

# =============================================================================
# Phase2 SPC トレンドをミニシミュレーションで再現
# =============================================================================

RNG = np.random.default_rng(7)

def _gen_spindle_current_p2(n_sg: int = 38) -> list:
    """Phase2 スピンドル電流 X-bar 系列 (SG13 から SG50)"""
    xbars = []
    n_phase1_cuts = 50
    for sg_p2 in range(n_sg):
        sg_abs  = 12 + sg_p2          # 0-indexed absolute SG
        n_worn  = sg_abs * SC_SUBG - n_phase1_cuts
        n_worn  = max(0, n_worn)
        W       = min(1.0, (n_worn / 150.0) ** 0.70)
        I_mean  = 1.20 + 0.80 * (1.0 + 1.60 * W)
        xbars.append(I_mean + RNG.normal(0.0, SC_SE))
    return xbars

def _gen_ttv_p2(n_sg: int = 3) -> list:
    """Phase2 TTV X-bar 系列 (SG06 から SG07)"""
    xbars = []
    for sg_p2 in range(n_sg):
        extra_sigma = 0.06 * (sg_p2 + 1)   # glazing drift per SG
        mean_ttv    = TTV_CL + 0.3 * (sg_p2 + 1)
        xbars.append(mean_ttv + RNG.normal(0.0, TTV_SE + extra_sigma / 4.0))
    return xbars

def _gen_kerf_width_p2(n_sg: int = 15) -> list:
    """Phase2 カーフ幅 X-bar 系列 (SG21 から SG35)"""
    xbars = []
    for sg_p2 in range(n_sg):
        drift  = 0.15 * (sg_p2 + 1)        # +0.15 μm/SG blade wear drift
        mean_k = KW_CL + drift
        xbars.append(mean_k + RNG.normal(0.0, KW_SE))
    return xbars

sc_p2  = _gen_spindle_current_p2(38)
ttv_p2 = _gen_ttv_p2(3)
kw_p2  = _gen_kerf_width_p2(15)

# =============================================================================
# ヘルスインデックスと RUL 計算
# =============================================================================

def health_index(xbar: float, cl: float, ucl: float) -> float:
    """H = (xbar - CL) / (UCL - CL) * 100  [0=新品相当, 100=UCL 到達]"""
    return max(0.0, min(100.0, (xbar - cl) / (ucl - cl) * 100.0))

def rul_estimate(xbars: list, cl: float, ucl: float, n_fit: int = 5) -> float:
    """
    直近 n_fit 点の線形回帰から UCL 到達までの残り SG 数を推定

    Returns
    -------
    RUL [SG]  (正: まだ余裕あり, 負: 既に UCL 超え, inf: 劣化傾向なし)
    """
    y = np.array(xbars[-n_fit:])
    x = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return float("inf")
    slope, intercept = np.polyfit(x, y, 1)
    if slope <= 1e-9:
        return float("inf")
    current = float(y[-1])
    return (ucl - current) / slope

# ---- ヘルスインデックス (現在値 = Phase2 最新 SG) ----
H_sc  = health_index(sc_p2[-1],  SC_CL,  SC_UCL)
H_ttv = health_index(ttv_p2[-1], TTV_CL, TTV_UCL)
H_kw  = health_index(kw_p2[-1],  KW_CL,  KW_UCL)

# ---- RUL [SG] ----
R_sc  = rul_estimate(sc_p2,  SC_CL,  SC_UCL)
R_ttv = rul_estimate(ttv_p2, TTV_CL, TTV_UCL)
R_kw  = rul_estimate(kw_p2,  KW_CL,  KW_UCL)

# =============================================================================
# アクション判定
# =============================================================================

H_CRITICAL = 90.0   # H >= 90: 即時対応
H_WARN     = 70.0   # H >= 70: 計画保全
RUL_URGENT =  2.0   # RUL < 2 SG: 即時
RUL_PLAN   = 10.0   # RUL < 10 SG: 計画

def action_level(H: float, rul: float) -> str:
    if H >= H_CRITICAL or rul < RUL_URGENT:
        return "IMMEDIATE"
    elif H >= H_WARN or rul < RUL_PLAN:
        return "PLAN"
    else:
        return "MONITOR"

act_sc  = action_level(H_sc,  R_sc)
act_ttv = action_level(H_ttv, R_ttv)
act_kw  = action_level(H_kw,  R_kw)

# =============================================================================
# 推奨アクション辞書
# =============================================================================

ACTIONS = {
    "SPINDLE_CURR": {
        "component"  : "ダイシングブレード (電流)",
        "signal"     : "スピンドル電流 X-bar",
        "unit"       : "A",
        "cl"         : SC_CL,  "ucl": SC_UCL,
        "xbar_now"   : sc_p2[-1],
        "H"          : H_sc,
        "rul"        : R_sc,
        "level"      : act_sc,
        "action_map" : {
            "IMMEDIATE": "ブレード即時交換 -> ドレッシング後リスタート",
            "PLAN"     : "次ロット終了後にブレード交換をスケジュール",
            "MONITOR"  : "SPC 継続監視 (次回 SG で再評価)",
        },
    },
    "GRIND_TTV": {
        "component"  : "研削ホイール (glazing)",
        "signal"     : "TTV X-bar",
        "unit"       : "μm",
        "cl"         : TTV_CL, "ucl": TTV_UCL,
        "xbar_now"   : ttv_p2[-1],
        "H"          : H_ttv,
        "rul"        : R_ttv,
        "level"      : act_ttv,
        "action_map" : {
            "IMMEDIATE": "砥石ドレッシング即時実施 (または砥石交換)",
            "PLAN"     : "次ウェーハ前にドレッシングをスケジュール",
            "MONITOR"  : "TTV トレンド継続監視",
        },
    },
    "KERF_WIDTH": {
        "component"  : "ダイシングブレード (カーフ幅)",
        "signal"     : "カーフ幅 X-bar",
        "unit"       : "μm",
        "cl"         : KW_CL,  "ucl": KW_UCL,
        "xbar_now"   : kw_p2[-1],
        "H"          : H_kw,
        "rul"        : R_kw,
        "level"      : act_kw,
        "action_map" : {
            "IMMEDIATE": "ブレード交換 -> キャリブレーションカット実施",
            "PLAN"     : "ロット間でブレード交換を計画",
            "MONITOR"  : "カーフ幅 SPC 継続監視",
        },
    },
}

PRIORITY_ORDER = {"IMMEDIATE": 0, "PLAN": 1, "MONITOR": 2}

# =============================================================================
# 出力
# =============================================================================

print("=" * 72)
print("  wafer-proc-sim  統合予防保全スケジューラー")
print("  %-20s  %6s  %6s  %6s  %10s" % ("コンポーネント", "H[%]", "RUL[SG]", "Xbar", "判定"))
print("=" * 72)

for key, info in sorted(ACTIONS.items(), key=lambda kv: PRIORITY_ORDER[kv[1]["level"]]):
    rul_str = "%.1f" % info["rul"] if math.isfinite(info["rul"]) else "  inf"
    print("  %-28s  %6.1f  %7s  %8.4f%s  [%s]" % (
        info["component"],
        info["H"],
        rul_str,
        info["xbar_now"],
        info["unit"][:1],
        info["level"]))

print()
print("  ─── アクション詳細 (優先順) ───")
for key, info in sorted(ACTIONS.items(), key=lambda kv: PRIORITY_ORDER[kv[1]["level"]]):
    lvl  = info["level"]
    rul_str = ("%.1f SG" % info["rul"]) if math.isfinite(info["rul"]) else "∞"
    marker = {"IMMEDIATE": "▶▶ IMMEDIATE", "PLAN": "▷  PLAN", "MONITOR": "○  MONITOR"}[lvl]
    print()
    print("  %s  |  %s" % (marker, info["component"]))
    print("    ヘルスインデックス : %.1f%%  (CL=%.4f  UCL=%.4f  現在=%.4f %s)" % (
        info["H"], info["cl"], info["ucl"], info["xbar_now"], info["unit"]))
    print("    推定 RUL          : %s" % rul_str)
    print("    推奨アクション    : %s" % info["action_map"][lvl])

print()
print("  ─── Phase2 トレンド履歴 (直近 SG) ───")
print()
print("  [スピンドル電流  CL=%.4fA  UCL=%.4fA]" % (SC_CL, SC_UCL))
print("  %-6s  %8s  %8s" % ("SG", "X-bar[A]", "H[%]"))
for i, xb in enumerate(sc_p2[-8:], start=len(sc_p2)-7):
    sg  = 12 + i
    H_i = health_index(xb, SC_CL, SC_UCL)
    mark = " <-- ALARM" if xb > SC_UCL else (" <-- Warn" if H_i > 70 else "")
    print("  SG%-4d  %8.4fA  %8.1f%%%s" % (sg, xb, H_i, mark))

print()
print("  [研削 TTV  CL=%.4fμm  UCL=%.4fμm]" % (TTV_CL, TTV_UCL))
print("  %-6s  %8s  %8s" % ("SG", "X-bar[μm]", "H[%]"))
for i, xb in enumerate(ttv_p2, start=5):
    H_i = health_index(xb, TTV_CL, TTV_UCL)
    mark = " <-- ALARM" if xb > TTV_UCL else (" <-- Warn" if H_i > 70 else "")
    print("  SG%-4d  %8.4fμm  %8.1f%%%s" % (i+1, xb, H_i, mark))

print()
print("  [カーフ幅  CL=%.3fμm  UCL=%.3fμm]" % (KW_CL, KW_UCL))
print("  %-6s  %8s  %8s" % ("SG", "X-bar[μm]", "H[%]"))
for i, xb in enumerate(kw_p2, start=20):
    H_i = health_index(xb, KW_CL, KW_UCL)
    mark = " <-- ALARM" if xb > KW_UCL else (" <-- Warn" if H_i > 70 else "")
    print("  SG%-4d  %8.3fμm  %8.1f%%%s" % (i+1, xb, H_i, mark))

print()
print("  ─── RUL 線形回帰 (直近5点) ───")
for key, info in ACTIONS.items():
    trend = {"SPINDLE_CURR": sc_p2, "GRIND_TTV": ttv_p2, "KERF_WIDTH": kw_p2}[key]
    y = np.array(trend[-5:])
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    rul_str = ("%.1f SG" % info["rul"]) if math.isfinite(info["rul"]) else "∞"
    print("  %-28s  slope=%+.4f %s/SG  RUL=%s" % (
        info["component"], slope, info["unit"], rul_str))

print("=" * 72)
