"""
alignment_demo.py  --  ウェーハアライメント補正
DISCO DFL7160-class  SiC ダイシングソー ビジョンアライメント

モデル:
  1. θ 補正: ノッチ検出 -> 角度誤差 -> ステージ回転補正
       theta_residual = theta_load - theta_measured = -eps_vision
  2. XY 補正: 初カーフ計測フィードバック -> ストリートオフセット除去
       x_residual = x_offset - x_measured = -eps_vision_xy
  3. 補正前後の残差統計 (N_WAFERS 枚ロット)
  4. ダイ歩留まりへの影響評価

DISCO DFL7160 ビジョンシステム仕様:
  - CCD カメラ 0.5 μm/pixel
  - ノッチ検出精度: σ = 0.005°
  - カーフ中心計測精度: σ = 0.30 μm
"""

import math
import numpy as np

# ---- ウェーハ / ダイシング仕様 ----
WAFER_R_MM        = 100.0    # ウェーハ半径 [mm]
STREET_PITCH_UM   = 120.0    # ダイシングストリートピッチ [μm]
KERF_WIDTH_UM     =  50.0    # カーフ幅 [μm]
MARGIN_UM         = (STREET_PITCH_UM - KERF_WIDTH_UM) / 2.0  # 片側余裕 [μm]

# ---- ロボットハンドリング誤差 ----
SIGMA_THETA_LOAD  =  0.080   # 搭載 角度誤差 1σ [°]
SIGMA_XY_LOAD     = 15.0     # 搭載 XY 誤差 1σ [μm]
X_SYS_UM          =  2.5     # チャックランナウト等 系統オフセット [μm]

# ---- ビジョンシステム精度 ----
SIGMA_VISION_TH   =  0.005   # ノッチ検出角度精度 1σ [°]
SIGMA_VISION_XY   =  0.30    # カーフ中心計測精度 1σ [μm]

# ---- アライメント規格 ----
THETA_SPEC_DEG    =  0.010   # θ 残差許容値 ±[°]
XY_SPEC_UM        =  2.0     # XY 残差許容値 ±[μm]

# ---- ロット設定 ----
N_WAFERS          =  30
EDGE_EVAL_MM      =  95.0    # ダイ端部 θ 誤差評価半径 [mm]


# =============================================================================
# アライメント補正シミュレーション
# =============================================================================

RNG = np.random.default_rng(99)

# 第1枚目で系統オフセット X_SYS を一回計測して全枚に適用
x_sys_meas = X_SYS_UM + RNG.normal(0.0, SIGMA_VISION_XY)

lot = []
for wid in range(N_WAFERS):

    # ---- ハンドリング誤差 (搭載ごとランダム) ----
    theta_load = RNG.normal(0.0, SIGMA_THETA_LOAD)   # [°]
    x_load     = RNG.normal(0.0, SIGMA_XY_LOAD)      # [μm]

    # ---- θ 補正 (ノッチ検出) ----
    # theta_measured = true + vision noise
    theta_meas   = theta_load + RNG.normal(0.0, SIGMA_VISION_TH)
    # stage rotates by -theta_meas -> residual = -eps_vision
    theta_resid  = theta_load - theta_meas  # ≈ -eps_vision

    # ウェーハ端部での横方向変位 [μm]
    disp_before = EDGE_EVAL_MM * 1e3 * math.tan(math.radians(abs(theta_load)))
    disp_after  = EDGE_EVAL_MM * 1e3 * math.tan(math.radians(abs(theta_resid)))

    # ---- XY 補正 ----
    # 真の X オフセット = ハンドリング + 系統
    x_true      = x_load + X_SYS_UM
    # 系統オフセット補正 (初枚学習値を引く)
    x_after_sys = x_true - x_sys_meas  # ≈ x_load + ε_sys_meas
    # カーフ計測フィードバック (per-wafer)
    x_meas_cut  = x_after_sys + RNG.normal(0.0, SIGMA_VISION_XY)
    x_resid     = x_after_sys - x_meas_cut  # ≈ -eps_vision_xy

    lot.append({
        "wid"          : wid + 1,
        "theta_load"   : theta_load,
        "theta_resid"  : theta_resid,
        "disp_before"  : disp_before,
        "disp_after"   : disp_after,
        "x_before"     : x_true,
        "x_resid"      : x_resid,
        "theta_ok"     : abs(theta_resid) <= THETA_SPEC_DEG,
        "xy_ok"        : abs(x_resid)     <= XY_SPEC_UM,
    })


# =============================================================================
# 統計サマリー
# =============================================================================

def arr_stats(vals):
    a = np.array(vals)
    return a.mean(), a.std(ddof=1), a.max()

th_b_mean, th_b_std, th_b_max = arr_stats([abs(w["theta_load"])  for w in lot])
th_a_mean, th_a_std, th_a_max = arr_stats([abs(w["theta_resid"]) for w in lot])
x_b_mean,  x_b_std,  x_b_max  = arr_stats([abs(w["x_before"])   for w in lot])
x_a_mean,  x_a_std,  x_a_max  = arr_stats([abs(w["x_resid"])    for w in lot])
dp_b_mean, dp_b_std, dp_b_max = arr_stats([w["disp_before"]     for w in lot])
dp_a_mean, dp_a_std, dp_a_max = arr_stats([w["disp_after"]      for w in lot])


def p_chip(mu: float, sigma: float, margin: float) -> float:
    """P(|N(mu, sigma)| > margin) by complementary error function approximation"""
    def _erfc(x: float) -> float:
        # Horner approximation for erfc (max err < 1.5e-7)
        t = 1.0 / (1.0 + 0.3275911 * abs(x))
        p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
            + t * (-1.453152027 + t * 1.061405429))))
        val = p * math.exp(-x * x)
        return val if x >= 0.0 else 2.0 - val

    s2 = sigma * math.sqrt(2.0)
    p_hi = 0.5 * _erfc((margin - mu) / s2)   # P(X > +margin)
    p_lo = 0.5 * _erfc((-margin - mu) / s2)  # P(X < -margin)
    return p_hi + (1.0 - p_lo - p_hi + p_lo)  # simplified: just P_hi + P_lo_tail

# Recompute simply:
def p_chip_simple(mu, sigma, margin):
    def _Q(x):
        t = 1.0 / (1.0 + 0.33267 * abs(x))
        p = (0.4361836 + t * (-0.1201676 + t * 0.9372980)) * t * math.exp(-x*x/2) / math.sqrt(2*math.pi) * math.sqrt(2*math.pi)
        # Use simpler formula
        return 0.0
    # Use erfc
    def erfc(x):
        t = 1.0 / (1.0 + 0.3275911 * x)
        return (0.254829592 + t*(-0.284496736 + t*(1.421413741
                + t*(-1.453152027 + t*1.061405429)))) * t * math.exp(-x*x)
    s2 = sigma * math.sqrt(2)
    p_pos = 0.5 * erfc((margin - mu) / s2)
    p_neg = 0.5 * erfc((margin + mu) / s2)
    return p_pos + p_neg

pc_before = p_chip_simple(abs(X_SYS_UM), SIGMA_XY_LOAD, MARGIN_UM)
pc_after  = p_chip_simple(0.0,           SIGMA_VISION_XY, MARGIN_UM)


# =============================================================================
# 出力
# =============================================================================

print("=" * 68)
print("  DISCO DFL7160  SiC ダイシング  ウェーハアライメント補正")
print("  ウェーハ: 4H-SiC 200mm  ピッチ: %.0f μm  カーフ: %.0f μm  余裕: %.0f μm" % (
    STREET_PITCH_UM, KERF_WIDTH_UM, MARGIN_UM))
print("  規格: θ ±%.3f°  XY ±%.1f μm" % (THETA_SPEC_DEG, XY_SPEC_UM))
print("=" * 68)

print()
print("  [ θ アライメント補正 (ノッチ検出 -> ステージ回転) ]")
print("  %-28s  %8s  %8s  %8s" % ("項目", "平均|値|", "1σ", "最大"))
print("  " + "-" * 60)
print("  %-28s  %8.4f°  %8.4f°  %8.4f°" % ("補正前 |θ_load|", th_b_mean, th_b_std, th_b_max))
print("  %-28s  %8.4f°  %8.4f°  %8.4f°" % ("補正後 |θ_residual|", th_a_mean, th_a_std, th_a_max))
print("  σ 改善率: %+.1f%%  (ビジョン限界 %.3f° 到達)" % (
    (1.0 - th_a_std / th_b_std) * 100, SIGMA_VISION_TH))
print()
print("  ウェーハ端部 r=%.0fmm 横方向変位:" % EDGE_EVAL_MM)
print("  %-28s  %8.2f μm  %8.2f μm  %8.2f μm" % ("補正前", dp_b_mean, dp_b_std, dp_b_max))
print("  %-28s  %8.3f μm  %8.3f μm  %8.3f μm" % ("補正後", dp_a_mean, dp_a_std, dp_a_max))
n_th_ok = sum(1 for w in lot if w["theta_ok"])
print("  θ 規格内 (±%.3f°): %d / %d (%.1f%%)" % (
    THETA_SPEC_DEG, n_th_ok, N_WAFERS, n_th_ok / N_WAFERS * 100))

print()
print("  [ XY アライメント補正 (カーフ計測フィードバック) ]")
print("  系統オフセット (チャックランナウト等): %.2f μm  学習値: %.2f μm" % (
    X_SYS_UM, x_sys_meas))
print()
print("  %-28s  %8s  %8s  %8s" % ("項目", "平均|値|", "1σ", "最大"))
print("  " + "-" * 60)
print("  %-28s  %8.2f μm  %8.2f μm  %8.2f μm" % ("補正前 |X_offset|", x_b_mean, x_b_std, x_b_max))
print("  %-28s  %8.3f μm  %8.3f μm  %8.3f μm" % ("補正後 |X_residual|", x_a_mean, x_a_std, x_a_max))
print("  σ 改善率: %+.1f%%  (ビジョン限界 %.2f μm 到達)" % (
    (1.0 - x_a_std / x_b_std) * 100, SIGMA_VISION_XY))
n_xy_ok = sum(1 for w in lot if w["xy_ok"])
print("  XY 規格内 (±%.1f μm): %d / %d (%.1f%%)" % (
    XY_SPEC_UM, n_xy_ok, N_WAFERS, n_xy_ok / N_WAFERS * 100))

print()
print("  [ ダイ歩留まりへの影響 ]")
print("  補正前 チッピング確率: %.4f%%  (σ=%.1f μm, μ=%.1f μm)" % (
    pc_before * 100, SIGMA_XY_LOAD, X_SYS_UM))
print("  補正後 チッピング確率: %.2e %%  (σ=%.2f μm)" % (
    pc_after * 100, SIGMA_VISION_XY))
print("  リスク低減率: %.0f× (補正前後比)" % max(1.0, pc_before / max(pc_after, 1e-30)))

print()
print("  [ ウェーハ別残差 (全 %d 枚) ]" % N_WAFERS)
print("  %-5s  %11s  %11s  %11s  %6s  %6s" % (
    "W#", "θ残差[°]", "端部変位[μm]", "X残差[μm]", "θ判定", "XY判定"))
print("  " + "-" * 60)
for w in lot:
    print("  W%-3d  %+11.5f  %11.3f  %+11.3f  %6s  %6s" % (
        w["wid"],
        w["theta_resid"],
        w["disp_after"],
        w["x_resid"],
        "OK" if w["theta_ok"] else "NG",
        "OK" if w["xy_ok"]    else "NG"))
print("=" * 68)
