"""
grinding_spc.py  --  研削 + SPC 統合: ウェーハ厚さムラ (TTV) モニタリング

物理モデル:
  - Preston 則による径方向 MRR 変動 -> 中央厚め (center-thick) プロファイル
  - 砥石傾き (tilt) による楔状厚さ分布
  - チャック平面度誤差 (Gaussian noise)

ロットシミュレーション:
  - 30枚ロット、n=4 でサブグループ化 (7.5 SG)
  - ホイール目詰まり (glazing) による TTV 増大ドリフト (Phase 2)
  - Phase 1/2 SPC で TTV を X-bar 管理図で管理
"""

import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import _spc_monitor_kernel as spc

# ---- グラインダー仕様 (DISCO DGP8761 class) ----
WAFER_R_MM   = 100.0    # ウェーハ半径 [mm]
WHEEL_RPM_0  = 5800.0   # 砥石回転数 [rpm] (#2000 仕上げ砥石)
CHUCK_RPM    = 300.0
WHEEL_D_M    = 0.200    # 砥石外径 [m]
OMEGA_CHUCK  = CHUCK_RPM * 2.0 * math.pi / 60.0   # チャック角速度 [rad/s]

# ---- 9点計測位置 (SEMI M1標準準拠) ----
# 中心(0) + 内輪4点(r=40mm) + 外輪4点(r=80mm)
MEAS_R_MM  = [0.0,  40.0,  40.0, 40.0, 40.0,  80.0,  80.0,  80.0,  80.0]
MEAS_TH_D  = [0.0,   0.0,  90.0, 180.0, 270.0,  0.0,  90.0, 180.0, 270.0]
MEAS_LABEL = ["Cnt", "+X40", "+Y40", "-X40", "-Y40",
               "+X80", "+Y80", "-X80", "-Y80"]

# ---- プロセス仕様 ----
TARGET_UM  = 100.0
USL_UM     = 103.0   # 中心厚さ上限
LSL_UM     =  97.0   # 中心厚さ下限
TTV_SPEC   =   2.0   # TTV (Total Thickness Variation) 規格上限 [um]

# ---- ロット設定 ----
N_WAFERS  = 30
N_PHASE1  = 20     # Phase 1 枚数 (安定期)
SUBGRP    =  4     # サブグループサイズ (X-bar 管理図)


# =============================================================================
# 物理モデル
# =============================================================================

def wheel_v_ms(rpm: float) -> float:
    """砥石周速 [m/s]"""
    return rpm * math.pi * WHEEL_D_M / 60.0


def v_rel_ms(r_mm: float, wheel_rpm: float) -> float:
    """
    ウェーハ面と砥石の相対速度 [m/s] at 半径 r_mm

    砥石とチャックが逆回転するため速度を加算する。
    v_rel(r) = v_wheel + omega_chuck * r
    """
    v_w = wheel_v_ms(wheel_rpm)
    v_c = OMEGA_CHUCK * (r_mm / 1000.0)
    return v_w + v_c


# 基準速度: r = ウェーハ半径/2 (50mm)
V_REF_MS = v_rel_ms(WAFER_R_MM / 2.0, WHEEL_RPM_0)


def thickness_9pt(
    wheel_rpm: float,
    tilt_um: float,
    sigma_chuck_um: float,
    rng: np.random.Generator,
) -> list:
    """
    9点厚さ計測をシミュレート [um]

    t(r, theta) = TARGET + delta_radial(r) + delta_tilt(r, theta) + eps_chuck

    Parameters
    ----------
    wheel_rpm      : 砥石回転数 [rpm]
    tilt_um        : 砥石傾きによる楔振幅 [um] (r=100mm での最大偏差)
    sigma_chuck_um : チャック平面度 1σ [um]
    rng            : numpy 乱数生成器

    Returns
    -------
    list of float, length 9 [um]
    """
    pts = []
    for r, th in zip(MEAS_R_MM, MEAS_TH_D):
        # 径方向効果: 中心は基準より遅い -> 除去量少 -> 厚め
        v       = v_rel_ms(r, wheel_rpm)
        d_rad   = (1.0 - v / V_REF_MS) * 2.5   # スケール係数 2.5um

        # 砥石傾き: 楔状オフセット (径×cos(theta))
        d_tilt  = tilt_um * (r / WAFER_R_MM) * math.cos(math.radians(th))

        # チャック平面度ランダム誤差
        eps     = rng.normal(0.0, sigma_chuck_um)

        pts.append(TARGET_UM + d_rad + d_tilt + eps)
    return pts


def ttv(pts: list) -> float:
    return max(pts) - min(pts)


# =============================================================================
# ロットシミュレーション (30枚, glazing ドリフトあり)
# =============================================================================

RNG  = np.random.default_rng(42)
lot  = []   # list of dict

for wid in range(N_WAFERS):
    # Phase 2 以降: 砥石目詰まり -> σ_chuck 増大 (TTV増加)
    if wid < N_PHASE1:
        extra_sigma = 0.0
    else:
        # glazing: 1枚ごとに 0.06um ずつ σ が増加
        extra_sigma = 0.06 * (wid - N_PHASE1 + 1)

    pts = thickness_9pt(
        wheel_rpm      = WHEEL_RPM_0,
        tilt_um        = 0.30,
        sigma_chuck_um = 0.25 + extra_sigma,
        rng            = RNG,
    )
    lot.append({
        "wid"      : wid + 1,
        "t_center" : pts[0],
        "ttv"      : ttv(pts),
        "pts"      : pts,
    })


# =============================================================================
# Phase 1/2 SPC (TTV の X-bar 管理図)
# =============================================================================

ttv_all  = [w["ttv"] for w in lot]
n_sg     = N_WAFERS  // SUBGRP
n_sg_p1  = N_PHASE1  // SUBGRP

# Phase 1: SPC カーネルで限界計算 (mm 単位)
p1 = spc.spc_update(
    measurements  = [v * 1e-3 for v in ttv_all[:N_PHASE1]],
    usl           = TTV_SPEC * 1e-3,
    lsl           = 0.0,
    subgroup_size = SUBGRP,
    ewma_lambda   = 0.2,
)

cl_um    = p1["xbar_bar"] * 1e3
ucl_um   = p1["ucl_x"]   * 1e3
sig_um   = p1["sigma_hat"]* 1e3
cpk_p1   = p1["cpk"]
se_um    = sig_um / (SUBGRP ** 0.5)


# =============================================================================
# 出力
# =============================================================================

print("=" * 68)
print("  DISCO SiC バックグラインド  厚さムラ (TTV) SPC モニタリング")
print("  ウェーハ: 4H-SiC 200mm  目標 %.0f um  TTV 規格 < %.1f um" % (TARGET_UM, TTV_SPEC))
print("=" * 68)

print()
print("  [ Phase 1  安定期ベースライン  ウェーハ 1-%d ]" % N_PHASE1)
print("    TTV 中心線 CL    : %.4f um" % cl_um)
print("    TTV UCL (3sigma) : %.4f um" % ucl_um)
print("    TTV sigma_hat    : %.4f um" % sig_um)
print("    Cpk              : %.3f  [%s]" % (cpk_p1, "OK (>=1.33)" if cpk_p1 >= 1.33 else "NG"))

print()
print("  [ Phase 2  リアルタイムモニタリング  ウェーハ %d-%d ]" % (N_PHASE1 + 1, N_WAFERS))
print()
print("   SG   TTV[um]   z     判定")
print("   " + "-" * 45)

first_alarm_sg = None
for sg in range(n_sg_p1, n_sg):
    start  = sg * SUBGRP
    end    = start + SUBGRP
    xb_ttv = sum(ttv_all[start:end]) / SUBGRP
    z      = (xb_ttv - cl_um) / se_um if se_um > 0.0 else 0.0

    if abs(z) > 3.0:
        status = "<-- ALARM 3sigma"
        if first_alarm_sg is None:
            first_alarm_sg = sg + 1
    elif z > 2.0:
        status = "<-- Warning"
    else:
        status = "OK"

    print("   SG%02d  %6.4f  %+5.2f  %s" % (sg + 1, xb_ttv, z, status))

print()
if first_alarm_sg is not None:
    print("  -> 最初のアラーム: SG%02d (ウェーハ %d 付近)" % (
        first_alarm_sg, first_alarm_sg * SUBGRP - 1))
    print("  -> glazing 検知: ドレッシングまたはホイール交換を推奨")
else:
    print("  -> 全サブグループ 管理内")

# ---- 9点プロファイル: 初枚 vs 最終枚 ----
print()
print("  [ 9点厚さプロファイル比較 ]")
print("  %-8s  %10s  %10s  %10s" % ("点", "1枚目[um]", "%d枚目[um]" % N_WAFERS, "差[um]"))
print("  " + "-" * 48)
first = lot[0]
last  = lot[-1]
for lbl, t1, tN in zip(MEAS_LABEL, first["pts"], last["pts"]):
    print("  %-8s  %10.4f  %10.4f  %+10.4f" % (lbl, t1, tN, tN - t1))

print()
print("  TTV 1枚目  : %.4f um  -> %s" % (
    first["ttv"], "OK" if first["ttv"] < TTV_SPEC else "NG"))
print("  TTV %d枚目 : %.4f um  -> %s" % (
    N_WAFERS, last["ttv"], "OK" if last["ttv"] < TTV_SPEC else "NG"))
print()
print("  Center-thick 効果 (r=0 vs r=80mm 外周平均):")
outer_avg_first = sum(first["pts"][5:]) / 4.0
outer_avg_last  = sum(last["pts"][5:])  / 4.0
print("    1枚目: 中心 %.4f um  外周平均 %.4f um  差 %+.4f um" % (
    first["pts"][0], outer_avg_first, first["pts"][0] - outer_avg_first))
print("    %d枚目: 中心 %.4f um  外周平均 %.4f um  差 %+.4f um" % (
    N_WAFERS, last["pts"][0], outer_avg_last, last["pts"][0] - outer_avg_last))
print("=" * 68)
