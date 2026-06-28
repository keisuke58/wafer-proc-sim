"""
SPC デモ: SiC ダイシング カーフ幅モニタリング
=============================================

DISCO DFL7160 で SiC ウェーハを 35 サブグループ（各 n=4 計測）分ダイシングする場面。
- サブグループ 1-20: 安定プロセス（Cpk > 1.33）
- サブグループ 21-: ブレード摩耗でカーフ幅が徐々に増大 -> WECO アラーム

DISCO 現場での実際の使われ方:
  計測器（レーザー変位計 or カメラ）-> spc_update() -> Cpk < 1.33 でレシピ補正発動
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import _spc_monitor_kernel as spc

RNG = np.random.default_rng(42)

# プロセス設定
TARGET_UM   = 50.0    # 目標カーフ幅 [um]
USL_UM      = 55.0    # 上限規格値
LSL_UM      = 45.0    # 下限規格値
SIGMA_NOISE = 0.8     # 計測ノイズ [um]

N_SUBGROUPS = 35
SUBGROUP_N  = 4       # 1サブグループあたり計測数

DRIFT_START = 20      # このサブグループからブレード摩耗ドリフト開始
DRIFT_RATE  = 0.15    # 摩耗による 1サブグループあたりの上昇 [um]

# データ生成
measurements = []
for g in range(N_SUBGROUPS):
    drift = max(0.0, (g - DRIFT_START) * DRIFT_RATE)
    center = TARGET_UM + drift
    meas   = RNG.normal(center, SIGMA_NOISE, size=SUBGROUP_N).tolist()
    measurements.extend(meas)

# SPC 計算（カーネルは mm 単位）
result = spc.spc_update(
    measurements  = [m * 1e-3 for m in measurements],
    usl           = USL_UM * 1e-3,
    lsl           = LSL_UM * 1e-3,
    subgroup_size = SUBGROUP_N,
    ewma_lambda   = 0.2,
)

# 結果表示
xbar_um   = [v * 1e3 for v in result["xbar"]]
ucl_x_um  = result["ucl_x"] * 1e3
lcl_x_um  = result["lcl_x"] * 1e3
xbar_bar  = result["xbar_bar"] * 1e3
sigma_hat = result["sigma_hat"] * 1e3

cpk = result["cpk"]
cpk_label = "OK (>=1.33)" if cpk >= 1.33 else "NG (<1.33)"

print("=" * 62)
print("  DISCO SiC ダイシング -- カーフ幅 SPC モニタリング")
print("  仕様: %.0f-%.0f um  (目標 %.0f um)" % (LSL_UM, USL_UM, TARGET_UM))
print("=" * 62)
print("  X-bar (全体平均) : %.2f um" % xbar_bar)
print("  sigma-hat (R/d2) : %.3f um" % sigma_hat)
print("  UCL_x            : %.2f um" % ucl_x_um)
print("  LCL_x            : %.2f um" % lcl_x_um)
print("  Cp               : %.3f" % result["cp"])
print("  Cpk              : %.3f  [%s]" % (cpk, cpk_label))
print()

# X-bar 管理図（ASCII）
WIDTH   = 50
lo, hi  = lcl_x_um - 0.5, ucl_x_um + 0.5
span    = hi - lo

def to_col(v):
    return int((v - lo) / span * (WIDTH - 1))

ucl_col = to_col(ucl_x_um)
lcl_col = to_col(lcl_x_um)
cl_col  = to_col(xbar_bar)

print("  X-bar 管理図 (各行=1サブグループ, ▲=WECOアラーム)")
print("  %*s CL %*s UCL" % (lcl_col, "LCL", ucl_col - cl_col - 2, ""))

flags = result["weco_flags"]
for g, (xb, fl) in enumerate(zip(xbar_um, flags)):
    col  = max(0, min(WIDTH - 1, to_col(xb)))
    line = [" "] * WIDTH
    if 0 <= lcl_col < WIDTH: line[lcl_col] = "|"
    if 0 <= cl_col  < WIDTH: line[cl_col]  = "|"
    if 0 <= ucl_col < WIDTH: line[ucl_col] = "|"
    marker = "A" if fl else "o"   # A = alarm
    line[col] = marker

    alarm_parts = []
    if fl & 0b0001: alarm_parts.append("Rule1:3sigma")
    if fl & 0b0010: alarm_parts.append("Rule2:2of3>2s")
    if fl & 0b0100: alarm_parts.append("Rule3:4of5>1s")
    if fl & 0b1000: alarm_parts.append("Rule4:8run")
    alarm_str = " [" + ", ".join(alarm_parts) + "]" if alarm_parts else ""
    wear_mark = " <- blade wear start" if g == DRIFT_START else ""
    print("  SG%02d %s  %5.2fum%s%s" % (g + 1, "".join(line), xb, alarm_str, wear_mark))

print()
print("  WECO アラーム: %s" % ("あり" if result["weco_alarm"] else "なし"))
n_alarm = sum(1 for f in flags if f)
print("  アラーム件数 : %d / %d サブグループ" % (n_alarm, N_SUBGROUPS))
print()

# 処置判断
print("  --- 処置判断 ---")
if cpk >= 1.67:
    print("  Cpk >= 1.67: プロセス優秀 -- 現行レシピ継続")
elif cpk >= 1.33:
    print("  Cpk >= 1.33: 合格 -- モニタリング継続")
elif cpk >= 1.00:
    print("  1.00 <= Cpk < 1.33: 警告 -- レシピ補正検討")
else:
    print("  Cpk < 1.00: 不合格 -- 生産停止 & ブレード交換")
if result["weco_alarm"]:
    print("  WECO アラーム: ブレード摩耗 or セットアップずれ -> 保守確認")
