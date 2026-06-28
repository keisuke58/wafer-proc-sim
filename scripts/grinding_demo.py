"""
grinding_demo.py  --  SiC ウェーハ裏面研削シミュレーション

DISCO DGP8761 クラス グラインダー想定。
  - プレストン則 (Preston 1927) で材料除去速度 (MRR) を計算
  - 表面粗さ Ra を砥粒モデルで推定
  - 表面下ダメージ深さ (SSD) を Ra の比例モデルで計算

ステップ:
  1. 粗研削 (#320砥石)  : 725 um -> 150 um
  2. 仕上げ研削 (#2000) : 150 um -> 100 um (目標)
"""

import math
import sys
import os

# ---------- グラインダー仕様 (DISCO DGP8761 class) ----------
WHEEL_DIAM_M     = 0.200    # 砥石外径 [m]
CHUCK_RPM        = 300.0    # チャック回転数 [rpm]
CHUCK_DIAM_M     = 0.200    # ウェーハ径 [m]

# 材料定数 (4H-SiC)
K_PRESTON        = 1.2e-13  # プレストン係数 [m^2/N]
# Ra = C * (infeed_um / grit_num)^alpha  (経験式)
RA_COEFF         = 0.55     # C
RA_ALPHA         = 0.72     # alpha
# SSD / Ra 比 (SiCの実験値: Blake & Scattergood 1990 に基づく)
SSD_RA_RATIO     = 12.0

# ---------- 工程パラメータ ----------
STEPS = [
    {
        "name"       : "粗研削 (#320)",
        "wheel_rpm"  : 4500,
        "infeed_um"  : 8.0,
        "grit"       : 320,
        "contact_N"  : 1.5,
        "start_um"   : 725.0,
        "target_um"  : 150.0,
    },
    {
        "name"       : "仕上げ研削 (#2000)",
        "wheel_rpm"  : 5800,
        "infeed_um"  : 0.5,
        "grit"       : 2000,
        "contact_N"  : 0.4,
        "start_um"   : 150.0,
        "target_um"  : 100.0,
    },
]

def wheel_velocity(wheel_rpm):
    """砥石周速 [m/s]"""
    return wheel_rpm * math.pi * WHEEL_DIAM_M / 60.0

def chuck_velocity():
    """チャック外周速 [m/s]（相対速度の代表値として使用）"""
    return CHUCK_RPM * math.pi * CHUCK_DIAM_M / 60.0

def mrr_um_per_rev(wheel_rpm, infeed_um, contact_N):
    """
    1チャック回転あたりの材料除去厚 [um/rev]
    Preston: MRR = K_p * P * v_rel
    P = contact_N / (接触面積 -- 簡略化で単位面積扱い)
    ここでは infeed_um を実測の除去量として
    Preston 係数で速度依存スケーリングのみ行う。
    """
    v_wheel = wheel_velocity(wheel_rpm)
    v_chuck = chuck_velocity()
    v_rel   = v_wheel + v_chuck           # 逆回転時の合算速度
    # 規格化: 基準速度 30 m/s, 基準荷重 1N での infeed をベースに速度・荷重スケール
    V_REF   = 30.0
    N_REF   = 1.0
    scale   = (v_rel / V_REF) * (contact_N / N_REF)
    return infeed_um * scale

def surface_roughness_nm(infeed_um, grit):
    """Ra 表面粗さ [nm] (経験式)"""
    return RA_COEFF * (infeed_um / grit) ** RA_ALPHA * 1000.0

def ssd_depth_nm(ra_nm):
    """表面下ダメージ深さ SSD [nm]"""
    return SSD_RA_RATIO * ra_nm

def simulate_step(step):
    """1研削ステップをシミュレート。結果 dict を返す。"""
    removal_um    = step["start_um"] - step["target_um"]
    eff_mrr       = mrr_um_per_rev(step["wheel_rpm"], step["infeed_um"], step["contact_N"])
    n_revs        = removal_um / eff_mrr if eff_mrr > 0 else float("inf")
    time_s        = n_revs / (CHUCK_RPM / 60.0)
    ra_nm         = surface_roughness_nm(step["infeed_um"], step["grit"])
    ssd_nm        = ssd_depth_nm(ra_nm)
    return {
        "name"       : step["name"],
        "removal_um" : removal_um,
        "eff_mrr"    : eff_mrr,
        "n_revs"     : n_revs,
        "time_s"     : time_s,
        "ra_nm"      : ra_nm,
        "ssd_nm"     : ssd_nm,
        "final_um"   : step["target_um"],
        "v_wheel"    : wheel_velocity(step["wheel_rpm"]),
    }

def print_results(results):
    print("=" * 60)
    print("  DISCO SiC バックグラインド シミュレーション")
    print("  ウェーハ: 4H-SiC 200mm  725um -> 100um")
    print("=" * 60)
    total_time = 0.0
    for r in results:
        print()
        print("  [ %s ]" % r["name"])
        print("    砥石周速       : %5.1f m/s" % r["v_wheel"])
        print("    除去量         : %5.1f um" % r["removal_um"])
        print("    実効 MRR       : %5.3f um/rev" % r["eff_mrr"])
        print("    チャック回転数 : %5.0f rev" % r["n_revs"])
        print("    加工時間       : %5.1f s  (%.1f min)" % (r["time_s"], r["time_s"]/60))
        print("    仕上げ厚さ     : %5.1f um" % r["final_um"])
        print("    表面粗さ Ra    : %5.2f nm" % r["ra_nm"])
        print("    表面下ダメージ : %5.1f nm" % r["ssd_nm"])
        total_time += r["time_s"]
    print()
    print("=" * 60)
    print("  合計加工時間   : %.1f s  (%.1f min)" % (total_time, total_time/60))
    print("  最終 Ra        : %.2f nm  ->  %s" % (
        results[-1]["ra_nm"],
        "OK (<5nm for power device)" if results[-1]["ra_nm"] < 5.0 else "要確認"))
    print("  最終 SSD       : %.1f nm  ->  %s" % (
        results[-1]["ssd_nm"],
        "OK (<100nm)" if results[-1]["ssd_nm"] < 100.0 else "要確認"))
    print("=" * 60)

if __name__ == "__main__":
    results = [simulate_step(s) for s in STEPS]
    print_results(results)
