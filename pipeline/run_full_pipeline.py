"""
完全統合パイプライン実行スクリプト
====================================
ASML → Lasertec → TEL CMP → Disco → Lasertec 後検査
→ TEL ALD → Advantest ATE → Socionext SoC → NVIDIA GPU

1コマンドで全モジュールを順次実行し、
各ステップの結果を次ステップへ受け渡す。

実行:
    python pipeline/run_full_pipeline.py
    python pipeline/run_full_pipeline.py --process stealth
    python pipeline/run_full_pipeline.py --process blade --plots
"""

import argparse
import os
import sys
import time

import numpy as np

# パス設定
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ─── カラー出力ヘルパー ──────────────────────────────────────────────────
BOLD  = "\033[1m";  RESET = "\033[0m"
GREEN = "\033[92m"; CYAN  = "\033[96m"
YELLOW= "\033[93m"; RED   = "\033[91m"

def step(n, title):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  Step {n}: {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def info(msg): print(f"  {YELLOW}→{RESET} {msg}")
def warn(msg): print(f"  {RED}⚠{RESET}  {msg}")


def run(dicing_process: str = "stealth", make_plots: bool = False) -> dict:
    t0 = time.time()
    results = {"process": dicing_process}

    # ════════════════════════════════════════════════════════════════════
    step(1, "ASML EUV リソグラフィー (2nm ノード)")
    # ════════════════════════════════════════════════════════════════════
    from fem.asml_model import aerial_image, process_window, cd_to_vth_sigma

    node_nm   = 2.0
    system    = "EXE:5000 (High-NA)"
    ai = aerial_image(pitch_nm=4.0, CD_nm=node_nm, NA=0.55,
                      wavelength_nm=13.5, sigma=0.90)
    pw = process_window(system, CD_target_nm=node_nm)

    sigma_Vth = cd_to_vth_sigma(pw["sigma_CD_nm"], node_nm)
    ok(f"k1={ai['k1']:.3f}  NILS={ai['NILS']:.2f}  "
       f"DOF={pw['DOF_nm']:.0f}nm  EL={pw['EL_pct']}%")
    ok(f"σ_CD={pw['sigma_CD_nm']:.2f}nm → σ_Vth={sigma_Vth:.1f}mV")
    results.update({"k1": ai["k1"], "sigma_Vth_mV": sigma_Vth,
                    "DOF_nm": pw["DOF_nm"]})

    # ════════════════════════════════════════════════════════════════════
    step(2, "Lasertec ACTIS — EUV マスク検査")
    # ════════════════════════════════════════════════════════════════════
    from fem.tel_cmp_lasertec import euv_mask_defect_density, euv_inspection_roi

    euv = euv_mask_defect_density(node_nm, fab="TSMC")
    roi = euv_inspection_roi(node_nm)
    ok(f"欠陥密度={euv['defect_density']:.4f}/cm²  "
       f"歩留まり損失={euv['yield_loss_pct']:.3f}%")
    ok(f"ACTIS ROI={roi['ROI']:.0f}×  ($検査 ${roi['actis_cost_k']:.0f}k "
       f"vs リスク ${roi['cost_per_defect_M']:.0f}M)")
    results.update({"euv_yield_loss_pct": euv["yield_loss_pct"],
                    "actis_roi": roi["ROI"]})

    # ════════════════════════════════════════════════════════════════════
    step(3, "TEL CMP — SiC 表面平坦化")
    # ════════════════════════════════════════════════════════════════════
    from fem.tel_cmp_lasertec import cmp_time, preston_mrr

    cmp = cmp_time(target_depth_nm=50, pressure_kPa=40,
                   velocity_m_s=0.5, material="4H-SiC")
    ok(f"MRR={cmp['MRR_nm_min']:.2f} nm/min  "
       f"研磨時間={cmp['time_min']:.1f} min  "
       f"Si比={cmp['cost_factor_vs_Si']}× コスト")
    ok(f"Post-CMP Ra={cmp['Ra_after_nm']:.2f} nm")
    results["cmp_time_min"] = cmp["time_min"]
    results["Ra_precmp_nm"] = cmp["Ra_after_nm"]

    # ════════════════════════════════════════════════════════════════════
    step(4, f"Disco ダイシング ({dicing_process})")
    # ════════════════════════════════════════════════════════════════════
    from ml.quantum_defect_model import (process_induced_trap_density,
                                          shockley_read_hall_lifetime)

    HAZ_map = {"blade":2.5,"laser_ns":2.0,"laser_ps":0.5,
               "laser_fs":0.2,"stealth":0.05,"plasma":0.8}
    Ra_map  = {"blade":80,"laser_ns":60,"laser_ps":30,
               "laser_fs":15,"stealth":5,"plasma":20}
    HAZ = HAZ_map.get(dicing_process, 2.5)
    Ra  = Ra_map.get(dicing_process, 80)

    traps = process_induced_trap_density(HAZ, Ra, dicing_process)
    tau   = shockley_read_hall_lifetime(traps["V_C (Z1/2)"])
    ok(f"HAZ={HAZ}µm  Ra={Ra}nm")
    ok(f"V_C 密度={traps['V_C (Z1/2)']:.2e} cm⁻³  "
       f"キャリア寿命={tau:.1f}µs")
    results.update({"HAZ_um": HAZ, "Ra_nm": Ra,
                    "N_VC": traps["V_C (Z1/2)"], "tau_us": tau})

    # ════════════════════════════════════════════════════════════════════
    step(5, "Lasertec SiC 後検査")
    # ════════════════════════════════════════════════════════════════════
    from fem.tel_cmp_lasertec import sic_postdice_inspection

    sic_insp = sic_postdice_inspection(dicing_process)
    grade_ok = sic_insp["grade"] in ["A", "B"]
    (ok if grade_ok else warn)(
        f"Grade {sic_insp['grade']}  "
        f"Ra={sic_insp['Ra_nm']:.1f}nm  "
        f"チッピング={sic_insp['chip_mean']:.3f}µm")
    info(sic_insp["feedback"])
    results.update({"inspection_grade": sic_insp["grade"],
                    "chip_mean_um": sic_insp["chip_mean"]})
    if not grade_ok:
        warn("Disco にプロセス改善フィードバック送信")

    # ════════════════════════════════════════════════════════════════════
    step(6, "TEL ALD + 熱酸化 → SiC MOSFET 製造")
    # ════════════════════════════════════════════════════════════════════
    from fem.tel_process_model import full_pipeline as tel_full

    tel = tel_full(dicing_process=dicing_process, ald_material="HfO2",
                   ald_cycles=50, ox_T_C=1150, ox_t_hr=2.0, anneal_T_C=950)
    ok(f"µ_ch={tel['mu_ch']:.1f} cm²/Vs  "
       f"EOT={tel['total_EOT_nm']:.2f}nm  "
       f"R_on={tel['R_on_mohm_mm2']:.3f} mΩ·mm²")
    results.update({"mu_ch": tel["mu_ch"], "EOT_nm": tel["total_EOT_nm"],
                    "R_on": tel["R_on_mohm_mm2"], "Dit": tel["Dit_cm2eV"]})

    # ════════════════════════════════════════════════════════════════════
    step(7, "Advantest ATE テスト → 歩留まり")
    # ════════════════════════════════════════════════════════════════════
    from fem.advantest_model import (simulate_device_params, run_ate_test,
                                      wafer_map, test_economics)

    wm   = wafer_map(tel["mu_ch"], tel["Dit_cm2eV"])
    econ = test_economics(wm["yield_pct"], wm["n_die"])
    n_good = int(wm["n_die"] * wm["yield_pct"] / 100)
    ok(f"ATE 歩留まり={wm['yield_pct']:.2f}%  "
       f"良品数={n_good}  "
       f"CPGD=${econ['cpgd_full_usd']:.2f}")
    ok(f"テストコスト比={econ['test_cost_pct']:.1f}%")
    results.update({"yield_pct": wm["yield_pct"], "n_good": n_good,
                    "cpgd_usd": econ["cpgd_full_usd"],
                    "test_cost_pct": econ["test_cost_pct"]})

    # ════════════════════════════════════════════════════════════════════
    step(8, "ソシオネクスト SoC タイミング解析")
    # ════════════════════════════════════════════════════════════════════
    from fem.semiconductor_ecosystem import socionext_timing_yield

    soc_yield = socionext_timing_yield("SC2300", sigma_Vth_mV=sigma_Vth)
    ok(f"SC2300 ADAS SoC  node={soc_yield['node_nm']}nm  "
       f"street={soc_yield['street_um']}µm")
    ok(f"タイミング歩留まり={soc_yield['timing_yield_pct']:.2f}%  "
       f"z_slack={soc_yield['z_slack']:.2f}")
    results.update({"soc_timing_yield": soc_yield["timing_yield_pct"],
                    "z_slack": soc_yield["z_slack"]})

    # ════════════════════════════════════════════════════════════════════
    step(9, "NVIDIA GPU サプライチェーン影響")
    # ════════════════════════════════════════════════════════════════════
    from fem.semiconductor_ecosystem import nvidia_gpu_bom

    gpu = nvidia_gpu_bom("Blackwell B200")
    ok(f"B200 GPU装置コスト: Disco ${gpu['equipment_usd']['Disco']}/GPU  "
       f"TEL ${gpu['equipment_usd']['TEL']}/GPU")
    ok(f"装置コスト計 ${gpu['total_equip_usd']}/GPU = "
       f"{gpu['equip_pct_bom']:.1f}% of BOM")
    results.update({"gpu_disco_cost": gpu["equipment_usd"]["Disco"],
                    "gpu_equip_pct": gpu["equip_pct_bom"]})

    # ════════════════════════════════════════════════════════════════════
    # サマリー
    # ════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  完全パイプライン結果 — プロセス: {dicing_process}{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")

    summary = [
        ("ASML k1",              f"{results['k1']:.3f}"),
        ("σ_Vth (ASML→回路)",   f"{results['sigma_Vth_mV']:.1f} mV"),
        ("CMP 研磨時間",          f"{results['cmp_time_min']:.1f} min"),
        ("Disco HAZ",             f"{results['HAZ_um']:.2f} µm"),
        ("キャリア寿命",           f"{results['tau_us']:.1f} µs"),
        ("Lasertec Grade",        results["inspection_grade"]),
        ("チャンネル移動度",       f"{results['mu_ch']:.1f} cm²/Vs"),
        ("ATE 歩留まり",          f"{results['yield_pct']:.2f} %"),
        ("CPGD",                 f"$ {results['cpgd_usd']:.2f}"),
        ("SoC タイミング yield",  f"{results['soc_timing_yield']:.2f} %"),
        ("GPU 装置コスト比",      f"{results['gpu_equip_pct']:.1f} %"),
    ]
    for k, v in summary:
        print(f"  {k:<24} {BOLD}{v}{RESET}")

    print(f"\n  {GREEN}実行時間: {elapsed:.1f}s{RESET}")
    return results


def compare_all(make_plots: bool = False):
    """全プロセスを比較実行。"""
    processes = ["blade", "laser_ps", "laser_fs", "stealth"]
    all_results = {}
    for proc in processes:
        print(f"\n{'#'*60}")
        print(f"# プロセス: {proc}")
        print(f"{'#'*60}")
        all_results[proc] = run(proc, make_plots=False)

    # 比較表
    print(f"\n{BOLD}{'═'*80}{RESET}")
    print(f"{BOLD}  プロセス比較サマリー{RESET}")
    print(f"{BOLD}{'═'*80}{RESET}")
    print(f"  {'指標':<24} {'ブレード':>12} {'レーザーps':>12} "
          f"{'レーザーfs':>12} {'ステルス':>12}")
    print(f"  {'─'*72}")

    metrics = [
        ("HAZ [µm]",        "HAZ_um",          ".2f"),
        ("寿命 [µs]",       "tau_us",           ".1f"),
        ("Grade",           "inspection_grade", "s"),
        ("µ_ch [cm²/Vs]",  "mu_ch",            ".1f"),
        ("Yield [%]",       "yield_pct",        ".2f"),
        ("CPGD [$]",        "cpgd_usd",         ".2f"),
        ("SoC yield [%]",   "soc_timing_yield", ".2f"),
    ]
    for label, key, fmt in metrics:
        vals = [all_results[p][key] for p in processes]
        row = f"  {label:<24}"
        for v in vals:
            if fmt == "s":
                row += f" {str(v):>12}"
            else:
                row += f" {v:{'>12' + fmt}}"
        print(row)

    if make_plots:
        _plot_comparison(all_results, processes)

    return all_results


def _plot_comparison(all_results: dict, processes: list):
    """比較バーチャートを生成。"""
    import matplotlib.pyplot as plt
    import os

    labels = {"blade":"ブレード","laser_ps":"レーザーps",
              "laser_fs":"レーザーfs","stealth":"ステルス"}
    colors = {"blade":"#d62728","laser_ps":"#ff7f0e",
              "laser_fs":"#2ca02c","stealth":"#2166ac"}

    metrics = [
        ("HAZ_um",          "HAZ [µm]",         False),
        ("tau_us",          "寿命 τ [µs]",       True),
        ("mu_ch",           "µ_ch [cm²/Vs]",    False),
        ("yield_pct",       "Yield [%]",         False),
        ("cpgd_usd",        "CPGD [$]",          False),
        ("soc_timing_yield","SoC Timing Yield%", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (key, ylabel, log_scale) in zip(axes.flat, metrics):
        vals = [all_results[p][key] for p in processes]
        bars = ax.bar([labels[p] for p in processes], vals,
                      color=[colors[p] for p in processes],
                      alpha=0.85, edgecolor="k", lw=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    v + max(vals)*0.01, f"{v:.2f}",
                    ha="center", fontsize=8, fontweight="bold")
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11)
        ax.grid(alpha=0.2, axis="y")

    fig.suptitle("完全パイプライン比較\n"
                 "ASML→Lasertec→TEL CMP→Disco→TEL ALD→Advantest→ソシオネクスト",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(ROOT, "results", "full_pipeline_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[✓] 比較チャート -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="半導体製造 完全パイプライン実行")
    ap.add_argument("--process", default="stealth",
                    choices=["blade","laser_ns","laser_ps","laser_fs","stealth","plasma"],
                    help="ダイシングプロセス")
    ap.add_argument("--all",    action="store_true",
                    help="全プロセスを比較実行")
    ap.add_argument("--plots",  action="store_true",
                    help="比較チャートを生成")
    args = ap.parse_args()

    if args.all:
        compare_all(make_plots=args.plots)
    else:
        run(args.process, make_plots=args.plots)


if __name__ == "__main__":
    main()
