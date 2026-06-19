"""
最強2nm ハイブリッドダイシング — 統合実行パイプライン
=====================================================
Stage A レーザーグルービング → Stage B 個片化 (stealth / plasma / blade) を
1本のチェーンで実行し、2nm制約で評価して「最強ルート」を定量比較する。

実行:
    python pipeline/hybrid_dicing_pipeline.py                  # 3ルートKPI比較
    python pipeline/hybrid_dicing_pipeline.py --route laser+stealth   # 単一ルート詳細
    python pipeline/hybrid_dicing_pipeline.py --optimize              # 多目的最適化
    python pipeline/hybrid_dicing_pipeline.py --optimize --plots      # + パレート図
    python pipeline/hybrid_dicing_pipeline.py --material SiC --thickness 110 --street 25
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sic.hybrid import Recipe, HybridDicer, Constraints2nm, ROUTES
from sic.hybrid.optimize import compare_routes, knee_point, strongest_route

# ─── color helpers ──────────────────────────────────────────────────────────
BOLD = "\033[1m"; RESET = "\033[0m"
GREEN = "\033[92m"; CYAN = "\033[96m"; YELLOW = "\033[93m"; RED = "\033[91m"


def _ok(flag: bool) -> str:
    return f"{GREEN}feasible{RESET}" if flag else f"{RED}INFEASIBLE{RESET}"


def banner(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─'*68}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*68}{RESET}")


def base_recipe(args) -> Recipe:
    return Recipe(
        material=args.material,
        wafer_thickness_um=args.thickness,
        street_width_um=args.street,
        lowk_stack_um=args.lowk,
    )


def show_single(dicer: HybridDicer, recipe: Recipe) -> None:
    res = dicer.run(recipe)
    banner(f"ROUTE {recipe.route}  ({recipe.material}, "
           f"t={recipe.wafer_thickness_um:.0f}µm, street={recipe.street_width_um:.0f}µm)")
    for tag, o in (("A  laser_groove", res.stage_a), (f"B  {res.stage_b.method}", res.stage_b)):
        print(f"  {BOLD}{tag:18s}{RESET} kerf={o.kerf_um:6.2f}  chip={o.chipping_um:6.2f}  "
              f"HAZ={o.haz_um:5.2f}  SSD={o.subsurface_crack_um:5.2f}  "
              f"σ_res={o.residual_stress_MPa:6.1f}MPa  WPH={o.throughput_wph:6.1f}")
    f = res.feasibility
    print(f"\n  {BOLD}KPI{RESET}  die_strength={f.die_strength_MPa:6.0f} MPa   "
          f"series_throughput={res.throughput_wph:6.2f} WPH   {_ok(f.feasible)}")
    if f.violations:
        for v in f.violations:
            print(f"    {YELLOW}✗ {v}{RESET}")


def show_compare(dicer: HybridDicer, recipe: Recipe) -> None:
    banner(f"3-ROUTE KPI COMPARISON  ({recipe.material}, t={recipe.wafer_thickness_um:.0f}µm, "
           f"street={recipe.street_width_um:.0f}µm, low-k={recipe.lowk_stack_um:.0f}µm)")
    results = dicer.run_all_routes(recipe)
    hdr = f"  {'route':16s}{'feasible':>10}{'σ_die[MPa]':>12}{'WPH':>9}{'chip[µm]':>10}{'HAZ[µm]':>9}"
    print(BOLD + hdr + RESET)
    for name, res in results.items():
        s = res.summary()
        col = GREEN if s["feasible"] else RED
        print(f"  {name:16s}{col}{str(s['feasible']):>10}{RESET}"
              f"{s['die_strength_MPa']:>12.0f}{s['throughput_wph']:>9.1f}"
              f"{s['total_chipping_um']:>10.2f}{s['max_haz_um']:>9.2f}")
    print(f"\n  {YELLOW}注: 既定レシピでの素点。最適化は --optimize{RESET}")


def show_optimize(dicer: HybridDicer, recipe: Recipe, n: int, seed: int,
                  plots: bool) -> None:
    banner(f"CONSTRAINED MULTI-OBJECTIVE OPTIMIZATION  (n={n}/route)")
    searches = compare_routes(base=recipe, n_samples=n, seed=seed, dicer=dicer)

    print(BOLD + f"  {'route':16s}{'feas/tot':>10}{'max σ_die':>11}{'max WPH':>9}"
                 f"{'min σ_res':>11}{'knee WPH':>10}{'knee σ_res':>12}" + RESET)
    for name, s in searches.items():
        if s.n_feasible == 0:
            print(f"  {name:16s}{RED}{'0/'+str(s.n_eval):>10}{RESET}"
                  f"{'—':>11}{'—':>9}{'—':>11}{'—':>10}{'—':>12}")
            continue
        knee = knee_point(s)
        print(f"  {name:16s}{GREEN}{str(s.n_feasible)+'/'+str(s.n_eval):>10}{RESET}"
              f"{s.best_strength.die_strength_MPa:>11.0f}"
              f"{s.best_throughput.throughput_wph:>9.1f}"
              f"{s.best_lowstress.residual_stress_MPa:>11.1f}"
              f"{knee.throughput_wph:>10.1f}{knee.residual_stress_MPa:>12.1f}")

    name, knee = strongest_route(searches)
    if knee is not None:
        print(f"\n  {BOLD}{GREEN}★ 最強2nmルート: {name}{RESET}  "
              f"σ_die={knee.die_strength_MPa:.0f} MPa, WPH={knee.throughput_wph:.1f}, "
              f"σ_res={knee.residual_stress_MPa:.1f} MPa")
        r = knee.recipe
        print(f"    recipe: groove {r.groove_power_W:.1f}W @ {r.groove_speed_mm_s:.0f}mm/s "
              f"×{r.groove_passes} ({r.groove_pulse_regime})  |  route knobs tuned")

    if plots:
        _plot_pareto(searches, recipe)


def _plot_pareto(searches, recipe) -> None:
    from sic.hybrid.viz import plot_pareto_dual
    out = plot_pareto_dual(searches, recipe,
                           os.path.join(ROOT, "results", "hybrid_2nm_pareto.png"))
    print(f"\n  {CYAN}wrote {out}{RESET}")


def show_sensitivity(dicer: HybridDicer, recipe: Recipe, n: int, seed: int) -> None:
    from sic.hybrid.sensitivity import analyze_all, TARGETS
    from sic.hybrid.viz import plot_sensitivity, write_sensitivity_md, pretty
    banner(f"VARIABLE-IMPORTANCE RANKING  (signed Spearman ρ, n={n}/route)")
    results = analyze_all(base=recipe, n_samples=n, seed=seed, dicer=dicer)

    for route, r in results.items():
        print(f"\n  {BOLD}{route}{RESET}  (feasible {r.n_feasible}/{len(r.X)})")
        for i, (v, sc) in enumerate(r.overall_ranking()[:4], 1):
            bar = "█" * int(round(sc * 24))
            print(f"    {i}. {pretty(v):26s} {sc:5.2f}  {CYAN}{bar}{RESET}")

    fig = plot_sensitivity(results, recipe,
                           os.path.join(ROOT, "results", "hybrid_2nm_sensitivity.png"))
    md = write_sensitivity_md(
        results, recipe,
        os.path.join(ROOT, "sic", "hybrid_dicing_2nm_sensitivity.md"),
        fig_rel="../results/hybrid_2nm_sensitivity.png")
    print(f"\n  {CYAN}wrote {fig}{RESET}")
    print(f"  {CYAN}wrote {md}{RESET}")


def show_dataset(dicer: HybridDicer, recipe: Recipe, n: int, seed: int) -> None:
    from sic.hybrid.dataset import (
        build_dataset, write_dataset, SEMICONDUCTORS, DEFAULT_GEOMETRIES)
    from sic.hybrid.viz import plot_materials, write_dataset_card
    # Material-relative die-strength spec so brittle materials (Si/Ga2O3) are held
    # to a brittle-material requirement → a real feasibility gradient across materials.
    mat_dicer = HybridDicer(constraints=Constraints2nm(strength_min_frac=0.6))
    banner(f"MATERIAL-SWEEP DATASET  ({', '.join(SEMICONDUCTORS)}  ×  "
           f"{len(ROUTES)} routes  ×  {len(DEFAULT_GEOMETRIES)} geometries  ×  {n} recipes)")
    df = build_dataset(n_per_cell=n, seed=seed, base=recipe, dicer=mat_dicer)
    out_dir = os.path.join(ROOT, "data", "hybrid_dicing")
    paths = write_dataset(df, out_dir)

    # quick console summary
    from sic.hybrid.dataset import summarize
    s = summarize(df)
    print(f"  {BOLD}{'material':9}{'route':16}{'feas%':>7}{'σ_die':>8}{'chip':>7}"
          f"{'σ_res':>8}{'WPH':>7}{RESET}")
    for _, r in s.iterrows():
        col = GREEN if r.feasible_rate > 0 else RED
        print(f"  {r.material:9}{r.route:16}{col}{r.feasible_rate*100:>6.0f}%{RESET}"
              f"{r.die_strength_med:>8.0f}{r.chipping_med:>7.2f}"
              f"{r.residual_med:>8.1f}{r.throughput_med:>7.2f}")

    fig = plot_materials(df, recipe, os.path.join(ROOT, "results", "hybrid_2nm_materials.png"))
    card = write_dataset_card(
        df, recipe, os.path.join(out_dir, "README.md"),
        csv_rel="hybrid_2nm_dataset.csv", fig_rel="../../results/hybrid_2nm_materials.png")
    print(f"\n  {CYAN}wrote {paths['full']}  ({paths['n_rows']} rows){RESET}")
    print(f"  {CYAN}wrote {paths['summary']}{RESET}")
    print(f"  {CYAN}wrote {fig}{RESET}")
    print(f"  {CYAN}wrote {card}{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid 2nm dicing pipeline")
    ap.add_argument("--route", choices=sorted(ROUTES), default=None,
                    help="run a single route in detail (default: compare all)")
    ap.add_argument("--optimize", action="store_true", help="run multi-objective search")
    ap.add_argument("--sensitivity", action="store_true",
                    help="rank process variables by influence (→ figure + md)")
    ap.add_argument("--dataset", action="store_true",
                    help="generate material-sweep ML dataset (→ CSVs + card + figure)")
    ap.add_argument("--plots", action="store_true", help="save Pareto figure")
    ap.add_argument("--n", type=int, default=200, help="samples per route for --optimize")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--material", default="SiC")
    ap.add_argument("--thickness", type=float, default=110.0, help="wafer thickness [µm]")
    ap.add_argument("--street", type=float, default=25.0, help="street width [µm]")
    ap.add_argument("--lowk", type=float, default=2.0, help="low-k stack thickness [µm]")
    args = ap.parse_args()

    dicer = HybridDicer()
    recipe = base_recipe(args)

    if args.dataset:
        show_dataset(dicer, recipe, args.n, args.seed)
    elif args.sensitivity:
        show_sensitivity(dicer, recipe, args.n, args.seed)
    elif args.optimize:
        show_optimize(dicer, recipe, args.n, args.seed, args.plots)
    elif args.route:
        show_single(dicer, recipe.copy(route=args.route))
    else:
        show_compare(dicer, recipe)


if __name__ == "__main__":
    main()
