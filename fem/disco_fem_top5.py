"""
DISCO FEM Top-5 — ranked registry & unified driver
==================================================
ONE entry point that *ranks* and *orchestrates* the five FEM analyses most
likely performed at DISCO (切る・削る・磨く equipment + wafer process).

Each analysis already lives elsewhere in this repo. Per the REPO_MAP reuse
rule this module does **not** re-implement any physics — it imports the owning
modules, declares the business-relevance ranking, and emits one ranked table +
portfolio report + figure. Where an analytic owner exists (TAIKO sag, die
strength) the demo calls it directly; the full FEM owner is always an ABAQUS
script and is recorded with its exact run command.

Ranking axis = DISCO business relevance:
    revenue tie  ×  product moat  ×  roadmap pull (HBM / SiC / thin-die)

Usage
-----
    python -m fem.disco_fem_top5 --list      # ranked ASCII table (default)
    python -m fem.disco_fem_top5 --demo      # run analytic proxies, print numbers
    python -m fem.disco_fem_top5 --report    # write results/disco_fem_top5.{md,png}
    python -m fem.disco_fem_top5 --run N      # show the ABAQUS run command for rank N
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(REPO_ROOT, "results")
ABAQUS = shutil.which("abaqus") or "/home/nishioka/DassaultSystemes/SIMULIA/Commands/abaqus"


# ════════════════════════════════════════════════════════════════════════════
# Analytic proxies — each calls an existing owner or is a textbook sanity check.
# The *full* model is always the ABAQUS owner module (see Entry.owner / run_cmd).
# ════════════════════════════════════════════════════════════════════════════

def _demo_warpage() -> dict:
    """#1 — TAIKO® self-weight sag (analytic owner: taiko.deflection).

    Proxy for the warpage story: the TAIKO edge ring stiffens a thinned wafer.
    Full residual-stress→warpage FEM = fem/grinding_warpage_2d.py (ABAQUS).
    """
    from taiko.deflection import solve_taiko_deflection, taiko_advantage

    h_thin, w_ring = 50e-6, 3e-3          # 50 µm ground inner disk, 3 mm ring
    res = solve_taiko_deflection(h_thin, w_ring)
    adv = taiko_advantage(h_thin, w_ring)
    return {
        "case": "h_thin=50 µm, ring=3 mm, 300 mm wafer (1 g)",
        "taiko_w_max_um": round(res["w_max"] * 1e6, 2),
        "uniform_vs_taiko_sag_x": round(adv, 2),
        "note": "ring acts ≈ built-in clamp → ~4× less sag (HBM thin-wafer handling)",
    }


def _demo_chipping() -> dict:
    """#2 — process → chipping → strength sensitivity (Griffith, analytic).

    Reuses the lit/real-data-anchored chipping law and the Griffith strength
    owner (both in fem.die_strength_model). Full brittle-fracture chipping FEM
    = fem/dicing_blade_2d.py (ABAQUS/Explicit).
    """
    import numpy as np
    from fem.die_strength_model import empirical_chipping_um, fracture_strength

    depth, spindle = 80.0, 30000.0       # µm step depth, rpm — fixed
    feeds = (2.0, 3.5, 5.0)              # mm/s
    sweep = {}
    for f in feeds:
        c = empirical_chipping_um(depth, f, spindle)
        sweep[f"feed_{f:g}mm_s"] = {
            "chipping_um": round(c, 2),
            "fracture_MPa": round(fracture_strength(c, "4H-SiC"), 1),
        }
    # strength sensitivity dσ_f/dc at a representative 5 µm chip (per +1 µm)
    c0 = 5.0
    dσ = fracture_strength(c0 + 0.5, "4H-SiC") - fracture_strength(c0 - 0.5, "4H-SiC")
    # real-data fit quality vs the 11 Grade-A experimental points
    try:
        from ml.surrogate_transformer import make_experimental_data
        X, y = make_experimental_data()
        pred = np.array([empirical_chipping_um(d, fd, s) for d, fd, s in X[:, :3]])
        mae = float(np.mean(np.abs(pred - y)))
    except Exception:
        mae = None
    return {
        "sweep_depth80um_30krpm": sweep,
        "strength_loss_MPa_per_um_at_5um": round(dσ, 1),
        "real_data_fit_MAE_um": None if mae is None else round(mae, 2),
        "note": "feed↑ → chipping↑ → strength↓; full chipping field needs the FEM",
    }


def _demo_die_strength() -> dict:
    """#3 — chipping → fracture strength → assembly failure (analytic owner)."""
    from fem.die_strength_model import assembly_failure_rate

    out = {}
    for chip in (2.0, 5.0, 10.0):         # µm kerf chipping
        r = assembly_failure_rate(chip, material="4H-SiC", sigma_assembly_MPa=80.0)
        out[f"chip_{chip:g}um"] = {
            "fracture_MPa": r["fracture_str_MPa"],
            "P_fail_ppm": r["P_failure_ppm"],
            "pass": r["pass"],
        }
    return out


def _demo_modal() -> dict:
    """#4 — fundamental frequency of a clamped circular chuck disk.

    Order-of-magnitude sanity check (Kirchhoff plate idealization).
    Full structural/modal FEM = fem/stage_vibration_modal.py (ABAQUS Lanczos).
    """
    # SiC porous chuck idealized as a clamped plate
    E, nu, rho = 410e9, 0.14, 3100.0      # SiC
    a, h = 0.15, 0.020                    # 300 mm chuck, 20 mm thick
    D = E * h ** 3 / (12.0 * (1.0 - nu ** 2))
    lam2 = 10.2158                        # clamped-edge first mode
    f1 = lam2 / (2.0 * math.pi * a ** 2) * math.sqrt(D / (rho * h))
    rpm = 30000.0                          # spindle / blade rotation
    f_blade = rpm / 60.0
    return {
        "chuck": "SiC clamped disk a=150 mm, h=20 mm (idealized)",
        "f1_Hz": round(f1, 1),
        "f_blade_Hz_at_30krpm": round(f_blade, 1),
        "separation_ratio": round(f1 / f_blade, 2),
        "design_rule": "f1 >> f_blade to avoid resonance (want ratio ≳ 3)",
    }


def _demo_thermal() -> dict:
    """#5 — peak surface temperature of a moving Gaussian laser source.

    Carslaw-&-Jaeger steady semi-infinite proxy for HAZ extent.
    Full transient ablation FEM = fem/laser_groove_thermal_2d.py (ABAQUS).
    """
    A, P, w = 0.7, 15.0, 10e-6            # absorptivity, 15 W, 10 µm beam radius
    k = 120.0                              # SiC thermal conductivity [W/m·K]
    dT_peak = A * P / (math.sqrt(2.0 * math.pi) * k * w)
    T_ablation = 3100.0                    # SiC sublimation ~3100 K
    return {
        "beam": "15 W, 0.7 absorptivity, 10 µm radius on SiC",
        "dT_peak_K": round(dT_peak, 0),
        "T_ablation_K": T_ablation,
        "ablates": dT_peak > T_ablation,
        "note": "steady proxy; pulsed peak fluence & HAZ width need the FEM",
    }


# ════════════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Entry:
    rank: int
    key: str
    title: str
    title_en: str
    disco_product: str
    tier: str
    priority: float                  # 0–100 business-relevance score (bar chart)
    physics: str
    solver: str                      # "ABAQUS/Standard" | "ABAQUS/Explicit" | "analytic"
    owner: str                       # repo-relative path of the full model
    run_cmd: Optional[str] = None    # how to run the full ABAQUS model
    demo: Optional[Callable[[], dict]] = field(default=None, repr=False)


TOP5: list[Entry] = [
    Entry(
        rank=1, key="warpage",
        title="裏面研削の残留応力 → ウェーハ反り",
        title_en="Back-grind residual stress → wafer warpage",
        disco_product="TAIKO® / 薄ウェーハ研削 (HBM 25→10 µm)",
        tier="Tier 1 — 売上直結",
        priority=100.0,
        physics="研削力→残留応力→チャック解放で反り。TAIKOリングで拘束。",
        solver="ABAQUS/Standard",
        owner="fem/grinding_warpage_2d.py (+_3d, taiko/deflection.py)",
        run_cmd="cd fem && abaqus cae noGUI=grinding_warpage_2d.py   # reads run_config.json",
        demo=_demo_warpage,
    ),
    Entry(
        rank=2, key="chipping",
        title="ダイシングのチッピング / き裂進展",
        title_en="Dicing kerf chipping / crack propagation",
        disco_product="ブレード + ステルスダイシング / カーフ品質",
        tier="Tier 1 — 競争力の核",
        priority=92.0,
        physics="Drucker-Prager脆性破壊 + G_c軟化。送り/切込みでチッピング量。",
        solver="ABAQUS/Explicit",
        owner="fem/dicing_blade_2d.py (+_3d, stealth_dicing_crack_model.py)",
        run_cmd="cd fem && abaqus cae noGUI=dicing_blade_2d.py   # reads run_config.json",
        demo=_demo_chipping,
    ),
    Entry(
        rank=3, key="die_strength",
        title="ダイ強度 / 薄ダイ破壊確率",
        title_en="Die strength / thin-die fracture probability",
        disco_product="車載品質保証 · 3D積層ハンドリング",
        tier="Tier 2 — 品質保証",
        priority=80.0,
        physics="Griffith σ_f=K_Ic/(Y√πa) → Weibull実装時破壊率(ppm)。",
        solver="analytic",
        owner="fem/die_strength_model.py (+ ml/weibull_analysis.py)",
        run_cmd="python -m fem.die_strength_model",
        demo=_demo_die_strength,
    ),
    Entry(
        rank=4, key="modal",
        title="装置の構造剛性 · モーダル解析",
        title_en="Equipment structural stiffness & modal analysis",
        disco_product="スピンドル / ステージ剛性 · 共振回避",
        tier="Tier 2 — 装置設計の土台",
        priority=72.0,
        physics="切断荷重で静たわみ + Lanczos固有値で f1 >> f_blade を保証。",
        solver="ABAQUS/Standard",
        owner="fem/stage_vibration_modal.py (+ C++ _frame_fem)",
        run_cmd="cd fem && abaqus cae noGUI=stage_vibration_modal.py -- --spindle_rpm 30000",
        demo=_demo_modal,
    ),
    Entry(
        rank=5, key="thermal",
        title="レーザ / 研削の熱解析 (HAZ)",
        title_en="Laser / grinding thermal analysis (HAZ)",
        disco_product="ステルス · KABRA® · 熱変形",
        tier="Tier 2 — レーザ工程",
        priority=66.0,
        physics="移動ガウス熱源→温度場→アブレーション/HAZ要素除去。多パス重畳。",
        solver="ABAQUS/Standard",
        owner="fem/laser_groove_thermal_2d.py (+ kabra_thermal_2d.py)",
        run_cmd="cd fem && abaqus cae noGUI=laser_groove_thermal_2d.py",
        demo=_demo_thermal,
    ),
]


def get(rank: int) -> Entry:
    for e in TOP5:
        if e.rank == rank:
            return e
    raise KeyError(f"no Top-5 entry with rank {rank} (valid: 1–{len(TOP5)})")


# ════════════════════════════════════════════════════════════════════════════
# Views
# ════════════════════════════════════════════════════════════════════════════

def _abaqus_available() -> bool:
    return os.path.exists(ABAQUS)


def render_table() -> str:
    head = (f"DISCO FEM — Top-5 by business relevance"
            f"   (ABAQUS: {'found' if _abaqus_available() else 'NOT found'})")
    lines = [head, "=" * len(head)]
    for e in TOP5:
        lines.append(f"#{e.rank}  {e.title}   [{e.tier}]")
        lines.append(f"     product : {e.disco_product}")
        lines.append(f"     solver  : {e.solver:16s} owner: {e.owner}")
        lines.append(f"     physics : {e.physics}")
        if e.run_cmd:
            lines.append(f"     run     : {e.run_cmd}")
        lines.append("")
    return "\n".join(lines)


def run_demos() -> dict:
    out = {}
    for e in TOP5:
        if e.demo is None:
            out[e.key] = {"_status": "ABAQUS-only — no analytic proxy",
                          "run": e.run_cmd}
            continue
        try:
            out[e.key] = e.demo()
        except Exception as exc:                       # never let one demo kill the run
            out[e.key] = {"_error": repr(exc)}
    return out


def _fmt(d: dict, indent: int = 6) -> str:
    pad = " " * indent
    rows = []
    for k, v in d.items():
        rows.append(f"{pad}{k}: {v}")
    return "\n".join(rows)


def write_report(out_md: Optional[str] = None, with_figure: bool = True) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    out_md = out_md or os.path.join(RESULTS, "disco_fem_top5.md")
    demos = run_demos()

    md = ["# DISCO FEM — Top-5 ランキング（業務直結度順）", "",
          "> wafer-proc-sim 既存FEMを **ビジネス重要度** で序列化した統合ビュー。",
          "> 物理は各 owner モジュールが所有（本ファイルは再実装しない）。", ""]
    md.append("| # | 解析 | DISCO製品 | Tier | Solver | Owner |")
    md.append("|---|------|-----------|------|--------|-------|")
    for e in TOP5:
        md.append(f"| {e.rank} | {e.title} | {e.disco_product} | "
                  f"{e.tier} | {e.solver} | `{e.owner}` |")
    md.append("")

    md.append("## 解析プロキシ実行結果（ABAQUS不要の数値チェック）")
    md.append("")
    for e in TOP5:
        md.append(f"### #{e.rank} {e.title}")
        md.append(f"- **物理**: {e.physics}")
        md.append(f"- **full model**: `{e.owner}` ({e.solver})")
        if e.run_cmd:
            md.append(f"- **run**: `{e.run_cmd}`")
        res = demos[e.key]
        md.append("- **proxy**:")
        md.append("")
        md.append("```")
        for k, v in res.items():
            md.append(f"{k}: {v}")
        md.append("```")
        md.append("")

    fig_path = None
    if with_figure:
        fig_path = _write_figure()
        if fig_path:
            md.append(f"![priority]({os.path.basename(fig_path)})")
            md.append("")

    text = "\n".join(md)
    with open(out_md, "w") as f:
        f.write(text)
    return out_md


def _write_figure() -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    # thesis_style sets usetex; fall back silently if LaTeX is unavailable
    try:
        import sys
        sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
        from thesis_style import use
        figsize = use(width_frac=1.0, aspect=0.5)
    except Exception:
        figsize = (6.0, 3.0)
    import matplotlib.pyplot as plt
    try:
        fig, ax = plt.subplots(figsize=figsize)
        # avoid LaTeX-special chars (#, &) when thesis_style enables usetex
        labels = [f"{e.rank}. {e.title_en.replace('&', 'and')}"
                  for e in reversed(TOP5)]
        vals = [e.priority for e in reversed(TOP5)]
        ax.barh(labels, vals, color="#2c6fbb")
        ax.set_xlabel("DISCO business-relevance score")
        ax.set_xlim(0, 100)
        fig.tight_layout()
        out_png = os.path.join(RESULTS, "disco_fem_top5.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return out_png
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="DISCO FEM Top-5 unified driver")
    ap.add_argument("--list", action="store_true", help="ranked ASCII table (default)")
    ap.add_argument("--demo", action="store_true", help="run analytic proxies")
    ap.add_argument("--report", action="store_true", help="write results/disco_fem_top5.{md,png}")
    ap.add_argument("--run", type=int, metavar="N", help="show ABAQUS run command for rank N")
    args = ap.parse_args(argv)

    if args.run is not None:
        e = get(args.run)
        print(f"#{e.rank} {e.title}\n  owner : {e.owner}\n  solver: {e.solver}")
        print(f"  run   : {e.run_cmd}")
        if e.solver.startswith("ABAQUS") and not _abaqus_available():
            print("  WARN  : abaqus not found on PATH")
        return

    if args.demo:
        import json
        print(json.dumps(run_demos(), indent=2, ensure_ascii=False, default=str))
        return

    if args.report:
        path = write_report()
        print(f"wrote {path}")
        return

    print(render_table())


if __name__ == "__main__":
    main()
