"""
sic.hybrid.viz — polished figures + markdown export for the hybrid 2nm study.

  plot_pareto_dual   : 2-panel Pareto (σ_die–WPH and σ_res–WPH) with knees
  plot_sensitivity   : per-route signed-importance heatmaps + overall tornado
  write_sensitivity_md : variable-importance ranking tables → markdown
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np

from sic.hybrid.base import Recipe
from sic.hybrid.constraints_2nm import Constraints2nm
from sic.hybrid.optimize import RouteSearch, knee_point, strongest_route
from sic.hybrid.sensitivity import SensitivityResult, TARGETS

ROUTE_COLORS = {
    "laser+stealth": "#1b7f79",
    "laser+plasma":  "#c08a1e",
    "laser+blade":   "#b5374a",
}
_TARGET_LABEL = {
    "feasible": "feasible",
    "die_strength_MPa": "σ_die",
    "throughput_wph": "WPH",
    "residual_stress_MPa": "σ_res",
}
_PRETTY_VAR = {
    "groove_power_W": "groove power [W]",
    "groove_speed_mm_s": "groove speed [mm/s]",
    "groove_passes": "groove passes",
    "stealth_power_W": "stealth power [W]",
    "stealth_speed_mm_s": "stealth speed [mm/s]",
    "stealth_focal_depth_um": "stealth focal depth [µm]",
    "stealth_layers": "stealth layers",
    "plasma_pressure_mTorr": "plasma pressure [mTorr]",
    "plasma_etch_rate_um_min": "plasma etch rate [µm/min]",
    "blade_feed_mm_s": "blade feed [mm/s]",
    "blade_W_um": "blade width [µm]",
}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#444", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def pretty(v: str) -> str:
    return _PRETTY_VAR.get(v, v)


# ── Pareto (dual panel) ─────────────────────────────────────────────────────────
def plot_pareto_dual(searches: Dict[str, RouteSearch], recipe: Recipe,
                     out: str) -> str:
    plt = _style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))
    champ_name, _ = strongest_route(searches)

    for name, s in searches.items():
        if not s.pareto:
            continue
        c = ROUTE_COLORS.get(name, "k")
        pts = sorted(s.pareto, key=lambda p: p.throughput_wph)
        wph = [p.throughput_wph for p in pts]
        sig = [p.die_strength_MPa for p in pts]
        res = [p.residual_stress_MPa for p in pts]
        lbl = name + ("  ★" if name == champ_name else "")
        axL.plot(sig, wph, "o-", color=c, lw=2, ms=5, label=lbl, alpha=.9)
        axR.plot(res, wph, "o-", color=c, lw=2, ms=5, label=lbl, alpha=.9)
        knee = knee_point(s)
        axL.scatter([knee.die_strength_MPa], [knee.throughput_wph], marker="*",
                    s=320, color=c, edgecolor="k", zorder=6, linewidths=1.1)
        axR.scatter([knee.residual_stress_MPa], [knee.throughput_wph], marker="*",
                    s=320, color=c, edgecolor="k", zorder=6, linewidths=1.1)

    axL.axvline(Constraints2nm().die_strength_min_MPa, ls="--", color="#888",
                lw=1.2, label="min σ_die")
    axL.set(xlabel="die break strength σ_die [MPa]  (→ stronger)",
            ylabel="series throughput [WPH]  (→ faster)",
            title="(a) strength vs throughput")
    axR.set(xlabel="process residual stress σ_res [MPa]  (← lower is better)",
            ylabel="series throughput [WPH]  (→ faster)",
            title="(b) residual stress vs throughput  — the active SiC trade")
    axR.invert_xaxis()
    for ax in (axL, axR):
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Hybrid 2nm dicing — Pareto fronts  ({recipe.material}, "
                 f"street {recipe.street_width_um:.0f}µm, t={recipe.wafer_thickness_um:.0f}µm)"
                 "   ★ = Pareto knee", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ── Sensitivity (heatmaps + tornado) ────────────────────────────────────────────
def plot_sensitivity(results: Dict[str, SensitivityResult], recipe: Recipe,
                     out: str) -> str:
    plt = _style()
    import matplotlib.pyplot as _plt
    routes = list(results)
    n = len(routes)
    fig, axes = _plt.subplots(2, n, figsize=(5.2 * n, 9.0),
                              gridspec_kw={"height_ratios": [1.25, 1.0]},
                              constrained_layout=True)
    if n == 1:
        axes = axes.reshape(2, 1)

    tcols = list(TARGETS)
    tlabels = [_TARGET_LABEL[t] for t in tcols]
    for ci, route in enumerate(routes):
        r = results[route]
        order = [v for v, _ in r.overall_ranking()]          # sort vars by importance
        idx = [r.var_names.index(v) for v in order]
        M = np.array([r.spearman[t][idx] for t in tcols]).T   # (d, n_targets) signed ρ

        # ── heatmap ──
        ax = axes[0, ci]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(tcols))); ax.set_xticklabels(tlabels, fontsize=9)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([pretty(v) for v in order], fontsize=8.5)
        for i in range(len(order)):
            for j in range(len(tcols)):
                val = M[i, j]
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7.5, color=("white" if abs(val) > 0.55 else "#222"))
        ax.set_title(f"{route}\nsigned Spearman ρ  (feas={r.n_feasible}/{len(r.X)})")
        ax.grid(False)

        # ── overall-importance tornado ──
        axb = axes[1, ci]
        rk = r.overall_ranking()
        names = [pretty(v) for v, _ in rk][::-1]
        scores = [s for _, s in rk][::-1]
        axb.barh(names, scores, color=ROUTE_COLORS.get(route, "#555"),
                 edgecolor="k", linewidth=0.6, alpha=.92)
        for y, s in enumerate(scores):
            axb.text(s + 0.01, y, f"{s:.2f}", va="center", fontsize=8)
        axb.set_xlim(0, max(0.6, max(scores) * 1.18))
        axb.set_xlabel("overall importance  mean|ρ|")
        axb.set_title("variable ranking")
        axb.grid(axis="x", alpha=.4)

    cbar = fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("signed rank correlation ρ  (red = ↑var raises target)")
    fig.suptitle("Hybrid 2nm dicing — process-variable importance per route\n"
                 f"({recipe.material}, street {recipe.street_width_um:.0f}µm, "
                 f"low-k {recipe.lowk_stack_um:.0f}µm)",
                 fontweight="bold", fontsize=14)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ── Material sweep ────────────────────────────────────────────────────────────────
def plot_materials(df, recipe: Recipe, out: str) -> str:
    """4-panel material comparison from the sweep dataframe."""
    import pandas as pd  # noqa: F401
    from data.materials.material_properties import ALL_MATERIALS
    plt = _style()
    from sic.hybrid.constraints_2nm import Constraints2nm
    from sic.hybrid.dataset import summarize

    mats = [m for m in ["Si", "GaN", "Ga2O3", "SiC", "Diamond"] if m in set(df.material)]
    routes = list(ROUTE_COLORS)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    c2 = Constraints2nm()

    # (a) die-strength distribution by material
    a = ax[0, 0]
    data = [df[df.material == m].die_strength_MPa.values for m in mats]
    bp = a.boxplot(data, labels=mats, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set(facecolor="#cfe3e1", edgecolor="#1b7f79")
    for m, x in zip(mats, range(1, len(mats) + 1)):
        sf = ALL_MATERIALS[m]["sigma_flex"] / 1e6
        a.plot([x - 0.3, x + 0.3], [sf, sf], color="#b5374a", lw=2)
    a.axhline(c2.die_strength_min_MPa, ls="--", color="#888", label="min σ_die (250)")
    a.set(ylabel="die break strength σ_die [MPa]",
          title="(a) die strength by material  (red = σ_flex cap)")
    a.legend(fontsize=8, frameon=False)

    # (b) feasible-rate heatmap material×route
    s = summarize(df)
    def _pivot(col):
        return s.pivot(index="material", columns="route", values=col).reindex(mats)[routes]
    b = ax[0, 1]
    F = _pivot("feasible_rate").values
    im = b.imshow(F, cmap="YlGn", vmin=0, vmax=max(0.01, F.max()), aspect="auto")
    b.set_xticks(range(len(routes))); b.set_xticklabels(routes, fontsize=8, rotation=12)
    b.set_yticks(range(len(mats))); b.set_yticklabels(mats)
    for i in range(len(mats)):
        for j in range(len(routes)):
            b.text(j, i, f"{F[i, j]*100:.0f}%", ha="center", va="center", fontsize=9)
    b.set_title("(b) feasible rate  (material × route)"); b.grid(False)
    fig.colorbar(im, ax=b, fraction=0.046, pad=0.04)

    # (c) median chipping heatmap
    c = ax[1, 0]
    C = _pivot("chipping_med").values
    im2 = c.imshow(C, cmap="OrRd", aspect="auto")
    c.set_xticks(range(len(routes))); c.set_xticklabels(routes, fontsize=8, rotation=12)
    c.set_yticks(range(len(mats))); c.set_yticklabels(mats)
    for i in range(len(mats)):
        for j in range(len(routes)):
            c.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=9,
                   color=("white" if C[i, j] > np.nanmax(C) * 0.6 else "#222"))
    c.set_title("(c) median total chipping [µm]"); c.grid(False)
    fig.colorbar(im2, ax=c, fraction=0.046, pad=0.04)

    # (d) die strength vs K_Ic (the physics driver)
    d = ax[1, 1]
    for m in mats:
        kic = ALL_MATERIALS[m]["K_Ic"] / 1e6
        med = df[df.material == m].die_strength_MPa.median()
        d.scatter([kic], [med], s=120, color="#1b7f79", edgecolor="k", zorder=5)
        d.annotate(m, (kic, med), textcoords="offset points", xytext=(6, 4), fontsize=9)
    d.axhline(c2.die_strength_min_MPa, ls="--", color="#888")
    d.set(xlabel="fracture toughness K_Ic [MPa·m$^{0.5}$]",
          ylabel="median σ_die [MPa]",
          title="(d) strength ← toughness  (brittle materials fall below the cap)")

    fig.suptitle("Hybrid 2nm dicing — material sweep  "
                 f"(street {recipe.street_width_um:.0f}µm, t={recipe.wafer_thickness_um:.0f}µm)",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def write_dataset_card(df, recipe: Recipe, out_md: str,
                       csv_rel: str, fig_rel: str | None = None) -> str:
    from sic.hybrid.dataset import summarize, FEATURE_COLS
    s = summarize(df)
    lines: list[str] = []
    A = lines.append
    A("# ハイブリッド2nmダイシング — 材料スイープ・データセット (Data Card)\n")
    n_geo = df[["street_width_um", "wafer_thickness_um"]].drop_duplicates().shape[0]
    n_rec = len(df) // max(1, df.material.nunique() * df.route.nunique() * n_geo)
    A(f"- 生成: `sic/hybrid/dataset.py` / `pipeline/hybrid_dicing_pipeline.py --dataset`")
    A(f"- 行数: **{len(df)}**  ( {df.material.nunique()} 材料 × {df.route.nunique()} ルート"
      f" × {n_geo} ジオメトリ × {n_rec} レシピ )")
    geos = sorted(set(zip(df.street_width_um, df.wafer_thickness_um)))
    A("- ジオメトリ (street µm / thickness µm): "
      + ", ".join(f"{int(s)}/{int(t)}" for s, t in geos))
    A("- 強度しきい値: 材料相対 (0.6·σ_flex) → 脆性材も可行勾配を持つ")
    A(f"- CSV: [`{os.path.basename(csv_rel)}`]({csv_rel})\n")
    if fig_rel:
        A(f"![material sweep]({fig_rel})\n")
    A("## スキーマ\n")
    A("**キー**: `material`, `route`")
    A("**特徴量 (レシピ)**: " + ", ".join(f"`{c}`" for c in FEATURE_COLS))
    A("**Stage出力**: `a_*`(レーザ溝), `b_*`(個片化)  例: `a_lowk_cleared_um`, "
      "`b_residual_MPa`, `b_ssd_um`")
    A("**ラベル/ターゲット**: `feasible`(0/1), `die_strength_MPa`, `total_chipping_um`, "
      "`throughput_wph`, `residual_stress_MPa`, `governing_crack_um`, `n_violations`\n")
    A("## 推奨タスク\n")
    A("- 分類: `feasible` (2nm制約充足) を特徴量から予測")
    A("- 回帰: `die_strength_MPa` / `total_chipping_um` / `residual_stress_MPa`")
    A("- 材料汎化: 4材料で学習→hold-out材料 (例 Ga2O3) でOODテスト\n")
    A("## 材料×ルート サマリ\n")
    A("| material | route | n | feasible% | σ_die中央 | chip中央[µm] | σ_res中央 | WPH中央 |")
    A("|---|---|--:|--:|--:|--:|--:|--:|")
    for _, r in s.iterrows():
        A(f"| {r.material} | {r.route} | {int(r.n)} | {r.feasible_rate*100:.0f}% | "
          f"{r.die_strength_med:.0f} | {r.chipping_med:.2f} | {r.residual_med:.1f} | "
          f"{r.throughput_med:.2f} |")
    A("")
    A("## 既知の限界（重要）\n")
    A("- **レーザ光学/溝モデルは SiC 校正**（屈折率・アブレーション閾値・溝の熱定数）。"
      "したがって `feasible`/`a_haz_um`/`a_lowk_cleared_um` の材料依存は弱い。")
    A("- **材料固有シグナルは破壊/熱KPI**（`die_strength_MPa`, `*_chip_um`, `b_ssd_um`, "
      "`*_residual_MPa`）に出る。各材料の K_Ic/H/E・熱物性で駆動。")
    A("- 物性は literature-typical の order-of-magnitude プロトタイプ。実fab/DISCOデータ校正前提。")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return out_md


# ── Markdown export ──────────────────────────────────────────────────────────────
def write_sensitivity_md(results: Dict[str, SensitivityResult], recipe: Recipe,
                         out_md: str, fig_rel: str | None = None) -> str:
    lines: list[str] = []
    A = lines.append
    A("# ハイブリッド2nmダイシング — 重要変数ランキング (感度分析)\n")
    A("> 各ルートでレシピ空間をLHSサンプリング→フルモデル実行し、各KPIへの"
      "影響度を **符号付きSpearman順位相関 ρ** で評価。\n")
    A(f"- 材料: **{recipe.material}**, ストリート {recipe.street_width_um:.0f}µm, "
      f"厚さ {recipe.wafer_thickness_um:.0f}µm, low-k {recipe.lowk_stack_um:.0f}µm")
    A("- 指標: |ρ|=0 影響なし … |ρ|→1 強い単調影響。符号 + は「変数↑でKPI↑」。")
    A("- KPI: `feasible`(2nm制約), `σ_die`(破壊強度), `WPH`(スループット), "
      "`σ_res`(プロセス残留応力)\n")
    if fig_rel:
        A(f"![variable importance]({fig_rel})\n")

    # headline: top driver per (route, target)
    A("## ヘッドライン — 各KPIの最重要ドライバ\n")
    A("| route | feasible | σ_die | WPH | σ_res |")
    A("|---|---|---|---|---|")
    for route, r in results.items():
        cells = [route]
        for t in TARGETS:
            rk = r.ranking(t)
            v, rho, _ = rk[0]
            cells.append(f"{pretty(v)} ({rho:+.2f})")
        A("| " + " | ".join(cells) + " |")
    A("")

    # per-route detail
    for route, r in results.items():
        A(f"## {route}  (feasible {r.n_feasible}/{len(r.X)})\n")
        A("### 総合ランキング (mean|ρ| over 4 KPI)\n")
        A("| 順位 | 変数 | 総合重要度 mean\\|ρ\\| |")
        A("|---:|---|---:|")
        for i, (v, sc) in enumerate(r.overall_ranking(), 1):
            A(f"| {i} | {pretty(v)} | {sc:.3f} |")
        A("")
        A("### KPI別 符号付きρ (と標準化β)\n")
        head = "| 変数 | " + " | ".join(_TARGET_LABEL[t] for t in TARGETS) + " |"
        A(head)
        A("|---|" + "---:|" * len(TARGETS))
        for v in [vv for vv, _ in r.overall_ranking()]:
            j = r.var_names.index(v)
            cells = [pretty(v)]
            for t in TARGETS:
                cells.append(f"{r.spearman[t][j]:+.2f} ({r.betas[t][j]:+.2f})")
            A("| " + " | ".join(cells) + " |")
        A("")

    A("## 読み方メモ\n")
    A("- **feasible列が実質的な設計ノブ**: 2nmで効くのは「low-kを切り切る溝深さ」"
      "と「HAZを抑える」変数。`groove_passes`/`groove_speed`が支配的なら、"
      "歩留りはレーザグルービング条件で決まる。")
    A("- **σ_dieがほぼ無相関 (|ρ|≈0)**: SiCはµm級欠陥を許容し強度が母材律速"
      "(pristine cap)になるため。強度は2nm SiCの律速KPIではない。")
    A("- **σ_res / WPH が実トレード**: 非接触ルート(stealth/plasma)はσ_resが低く、"
      "WPHはStage Aグルービングのpass数/速度で律速される。")

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return out_md
