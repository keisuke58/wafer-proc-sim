"""
TEL 装置競争力 — 装置スペック → デバイス FOM / 歩留まり 写像
==================================================================
Tokyo Electron (TEL) の前工程装置が「装置スペックの差」をどれだけデバイス性能と
歩留まりに変換するかを、検証済み forward model (fem/tel_feol_recipe.py) の上で
定量化する。市場予測ではなく「物理 → 歩留まり」感度解析である点に注意。

評価する 3 軸 (TEL 装置 vs 競合装置):
    (a) ALD 膜厚均一性
        TEL Triase+ (~±0.3%) vs 競合 (~±1.0%)
        → ald_gpc_scale のウェハ間ばらつき → EOT スプレッド / µ_ch 分布
    (b) 洗浄効果 (CELLESTA)
        TEL pregate_sic (SiC 専用強洗浄) vs 競合の弱シーケンス pregate_si
        → 残留炭素由来 Dit → µ_ch
    (c) 低ダメージエッチ
        TEL 低損傷 Cl2 vs 競合 高損傷 SF6
        → RIE 損傷/HAZ 残渣 → Dit

各軸でモンテカルロ (~300 ウェハ) を回し、装置由来の
プロセスばらつき (ALD GPC ドリフト / 酸化炉温度ドリフト) を
simulate_recipe(tool_state=...) に伝播。デバイス FOM 分布と
    YIELD = fraction( µ_ch >= threshold AND V_th ∈ [2,4] )
を計算し、TEL vs 競合の競争力スコア表 (yield delta) を作る。

この forward model の特性 (検証済):
    - µ_ch はほぼ Dit 律速 (Coulomb 散乱) → 洗浄軸が最大レバー
    - V_th は製造ウィンドウ [2,4]V にほぼ常に収まる (Dit/Cox 項が小さい)
      → 歩留まりは実質 µ_ch 基準で決まる
    - ALD 均一性軸は EOT スプレッド (プロセスマージン) が主指標
    - エッチ軸は本モデルでは小レバー (正直に小さい delta として報告)

出力:
    results/tel_equipment_competitiveness.png   yield + µ_ch 分布 (TEL vs 競合 ×3軸)
    results/tel_equipment_competitiveness.json  競争力スコア表

実行:
    python fem/tel_equipment_competitiveness.py
    python fem/tel_equipment_competitiveness.py --wafers 500 --seed 7

参考: Kimoto & Cooper (2014); Saks (1999); George (2010);
      TEL Triase+ / CELLESTA / Tactras spec sheets (公開値の桁感)
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, "/home/nishioka/git/wafer-proc-sim")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fem.tel_feol_recipe import simulate_recipe, default_recipe

OUT_DIR = "/home/nishioka/git/wafer-proc-sim/results"
FIG_PATH = os.path.join(OUT_DIR, "tel_equipment_competitiveness.png")
JSON_PATH = os.path.join(OUT_DIR, "tel_equipment_competitiveness.json")

# ════════════════════════════════════════════════════════════════════════════
# 共通: ウェハ間プロセスばらつきの装置由来モデル
# ════════════════════════════════════════════════════════════════════════════
# どの軸でも残る基礎ノイズ (酸化炉温度ドリフト)。装置由来ばらつきの土台。
OX_DRIFT_SIGMA_C = 4.0      # 酸化炉温度ドリフト 1σ [°C]
ALD_BASE_SIGMA = 0.003      # ALD GPC の基礎ばらつき (TEL クラス) 1σ

# 製造ウィンドウ
VTH_WINDOW = (2.0, 4.0)


def _run_wafers(base_recipe, ald_sigma, n_wafers, rng, clean_seq=None, rie_gas=None):
    """
    1 シナリオ分のモンテカルロ。装置由来ばらつきを tool_state に伝播して
    n_wafers 枚のウェハ FOM を返す。

    ald_sigma : ALD GPC ばらつき 1σ (装置の膜厚均一性スペックを表す)
    clean_seq / rie_gas : 指定すれば recipe を上書き (洗浄/エッチ軸用)
    """
    recipe = copy.deepcopy(base_recipe)
    if clean_seq is not None:
        recipe["clean_seq"] = clean_seq
    if rie_gas is not None:
        recipe["rie_gas"] = rie_gas

    mu = np.empty(n_wafers)
    eot = np.empty(n_wafers)
    vth = np.empty(n_wafers)
    dit = np.empty(n_wafers)
    ron = np.empty(n_wafers)
    for i in range(n_wafers):
        ts = {
            "ald_gpc_scale": float(rng.normal(1.0, ald_sigma)),
            "ox_T_offset_C": float(rng.normal(0.0, OX_DRIFT_SIGMA_C)),
        }
        f = simulate_recipe(recipe, ts)
        mu[i] = f["mu_ch"]
        eot[i] = f["EOT_nm"]
        vth[i] = f["V_th"]
        dit[i] = f["Dit"]
        ron[i] = f["R_on_sp"]
    return {"mu_ch": mu, "EOT_nm": eot, "V_th": vth, "Dit": dit, "R_on_sp": ron}


def _yield(samples, mu_thresh, eot_window=None):
    """
    YIELD = fraction( µ_ch >= thresh AND V_th ∈ [2,4] [AND EOT ∈ eot_window] )。

    eot_window : (lo, hi) を渡すと EOT 仕様窓も歩留まり条件に加える。ALD 膜厚
                 均一性は EOT スペックを直接律速するので、この軸ではこれが本質。
    """
    ok = samples["mu_ch"] >= mu_thresh
    ok &= (samples["V_th"] >= VTH_WINDOW[0]) & (samples["V_th"] <= VTH_WINDOW[1])
    if eot_window is not None:
        lo, hi = eot_window
        ok &= (samples["EOT_nm"] >= lo) & (samples["EOT_nm"] <= hi)
    return float(np.mean(ok))


def _stats(arr):
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3 軸の競争力スタディ
# ════════════════════════════════════════════════════════════════════════════

def run_competitiveness(n_wafers=300, seed=0):
    """TEL vs 競合を 3 軸 (ALD/洗浄/エッチ) で評価し、スコア表とサンプルを返す。"""
    rng = np.random.default_rng(seed)
    base = default_recipe()  # Al2O3, blade, pregate_sic, CF4/O2

    axes = {}

    # ── (a) ALD 膜厚均一性軸 ───────────────────────────────────────────
    # TEL Triase+ ±0.3% vs 競合 ±1.0%。膜厚均一性は EOT を直接律速する。
    # µ_ch はほぼ Dit 律速で gpc にほぼ不感なので、この軸の歩留まりは
    # 「EOT が仕様窓に入るか」で決まる (gate スタックの容量整合)。
    # 仕様窓 = 公称 EOT ±1.5% (高密度パワーデバイスの厳しい EOT 管理幅)。
    # この窓だと TEL(±0.3%)≒5σ で 100% 通すが、競合(±1.0%)は ~1.5σ で歩留落ち。
    ald_base = copy.deepcopy(base)
    eot_nominal = simulate_recipe(ald_base, {"ald_gpc_scale": 1.0})["EOT_nm"]
    eot_win = (eot_nominal * 0.985, eot_nominal * 1.015)
    tel_ald = _run_wafers(ald_base, ald_sigma=0.003, n_wafers=n_wafers, rng=rng)
    cmp_ald = _run_wafers(ald_base, ald_sigma=0.010, n_wafers=n_wafers, rng=rng)
    # µ_ch 閾値は緩く (両者の最小値以下) し、EOT 仕様窓を主条件にする
    th_ald = float(min(tel_ald["mu_ch"].min(), cmp_ald["mu_ch"].min()) - 1.0)
    axes["ALD_uniformity"] = {
        "description": "ALD thickness uniformity: TEL Triase+ (~+/-0.3%) vs competitor (~+/-1.0%)",
        "lever": "ald_gpc_scale wafer-to-wafer sigma -> EOT spread -> EOT-spec yield",
        "tel_spec": "GPC sigma = 0.3% (Triase+)",
        "competitor_spec": "GPC sigma = 1.0%",
        "mu_threshold": th_ald,
        "eot_nominal_nm": float(eot_nominal),
        "eot_spec_window_nm": [float(eot_win[0]), float(eot_win[1])],
        "tel": {
            "yield": _yield(tel_ald, th_ald, eot_window=eot_win),
            "mu_ch": _stats(tel_ald["mu_ch"]),
            "EOT_nm": _stats(tel_ald["EOT_nm"]),
            "EOT_3sigma_spread_nm": float(6.0 * np.std(tel_ald["EOT_nm"])),
        },
        "competitor": {
            "yield": _yield(cmp_ald, th_ald, eot_window=eot_win),
            "mu_ch": _stats(cmp_ald["mu_ch"]),
            "EOT_nm": _stats(cmp_ald["EOT_nm"]),
            "EOT_3sigma_spread_nm": float(6.0 * np.std(cmp_ald["EOT_nm"])),
        },
        "samples": {"tel": tel_ald, "competitor": cmp_ald},
    }
    axes["ALD_uniformity"]["yield_delta"] = (
        axes["ALD_uniformity"]["tel"]["yield"]
        - axes["ALD_uniformity"]["competitor"]["yield"]
    )
    # EOT 均一性比 (小さいほど良い競合に対する TEL の優位)
    axes["ALD_uniformity"]["EOT_spread_ratio_comp_over_tel"] = (
        axes["ALD_uniformity"]["competitor"]["EOT_3sigma_spread_nm"]
        / max(axes["ALD_uniformity"]["tel"]["EOT_3sigma_spread_nm"], 1e-9)
    )

    # ── (b) 洗浄効果軸 (CELLESTA) ──────────────────────────────────────
    # TEL pregate_sic (SiC 専用強洗浄) vs 競合 pregate_si (弱・残留炭素多い)。
    # 残留炭素 → Dit → µ_ch の最大レバー。
    tel_cln = _run_wafers(base, ald_sigma=ALD_BASE_SIGMA, n_wafers=n_wafers, rng=rng,
                          clean_seq="pregate_sic")
    cmp_cln = _run_wafers(base, ald_sigma=ALD_BASE_SIGMA, n_wafers=n_wafers, rng=rng,
                          clean_seq="pregate_si")
    # 閾値: 弱洗浄の µ_ch を上回り、強洗浄なら通る位置 (両分布の中間)
    th_cln = 0.5 * (np.mean(tel_cln["mu_ch"]) + np.mean(cmp_cln["mu_ch"]))
    th_cln = float(th_cln)
    axes["clean_effectiveness"] = {
        "description": "Cleaning effectiveness: TEL CELLESTA pregate_sic vs weaker pregate_si",
        "lever": "residual carbon Dit -> mu_ch",
        "tel_spec": "pregate_sic (SiC-dedicated strong clean)",
        "competitor_spec": "pregate_si (weaker, more residual carbon)",
        "mu_threshold": th_cln,
        "tel": {
            "yield": _yield(tel_cln, th_cln),
            "mu_ch": _stats(tel_cln["mu_ch"]),
            "Dit": _stats(tel_cln["Dit"]),
        },
        "competitor": {
            "yield": _yield(cmp_cln, th_cln),
            "mu_ch": _stats(cmp_cln["mu_ch"]),
            "Dit": _stats(cmp_cln["Dit"]),
        },
        "samples": {"tel": tel_cln, "competitor": cmp_cln},
    }
    axes["clean_effectiveness"]["yield_delta"] = (
        axes["clean_effectiveness"]["tel"]["yield"]
        - axes["clean_effectiveness"]["competitor"]["yield"]
    )
    axes["clean_effectiveness"]["mu_ch_gain_pct"] = float(
        100.0 * (np.mean(tel_cln["mu_ch"]) - np.mean(cmp_cln["mu_ch"]))
        / np.mean(cmp_cln["mu_ch"]))

    # ── (c) 低ダメージエッチ軸 ─────────────────────────────────────────
    # TEL 低損傷 Cl2 vs 競合 高損傷 SF6。HAZ が小さく表面状態が支配的になる
    # stealth ダイシング上で評価。歩留まり閾値は両分布の下に置き (合格ライン)、
    # 軸の本質は µ_ch/Dit の差そのものとして正直に報告する。
    # 重要: この forward model では RIE ガスは小レバー。実際 etch_nm の大きい
    # SF6 は HAZ をより多く除去して Dit を僅かに下げる (damage_nm は Dit に
    # 未結合)。よって本モデル上は SF6 がごく僅かに µ_ch 高 = TEL 優位ではない。
    # 物理予測ではなくモデル感度として、この事実を隠さず delta を出す。
    etch_base = copy.deepcopy(base)
    etch_base["dicing_process"] = "stealth"
    tel_etch = _run_wafers(etch_base, ald_sigma=ALD_BASE_SIGMA, n_wafers=n_wafers, rng=rng,
                           rie_gas="Cl2")
    cmp_etch = _run_wafers(etch_base, ald_sigma=ALD_BASE_SIGMA, n_wafers=n_wafers, rng=rng,
                           rie_gas="SF6")
    # 閾値は両分布の最小値以下: V_th 窓のみが効き、µ_ch 基準は両者合格
    th_etch = float(min(tel_etch["mu_ch"].min(), cmp_etch["mu_ch"].min()) - 1.0)
    axes["low_damage_etch"] = {
        "description": "Low-damage etch: TEL low-damage Cl2 vs high-damage SF6",
        "lever": "RIE damage / HAZ residue -> Dit (MINOR lever on this forward model)",
        "tel_spec": "Cl2 low-damage RIE",
        "competitor_spec": "SF6 high-damage RIE",
        "mu_threshold": th_etch,
        "tel": {
            "yield": _yield(tel_etch, th_etch),
            "mu_ch": _stats(tel_etch["mu_ch"]),
            "Dit": _stats(tel_etch["Dit"]),
        },
        "competitor": {
            "yield": _yield(cmp_etch, th_etch),
            "mu_ch": _stats(cmp_etch["mu_ch"]),
            "Dit": _stats(cmp_etch["Dit"]),
        },
        "samples": {"tel": tel_etch, "competitor": cmp_etch},
    }
    axes["low_damage_etch"]["yield_delta"] = (
        axes["low_damage_etch"]["tel"]["yield"]
        - axes["low_damage_etch"]["competitor"]["yield"]
    )
    axes["low_damage_etch"]["mu_ch_delta_cm2Vs"] = float(
        np.mean(tel_etch["mu_ch"]) - np.mean(cmp_etch["mu_ch"]))
    axes["low_damage_etch"]["note"] = (
        "MINOR lever on this forward model: damage_nm is not coupled into Dit; only "
        "etch_nm (HAZ removal) is, so high-etch SF6 looks ~equal or marginally better "
        "on mu_ch. Yield is set by the V_th window only here. Reported honestly as a "
        "near-tie -- the model does not support a strong low-damage-etch yield claim."
    )

    meta = {
        "n_wafers": int(n_wafers),
        "seed": int(seed),
        "base_recipe": base,
        "ox_drift_sigma_C": OX_DRIFT_SIGMA_C,
        "vth_window": list(VTH_WINDOW),
        "yield_definition": "fraction(mu_ch >= axis_threshold AND V_th in [2,4])",
        "caveat": ("Physics-to-yield sensitivity study on a verified forward model, "
                   "NOT a market forecast. mu_ch is Dit-limited so the clean axis "
                   "dominates; V_th stays inside the window so yield tracks mu_ch."),
    }
    return axes, meta


# ════════════════════════════════════════════════════════════════════════════
# 出力: JSON 表 + 図
# ════════════════════════════════════════════════════════════════════════════

def _strip_samples(axes):
    """JSON 用に巨大な MC サンプル配列を除いたコピーを返す。"""
    out = {}
    for name, ax in axes.items():
        out[name] = {k: v for k, v in ax.items() if k != "samples"}
    return out


def save_json(axes, meta):
    table = {"meta": meta, "axes": _strip_samples(axes)}
    with open(JSON_PATH, "w") as fp:
        json.dump(table, fp, indent=2, ensure_ascii=False)
    print(f"[OK] saved -> {JSON_PATH}")
    return table


def make_figure(axes):
    axis_keys = ["ALD_uniformity", "clean_effectiveness", "low_damage_etch"]
    titles = ["(a) ALD uniformity\nTriase+ vs competitor",
              "(b) Clean (CELLESTA)\npregate_sic vs pregate_si",
              "(c) Low-damage etch\nCl2 vs SF6"]
    tel_c = "#1f77b4"
    cmp_c = "#d62728"

    fig, axarr = plt.subplots(2, 3, figsize=(14, 8))

    # 上段: 歩留まりを律速する量の分布。
    #   ALD 軸 → EOT (仕様窓が歩留まりを決める) / 洗浄・エッチ軸 → µ_ch。
    for j, key in enumerate(axis_keys):
        ax = axarr[0, j]
        s = axes[key]["samples"]
        if key == "ALD_uniformity":
            metric = "EOT_nm"
            data = [s["tel"]["EOT_nm"], s["competitor"]["EOT_nm"]]
        else:
            metric = "mu_ch"
            data = [s["tel"]["mu_ch"], s["competitor"]["mu_ch"]]
        parts = ax.violinplot(data, positions=[1, 2], showmeans=True,
                              showextrema=True, widths=0.8)
        for k, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(tel_c if k == 0 else cmp_c)
            pc.set_alpha(0.55)
        for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
            parts[partname].set_color("#333333")
            parts[partname].set_linewidth(1.0)
        if key == "ALD_uniformity":
            lo, hi = axes[key]["eot_spec_window_nm"]
            ax.axhspan(lo, hi, color="green", alpha=0.12,
                       label=f"EOT spec ±1.5% [{lo:.2f},{hi:.2f}]")
            ax.axhline(lo, color="green", ls="--", lw=1.0)
            ax.axhline(hi, color="green", ls="--", lw=1.0)
            ax.set_ylabel("EOT  [nm]  (drives ALD yield)")
        else:
            th = axes[key]["mu_threshold"]
            ax.axhline(th, color="green", ls="--", lw=1.2,
                       label=f"µ_ch thresh = {th:.1f}")
            if j == 1:
                ax.set_ylabel("channel mobility  µ_ch  [cm²/Vs]")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["TEL", "competitor"])
        ax.set_title(titles[j], fontsize=10)
        ax.legend(fontsize=7.5, loc="lower left")
        ax.grid(alpha=0.25, axis="y")

    # 下段: 歩留まりバー (TEL vs 競合) + yield delta 注記
    for j, key in enumerate(axis_keys):
        ax = axarr[1, j]
        y_tel = axes[key]["tel"]["yield"] * 100.0
        y_cmp = axes[key]["competitor"]["yield"] * 100.0
        bars = ax.bar([0, 1], [y_tel, y_cmp], color=[tel_c, cmp_c],
                      width=0.6, alpha=0.85)
        for b, v in zip(bars, [y_tel, y_cmp]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=9)
        delta = axes[key]["yield_delta"] * 100.0
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["TEL", "competitor"])
        ax.set_ylim(0, 109)
        if j == 0:
            ax.set_ylabel("YIELD  [%]\n(spec & V_th∈[2,4])")
        ax.set_title(f"Δyield = {delta:+.1f} pts", fontsize=10)
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle("TEL equipment competitiveness — tool spec → device FOM → yield "
                 "(physics-to-yield sensitivity, not market forecast)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_PATH, dpi=130)
    plt.close(fig)
    print(f"[OK] saved -> {FIG_PATH}")


def print_table(axes, meta):
    print(f"\n=== TEL equipment competitiveness  (n_wafers={meta['n_wafers']}, "
          f"seed={meta['seed']}) ===")
    print(f"{'axis':22} {'TEL yield':>10} {'comp yield':>11} {'Δyield':>9} "
          f"{'µ_ch TEL':>9} {'µ_ch comp':>10}")
    for key, ax in axes.items():
        print(f"{key:22} {ax['tel']['yield']*100:9.1f}% "
              f"{ax['competitor']['yield']*100:10.1f}% "
              f"{ax['yield_delta']*100:+8.1f} "
              f"{ax['tel']['mu_ch']['mean']:9.1f} "
              f"{ax['competitor']['mu_ch']['mean']:10.1f}")
    a = axes["ALD_uniformity"]
    win = a["eot_spec_window_nm"]
    print(f"\nALD axis  (yield = EOT in spec window [{win[0]:.3f}, {win[1]:.3f}] nm, "
          f"±1.5% of {a['eot_nominal_nm']:.2f} nm):")
    print(f"  EOT 3σ spread:  TEL {a['tel']['EOT_3sigma_spread_nm']:.3f} nm  "
          f"vs competitor {a['competitor']['EOT_3sigma_spread_nm']:.3f} nm  "
          f"(×{a['EOT_spread_ratio_comp_over_tel']:.2f} wider)")
    c = axes["clean_effectiveness"]
    print(f"Clean axis: µ_ch gain (TEL over competitor): "
          f"{c['mu_ch_gain_pct']:+.1f}%  (DOMINANT lever)")
    e = axes["low_damage_etch"]
    print(f"Etch axis:  µ_ch delta (Cl2 - SF6): "
          f"{e['mu_ch_delta_cm2Vs']:+.2f} cm²/Vs  (MINOR lever; see note)")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="TEL equipment competitiveness: tool spec -> device FOM -> yield")
    ap.add_argument("--wafers", type=int, default=300,
                    help="Monte-Carlo ウェハ枚数 / シナリオ (default 300)")
    ap.add_argument("--seed", type=int, default=0, help="乱数シード")
    args = ap.parse_args()

    axes, meta = run_competitiveness(n_wafers=args.wafers, seed=args.seed)
    print_table(axes, meta)
    save_json(axes, meta)
    make_figure(axes)


if __name__ == "__main__":
    main()
