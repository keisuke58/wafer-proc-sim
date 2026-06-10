"""
TEL 洗浄モデル — CELLESTA / CLEAN TRACK 相当
=============================================
半導体前工程の洗浄ステップを物理モデル化。

洗浄シーケンス:
    [1] Post-CMP 洗浄      研磨スラリー + パーティクル除去
    [2] Post-Dicing 洗浄   ダイシング屑 + 冷却水残留物
    [3] Pre-Gate 洗浄      ゲート酸化前の最重要洗浄 (Dit に直結)

SiC 固有の課題:
    - 化学的安定性が高い → 通常の RCA では不十分
    - Piranha + HF シーケンスが標準
    - Megasonic 照射が粒子除去に必須
    - C-face と Si-face で最適シーケンスが異なる

実行:
    python fem/tel_cleaning_model.py
    python fem/tel_cleaning_model.py --sequence pregate
    python fem/tel_cleaning_model.py --compare-materials
"""

import argparse
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

kB = 8.617e-5  # eV/K

# ── 洗浄薬液パラメータ ──────────────────────────────────────────────────────
CHEMICALS = {
    "SC1":      {"T_C": 70,  "time_min": 10, "target": "particle+organic",
                 "etch_SiO2_nm_min": 0.5,  "E_act_eV": 0.65,
                 "particle_removal": 0.90, "metal_removal": 0.30,
                 "carbon_removal":   0.85, "color": "#2166ac",
                 "formula": "NH₄OH:H₂O₂:H₂O = 1:1:5"},
    "SC2":      {"T_C": 70,  "time_min": 10, "target": "metal",
                 "etch_SiO2_nm_min": 0.1,  "E_act_eV": 0.55,
                 "particle_removal": 0.30, "metal_removal": 0.99,
                 "carbon_removal":   0.20, "color": "#d62728",
                 "formula": "HCl:H₂O₂:H₂O = 1:1:6"},
    "Piranha":  {"T_C": 120, "time_min": 15, "target": "heavy organic",
                 "etch_SiO2_nm_min": 0.2,  "E_act_eV": 0.80,
                 "particle_removal": 0.70, "metal_removal": 0.60,
                 "carbon_removal":   0.999,"color": "#ff7f0e",
                 "formula": "H₂SO₄:H₂O₂ = 4:1"},
    "HF dip":   {"T_C": 25,  "time_min": 1,  "target": "native oxide",
                 "etch_SiO2_nm_min": 2.0,  "E_act_eV": 0.20,
                 "particle_removal": 0.20, "metal_removal": 0.10,
                 "carbon_removal":   0.10, "color": "#9467bd",
                 "formula": "HF:H₂O = 1:50"},
    "Megasonic":{"T_C": 40,  "time_min": 5,  "target": "particle",
                 "etch_SiO2_nm_min": 0.0,  "E_act_eV": 0.0,
                 "particle_removal": 0.99, "metal_removal": 0.05,
                 "carbon_removal":   0.30, "color": "#2ca02c",
                 "formula": "DIW + 800kHz 超音波"},
    "O3 water": {"T_C": 25,  "time_min": 5,  "target": "organic",
                 "etch_SiO2_nm_min": 0.1,  "E_act_eV": 0.35,
                 "particle_removal": 0.40, "metal_removal": 0.20,
                 "carbon_removal":   0.95, "color": "#8c564b",
                 "formula": "O₃(10ppm)/DIW"},
    "H2 water": {"T_C": 25,  "time_min": 10, "target": "particle (redeposition prevention)",
                 "etch_SiO2_nm_min": 0.0,  "E_act_eV": 0.10,
                 # H₂ が表面を負帯電 → 粒子（負帯電）を静電反発で再付着防止
                 "particle_removal": 0.92, "metal_removal": 0.85,
                 "carbon_removal":   0.30, "color": "#00aaff",
                 "formula": "H₂(1ppm)/DIW — 還元性機能水"},
}

# ── 標準洗浄シーケンス ─────────────────────────────────────────────────────
SEQUENCES = {
    "post_cmp": {
        "label": "Post-CMP 洗浄 (TEL CELLESTA)",
        # H₂水 + Megasonic でスラリー粒子を再付着させずに除去、最後に SC2 で金属
        "steps": ["H2 water", "Megasonic", "HF dip", "SC2"],
        "target": "スラリーパーティクル + 金属汚染",
    },
    "post_dicing": {
        "label": "Post-Dicing 洗浄",
        # ダイシング屑は粒子が大きく H₂水 + Megasonic が最効率
        "steps": ["H2 water", "Megasonic", "SC2"],
        "target": "ダイシング屑 + 冷却水ミネラル",
    },
    "pregate_si": {
        "label": "Pre-Gate 洗浄 (Si 標準 RCA)",
        "steps": ["SC1", "HF dip", "SC2", "HF dip"],
        "target": "有機物 + 酸化膜 + 金属 (Si 用)",
    },
    "pregate_sic": {
        "label": "Pre-Gate 洗浄 (SiC 最適化)",
        "steps": ["Piranha", "HF dip", "SC2", "O3 water", "HF dip"],
        "target": "C クラスター + 表面炭素 + 金属 (SiC 専用)",
    },
}


# ════════════════════════════════════════════════════════════════════════════
# 1. 洗浄効果モデル
# ════════════════════════════════════════════════════════════════════════════

class SurfaceState:
    """ウェーハ表面の汚染状態を保持するクラス。"""
    def __init__(self, process: str = "blade"):
        # 初期汚染量（プロセスに依存）
        init = {
            "blade":     {"particle": 1e10, "metal": 1e11, "carbon": 5e13, "oxide_nm": 2.0},
            "laser_ps":  {"particle": 5e9,  "metal": 5e10, "carbon": 2e13, "oxide_nm": 1.5},
            "stealth":   {"particle": 1e9,  "metal": 1e10, "carbon": 5e12, "oxide_nm": 1.0},
            "post_cmp":  {"particle": 5e10, "metal": 2e11, "carbon": 1e13, "oxide_nm": 3.0},
        }
        d = init.get(process, init["blade"])
        self.particle_cm2  = d["particle"]   # cm⁻²
        self.metal_cm2     = d["metal"]      # cm⁻²
        self.carbon_cm2    = d["carbon"]     # cm⁻²  (C cluster, Dit 前駆体)
        self.oxide_nm      = d["oxide_nm"]   # 自然酸化膜厚
        self.history       = []

    def apply_step(self, chemical: str, T_C: float = None,
                   megasonic: bool = False) -> dict:
        """1洗浄ステップを適用し表面状態を更新。"""
        c  = CHEMICALS[chemical]
        T  = T_C if T_C else c["T_C"]

        # 温度依存の反応速度補正
        if c["E_act_eV"] > 0:
            rate_factor = math.exp(-c["E_act_eV"] / (kB * (T + 273)))
            rate_factor /= math.exp(-c["E_act_eV"] / (kB * (c["T_C"] + 273)))
        else:
            rate_factor = 1.0

        # Megasonic 追加効果（粒子除去 ×3）
        ms_boost = 3.0 if megasonic and chemical not in ["Megasonic"] else 1.0

        pr = min(1.0, c["particle_removal"] * rate_factor * ms_boost)
        mr = min(1.0, c["metal_removal"]    * rate_factor)
        cr = min(1.0, c["carbon_removal"]   * rate_factor)

        before = {
            "particle": self.particle_cm2,
            "metal":    self.metal_cm2,
            "carbon":   self.carbon_cm2,
            "oxide_nm": self.oxide_nm,
        }

        self.particle_cm2 *= (1 - pr)
        self.metal_cm2    *= (1 - mr)
        self.carbon_cm2   *= (1 - cr)
        self.oxide_nm     = max(0, self.oxide_nm - c["etch_SiO2_nm_min"] * c["time_min"])

        after = {
            "particle": self.particle_cm2,
            "metal":    self.metal_cm2,
            "carbon":   self.carbon_cm2,
            "oxide_nm": self.oxide_nm,
        }
        entry = {"step": chemical, "before": before, "after": after}
        self.history.append(entry)
        return entry

    def dit_contribution(self) -> float:
        """
        残留炭素汚染 → SiC/SiO₂ 界面 Dit への寄与推定。
        Saks et al. (1999): Dit ∝ C_surface^0.7 (炭素クラスター起源)
        """
        C_ref = 5e12   # cm⁻² (無処理時の典型値)
        Dit_ref = 2e12  # cm⁻²eV⁻¹
        return Dit_ref * (self.carbon_cm2 / C_ref) ** 0.7


def run_sequence(seq_name: str, process: str = "blade") -> dict:
    """洗浄シーケンスを実行し、前後の表面状態を返す。"""
    seq   = SEQUENCES[seq_name]
    state = SurfaceState(process)

    before = {
        "particle": state.particle_cm2,
        "metal":    state.metal_cm2,
        "carbon":   state.carbon_cm2,
        "oxide_nm": state.oxide_nm,
    }

    for step in seq["steps"]:
        state.apply_step(step)

    after = {
        "particle": state.particle_cm2,
        "metal":    state.metal_cm2,
        "carbon":   state.carbon_cm2,
        "oxide_nm": state.oxide_nm,
        "dit_contrib": state.dit_contribution(),
    }

    reduction = {k: (1 - after[k] / max(before[k], 1e-20)) * 100
                 for k in ["particle", "metal", "carbon"]}

    return {
        "seq_name":  seq_name,
        "label":     seq["label"],
        "process":   process,
        "before":    before,
        "after":     after,
        "reduction": reduction,
        "history":   state.history,
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. 洗浄 → Dit → µ_ch パイプライン接続
# ════════════════════════════════════════════════════════════════════════════

def cleaning_to_device(dicing_process: str,
                        cleaning_seq: str = "pregate_sic") -> dict:
    """
    洗浄品質 → Dit 低減 → チャンネル移動度向上 の定量評価。
    fel_process_model の dit_from_oxidation と組み合わせる。
    """
    from market_analysis.tel_process_model import dit_from_oxidation, channel_mobility, ald_film

    # 洗浄なし
    state_dirty = SurfaceState(dicing_process)
    dit_no_clean = state_dirty.dit_contribution()

    # 洗浄あり
    result_clean = run_sequence(cleaning_seq, dicing_process)
    dit_cleaned  = result_clean["after"]["dit_contrib"]

    # 熱酸化 + ALD
    dit_ox  = dit_from_oxidation(1150, 2.0, anneal_T_C=950)
    ald     = ald_film(50, 250, "HfO2")

    mu_no_clean = channel_mobility(
        (dit_ox + dit_no_clean) * 0.2, ald["Cox_fF_um2"])
    mu_cleaned  = channel_mobility(
        (dit_ox + dit_cleaned)  * 0.2, ald["Cox_fF_um2"])

    return {
        "dicing":        dicing_process,
        "cleaning":      cleaning_seq,
        "dit_no_clean":  dit_no_clean,
        "dit_cleaned":   dit_cleaned,
        "dit_reduction": (1 - dit_cleaned / max(dit_no_clean, 1)) * 100,
        "mu_no_clean":   round(mu_no_clean, 1),
        "mu_cleaned":    round(mu_cleaned, 1),
        "mu_gain":       round(mu_cleaned - mu_no_clean, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    procs   = ["blade", "laser_ps", "stealth"]
    labels  = {"blade":"ブレード","laser_ps":"レーザーps","stealth":"ステルス"}
    colors  = {"blade":"#d62728","laser_ps":"#2ca02c","stealth":"#2166ac"}
    seqs    = ["post_cmp", "post_dicing", "pregate_si", "pregate_sic"]

    # ── Panel 1: 洗浄シーケンス別 炭素除去率 ─────────────────────────
    ax = axes[0, 0]
    x = np.arange(len(seqs))
    w = 0.28
    for k, proc in enumerate(procs):
        reductions = []
        for seq in seqs:
            r = run_sequence(seq, proc)
            reductions.append(r["reduction"]["carbon"])
        ax.bar(x + (k-1)*w, reductions, w, color=colors[proc],
               label=labels[proc], alpha=0.85, edgecolor="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SEQUENCES[s]["label"].split("(")[0][:12]
                        for s in seqs], fontsize=8, rotation=15)
    ax.set_ylabel("炭素除去率 [%]", fontsize=10)
    ax.set_title("洗浄シーケンス別 炭素除去率\n(SiC/SiO₂ Dit の前駆体)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2, axis="y")
    ax.axhline(99, color="purple", ls="--", lw=1.2, label=">99% 目標")

    # ── Panel 2: ステップ別 汚染推移 (SiC Pre-Gate) ───────────────────
    ax2 = axes[0, 1]
    r_sic = run_sequence("pregate_sic", "blade")
    steps = ["初期"] + [h["step"] for h in r_sic["history"]]
    vals_p = [r_sic["before"]["particle"]] + \
             [h["after"]["particle"] for h in r_sic["history"]]
    vals_c = [r_sic["before"]["carbon"]] + \
             [h["after"]["carbon"]   for h in r_sic["history"]]
    vals_m = [r_sic["before"]["metal"]] + \
             [h["after"]["metal"]    for h in r_sic["history"]]
    x2 = range(len(steps))
    ax2.semilogy(x2, vals_p, "o-", color="#2166ac", lw=2, label="Particle [cm⁻²]")
    ax2.semilogy(x2, vals_c, "s-", color="#d62728", lw=2, label="Carbon [cm⁻²]")
    ax2.semilogy(x2, vals_m, "^-", color="#2ca02c", lw=2, label="Metal [cm⁻²]")
    ax2.set_xticks(x2); ax2.set_xticklabels(steps, fontsize=8, rotation=20)
    ax2.set_ylabel("汚染密度 [cm⁻²]", fontsize=10)
    ax2.set_title("SiC Pre-Gate 洗浄 (Piranha→HF→SC2→O₃→HF)\n汚染推移", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2, which="both")

    # ── Panel 3: Si vs SiC 洗浄シーケンス比較 ────────────────────────
    ax3 = axes[0, 2]
    for seq, col, lbl in [("pregate_si",  "#d62728", "RCA (Si標準)"),
                           ("pregate_sic", "#2166ac", "SiC 最適化")]:
        r = run_sequence(seq, "blade")
        contams = ["particle","metal","carbon"]
        vals_b = [r["before"][c] for c in contams]
        vals_a = [r["after"][c]  for c in contams]
        x3 = np.arange(3)
        w3 = 0.35
        off = -w3/2 if "si\"" in seq else w3/2
        ax3.bar(x3 - w3/2 if "si\"" not in seq else x3 + w3/2,
                [r["reduction"][c] for c in contams],
                w3, color=col, alpha=0.85, label=lbl, edgecolor="k", lw=0.5)
    ax3.set_xticks(np.arange(3))
    ax3.set_xticklabels(["Particle", "Metal", "Carbon"], fontsize=10)
    ax3.set_ylabel("除去率 [%]", fontsize=10)
    ax3.set_title("Si RCA vs SiC 最適化洗浄\n(SiC は C 除去に特化が必要)", fontsize=10)
    ax3.legend(fontsize=9); ax3.grid(alpha=0.2, axis="y")

    # ── Panel 4: 洗浄 → Dit → µ_ch 改善 ─────────────────────────────
    ax4 = axes[1, 0]
    for k, proc in enumerate(procs):
        res_no  = cleaning_to_device(proc, "pregate_sic")
        mu_vals = [res_no["mu_no_clean"], res_no["mu_cleaned"]]
        x4 = [k - 0.15, k + 0.15]
        ax4.bar(x4[0], mu_vals[0], 0.28, color=colors[proc], alpha=0.4,
                edgecolor="k", lw=0.8, label="洗浄なし" if k==0 else "")
        ax4.bar(x4[1], mu_vals[1], 0.28, color=colors[proc], alpha=0.9,
                edgecolor="k", lw=0.8, label="洗浄あり" if k==0 else "")
        ax4.text(k, max(mu_vals)+0.5,
                 f"+{res_no['mu_gain']:.1f}", ha="center", fontsize=9,
                 color=colors[proc], fontweight="bold")
    ax4.set_xticks(range(3))
    ax4.set_xticklabels([labels[p] for p in procs])
    ax4.set_ylabel("µ_ch [cm²/Vs]", fontsize=10)
    ax4.set_title("洗浄 → Dit 低減 → µ_ch 向上\n(薄色=洗浄なし, 濃色=洗浄あり)", fontsize=10)
    ax4.legend(fontsize=8); ax4.grid(alpha=0.2, axis="y")
    ax4.set_ylim(850, 910)

    # ── Panel 5: 洗浄時間 vs 炭素残留 (温度依存) ─────────────────────
    ax5 = axes[1, 1]
    t_arr = np.linspace(1, 30, 80)
    for T_C, col, lbl in [(80,"#d62728","80°C"),
                           (100,"#ff7f0e","100°C"),
                           (120,"#2ca02c","120°C (Piranha)")]:
        c0 = 5e13
        E  = CHEMICALS["Piranha"]["E_act_eV"]
        k_rate = CHEMICALS["Piranha"]["carbon_removal"] * \
                 math.exp(-E/(kB*(T_C+273))) / math.exp(-E/(kB*(120+273)))
        carbons = [c0 * (1 - k_rate)**t for t in t_arr]
        ax5.semilogy(t_arr, carbons, color=col, lw=2, label=lbl)
    ax5.axhline(1e11, color="purple", ls="--", lw=1.5,
                label="目標 <1e11 cm⁻²")
    ax5.set_xlabel("Piranha 処理時間 [min]", fontsize=10)
    ax5.set_ylabel("Carbon 残留量 [cm⁻²]", fontsize=10)
    ax5.set_title("Piranha 炭素除去\n温度 × 時間 依存性", fontsize=10)
    ax5.legend(fontsize=8); ax5.grid(alpha=0.2, which="both")

    # ── Panel 6: 全パイプライン Dit サマリー ─────────────────────────
    ax6 = axes[1, 2]
    ax6.axis("off")
    rows = [["工程", "Dit 寄与 [cm⁻²eV⁻¹]", "洗浄後 Dit", "改善率"]]
    for proc in procs:
        r = cleaning_to_device(proc, "pregate_sic")
        rows.append([
            labels[proc],
            f"{r['dit_no_clean']:.1e}",
            f"{r['dit_cleaned']:.1e}",
            f"{r['dit_reduction']:.0f}%",
        ])
    tbl = ax6.table(cellText=rows[1:], colLabels=rows[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1.2, 2.3)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#333"); cell.set_text_props(color="white")
        elif r <= 3:
            cell.set_facecolor(
                [None, "#ffcccc", "#fff4cc", "#cce4ff"][r])
    ax6.set_title("洗浄 → Dit 低減 サマリー\n(SiC Pre-Gate 最適シーケンス)", fontsize=10)

    fig.suptitle("TEL 洗浄モデル (CELLESTA 相当) — SiC 前工程\n"
                 "Post-CMP / Post-Dicing / Pre-Gate × Piranha+HF+SC2+O₃",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "tel_cleaning.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # サマリー出力
    print("\n  洗浄シーケンス効果サマリー:")
    print(f"  {'シーケンス':<20} {'炭素除去':>9}  {'金属除去':>9}  {'粒子除去':>9}  Dit寄与")
    print("  " + "─"*60)
    for seq in seqs:
        r = run_sequence(seq, "blade")
        d = SurfaceState("blade")
        for step in SEQUENCES[seq]["steps"]:
            d.apply_step(step)
        print(f"  {SEQUENCES[seq]['label'][:20]:<20} "
              f"{r['reduction']['carbon']:>8.1f}%  "
              f"{r['reduction']['metal']:>8.1f}%  "
              f"{r['reduction']['particle']:>8.1f}%  "
              f"{d.dit_contribution():.1e}")
    print(f"\n[✓] TEL cleaning -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence",         default=None)
    ap.add_argument("--compare-materials",action="store_true")
    args = ap.parse_args()
    plot_all()


if __name__ == "__main__":
    main()
