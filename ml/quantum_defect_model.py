"""
Quantum Defect Model — SiC / GaN 点欠陥の電子構造解析
=======================================================
DFT 文献値をベースとした解析的欠陥モデル。
真の第一原理計算（VASP/QE）の代わりに DFT-calibrated パラメータを用いて
欠陥形成エネルギー・荷電状態遷移レベル・キャリア寿命への影響を計算する。

物理背景:
    半導体中の点欠陥はバンドギャップ内にトラップ準位を作る。
    ダイシング・プラズマ・研削などの加工プロセスが結晶損傷を誘起し、
    欠陥密度を増加させることでデバイス特性（キャリア寿命・移動度）を劣化させる。

4H-SiC の主要欠陥（2nm ノードパワーデバイスで重要）:
    Z1/2 center  : 炭素空孔 V_C (k サイト)  E_c - 0.63 eV — 最大寿命キラー
    EH6/7 center : ダイ空孔対 V_Si-V_C      E_c - 1.55 eV
    V_Si center  : シリコン空孔（量子スピン量子ビット候補）
    積層欠陥     : 順バイアス下で拡張するプラナー欠陥

GaN の主要欠陥:
    V_N          : 窒素空孔  浅いドナー
    V_Ga         : ガリウム空孔  深いアクセプター  E_c - 1.1 eV
    Threading dislocation: 密度 10^8-10^10 /cm² が移動度を制限

実行:
    python ml/quantum_defect_model.py --plot
    python ml/quantum_defect_model.py --process-damage --haz 2.0
    python ml/quantum_defect_model.py --lifetime-map

参考文献:
    Kimoto & Cooper (2014) Fundamentals of Silicon Carbide Technology
    Iwamoto & Kimoto (2020) Appl Phys Express — Z1/2 DFT
    Dreyer et al. (2018) Rev Mod Phys — GaN defect review
    Davidsson et al. (2021) npj Comput Mater — V_Si spin qubits
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── 物理定数 ────────────────────────────────────────────────────────────────
kB   = 8.617e-5   # eV/K
T    = 300.0       # K（室温）
kT   = kB * T

# ── 材料バンドギャップ ────────────────────────────────────────────────────
BANDGAP = {
    "4H-SiC": 3.26,   # eV
    "GaN":    3.44,   # eV
    "Si":     1.12,   # eV（比較用）
}

# ── DFT 文献値: 欠陥形成エネルギーパラメータ ───────────────────────────────
# 形式: {名前: {荷電状態q: (E_f0, slope), ...}, "level": 遷移エネルギー, ...}
#   E_f(q, E_F) = E_f0 + q * E_F  （E_F はVBMからの Fermiレベル [eV]）
#   遷移レベル ε(q/q') : E_f(q) = E_f(q') の E_F 値
#
# 出所: Iwamoto & Kimoto (2020), Bockstedte et al. (2010), Dreyer et al. (2018)

DEFECTS_SIC = {
    "V_C (Z1/2)": {
        "color": "#d62728",
        "charges": {
            0:  (3.20,  0),    # neutral
            -1: (3.83, -1),    # negative (dominant at n-type)
            -2: (4.72, -2),    # doubly negative
        },
        "transitions": {
            "ε(0/-)":  3.26 - 0.63,   # E_c - 0.63 eV  = 2.63 eV from VBM
            "ε(-/-2)": 3.26 - 1.13,   # E_c - 1.13 eV
        },
        "description": "炭素空孔 (k-site) — 最大ライフタイムキラー",
        "dft_ref": "Iwamoto & Kimoto (2020) Appl Phys Express",
        "process_sensitivity": 1.0,   # 加工誘起生成の相対感度
    },
    "V_Si (spin qubit)": {
        "color": "#2166ac",
        "charges": {
            0:  (4.50,  0),
            -1: (5.00, -1),
            -2: (5.80, -2),
        },
        "transitions": {
            "ε(0/-)":  3.26 - 0.44,
            "ε(-/-2)": 3.26 - 0.93,
        },
        "description": "シリコン空孔 — 量子スピン量子ビット候補 (S=3/2)",
        "dft_ref": "Davidsson et al. (2021) npj Comput Mater",
        "process_sensitivity": 0.8,
    },
    "V_Si-V_C (EH6/7)": {
        "color": "#2ca02c",
        "charges": {
            0:  (6.10,  0),
            -1: (6.70, -1),
            -2: (7.50, -2),
        },
        "transitions": {
            "ε(0/-)":  3.26 - 1.55,
            "ε(-/-2)": 3.26 - 1.70,
        },
        "description": "ダイ空孔対 — 深いトラップ (EH6/7)",
        "dft_ref": "Bockstedte et al. (2010) Phys Rev B",
        "process_sensitivity": 1.2,   # ダイシング衝撃で生成されやすい
    },
    "C_Si antisite": {
        "color": "#ff7f0e",
        "charges": {
            0:  (1.20,  0),
            1:  (0.60,  1),
            2:  (0.10,  2),
        },
        "transitions": {
            "ε(+2/+1)": 0.50,
            "ε(+1/0)":  0.60,
        },
        "description": "炭素置換サイト — 浅いドナー",
        "dft_ref": "Bockstedte et al. (2010)",
        "process_sensitivity": 0.5,
    },
}

DEFECTS_GAN = {
    "V_N": {
        "color": "#d62728",
        "charges": {
            3:  (0.50,  3),
            1:  (1.50,  1),
            0:  (2.20,  0),
        },
        "transitions": {
            "ε(+3/+1)": 0.50,
            "ε(+1/0)":  0.70,
        },
        "description": "窒素空孔 — 浅いドナー",
        "dft_ref": "Dreyer et al. (2018) Rev Mod Phys",
        "process_sensitivity": 1.0,
    },
    "V_Ga": {
        "color": "#2166ac",
        "charges": {
            0:  (5.50,  0),
            -1: (6.00, -1),
            -2: (6.70, -2),
            -3: (7.60, -3),
        },
        "transitions": {
            "ε(0/-)":   3.44 - 1.10,
            "ε(-/-2)":  3.44 - 1.50,
            "ε(-2/-3)": 3.44 - 2.10,
        },
        "description": "ガリウム空孔 — 深いアクセプター (E_c - 1.1 eV)",
        "dft_ref": "Dreyer et al. (2018)",
        "process_sensitivity": 1.3,
    },
    "N_Ga antisite": {
        "color": "#2ca02c",
        "charges": {
            0:  (3.80,  0),
            -1: (4.50, -1),
        },
        "transitions": {
            "ε(0/-)": 3.44 - 1.80,
        },
        "description": "N-on-Ga 反サイト欠陥",
        "dft_ref": "Dreyer et al. (2018)",
        "process_sensitivity": 0.6,
    },
}


# ── 欠陥形成エネルギー計算 ─────────────────────────────────────────────────

def formation_energy(defect: dict, q: int, E_F: float,
                     mu_offset: float = 0.0) -> float:
    """
    E_f(q, E_F) = E_f0 + q * E_F + mu_offset
    mu_offset: 化学ポテンシャル補正（Si-rich / C-rich 条件）
    """
    E_f0, slope = defect["charges"][q]
    return E_f0 + slope * E_F + mu_offset


def stable_charge(defect: dict, E_F: float,
                  mu_offset: float = 0.0) -> int:
    """最安定荷電状態（最小形成エネルギーを持つ q）を返す。"""
    return min(defect["charges"].keys(),
               key=lambda q: formation_energy(defect, q, E_F, mu_offset))


# ── キャリア寿命モデル ─────────────────────────────────────────────────────

def shockley_read_hall_lifetime(N_trap: float,
                                sigma: float = 1e-15,  # cm²
                                v_th: float  = 2e7,    # cm/s (4H-SiC 300K)
                                ) -> float:
    """
    SRH 再結合寿命 [µs]:
        τ = 1 / (σ × v_th × N_trap)

    N_trap: トラップ密度 [cm⁻³]
    """
    if N_trap <= 0:
        return float("inf")
    return 1.0 / (sigma * v_th * N_trap) * 1e6   # → µs


def process_induced_trap_density(HAZ_depth_um: float,
                                 Ra_nm: float,
                                 process: str = "blade") -> dict:
    """
    加工プロセス → 欠陥密度の経験的モデル。

    ダイシング誘起欠陥 (Kimoto & Cooper 2014, empirical):
        ブレード: 機械的衝撃 → Z1/2 + V_Si-V_C が優勢
        レーザー: 熱誘起   → V_C (ns) / V_Si (ps/fs) が優勢
        プラズマ: イオン照射 → V_N (GaN) / V_C (SiC) が優勢

    Returns: {欠陥名: 密度 [cm⁻³]}
    """
    # 基準密度（未加工 4H-SiC エピ層）
    base = {
        "V_C (Z1/2)":    1e11,   # cm⁻³ (典型的エピ品質)
        "V_Si (spin qubit)": 5e10,
        "V_Si-V_C (EH6/7)": 2e11,
    }

    # 加工係数: HAZ 深さ・粗さ依存
    haz_factor  = 1.0 + (HAZ_depth_um / 1.0) ** 1.5   # HAZ 1µm → 2倍
    Ra_factor   = 1.0 + (Ra_nm / 50.0) ** 0.8          # Ra 50nm → 2倍

    process_factors = {
        "blade":  {"V_C (Z1/2)": 3.0, "V_Si (spin qubit)": 2.0,
                   "V_Si-V_C (EH6/7)": 4.0},   # 衝撃でダイ空孔対が多い
        "laser_ns": {"V_C (Z1/2)": 2.5, "V_Si (spin qubit)": 1.5,
                     "V_Si-V_C (EH6/7)": 2.0},
        "laser_ps": {"V_C (Z1/2)": 1.5, "V_Si (spin qubit)": 2.5,
                     "V_Si-V_C (EH6/7)": 1.2},
        "laser_fs": {"V_C (Z1/2)": 1.1, "V_Si (spin qubit)": 3.0,
                     "V_Si-V_C (EH6/7)": 1.05},   # アサーマル → V_Si 優勢
        "plasma":   {"V_C (Z1/2)": 2.0, "V_Si (spin qubit)": 1.2,
                     "V_Si-V_C (EH6/7)": 1.8},
        "stealth":  {"V_C (Z1/2)": 1.05, "V_Si (spin qubit)": 4.0,
                     "V_Si-V_C (EH6/7)": 1.02},   # 内部改質層 → V_Si 局所集中
    }
    pf = process_factors.get(process, process_factors["blade"])

    result = {}
    for name, n0 in base.items():
        factor = pf.get(name, 1.0) * haz_factor * Ra_factor
        result[name] = n0 * factor
    return result


# ── プロット ─────────────────────────────────────────────────────────────────

def plot_formation_energies(out_dir: str = OUT_DIR):
    """欠陥形成エネルギー図（E_F vs E_f）を 4H-SiC と GaN で並べて描く。"""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (material, defects, cond) in zip(axes, [
        ("4H-SiC", DEFECTS_SIC, "C-rich"),
        ("GaN",    DEFECTS_GAN, "N-rich"),
    ]):
        Eg  = BANDGAP[material]
        EFs = np.linspace(0, Eg, 300)

        ax.axvspan(0, 0, alpha=0)   # placeholder

        for dname, d in defects.items():
            # 各 E_F で最安定荷電状態の形成エネルギー（凸包）
            ef_vals = []
            for EF in EFs:
                ef_min = min(formation_energy(d, q, EF)
                             for q in d["charges"])
                ef_vals.append(ef_min)
            ef_vals = np.array(ef_vals)
            ax.plot(EFs, ef_vals, color=d["color"], lw=2.5, label=dname)

            # 遷移レベルの縦線
            for label, ev in d["transitions"].items():
                if 0 <= ev <= Eg:
                    ax.axvline(ev, color=d["color"], ls=":", lw=0.8, alpha=0.6)
                    ax.text(ev, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 8,
                            label, fontsize=6, color=d["color"],
                            rotation=90, va="top", ha="right")

        # バンド端の表示
        ax.axvline(0,  color="gray", ls="-", lw=1.5, alpha=0.4)
        ax.axvline(Eg, color="gray", ls="-", lw=1.5, alpha=0.4)
        ax.text(0.02,  0.02, "VBM", transform=ax.transAxes, fontsize=9, color="gray")
        ax.text(0.95,  0.02, "CBM", transform=ax.transAxes, fontsize=9,
                color="gray", ha="right")

        # E_f = 0 ライン（この下は自発的に形成）
        ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.4)

        ax.set_xlabel("Fermi Level $E_F$ (eV from VBM)", fontsize=11)
        ax.set_ylabel("Formation Energy $E_f$ (eV)", fontsize=11)
        ax.set_title(f"{material} 点欠陥 形成エネルギー図\n({cond} 条件, 300K)",
                     fontsize=10)
        ax.set_xlim(0, Eg)
        ax.set_ylim(0, 8)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)

    fig.suptitle("SiC / GaN 点欠陥 量子的電子構造 — DFT 文献値ベース\n"
                 "加工プロセス誘起欠陥とデバイス特性への影響",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_defect_formation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] 形成エネルギー図 → {out}")


def plot_process_damage(HAZ_depth_um: float = 2.0,
                        Ra_nm: float = 50.0,
                        out_dir: str = OUT_DIR):
    """プロセス別 欠陥密度 → キャリア寿命 比較。"""
    os.makedirs(out_dir, exist_ok=True)
    processes = ["blade", "laser_ns", "laser_ps", "laser_fs", "plasma", "stealth"]
    labels    = ["ブレード", "レーザーns", "レーザーps", "レーザーfs",
                 "プラズマ", "ステルス"]
    colors    = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: 欠陥密度（棒グラフ）
    ax = axes[0]
    defect_names = list(process_induced_trap_density(HAZ_depth_um, Ra_nm, "blade").keys())
    x = np.arange(len(defect_names))
    w = 0.12
    for i, (proc, lbl, col) in enumerate(zip(processes, labels, colors)):
        dens = process_induced_trap_density(HAZ_depth_um, Ra_nm, proc)
        vals = [dens[n] for n in defect_names]
        ax.bar(x + i*w - w*2.5, vals, width=w, label=lbl, color=col, alpha=0.85)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(" ")[0] for n in defect_names], fontsize=10)
    ax.set_ylabel("欠陥密度 [cm⁻³]", fontsize=11)
    ax.set_title(f"プロセス別 加工誘起欠陥密度\n(HAZ={HAZ_depth_um}µm, Ra={Ra_nm}nm)",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2, which="both")
    ax.axhline(1e12, color="red", ls="--", lw=1.2, label="デバイス限界")
    ax.text(2.5, 1.2e12, "デバイス特性劣化限界", color="red", fontsize=8)

    # Panel 2: キャリア寿命（各プロセス × Z1/2 支配）
    ax2 = axes[1]
    haz_range = np.linspace(0.1, 10.0, 50)
    for proc, lbl, col in zip(processes, labels, colors):
        lifetimes = []
        for haz in haz_range:
            dens = process_induced_trap_density(haz, Ra_nm, proc)
            N_z12 = dens["V_C (Z1/2)"]
            tau   = shockley_read_hall_lifetime(N_z12)
            lifetimes.append(tau)
        ax2.semilogy(haz_range, lifetimes, color=col, lw=2, label=lbl)

    ax2.axhline(1.0,   color="purple", ls="--", lw=1.5, label="1µs (SBD最低限)")
    ax2.axhline(10.0,  color="green",  ls=":",  lw=1.2, label="10µs (MOSFET目標)")
    ax2.axvline(2.0,   color="gray",   ls=":",  lw=1.0)
    ax2.text(2.1, 1e4, "HAZ=2µm\n(ps laser)", fontsize=8, color="gray")
    ax2.set_xlabel("HAZ 深さ [µm]", fontsize=11)
    ax2.set_ylabel("SRH キャリア寿命 τ [µs]", fontsize=11)
    ax2.set_title("HAZ 深さ vs キャリア寿命\n(Z1/2 支配, SRH モデル)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2, which="both")

    fig.suptitle("加工プロセス × 量子欠陥 × デバイス特性ブリッジモデル\n"
                 "4H-SiC パワーデバイス（Disco DFL/DAD プロセス評価）",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_process_damage.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] プロセス損傷図 → {out}")

    # テキストサマリー
    print(f"\n  プロセス別 キャリア寿命 (HAZ={HAZ_depth_um}µm, Ra={Ra_nm}nm)")
    print(f"  {'プロセス':>12}  {'V_C密度[cm⁻³]':>15}  {'τ [µs]':>10}  評価")
    for proc, lbl in zip(processes, labels):
        dens  = process_induced_trap_density(HAZ_depth_um, Ra_nm, proc)
        N_z12 = dens["V_C (Z1/2)"]
        tau   = shockley_read_hall_lifetime(N_z12)
        grade = "◎" if tau > 10 else ("○" if tau > 1 else "✗")
        print(f"  {lbl:>12}  {N_z12:>15.2e}  {tau:>10.2f}  {grade}")


def plot_lifetime_map(out_dir: str = OUT_DIR):
    """
    HAZ 深さ × 粗さ Ra のパラメータマップ → キャリア寿命ヒートマップ。
    2nm ノードデバイス設計への直接フィードバック。
    """
    os.makedirs(out_dir, exist_ok=True)
    HAZ_arr = np.linspace(0.05, 8.0, 60)
    Ra_arr  = np.linspace(5.0, 200.0, 50)
    HAZ_g, Ra_g = np.meshgrid(HAZ_arr, Ra_arr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, proc, title in zip(axes,
                               ["blade", "laser_ps", "stealth"],
                               ["ブレードダイシング", "レーザー(ps)", "ステルスダイシング"]):
        tau_map = np.zeros_like(HAZ_g)
        for i in range(Ra_g.shape[0]):
            for j in range(HAZ_g.shape[1]):
                dens = process_induced_trap_density(HAZ_g[i,j], Ra_g[i,j], proc)
                tau_map[i,j] = shockley_read_hall_lifetime(dens["V_C (Z1/2)"])

        im = ax.contourf(HAZ_g, Ra_g, np.log10(np.clip(tau_map, 1e-3, 1e5)),
                         levels=20, cmap="RdYlGn")
        plt.colorbar(im, ax=ax, label="log₁₀(τ [µs])")
        ax.contour(HAZ_g, Ra_g, tau_map, levels=[1.0, 10.0],
                   colors=["yellow", "white"], linewidths=[1.5, 2.0])
        ax.set_xlabel("HAZ 深さ [µm]", fontsize=10)
        ax.set_ylabel("表面粗さ Ra [nm]", fontsize=10)
        ax.set_title(f"{title}\n黄:1µs 白:10µs ライン", fontsize=10)

    fig.suptitle("2nm ノード SiC パワーデバイス — プロセスウィンドウマップ\n"
                 "HAZ×Ra → SRH 寿命 (Z1/2 欠陥支配, 300K)",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_lifetime_map.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] 寿命マップ → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot",           action="store_true",
                    help="欠陥形成エネルギー図")
    ap.add_argument("--process-damage", action="store_true",
                    help="プロセス別損傷 × キャリア寿命")
    ap.add_argument("--lifetime-map",   action="store_true",
                    help="HAZ×Ra → 寿命ヒートマップ")
    ap.add_argument("--haz",  type=float, default=2.0,
                    help="HAZ 深さ [µm]")
    ap.add_argument("--ra",   type=float, default=50.0,
                    help="表面粗さ Ra [nm]")
    args = ap.parse_args()

    run_all = not (args.plot or args.process_damage or args.lifetime_map)

    if args.plot or run_all:
        plot_formation_energies()

    if args.process_damage or run_all:
        plot_process_damage(args.haz, args.ra)

    if args.lifetime_map or run_all:
        plot_lifetime_map()


if __name__ == "__main__":
    main()
