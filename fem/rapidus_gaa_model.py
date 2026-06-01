"""
Rapidus / TSMC 2nm GAA MOSFET モデル
=====================================
Gate-All-Around (GAA) Nanosheet トランジスタの物理モデル化。
Rapidus の 2027 年 2nm 量産目標を想定した理論限界計算。

FinFET (7nm 以降) → GAA Nanosheet (2nm) の移行を定量化。

モデル内容:
    1. 量子閉じ込めエネルギー  — 薄膜ナノシートでの副帯エネルギー
    2. 短チャンネル効果 (DIBL) — ドレイン誘起障壁低下
    3. 移動度 vs ナノシート厚 — 表面粗さ散乱
    4. オン電流 vs ゲート長   — 長チャンネル→弾道輸送
    5. FinFET vs GAA 比較      — 2nm ノードの利点を定量化

実行:
    python fem/rapidus_gaa_model.py
"""

import math, os, sys
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

hbar = 1.055e-34   # J·s
m0   = 9.109e-31   # kg
q    = 1.602e-19   # C
kB_J = 1.381e-23   # J/K
eps0 = 8.854e-12

# ── 材料定数 (Si channel) ────────────────────────────────────────────────────
SI_PARAMS = {
    "m_eff": 0.19 * 9.109e-31,   # Si 横方向有効質量
    "m_t":   0.916 * 9.109e-31,  # Si 縦方向有効質量
    "Eg_eV": 1.12,
    "mu_bulk_cm2Vs": 1400,       # バルク電子移動度
    "eps_r": 11.7,
    "v_th_m_s": 1.0e5,           # 実効注入速度 [m/s] (= 1e7 cm/s)
}

# ── GAA Nanosheet 構造パラメータ ──────────────────────────────────────────────
GAA_NODES = {
    "5nm_FinFET": {
        "Lg_nm": 18, "Tsi_nm": 7,  "Wsi_nm": 30,
        "Tox_nm": 0.7, "k_gate": 22,  # HfO₂
        "n_sheets": 1, "structure": "FinFET",
        "color": "#9467bd", "year": 2019,
    },
    "3nm_GAA": {
        "Lg_nm": 12, "Tsi_nm": 5,  "Wsi_nm": 20,
        "Tox_nm": 0.6, "k_gate": 22,
        "n_sheets": 3, "structure": "GAA",
        "color": "#2166ac", "year": 2022,
    },
    "2nm_GAA": {
        "Lg_nm": 8,  "Tsi_nm": 4,  "Wsi_nm": 15,
        "Tox_nm": 0.5, "k_gate": 25,  # High-k next gen
        "n_sheets": 3, "structure": "GAA",
        "color": "#d62728", "year": 2025,
    },
    "1nm_CFET": {
        "Lg_nm": 6,  "Tsi_nm": 3,  "Wsi_nm": 10,
        "Tox_nm": 0.4, "k_gate": 30,
        "n_sheets": 2, "structure": "CFET",
        "color": "#ff7f0e", "year": 2028,
    },
}


# ════════════════════════════════════════════════════════════════════════════
# 1. 量子閉じ込めエネルギー
# ════════════════════════════════════════════════════════════════════════════

def quantum_confinement_energy(Tsi_nm: float, subband: int = 1) -> float:
    """
    無限量子井戸近似: E_n = (n²π²ħ²) / (2m* Tsi²) [eV]
    Si ナノシート方向 (100) → m* = m_t
    """
    m_eff = SI_PARAMS["m_eff"]
    Tsi_m = Tsi_nm * 1e-9
    E = (subband**2 * math.pi**2 * hbar**2) / (2 * m_eff * Tsi_m**2)
    return E / q  # eV


def threshold_voltage_shift(Tsi_nm: float) -> float:
    """量子閉じ込めによる閾値電圧シフト [V]。ΔVth = E1 - E1_bulk"""
    E1_bulk = quantum_confinement_energy(10.0)  # 参照厚さ 10nm
    E1_thin = quantum_confinement_energy(Tsi_nm)
    return E1_thin - E1_bulk


# ════════════════════════════════════════════════════════════════════════════
# 2. 短チャンネル効果 (DIBL)
# ════════════════════════════════════════════════════════════════════════════

def dibl(Lg_nm: float, Tsi_nm: float, Tox_nm: float, k_gate: float) -> float:
    """
    DIBL [mV/V] — ドレイン誘起障壁低下。
    GAA は全周囲ゲートで FinFET より DIBL が低い。
    簡易モデル: DIBL ∝ exp(-Lg / lambda)
    lambda (特性長) = sqrt(eps_Si * Tsi * Tox / (4 * k_gate))
    """
    eps_Si = SI_PARAMS["eps_r"]
    # GAA の特性長は FinFET より小さい (全周囲ゲート制御)
    lambda_nm = math.sqrt(eps_Si * Tsi_nm * Tox_nm / (4 * k_gate / 3.9)) * 0.7
    DIBL = 200 * math.exp(-Lg_nm / (2.5 * lambda_nm))  # mV/V
    return DIBL


def subthreshold_swing(DIBL_mV_V: float, T_K: float = 300) -> float:
    """SS [mV/dec] = (kT/q) * ln(10) * (1 + Cd/Cox) + DIBL 寄与"""
    SS_ideal = kB_J * T_K / q * math.log(10) * 1000  # ~60 mV/dec
    return SS_ideal + DIBL_mV_V * 0.1


# ════════════════════════════════════════════════════════════════════════════
# 3. 移動度 vs ナノシート厚
# ════════════════════════════════════════════════════════════════════════════

def nanosheet_mobility(Tsi_nm: float) -> float:
    """
    ナノシート移動度 [cm²/Vs] — Matthiessen 則。
    1/µ = 1/µ_bulk + 1/µ_SR + 1/µ_QC
    µ_SR: 表面粗さ散乱 ∝ Tsi^6 (薄いほど悪化)
    µ_QC: 量子閉じ込め散乱 ∝ Tsi^2
    """
    mu_bulk = SI_PARAMS["mu_bulk_cm2Vs"]
    # 表面粗さ散乱 (Tsi^6 依存)
    mu_SR = mu_bulk * (Tsi_nm / 10) ** 6
    # 量子閉じ込め散乱
    mu_QC = mu_bulk * (Tsi_nm / 10) ** 2 * 2
    return 1 / (1/mu_bulk + 1/mu_SR + 1/mu_QC)


# ════════════════════════════════════════════════════════════════════════════
# 4. オン電流
# ════════════════════════════════════════════════════════════════════════════

def on_current(node_name: str, Vdd: float = 0.7) -> dict:
    """
    GAA オン電流 [µA/µm]。
    長チャンネル近似 → 弾道輸送補正。
    I_on = n_sheets * Cox * µ * (Vgs-Vth)² / (2*Lg)  (long-channel)
    """
    n = GAA_NODES[node_name]
    Lg_m   = n["Lg_nm"] * 1e-9
    Tox_m  = n["Tox_nm"] * 1e-9
    W_m    = n["Wsi_nm"] * 1e-9 * n["n_sheets"]

    # Tox_nm は EOT (等価酸化膜厚) として扱う → k=3.9 (SiO₂) で Cox 計算
    Cox = 3.9 * eps0 / Tox_m   # F/m²
    mu  = nanosheet_mobility(n["Tsi_nm"]) * 1e-4   # cm² → m²

    Vth = 0.2 + threshold_voltage_shift(n["Tsi_nm"])
    Vov = max(Vdd - Vth, 0.05)

    # 長チャンネル
    I_lc = Cox * mu * W_m / Lg_m * Vov**2 / 2

    # 弾道輸送補正 (Lg < 10nm で顕著)
    v_inj = SI_PARAMS["v_th_m_s"] * 0.4  # 実効注入速度
    I_bal = Cox * v_inj * W_m * Vov
    # 実電流は両者の調和平均
    I_on = 1 / (1/I_lc + 1/I_bal) * 1e6 / (W_m * 1e6)  # µA/µm

    DIBL_val = dibl(n["Lg_nm"], n["Tsi_nm"], n["Tox_nm"], n["k_gate"])
    SS_val   = subthreshold_swing(DIBL_val)

    return {
        "node": node_name,
        "I_on_uA_um":  round(I_on, 0),
        "DIBL_mV_V":   round(DIBL_val, 1),
        "SS_mV_dec":   round(SS_val, 1),
        "Vth_shift_V": round(threshold_voltage_shift(n["Tsi_nm"]), 3),
        "mu_cm2Vs":    round(nanosheet_mobility(n["Tsi_nm"]), 0),
        "E1_eV":       round(quantum_confinement_energy(n["Tsi_nm"]), 3),
    }


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel 1: 量子閉じ込めエネルギー vs Tsi
    ax = axes[0, 0]
    Tsi_arr = np.linspace(2, 15, 100)
    for n_sub, col, lbl in [(1,"#2166ac","n=1 (ground)"),(2,"#d62728","n=2"),(3,"#ff7f0e","n=3")]:
        E = [quantum_confinement_energy(T, n_sub) for T in Tsi_arr]
        ax.plot(Tsi_arr, E, lw=2.5, color=col, label=lbl)
    for node_name, nd in GAA_NODES.items():
        ax.axvline(nd["Tsi_nm"], color=nd["color"], ls=":", lw=1.5, alpha=0.7,
                   label=f"{node_name}")
    ax.set_xlabel("Nanosheet thickness Tsi [nm]", fontsize=10)
    ax.set_ylabel("Confinement energy E_n [eV]", fontsize=10)
    ax.set_title("Quantum Confinement Energy\nvs Nanosheet Thickness", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # Panel 2: DIBL vs ゲート長
    ax2 = axes[0, 1]
    Lg_arr = np.linspace(4, 30, 100)
    for Tsi, col, lbl in [(4,"#d62728","Tsi=4nm (2nm GAA)"),(5,"#ff7f0e","Tsi=5nm (3nm GAA)"),(7,"#2166ac","Tsi=7nm (5nm Fin)")]:
        dibl_arr = [dibl(Lg, Tsi, 0.6, 22) for Lg in Lg_arr]
        ax2.plot(Lg_arr, dibl_arr, lw=2.5, color=col, label=lbl)
    ax2.axhline(30, color="red",   ls="--", lw=1.5, label="DIBL limit 30mV/V")
    ax2.axhline(60, color="orange",ls=":",  lw=1.5, label="Warning 60mV/V")
    ax2.set_xlabel("Gate length Lg [nm]", fontsize=10)
    ax2.set_ylabel("DIBL [mV/V]", fontsize=10)
    ax2.set_title("DIBL vs Gate Length\n(GAA は短チャンネル効果を抑制)", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2)

    # Panel 3: 移動度 vs ナノシート厚
    ax3 = axes[0, 2]
    Tsi_arr2 = np.linspace(2, 12, 100)
    mu_arr = [nanosheet_mobility(T) for T in Tsi_arr2]
    ax3.plot(Tsi_arr2, mu_arr, "b-", lw=2.5)
    for node_name, nd in GAA_NODES.items():
        mu = nanosheet_mobility(nd["Tsi_nm"])
        ax3.scatter(nd["Tsi_nm"], mu, s=150, color=nd["color"], zorder=5,
                    label=f"{node_name}: {mu:.0f} cm²/Vs")
    ax3.set_xlabel("Nanosheet thickness [nm]", fontsize=10)
    ax3.set_ylabel("Electron mobility [cm²/Vs]", fontsize=10)
    ax3.set_title("Nanosheet Mobility\n(Surface Roughness Scattering)", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(alpha=0.2)

    # Panel 4: オン電流 vs ゲート長（ノード比較）
    ax4 = axes[1, 0]
    results = {n: on_current(n) for n in GAA_NODES}
    node_names = list(GAA_NODES.keys())
    I_ons = [results[n]["I_on_uA_um"] for n in node_names]
    colors4 = [GAA_NODES[n]["color"] for n in node_names]
    bars = ax4.bar(node_names, I_ons, color=colors4, alpha=0.85, edgecolor="k", lw=0.8)
    for bar, val in zip(bars, I_ons):
        ax4.text(bar.get_x()+bar.get_width()/2, val+20, f"{val:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax4.set_ylabel("I_on [µA/µm]", fontsize=10)
    ax4.set_title("On-Current vs Node\nGAA のスケーリング", fontsize=10)
    ax4.set_xticklabels([n.replace("_","\n") for n in node_names], fontsize=8)
    ax4.grid(alpha=0.2, axis="y")

    # Panel 5: パフォーマンスサマリーレーダー風
    ax5 = axes[1, 1]
    metrics_arr = np.arange(len(node_names))
    w = 0.2
    metrics = {
        "I_on [µA/µm]":   [results[n]["I_on_uA_um"]/10 for n in node_names],
        "DIBL (low)":     [100 - results[n]["DIBL_mV_V"] for n in node_names],
        "SS (low)":       [120 - results[n]["SS_mV_dec"] for n in node_names],
        "µ [cm²/Vs]/10":  [results[n]["mu_cm2Vs"]/10 for n in node_names],
    }
    colors5 = ["#2166ac","#d62728","#ff7f0e","#2ca02c"]
    for k, (metric, vals) in enumerate(metrics.items()):
        offset = (k - len(metrics)/2) * w
        ax5.bar(metrics_arr + offset, vals, w, alpha=0.85,
                color=colors5[k], label=metric, edgecolor="k", lw=0.5)
    ax5.set_xticks(metrics_arr)
    ax5.set_xticklabels([n.replace("_","\n") for n in node_names], fontsize=8)
    ax5.set_ylabel("Score (normalized)", fontsize=10)
    ax5.set_title("Multi-metric Performance\n(高いほど良い)", fontsize=10)
    ax5.legend(fontsize=7); ax5.grid(alpha=0.2, axis="y")

    # Panel 6: Rapidus ロードマップ + サマリー表
    ax6 = axes[1, 2]
    ax6.axis("off")
    rows = [["ノード", "Lg[nm]", "Tsi[nm]", "I_on\n[µA/µm]", "DIBL\n[mV/V]", "SS\n[mV/dec]"]]
    for node_name in node_names:
        nd = GAA_NODES[node_name]
        r  = results[node_name]
        rows.append([
            node_name.replace("_","\n"),
            str(nd["Lg_nm"]),
            str(nd["Tsi_nm"]),
            f"{r['I_on_uA_um']:.0f}",
            f"{r['DIBL_mV_V']:.1f}",
            f"{r['SS_mV_dec']:.1f}",
        ])
    tbl = ax6.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.2, 2.1)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_facecolor("#333"); cell.set_text_props(color="w")
        elif r == 3: cell.set_facecolor("#ffe0e0")  # 2nm GAA 強調
        elif r % 2 == 0: cell.set_facecolor("#f5f5f5")
    ax6.set_title("GAA Node Roadmap\n(Rapidus 2nm → 2027年量産目標)", fontsize=10)

    fig.suptitle("Rapidus / TSMC 2nm GAA Nanosheet MOSFET モデル\n"
                 "量子閉じ込め / DIBL / 移動度 / オン電流スケーリング",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "rapidus_gaa.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print("\n  GAA ノード別パフォーマンス:")
    print(f"  {'ノード':<16} {'I_on':>8} {'DIBL':>8} {'SS':>8} {'µ':>8}")
    print("  " + "─" * 52)
    for node_name in node_names:
        r = results[node_name]
        print(f"  {node_name:<16} {r['I_on_uA_um']:>6.0f}µA/µm  "
              f"{r['DIBL_mV_V']:>5.1f}mV/V  {r['SS_mV_dec']:>5.1f}mV/dec  "
              f"{r['mu_cm2Vs']:>5.0f}cm²/Vs")
    print(f"\n[OK] Rapidus GAA -> {out}")


if __name__ == "__main__":
    plot_all()
