"""
Si vs 4H-SiC Dicing/Grinding — Quantitative Processing Comparison.

Why SiC is fundamentally harder to process than Si:
  1. 3× harder (Vickers hardness)
  2. 4× higher fracture toughness → more energy to propagate cracks
  3. 3× higher thermal conductivity → heat dissipates faster (less thermal assist)
  4. Crystal anisotropy → dicing direction matters

Reference materials:
  Si  : Hopcroft et al. (2010) JMEMS
  SiC : Zhao et al. (2021) Nanotechnology and Precision Engineering
        Wang et al. (2020) Int J Mach Tools Manuf

Usage:
    python sic/sic_vs_si_analysis.py
    python sic/sic_vs_si_analysis.py --plot
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.materials.material_properties import Si, SiC, GaN


# ── Material comparison table ─────────────────────────────────────────────────

COMPARE_PROPS = [
    ("Young's modulus [GPa]",       "E",               1e9,  "GPa"),
    ("Tensile strength [MPa]",      "sigma_tensile",   1e6,  "MPa"),
    ("Compressive strength [GPa]",  "sigma_compress",  1e9,  "GPa"),
    ("Vickers hardness [GPa]",      "H_v",             1e9,  "GPa"),
    ("Fracture toughness [MPa√m]",  "K_Ic",            1e6,  "MPa√m"),
    ("Thermal conductivity [W/mK]", "k_thermal",       1.0,  "W/mK"),
    ("Thermal expansion [ppm/K]",   "alpha_thermal",   1e-6, "ppm/K"),
    ("D-P friction angle [°]",      "dp_friction_angle", 1.0, "°"),
]


def print_comparison():
    mats = {"Si": Si, "4H-SiC": SiC, "GaN": GaN}
    col_w = 32

    header = f"{'Property':<{col_w}}" + "".join(f"{'  ' + k:>12}" for k in mats)
    print("\n" + "=" * (col_w + 12 * len(mats)))
    print("  Si vs 4H-SiC vs GaN — Dicing/Grinding Material Properties")
    print("=" * (col_w + 12 * len(mats)))
    print(header)
    print("-" * (col_w + 12 * len(mats)))

    for label, key, scale, unit in COMPARE_PROPS:
        row = f"{label:<{col_w}}"
        vals = []
        for mat in mats.values():
            v = mat.get(key, float("nan"))
            vals.append(v / scale)
            row += f"{v / scale:>12.2f}"
        print(row)

    print("\n  SiC/Si ratios (key processing implications):")
    ratios = {
        "Hardness (H_v)":         (SiC["H_v"],         Si["H_v"]),
        "Fracture toughness":     (SiC["K_Ic"],        Si["K_Ic"]),
        "Thermal conductivity":   (SiC["k_thermal"],   Si["k_thermal"]),
        "Young's modulus":        (SiC["E"],            Si["E"]),
        "Tensile strength":       (SiC["sigma_tensile"], Si["sigma_tensile"]),
    }
    for name, (sic_v, si_v) in ratios.items():
        ratio = sic_v / si_v
        direction = "harder" if ratio > 1 else "easier"
        print(f"    {name:<28s}: SiC/Si = {ratio:.2f}×  ({direction} to process)")


# ── Cutting force model ───────────────────────────────────────────────────────
# Grinding force per unit width (Ft/b) empirical model:
#   Ft/b = C_t * (vf / vs)^0.6 * ap^0.4   [N/m]
# where vf = workpiece feed speed, vs = wheel peripheral speed, ap = depth of cut
# C_t captures material-specific resistance.

def cutting_force_comparison(
    ap_um: float = 150.0,
    vf_mm_s: float = 1.0,
    vs_m_s: float = 30.0,
) -> dict:
    """
    Estimate tangential grinding force per unit width for Si and SiC.

    Parameters
    ----------
    ap_um   : depth of cut [µm]
    vf_mm_s : workpiece feed speed [mm/s]
    vs_m_s  : wheel peripheral speed [m/s]

    Returns
    -------
    dict with Ft/b [N/m] for each material
    """
    ap   = ap_um * 1e-6    # m
    vf   = vf_mm_s * 1e-3  # m/s
    results = {}
    for name, mat in [("Si", Si), ("4H-SiC", SiC), ("GaN", GaN)]:
        C_t = mat.get("C_t_grinding", 1200.0)
        Ft_per_b = C_t * (vf / vs_m_s) ** 0.6 * ap ** 0.4
        results[name] = Ft_per_b
    return results


def print_force_comparison(ap_um=150.0, vf_mm_s=1.0, vs_m_s=30.0):
    forces = cutting_force_comparison(ap_um, vf_mm_s, vs_m_s)
    si_f = forces["Si"]
    print(f"\n  Tangential grinding force Ft/b  "
          f"(ap={ap_um}µm, vf={vf_mm_s}mm/s, vs={vs_m_s}m/s):")
    for name, f in forces.items():
        ratio = f / si_f
        bar = "█" * int(ratio * 10) + "░" * max(0, 30 - int(ratio * 10))
        print(f"    {name:<8s}: {f:8.1f} N/m  [{bar}]  ×{ratio:.2f} vs Si")
    print(f"\n  → SiC requires {forces['4H-SiC']/si_f:.1f}× higher cutting force than Si")
    print(f"  → Blade wear rate scales with force → SiC blade life << Si blade life")
    print(f"  → Reason DISCO developed KABRA (laser) for SiC ingot slicing")


# ── Thermal damage zone estimate ─────────────────────────────────────────────
# During blade dicing, friction heat generates a HAZ (heat affected zone).
# Simplified 1D steady-state:  HAZ depth ≈ sqrt(α * t_contact)
# t_contact ≈ blade_contact_length / feed_speed

def thermal_haz_estimate(
    feed_speed_m_s: float = 0.1,
    blade_contact_len_um: float = 500.0,
) -> dict:
    """
    Rough HAZ depth estimate for blade dicing via thermal diffusion.
    HAZ_depth ≈ sqrt(α * t_contact)
    """
    t_contact = (blade_contact_len_um * 1e-6) / feed_speed_m_s
    results = {}
    for name, mat in [("Si", Si), ("4H-SiC", SiC), ("GaN", GaN)]:
        rho = mat["density"]
        Cp  = mat["Cp"]
        k   = mat["k_thermal"]
        alpha = k / (rho * Cp)           # thermal diffusivity [m²/s]
        haz   = np.sqrt(alpha * t_contact) * 1e6   # µm
        results[name] = {"alpha_mm2s": alpha * 1e6, "haz_um": haz}
    return results


def print_thermal_haz(feed_speed_m_s=0.1):
    haz = thermal_haz_estimate(feed_speed_m_s)
    print(f"\n  Thermal HAZ estimate (blade dicing, feed={feed_speed_m_s}m/s):")
    for name, v in haz.items():
        print(f"    {name:<8s}: α={v['alpha_mm2s']:.3f} mm²/s  "
              f"HAZ≈{v['haz_um']:.1f}µm")
    sic_haz = haz["4H-SiC"]["haz_um"]
    si_haz  = haz["Si"]["haz_um"]
    print(f"\n  SiC HAZ is {sic_haz/si_haz:.2f}× SMALLER than Si")
    print(f"  → High k_thermal dissipates heat quickly → less thermal damage")
    print(f"  → But high hardness → more mechanical damage (chips, cracks)")
    print(f"  → KABRA avoids blade contact entirely → eliminates mechanical damage")


# ── Optimal process recommendation ───────────────────────────────────────────

def process_recommendation():
    print("\n  Process Recommendation by Material & Application:")
    print(f"  {'Material':<10} {'Ingot slicing':<20} {'Wafer dicing':<20} {'Notes'}")
    print("  " + "-" * 72)
    recs = [
        ("Si",     "Wire saw",   "Blade / Stealth",   "Standard process"),
        ("4H-SiC", "KABRA®",     "Blade (thin) / Plasma", "3× harder → laser preferred"),
        ("GaN",    "Laser / Blade", "Stealth / Plasma", "Brittle, thermally sensitive"),
    ]
    for mat, ingot, dice, note in recs:
        print(f"  {mat:<10} {ingot:<20} {dice:<20} {note}")

    print("\n  DISCO KABRA® advantages for SiC ingot slicing vs wire saw:")
    kabra_adv = [
        ("Kerf loss",     "~140µm (wire saw)", "~140µm → ~30µm (KABRA)", "5× material saving"),
        ("Slice speed",   "8 hours/ingot",     "2 hours/ingot",          "4× throughput"),
        ("Wafer quality", "subsurface damage",  "minimal damage",         "higher die yield"),
        ("Process",       "wet, abrasive slurry", "dry laser",           "cleaner, no slurry"),
    ]
    for prop, wire, kabra, benefit in kabra_adv:
        print(f"    {prop:<18}: {wire:<25} → {kabra:<25} ({benefit})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Si vs SiC processing analysis")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    print_comparison()
    print_force_comparison(ap_um=150, vf_mm_s=1.0, vs_m_s=30.0)
    print_force_comparison(ap_um=300, vf_mm_s=2.0, vs_m_s=30.0)
    print_thermal_haz(feed_speed_m_s=0.05)
    print_thermal_haz(feed_speed_m_s=0.2)
    process_recommendation()

    if args.plot:
        _plot_comparison()


def _plot_comparison():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Si vs 4H-SiC vs GaN — Processing Property Comparison", fontsize=13)

    mats = {"Si": Si, "4H-SiC": SiC, "GaN": GaN}
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    # Panel 1: Key mechanical properties (normalised to Si)
    ax = axes[0]
    props = ["H_v", "K_Ic", "E", "sigma_tensile"]
    labels = ["Hardness", "Fracture\nToughness", "Young's\nModulus", "Tensile\nStrength"]
    x = np.arange(len(props))
    w = 0.25
    for i, (name, mat) in enumerate(mats.items()):
        si_vals = [Si[p] for p in props]
        vals    = [mat[p] / si_vals[j] for j, p in enumerate(props)]
        ax.bar(x + i * w, vals, w, label=name, color=colors[i], alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x + w)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Ratio (normalised to Si)")
    ax.set_title("Mechanical Properties")
    ax.legend(fontsize=8)

    # Panel 2: Cutting force vs depth of cut
    ax = axes[1]
    ap_range = np.linspace(50, 400, 50)
    for i, (name, mat) in enumerate(mats.items()):
        C_t = mat.get("C_t_grinding", 1200.0)
        vf, vs = 1e-3, 30.0
        forces = [C_t * (vf / vs) ** 0.6 * (ap * 1e-6) ** 0.4 for ap in ap_range]
        ax.plot(ap_range, forces, color=colors[i], label=name, linewidth=2)
    ax.set_xlabel("Depth of cut [µm]")
    ax.set_ylabel("Tangential force Ft/b [N/m]")
    ax.set_title("Grinding Force vs Depth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Thermal properties
    ax = axes[2]
    t_props = ["k_thermal", "alpha_thermal", "Cp"]
    t_labels = ["Thermal\nConductivity\n[W/mK]", "Thermal\nExpansion\n[ppm/K]", "Heat\nCapacity\n[J/kgK]"]
    scales   = [1.0, 1e6, 1.0]
    x = np.arange(len(t_props))
    for i, (name, mat) in enumerate(mats.items()):
        vals = [mat[p] * scales[j] for j, p in enumerate(t_props)]
        ax.bar(x + i * w, vals, w, label=name, color=colors[i], alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(t_labels, fontsize=7)
    ax.set_title("Thermal Properties")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "results", "sic_vs_si_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
