"""
Curated experimental dataset for SiC blade dicing validation.
All data extracted from open-access publications.

Format:
    Each entry: {
        "source"         : citation key,
        "material"       : "4H-SiC" | "Si" | "6H-SiC",
        "blade_W_um"     : blade kerf width [µm],
        "cut_depth_um"   : penetration depth [µm],
        "feed_mm_s"      : workpiece feed speed [mm/s],
        "spindle_rpm"    : spindle rotation speed [rpm],
        "chipping_um"    : measured chipping width [µm] (topside, mean),
        "chipping_std_um": std dev of chipping width [µm] (None if not reported),
        "notes"          : additional context,
    }

Sources:
  [Micro2026]  "Processing Characteristics of Ultra-Precision Cutting of 4H-SiC
                Wafers by Dicing Blade" Micromachines 17(2):187, 2026.
                DOI: 10.3390/mi17020187  (Open Access, PMC12943408)
                Parameters: depth 80-390µm, feed 0.5-2.5mm/s, spindle 22-38krpm
                Blade: Ni-bond, 23µm kerf, grit 3000 (4.5µm), D=56.32mm
                NOTE: intermediate feed/spindle points (1.5, 2.0mm/s; 26k, 34krpm)
                      estimated by linear interpolation of digitized endpoints.

  [Mat2022]    "High-Speed Dicing of SiC Wafers with 0.048mm Diamond Blades
                via Rolling-Slitting" Materials 15(22):8083, 2022.
                DOI: 10.3390/ma15228083  (Open Access, PMC9694500)
                Parameters: depth 100-350µm, feed 1-7mm/s, spindle 10-28krpm
                Blade: resin-bond rolling-slitting, 48µm kerf, grit 10µm, D=52mm
                NOTE: depth/feed chipping digitized; spindle sweep estimated from
                      kerf width trend (no direct chipping table in paper).

  [AIP2021]    "Study on Precision Dicing Process of SiC Wafer with Diamond
                Dicing Blades" Nanotechnology and Precision Engineering 4(3):033004
                DOI: 10.1063/5.0055498 (2021)

Usage:
    from validation.experimental_data import CHIPPING_DATA, FORCE_DATA
    import pandas as pd
    df = pd.DataFrame(CHIPPING_DATA)
"""

import numpy as np

# ── Quality grades ────────────────────────────────────────────────────────────
# A: direct measurement with reported std              → noise_sigma = chipping_std_um
# B: digitized from figure, no interpolation           → noise_sigma = 1.5 µm
# C: linearly interpolated between measured endpoints  → noise_sigma = 2.5 µm
# D: rough estimate from qualitative description       → noise_sigma = 4.0 µm
#
# cut_type: "incomplete" = partial penetration (standard dicing)
#           "complete"   = fully severed wafer (different fracture regime)

QUALITY_NOISE = {"A": None, "B": 1.5, "C": 2.5, "D": 4.0}  # None → use std_um


def point_noise(entry: dict) -> float:
    """Return noise σ [µm] for a data point based on quality grade."""
    q = entry.get("quality", "B")
    if q == "A" and entry.get("chipping_std_um") is not None:
        return float(entry["chipping_std_um"])
    return QUALITY_NOISE.get(q, 2.0)


# ── Chipping width data ───────────────────────────────────────────────────────
CHIPPING_DATA = [

    # ── Micromachines 2026 (4H-SiC, blade_W=23µm) ── digitized from figures ──
    # Depth of cut sweep (front chipping, feed=1mm/s, spindle=30,000rpm)
    {"source": "Micro2026", "material": "4H-SiC", "quality": "A", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um":  80, "feed_mm_s": 1.0,
     "spindle_rpm": 30000, "chipping_um": 2.5, "chipping_std_um": 0.5,
     "notes": "digitized from Fig depth sweep; front chipping"},
    {"source": "Micro2026", "material": "4H-SiC", "quality": "A", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um": 150, "feed_mm_s": 1.0,
     "spindle_rpm": 30000, "chipping_um": 3.5, "chipping_std_um": 0.5,
     "notes": "digitized from Fig depth sweep; front chipping"},
    {"source": "Micro2026", "material": "4H-SiC", "quality": "A", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um": 220, "feed_mm_s": 1.0,
     "spindle_rpm": 30000, "chipping_um": 4.5, "chipping_std_um": 0.5,
     "notes": "digitized from Fig depth sweep; front chipping"},
    {"source": "Micro2026", "material": "4H-SiC", "quality": "A", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um": 290, "feed_mm_s": 1.0,
     "spindle_rpm": 30000, "chipping_um": 6.0, "chipping_std_um": 1.0,
     "notes": "digitized from Fig depth sweep; front chipping"},
    {"source": "Micro2026", "material": "4H-SiC", "quality": "A", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um": 360, "feed_mm_s": 1.0,
     "spindle_rpm": 30000, "chipping_um": 7.5, "chipping_std_um": 1.0,
     "notes": "digitized from Fig depth sweep; front chipping"},

    # Feed speed sweep (depth=390µm, spindle=30,000rpm) — front chipping
    {"source": "Micro2026", "material": "4H-SiC", "quality": "A", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um": 390, "feed_mm_s": 0.5,
     "spindle_rpm": 30000, "chipping_um": 8.0,  "chipping_std_um": 1.0,
     "notes": "digitized from Fig feed sweep; front chipping"},
    {"source": "Micro2026", "material": "4H-SiC", "quality": "A", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um": 390, "feed_mm_s": 1.0,
     "spindle_rpm": 30000, "chipping_um": 10.0, "chipping_std_um": 1.0,
     "notes": "digitized; front chipping"},
    {"source": "Micro2026", "material": "4H-SiC", "quality": "C", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um": 390, "feed_mm_s": 1.5,
     "spindle_rpm": 30000, "chipping_um": 11.5, "chipping_std_um": 1.5,
     "notes": "estimated by linear interpolation (0.5→2.5mm/s trend); front chipping"},
    {"source": "Micro2026", "material": "4H-SiC", "quality": "C", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um": 390, "feed_mm_s": 2.0,
     "spindle_rpm": 30000, "chipping_um": 13.0, "chipping_std_um": 1.5,
     "notes": "estimated by linear interpolation; front chipping"},
    {"source": "Micro2026", "material": "4H-SiC", "quality": "A", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um": 390, "feed_mm_s": 2.5,
     "spindle_rpm": 30000, "chipping_um": 15.0, "chipping_std_um": 2.0,
     "notes": "digitized; front chipping (backside=26µm explicitly stated)"},

    # Spindle speed sweep (depth=390µm, feed=1mm/s) — front chipping
    {"source": "Micro2026", "material": "4H-SiC", "quality": "A", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um": 390, "feed_mm_s": 1.0,
     "spindle_rpm": 22000, "chipping_um": 12.0, "chipping_std_um": 1.5,
     "notes": "digitized; front chipping"},
    {"source": "Micro2026", "material": "4H-SiC", "quality": "C", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um": 390, "feed_mm_s": 1.0,
     "spindle_rpm": 26000, "chipping_um": 11.0, "chipping_std_um": 1.5,
     "notes": "estimated by interpolation (22000→38000 rpm trend); front chipping"},
    {"source": "Micro2026", "material": "4H-SiC", "quality": "A", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um": 390, "feed_mm_s": 1.0,
     "spindle_rpm": 30000, "chipping_um": 10.0, "chipping_std_um": 1.0,
     "notes": "digitized; front chipping"},
    {"source": "Micro2026", "material": "4H-SiC", "quality": "C", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um": 390, "feed_mm_s": 1.0,
     "spindle_rpm": 34000, "chipping_um": 9.5,  "chipping_std_um": 1.0,
     "notes": "estimated by interpolation; front chipping"},
    {"source": "Micro2026", "material": "4H-SiC", "quality": "A", "cut_type": "incomplete",
     "blade_W_um": 23, "cut_depth_um": 390, "feed_mm_s": 1.0,
     "spindle_rpm": 38000, "chipping_um": 9.0,  "chipping_std_um": 1.0,
     "notes": "digitized; front chipping (higher spindle = less chipping)"},

    # Complete cut — different fracture regime, excluded from incomplete-cut models
    {"source": "Micro2026", "material": "4H-SiC", "quality": "B", "cut_type": "complete",
     "blade_W_um": 23, "cut_depth_um": 390, "feed_mm_s": 1.0,
     "spindle_rpm": 30000, "chipping_um": 7.0,  "chipping_std_um": 1.0,
     "notes": "complete cut (wafer fully severed); front chipping from Fig 9"},

    # ── Mat2022 (SiC, 0.048mm=48µm blade) ────────────────────────────────────
    # All values estimated/qualitative — quality D
    {"source": "Mat2022", "material": "SiC", "quality": "D", "cut_type": "incomplete",
     "blade_W_um": 48, "cut_depth_um": 100, "feed_mm_s": 5.0,
     "spindle_rpm": 22000, "chipping_um": 8.0,  "chipping_std_um": None,
     "notes": "estimated from 'acceptable' (<15µm) range, shallow depth"},
    {"source": "Mat2022", "material": "SiC", "quality": "D", "cut_type": "incomplete",
     "blade_W_um": 48, "cut_depth_um": 200, "feed_mm_s": 5.0,
     "spindle_rpm": 22000, "chipping_um": 10.0, "chipping_std_um": None,
     "notes": "optimal condition, chipping < 15µm threshold"},
    {"source": "Mat2022", "material": "SiC", "quality": "D", "cut_type": "incomplete",
     "blade_W_um": 48, "cut_depth_um": 300, "feed_mm_s": 5.0,
     "spindle_rpm": 22000, "chipping_um": 13.0, "chipping_std_um": None,
     "notes": "estimated, approaching threshold"},
    {"source": "Mat2022", "material": "SiC", "quality": "D", "cut_type": "incomplete",
     "blade_W_um": 48, "cut_depth_um": 350, "feed_mm_s": 5.0,
     "spindle_rpm": 22000, "chipping_um": 16.0, "chipping_std_um": None,
     "notes": "estimated, exceeding threshold"},
    {"source": "Mat2022", "material": "SiC", "quality": "D", "cut_type": "incomplete",
     "blade_W_um": 48, "cut_depth_um": 200, "feed_mm_s": 1.0,
     "spindle_rpm": 22000, "chipping_um": 7.0,  "chipping_std_um": None,
     "notes": "low feed, low chipping"},
    {"source": "Mat2022", "material": "SiC", "quality": "D", "cut_type": "incomplete",
     "blade_W_um": 48, "cut_depth_um": 200, "feed_mm_s": 3.0,
     "spindle_rpm": 22000, "chipping_um": 9.0,  "chipping_std_um": None,
     "notes": "medium feed"},
    {"source": "Mat2022", "material": "SiC", "quality": "D", "cut_type": "incomplete",
     "blade_W_um": 48, "cut_depth_um": 200, "feed_mm_s": 7.0,
     "spindle_rpm": 22000, "chipping_um": 20.0, "chipping_std_um": None,
     "notes": "high feed, overloaded blade (kerf > 60µm)"},
    {"source": "Mat2022", "material": "SiC", "quality": "D", "cut_type": "incomplete",
     "blade_W_um": 48, "cut_depth_um": 200, "feed_mm_s": 1.0,
     "spindle_rpm": 10000, "chipping_um": 13.0, "chipping_std_um": None,
     "notes": "estimated; low spindle, wider kerf (56µm) → higher chipping"},
    {"source": "Mat2022", "material": "SiC", "quality": "D", "cut_type": "incomplete",
     "blade_W_um": 48, "cut_depth_um": 200, "feed_mm_s": 1.0,
     "spindle_rpm": 16000, "chipping_um": 11.0, "chipping_std_um": None,
     "notes": "estimated; moderate spindle"},
    {"source": "Mat2022", "material": "SiC", "quality": "D", "cut_type": "incomplete",
     "blade_W_um": 48, "cut_depth_um": 200, "feed_mm_s": 1.0,
     "spindle_rpm": 28000, "chipping_um": 9.0,  "chipping_std_um": None,
     "notes": "estimated; higher spindle → lower chipping (Mat2022 trend)"},
]

# ── Key qualitative findings (for trend validation) ───────────────────────────
QUALITATIVE_TRENDS = {
    "depth_effect":   "chipping INCREASES with cut depth (dominant effect)",
    "feed_effect":    "chipping INCREASES with feed speed (second effect)",
    "spindle_effect": "chipping DECREASES with spindle speed (minor effect)",
    "blade_effect":   "chipping INCREASES with larger grit size",
    "parameter_rank": "depth > feed > spindle (most to least influential)",
    "ductile_brittle_transition_nm": 16.2,  # critical chip thickness for 4H-SiC [nm]
    "acceptable_chipping_um": 15.0,          # production threshold [µm]
    "sources": ["Micro2026", "Mat2022"],
}

# ── Blade specifications from literature ──────────────────────────────────────
BLADE_SPECS = {
    "Micro2026": {
        "type":         "Nickel bond",
        "thickness_um":  23,
        "diameter_mm":   56.32,
        "grit_grade":    3000,
        "grit_size_um":  4.5,
        "concentration": 70,
    },
    "Mat2022": {
        "type":         "Resin bond (rolling-slitting)",
        "thickness_um":  48,
        "diameter_mm":   52,
        "grit_size_um":  10,
        "diamond_wt_frac": 0.03,
    },
}

# ── Drucker-Prager parameters from literature (for model validation) ──────────
DP_PARAMS_LITERATURE = {
    "4H-SiC": {
        "friction_angle_deg": 68.8,    # β, from SiN analogue (Crystals 2021)
        "cohesion_GPa":        1.0,    # d, estimated range 0.5–2.0 GPa
        "dilation_deg":        0.0,    # ψ, non-associated flow
        "sigma_tensile_MPa": 350.0,    # tensile strength
        "sigma_compress_MPa": 3000.0,  # compressive strength
        "K_Ic_MPa_sqrt_m":    2.8,
        "notes": "Wang 2020, Huang 2021, Crystals 2021 (3C-SiC analogue)",
    },
    "Si": {
        "friction_angle_deg": 55.0,
        "cohesion_GPa":        0.5,
        "dilation_deg":        0.0,
        "sigma_tensile_MPa": 150.0,
        "sigma_compress_MPa": 1000.0,
        "K_Ic_MPa_sqrt_m":    0.83,
        "notes": "literature estimates",
    },
}


def get_chipping_dataframe():
    """Return CHIPPING_DATA as a pandas DataFrame."""
    import pandas as pd
    df = pd.DataFrame(CHIPPING_DATA)
    df["chipping_flag_estimated"] = df["notes"].str.contains("estimated")
    return df


def print_summary():
    import pandas as pd
    df = get_chipping_dataframe()
    print("=== Experimental Dataset Summary ===")
    print(f"Total data points: {len(df)}")
    print(f"Sources: {df['source'].unique().tolist()}")
    print(f"Blade widths [µm]: {sorted(df['blade_W_um'].unique().tolist())}")
    print(f"Feed speeds [mm/s]: {sorted(df['feed_mm_s'].unique().tolist())}")
    print(f"Chipping range [µm]: {df['chipping_um'].min():.1f} – {df['chipping_um'].max():.1f}")
    print(f"\nQualitative trends:")
    for k, v in QUALITATIVE_TRENDS.items():
        if isinstance(v, str):
            print(f"  {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════════════
# HBM paper: wafer thinning & chip fracture strength
# Source: Materials 17(22):5529, MDPI 2024. PMC11595813.
#   "Si Characterization on Thinning and Singulation Processes for 2.5/3D HBM"
#   Test: 3-point bending, chip 4×8mm, span=2.4mm, Si(111) 12-inch
# ═══════════════════════════════════════════════════════════════════════════════

# 3PB geometry
_3PB_SPAN_m  = 2.4e-3
_3PB_WIDTH_m = 4.0e-3


def gf_to_mpa(force_gf: float, thickness_um: float) -> float:
    """Convert gram-force 3PB reading → flexural stress [MPa]."""
    F = force_gf * 9.81e-3
    h = thickness_um * 1e-6
    return 3.0 * F * _3PB_SPAN_m / (2.0 * _3PB_WIDTH_m * h ** 2) / 1e6


# Raw 3PB force [gram-force] per (singulation method, wafer thickness µm)
_CHIP_GF = {
    ("stealth_dicing",  60): 153.0,
    ("stealth_dicing",  90): 105.0,
    ("stealth_dicing", 120): 108.0,
    ("blade_dicing",    60):  65.0,
    ("blade_dicing",    90):  45.0,
    ("blade_dicing",   120):  60.0,
    ("laser_grooving",  60):  20.0,
    ("laser_grooving",  90):  18.0,
    ("laser_grooving", 120):  15.0,
}

# MPa values (auto-converted)
CHIP_STRENGTH_MPa = {k: gf_to_mpa(v, k[1]) for k, v in _CHIP_GF.items()}

# Thinned-wafer flexural strength (before singulation) [gf]
_WAFER_GF = {
    ("fine_grinding",  60): 14.5,
    ("fine_grinding",  90): 11.2,
    ("fine_grinding", 120): 10.8,
    ("poly_grinding",  60): 15.3,
    ("poly_grinding",  90): 10.2,
    ("poly_grinding", 120): 15.5,
    ("polishing",      60): 21.2,
    ("polishing",      90): 14.5,
    ("polishing",     120): 15.2,
}
WAFER_STRENGTH_MPa = {k: gf_to_mpa(v, k[1]) for k, v in _WAFER_GF.items()}

# Surface roughness [nm]: {Ra, Rt, Rz}
SURFACE_ROUGHNESS_nm = {
    ("fine_grinding",  60): {"Ra": 2.23, "Rt": 18.61, "Rz": 17.07},
    ("fine_grinding",  90): {"Ra": 2.74, "Rt": 18.61, "Rz": 17.07},
    ("fine_grinding", 120): {"Ra": 2.23, "Rt": 18.61, "Rz": 17.07},
    ("poly_grinding",  60): {"Ra": 2.49, "Rt": 21.19, "Rz": 19.27},
    ("poly_grinding",  90): {"Ra": 3.10, "Rt": 21.19, "Rz": 19.27},
    ("poly_grinding", 120): {"Ra": 2.49, "Rt": 21.19, "Rz": 19.27},
    ("polishing",      60): {"Ra": 0.87, "Rt":  8.75, "Rz":  7.83},
    ("polishing",      90): {"Ra": 1.10, "Rt":  8.75, "Rz":  7.83},
    ("polishing",     120): {"Ra": 0.87, "Rt":  8.75, "Rz":  7.83},
}


def effective_flaw_size_um(method: str, thickness_um: int,
                            K_Ic_MPa_m05: float = 0.83) -> float:
    """
    Back-calculate effective Griffith crack depth from chip fracture stress.
        a_eff = (K_Ic / σ_f)² / π     [µm]
    Larger → more surface damage introduced by the singulation process.
    """
    sigma = CHIP_STRENGTH_MPa.get((method, thickness_um))
    if sigma is None:
        return float("nan")
    return (K_Ic_MPa_m05 / sigma) ** 2 / np.pi * 1e6


def print_fracture_summary():
    print("\n" + "=" * 72)
    print("  Chip Fracture Strength [MPa] — Si(111), 3PB, span=2.4mm, 4×8mm chip")
    print("  Source: Materials 17(22):5529, MDPI 2024 (PMC11595813)")
    print("=" * 72)
    print(f"  {'Method':<18} {'60µm':>8} {'90µm':>8} {'120µm':>8}  {'a_eff@60µm':>12}")
    print("  " + "-" * 60)
    for m in ("stealth_dicing", "blade_dicing", "laser_grooving"):
        vals = [CHIP_STRENGTH_MPa.get((m, t), float("nan")) for t in (60, 90, 120)]
        a60  = effective_flaw_size_um(m, 60)
        sym  = {"stealth_dicing": "★ best", "blade_dicing": "  mid",
                "laser_grooving": "  worst"}[m]
        print(f"  {m:<18} "
              f"{vals[0]:>7.0f}M {vals[1]:>7.0f}M {vals[2]:>7.0f}M  "
              f"{a60:>9.2f}µm  {sym}")
    print()
    print("  Wafer Strength (thinning only, before singulation):")
    for m in ("fine_grinding", "poly_grinding", "polishing"):
        vals = [WAFER_STRENGTH_MPa.get((m, t), float("nan")) for t in (60, 90, 120)]
        print(f"  {m:<18} {vals[0]:>7.0f}M {vals[1]:>7.0f}M {vals[2]:>7.0f}M")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Leeftink (arXiv:2511.23141): BO on LASER1205 D-UVP for laser dicing
# ═══════════════════════════════════════════════════════════════════════════════

LEEFTINK_OBJECTIVES = {
    "front_strength_MPa": "maximize",
    "back_strength_MPa":  "maximize",
    "dicing_width_um":    "minimize",
    "mod_width_um":       "minimize",
    "burr_height_um":     "minimize",
    "throughput_wph":     "maximize",
}

# Expert utility weights by scenario (Table A.6)
LEEFTINK_WEIGHTS = {
    "bare_Si": {
        "dicing_width": 0.075, "mod_width": 0.075, "burr_height": 1.000,
        "throughput":   0.010, "front_str": 0.500, "back_str":   0.500,
    },
    "BOLD_A_balanced": {
        "dicing_width": 0.05,  "mod_width": 0.05,  "burr_height": 0.10,
        "throughput":   0.30,  "front_str": 0.25,  "back_str":   0.25,
    },
    "BOLD_B_strength": {
        "dicing_width": 0.005, "mod_width": 0.005, "burr_height": 0.010,
        "throughput":   0.098, "front_str": 0.450, "back_str":   0.450,
    },
    "BOLD_C_speed": {
        "dicing_width": 0.005, "mod_width": 0.005, "burr_height": 0.010,
        "throughput":   0.780, "front_str": 0.100, "back_str":   0.100,
    },
}

# Results (production wafer, 53 µm thick Si)
LEEFTINK_RESULTS = {
    "expert_baseline":   {"throughput_wph": 3.01},
    "BOLD_config":       {"throughput_wph": 2.81, "front_MPa": 496, "back_MPa": 394,
                          "dicing_width_um": 28.9},
    "BOLD_A_map":        {"throughput_wph": 3.25, "front_MPa": 475, "back_MPa": 424,
                          "dicing_width_um": 28.9, "improvement_pct": 8.0},
    "BOLD_plus_expert":  {"throughput_wph": 4.03, "front_MPa": 444, "back_MPa": 388,
                          "improvement_pct": 34.0, "note": "human-in-the-loop"},
}

LEEFTINK_BUDGET = {
    "n_params": 11, "n_objectives": 6, "n_constraints": 10,
    "bare_Si_iters": 109, "production_iters": 174,
    "wafers_total": 4, "weeks_total": 3,
    "surrogate": "GP Matern-5/2", "acq": "Thompson Sampling (TuRBO+SCBO)",
    "batch_size_bare": 2, "batch_size_prod": 3,
}


if __name__ == "__main__":
    print_summary()
    print_fracture_summary()
    r = LEEFTINK_RESULTS["BOLD_plus_expert"]
    print(f"  Leeftink best (BOLD+Expert): {r['throughput_wph']:.2f} wph "
          f"(+{r['improvement_pct']:.0f}% vs expert)")
