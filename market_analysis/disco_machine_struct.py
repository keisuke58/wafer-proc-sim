"""
DISCO dicing saw — mechanical structure model
Covers the core メカ scope: spindle housing stiffness, frame dynamics,
bearing selection, thermal expansion alignment, and vibration isolation.

Subsystems modelled
-------------------
1. Spindle housing — static stiffness (Castigliano / Euler-Bernoulli beam)
2. Angular contact bearing pair — preload, stiffness, speed limit (dm·n)
3. Frame modal analysis — lumped 2-DOF model, natural frequencies
4. Vibration isolation — air mount transmissibility
5. Thermal expansion — spindle axis drift from heat sources
6. Tolerance stack-up — kerf position error budget (RSS)

References
----------
Harris & Kotzalas (2007) Rolling Element Bearings
Ewins (2000) Modal Testing
DISCO DFD6361: spindle overhang 18 mm, bearing span 40 mm
NSK catalog: 7008 CTYNSULP4 angular contact pair
ISO 1940-1:2003 (balance quality grades)
"""

import numpy as np

# ── 1. Spindle Housing Stiffness ─────────────────────────────────────────────

def spindle_housing_stiffness(
    E_GPa: float = 200.0,       # steel [GPa]
    D_outer_mm: float = 40.0,   # shaft outer diameter [mm]
    D_inner_mm: float = 0.0,    # hollow shaft inner diameter (0 = solid)
    L_overhang_mm: float = 18.0,  # blade nose to front bearing [mm]
    L_span_mm: float = 40.0,    # front to rear bearing [mm]
    F_radial_N: float = 20.0,   # radial cutting force at blade tip
) -> dict:
    """
    Spindle shaft: simply supported beam (bearing span) + cantilever (overhang).
    Deflection at blade tip using superposition.

    δ_tip = F × a² × (3L + a) / (3EI)     (cantilever a beyond support at L)
    Static stiffness k = F / δ_tip  [N/µm]
    """
    E = E_GPa * 1e9          # Pa
    a = L_overhang_mm * 1e-3  # m
    L = L_span_mm * 1e-3      # m
    D = D_outer_mm * 1e-3
    d = D_inner_mm * 1e-3
    I = np.pi * (D**4 - d**4) / 64  # second moment of area [m^4]

    delta_m = F_radial_N * a**2 * (3 * L + a) / (3 * E * I)
    k_N_um  = F_radial_N / delta_m * 1e-6  # N/µm

    return {
        "I_m4": I,
        "delta_um": round(delta_m * 1e6, 3),
        "k_N_um": round(k_N_um, 1),
        "k_adequate": k_N_um > 10,  # >10 N/µm typical spec
        "F_radial_N": F_radial_N,
    }


# ── 2. Angular Contact Bearing Pair ──────────────────────────────────────────

def bearing_pair(
    d_mm: float = 20.0,         # bore diameter [mm] (7004 series; 60k rpm spindle uses small bore)
    D_mm: float = 42.0,         # outer diameter [mm]
    n_balls: int = 12,
    D_ball_mm: float = 5.0,
    contact_angle_deg: float = 15.0,
    preload_N: float = 80.0,    # axial preload per bearing
    rpm_target: float = 60_000.0,
    grease: bool = False,       # oil-air lubrication for ≥30 000 rpm
) -> dict:
    """
    Angular contact bearing stiffness (Hertz contact, axial preload).
    k_a = 1.5 × Q / δ_a   (linearised about preload point)
    dm·n limit: grease lubrication typically 0.8×10^6 mm·rpm.

    Radial stiffness from axial preload (back-to-back pair):
    k_r ≈ k_a × tan²(α)   (simplified)
    """
    dm_mm = (d_mm + D_mm) / 2  # pitch diameter
    # DISCO high-speed spindles use oil-air lubrication (≥30 000 rpm)
    # Grease: 0.8×10^6 mm·rpm; Oil-air: 2.0×10^6 mm·rpm
    dn_limit = 0.8e6 if grease else 2.0e6  # mm·rpm
    dn = dm_mm * rpm_target
    speed_ok = dn < dn_limit

    alpha = np.radians(contact_angle_deg)
    # Hertz approach: approximate stiffness from Palmgren
    # k_a ≈ n_balls × sin(α) × (preload / n_balls)^(1/3) × K_H
    K_H = 7.86e4  # N^(2/3)/µm, for 52100 steel (Palmgren const)
    Q_per_ball = preload_N / (n_balls * np.sin(alpha))
    k_a_N_um = n_balls * np.sin(alpha) * K_H * Q_per_ball**(1/3) * 1e-6  # rough
    k_r_N_um = k_a_N_um * np.tan(alpha)**2

    return {
        "dm_mm": dm_mm,
        "dn_value": round(dn / 1e6, 3),
        "dn_limit_1e6": dn_limit / 1e6,
        "speed_ok": speed_ok,
        "k_axial_N_um": round(k_a_N_um, 1),
        "k_radial_N_um": round(k_r_N_um, 2),
        "balance_grade": "G0.4",  # ISO 1940-1: precision spindle
    }


# ── 3. Frame Modal Analysis (2-DOF lumped) ───────────────────────────────────

def frame_modal(
    m_spindle_kg: float = 8.0,    # spindle carriage mass
    m_stage_kg: float = 12.0,     # XY stage + chuck table
    k_spindle_Num: float = 5e6,   # Z-axis spindle feed stiffness [N/m]
    k_stage_Num: float = 8e6,     # XY stage drive stiffness [N/m]
    c_spindle: float = 500.0,     # damping coefficient [N·s/m]
    c_stage: float = 800.0,
) -> dict:
    """
    2-DOF undamped natural frequencies.
    [m1  0 ][ẍ1]   [k1+k12  -k12][x1]   [0]
    [0   m2][ẍ2] + [-k12   k2+k12][x2] = [0]
    Here treat as two independent 1-DOF for first estimate.
    """
    omega1 = np.sqrt(k_spindle_Num / m_spindle_kg)
    omega2 = np.sqrt(k_stage_Num / m_stage_kg)
    f1_Hz = omega1 / (2 * np.pi)
    f2_Hz = omega2 / (2 * np.pi)

    # Spindle excitation at blade pass frequency
    n_slots = 4
    rpm_range = np.array([20_000, 30_000, 40_000, 60_000])
    f_blade = rpm_range / 60 * n_slots  # [Hz]
    resonance_risk = [abs(f - f1_Hz) < 20 or abs(f - f2_Hz) < 20 for f in f_blade]

    zeta1 = c_spindle / (2 * np.sqrt(k_spindle_Num * m_spindle_kg))
    zeta2 = c_stage   / (2 * np.sqrt(k_stage_Num   * m_stage_kg))

    return {
        "f1_Hz": round(f1_Hz, 1),
        "f2_Hz": round(f2_Hz, 1),
        "zeta1": round(zeta1, 3),
        "zeta2": round(zeta2, 3),
        "blade_pass_Hz": dict(zip(rpm_range.tolist(), (f_blade).tolist())),
        "resonance_risk_rpm": [int(r) for r, risk in zip(rpm_range, resonance_risk) if risk],
    }


# ── 4. Vibration Isolation (air mount) ───────────────────────────────────────

def vibration_isolation(
    f_nat_Hz: float = 1.5,        # air mount natural frequency [Hz]
    damping_ratio: float = 0.05,
    f_disturb_Hz: float = 60.0,   # floor vibration (pump, HVAC)
) -> dict:
    """
    Transmissibility of a 1-DOF isolation system.
    TR(ω) = sqrt[(1+(2ζr)²) / ((1-r²)²+(2ζr)²)]  where r = f/f_nat
    Attenuation above √2 × f_nat.
    """
    r = f_disturb_Hz / f_nat_Hz
    zeta = damping_ratio
    TR = np.sqrt((1 + (2*zeta*r)**2) / ((1 - r**2)**2 + (2*zeta*r)**2))
    TR_dB = 20 * np.log10(TR)

    f_cross_Hz = f_nat_Hz * np.sqrt(2)  # crossover frequency

    return {
        "f_nat_Hz": f_nat_Hz,
        "f_disturb_Hz": f_disturb_Hz,
        "TR": round(TR, 4),
        "TR_dB": round(TR_dB, 1),
        "isolation_ok": TR_dB < -20,  # >20 dB attenuation target
        "crossover_Hz": round(f_cross_Hz, 2),
        "note": f"Isolation only above {f_cross_Hz:.1f} Hz; worsen below.",
    }


# ── 5. Thermal Axis Drift ─────────────────────────────────────────────────────

def thermal_drift(
    dT_spindle_K: float = 15.0,   # spindle temperature rise from idle [K]
    dT_stage_K: float = 5.0,      # stage table temperature rise
    alpha_steel: float = 11.7e-6, # CTE of steel [1/K]
    alpha_Al: float = 23.1e-6,    # CTE of aluminium (stage) [1/K]
    L_spindle_mm: float = 120.0,  # effective thermal length of spindle [mm]
    L_stage_mm: float = 200.0,    # stage length [mm]
) -> dict:
    """
    Thermal drift of blade tip relative to chuck table.
    δ_thermal = α × ΔT × L
    Kerf position budget: target <1 µm/hr after warm-up.
    """
    drift_spindle_um = alpha_steel * dT_spindle_K * (L_spindle_mm * 1e-3) * 1e6
    drift_stage_um   = alpha_Al   * dT_stage_K   * (L_stage_mm   * 1e-3) * 1e6
    total_drift_um   = drift_spindle_um + drift_stage_um

    return {
        "drift_spindle_um": round(drift_spindle_um, 2),
        "drift_stage_um":   round(drift_stage_um, 2),
        "total_drift_um":   round(total_drift_um, 2),
        "compensation_needed": total_drift_um > 5.0,  # >5 µm needs ATC
        "note": "Use linear encoder feedback or ATC (Auto Thermal Compensation) if >5 µm",
    }


# ── 6. Tolerance Stack-Up (kerf position) ────────────────────────────────────

def tolerance_stackup(components: list[dict] | None = None) -> dict:
    """
    RSS tolerance stack-up for kerf position accuracy.
    T_total_RSS = sqrt(ΣT_i²)

    Default components represent a DISCO dicing saw kerf budget:
      - Blade run-out (mounted)
      - Stage positioning (encoder + drive)
      - Thermal drift (compensated residual)
      - Vision alignment (fiducial detect error)
      - Wafer chuck flatness
    """
    if components is None:
        components = [
            {"name": "blade_runout",       "tol_um": 0.5},
            {"name": "stage_position",     "tol_um": 0.3},
            {"name": "thermal_residual",   "tol_um": 0.8},
            {"name": "vision_alignment",   "tol_um": 0.4},
            {"name": "chuck_flatness",     "tol_um": 0.3},
        ]
    T_rss = np.sqrt(sum(c["tol_um"]**2 for c in components))
    T_wc  = sum(c["tol_um"] for c in components)

    return {
        "components": components,
        "T_RSS_um": round(T_rss, 3),
        "T_worst_case_um": round(T_wc, 2),
        "spec_um": 2.0,
        "RSS_ok": T_rss < 2.0,
    }


# ── System summary ─────────────────────────────────────────────────────────────

def system_summary() -> dict:
    stiff  = spindle_housing_stiffness()
    brng   = bearing_pair(grease=False)  # oil-air lube for 60 000 rpm
    modal  = frame_modal()
    isol   = vibration_isolation()
    drift  = thermal_drift()
    tol    = tolerance_stackup()

    print("=" * 60)
    print("DISCO dicing saw — Mechanical Structure Summary")
    print("=" * 60)

    print(f"\n[Spindle Housing Stiffness]")
    st = "OK" if stiff["k_adequate"] else "NG"
    print(f"  δ_tip = {stiff['delta_um']} µm @ {stiff['F_radial_N']} N  →  k = {stiff['k_N_um']} N/µm  [{st}]")

    print(f"\n[Bearing Pair (angular contact)]")
    sp = "OK" if brng["speed_ok"] else "OVER LIMIT"
    print(f"  dm·n = {brng['dn_value']:.3f}×10⁶ mm·rpm  (limit {brng['dn_limit_1e6']}×10⁶)  [{sp}]")
    print(f"  k_axial = {brng['k_axial_N_um']} N/µm,  k_radial = {brng['k_radial_N_um']} N/µm")
    print(f"  Balance grade: {brng['balance_grade']} (ISO 1940-1)")

    print(f"\n[Frame Modal Analysis]")
    print(f"  f1 = {modal['f1_Hz']} Hz (spindle Z),  f2 = {modal['f2_Hz']} Hz (stage XY)")
    if modal["resonance_risk_rpm"]:
        print(f"  ⚠ Resonance risk at: {modal['resonance_risk_rpm']} rpm")
    else:
        print(f"  No resonance risk in operating range")

    print(f"\n[Vibration Isolation]")
    iso_st = "OK" if isol["isolation_ok"] else "NG"
    print(f"  TR = {isol['TR_dB']} dB @ {isol['f_disturb_Hz']} Hz  [{iso_st}]")

    print(f"\n[Thermal Axis Drift]")
    comp = "→ ATC required" if drift["compensation_needed"] else "→ within spec"
    print(f"  Total = {drift['total_drift_um']} µm  {comp}")

    print(f"\n[Kerf Tolerance Stack-up (RSS)]")
    tol_st = "PASS" if tol["RSS_ok"] else "FAIL"
    print(f"  T_RSS = {tol['T_RSS_um']} µm  (spec {tol['spec_um']} µm)  [{tol_st}]")
    for c in tol["components"]:
        print(f"    {c['name']:25s} {c['tol_um']} µm")
    print()

    return {
        "stiffness": stiff, "bearing": brng, "modal": modal,
        "isolation": isol, "thermal": drift, "tolerance": tol,
    }


if __name__ == "__main__":
    system_summary()
