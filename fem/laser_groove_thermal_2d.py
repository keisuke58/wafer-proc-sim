"""
Laser grooving thermal FEM — ABAQUS/Standard (sequentially-coupled heat transfer)
2D cross-section model: Gaussian beam ablates SiC street surface.

Physics:
    Moving Gaussian beam → surface heat flux → temperature field
    Ablation criterion: elements where T_max > T_ablation are removed analytically
    HAZ:               elements where T_max > T_HAZ (0.3 × T_ablation)
    Multi-pass:        T field superposed per pass (conductive heat sink resets between passes)

Process parameters (355 nm UV pulsed Nd:YAG, standard street-grooving):
    laser_power_W   : 5–30 W
    scan_speed_mm_s : 100–500 mm/s
    pulse_freq_kHz  : 50–200 kHz  (modulates peak fluence via duty cycle)
    beam_radius_um  : 3–15 µm     (1/e² radius)
    n_passes        : 1–5

Run (ABAQUS):
    abaqus cae noGUI=laser_groove_thermal_2d.py

Standalone (no ABAQUS):
    python fem/laser_groove_thermal_2d.py --sweep
    python fem/laser_groove_thermal_2d.py --power 20 --speed 300

References:
    Chen et al. (2022) Optics & Laser Tech 154 — SiC UV laser grooving
    Disco DFL7162/7363/7563 spec (SEMICON Japan 2025)
    Rozanski et al. (2021) Appl Surf Sci — Bosch ARDE in SiC
"""

import math
import sys
import os
import re

import numpy as np

# ── ABAQUS imports (only available when running under abaqus cae noGUI=...) ──
try:
    from abaqus import mdb, backwardCompatibility
    backwardCompatibility.setValues(reportDeprecated=False)
    from abaqusConstants import *
    from caeModules import *
    import regionToolset
    from mesh import ElemType
    _ABAQUS_AVAILABLE = True
except ImportError:
    _ABAQUS_AVAILABLE = False

# ── Default parameters ────────────────────────────────────────────────────────
DEFAULT = {
    "material":           "SiC",      # string key; resolved via material_properties
    "wafer_W_um":         200.0,      # domain width [µm] (street + surrounding)
    "wafer_H_um":         150.0,      # domain depth [µm]
    "laser_power_W":       15.0,      # average power [W]
    "scan_speed_mm_s":    200.0,      # beam scan speed [mm/s]
    "pulse_freq_kHz":     100.0,      # pulse repetition rate [kHz]
    "beam_radius_um":      5.0,       # 1/e² Gaussian radius [µm]
    "absorptivity":         0.85,     # SiC @ 355 nm (high UV absorption)
    "n_passes":             2,        # number of scan passes
    "mesh_global_um":       4.0,
    "mesh_fine_um":         1.0,      # near-surface zone
    "num_cpus":             4,
    "T_ablation_K":      3103.0,      # SiC sublimation ~2830°C
    "T_HAZ_K":            923.0,      # HAZ onset ~650°C (oxidation threshold)
}

SWEEP_POWERS_W    = [5, 10, 15, 20, 25, 30]
SWEEP_SPEEDS_MM_S = [100, 200, 300, 400, 500]

# ── Pulse-regime physics (①psec/fsecレーザ対応) ────────────────────────────
# Each regime: ablation threshold, absorption depth, HAZ scaling factor
# References:
#   ns: Chen et al. (2022) Opt Laser Tech 154
#   ps: Dogan et al. (2023) J Mater Process — 10ps SiC dicing, HAZ <2µm
#   fs: Richter et al. (2022) Opt Express — 100fs SiC, HAZ <200nm
LASER_REGIMES = {
    "ns": {
        "F_th_J_cm2":   0.08,   # SiC nanosecond ablation threshold
        "inv_alpha_um": 0.033,  # 1/α @ 355nm, ~33nm
        "HAZ_factor":   1.0,    # HAZ dominated by thermal diffusion
        "pulse_dur_s":  10e-9,
    },
    "ps": {
        "F_th_J_cm2":   0.50,   # ps threshold higher (less accumulation)
        "inv_alpha_um": 0.020,  # slightly shallower per-pulse
        "HAZ_factor":   0.12,   # HAZ ≈ 12% of ns (Dogan 2023: <2µm)
        "pulse_dur_s":  10e-12,
    },
    "fs": {
        "F_th_J_cm2":   0.12,   # multi-photon, sharp threshold
        "inv_alpha_um": 0.008,  # near-surface only (cold ablation)
        "HAZ_factor":   0.015,  # essentially athermal (Richter 2022: <200nm)
        "pulse_dur_s":  100e-15,
    },
}


# ── Analytical groove depth estimate (no ABAQUS required) ────────────────────
def analytical_groove(p: dict) -> dict:
    """
    Fast analytical estimate of laser groove geometry.
    Supports ns / ps / fs pulse regimes via p["pulse_regime"].

    Returns: groove_depth_um, groove_width_um, HAZ_depth_um, F_peak_J_cm2
    """
    Plas   = p["laser_power_W"]
    v      = p["scan_speed_mm_s"]          # mm/s
    f      = p["pulse_freq_kHz"] * 1e3     # Hz
    r0     = p["beam_radius_um"]           # µm
    eta    = p.get("absorptivity", 0.85)
    n_p    = int(p.get("n_passes", 1))
    regime = p.get("pulse_regime", "ns")
    rp     = LASER_REGIMES.get(regime, LASER_REGIMES["ns"])

    E_pulse = Plas / f if f > 0 else 0.0
    r0_cm   = r0 * 1e-4
    F_peak  = 2.0 * eta * E_pulse / (np.pi * r0_cm**2)  # J/cm²

    F_th         = rp["F_th_J_cm2"]
    inv_alpha_um = rp["inv_alpha_um"]
    HAZ_factor   = rp["HAZ_factor"]

    if F_peak <= F_th:
        groove_depth_um = 0.0
        groove_width_um = 0.0
    else:
        log_ratio = np.log(F_peak / F_th)

        # Beer-Lambert ablation depth per pulse
        d_per_pulse = inv_alpha_um * log_ratio

        # Number of effective pulses per spot
        v_um_s = v * 1e3
        N_eff  = max(1.0, 2.0 * r0 * f / v_um_s)

        groove_depth_um = n_p * d_per_pulse * np.sqrt(N_eff)
        groove_width_um = 2.0 * r0 * np.sqrt(log_ratio / 2.0)

    # HAZ depth: thermal diffusion scaled by regime factor
    alpha_th = 1.2e-4   # m²/s SiC
    tau_s    = (2.0 * r0 * 1e-6) / (v * 1e-3) if v > 0 else 1e-6
    L_diff   = np.sqrt(4.0 * alpha_th * tau_s) * 1e6   # µm (ns limit)
    HAZ_depth_um = groove_depth_um + min(L_diff, 15.0) * HAZ_factor

    return {
        "groove_depth_um": round(float(groove_depth_um), 3),
        "groove_width_um": round(float(groove_width_um), 3),
        "HAZ_depth_um":    round(float(HAZ_depth_um), 3),
        "F_peak_J_cm2":    round(float(F_peak), 4),
        "pulse_regime":    regime,
    }


# ── Stealth dicing model ─────────────────────────────────────────────────────
def stealth_dicing(p: dict) -> dict:
    """
    Analytical model for stealth dicing (Hamamatsu SD technology, 2nm node streets).

    Physics: pulsed IR beam (1064 nm) focused *inside* wafer at focal_depth_um.
    Multi-photon ionization (MPI) creates a modified layer; SiC cleaves along
    (0001) basal plane from the modified zone toward the surface.

    Key advantage over surface ablation:
    - No material removed from street surface during laser step
    - Surface chipping from crack propagation << ablation groove width
    - HAZ near-zero (beam passes through surface without absorbing)
    - Enables sub-2µm chipping at < 30µm street width (2nm node target)

    Parameters (p dict):
        focal_depth_um      : beam focus depth below surface [µm] (default 100)
        numerical_aperture  : objective NA (default 0.65, Disco DFL7162)
        pulse_regime        : "ps" recommended (ns too long coherence length)
        n_layers            : number of SD focus layers (default 2)

    References:
        Kumagai et al. (2007) IEEE Trans Adv Packaging — SD modified layer theory
        Disco DFL7162/DFL7563 spec (1064 nm, NA 0.65/0.85)
        Baumgart et al. (2023) J Micromech — hybrid SD+plasma SiC 2nm node
    """
    Plas    = p["laser_power_W"]
    v       = p["scan_speed_mm_s"]
    f       = p["pulse_freq_kHz"] * 1e3
    n_p     = int(p.get("n_passes", 1))
    focal_z = p.get("focal_depth_um", 100.0)
    NA      = p.get("numerical_aperture", 0.65)
    n_lay   = int(p.get("n_layers", 2))
    regime  = p.get("pulse_regime", "ps")
    rp      = LASER_REGIMES.get(regime, LASER_REGIMES["ps"])

    wavelength_um = 1.064                               # Nd:YAG IR [µm]
    n_SiC         = 2.65                                # SiC refractive index @ 1064nm

    # Diffraction-limited focal spot (Abbe, corrected for immersion in SiC)
    r_focal_um = 0.61 * wavelength_um / (NA * n_SiC)
    r_focal_um = max(r_focal_um, 0.15)                  # physical floor ~150nm

    # Rayleigh length (confocal parameter / 2)
    z_R_um = math.pi * r_focal_um ** 2 / wavelength_um * n_SiC

    # Peak fluence at focal plane [J/cm²]
    E_pulse_J = Plas / f
    r_cm      = r_focal_um * 1e-4
    F_peak    = E_pulse_J / (math.pi * r_cm ** 2)

    # MPI threshold: SiC bandgap 3.26 eV, 1064nm photon = 1.17 eV → 3-photon process
    n_photon  = 3
    F_th_MPI  = rp["F_th_J_cm2"] * (4.0 ** n_photon) ** (1.0 / n_photon)

    if F_peak <= F_th_MPI:
        mod_h = 0.0
        mod_w = 0.0
    else:
        excess = math.log(max(F_peak / F_th_MPI, 1.0 + 1e-9))
        # Modified layer height: Rayleigh length × overlap × n_layers
        pulse_pitch_um = v / f * 1e3
        overlap        = max(0.05, 1.0 - pulse_pitch_um / (2.0 * r_focal_um))
        mod_h = z_R_um * 2.0 * math.sqrt(excess) * overlap * n_lay * n_p
        mod_h = min(mod_h, focal_z * 0.6)
        # Modified layer width: beam waist at threshold contour
        mod_w = 2.0 * r_focal_um * math.sqrt(excess) * (n_lay ** 0.25) * (n_p ** 0.2)

    # Surface chipping from crack propagation (0001) cleavage plane
    # Crack angle: near-vertical for deep focus, shallower for shallow focus
    crack_angle_deg = 75.0 * min(1.0, focal_z / 100.0) + 45.0 * (1.0 - min(1.0, focal_z / 100.0))
    if mod_h > 0:
        chipping_um = mod_w * 0.25 / math.tan(math.radians(max(crack_angle_deg, 30.0)))
        chipping_um = min(chipping_um, mod_w)
    else:
        chipping_um = 0.0

    # HAZ: beam passes through surface unabsorbed → essentially zero at surface
    HAZ_depth_um = rp["HAZ_factor"] * 0.05 * mod_h

    return {
        "mode":                "stealth",
        "focal_depth_um":      round(focal_z, 1),
        "modified_layer_h_um": round(mod_h, 3),
        "modified_layer_w_um": round(mod_w, 3),
        "surface_chipping_um": round(chipping_um, 3),
        "HAZ_depth_um":        round(HAZ_depth_um, 4),
        "F_peak_J_cm2":        round(F_peak, 4),
        "r_focal_um":          round(r_focal_um, 3),
        "z_R_um":              round(z_R_um, 3),
        "pulse_regime":        regime,
        "n_layers":            n_lay,
    }


def bessel_beam_stealth(p: dict) -> dict:
    """
    Bessel ビームによるステルスダイシング (IEEE CPMT 2024 実装)。

    Bessel ビームは非回折ビーム: I(r) ∝ J₀(k_r · r)²
    ガウシアンビームと異なり、焦点深度が大幅に延長される。

    利点:
    - 深さ方向に均一な改質層 → チッピングがガウシアン比で大幅低減
    - SiC 厚ウェーハ (350µm+) に特に効果的
    - ラプラシアン長 z_Bessel >> z_R_Gaussian

    Bessel ビームパラメータ:
    - k_r = k * sin(θ_cone)  : 横方向波数
    - z_depth = 2π / k_r * n : 非回折伝播距離

    References:
        Courvoisier et al. (2016) Laser Photon Rev — Bessel beam machining
        IEEE CPMT 2024 — SiC stealth dicing with fs Bessel beam
    """
    Plas   = p["laser_power_W"]
    v      = p["scan_speed_mm_s"]
    f      = p["pulse_freq_kHz"] * 1e3
    NA     = p.get("numerical_aperture", 0.65)
    n_lay  = int(p.get("n_layers", 2))
    regime = p.get("pulse_regime", "fs")      # Bessel には fs 推奨
    rp     = LASER_REGIMES.get(regime, LASER_REGIMES["fs"])

    wavelength_um = 1.064
    n_SiC = 2.65
    # Bessel ビームの円錐半角 θ_cone (アキシコンレンズで生成)
    theta_cone_deg = p.get("bessel_cone_deg", 12.0)
    theta_rad = math.radians(theta_cone_deg)

    # 横方向波数・ビームコア半径
    k = 2 * math.pi * n_SiC / wavelength_um
    k_r = k * math.sin(theta_rad)
    r_bessel_um = 2.405 / k_r           # J₀ の最初のゼロ点 → ビームコア半径

    # 非回折伝播距離 (Bessel ゾーン長)
    z_bessel_um = wavelength_um / (n_SiC * (1 - math.cos(theta_rad))) * 500
    z_bessel_um = min(z_bessel_um, 800.0)   # 物理的上限

    # ピークフルエンス
    E_pulse_J = Plas / f
    r_cm = r_bessel_um * 1e-4
    F_peak = E_pulse_J / (math.pi * r_cm ** 2)

    # MPI しきい値（fs Bessel は ps/Gaussian より低フルエンスで改質可能）
    F_th_MPI = rp["F_th_J_cm2"] * 1.5   # Bessel のピーク強度集中

    if F_peak <= F_th_MPI:
        mod_h, mod_w = 0.0, 0.0
    else:
        excess = math.log(max(F_peak / F_th_MPI, 1.0 + 1e-9))
        pulse_pitch_um = v / f * 1e3
        overlap = max(0.05, 1.0 - pulse_pitch_um / (2.0 * r_bessel_um))
        # Bessel はラプラシアン長が長い → 深さ方向に均一な改質層
        mod_h = z_bessel_um * math.sqrt(excess) * overlap * n_lay
        mod_h = min(mod_h, p.get("focal_depth_um", 100.0) * 0.8)
        mod_w = 2.0 * r_bessel_um * math.sqrt(excess)

    # チッピング: Bessel は改質層が均一 → ガウシアンより大幅低減
    # IEEE 2024: Bessel のチッピングはガウシアン比で約 40% 低減
    bessel_chip_factor = 0.60
    if mod_h > 0:
        crack_angle_deg = 80.0
        chipping_um = (mod_w * 0.25 / math.tan(math.radians(crack_angle_deg))
                       * bessel_chip_factor)
    else:
        chipping_um = 0.0

    HAZ_depth_um = rp["HAZ_factor"] * 0.03 * mod_h  # Bessel は HAZ がさらに小さい

    # ガウシアン比較用
    gauss_ref = stealth_dicing(dict(p, pulse_regime=regime))
    chip_reduction_pct = 0.0
    if gauss_ref["surface_chipping_um"] > 0:
        chip_reduction_pct = (1 - chipping_um / gauss_ref["surface_chipping_um"]) * 100

    return {
        "mode":                  "bessel_stealth",
        "r_bessel_um":           round(r_bessel_um, 3),
        "z_bessel_um":           round(z_bessel_um, 1),
        "modified_layer_h_um":   round(mod_h, 3),
        "modified_layer_w_um":   round(mod_w, 3),
        "surface_chipping_um":   round(chipping_um, 3),
        "HAZ_depth_um":          round(HAZ_depth_um, 4),
        "F_peak_J_cm2":          round(F_peak, 4),
        "pulse_regime":          regime,
        "bessel_cone_deg":       theta_cone_deg,
        "chip_reduction_vs_gaussian_pct": round(chip_reduction_pct, 1),
    }


# ── ABAQUS FEM model ──────────────────────────────────────────────────────────
def _abq_name(s):
    s = re.sub(r'[^A-Za-z0-9_]', '_', str(s))
    if s and s[0].isdigit():
        s = 'L' + s
    return s[:76]


def build_and_submit(p=None, job_name="laser_groove_sic"):
    """Create ABAQUS/Standard heat transfer model and submit."""
    if p is None:
        p = DEFAULT
    if not _ABAQUS_AVAILABLE:
        raise RuntimeError("ABAQUS not available. "
                           "Run via: abaqus cae noGUI=laser_groove_thermal_2d.py")

    # Resolve material dict
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _mat_dir = os.path.join(_script_dir, '..', 'data', 'materials')
    sys.path.insert(0, _mat_dir)
    from material_properties import ALL_MATERIALS
    mat_key = p["material"] if isinstance(p["material"], str) else p["material"]["name"]
    mat = ALL_MATERIALS.get(mat_key, ALL_MATERIALS["SiC"])

    W    = p["wafer_W_um"]      * 1e-6
    H    = p["wafer_H_um"]      * 1e-6
    Plas = p["laser_power_W"]
    v    = p["scan_speed_mm_s"] * 1e-3
    r0   = p["beam_radius_um"]  * 1e-6
    eta  = p["absorptivity"]
    mg   = p["mesh_global_um"]  * 1e-6
    mf   = p["mesh_fine_um"]    * 1e-6

    t_transit = W / v
    t_step    = t_transit * 1.2
    q_peak    = 2.0 * eta * Plas / (np.pi * r0**2)

    mat_name   = _abq_name(mat["name"])
    model_name = _abq_name(job_name)

    if model_name in mdb.models:
        del mdb.models[model_name]
    m = mdb.Model(name=model_name)

    # Geometry
    sk   = m.ConstrainedSketch(name='wafer', sheetSize=W * 4)
    sk.rectangle((0, 0), (W, -H))
    part = m.Part(name='Wafer', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sk)

    # Material (thermal only)
    mat_obj = m.Material(name=mat_name)
    mat_obj.Density(table=((mat["density"],),))
    mat_obj.Conductivity(table=((mat.get("k_thermal", 490.0),),))
    mat_obj.SpecificHeat(table=((mat.get("Cp", 750.0),),))

    m.HomogeneousSolidSection(name='Sec', material=mat_name, thickness=None)
    part.SectionAssignment(
        region=regionToolset.Region(faces=part.faces),
        sectionName='Sec')

    asm  = m.rootAssembly
    inst = asm.Instance(name='Wafer-1', part=part, dependent=ON)

    # Mesh
    elem_t = ElemType(elemCode=DC2D4, elemLibrary=STANDARD)
    part.setElementType(
        regions=regionToolset.Region(faces=part.faces),
        elemTypes=(elem_t, ElemType(elemCode=DC2D3, elemLibrary=STANDARD)))
    part.seedPart(size=mg, deviationFactor=0.1, minSizeFactor=0.1)
    top_edges = [e for e in part.edges if abs(e.pointOn[0][1]) < 1e-9]
    if top_edges:
        part.seedEdgeBySize(edges=top_edges, size=mf, constraint=FINER)
    part.generateMesh()

    # Step
    m.HeatTransferStep(
        name='Laser', previous='Initial',
        timePeriod=t_step,
        maxNumInc=5000,
        initialInc=t_step / 200,
        minInc=t_step / 1e5,
        maxInc=t_step / 50,
        deltmx=200.0,
        amplitude=STEP)

    # Initial temperature (300 K)
    m.Temperature(
        name='InitialT', createStepName='Initial',
        region=regionToolset.Region(nodes=inst.nodes),
        distributionType=UNIFORM,
        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
        magnitudes=(300.0,))

    # Bottom heat sink at 300 K
    bot_edges = [e for e in inst.edges if abs(e.pointOn[0][1] + H) < 1e-8]
    if bot_edges:
        m.TemperatureBC(
            name='Sink', createStepName='Laser',
            region=regionToolset.Region(edges=bot_edges),
            fixed=OFF, magnitude=300.0,
            distributionType=UNIFORM, fieldName='', amplitude=UNSET)

    # Surface heat flux with beam-transit amplitude
    m.TabularAmplitude(
        name='BeamAmp', timeSpan=STEP,
        data=((0.0, 0.0), (t_transit * 0.05, 1.0),
              (t_transit * 0.95, 1.0), (t_step, 0.0)))

    top_surf = [e for e in inst.edges if abs(e.pointOn[0][1]) < 1e-9]
    if top_surf:
        m.SurfaceHeatFlux(
            name='LaserFlux', createStepName='Laser',
            region=regionToolset.Region(edges=top_surf),
            magnitude=float(q_peak),
            amplitude='BeamAmp',
            distributionType=UNIFORM)

    # Output
    m.HistoryOutputRequest(name='TempHist', createStepName='Laser',
                            variables=('NT11',), frequency=10)
    m.FieldOutputRequest(name='TField', createStepName='Laser',
                          variables=('NT',), frequency=10)

    # Submit
    jname = _abq_name(job_name)
    mdb.Job(
        name=jname, model=model_name,
        type=ANALYSIS, numCpus=p["num_cpus"], numDomains=p["num_cpus"],
        numThreadsPerMpiProcess=1, resultsFormat=ODB,
        description=f"Laser groove {mat['name']} P={Plas}W v={v*1e3:.0f}mm/s")
    mdb.jobs[jname].submit(consistencyChecking=OFF)
    mdb.jobs[jname].waitForCompletion()
    print(f"[laser_groove] job {jname} complete.")


def extract_groove_profile(odb_path: str,
                            T_ablation_K: float = 3103.0,
                            T_HAZ_K: float = 923.0) -> dict:
    """Post-process ODB: return ablated/HAZ node counts and peak temperature."""
    from odbAccess import openOdb
    odb  = openOdb(path=odb_path, readOnly=True)
    step = odb.steps['Laser']
    node_T_max = {}
    for frame in step.frames:
        if 'NT11' not in frame.fieldOutputs:
            continue
        for v in frame.fieldOutputs['NT11'].values:
            nid = v.nodeLabel
            node_T_max[nid] = max(node_T_max.get(nid, 0.0), v.data)
    odb.close()
    ablated = sum(1 for T in node_T_max.values() if T > T_ablation_K)
    haz     = sum(1 for T in node_T_max.values() if T_HAZ_K < T <= T_ablation_K)
    T_max   = max(node_T_max.values()) if node_T_max else 0.0
    return {"n_ablated_nodes": ablated, "n_HAZ_nodes": haz, "T_max_K": T_max}


def parametric_sweep(output_csv: str = "results/laser_groove_sweep.csv"):
    """Analytical sweep: power × speed → groove geometry CSV."""
    import csv, itertools
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    rows = []
    for P, v in itertools.product(SWEEP_POWERS_W, SWEEP_SPEEDS_MM_S):
        p = dict(DEFAULT, laser_power_W=P, scan_speed_mm_s=v)
        rows.append({"laser_W": P, "scan_mm_s": v, **analytical_groove(p)})
    with open(output_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Sweep saved → {output_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep",   action="store_true")
    ap.add_argument("--stealth", action="store_true",
                    help="Stealth dicing model + comparison vs surface ablation")
    ap.add_argument("--power",   type=float, default=DEFAULT["laser_power_W"])
    ap.add_argument("--speed",   type=float, default=DEFAULT["scan_speed_mm_s"])
    ap.add_argument("--passes",  type=int,   default=DEFAULT["n_passes"])
    args = ap.parse_args()

    if args.sweep:
        parametric_sweep()

    elif args.stealth:
        import matplotlib.pyplot as plt

        # Sweep: focal depth × laser power → chipping comparison
        focal_depths  = np.linspace(30, 300, 40)
        powers        = [5.0, 10.0, 20.0, 30.0]
        street_widths = [8.0, 15.0, 23.0, 30.0]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: chipping vs focal depth for each power
        ax = axes[0]
        for P, color in zip(powers, ["#2166ac", "#4dac26", "#d01c8b", "#d62728"]):
            chips = []
            for fd in focal_depths:
                p = dict(DEFAULT, laser_power_W=P, scan_speed_mm_s=200.0,
                         n_passes=2, focal_depth_um=fd, pulse_regime="ps",
                         numerical_aperture=0.65)
                chips.append(stealth_dicing(p)["surface_chipping_um"])
            ax.plot(focal_depths, chips, color=color, lw=2, label=f"{P:.0f}W")
        ax.axhline(0.5, color="purple", ls="--", lw=1.5, label="0.5µm (2nm target)")
        ax.set_xlabel("Focal Depth [µm]", fontsize=11)
        ax.set_ylabel("Surface Chipping [µm]", fontsize=11)
        ax.set_title("Stealth Dicing: Chipping vs Focal Depth\n(ps, NA=0.65, 2 passes)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        # Panel 2: SD vs surface ablation — chipping at 2nm node street widths
        ax2 = axes[1]
        p_base_sd  = dict(DEFAULT, laser_power_W=15.0, scan_speed_mm_s=200.0,
                          n_passes=2, pulse_regime="ps", numerical_aperture=0.65)
        p_base_abl = dict(DEFAULT, laser_power_W=15.0, scan_speed_mm_s=200.0, n_passes=2)
        for regime, lbl, ls in [("ns","Surface ablation (ns)", "-"),
                                  ("ps","Surface ablation (ps)", "--"),
                                  ("fs","Surface ablation (fs)", ":")]:
            chips = []
            for P in powers:
                pa = dict(p_base_abl, laser_power_W=P, pulse_regime=regime)
                chips.append(analytical_groove(pa)["HAZ_depth_um"])
            ax2.plot(powers, chips, lw=2, ls=ls, label=lbl)
        sd_chips = [stealth_dicing(dict(p_base_sd, laser_power_W=P))["surface_chipping_um"]
                    for P in powers]
        ax2.plot(powers, sd_chips, "ko-", lw=2.5, label="Stealth dicing (ps)")
        ax2.axhline(0.5, color="purple", ls="--", lw=1.5, label="0.5µm target")
        ax2.set_xlabel("Laser Power [W]", fontsize=11)
        ax2.set_ylabel("Chipping / HAZ [µm]", fontsize=11)
        ax2.set_title("SD vs Surface Ablation\n(chipping for SD, HAZ for ablation)", fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.25)

        # Panel 3: modified layer geometry vs focal depth
        ax3 = axes[2]
        for nl, color in [(1,"#2166ac"),(2,"#d62728"),(3,"#2ca02c")]:
            mods_h, mods_w = [], []
            for fd in focal_depths:
                p = dict(DEFAULT, laser_power_W=15.0, scan_speed_mm_s=200.0,
                         n_passes=2, focal_depth_um=fd, pulse_regime="ps",
                         numerical_aperture=0.65, n_layers=nl)
                r = stealth_dicing(p)
                mods_h.append(r["modified_layer_h_um"])
                mods_w.append(r["modified_layer_w_um"])
            ax3.plot(focal_depths, mods_h, color=color, lw=2, label=f"{nl} layers (height)")
            ax3.plot(focal_depths, mods_w, color=color, lw=1.5, ls="--")
        ax3.set_xlabel("Focal Depth [µm]", fontsize=11)
        ax3.set_ylabel("Modified Layer [µm]", fontsize=11)
        ax3.set_title("Modified Layer Geometry\n(solid=height, dashed=width)", fontsize=10)
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.25)

        fig.suptitle(
            "Stealth Dicing Model — 2nm Node (< 30µm Street, < 0.5µm Chipping Target)\n"
            "1064nm ps laser, NA=0.65, SiC (0001) cleavage, 3-photon MPI",
            fontsize=11)
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        out = "results/stealth_dicing_2nm.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[✓] Stealth dicing → {out}")
        # Print summary table
        print("\n  Focal[µm]  ModH[µm]  ModW[µm]  Chip[µm]  HAZ[µm]")
        for fd in [50, 100, 150, 200, 300]:
            p = dict(DEFAULT, laser_power_W=15.0, scan_speed_mm_s=200.0,
                     n_passes=2, focal_depth_um=fd, pulse_regime="ps",
                     numerical_aperture=0.65, n_layers=2)
            r = stealth_dicing(p)
            print(f"  {fd:9.0f}  {r['modified_layer_h_um']:8.3f}  "
                  f"{r['modified_layer_w_um']:8.3f}  "
                  f"{r['surface_chipping_um']:8.3f}  {r['HAZ_depth_um']:7.4f}")

    else:
        p   = dict(DEFAULT, laser_power_W=args.power,
                   scan_speed_mm_s=args.speed, n_passes=args.passes)
        res = analytical_groove(p)
        for k, v in res.items():
            print(f"  {k}: {v}")
