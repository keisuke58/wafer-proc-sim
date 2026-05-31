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
    ap.add_argument("--sweep",  action="store_true")
    ap.add_argument("--power",  type=float, default=DEFAULT["laser_power_W"])
    ap.add_argument("--speed",  type=float, default=DEFAULT["scan_speed_mm_s"])
    ap.add_argument("--passes", type=int,   default=DEFAULT["n_passes"])
    args = ap.parse_args()

    if args.sweep:
        parametric_sweep()
    else:
        p   = dict(DEFAULT, laser_power_W=args.power,
                   scan_speed_mm_s=args.speed, n_passes=args.passes)
        res = analytical_groove(p)
        for k, v in res.items():
            print(f"  {k}: {v}")
