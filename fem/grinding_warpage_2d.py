"""
2D plane-strain grinding FEM — ABAQUS/Standard (warpage prediction)
Si/SiC wafer backside grinding: residual stress → warpage after chuck release.

Dimensionality: TWO_D_PLANAR with CPE4R elements (plane strain).
Note: AXISYMMETRIC (CAX4R) gives the same trend but fails on this installation.
      Plane-strain overestimates out-of-plane stress slightly; warpage trend is preserved.

Physics:
  Oxford 2023 force model → distributed pressure on top surface.
  Step 1 "Grind"  : load + vacuum chuck → stress buildup.
  Step 2 "Release": chuck removed → wafer warps freely.
  TAIKO variant   : edge ring geometry to constrain warpage.

Outputs (via extract_grinding_results.py):
  warpage_um               peak-to-valley of U2 on top surface [µm]
  max_residual_stress_MPa  max S22 after Release step [MPa]

Run:
    abaqus cae noGUI=grinding_warpage_2d.py   (reads run_config.json in cwd)

run_config.json keys:
    material, wheel_speed_rpm, feed_rate_um_min, cut_depth_um,
    wafer_diameter_mm, wafer_thickness_um,
    taiko_mode, taiko_edge_width_mm, taiko_thin_um,
    mesh_global_um, mesh_fine_um, num_cpus, study, job_name
"""

import sys
import os
import json
import math

# ── Script path resolution (ABAQUS noGUI: __file__ undefined) ─────────────────
_abq_script = None
for _i, _a in enumerate(sys.argv[:-1]):
    if _a == '-noGUI':
        _abq_script = os.path.abspath(sys.argv[_i + 1])
        break
if _abq_script is None:
    raise RuntimeError("Cannot locate -noGUI script: " + str(sys.argv))

_mat_dir = os.path.normpath(
    os.path.join(os.path.dirname(_abq_script), '..', 'data', 'materials'))
sys.path.insert(0, _mat_dir)
from material_properties import Si, SiC, GaN, ALL_MATERIALS

from abaqus import mdb, backwardCompatibility
backwardCompatibility.setValues(reportDeprecated=False)
from abaqusConstants import *
from caeModules import *
import regionToolset
from mesh import ElemType

DEFAULT = {
    "material":            Si,
    "wheel_speed_rpm":     4000.0,    # grinding spindle speed [rpm]
    "feed_rate_um_min":    300.0,     # workpiece feed rate [µm/min]
    "cut_depth_um":        20.0,      # cut depth per pass [µm]
    "wafer_diameter_mm":   300.0,     # wafer diameter [mm]
    "wafer_thickness_um":  500.0,     # wafer thickness before grinding [µm]
    "wheel_diameter_mm":   200.0,     # grinding wheel outer diameter [mm]
    "wheel_width_mm":      5.0,       # wheel contact width [mm]
    "grinding_mu":         0.35,      # grinding friction coefficient
    "taiko_mode":          False,     # True → TAIKO edge-ring geometry
    "taiko_edge_width_mm": 3.0,       # TAIKO edge ring radial width [mm]
    "taiko_thin_um":       50.0,      # TAIKO thinned inner region thickness [µm]
    "mesh_global_um":      2000.0,    # global seed [µm] (2mm for 300mm wafer)
    "mesh_fine_um":        100.0,     # fine seed at grind zone and edge [µm]
    "t_ssd_um":            10.0,      # subsurface damage layer thickness [µm]
    "E_damage_factor":      0.5,      # E_damaged / E_bulk (microcracks; lit: 0.3–0.7)
    "num_cpus":            4,
}

SWEEP_WHEEL_SPEEDS  = [2000, 4000, 6000, 8000]   # rpm
SWEEP_FEED_RATES    = [200,  300,  500,  800]     # µm/min
SWEEP_CUT_DEPTHS    = [10,   20,   30,   50]      # µm


def _grinding_force(p):
    """
    Normal and tangential grinding force per unit width [N/m].
    Simplified Oxford 2023 formulation (eq. 2-3, backside grinding).

    Ft/b = C_t × (v_f / v_s)^0.6 × a_p^0.4
    Fn   = Ft / mu_g

    Returns (Fn_per_width [N/m], Ft_per_width [N/m], contact_length [m]).
    """
    v_s = math.pi * p["wheel_diameter_mm"] * 1e-3 * p["wheel_speed_rpm"] / 60.0
    v_f = p["feed_rate_um_min"] * 1e-6 / 60.0  # m/s
    a_p = p["cut_depth_um"] * 1e-6              # m

    C_t = p["material"].get("C_t_grinding", 1800.0)   # material-dependent [N/m]

    Ft_per_width = C_t * ((v_f / v_s) ** 0.6) * (a_p ** 0.4)
    Fn_per_width = Ft_per_width / p["grinding_mu"]

    # Geometric contact length (Malkin 2008)
    l_contact = math.sqrt(a_p * p["wheel_diameter_mm"] * 1e-3)

    return Fn_per_width, Ft_per_width, l_contact


def _abq_name(s):
    import re
    s = re.sub(r'[^A-Za-z0-9_]', '_', str(s))
    if s and s[0].isdigit():
        s = 'M_' + s
    return s


def build_model(p=DEFAULT, job_name="grind_Si_ws4000_f300_d020"):
    """Build ABAQUS/Standard axisymmetric warpage model. Returns job_name."""

    R   = p["wafer_diameter_mm"] * 1e-3 / 2.0   # wafer radius [m]
    t   = p["wafer_thickness_um"] * 1e-6         # full thickness [m]
    mat = p["material"]
    mat_name = _abq_name(mat["name"])

    taiko     = bool(p.get("taiko_mode", False))
    t_thin    = p["taiko_thin_um"]      * 1e-6 if taiko else t
    w_edge    = p["taiko_edge_width_mm"] * 1e-3 if taiko else 0.0
    r_step    = R - w_edge  # radius of TAIKO step

    mg = p["mesh_global_um"] * 1e-6
    mf = p["mesh_fine_um"]   * 1e-6

    # Grinding zone: centered at R/2, width = wheel contact width
    bw    = p["wheel_width_mm"] * 1e-3
    r_ctr = R / 2.0
    r_lo  = max(0.0, r_ctr - bw / 2.0)
    r_hi  = min(R,   r_ctr + bw / 2.0)

    Fn_pw, _Ft_pw, l_c = _grinding_force(p)
    p_grind = Fn_pw / l_c  # contact pressure [Pa]

    # ── Create / reset model ──────────────────────────────────────────────────
    if job_name in mdb.models:
        del mdb.models[job_name]
    model = mdb.Model(name=job_name)
    if "Model-1" in mdb.models and job_name != "Model-1":
        del mdb.models["Model-1"]

    # ═══ PART: Wafer (axisymmetric cross-section) ════════════════════════════
    sk = model.ConstrainedSketch(name="wafer_sk", sheetSize=R * 3)

    if taiko:
        # L-shaped cross-section: thin inner disk + thick edge ring
        # Outline (CCW): origin → outer bottom → outer top →
        #                step-top-outer → step height → inner top → origin
        sk.Line(point1=(0.0,    0.0),    point2=(R,      0.0))
        sk.Line(point1=(R,      0.0),    point2=(R,      t))
        sk.Line(point1=(R,      t),      point2=(r_step, t))
        sk.Line(point1=(r_step, t),      point2=(r_step, t_thin))
        sk.Line(point1=(r_step, t_thin), point2=(0.0,    t_thin))
        sk.Line(point1=(0.0,    t_thin), point2=(0.0,    0.0))
    else:
        sk.rectangle(point1=(0.0, 0.0), point2=(R, t))

    wafer = model.Part(name="Wafer", dimensionality=TWO_D_PLANAR,
                       type=DEFORMABLE_BODY)
    wafer.BaseShell(sketch=sk)

    # Partitions at grind zone boundaries for mesh refinement
    top_y = t_thin if taiko else t
    for r_cut in (r_lo, r_hi):
        if 0.0 < r_cut < R:
            wafer.PartitionFaceByShortestPath(
                point1=(r_cut, 0.0,   0.0),
                point2=(r_cut, top_y, 0.0),
                faces=wafer.faces[:])

    # ── Subsurface damage (SSD) partition ─────────────────────────────────────
    # Horizontal cut at y = top_y - t_ssd separates damaged layer from bulk.
    # Chen & Wolf (2003) Semicond. Sci. Technol. 18:261 show E_damaged ≈ 0.3–0.7 × E_bulk.
    t_ssd  = p.get("t_ssd_um", 10.0) * 1e-6
    E_dmg  = p.get("E_damage_factor", 0.5)
    y_ssd  = top_y - t_ssd

    if 0.0 < t_ssd < top_y * 0.5:
        r_end = r_step if taiko else R
        wafer.PartitionFaceByShortestPath(
            point1=(0.0,   y_ssd, 0.0),
            point2=(r_end, y_ssd, 0.0),
            faces=wafer.faces[:])

    # ═══ MATERIALS ════════════════════════════════════════════════════════════
    # Bulk (undamaged) Si
    m_bulk = model.Material(name=mat_name)
    m_bulk.Elastic(table=((mat["E"], mat["nu"]),))
    m_bulk.Density(table=((mat["density"],),))

    # Damaged surface layer — reduced E from microcrack network
    mat_dam = mat_name + "_dam"
    m_dam   = model.Material(name=mat_dam)
    m_dam.Elastic(table=((mat["E"] * E_dmg, mat["nu"]),))
    m_dam.Density(table=((mat["density"],),))

    model.HomogeneousSolidSection(name="SecBulk", material=mat_name,  thickness=1.0)
    model.HomogeneousSolidSection(name="SecDam",  material=mat_dam,   thickness=1.0)

    # Assign sections using getByBoundingBox (returns GeomSequence, not tuple)
    use_ssd = (0.0 < t_ssd < top_y * 0.5)
    eps_y   = t_ssd * 0.05 if use_ssd else 1e-9

    if use_ssd:
        dam_faces  = wafer.faces.getByBoundingBox(
            xMin=-eps_y, xMax=R + eps_y,
            yMin=y_ssd + eps_y, yMax=top_y + eps_y)
        bulk_faces = wafer.faces.getByBoundingBox(
            xMin=-eps_y, xMax=R + eps_y,
            yMin=-eps_y, yMax=y_ssd - eps_y)
        if len(dam_faces) > 0:
            wafer.SectionAssignment(
                region=regionToolset.Region(faces=dam_faces),
                sectionName="SecDam",
                offset=0.0, offsetType=MIDDLE_SURFACE,
                offsetField="", thicknessAssignment=FROM_SECTION)
        wafer.SectionAssignment(
            region=regionToolset.Region(
                faces=bulk_faces if len(bulk_faces) > 0 else wafer.faces[:]),
            sectionName="SecBulk",
            offset=0.0, offsetType=MIDDLE_SURFACE,
            offsetField="", thicknessAssignment=FROM_SECTION)
    else:
        wafer.SectionAssignment(
            region=regionToolset.Region(faces=wafer.faces[:]),
            sectionName="SecBulk",
            offset=0.0, offsetType=MIDDLE_SURFACE,
            offsetField="", thicknessAssignment=FROM_SECTION)

    # ═══ ASSEMBLY ═════════════════════════════════════════════════════════════
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    wafer_inst = a.Instance(name="Wafer-1", part=wafer, dependent=ON)

    # ═══ STEPS ════════════════════════════════════════════════════════════════
    model.StaticStep(name="Grind", previous="Initial",
                     nlgeom=ON, maxNumInc=200,
                     initialInc=0.1, minInc=1e-8, maxInc=1.0)
    model.StaticStep(name="Release", previous="Grind",
                     nlgeom=ON, maxNumInc=200,
                     initialInc=0.1, minInc=1e-8, maxInc=1.0)

    # ═══ BOUNDARY CONDITIONS ══════════════════════════════════════════════════
    # Vacuum chuck: bottom face fixed in z (U2=0), free radially
    bot_edges = wafer_inst.edges.getByBoundingBox(
        xMin=0.0, xMax=R * 1.01, yMin=-1e-7, yMax=1e-7)
    if len(bot_edges) == 0:
        bot_edges = wafer_inst.edges.findAt(((R / 4, 0.0, 0.0),))
    chuck_region = regionToolset.Region(edges=bot_edges)
    model.DisplacementBC(name="ChuckBC", createStepName="Initial",
                         region=chuck_region, u2=0.0)
    # Chuck released in Release step → wafer warps freely
    model.boundaryConditions["ChuckBC"].deactivate("Release")

    # Pin inner-bottom corner to suppress rigid-body axial translation in Release
    ctr_verts = wafer_inst.vertices.findAt(((0.0, 0.0, 0.0),))
    if len(ctr_verts) > 0:
        model.DisplacementBC(name="AxisPin", createStepName="Initial",
                             region=regionToolset.Region(vertices=ctr_verts),
                             u2=0.0)

    # ═══ LOAD: grinding pressure on top surface at grind zone ════════════════
    top_grind_edges = wafer_inst.edges.getByBoundingBox(
        xMin=r_lo * 0.99, xMax=r_hi * 1.01,
        yMin=top_y - 1e-7, yMax=top_y + 1e-7)
    if len(top_grind_edges) > 0:
        model.Pressure(name="GrindPress", createStepName="Grind",
                       region=regionToolset.Region(side1Edges=top_grind_edges),
                       magnitude=p_grind)
        print("[INP] Grinding pressure: %.2f MPa over r=[%.1f, %.1f] mm" % (
              p_grind / 1e6, r_lo * 1e3, r_hi * 1e3))
    else:
        print("[WARN] No grind-zone edges found — check r_lo/r_hi vs mesh")

    # ═══ MESH (plane strain: CPE4R) ═══════════════════════════════════════════
    elem_q = ElemType(elemCode=CPE4R, elemLibrary=STANDARD,
                      secondOrderAccuracy=OFF, hourglassControl=ENHANCED)
    elem_t = ElemType(elemCode=CPE3,  elemLibrary=STANDARD)
    wafer.setElementType(
        regions=regionToolset.Region(faces=wafer.faces[:]),
        elemTypes=(elem_q, elem_t))
    wafer.seedPart(size=mg, deviationFactor=0.1, minSizeFactor=0.1)

    # ── Thickness (Y) direction: explicit seeding to avoid extreme aspect ratio ─
    # Global seed mg (2–5mm) vs wafer thickness 0.5mm → aspect ratio 4–10× bad.
    # Find all near-vertical edges (length ≈ top_y) and seed them by number.
    n_tck = max(3, int(round(top_y / mf)))   # ≥3 elements through thickness
    y_thick = [e for e in wafer.edges[:]
               if top_y * 0.05 < e.getSize() < top_y * 1.5]
    if y_thick:
        wafer.seedEdgeByNumber(edges=y_thick, number=n_tck)

    # Finer mesh in grinding zone and near TAIKO step (stress concentration)
    fine_bb_edges = wafer.edges.getByBoundingBox(
        xMin=r_lo * 0.95, xMax=r_hi * 1.05,
        yMin=top_y * 0.5,  yMax=top_y + mf)
    if len(fine_bb_edges) > 0:
        wafer.seedEdgeBySize(edges=fine_bb_edges, size=mf, constraint=FINER)
    if taiko:
        step_edges = wafer.edges.getByBoundingBox(
            xMin=r_step - bw, xMax=r_step + bw,
            yMin=0.0,          yMax=t + mf)
        if len(step_edges) > 0:
            wafer.seedEdgeBySize(edges=step_edges, size=mf, constraint=FINER)

    # SSD layer: 3 elements through t_ssd thickness (Y-direction short edges)
    if 0.0 < t_ssd < top_y * 0.5:
        ssd_short = [e for e in wafer.edges[:]
                     if (y_ssd - t_ssd * 0.1) < e.pointOn[0][1] < (top_y + mf)
                     and e.getSize() < t_ssd * 2.0]
        if ssd_short:
            wafer.seedEdgeByNumber(edges=ssd_short, number=3)

    wafer.generateMesh()

    # ═══ OUTPUT REQUESTS ══════════════════════════════════════════════════════
    model.fieldOutputRequests["F-Output-1"].setValues(
        variables=("S", "U", "E", "RF"),
        timeInterval=0.5)

    # Track top-surface axial displacement for warpage (Release step)
    top_all_edges = wafer_inst.edges.getByBoundingBox(
        xMin=0.0, xMax=R * 1.01,
        yMin=top_y - 1e-7, yMax=top_y + 1e-7)
    if len(top_all_edges) > 0:
        model.HistoryOutputRequest(
            name="TopDisp", createStepName="Release",
            region=regionToolset.Region(edges=top_all_edges),
            variables=("U2",), numIntervals=20)

    # ═══ WRITE INP ════════════════════════════════════════════════════════════
    _ncpu = p.get("num_cpus", 1)
    job = mdb.Job(
        name=job_name, model=job_name,
        numCpus=_ncpu,
        numDomains=_ncpu,   # Standard: numDomains must equal numCpus
        memory=80, memoryUnits=PERCENTAGE,
        nodalOutputPrecision=FULL)
    job.writeInput(consistencyChecking=OFF)
    print("[INP] Written: %s.inp" % job_name)
    return job_name


def submit_and_wait(job_name, num_cpus=1):
    """Submit pre-written INP and wait for completion."""
    job = mdb.JobFromInputFile(
        name=job_name + "_run",
        inputFileName=job_name + ".inp",
        numCpus=num_cpus, numDomains=num_cpus,
        memory=80, memoryUnits=PERCENTAGE,
        nodalOutputPrecision=FULL)
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    run_odb = job_name + "_run.odb"
    out_odb = job_name + ".odb"
    if os.path.exists(run_odb) and not os.path.exists(out_odb):
        os.rename(run_odb, out_odb)
    print("[OK] Completed: %s" % job_name)
    return job_name


def build_and_submit(p=DEFAULT, job_name="grind_Si_ws4000_f300_d020",
                     inp_only=False):
    build_model(p=p, job_name=job_name)
    if not inp_only:
        submit_and_wait(job_name, num_cpus=p.get("num_cpus", 1))
    return job_name


def parametric_study(material=Si, base_cfg=None, inp_only=False):
    """
    Full parameter sweep.
    inp_only=True: write all INP files without running solver.
                   Also writes submit_all.sh for batch execution.
    """
    tag     = material["name"].replace("-", "").replace(" ", "")
    results = []
    for ws in SWEEP_WHEEL_SPEEDS:
        for fr in SWEEP_FEED_RATES:
            for cd in SWEEP_CUT_DEPTHS:
                p = DEFAULT.copy()
                if base_cfg:
                    for k in ("wafer_diameter_mm", "wafer_thickness_um",
                              "taiko_mode", "taiko_edge_width_mm", "taiko_thin_um",
                              "mesh_global_um", "mesh_fine_um", "t_ssd_um",
                              "E_damage_factor", "num_cpus"):
                        if k in base_cfg:
                            p[k] = base_cfg[k]
                p["material"]         = material
                p["wheel_speed_rpm"]  = ws
                p["feed_rate_um_min"] = fr
                p["cut_depth_um"]     = cd
                name = "grind_%s_ws%04d_f%04d_d%03d" % (tag, ws, fr, cd)
                print("\n[->] %s  (ws=%drpm  f=%dum/min  d=%dum)" % (
                      name, ws, fr, cd))
                build_and_submit(p=p, job_name=name, inp_only=inp_only)
                results.append({
                    "material":         tag,
                    "wheel_speed_rpm":  ws,
                    "feed_rate_um_min": fr,
                    "cut_depth_um":     cd,
                    "job":              name})

    manifest = "jobs_grind_%s.json" % tag
    with open(manifest, "w") as f:
        json.dump(results, f, indent=2)
    print("[OK] Manifest: %s" % manifest)

    if inp_only:
        # Write a shell script to submit all jobs
        sh_path = "submit_all_grind_%s.sh" % tag
        with open(sh_path, "w") as sh:
            sh.write("#!/bin/bash\n# Auto-generated by grinding_warpage_2d.py\n")
            sh.write("# Run: bash %s\n\n" % sh_path)
            for r in results:
                sh.write("abaqus job=%s cpus=1 interactive && \\\n" % r["job"])
                sh.write("  echo '[OK] %s' || echo '[FAIL] %s'\n\n" % (
                          r["job"], r["job"]))
        print("[OK] Submit script: %s  (%d jobs)" % (sh_path, len(results)))

    return results


# ── Entry point ────────────────────────────────────────────────────────────────
_config_path = os.path.join(os.getcwd(), "run_config.json")
_cfg = json.load(open(_config_path)) if os.path.exists(_config_path) else {}

_mat = ALL_MATERIALS.get(_cfg.get("material", "Si"), Si)

if _cfg.get("study", False):
    parametric_study(material=_mat, base_cfg=_cfg,
                     inp_only=bool(_cfg.get("inp_only", False)))
else:
    _p = DEFAULT.copy()
    _p["material"] = _mat
    for _k in ("wheel_speed_rpm", "feed_rate_um_min", "cut_depth_um",
               "wafer_diameter_mm", "wafer_thickness_um",
               "taiko_mode", "taiko_edge_width_mm", "taiko_thin_um",
               "mesh_global_um", "mesh_fine_um", "num_cpus"):
        if _k in _cfg:
            _p[_k] = _cfg[_k]
    _tag  = _mat["name"].replace("-", "").replace(" ", "")
    _name = _cfg.get("job_name",
                     "grind_%s_ws%04d_f%04d_d%03d" % (
                         _tag,
                         int(_p["wheel_speed_rpm"]),
                         int(_p["feed_rate_um_min"]),
                         int(_p["cut_depth_um"])))
    build_and_submit(p=_p, job_name=_name)
