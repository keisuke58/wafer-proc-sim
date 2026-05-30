"""
3D grinding FEM — ABAQUS/Standard (warpage validation model)

Geometry : quarter-wafer (two symmetry planes) — X=[0,R], Y=[0,t], Z=[0,R]
Symmetry : X=0 face (u1=0), Z=0 face (u3=0)
Load     : grinding pressure strip on top face Y=t, X∈[r_lo,r_hi], all Z
Steps    : Grind (chuck clamped, Y=0 fixed) → Release (chuck off → warpage)
TAIKO    : L-shaped cross-section sketched in XY, extruded in Z
           (rectangular approximation of circular edge ring — valid for trend validation)

Output   : used with extract_grinding_results.py
           warpage_um              peak-to-valley U2 on Y=t surface after Release
           max_residual_stress_MPa max |S22| after Release step

Difference from 2D axisymmetric (grinding_warpage_2d.py):
  - Captures asymmetric warpage modes (off-center grinding pass)
  - TAIKO 3D rectangular ring vs. circular ring comparison
  - ~3–5 × slower than 2D; use for spot-check at GP-optimal points

Run:
    abaqus cae noGUI=grinding_warpage_3d.py   (reads run_config_grinding_3d.json in cwd)

run_config_grinding_3d.json keys: same as 2D + "mesh_thick_n" (elements through thickness)
"""

import sys
import os
import json
import math

# ── Script path resolution ────────────────────────────────────────────────────
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

DEFAULT_3D = {
    "material":            Si,
    "wheel_speed_rpm":     4000.0,
    "feed_rate_um_min":    300.0,
    "cut_depth_um":        20.0,
    "wafer_radius_mm":     150.0,    # full-wafer radius; model = quarter (R × t × R)
    "wafer_thickness_um":  500.0,
    "wheel_diameter_mm":   200.0,
    "wheel_width_mm":      5.0,      # wheel contact width in X [mm]
    "grinding_mu":         0.35,
    "taiko_mode":          False,
    "taiko_edge_width_mm": 3.0,
    "taiko_thin_um":       50.0,
    "mesh_global_um":      5000.0,   # 5 mm global seed (radial / Z)
    "mesh_thick_n":        4,        # element count through wafer thickness
    "mesh_fine_um":        500.0,    # 0.5 mm seed in grinding zone
    "t_ssd_um":            10.0,     # subsurface damage layer thickness [µm]
    "E_damage_factor":      0.5,     # E_damaged / E_bulk (microcracks; lit: 0.3–0.7)
    "num_cpus":            4,
}

SWEEP_WHEEL_SPEEDS = [2000, 4000, 8000]    # rpm   (reduced subset for 3D)
SWEEP_FEED_RATES   = [200,  500]           # µm/min
SWEEP_CUT_DEPTHS   = [10,   30]            # µm


# ── Shared force model (identical to 2D) ─────────────────────────────────────
def _grinding_force(p):
    v_s = math.pi * p["wheel_diameter_mm"] * 1e-3 * p["wheel_speed_rpm"] / 60.0
    v_f = p["feed_rate_um_min"] * 1e-6 / 60.0
    a_p = p["cut_depth_um"] * 1e-6
    C_t = p["material"].get("C_t_grinding", 1800.0)
    Ft_pw = C_t * ((v_f / v_s) ** 0.6) * (a_p ** 0.4)
    Fn_pw = Ft_pw / p["grinding_mu"]
    l_c   = math.sqrt(a_p * p["wheel_diameter_mm"] * 1e-3)
    return Fn_pw, Ft_pw, l_c


def _abq_name(s):
    import re
    s = re.sub(r'[^A-Za-z0-9_]', '_', str(s))
    if s and s[0].isdigit():
        s = 'M_' + s
    return s


def build_model_3d(p=DEFAULT_3D, job_name="grind3d_Si_ws4000_f300_d020"):
    """Build ABAQUS/Standard 3D quarter-wafer grinding model. Returns job_name."""

    R   = p["wafer_radius_mm"] * 1e-3      # quarter-wafer X/Z extent [m]
    t   = p["wafer_thickness_um"] * 1e-6   # full thickness [m]
    Lz  = R                                 # Z extent = R (quarter symmetry)
    mat = p["material"]
    mat_name = _abq_name(mat["name"])

    taiko  = bool(p.get("taiko_mode", False))
    t_thin = p["taiko_thin_um"] * 1e-6 if taiko else t
    w_edge = p["taiko_edge_width_mm"] * 1e-3 if taiko else 0.0
    r_step = R - w_edge

    mg    = p["mesh_global_um"] * 1e-6
    mf    = p["mesh_fine_um"]   * 1e-6
    n_tck = max(2, int(p.get("mesh_thick_n", 4)))

    # Grinding zone in X: centered at R/2, width = wheel_width_mm
    bw    = p["wheel_width_mm"] * 1e-3
    r_ctr = R / 2.0
    r_lo  = max(0.0,  r_ctr - bw / 2.0)
    r_hi  = min(R,    r_ctr + bw / 2.0)

    Fn_pw, _Ft_pw, l_c = _grinding_force(p)
    p_grind = Fn_pw / l_c

    # ── Create / reset model ──────────────────────────────────────────────────
    if job_name in mdb.models:
        del mdb.models[job_name]
    model = mdb.Model(name=job_name)
    if "Model-1" in mdb.models and job_name != "Model-1":
        del mdb.models["Model-1"]

    # ═══ PART: Wafer (3D solid) ════════════════════════════════════════════════
    # Sketch the XY cross-section, extrude in Z = Lz.
    # Non-TAIKO : rectangle (0,0)→(R,t)
    # TAIKO     : L-shape (inner thin + outer full ring, rectangular approx)
    sk = model.ConstrainedSketch(name="wafer_sk", sheetSize=R * 3)

    if taiko:
        # L-shaped outline (CW from origin so extrusion gives outward +Z normal)
        sk.Line(point1=(0.0,    0.0),    point2=(R,      0.0))
        sk.Line(point1=(R,      0.0),    point2=(R,      t))
        sk.Line(point1=(R,      t),      point2=(r_step, t))
        sk.Line(point1=(r_step, t),      point2=(r_step, t_thin))
        sk.Line(point1=(r_step, t_thin), point2=(0.0,    t_thin))
        sk.Line(point1=(0.0,    t_thin), point2=(0.0,    0.0))
    else:
        sk.rectangle(point1=(0.0, 0.0), point2=(R, t))

    wafer = model.Part(name="Wafer", dimensionality=THREE_D,
                       type=DEFORMABLE_BODY)
    wafer.BaseSolidExtrude(sketch=sk, depth=Lz)

    # ── Partitions: datum planes at r_lo, r_hi (+ r_step for TAIKO) ────────────
    part_xs = [r_lo, r_hi]
    if taiko:
        part_xs.append(r_step)
    for xval in part_xs:
        if 0.0 < xval < R:
            dp = wafer.DatumPlaneByPrincipalPlane(
                principalPlane=YZPLANE, offset=xval)
            wafer.PartitionCellByDatumPlane(
                datumPlane=wafer.datums[dp.id],
                cells=wafer.cells[:])

    # ── SSD partition: horizontal datum plane at Y = top_y - t_ssd ───────────
    t_ssd  = p.get("t_ssd_um", 10.0) * 1e-6
    E_dmg  = p.get("E_damage_factor", 0.5)
    top_y  = t_thin if taiko else t
    y_ssd  = top_y - t_ssd

    if 0.0 < t_ssd < top_y * 0.5:
        dp_ssd = wafer.DatumPlaneByPrincipalPlane(
            principalPlane=XZPLANE, offset=y_ssd)
        wafer.PartitionCellByDatumPlane(
            datumPlane=wafer.datums[dp_ssd.id],
            cells=wafer.cells[:])

    # ═══ MATERIALS ════════════════════════════════════════════════════════════
    m_bulk = model.Material(name=mat_name)
    m_bulk.Elastic(table=((mat["E"], mat["nu"]),))
    m_bulk.Density(table=((mat["density"],),))

    mat_dam = mat_name + "_dam"
    m_dam   = model.Material(name=mat_dam)
    m_dam.Elastic(table=((mat["E"] * E_dmg, mat["nu"]),))
    m_dam.Density(table=((mat["density"],),))

    model.HomogeneousSolidSection(name="SecBulk", material=mat_name)
    model.HomogeneousSolidSection(name="SecDam",  material=mat_dam)

    # Classify cells: pointOn Y > y_ssd → damaged; else → bulk
    all_c   = wafer.cells[:]
    dam_c   = tuple(c for c in all_c if c.pointOn[0][1] > y_ssd)
    bulk_c  = tuple(c for c in all_c if c.pointOn[0][1] <= y_ssd)

    if dam_c:
        wafer.SectionAssignment(
            region=regionToolset.Region(cells=dam_c),
            sectionName="SecDam",
            offset=0.0, offsetType=MIDDLE_SURFACE,
            offsetField="", thicknessAssignment=FROM_SECTION)
    wafer.SectionAssignment(
        region=regionToolset.Region(cells=bulk_c if bulk_c else wafer.cells[:]),
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
    eps = 1e-7

    # Vacuum chuck: Y=0 face, fixed in Y (u2=0)
    chuck_faces = wafer_inst.faces.getByBoundingBox(
        xMin=0.0, xMax=R + eps, yMin=-eps, yMax=eps, zMin=0.0, zMax=Lz + eps)
    if len(chuck_faces) == 0:
        chuck_faces = wafer_inst.faces.findAt(((R / 4, 0.0, Lz / 4),))
    model.DisplacementBC(name="ChuckBC", createStepName="Initial",
                         region=regionToolset.Region(faces=chuck_faces),
                         u2=0.0)
    # Release chuck in Release step → wafer warps
    model.boundaryConditions["ChuckBC"].deactivate("Release")

    # Symmetry plane X=0 (u1=0) — prevents rigid body translation in X
    xsym_faces = wafer_inst.faces.getByBoundingBox(
        xMin=-eps, xMax=eps, yMin=0.0, yMax=t + eps, zMin=0.0, zMax=Lz + eps)
    if len(xsym_faces) == 0:
        t_mid = t_thin / 2.0 if taiko else t / 2.0
        xsym_faces = wafer_inst.faces.findAt(((0.0, t_mid, Lz / 4),))
    model.DisplacementBC(name="SymX", createStepName="Initial",
                         region=regionToolset.Region(faces=xsym_faces),
                         u1=0.0)

    # Symmetry plane Z=0 (u3=0)
    zsym_faces = wafer_inst.faces.getByBoundingBox(
        xMin=0.0, xMax=R + eps, yMin=0.0, yMax=t + eps, zMin=-eps, zMax=eps)
    if len(zsym_faces) == 0:
        zsym_faces = wafer_inst.faces.findAt(((R / 4, t / 2.0, 0.0),))
    model.DisplacementBC(name="SymZ", createStepName="Initial",
                         region=regionToolset.Region(faces=zsym_faces),
                         u3=0.0)

    # Pin origin corner to suppress residual rigid-body Y after chuck release
    # (one node on X=0, Y=0, Z=0 corner pinned in all directions)
    origin_verts = wafer_inst.vertices.findAt(((0.0, 0.0, 0.0),))
    if len(origin_verts) > 0:
        model.DisplacementBC(name="OriginPin", createStepName="Initial",
                             region=regionToolset.Region(vertices=origin_verts),
                             u1=0.0, u2=0.0, u3=0.0)

    # ═══ LOAD: grinding pressure on Y=t top face, X∈[r_lo, r_hi] ═══════════
    top_y = t_thin if taiko else t
    grind_faces = wafer_inst.faces.getByBoundingBox(
        xMin=r_lo - eps, xMax=r_hi + eps,
        yMin=top_y - eps, yMax=top_y + eps,
        zMin=0.0,          zMax=Lz + eps)
    if len(grind_faces) > 0:
        # side1Faces: positive pressure = compressive (into +Y direction) ✓
        model.Pressure(
            name="GrindPress", createStepName="Grind",
            region=regionToolset.Region(side1Faces=grind_faces),
            magnitude=p_grind)
        print("[INP] Grinding pressure: %.2f MPa  X=[%.1f, %.1f] mm" % (
              p_grind / 1e6, r_lo * 1e3, r_hi * 1e3))
    else:
        print("[WARN] No top faces found in grind zone [%.2f, %.2f] mm — "
              "check r_lo/r_hi vs partitions" % (r_lo * 1e3, r_hi * 1e3))

    # ═══ MESH (C3D8R — hex with reduced integration) ══════════════════════════
    elem_hex = ElemType(elemCode=C3D8R, elemLibrary=STANDARD,
                        secondOrderAccuracy=OFF, hourglassControl=ENHANCED)
    elem_tet = ElemType(elemCode=C3D4,  elemLibrary=STANDARD)
    wafer.setElementType(
        regions=regionToolset.Region(cells=wafer.cells[:]),
        elemTypes=(elem_hex, elem_tet))

    # Global radial / Z seed
    wafer.seedPart(size=mg, deviationFactor=0.1, minSizeFactor=0.1)

    # Thickness (Y) direction: fixed element count regardless of global seed
    # Y-edges: run from (x, 0, z) to (x, t, z)
    y_edges = wafer.edges.getByBoundingBox(
        xMin=0.0, xMax=R + eps,
        yMin=0.0, yMax=t + eps,
        zMin=0.0, zMax=Lz + eps)
    y_only = [e for e in y_edges
              if abs(e.pointOn[0][0] - e.pointOn[0][0]) < eps   # same X
              and abs(e.pointOn[0][2] - e.pointOn[0][2]) < eps  # same Z
              and e.getSize() < t * 1.5]                        # length ~ t
    if len(y_only) > 0:
        wafer.seedEdgeByNumber(edges=y_only, number=n_tck)
    else:
        # Fallback: seed all short edges (length < 2t) by number
        all_edges = wafer.edges[:]
        short = [e for e in all_edges if e.getSize() < t * 2.0]
        if short:
            wafer.seedEdgeByNumber(edges=short, number=n_tck)

    # Finer seed in grinding zone (X direction near r_lo/r_hi)
    fine_edges = wafer.edges.getByBoundingBox(
        xMin=r_lo - bw, xMax=r_hi + bw,
        yMin=0.0,        yMax=top_y + mf,
        zMin=0.0,        zMax=Lz + eps)
    if len(fine_edges) > 0:
        wafer.seedEdgeBySize(edges=fine_edges, size=mf, constraint=FINER)

    wafer.generateMesh()

    n_elem = len(wafer.elements)
    print("[3D] Wafer elements: %d" % n_elem)

    # ═══ OUTPUT REQUESTS ══════════════════════════════════════════════════════
    model.fieldOutputRequests["F-Output-1"].setValues(
        variables=("S", "U", "E", "RF"),
        timeInterval=0.5)

    # Track top-surface nodal U2 for warpage map (Release step)
    top_all_faces = wafer_inst.faces.getByBoundingBox(
        xMin=0.0, xMax=R + eps,
        yMin=top_y - eps, yMax=top_y + eps,
        zMin=0.0, zMax=Lz + eps)
    if len(top_all_faces) > 0:
        model.HistoryOutputRequest(
            name="TopDisp3D", createStepName="Release",
            region=regionToolset.Region(faces=top_all_faces),
            variables=("U2",), numIntervals=10)

    # ═══ WRITE INP ════════════════════════════════════════════════════════════
    job = mdb.Job(
        name=job_name, model=job_name,
        numCpus=p.get("num_cpus", 1),
        numDomains=1,
        memory=80, memoryUnits=PERCENTAGE,
        nodalOutputPrecision=FULL)
    job.writeInput(consistencyChecking=OFF)
    print("[INP] Written: %s.inp" % job_name)
    return job_name


def submit_and_wait_3d(job_name, num_cpus=1):
    job = mdb.JobFromInputFile(
        name=job_name + "_run",
        inputFileName=job_name + ".inp",
        numCpus=num_cpus, numDomains=1,
        memory=80, memoryUnits=PERCENTAGE,
        nodalOutputPrecision=FULL)
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    run_odb = job_name + "_run.odb"
    out_odb = job_name + ".odb"
    if os.path.exists(run_odb) and not os.path.exists(out_odb):
        os.rename(run_odb, out_odb)
    print("[OK] 3D job completed: %s" % job_name)
    return job_name


def build_and_submit_3d(p=DEFAULT_3D, job_name="grind3d_Si_ws4000_f300_d020"):
    build_model_3d(p=p, job_name=job_name)
    submit_and_wait_3d(job_name, num_cpus=p.get("num_cpus", 1))
    return job_name


def parametric_study_3d(material=Si, base_cfg=None):
    """Spot-check sweep — smaller than 2D (3D is 3–5× slower)."""
    tag     = material["name"].replace("-", "").replace(" ", "")
    results = []
    for ws in SWEEP_WHEEL_SPEEDS:
        for fr in SWEEP_FEED_RATES:
            for cd in SWEEP_CUT_DEPTHS:
                p = DEFAULT_3D.copy()
                if base_cfg:
                    for k in ("wafer_radius_mm", "wafer_thickness_um",
                              "taiko_mode", "taiko_edge_width_mm", "taiko_thin_um",
                              "mesh_global_um", "mesh_fine_um",
                              "mesh_thick_n", "num_cpus"):
                        if k in base_cfg:
                            p[k] = base_cfg[k]
                p["material"]         = material
                p["wheel_speed_rpm"]  = ws
                p["feed_rate_um_min"] = fr
                p["cut_depth_um"]     = cd
                name = "grind3d_%s_ws%04d_f%04d_d%03d" % (tag, ws, fr, cd)
                print("\n[->] %s  (ws=%drpm  f=%dum/min  d=%dum)" % (
                      name, ws, fr, cd))
                build_and_submit_3d(p=p, job_name=name)
                results.append({
                    "material":         tag,
                    "wheel_speed_rpm":  ws,
                    "feed_rate_um_min": fr,
                    "cut_depth_um":     cd,
                    "job":              name})

    manifest = "jobs_grind3d_%s.json" % tag
    with open(manifest, "w") as f:
        json.dump(results, f, indent=2)
    print("[OK] Manifest: %s" % manifest)
    return results


# ── Entry point ────────────────────────────────────────────────────────────────
_cfg_path = os.path.join(os.getcwd(), "run_config_grinding_3d.json")
_cfg = json.load(open(_cfg_path)) if os.path.exists(_cfg_path) else {}

_mat = ALL_MATERIALS.get(_cfg.get("material", "Si"), Si)

if _cfg.get("study", False):
    parametric_study_3d(material=_mat, base_cfg=_cfg)
else:
    _p = DEFAULT_3D.copy()
    _p["material"] = _mat
    for _k in ("wheel_speed_rpm", "feed_rate_um_min", "cut_depth_um",
               "wafer_radius_mm", "wafer_thickness_um",
               "taiko_mode", "taiko_edge_width_mm", "taiko_thin_um",
               "mesh_global_um", "mesh_fine_um", "mesh_thick_n", "num_cpus"):
        if _k in _cfg:
            _p[_k] = _cfg[_k]
    _tag  = _mat["name"].replace("-", "").replace(" ", "")
    _name = _cfg.get("job_name",
                     "grind3d_%s_ws%04d_f%04d_d%03d" % (
                         _tag,
                         int(_p["wheel_speed_rpm"]),
                         int(_p["feed_rate_um_min"]),
                         int(_p["cut_depth_um"])))
    build_and_submit_3d(p=_p, job_name=_name)
