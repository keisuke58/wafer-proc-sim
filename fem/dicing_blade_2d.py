"""
2D plane-strain blade dicing FEM — ABAQUS/Explicit
SiC (or Si) wafer, element deletion via ductile damage (brittle approximation).

Run:
    abaqus cae noGUI=dicing_blade_2d.py
    abaqus cae noGUI=dicing_blade_2d.py -- --study
    abaqus cae noGUI=dicing_blade_2d.py -- --depth 30 --kerf 30

Physics note:
    SiC is brittle (no real plasticity). We approximate brittle fracture by:
    - Elastic-perfectly-plastic with yield stress = fracture stress (21 GPa)
    - DUCTILE damage onset at small eps_eq, energy-based softening → element deletion
    This approach is standard in ceramic machining FEM literature.
"""

import sys
import os
import json
import argparse

# ── Locate materials from ABAQUS sys.argv (-noGUI <script>) ──────────────────
_abq_script = None
for _i, _a in enumerate(sys.argv[:-1]):
    if _a == '-noGUI':
        _abq_script = os.path.abspath(sys.argv[_i + 1])
        break
if _abq_script is None:
    raise RuntimeError("Cannot locate -noGUI script in sys.argv: " + str(sys.argv))
_mat_dir = os.path.normpath(os.path.join(os.path.dirname(_abq_script),
                                          '..', 'data', 'materials'))
sys.path.insert(0, _mat_dir)
from material_properties import SiC, Si

# ── ABAQUS imports (all at module level) ──────────────────────────────────────
from abaqus import mdb, backwardCompatibility
backwardCompatibility.setValues(reportDeprecated=False)
from abaqusConstants import *
from caeModules import *
import regionToolset
from mesh import ElemType

# ── Default parameters ────────────────────────────────────────────────────────
DEFAULT = {
    "material":        SiC,
    "wafer_W_um":      500.0,
    "wafer_H_um":      200.0,
    "cut_depth_um":     30.0,
    "blade_W_um":       30.0,
    "feed_speed_m_s":    0.05,
    "mesh_global_um":    5.0,
    "mesh_fine_um":      2.0,
    "friction_coeff":    0.3,
    "num_cpus":          4,
}

SWEEP_CUT_DEPTHS_UM   = [20, 30, 40, 50, 60]
SWEEP_BLADE_WIDTHS_UM = [20, 30, 40]


# ─────────────────────────────────────────────────────────────────────────────
def build_and_submit(p=DEFAULT, job_name="dicing_sic_d030"):
    W   = p["wafer_W_um"]      * 1e-6
    H   = p["wafer_H_um"]      * 1e-6
    d   = p["cut_depth_um"]    * 1e-6
    bw  = p["blade_W_um"]      * 1e-6
    v   = p["feed_speed_m_s"]
    mg  = p["mesh_global_um"]  * 1e-6
    mf  = p["mesh_fine_um"]    * 1e-6
    mat = p["material"]

    xc        = W / 2
    fine_half = bw * 3
    travel    = W * 0.8
    t_plunge  = d / v
    t_feed    = travel / v
    t_total   = t_plunge + t_feed
    G_c       = mat["K_Ic"] ** 2 / mat["E"]   # fracture energy J/m²
    chamfer   = bw * 0.15

    # ABAQUS names: letters/digits/underscore only, must start with letter
    import re as _re
    def _abq_name(s):
        s = _re.sub(r'[^A-Za-z0-9_]', '_', str(s))
        if s and s[0].isdigit():
            s = 'M_' + s
        return s
    mat_name = _abq_name(mat["name"])

    # ── Model ─────────────────────────────────────────────────────────────────
    if job_name in mdb.models:
        del mdb.models[job_name]
    model = mdb.Model(name=job_name)
    if "Model-1" in mdb.models and job_name != "Model-1":
        del mdb.models["Model-1"]

    # ═══ PART: Wafer ══════════════════════════════════════════════════════════
    sk_w = model.ConstrainedSketch(name="wafer_sk", sheetSize=W * 4)
    sk_w.rectangle(point1=(0.0, 0.0), point2=(W, H))
    wafer = model.Part(name="Wafer", dimensionality=TWO_D_PLANAR,
                       type=DEFORMABLE_BODY)
    wafer.BaseShell(sketch=sk_w)

    # Partition → two vertical lines to isolate fine-mesh zone
    # PartitionFaceByShortestPath is more reliable than sketch-based for 2D planar
    wafer.PartitionFaceByShortestPath(
        point1=(xc - fine_half, 0.0, 0.0),
        point2=(xc - fine_half, H,   0.0),
        faces=wafer.faces[:])
    wafer.PartitionFaceByShortestPath(
        point1=(xc + fine_half, 0.0, 0.0),
        point2=(xc + fine_half, H,   0.0),
        faces=wafer.faces[:])

    # ═══ MATERIAL ═════════════════════════════════════════════════════════════
    m = model.Material(name=mat_name)
    m.Elastic(table=((mat["E"], mat["nu"]),))
    m.Density(table=((mat["density"],),))

    # Brittle fracture via Max Principal Stress criterion
    # MaxpsDamageInitiation: damage onset when sigma_max > sigma_fracture
    # DamageEvolution: energy-based linear softening → element deletion at D=1
    m.MaxpsDamageInitiation(table=((mat["sigma_y"],),))
    m.maxpsDamageInitiation.DamageEvolution(
        type=ENERGY, softening=LINEAR, table=((G_c,),))

    # ═══ SECTION ══════════════════════════════════════════════════════════════
    model.HomogeneousSolidSection(name="WaferSec",
                                   material=mat_name, thickness=1.0e-6)
    wafer_all = wafer.Set(faces=wafer.faces[:], name="WaferAll")
    wafer.SectionAssignment(region=wafer_all, sectionName="WaferSec",
                            offset=0.0, offsetType=MIDDLE_SURFACE,
                            offsetField="",
                            thicknessAssignment=FROM_SECTION)

    # ═══ PART: Blade (discrete rigid) ═════════════════════════════════════════
    half  = bw / 2
    sk_b  = model.ConstrainedSketch(name="blade_sk", sheetSize=W * 4)
    sk_b.Line(point1=(-half,              0.0),
              point2=(-half + chamfer, -chamfer))
    sk_b.Line(point1=(-half + chamfer, -chamfer),
              point2=( half - chamfer, -chamfer))
    sk_b.Line(point1=( half - chamfer, -chamfer),
              point2=( half,              0.0))

    blade = model.Part(name="Blade", dimensionality=TWO_D_PLANAR,
                       type=DISCRETE_RIGID_SURFACE)
    blade.BaseWire(sketch=sk_b)
    # DISCRETE_RIGID_SURFACE parts do not need a section assignment in CAE
    blade.ReferencePoint(point=(0.0, 0.0, 0.0))
    blade_rp_key = list(blade.referencePoints.keys())[-1]

    # ═══ ASSEMBLY ═════════════════════════════════════════════════════════════
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    wafer_inst = a.Instance(name="Wafer-1", part=wafer, dependent=ON)
    blade_inst = a.Instance(name="Blade-1", part=blade, dependent=ON)
    # Start blade at left edge, above wafer
    a.translate(instanceList=("Blade-1",), vector=(bw, H + chamfer, 0.0))

    # ═══ STEPS ════════════════════════════════════════════════════════════════
    ms_table = ((SEMI_AUTOMATIC, MODEL, AT_BEGINNING,
                 0.0, 0.0, BELOW_MIN, 1, 0, 0.0, 0.0, 0, None),)

    model.ExplicitDynamicsStep(
        name="Plunge", previous="Initial",
        timePeriod=t_plunge, massScaling=ms_table,
        improvedDtMethod=ON)

    model.ExplicitDynamicsStep(
        name="Feed", previous="Plunge",
        timePeriod=t_feed, massScaling=ms_table,
        improvedDtMethod=ON)

    # ═══ BOUNDARY CONDITIONS ══════════════════════════════════════════════════
    bot_edge = wafer_inst.edges.findAt(((W/2, 0.0, 0.0),))
    model.EncastreBC(name="WaferBottom", createStepName="Initial",
                     region=regionToolset.Region(edges=bot_edge))

    lft_edge = wafer_inst.edges.findAt(((0.0, H/2, 0.0),))
    rgt_edge = wafer_inst.edges.findAt(((W,   H/2, 0.0),))
    model.DisplacementBC(name="WaferLeft",  createStepName="Initial",
                         region=regionToolset.Region(edges=lft_edge), u1=0.0)
    model.DisplacementBC(name="WaferRight", createStepName="Initial",
                         region=regionToolset.Region(edges=rgt_edge), u1=0.0)

    blade_rp_inst = a.instances["Blade-1"].referencePoints[blade_rp_key]
    blade_region  = regionToolset.Region(referencePoints=(blade_rp_inst,))

    # Fix rotation throughout
    model.DisplacementBC(name="BladeRot", createStepName="Initial",
                         region=blade_region, ur3=0.0)

    # Plunge: move down by cut depth, hold horizontal
    model.DisplacementBC(name="BladeMotion", createStepName="Plunge",
                         region=blade_region, u1=0.0, u2=-d)

    # Feed: modify same BC → horizontal travel, maintain depth
    # setValuesInStep updates BC in Feed step without creating a new one
    model.boundaryConditions["BladeMotion"].setValuesInStep(
        stepName="Feed", u1=travel, u2=-d)

    # ═══ CONTACT ══════════════════════════════════════════════════════════════
    top_edge = wafer_inst.edges.findAt(((W/2, H, 0.0),))
    a.Surface(side1Edges=(top_edge,), name="WaferTop")
    a.Surface(side1Edges=blade_inst.edges[:], name="BladeSurf")

    model.ContactProperty("FricContact")
    model.interactionProperties["FricContact"].TangentialBehavior(
        formulation=PENALTY, directionality=ISOTROPIC,
        slipRateDependency=OFF, pressureDependency=OFF,
        temperatureDependency=OFF, dependencies=0,
        table=((p["friction_coeff"],),),
        shearStressLimit=None,
        maximumElasticSlip=FRACTION, fraction=0.005,
        elasticSlipStiffness=None)
    model.interactionProperties["FricContact"].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON)

    model.SurfaceToSurfaceContactExp(
        name="BladeCut", createStepName="Plunge",
        main=a.surfaces["BladeSurf"],
        secondary=a.surfaces["WaferTop"],
        sliding=FINITE,
        interactionProperty="FricContact")

    # ═══ MESH ═════════════════════════════════════════════════════════════════
    elem_quad = ElemType(elemCode=CPE4R, elemLibrary=EXPLICIT,
                         secondOrderAccuracy=OFF, hourglassControl=ENHANCED,
                         distortionControl=ON)
    elem_tri  = ElemType(elemCode=CPE3, elemLibrary=EXPLICIT)
    wafer.setElementType(
        regions=regionToolset.Region(faces=wafer.faces[:]),
        elemTypes=(elem_quad, elem_tri))
    wafer.seedPart(size=mg, deviationFactor=0.1, minSizeFactor=0.1)

    fine_edges = wafer.edges.getByBoundingBox(
        xMin=xc - fine_half * 1.05, xMax=xc + fine_half * 1.05,
        yMin=H - d * 5,             yMax=H + mf)
    if len(fine_edges) > 0:
        wafer.seedEdgeBySize(edges=fine_edges, size=mf, constraint=FINER)
    wafer.generateMesh()

    blade_r2d2 = ElemType(elemCode=R2D2, elemLibrary=EXPLICIT)
    blade.setElementType(
        regions=regionToolset.Region(edges=blade.edges[:]),
        elemTypes=(blade_r2d2,))
    blade.seedPart(size=mf)
    blade.generateMesh()

    # ═══ OUTPUT ═══════════════════════════════════════════════════════════════
    model.fieldOutputRequests["F-Output-1"].setValues(
        variables=("S", "U", "STATUS", "PEEQ", "ENER"),
        timeInterval=t_total / 50)
    model.HistoryOutputRequest(
        name="BladeForce", createStepName="Plunge",
        region=blade_region,
        variables=("RF1", "RF2", "U1", "U2"),
        numIntervals=200)

    # ═══ SAVE + SUBMIT ════════════════════════════════════════════════════════
    mdb.saveAs(pathName=job_name + ".cae")
    print("[Model saved] %s.cae" % job_name)

    job = mdb.Job(
        name=job_name, model=job_name,
        numCpus=p.get("num_cpus", 4), numDomains=p.get("num_cpus", 4),
        memory=80, memoryUnits=PERCENTAGE,
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE)
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    print("[OK] Job completed: %s" % job_name)
    return job_name


# ─────────────────────────────────────────────────────────────────────────────
def parametric_study(material=SiC, base_cfg=None):
    """Run full sweep; base_cfg overrides DEFAULT for shared params (num_cpus etc.)."""
    tag     = material["name"].replace("-", "").replace(" ", "")
    results = []
    for d in SWEEP_CUT_DEPTHS_UM:
        for bw in SWEEP_BLADE_WIDTHS_UM:
            p = DEFAULT.copy()
            if base_cfg:   # apply shared overrides from run_config.json
                for k in ("feed_speed_m_s", "mesh_global_um",
                          "mesh_fine_um", "friction_coeff", "num_cpus"):
                    if k in base_cfg:
                        p[k] = base_cfg[k]
            p["material"]     = material
            p["cut_depth_um"] = d
            p["blade_W_um"]   = bw
            name = "dicing_%s_d%03d_bw%02d" % (tag, d, bw)
            print("[->] %s  (cpus=%d, v=%.1fm/s)" % (
                name, p["num_cpus"], p["feed_speed_m_s"]))
            build_and_submit(p=p, job_name=name)
            results.append({"material": tag, "cut_depth_um": d,
                            "blade_W_um": bw, "job": name})
    with open("jobs_%s.json" % tag, "w") as f:
        json.dump(results, f, indent=2)
    print("[OK] Manifest saved: jobs_%s.json" % tag)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ABAQUS drops args after noGUI=script.py, so we use run_config.json instead.
# Create run_config.json in cwd before running:
#   {"cut_depth_um": 30, "blade_W_um": 30, "study": false}
# ─────────────────────────────────────────────────────────────────────────────
_config_path = os.path.join(os.getcwd(), "run_config.json")
if os.path.exists(_config_path):
    with open(_config_path) as _f:
        _cfg = json.load(_f)
else:
    _cfg = {}

from material_properties import ALL_MATERIALS
_mat_name = _cfg.get("material", "SiC")
_mat      = ALL_MATERIALS.get(_mat_name, SiC)

if _cfg.get("study", False):
    parametric_study(material=_mat, base_cfg=_cfg)
else:
    _p = DEFAULT.copy()
    _p["material"] = _mat
    for _k in ("cut_depth_um", "blade_W_um", "feed_speed_m_s",
                "mesh_global_um", "mesh_fine_um", "friction_coeff", "num_cpus"):
        if _k in _cfg:
            _p[_k] = _cfg[_k]
    _tag  = _mat["name"].replace("-", "").replace(" ", "")
    _name = _cfg.get("job_name",
                     "dicing_%s_d%03d_bw%03d" % (
                         _tag, int(_p["cut_depth_um"]), int(_p["blade_W_um"])))
    build_and_submit(p=_p, job_name=_name)
