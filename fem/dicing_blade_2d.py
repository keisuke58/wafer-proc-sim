"""
2D plane-strain blade dicing FEM — ABAQUS/Explicit (Accuracy v2)
SiC/Si/GaN wafer, physically-based brittle fracture model.

Accuracy improvements over v1:
  Material : Drucker-Prager pressure-dependent plasticity (ceramics standard)
             Triaxiality-dependent fracture strain table
             Correct strength values: σ_t=350MPa, σ_c=3GPa (not hardness 21GPa)
  Dynamics : Target DT=1e-8s injected into INP → stable quasi-static solution
             v=0.5 m/s (realistic feed), double precision solver
  Fracture : Energy-based softening calibrated from K_Ic (G_c=17.5 J/m² for SiC)
  Contact  : Coulomb friction µ=0.3, hard normal contact

Run:
    abaqus cae noGUI=dicing_blade_2d.py   (reads run_config.json in cwd)

run_config.json keys:
    material, cut_depth_um, blade_W_um, feed_speed_m_s,
    mesh_global_um, mesh_fine_um, num_cpus, study, job_name
"""

import sys
import os
import json
import re

# ── Script path resolution (ABAQUS noGUI: __file__ undefined) ────────────────
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
from material_properties import SiC, Si, GaN, ALL_MATERIALS

# ── ABAQUS module imports (must be at module level) ───────────────────────────
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
    "feed_speed_m_s":    0.5,    # realistic; mass scaling handles quasi-statics
    "mesh_global_um":    5.0,
    "mesh_fine_um":      1.5,    # finer than v1 for better fracture resolution
    "target_dt_s":       1e-8,   # injected into INP mass scaling keyword
    "friction_coeff":    0.3,
    "num_cpus":          1,
}

SWEEP_CUT_DEPTHS_UM   = [20, 30, 40, 50, 60]
SWEEP_BLADE_WIDTHS_UM = [20, 30, 40]


# ─────────────────────────────────────────────────────────────────────────────
def _abq_name(s):
    """Make string a valid ABAQUS name (letters/digits/underscore, start letter)."""
    s = re.sub(r'[^A-Za-z0-9_]', '_', str(s))
    if s and s[0].isdigit():
        s = 'M_' + s
    return s


def build_model(p=DEFAULT, job_name="dicing_sic_d030"):
    """Build ABAQUS CAE model and write INP. Returns job_name."""

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
    chamfer   = bw * 0.15
    mat_name  = _abq_name(mat["name"])

    # ── Create / reset model ──────────────────────────────────────────────────
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

    # Vertical partitions isolating fine-mesh zone around cut center
    wafer.PartitionFaceByShortestPath(
        point1=(xc - fine_half, 0.0, 0.0),
        point2=(xc - fine_half, H,   0.0),
        faces=wafer.faces[:])
    wafer.PartitionFaceByShortestPath(
        point1=(xc + fine_half, 0.0, 0.0),
        point2=(xc + fine_half, H,   0.0),
        faces=wafer.faces[:])

    # ═══ MATERIAL (Drucker-Prager + triaxiality-dependent fracture) ═══════════
    m = model.Material(name=mat_name)
    m.Elastic(table=((mat["E"], mat["nu"]),))
    m.Density(table=((mat["density"],),))

    # Drucker-Prager pressure-dependent plasticity (ceramics standard)
    # β = friction angle, K_dp = yield ratio, ψ = dilation angle
    m.DruckerPrager(
        shearCriterion=LINEAR,
        eccentricity=0.1,
        testData=OFF,
        temperatureDependency=OFF,
        dependencies=0,
        table=((mat["dp_friction_angle"],
                mat["K_dp"],
                mat["dp_dilation_angle"]),))

    # Perfectly-plastic hardening: yield stress = dp_cohesion at eps_p = 0
    # DruckerPragerHardening is a child of druckerPrager (lowercase attribute)
    m.druckerPrager.DruckerPragerHardening(
        type=SHEAR,
        temperatureDependency=OFF,
        dependencies=0,
        table=((mat["dp_cohesion_Pa"], 0.0),))

    # Triaxiality-dependent ductile damage initiation
    # fracture_table: ((eps_f, triaxiality, strain_rate), ...)
    m.DuctileDamageInitiation(
        table=mat["fracture_table"],
        temperatureDependency=OFF,
        dependencies=0)

    # Energy-based linear softening calibrated from K_Ic
    m.ductileDamageInitiation.DamageEvolution(
        type=ENERGY,
        softening=LINEAR,
        table=((mat["G_c"],),))

    # ═══ SECTION ══════════════════════════════════════════════════════════════
    model.HomogeneousSolidSection(
        name="WaferSec", material=mat_name, thickness=1.0e-6)
    wafer_all = wafer.Set(faces=wafer.faces[:], name="WaferAll")
    wafer.SectionAssignment(
        region=wafer_all, sectionName="WaferSec",
        offset=0.0, offsetType=MIDDLE_SURFACE,
        offsetField="", thicknessAssignment=FROM_SECTION)

    # ═══ PART: Blade (discrete rigid, trapezoidal tip) ════════════════════════
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
    blade.ReferencePoint(point=(0.0, 0.0, 0.0))
    blade_rp_key = list(blade.referencePoints.keys())[-1]

    # ═══ ASSEMBLY ═════════════════════════════════════════════════════════════
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    wafer_inst = a.Instance(name="Wafer-1", part=wafer, dependent=ON)
    blade_inst = a.Instance(name="Blade-1", part=blade, dependent=ON)
    a.translate(instanceList=("Blade-1",), vector=(bw, H + chamfer, 0.0))

    # ═══ STEPS (mass scaling table placeholder — DT injected into INP later) ══
    # Use SEMI_AUTOMATIC, MODEL, AT_BEGINNING as placeholder
    # The actual DT is added by _inject_mass_scaling_dt()
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
    model.DisplacementBC(name="BladeRot", createStepName="Initial",
                         region=blade_region, ur3=0.0)
    model.DisplacementBC(name="BladeMotion", createStepName="Plunge",
                         region=blade_region, u1=0.0, u2=-d)
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
        sliding=FINITE, interactionProperty="FricContact")

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
        yMin=H - d * 6,             yMax=H + mf)
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
        variables=("S", "U", "STATUS", "PEEQ", "ENER", "SDV"),
        timeInterval=(t_plunge + t_feed) / 50)
    model.HistoryOutputRequest(
        name="BladeForce", createStepName="Plunge",
        region=blade_region,
        variables=("RF1", "RF2", "U1", "U2"),
        numIntervals=300)

    # ═══ WRITE INP (do NOT submit yet) ════════════════════════════════════════
    job = mdb.Job(
        name=job_name, model=job_name,
        numCpus=p.get("num_cpus", 1),
        numDomains=p.get("num_cpus", 1),
        memory=80, memoryUnits=PERCENTAGE,
        explicitPrecision=DOUBLE,          # double precision (important!)
        nodalOutputPrecision=FULL)
    job.writeInput(consistencyChecking=OFF)
    print("[INP] Written: %s.inp" % job_name)
    return job_name


def _inject_mass_scaling_dt(inp_path, target_dt):
    """
    Replace *Fixed Mass Scaling with DT-specified version in the INP file.
    This is the reliable way to set target time increment in ABAQUS Python.
    """
    with open(inp_path, "r") as f:
        content = f.read()

    # Replace all occurrences of *Fixed Mass Scaling (no DT) with DT version
    pattern  = r'\*Fixed Mass Scaling\s*\n'
    replacement = '*Fixed Mass Scaling, dt=%.2e, type=BELOW MIN\n' % target_dt
    new_content, n = re.subn(pattern, replacement, content)

    if n == 0:
        print("[WARN] No '*Fixed Mass Scaling' found in INP — mass scaling not set")
    else:
        with open(inp_path, "w") as f:
            f.write(new_content)
        print("[INP] Injected DT=%.2e into %d mass scaling block(s)" % (target_dt, n))
    return n


def submit_and_wait(job_name, num_cpus=1):
    """Submit pre-written INP file and wait for completion."""
    job = mdb.JobFromInputFile(
        name=job_name + "_run",
        inputFileName=job_name + ".inp",
        numCpus=num_cpus, numDomains=num_cpus,
        memory=80, memoryUnits=PERCENTAGE,
        explicitPrecision=DOUBLE,
        nodalOutputPrecision=FULL)
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()

    # Rename ODB to match job_name (JobFromInputFile appends _run)
    run_odb = job_name + "_run.odb"
    out_odb = job_name + ".odb"
    if os.path.exists(run_odb) and not os.path.exists(out_odb):
        os.rename(run_odb, out_odb)
    print("[OK] Completed: %s" % job_name)
    return job_name


def build_and_submit(p=DEFAULT, job_name="dicing_sic_d030"):
    """Build model, inject mass scaling DT into INP, submit, wait."""
    build_model(p=p, job_name=job_name)

    inp_path = os.path.join(os.getcwd(), job_name + ".inp")
    _inject_mass_scaling_dt(inp_path, p.get("target_dt_s", 1e-8))

    submit_and_wait(job_name, num_cpus=p.get("num_cpus", 1))
    return job_name


# ─────────────────────────────────────────────────────────────────────────────
def parametric_study(material=SiC, base_cfg=None):
    """Run full sweep; base_cfg overrides DEFAULT for shared params."""
    tag     = material["name"].replace("-", "").replace(" ", "")
    results = []
    for d in SWEEP_CUT_DEPTHS_UM:
        for bw in SWEEP_BLADE_WIDTHS_UM:
            p = DEFAULT.copy()
            if base_cfg:
                for k in ("feed_speed_m_s", "mesh_global_um", "mesh_fine_um",
                          "friction_coeff", "num_cpus", "target_dt_s"):
                    if k in base_cfg:
                        p[k] = base_cfg[k]
            p["material"]     = material
            p["cut_depth_um"] = d
            p["blade_W_um"]   = bw
            name = "dicing_%s_d%03d_bw%02d" % (tag, d, bw)
            print("\n[->] %s  (depth=%dum kerf=%dum v=%.1fm/s DT=%.0e)" % (
                name, d, bw, p["feed_speed_m_s"], p.get("target_dt_s", 1e-8)))
            build_and_submit(p=p, job_name=name)
            results.append({"material": tag, "cut_depth_um": d,
                            "blade_W_um": bw, "job": name})

    manifest = "jobs_%s.json" % tag
    with open(manifest, "w") as f:
        json.dump(results, f, indent=2)
    print("[OK] Manifest: %s" % manifest)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Read run_config.json from cwd
# ─────────────────────────────────────────────────────────────────────────────
_config_path = os.path.join(os.getcwd(), "run_config.json")
_cfg = json.load(open(_config_path)) if os.path.exists(_config_path) else {}

from material_properties import ALL_MATERIALS
_mat = ALL_MATERIALS.get(_cfg.get("material", "SiC"), SiC)

if _cfg.get("study", False):
    parametric_study(material=_mat, base_cfg=_cfg)
else:
    _p = DEFAULT.copy()
    _p["material"] = _mat
    for _k in ("cut_depth_um", "blade_W_um", "feed_speed_m_s",
                "mesh_global_um", "mesh_fine_um", "friction_coeff",
                "num_cpus", "target_dt_s"):
        if _k in _cfg:
            _p[_k] = _cfg[_k]
    _tag  = _mat["name"].replace("-", "").replace(" ", "")
    _name = _cfg.get("job_name",
                     "dicing_%s_d%03d_bw%03d" % (
                         _tag, int(_p["cut_depth_um"]), int(_p["blade_W_um"])))
    build_and_submit(p=_p, job_name=_name)
