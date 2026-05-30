"""
3D blade dicing FEM — ABAQUS/Explicit (coarse validation model)
Purpose: validate 2D surrogate trends; run 3~5 jobs near GP-identified optimum.

Geometry:
    X : feed direction  (wafer width)
    Y : depth direction (wafer thickness, blade cuts downward)
    Z : out-of-plane    (blade axis, perpendicular to feed)

Blade: trapezoidal profile (same as 2D) extruded in Z = 3 × kerf widths.
Wafer: same Drucker-Prager material as 2D for direct comparison.
Mesh : 5 µm global (coarse) — ~72,000 C3D8R elements for default params.
Time : ~2.5 h per job (single CPU, consistent with estimate).

Run:
    # Create run_config_3d.json in cwd, then:
    abaqus cae noGUI=dicing_blade_3d.py

run_config_3d.json keys: same as 2D + "lz_factor" (Z extent = lz_factor × blade_W_um)
"""

import sys
import os
import json
import re

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
from material_properties import SiC, Si, ALL_MATERIALS

from abaqus import mdb, backwardCompatibility
backwardCompatibility.setValues(reportDeprecated=False)
from abaqusConstants import *
from caeModules import *
import regionToolset
from mesh import ElemType

DEFAULT_3D = {
    "material":        SiC,
    "wafer_W_um":      300.0,   # shorter X for 3D (saves time vs 500µm)
    "wafer_H_um":      150.0,   # wafer thickness
    "lz_factor":         3.0,   # Z extent = lz_factor × blade_W_um
    "cut_depth_um":     30.0,
    "blade_W_um":       30.0,
    "feed_speed_m_s":    0.5,
    "mesh_global_um":    5.0,   # coarse
    "target_dt_s":       1e-8,
    "friction_coeff":    0.3,
    "num_cpus":          1,
}


def _abq_name(s):
    s = re.sub(r'[^A-Za-z0-9_]', '_', str(s))
    if s and s[0].isdigit():
        s = 'M_' + s
    return s


def build_model_3d(p=DEFAULT_3D, job_name="dicing3d_sic_d030"):
    W   = p["wafer_W_um"]  * 1e-6
    H   = p["wafer_H_um"]  * 1e-6
    d   = p["cut_depth_um"] * 1e-6
    bw  = p["blade_W_um"]  * 1e-6
    v   = p["feed_speed_m_s"]
    mg  = p["mesh_global_um"] * 1e-6
    mat = p["material"]
    Lz  = p["lz_factor"] * bw   # out-of-plane extent

    _CLEARANCE = 2e-6          # initial gap to ensure explicit contact detection
    travel   = W * 0.8
    t_plunge = (d + _CLEARANCE) / v
    t_feed   = travel / v
    chamfer  = bw * 0.15
    half     = bw / 2
    mat_name = _abq_name(mat["name"])

    # ── Model ─────────────────────────────────────────────────────────────────
    if job_name in mdb.models:
        del mdb.models[job_name]
    model = mdb.Model(name=job_name)
    if "Model-1" in mdb.models and job_name != "Model-1":
        del mdb.models["Model-1"]

    # ═══ PART: Wafer (3D solid) ════════════════════════════════════════════════
    sk_w = model.ConstrainedSketch(name="wafer_sk", sheetSize=W * 4)
    sk_w.rectangle(point1=(0.0, 0.0), point2=(W, H))
    wafer = model.Part(name="Wafer", dimensionality=THREE_D,
                       type=DEFORMABLE_BODY)
    wafer.BaseSolidExtrude(sketch=sk_w, depth=Lz)

    # ═══ MATERIAL ═════════════════════════════════════════════════════════════
    m = model.Material(name=mat_name)
    m.Elastic(table=((mat["E"], mat["nu"]),))
    m.Density(table=((mat["density"],),))
    m.DruckerPrager(
        shearCriterion=LINEAR, eccentricity=0.1,
        testData=OFF, temperatureDependency=OFF, dependencies=0,
        table=((mat["dp_friction_angle"], mat["K_dp"],
                mat["dp_dilation_angle"]),))
    m.druckerPrager.DruckerPragerHardening(
        type=SHEAR, temperatureDependency=OFF, dependencies=0,
        table=((mat["dp_cohesion_Pa"], 0.0),))
    m.DuctileDamageInitiation(table=mat["fracture_table"],
                               temperatureDependency=OFF, dependencies=0)
    m.ductileDamageInitiation.DamageEvolution(
        type=ENERGY, softening=LINEAR, table=((mat["G_c"],),))

    # ═══ SECTION ══════════════════════════════════════════════════════════════
    model.HomogeneousSolidSection(name="WaferSec", material=mat_name)
    wafer_all = wafer.Set(cells=wafer.cells[:], name="WaferAll")
    wafer.SectionAssignment(region=wafer_all, sectionName="WaferSec",
                            offset=0.0, offsetType=MIDDLE_SURFACE,
                            offsetField="", thicknessAssignment=FROM_SECTION)

    # ═══ PART: Blade (3D discrete rigid shell, extruded trapezoidal tip) ══════
    sk_b = model.ConstrainedSketch(name="blade_sk", sheetSize=W * 4)
    sk_b.Line(point1=(-half,              0.0),
              point2=(-half + chamfer, -chamfer))
    sk_b.Line(point1=(-half + chamfer, -chamfer),
              point2=( half - chamfer, -chamfer))
    sk_b.Line(point1=( half - chamfer, -chamfer),
              point2=( half,              0.0))

    blade = model.Part(name="Blade", dimensionality=THREE_D,
                       type=DISCRETE_RIGID_SURFACE)
    blade.BaseShellExtrude(sketch=sk_b, depth=Lz)

    # Reference point at blade top-center
    blade.ReferencePoint(point=(0.0, 0.0, Lz / 2))
    blade_rp_key = list(blade.referencePoints.keys())[-1]

    # ═══ ASSEMBLY ═════════════════════════════════════════════════════════════
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    wafer_inst = a.Instance(name="Wafer-1", part=wafer, dependent=ON)
    blade_inst = a.Instance(name="Blade-1", part=blade, dependent=ON)

    # Centre blade at X=W/2 (within wafer top face) with 2µm clearance above wafer.
    # X=bw (left edge) was outside the fine-mesh zone and caused contact miss.
    a.translate(instanceList=("Blade-1",),
                vector=(W / 2, H + chamfer + _CLEARANCE, 0.0))

    # ═══ STEPS ════════════════════════════════════════════════════════════════
    ms_table = ((SEMI_AUTOMATIC, MODEL, AT_BEGINNING,
                 0.0, 0.0, BELOW_MIN, 1, 0, 0.0, 0.0, 0, None),)
    model.ExplicitDynamicsStep(name="Plunge", previous="Initial",
                                timePeriod=t_plunge, massScaling=ms_table,
                                improvedDtMethod=ON)
    model.ExplicitDynamicsStep(name="Feed",   previous="Plunge",
                                timePeriod=t_feed,   massScaling=ms_table,
                                improvedDtMethod=ON)

    # ═══ BOUNDARY CONDITIONS ══════════════════════════════════════════════════
    # Wafer bottom (Y=0 face)
    bot_face = wafer_inst.faces.findAt(((W/2, 0.0, Lz/2),))
    model.EncastreBC(name="WaferBottom", createStepName="Initial",
                     region=regionToolset.Region(faces=bot_face))

    # Wafer X-faces: roller (prevent rigid body in X)
    lft_face = wafer_inst.faces.findAt(((0.0, H/2, Lz/2),))
    rgt_face = wafer_inst.faces.findAt(((W,   H/2, Lz/2),))
    model.DisplacementBC(name="WaferLeft",  createStepName="Initial",
                         region=regionToolset.Region(faces=lft_face), u1=0.0)
    model.DisplacementBC(name="WaferRight", createStepName="Initial",
                         region=regionToolset.Region(faces=rgt_face), u1=0.0)

    # Wafer Z-faces: symmetry (blade cuts symmetrically in Z)
    zmin_face = wafer_inst.faces.findAt(((W/2, H/2, 0.0),))
    zmax_face = wafer_inst.faces.findAt(((W/2, H/2, Lz),))
    model.DisplacementBC(name="WaferZmin", createStepName="Initial",
                         region=regionToolset.Region(faces=zmin_face), u3=0.0)
    model.DisplacementBC(name="WaferZmax", createStepName="Initial",
                         region=regionToolset.Region(faces=zmax_face), u3=0.0)

    # Blade: fix rotation and Z-translation; drive X-Y via VelocityBC.
    # DisplacementBC on rigid body RP is silently ignored in ABAQUS/Explicit
    # when applied in a non-Initial step — always use VelocityBC instead.
    blade_rp_inst = a.instances["Blade-1"].referencePoints[blade_rp_key]
    blade_region  = regionToolset.Region(referencePoints=(blade_rp_inst,))
    model.DisplacementBC(name="BladeRot", createStepName="Initial",
                         region=blade_region, ur1=0.0, ur2=0.0, ur3=0.0, u3=0.0)
    model.VelocityBC(name="BladeMotion", createStepName="Plunge",
                     region=blade_region, v1=0.0, v2=-v)
    model.boundaryConditions["BladeMotion"].setValuesInStep(
        stepName="Feed", v1=(travel / t_feed), v2=0.0)

    # ═══ CONTACT ══════════════════════════════════════════════════════════════
    top_face = wafer_inst.faces.findAt(((W/2, H, Lz/2),))
    a.Surface(side1Faces=(top_face,), name="WaferTop")
    a.Surface(side1Faces=blade_inst.faces[:], name="BladeSurf")

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
        sliding=FINITE, interactionProperty="FricContact",
        mechanicalConstraint=PENALTY)

    # ═══ MESH ═════════════════════════════════════════════════════════════════
    elem_hex  = ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT,
                         secondOrderAccuracy=OFF, hourglassControl=ENHANCED,
                         distortionControl=ON)
    elem_tet  = ElemType(elemCode=C3D4,  elemLibrary=EXPLICIT)
    wafer.setElementType(
        regions=regionToolset.Region(cells=wafer.cells[:]),
        elemTypes=(elem_hex, elem_tet))
    wafer.seedPart(size=mg, deviationFactor=0.1, minSizeFactor=0.1)
    wafer.generateMesh()

    # Blade: DISCRETE_RIGID_SURFACE auto-assigns R3D4/R3D3 — no setElementType needed
    blade.seedPart(size=mg)
    blade.generateMesh()

    # Element count estimate
    n_wafer = len(wafer.elements)
    print("[3D] Wafer elements: %d" % n_wafer)

    # ═══ OUTPUT ═══════════════════════════════════════════════════════════════
    model.fieldOutputRequests["F-Output-1"].setValues(
        variables=("S", "U", "STATUS", "PEEQ", "ENER"),
        timeInterval=(t_plunge + t_feed) / 30)
    model.HistoryOutputRequest(
        name="BladeForce", createStepName="Plunge",
        region=blade_region,
        variables=("RF1", "RF2", "RF3", "U1", "U2"),
        numIntervals=300)

    # ═══ WRITE INP ════════════════════════════════════════════════════════════
    job = mdb.Job(
        name=job_name, model=job_name,
        numCpus=p.get("num_cpus", 1), numDomains=p.get("num_cpus", 1),
        memory=80, memoryUnits=PERCENTAGE,
        explicitPrecision=DOUBLE, nodalOutputPrecision=FULL)
    job.writeInput(consistencyChecking=OFF)
    print("[INP] Written: %s.inp" % job_name)
    return job_name


def _inject_dt(inp_path, target_dt):
    with open(inp_path, "r") as f:
        content = f.read()
    new_content, n = re.subn(
        r'\*Fixed Mass Scaling\s*\n',
        '*Fixed Mass Scaling, dt=%.2e, type=BELOW MIN\n' % target_dt,
        content)
    if n:
        with open(inp_path, "w") as f:
            f.write(new_content)
        print("[INP] DT=%.2e injected (%d blocks)" % (target_dt, n))


def submit_and_wait_3d(job_name, num_cpus=1):
    job = mdb.JobFromInputFile(
        name=job_name + "_run",
        inputFileName=job_name + ".inp",
        numCpus=num_cpus, numDomains=num_cpus,
        memory=80, memoryUnits=PERCENTAGE,
        explicitPrecision=DOUBLE, nodalOutputPrecision=FULL)
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    run_odb = job_name + "_run.odb"
    out_odb = job_name + ".odb"
    if os.path.exists(run_odb) and not os.path.exists(out_odb):
        os.rename(run_odb, out_odb)
    print("[OK] 3D job completed: %s" % job_name)
    return job_name


# ─────────────────────────────────────────────────────────────────────────────
_cfg_path = os.path.join(os.getcwd(), "run_config_3d.json")
_cfg = json.load(open(_cfg_path)) if os.path.exists(_cfg_path) else {}

_mat = ALL_MATERIALS.get(_cfg.get("material", "SiC"), SiC)
_p   = DEFAULT_3D.copy()
_p["material"] = _mat
for _k in ("cut_depth_um", "blade_W_um", "wafer_W_um", "wafer_H_um",
           "lz_factor", "feed_speed_m_s", "mesh_global_um",
           "target_dt_s", "friction_coeff", "num_cpus"):
    if _k in _cfg:
        _p[_k] = _cfg[_k]

_tag  = _mat["name"].replace("-", "").replace(" ", "")
_name = _cfg.get("job_name",
                 "dicing3d_%s_d%03d_bw%03d" % (
                     _tag, int(_p["cut_depth_um"]), int(_p["blade_W_um"])))

build_model_3d(p=_p, job_name=_name)
_inject_dt(os.path.join(os.getcwd(), _name + ".inp"), _p.get("target_dt_s", 1e-8))
submit_and_wait_3d(_name, num_cpus=_p.get("num_cpus", 1))
