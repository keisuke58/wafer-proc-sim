"""
KABRA® laser-slicing thermo-mechanical FEM — ABAQUS/Standard + Explicit
2D plane-strain model of laser-induced stress fracture in SiC.

Physics:
    Step 1 (Heat): moving Gaussian laser beam → temperature field
                   (sequentially-coupled thermal-stress, ABAQUS/Standard)
    Step 2 (Mech): thermal strains → stress field → crack propagation
                   (progressive damage, max principal stress criterion)

KABRA process parameters:
    Wavelength : 1064 nm (Nd:YAG, absorbed in subsurface SiC)
    Focus depth: ~50–150 µm below surface
    Scan speed : 100–400 mm/s
    Power      : 5–30 W

Run:
    abaqus cae noGUI=kabra_thermal_2d.py
    abaqus cae noGUI=kabra_thermal_2d.py -- --power 15 --speed 200 --depth 100

References:
    Disco KABRA® patent (JP2016-111143)
    Stealth Dicing review: Kumagai et al. (2007) IEEE TCPMT
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.materials.material_properties import SiC

# ── Default KABRA parameters ──────────────────────────────────────────────────
DEFAULT_KABRA = {
    "material":          SiC,
    "wafer_W_um":        500.0,    # domain width [µm]
    "wafer_H_um":        200.0,    # wafer thickness [µm]
    "laser_power_W":      15.0,    # [W]
    "scan_speed_mm_s":   200.0,    # [mm/s]
    "focus_depth_um":    100.0,    # depth of laser focus below surface [µm]
    "beam_radius_um":     10.0,    # 1/e² Gaussian beam radius [µm]
    "absorptivity":        0.35,   # SiC @ 1064 nm (subsurface absorption)
    "mesh_global_um":      5.0,
    "mesh_fine_um":        2.0,    # near focus zone
    "num_cpus":            4,
}

SWEEP_POWERS_W    = [5, 10, 15, 20, 25]
SWEEP_SPEEDS_MM_S = [100, 200, 300, 400]


def build_and_submit(p=DEFAULT_KABRA, job_name="kabra_sic_p15_v200"):
    from abaqus import mdb, backwardCompatibility
    backwardCompatibility.setValues(reportDeprecated=False)

    from abaqusConstants import (
        TWO_D_PLANAR, DEFORMABLE_BODY, PLANE_STRAIN,
        ON, OFF, CARTESIAN, MIDDLE_SURFACE, FROM_SECTION,
        MAX_PRINCIPAL_STRESS, ENERGY, LINEAR,
        FIXED, UNSET, ENCASTRE,
        PERCENTAGE, SINGLE,
        LAGRANGIAN,
    )
    from caeModules import *
    import regionToolset
    from mesh import ElemType
    from abaqusConstants import CPE4T, CPE3T, CPE4R, CPE3   # thermo-mechanical elements

    # ── Unit conversion ───────────────────────────────────────────────────────
    W  = p["wafer_W_um"]      * 1e-6
    H  = p["wafer_H_um"]      * 1e-6
    P  = p["laser_power_W"]
    v  = p["scan_speed_mm_s"] * 1e-3   # m/s
    zf = p["focus_depth_um"]  * 1e-6   # focus depth below top surface
    r0 = p["beam_radius_um"]  * 1e-6   # Gaussian beam radius
    eta = p["absorptivity"]
    mg  = p["mesh_global_um"] * 1e-6
    mf  = p["mesh_fine_um"]   * 1e-6
    mat = p["material"]

    # Effective heat source intensity [W/m²] at focus
    # Gaussian: Q(r, z) = Q0 * exp(-2r²/r0²) * exp(-alpha*(H-zf-z))
    # Here: use surface flux approximation with absorbed power = eta * P
    Q0 = eta * P / (3.14159 * r0 ** 2)   # peak intensity [W/m²]

    # Scan duration: beam travels full width
    t_scan  = W / v
    G_c     = mat["K_Ic"] ** 2 / mat["E"]
    xc_start = r0 * 2   # start position (beam center)

    # ── Create model ──────────────────────────────────────────────────────────
    if job_name in mdb.models:
        del mdb.models[job_name]
    model = mdb.Model(name=job_name)
    if "Model-1" in mdb.models and job_name != "Model-1":
        del mdb.models["Model-1"]

    # ═══════════════════════════════════════════════════════════════════════════
    #  PART: Wafer
    # ═══════════════════════════════════════════════════════════════════════════
    sk = model.ConstrainedSketch(name="wafer_sk", sheetSize=W * 4)
    sk.rectangle(point1=(0.0, 0.0), point2=(W, H))
    wafer = model.Part(name="Wafer", dimensionality=TWO_D_PLANAR,
                       type=DEFORMABLE_BODY)
    wafer.BaseShell(sketch=sk)

    # Partition: horizontal line at focus depth (from top: H - zf)
    focus_y = H - zf
    fine_half_x = r0 * 5

    f0 = wafer.faces[0]
    t_sk = wafer.MakeSketchTransform(
        sketchPlane=f0, sketchUpEdge=wafer.edges.findAt((W/2, H, 0.0)),
        sketchPlaneSide=SIDE1, origin=(0.0, 0.0, 0.0))
    sk_p = model.ConstrainedSketch(name="part_sk", sheetSize=W*4,
                                    gridSpacing=mf, transform=t_sk)
    sk_p.Line(point1=(0.0, focus_y), point2=(W, focus_y))
    wafer.PartitionFaceBySketch(faces=wafer.faces[:], sketch=sk_p)

    # ═══════════════════════════════════════════════════════════════════════════
    #  MATERIAL (thermo-mechanical)
    # ═══════════════════════════════════════════════════════════════════════════
    m = model.Material(name=mat["name"])
    m.Elastic(table=((mat["E"], mat["nu"]),))
    m.Density(table=((mat["density"],),))
    m.Conductivity(table=((mat["k_thermal"],),))
    m.SpecificHeat(table=((750.0,),))   # J/(kg·K), typical SiC
    m.Expansion(table=((mat["alpha_thermal"],),))

    # Progressive damage for mechanical step
    m.DamageInitiation(criterion=MAX_PRINCIPAL_STRESS,
                       table=((mat["sigma_y"],),))
    m.damageInitiation.DamageEvolution(
        type=ENERGY, softening=LINEAR, table=((G_c,),))

    # ═══════════════════════════════════════════════════════════════════════════
    #  SECTION (thermo-mechanical solid, plane strain)
    # ═══════════════════════════════════════════════════════════════════════════
    model.HomogeneousSolidSection(name="WaferSec",
                                   material=mat["name"], thickness=1.0e-6)
    wafer_all = wafer.Set(faces=wafer.faces[:], name="WaferAll")
    wafer.SectionAssignment(region=wafer_all, sectionName="WaferSec",
                            offset=0.0, offsetType=MIDDLE_SURFACE,
                            offsetField="", thicknessAssignment=FROM_SECTION)

    # ═══════════════════════════════════════════════════════════════════════════
    #  ASSEMBLY
    # ═══════════════════════════════════════════════════════════════════════════
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    wafer_inst = a.Instance(name="Wafer-1", part=wafer, dependent=ON)

    # ═══════════════════════════════════════════════════════════════════════════
    #  STEP 1: Heat Transfer (moving laser source)
    # ═══════════════════════════════════════════════════════════════════════════
    model.HeatTransferStep(
        name="LaserScan",
        previous="Initial",
        timePeriod=t_scan,
        maxNumInc=10000,
        initialInc=t_scan / 500,
        minInc=1e-8,
        maxInc=t_scan / 50,
        deltmx=50.0)   # max temperature change per increment [K]

    # Initial temperature field (room temp 300 K)
    model.Temperature(
        name="InitTemp",
        createStepName="Initial",
        region=regionToolset.Region(instances=(wafer_inst,)),
        distributionType=UNIFORM,
        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
        magnitudes=(300.0,))

    # Moving Gaussian heat flux via DFLUX user subroutine or tabular amplitude
    # Here: approximate with tabular heat flux at focus depth (y = focus_y)
    # Use ABAQUS analytical field to represent Gaussian × moving window
    #
    # Amplitude: laser traverses from x=0 to x=W in t_scan seconds
    # We apply a body heat flux Q(x, y, t) via an AnalyticalField
    #
    # Simplified: uniform volumetric heat source in focus zone, swept by amplitude
    focus_zone_faces = wafer.faces.getByBoundingBox(
        xMin=0, xMax=W/10, yMin=focus_y - r0*2, yMax=focus_y + r0*2)

    # Create surface heat flux on top (beam enters from top, absorbed at focus)
    top_edge = wafer_inst.edges.findAt(((W/2, H, 0.0),))
    top_surf  = a.Surface(side1Edges=(top_edge,), name="WaferTop")

    # Amplitude: ramp ON at t=0 (beam enters), stays on for full scan
    # More accurate implementation would use DFLUX subroutine in Fortran/C
    model.TabularAmplitude(
        name="LaserAmp",
        timeSpan=STEP,
        data=((0.0, 1.0), (t_scan, 1.0)))

    model.SurfaceHeatFlux(
        name="LaserFlux",
        createStepName="LaserScan",
        region=top_surf,
        magnitude=Q0,
        amplitude="LaserAmp")

    # BCs: adiabatic bottom, convective top (coolant)
    bot_edge = wafer_inst.edges.findAt(((W/2, 0.0, 0.0),))
    model.FilmCondition(
        name="Coolant",
        createStepName="LaserScan",
        surface=a.Surface(side1Edges=(bot_edge,), name="WaferBot"),
        definition=EMBEDDED_COEFF,
        filmCoeff=1000.0,       # W/(m²·K), water coolant
        filmCoeffAmplitude="",
        sinkTemperature=300.0,
        sinkAmplitude="")

    # ═══════════════════════════════════════════════════════════════════════════
    #  STEP 2: Mechanical (thermal strains → fracture)
    # ═══════════════════════════════════════════════════════════════════════════
    model.StaticStep(
        name="ThermalStress",
        previous="LaserScan",
        maxNumInc=1000,
        initialInc=0.01,
        minInc=1e-6,
        maxInc=0.1,
        nlgeom=ON)

    # Mechanical BCs: bottom encastre, left/right x-roller
    model.EncastreBC(name="WaferBottom", createStepName="Initial",
                     region=regionToolset.Region(edges=bot_edge))
    lft = wafer_inst.edges.findAt(((0.0, H/2, 0.0),))
    rgt = wafer_inst.edges.findAt(((W,   H/2, 0.0),))
    model.DisplacementBC(name="WaferLeft",  createStepName="Initial",
                         region=regionToolset.Region(edges=lft), u1=0.0)
    model.DisplacementBC(name="WaferRight", createStepName="Initial",
                         region=regionToolset.Region(edges=rgt), u1=0.0)

    # ═══════════════════════════════════════════════════════════════════════════
    #  MESH — CPE4T: plane-strain, thermal-mechanical coupled elements
    # ═══════════════════════════════════════════════════════════════════════════
    elem_quad = ElemType(elemCode=CPE4T,  elemLibrary=STANDARD)
    elem_tri  = ElemType(elemCode=CPE3T,  elemLibrary=STANDARD)

    wafer.setElementType(
        regions=regionToolset.Region(faces=wafer.faces[:]),
        elemTypes=(elem_quad, elem_tri))

    wafer.seedPart(size=mg, deviationFactor=0.1, minSizeFactor=0.1)

    # Fine mesh near focus zone (horizontal band)
    focus_edges = wafer.edges.getByBoundingBox(
        xMin=0, xMax=W,
        yMin=focus_y - r0 * 3, yMax=focus_y + r0 * 3)
    if len(focus_edges) > 0:
        wafer.seedEdgeBySize(edges=focus_edges, size=mf, constraint=FINER)

    wafer.generateMesh()

    # ═══════════════════════════════════════════════════════════════════════════
    #  OUTPUT REQUESTS
    # ═══════════════════════════════════════════════════════════════════════════
    # Thermal step: temperature field
    model.fieldOutputRequests["F-Output-1"].setValues(
        variables=("NT", "HFL"),
        timeInterval=t_scan / 20)

    # Mechanical step: stress, displacement, damage
    model.FieldOutputRequest(
        name="F-Mech",
        createStepName="ThermalStress",
        variables=("S", "U", "STATUS", "DMICRT", "ENER", "LE"),
        timeInterval=0.05)

    # ═══════════════════════════════════════════════════════════════════════════
    #  SAVE + SUBMIT
    # ═══════════════════════════════════════════════════════════════════════════
    mdb.saveAs(pathName=job_name + ".cae")
    print(f"[*] Model saved: {job_name}.cae")
    print(f"    Q0 = {Q0/1e6:.2f} MW/m²  |  scan time = {t_scan*1e3:.2f} ms")

    job = mdb.Job(
        name=job_name, model=job_name,
        numCpus=p.get("num_cpus", 4),
        numDomains=p.get("num_cpus", 4),
        memory=80, memoryUnits=PERCENTAGE)

    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    print(f"[✓] Job '{job_name}' completed.")
    return job_name


def parametric_study():
    import json
    results = []
    for pw in SWEEP_POWERS_W:
        for spd in SWEEP_SPEEDS_MM_S:
            p = DEFAULT_KABRA.copy()
            p["laser_power_W"]   = pw
            p["scan_speed_mm_s"] = spd
            name = f"kabra_sic_p{pw:02d}_v{spd:03d}"
            print(f"\n[→] {name}  (P={pw}W, v={spd}mm/s)")
            build_and_submit(p=p, job_name=name)
            results.append({"power_W": pw, "speed_mm_s": spd, "job": name})

    with open("jobs_kabra_SiC.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[✓] Manifest saved: jobs_kabra_SiC.json")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--study",   action="store_true")
    parser.add_argument("--power",   type=float, default=15.0)
    parser.add_argument("--speed",   type=float, default=200.0)
    parser.add_argument("--depth",   type=float, default=100.0)
    args = parser.parse_args()

    if args.study:
        parametric_study()
    else:
        p = DEFAULT_KABRA.copy()
        p["laser_power_W"]   = args.power
        p["scan_speed_mm_s"] = args.speed
        p["focus_depth_um"]  = args.depth
        name = f"kabra_sic_p{int(args.power):02d}_v{int(args.speed):03d}"
        build_and_submit(p=p, job_name=name)
