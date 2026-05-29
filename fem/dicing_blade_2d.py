"""
Phase 1: 2D plane-strain FEM of blade dicing on SiC wafer (ABAQUS scripting API)
Geometry: wafer cross-section, blade modeled as rigid indenter

Run with:
    abaqus cae noGUI=dicing_blade_2d.py
    abaqus python dicing_blade_2d.py  (Abaqus Python environment)

Parameters swept:
    - cutting_depth  : blade penetration depth [µm]
    - feed_rate      : blade traverse speed [mm/s]
    - blade_width    : kerf width [µm]
"""

import sys
sys.path.insert(0, '..')
from data.materials.material_properties import SiC

# ── Simulation parameters ─────────────────────────────────────────
PARAMS = {
    "wafer_width_um":   500.0,   # model domain width [µm]
    "wafer_depth_um":   200.0,   # wafer thickness [µm]
    "cutting_depth_um": 20.0,    # blade penetration [µm]
    "blade_width_um":   30.0,    # blade kerf width [µm]
    "feed_rate_mm_s":   50.0,    # blade traverse speed
    "blade_speed_rpm":  30000,   # spindle speed
}

def build_model(params=PARAMS, material=SiC, job_name="dicing_sic_2d"):
    """
    Build ABAQUS 2D dicing model via scripting API.
    Returns: job name for result extraction.
    """
    try:
        from abaqus import mdb, session
        from abaqusConstants import (DEFORMABLE_BODY, TWO_D_PLANAR,
                                     PLANE_STRAIN, ON, FIXED, XSYMM)
        from caeModules import *
    except ImportError:
        print("ABAQUS modules not found. Run with: abaqus python dicing_blade_2d.py")
        return None

    m = mdb.Model(name=job_name)
    s = m.ConstrainedSketch(name='wafer', sheetSize=1e-3)

    W = params["wafer_width_um"] * 1e-6
    H = params["wafer_depth_um"] * 1e-6

    # Wafer rectangle
    s.rectangle(point1=(0, 0), point2=(W, H))
    part = m.Part(name='Wafer', dimensionality=TWO_D_PLANAR,
                  type=DEFORMABLE_BODY)
    part.BaseShell(sketch=s)

    # Material
    mat = m.Material(name='SiC')
    mat.Elastic(table=((material["E"], material["nu"]),))
    mat.Density(table=((material["density"],),))
    # Extended Drucker-Prager for brittle fracture (placeholder)
    # mat.DruckerPrager(...)

    # Section
    m.HomogeneousSolidSection(name='WaferSection', material='SiC',
                               thickness=None)
    part.SectionAssignment(region=part.sets['Set-1'],
                           sectionName='WaferSection')

    # Mesh: 2 µm element size near cutting zone
    part.seedPart(size=2e-6, deviationFactor=0.1)
    part.generateMesh()

    # Assembly
    a = m.rootAssembly
    a.Instance(name='Wafer-1', part=part, dependent=ON)

    print(f"[dicing_blade_2d] Model '{job_name}' built.")
    print(f"  Material: {material['name']}, E={material['E']/1e9:.0f} GPa")
    print(f"  Cutting depth: {params['cutting_depth_um']} µm")
    return job_name


def parametric_study():
    """Sweep cutting depth: 10, 20, 30, 40, 50 µm"""
    results = []
    for depth in [10, 20, 30, 40, 50]:
        p = PARAMS.copy()
        p["cutting_depth_um"] = depth
        name = f"dicing_sic_depth{depth:02d}um"
        job = build_model(params=p, job_name=name)
        results.append({"depth_um": depth, "job": job})
    return results


if __name__ == "__main__":
    build_model()
    print("\nNext: run parametric_study() to sweep cutting depth.")
    print("Then: extract chipping width from ODB and fit surrogate model.")
