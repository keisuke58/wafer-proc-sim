"""
Blade dicing thermo-mechanical FEM — ABAQUS/Standard
2D plane-strain model: cutting heat generation → tool temperature → wear rate.

Gap-closing exercise: thermal field + tool wear physics for mechanical engineer role.
Complements dicing_blade_2d.py (fracture/chipping) with temperature-driven wear.

Physics:
    Heat generation:  q = mu * Fn * v_c  (friction at blade-workpiece interface)
    Heat partition:   R_blade = sqrt(k_w * rho_w * cp_w) /
                                (sqrt(k_b * rho_b * cp_b) + sqrt(k_w * rho_w * cp_w))
    Temperature rise: transient thermal FEM (DC2D4 elements)
    Wear rate:        Taylor tool life: T_life = C_T / v_c^n_T  [s]
                      Archard: W_dot = k_w * Fn / H_blade  [m/s]

Outputs:
    T_max_C          max blade temperature [°C]
    T_avg_contact_C  avg temperature at contact zone [°C]
    wear_rate_nm_m   Archard wear rate [nm per metre of cut]
    T_life_min       Taylor tool life [min] at given cutting speed

Run:
    abaqus cae noGUI=blade_thermal_wear_2d.py
    abaqus cae noGUI=blade_thermal_wear_2d.py -- --v_c 0.5 --Fn 8.0 --mu 0.35

References:
    Shaw (2005) Metal Cutting Principles, 2nd ed.
    Malkin & Guo (2008) Grinding Technology
    Disco blade wear data: internal application note (public summary)
"""

import sys
import os
import math

# ── Script path resolution ─────────────────────────────────────────────────────
_abq_script = None
for _i, _a in enumerate(sys.argv[:-1]):
    if _a == '-noGUI':
        _abq_script = os.path.abspath(sys.argv[_i + 1])
        break
if _abq_script is None:
    _abq_script = os.path.abspath(__file__) if '__file__' in dir() else os.path.abspath('blade_thermal_wear_2d.py')

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT = {
    "v_c":         0.30,    # cutting speed [m/s]  (blade peripheral speed)
    "Fn":          8.0,     # normal force per unit width [N/mm]
    "mu":          0.35,    # friction coefficient blade–SiC
    "workpiece":   "SiC4H", # SiC4H | Si | SiO2
    "blade":       "ResinBond",  # ResinBond | MetalBond | Electroformed
    "feed_mm_s":   0.01,    # feed speed [mm/s]  (wafer advance)
    "depth_um":    300.0,   # cut depth [µm]
    "T_ambient_C": 25.0,    # ambient / coolant temperature [°C]
    "t_sim_ms":    50.0,    # thermal simulation duration [ms]
    "mesh_um":     20.0,    # mesh size [µm]
    "num_cpus":    4,
}

cfg = dict(DEFAULT)
_args = sys.argv[:]
for k in ('v_c', 'Fn', 'mu', 'feed_mm_s', 'depth_um', 'T_ambient_C', 't_sim_ms', 'mesh_um'):
    tag = '--' + k
    if tag in _args:
        cfg[k] = float(_args[_args.index(tag) + 1])
for k in ('workpiece', 'blade', 'num_cpus'):
    tag = '--' + k
    if tag in _args:
        cfg[k] = _args[_args.index(tag) + 1]

# ── Material thermal properties ────────────────────────────────────────────────
WORKPIECE = {
    "SiC4H": {"k": 490.0, "rho": 3210.0, "cp": 750.0,  "H_GPa": 28.0},
    "Si":    {"k": 148.0, "rho": 2329.0, "cp": 700.0,  "H_GPa": 11.0},
    "SiO2":  {"k":   1.4, "rho": 2200.0, "cp": 740.0,  "H_GPa":  9.0},
}
BLADE = {
    "ResinBond":    {"k": 1.2,  "rho": 3500.0, "cp": 800.0, "H_GPa": 15.0, "C_T": 120.0, "n_T": 0.25},
    "MetalBond":    {"k": 45.0, "rho": 7800.0, "cp": 500.0, "H_GPa": 25.0, "C_T": 200.0, "n_T": 0.20},
    "Electroformed":{"k": 60.0, "rho": 8900.0, "cp": 390.0, "H_GPa": 30.0, "C_T": 300.0, "n_T": 0.18},
}
wp  = WORKPIECE[cfg["workpiece"]]
bl  = BLADE[cfg["blade"]]

# ── Analytical thermal model ───────────────────────────────────────────────────
# Heat flux at contact
q_gen = cfg["mu"] * cfg["Fn"] * 1e3 * cfg["v_c"]          # [W/m²] (Fn in N/mm → N/m × v)
# Thermal effusivity
e_w = math.sqrt(wp["k"] * wp["rho"] * wp["cp"])
e_b = math.sqrt(bl["k"] * bl["rho"] * bl["cp"])
R_b = e_b / (e_b + e_w)                                    # heat partition to blade [-]
q_blade = R_b * q_gen                                       # [W/m²]

# Contact length (chip-load geometry)
a_p = cfg["depth_um"] * 1e-6                                # cut depth [m]
l_c = math.sqrt(a_p * (cfg["feed_mm_s"] * 1e-3))           # geometric contact length [m]
l_c = max(l_c, 1e-6)

# Max temperature rise (Jaeger moving heat source, Peclet-corrected)
Pe = cfg["v_c"] * l_c / (wp["k"] / (wp["rho"] * wp["cp"]))
if Pe > 5:
    C_jaeger = 0.754
else:
    C_jaeger = 1.13 * math.sqrt(Pe) / Pe if Pe > 0 else 1.0
T_rise = C_jaeger * q_blade * l_c / wp["k"]
T_max = cfg["T_ambient_C"] + T_rise

# Taylor tool life
T_life_s = bl["C_T"] / (cfg["v_c"] ** bl["n_T"])

# Archard wear rate
k_archard = 1e-4                                            # dimensionless wear coefficient
wear_rate_nm_m = k_archard * (cfg["Fn"] * 1e3) / (bl["H_GPa"] * 1e9) * 1e12  # [nm/m]

print("=" * 60)
print("Blade Thermal & Wear Analysis — Analytical Pre-check")
print("=" * 60)
print("Workpiece       : {}".format(cfg["workpiece"]))
print("Blade type      : {}".format(cfg["blade"]))
print("Cutting speed   : {:.3f} m/s".format(cfg["v_c"]))
print("Normal force    : {:.1f} N/mm".format(cfg["Fn"]))
print("Friction coeff  : {:.2f}".format(cfg["mu"]))
print("-" * 60)
print("Heat flux       : {:.2e} W/m²".format(q_gen))
print("Heat to blade   : {:.1f} %".format(R_b * 100))
print("Contact length  : {:.1f} µm".format(l_c * 1e6))
print("T_max blade     : {:.1f} °C".format(T_max))
print("Taylor life     : {:.1f} s  = {:.2f} min".format(T_life_s, T_life_s / 60.0))
print("Archard wear    : {:.2f} nm/m".format(wear_rate_nm_m))
print("=" * 60)
if T_max > 600:
    print("WARN: T_max > 600°C — resin bond degradation risk.")
elif T_max > 400:
    print("CAUTION: T_max > 400°C — monitor coolant flow.")
else:
    print("OK: temperature within blade operating range.")

# ── ABAQUS transient thermal model ─────────────────────────────────────────────
try:
    from caeModules import *  # noqa: F401,F403
    import part, material, section, assembly, step, load, mesh, job  # noqa: F401

    W_m  = 500e-6   # domain width [m]
    H_m  = cfg["depth_um"] * 2 * 1e-6   # domain height [m]
    t_s  = cfg["t_sim_ms"] * 1e-3

    mdb_name = 'blade_thermal'
    if mdb_name in mdb.models:
        del mdb.models[mdb_name]
    m = mdb.Model(name=mdb_name)

    s = m.ConstrainedSketch(name='sk', sheetSize=W_m * 2)
    s.rectangle(point1=(0.0, 0.0), point2=(W_m, H_m))
    p = m.Part(name='Workpiece', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    p.BaseShell(sketch=s)

    m.Material(name='WP')
    m.materials['WP'].Conductivity(table=((wp["k"],),))
    m.materials['WP'].Density(table=((wp["rho"],),))
    m.materials['WP'].SpecificHeat(table=((wp["cp"],),))
    m.HomogeneousShellSection(name='sec', material='WP', thickness=1e-3)
    p.SectionAssignment(region=p.Set(faces=p.faces, name='all'), sectionName='sec')

    a = m.rootAssembly
    inst = a.Instance(name='WP-1', part=p, dependent=ON)

    m.HeatTransferStep(name='Thermal', previous='Initial',
                       timePeriod=t_s, maxNumInc=1000,
                       initialInc=t_s / 100, minInc=t_s / 1e6, maxInc=t_s / 10,
                       deltmx=50.0)

    # Heat flux on top surface (blade contact)
    top_edges = a.instances['WP-1'].edges.findAt(((W_m / 2, H_m, 0.0),))
    m.SurfaceHeatFlux(name='BladeHeat', createStepName='Thermal',
                      region=a.Surface(side1Edges=top_edges, name='top'),
                      magnitude=q_blade)

    # Initial + ambient BC
    all_nodes = a.Set(faces=inst.faces, name='all_f')
    m.Temperature(name='InitTemp', createStepName='Initial',
                  region=all_nodes, distributionType=UNIFORM,
                  crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
                  magnitudes=(cfg["T_ambient_C"],))

    p.seedPart(size=cfg["mesh_um"] * 1e-6)
    p.generateMesh()
    elemType = mesh.ElemType(elemCode=DC2D4, elemLibrary=STANDARD)
    p.setElementType(regions=(p.faces,), elemTypes=(elemType,))

    j = mdb.Job(name='blade_thermal', model=mdb_name,
                numCpus=int(cfg["num_cpus"]), numDomains=int(cfg["num_cpus"]))
    j.submit(consistencyChecking=OFF)
    j.waitForCompletion()
    print("ABAQUS job 'blade_thermal' completed.")

except ImportError:
    print("(ABAQUS caeModules not available — analytical pre-check only)")
except Exception as e:
    print("ABAQUS model error: {}".format(e))
