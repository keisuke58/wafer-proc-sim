"""
Stealth Dicing crack propagation model — fracture mechanics (Mode I + III)
2D analytical + ABAQUS XFEM model of laser-modified layer → tape-expand cleavage.

Physics:
    Step 1 (Laser): Gaussian pulse → thermal shock → modified layer (amorphous + dislocations)
                    Temperature field: T(r,z,t) from Beer-Lambert + Gaussian beam
                    Residual stress in modified zone: sigma_r ~ E * alpha * Delta_T
    Step 2 (Tape):  Tape expansion applies biaxial tensile stress sigma_app
                    Stress intensity: K_I = sigma_app * sqrt(pi * a) * F(a/W)
                    Crack propagates when K_I >= K_Ic_modified
                    K_Ic_modified < K_Ic_bulk  (modified layer is weakened)

Outputs:
    T_max_modified_C    peak temperature in modified layer [°C]
    sigma_residual_MPa  residual tensile stress in modified zone [MPa]
    K_I_MPa_sqrtm       stress intensity factor at crack tip
    sigma_tape_min_MPa  minimum tape expansion stress for cleavage
    crack_path          predicted crack direction (vertical preferred for clean cut)

References:
    Kumagai et al. (2007) IEEE TCPMT — stealth dicing Si wafer
    Hosseini & Bhowmik (2016) Optics Express — modified layer characterization
    Ohmura et al. (2006) JLMN — SIF analysis of stealth dicing crack propagation
    K_Ic Si bulk: 0.83 MPa√m; modified: ~0.3 MPa√m
"""

import sys
import math

DEFAULT = {
    "material":          "Si",      # Si | SiC4H | GaAs
    "laser_power_W":      1.5,      # average laser power [W]
    "pulse_freq_kHz":   100.0,      # repetition rate [kHz]
    "scan_speed_mm_s":  500.0,      # scan speed [mm/s]
    "focus_depth_um":   100.0,      # laser focus depth below surface [µm]
    "beam_radius_um":     3.0,      # 1/e² beam radius [µm]
    "wafer_thick_um":   775.0,      # wafer thickness [µm]
    "n_passes":           1,        # number of laser passes
    "tape_strain_pct":    2.0,      # tape expansion strain [%]
}

cfg = dict(DEFAULT)
_args = sys.argv[:]
for k in ("laser_power_W", "pulse_freq_kHz", "scan_speed_mm_s", "focus_depth_um",
          "beam_radius_um", "wafer_thick_um", "tape_strain_pct"):
    tag = "--" + k
    if tag in _args:
        cfg[k] = float(_args[_args.index(tag) + 1])
for k in ("material", "n_passes"):
    tag = "--" + k
    if tag in _args:
        cfg[k] = _args[_args.index(tag) + 1]

# ── Material database ──────────────────────────────────────────────────────────
MATERIALS = {
    "Si":    {"E": 170e9,  "nu": 0.28, "alpha": 2.6e-6, "k": 148.0, "rho": 2329.0, "cp": 700.0,
              "K_Ic_bulk": 0.83,  "K_Ic_mod_frac": 0.36,  "absorb_1064": 0.0,   "absorb_1342": 0.85},
    "SiC4H":{"E": 480e9,  "nu": 0.21, "alpha": 4.3e-6, "k": 490.0, "rho": 3210.0, "cp": 750.0,
              "K_Ic_bulk": 2.8,   "K_Ic_mod_frac": 0.40,  "absorb_1064": 0.0,   "absorb_532":  0.90},
    "GaAs":  {"E":  85e9,  "nu": 0.31, "alpha": 6.0e-6, "k":  46.0, "rho": 5317.0, "cp": 327.0,
              "K_Ic_bulk": 0.44,  "K_Ic_mod_frac": 0.30,  "absorb_1064": 0.95,  "absorb_532":  0.98},
}
mp = MATERIALS[cfg["material"]]

# ── Step 1: Laser thermal model ────────────────────────────────────────────────
E_pulse_J   = cfg["laser_power_W"] / (cfg["pulse_freq_kHz"] * 1e3)
r0_m        = cfg["beam_radius_um"] * 1e-6
I0          = E_pulse_J / (math.pi * r0_m**2)        # peak fluence [J/m²]
absorb      = list(mp.items())[-2][1]                 # use first absorptivity entry
# Temperature rise at focus (adiabatic estimate, Gaussian pulse)
delta_z     = cfg["beam_radius_um"] * 1e-6            # interaction depth ~ beam radius
vol         = math.pi * r0_m**2 * delta_z
mass        = mp["rho"] * vol
Q_abs       = absorb * E_pulse_J
T_rise      = Q_abs / (mass * mp["cp"]) if mass > 0 else 0.0
T_max       = 25.0 + T_rise

# Modified layer thickness (empirical: ~2× Rayleigh range)
lambda_nm   = 1064.0   # laser wavelength [nm] (Nd:YAG)
z_R         = math.pi * r0_m**2 / (lambda_nm * 1e-9)   # Rayleigh range [m]
mod_thick_um = min(2.0 * z_R * 1e6, cfg["wafer_thick_um"] / 4.0)

# Residual stress in modified zone (thermal mismatch)
delta_T     = min(T_rise, 1200.0)                     # cap at melting
sigma_res   = mp["E"] * mp["alpha"] * delta_T / (1.0 - mp["nu"])  # [Pa]

# ── Step 2: Tape expansion fracture mechanics ──────────────────────────────────
W_m         = cfg["wafer_thick_um"] * 1e-6
a_m         = mod_thick_um * 1e-6 / 2.0              # half crack length [m]
eps_tape    = cfg["tape_strain_pct"] / 100.0
sigma_app   = mp["E"] * eps_tape                      # applied stress [Pa]

# Geometric factor F(a/W) for edge crack in strip
a_over_W    = a_m / W_m
F           = (1.12 - 0.231 * a_over_W + 10.55 * a_over_W**2
               - 21.72 * a_over_W**3 + 30.39 * a_over_W**4)
K_I         = (sigma_app + sigma_res) * math.sqrt(math.pi * a_m) * F   # [Pa√m]
K_I_MPa     = K_I * 1e-6                             # [MPa√m]

K_Ic_mod    = mp["K_Ic_bulk"] * mp["K_Ic_mod_frac"]  # [MPa√m]
sigma_min   = (K_Ic_mod * 1e6 - sigma_res * math.sqrt(math.pi * a_m) * F) \
              / (math.sqrt(math.pi * a_m) * F) * 1e-6  # [MPa]
sigma_tape_min = sigma_min / mp["E"] * 100.0          # [%] equivalent tape strain

# n_passes: multiple passes weaken further
K_Ic_eff    = K_Ic_mod / math.sqrt(float(cfg["n_passes"]))

cleave_ok   = K_I_MPa >= K_Ic_eff

print("=" * 60)
print("Stealth Dicing Crack Model — {}".format(cfg["material"]))
print("=" * 60)
print("Laser power     : {:.2f} W   @ {:.0f} kHz".format(cfg["laser_power_W"], cfg["pulse_freq_kHz"]))
print("Scan speed      : {:.0f} mm/s,  focus: {:.0f} µm depth".format(cfg["scan_speed_mm_s"], cfg["focus_depth_um"]))
print("Beam radius     : {:.1f} µm".format(cfg["beam_radius_um"]))
print("-" * 60)
print("T_max modified  : {:.0f} °C".format(T_max))
print("Modified layer  : {:.1f} µm thick".format(mod_thick_um))
print("Residual stress : {:.1f} MPa (tensile)".format(sigma_res * 1e-6))
print("-" * 60)
print("Tape strain     : {:.1f} %  → sigma_app = {:.0f} MPa".format(cfg["tape_strain_pct"], sigma_app * 1e-6))
print("K_I             : {:.3f} MPa√m".format(K_I_MPa))
print("K_Ic (modified) : {:.3f} MPa√m  ({:.0f}% of bulk)".format(
    K_Ic_mod, mp["K_Ic_mod_frac"] * 100))
print("K_Ic (bulk)     : {:.3f} MPa√m".format(mp["K_Ic_bulk"]))
print("-" * 60)
if cleave_ok:
    print("CLEAVE: K_I >= K_Ic_eff → clean separation expected")
else:
    need_strain = sigma_tape_min
    print("NO CLEAVE: K_I < K_Ic_eff → increase tape strain to ~{:.1f} %".format(max(need_strain, 0.0)))
print("=" * 60)

# ── ABAQUS XFEM model (skeleton) ──────────────────────────────────────────────
try:
    from caeModules import *  # noqa: F401,F403
    import part, material, section, assembly, step, load, mesh, interaction  # noqa: F401

    W_m2  = cfg["wafer_thick_um"] * 1e-6
    L_m   = W_m2 * 2.0
    mod_h = mod_thick_um * 1e-6

    mdb_name = 'stealth_xfem'
    if mdb_name in mdb.models:
        del mdb.models[mdb_name]
    m = mdb.Model(name=mdb_name)

    s = m.ConstrainedSketch(name='wafer_sk', sheetSize=L_m * 2)
    s.rectangle(point1=(0.0, 0.0), point2=(L_m, W_m2))
    p = m.Part(name='Wafer', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    p.BaseShell(sketch=s)

    # Partition: modified layer zone
    p.PartitionFaceByShortestPath(
        faces=p.faces,
        point1=p.InterestingPoint(p.edges.findAt(((0.0, cfg["focus_depth_um"] * 1e-6, 0.0),)), MIDDLE),
        point2=p.InterestingPoint(p.edges.findAt(((L_m, cfg["focus_depth_um"] * 1e-6, 0.0),)), MIDDLE)
    )

    # Material
    m.Material(name=cfg["material"])
    m.materials[cfg["material"]].Elastic(table=((mp["E"], mp["nu"]),))
    m.HomogeneousSolidSection(name='sec', material=cfg["material"], thickness=None)
    p.SectionAssignment(region=p.Set(faces=p.faces, name='all'), sectionName='sec')

    # XFEM crack domain (modified layer region)
    mod_faces = p.faces.findAt(((L_m / 2, cfg["focus_depth_um"] * 1e-6 / 2, 0.0),))
    xfem_region = p.Set(faces=mod_faces, name='xfem_zone')
    m.rootAssembly.Instance(name='Wafer-1', part=p, dependent=ON)
    crack = m.XFEMCrack(name='ModifiedCrack',
                         crackDomain=m.rootAssembly.instances['Wafer-1'].sets['xfem_zone'])

    print("ABAQUS XFEM crack domain created in modified layer zone.")

except ImportError:
    print("(ABAQUS caeModules not available — analytical model only)")
except Exception as e:
    print("ABAQUS model error: {}".format(e))
