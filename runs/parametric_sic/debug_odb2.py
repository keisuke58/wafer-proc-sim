"""Deep debug: check blade displacement and wafer deformation."""
from odbAccess import openOdb
import os

ODB = "/home/nishioka/git/wafer-proc-sim/runs/parametric_sic/dicing_4HSiC_d030_bw30.odb"
odb = openOdb(path=ODB, readOnly=True)

# ── Blade U2 in Feed step ──────────────────────────────────────────────────────
print("=== BLADE DISPLACEMENTS (Feed step) ===")
feed = odb.steps["Feed"]
for rname in feed.historyRegions.keys():
    reg = feed.historyRegions[rname]
    if "U2" in reg.historyOutputs.keys():
        u2 = reg.historyOutputs["U2"].data
        u2_vals = [v[1] for v in u2]
        print("Region:", rname)
        print("  U2 first 5:", u2_vals[:5])
        print("  U2 last  5:", u2_vals[-5:])
        print("  U2 max:", max(abs(v) for v in u2_vals) if u2_vals else "empty")
    if "U1" in reg.historyOutputs.keys():
        u1 = reg.historyOutputs["U1"].data
        u1_vals = [v[1] for v in u1]
        if max(abs(v) for v in u1_vals) > 1e-10:
            print("  U1 non-zero! max =", max(abs(v) for v in u1_vals))
        else:
            print("  U1 = 0 throughout")

# ── Energy in entire model ─────────────────────────────────────────────────────
print("\n=== MODEL ENERGY (Plunge step) ===")
plunge = odb.steps["Plunge"]
for rname in plunge.historyRegions.keys():
    reg = plunge.historyRegions[rname]
    if "ALLIE" in reg.historyOutputs.keys():
        allie = reg.historyOutputs["ALLIE"].data
        vals  = [v[1] for v in allie]
        print("ALLIE max:", max(abs(v) for v in vals) if vals else 0)
    if "ALLKE" in reg.historyOutputs.keys():
        allke = reg.historyOutputs["ALLKE"].data
        vals  = [v[1] for v in allke]
        print("ALLKE max:", max(abs(v) for v in vals) if vals else 0)
    # All available outputs
    keys = list(reg.historyOutputs.keys())
    if len(keys) > 3:
        print("All history outputs:", keys)

# ── Wafer displacement in last frame ──────────────────────────────────────────
print("\n=== WAFER DISPLACEMENT LAST FRAME ===")
last_frame = odb.steps["Feed"].frames[-1]
if "U" in last_frame.fieldOutputs.keys():
    u_fo   = last_frame.fieldOutputs["U"]
    inst   = list(odb.rootAssembly.instances.values())[0]
    u_vals = u_fo.getSubset(region=inst).values
    u_mags = [(v.data[0]**2 + v.data[1]**2)**0.5 for v in u_vals]
    print("Max displacement magnitude:", max(u_mags) if u_mags else 0, "m")
    print("Non-zero disp nodes:", sum(1 for v in u_mags if v > 1e-10))

odb.close()
