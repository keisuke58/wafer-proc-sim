"""Debug: show what's inside one ODB."""
import os
from odbAccess import openOdb

ODB = "/home/nishioka/git/wafer-proc-sim/runs/parametric_sic/dicing_4HSiC_d030_bw30.odb"
odb = openOdb(path=ODB, readOnly=True)

print("=== STEPS ===")
for sname in odb.steps.keys():
    step = odb.steps[sname]
    print("  Step:", sname)
    print("  History regions:", list(step.historyRegions.keys())[:5])
    for rname in list(step.historyRegions.keys())[:3]:
        reg = step.historyRegions[rname]
        print("    Region:", rname, "-> outputs:", list(reg.historyOutputs.keys())[:5])
        for hname in list(reg.historyOutputs.keys())[:3]:
            data = reg.historyOutputs[hname].data
            vals = [v[1] for v in data]
            nz   = [v for v in vals if abs(v) > 1e-10]
            print("      %s: %d pts, non-zero=%d, max=%.4g" % (
                hname, len(vals), len(nz), max(abs(v) for v in vals) if vals else 0))

print("\n=== LAST FRAME FIELD OUTPUTS ===")
last = odb.steps[list(odb.steps.keys())[-1]].frames[-1]
print("  Fields:", list(last.fieldOutputs.keys()))

if "STATUS" in last.fieldOutputs.keys():
    sf = last.fieldOutputs["STATUS"]
    inst = list(odb.rootAssembly.instances.values())[0]
    print("  STATUS values (first 5):", [v.data for v in sf.values[:5]])
    try:
        sub = sf.getSubset(region=inst)
        vals = [v.data for v in sub.values]
        print("  STATUS subset len:", len(vals), "zeros:", sum(1 for v in vals if v==0))
    except Exception as e:
        print("  getSubset error:", e)

if "PEEQ" in last.fieldOutputs.keys():
    pf   = last.fieldOutputs["PEEQ"]
    vals = [v.data for v in pf.values[:10]]
    print("  PEEQ (first 10):", vals)

odb.close()
