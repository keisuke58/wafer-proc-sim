"""Check all history regions for any non-zero values."""
from odbAccess import openOdb
odb = openOdb("/home/nishioka/git/wafer-proc-sim/runs/parametric_sic/dicing_4HSiC_d020_bw20.odb", readOnly=True)

plunge = odb.steps["Plunge"]
print("=== ALL history regions in Plunge ===")
for rname in plunge.historyRegions.keys():
    reg = plunge.historyRegions[rname]
    for k in reg.historyOutputs.keys():
        data = reg.historyOutputs[k].data
        mx = max(abs(v[1]) for v in data) if data else 0
        if mx > 0:
            print("  [%s][%s] max=%.4g n=%d" % (rname[:30], k, mx, len(data)))

print("\n=== ALL history regions in Feed ===")
feed = odb.steps["Feed"]
for rname in feed.historyRegions.keys():
    reg = feed.historyRegions[rname]
    for k in reg.historyOutputs.keys():
        data = reg.historyOutputs[k].data
        mx = max(abs(v[1]) for v in data) if data else 0
        if mx > 0:
            print("  [%s][%s] max=%.4g" % (rname[:30], k, mx))

# Check the INP contact pair is in Plunge step (not initial)
print("\n=== Contact check from INP surface definition ===")
print("  BladeSurf surface: %s" % str(list(odb.rootAssembly.surfaces.keys())))

odb.close()
