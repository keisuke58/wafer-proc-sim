"""Check minimal contact test ODB for contact forces."""
from odbAccess import openOdb

odb = openOdb("/home/nishioka/git/wafer-proc-sim/runs/parametric_sic/test_contact_minimal.odb", readOnly=True)
plunge = odb.steps["Plunge"]

print("=== TOOL RP HISTORY ===")
for rname in plunge.historyRegions.keys():
    reg = plunge.historyRegions[rname]
    for k in reg.historyOutputs.keys():
        data = reg.historyOutputs[k].data
        mx = max(abs(v[1]) for v in data) if data else 0
        if mx > 0:
            last5 = [(data[i][0], data[i][1]) for i in range(max(0,len(data)-3), len(data))]
            print("  [%s][%s] max=%.4g  last3=%s" % (rname[:20], k, mx, last5))

print("\n=== BLOCK TOP NODE U (last frame) ===")
last = plunge.frames[-1]
block_inst = odb.rootAssembly.instances["BLOCK-1"]
if "U" in last.fieldOutputs.keys():
    sub = last.fieldOutputs["U"].getSubset(region=block_inst)
    H = 200e-6
    for v in sub.values:
        node = block_inst.getNodeFromLabel(v.nodeLabel)
        if abs(node.coordinates[1] - H) < 1e-7:
            try:
                print("  node%d x=%.4e U=%s" % (v.nodeLabel, node.coordinates[0], list(v.data)))
            except:
                print("  node%d x=%.4e U=%s" % (v.nodeLabel, node.coordinates[0], list(v.dataDouble)))

print("\n=== KE at t=5-6 μs (contact zone) ===")
for rname in plunge.historyRegions.keys():
    reg = plunge.historyRegions[rname]
    if "ALLKE" in reg.historyOutputs.keys():
        data = reg.historyOutputs["ALLKE"].data
        for t, v in data:
            if 4.5e-6 <= t <= 6.5e-6:
                print("  t=%.4g KE=%.4g" % (t, v))

odb.close()
