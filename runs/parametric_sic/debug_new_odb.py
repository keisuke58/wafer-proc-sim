"""Check new-code ODB: history + field outputs + U2 direct check."""
from odbAccess import openOdb

ODB = "/home/nishioka/git/wafer-proc-sim/runs/parametric_sic/dicing_4HSiC_d020_bw20.odb"
odb = openOdb(path=ODB, readOnly=True)

# ── history: all non-zero ──────────────────────────────────────────────────
print("=== HISTORY: all non-zero values ===")
for sname in odb.steps.keys():
    step = odb.steps[sname]
    for rname in step.historyRegions.keys():
        reg = step.historyRegions[rname]
        for k in reg.historyOutputs.keys():
            vals = [v[1] for v in reg.historyOutputs[k].data]
            mx = max(abs(v) for v in vals) if vals else 0
            if mx > 1e-15:
                print("  [%s][%s][%s] max=%.4g n=%d" % (
                    sname[:6], rname[:22], k, mx, len(vals)))

# ── U2 of blade RP: first 10 non-zero or all values max ───────────────────
print("\n=== U2 BLADE RP detail ===")
for sname in odb.steps.keys():
    step = odb.steps[sname]
    for rname in step.historyRegions.keys():
        reg = step.historyRegions[rname]
        if "U2" in reg.historyOutputs.keys():
            data = reg.historyOutputs["U2"].data
            vals = [v[1] for v in data]
            mx = max(abs(v) for v in vals) if vals else 0
            first = [(data[i][0], data[i][1]) for i in range(min(3, len(data)))]
            last  = [(data[i][0], data[i][1]) for i in range(max(0, len(data)-3), len(data))]
            print("  [%s][%s] max_abs=%.4g  first3=%s  last3=%s" % (
                sname[:6], rname[:22], mx, first, last))

# ── blade RP node coordinates: frame 0 vs last frame ─────────────────────
print("\n=== BLADE RP NODE POSITION (coordinates + displacement) ===")
all_steps = list(odb.steps.keys())
first_frame = odb.steps[all_steps[0]].frames[0]
last_frame  = odb.steps[all_steps[-1]].frames[-1]

blade_inst = None
for iname in odb.rootAssembly.instances.keys():
    if "BLADE" in iname.upper():
        blade_inst = odb.rootAssembly.instances[iname]
        print("  Blade instance: %s" % iname)
        break

if blade_inst is not None:
    # Print node 17 position (RP) from initial coords
    for node in blade_inst.nodes:
        if node.label == 17:
            print("  RP initial coord: %s" % str(node.coordinates))
            break

    # U in first frame and last frame for blade RP
    for label, fname, frame in [("FIRST", all_steps[0], first_frame),
                                 ("LAST",  all_steps[-1], last_frame)]:
        if "U" in frame.fieldOutputs.keys():
            fo = frame.fieldOutputs["U"]
            sub = fo.getSubset(region=blade_inst)
            for i, v in enumerate(sub.values):
                if v.nodeLabel == 17:
                    try:
                        print("  [%s] RP node17 U=%s (stepTime=%.4g)" % (
                            label, v.data, frame.frameValue))
                    except Exception:
                        print("  [%s] RP node17 U=%s (stepTime=%.4g)" % (
                            label, v.dataDouble, frame.frameValue))
                    break
                if i > 50:
                    break

# ── last-frame wafer displacement ─────────────────────────────────────────
print("\n=== WAFER U (last frame, first 5 nodes) ===")
wafer_inst = None
for iname in odb.rootAssembly.instances.keys():
    if "WAFER" in iname.upper():
        wafer_inst = odb.rootAssembly.instances[iname]
        break

if wafer_inst and "U" in last_frame.fieldOutputs.keys():
    sub = last_frame.fieldOutputs["U"].getSubset(region=wafer_inst)
    for i, v in enumerate(sub.values):
        if i >= 5:
            break
        try:
            print("  node%d U=%s" % (v.nodeLabel, v.data))
        except Exception:
            print("  node%d U=%s" % (v.nodeLabel, v.dataDouble))

# ── step durations / frame count ──────────────────────────────────────────
print("\n=== STEP DURATIONS ===")
for sname in odb.steps.keys():
    dur = odb.steps[sname].timePeriod
    nf  = len(odb.steps[sname].frames)
    print("  %s: %.3g s  (%d frames)" % (sname, dur, nf))

odb.close()
