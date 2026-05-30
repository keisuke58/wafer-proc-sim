"""Check contact forces and wafer top node displacements."""
from odbAccess import openOdb

ODB = "/home/nishioka/git/wafer-proc-sim/runs/parametric_sic/dicing_4HSiC_d020_bw20.odb"
odb = openOdb(path=ODB, readOnly=True)

def get_data(v):
    try:
        return v.data
    except Exception:
        return v.dataDouble

# ── RF at blade RP (Plunge history) ───────────────────────────────────────
print("=== RF1 RF2 at blade RP (Plunge, max over all time) ===")
plunge = odb.steps["Plunge"]
for k in ["RF1", "RF2", "U1", "U2"]:
    hr = plunge.historyRegions["Node BLADE-1.17"] if "Node BLADE-1.17" in plunge.historyRegions.keys() else None
    if hr and k in hr.historyOutputs.keys():
        data = hr.historyOutputs[k].data
        mx = max(abs(v[1]) for v in data) if data else 0
        t_end, v_end = (data[-1][0], data[-1][1]) if data else (0, 0)
        print("  %s: max_abs=%.6g  at_end=(t=%.4g, v=%.6g)" % (k, mx, t_end, v_end))

# ── Wafer top-edge nodes: displacement in last Plunge frame ───────────────
print("\n=== WAFER TOP NODES U (last Plunge frame) ===")
last_plunge = plunge.frames[-1]
wafer_inst = None
for iname in odb.rootAssembly.instances.keys():
    if "WAFER" in iname.upper():
        wafer_inst = odb.rootAssembly.instances[iname]
        break

H = 200e-6
if wafer_inst and "U" in last_plunge.fieldOutputs.keys():
    sub = last_plunge.fieldOutputs["U"].getSubset(region=wafer_inst)
    top_nodes = []
    for v in sub.values:
        node = wafer_inst.getNodeFromLabel(v.nodeLabel)
        if abs(node.coordinates[1] - H) < 1e-7:
            try:
                u = list(get_data(v))
            except Exception:
                u = ["err"]
            top_nodes.append((v.nodeLabel, node.coordinates[0], u))
    top_nodes.sort(key=lambda x: x[1])
    max_u2 = max(abs(n[2][1]) for n in top_nodes if len(n[2]) > 1 and isinstance(n[2][1], float)) if top_nodes else 0
    print("  Total top nodes: %d, max |U2| = %.4g m" % (len(top_nodes), max_u2))
    print("  nodeLabel  x[m]          U=[u1,u2]")
    for nl, x, u in top_nodes[:10]:
        print("    %6d   %.4e   %s" % (nl, x, u))

# ── Wafer top-edge nodes U: last Feed frame ─────────────────────────────
print("\n=== WAFER TOP NODES U (last Feed frame) ===")
last_feed = odb.steps["Feed"].frames[-1]
if wafer_inst and "U" in last_feed.fieldOutputs.keys():
    sub2 = last_feed.fieldOutputs["U"].getSubset(region=wafer_inst)
    top_nodes2 = []
    for v in sub2.values:
        node = wafer_inst.getNodeFromLabel(v.nodeLabel)
        if abs(node.coordinates[1] - H) < 1e-7:
            try:
                u = list(get_data(v))
            except Exception:
                u = ["err"]
            top_nodes2.append((v.nodeLabel, node.coordinates[0], u))
    top_nodes2.sort(key=lambda x: x[1])
    max_u2_feed = max(abs(n[2][1]) for n in top_nodes2 if len(n[2]) > 1 and isinstance(n[2][1], float)) if top_nodes2 else 0
    print("  max |U2| = %.4g m" % max_u2_feed)
    # Show only nodes with non-trivial displacement
    for nl, x, u in top_nodes2:
        if len(u) > 1 and isinstance(u[1], float) and abs(u[1]) > 1e-8:
            print("    node%d x=%.4e U=%s" % (nl, x, u))

# ── Check STATUS of all wafer elements in last Feed frame ────────────────
print("\n=== WAFER ELEMENT STATUS (last Feed frame) ===")
if wafer_inst and "STATUS" in last_feed.fieldOutputs.keys():
    sub_s = last_feed.fieldOutputs["STATUS"].getSubset(region=wafer_inst)
    cnt = {1.0: 0, 0.0: 0}
    for v in sub_s.values:
        try:
            d = float(get_data(v))
        except Exception:
            d = -1.0
        cnt[d] = cnt.get(d, 0) + 1
    for k2, n in sorted(cnt.items()):
        label = "INTACT" if k2 == 1.0 else "DELETED" if k2 == 0.0 else "other"
        print("  STATUS=%.1f (%s): %d elements" % (k2, label, n))
else:
    print("  STATUS field not found")

odb.close()
