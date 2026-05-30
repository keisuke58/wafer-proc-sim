"""Extract results from extended FEM runs and update parametric_summary_extended.csv."""
import sys, os, csv, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from odbAccess import openOdb

ODB_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS  = os.path.join(ODB_DIR, '..', '..', 'results')
MANIFEST = os.path.join(ODB_DIR, 'jobs_4HSiC.json')

jobs = json.load(open(MANIFEST))
summary = []

def wafer_inst(assembly):
    for key in ("WAFER-1", "Wafer-1"):
        if key in assembly.instances.keys():
            return assembly.instances[key]
    for inst in assembly.instances.values():
        if inst.type != "RIGID_BODY":
            return inst
    return list(assembly.instances.values())[0]

for job_info in jobs:
    job = job_info["job"]
    odb_path = os.path.join(ODB_DIR, job + "_run.odb")
    if not os.path.exists(odb_path):
        print("[!] Missing: " + odb_path)
        continue

    print("[->] " + job)
    odb = openOdb(path=odb_path, readOnly=True)

    # Forces from history
    rf1_max, rf2_max = 0.0, 0.0
    for sname in odb.steps.keys():
        step = odb.steps[sname]
        for rname in step.historyRegions.keys():
            reg = step.historyRegions[rname]
            ho  = reg.historyOutputs
            for key, attr in (("RF1", "rf1_max"), ("RF2", "rf2_max")):
                if key in ho.keys() and ho[key].data:
                    for v in ho[key].data:
                        val = abs(v[1])
                        if key == "RF1" and val > rf1_max: rf1_max = val
                        if key == "RF2" and val > rf2_max: rf2_max = val

    # Chipping from STATUS
    last_frame = odb.steps[list(odb.steps.keys())[-1]].frames[-1]
    fields = last_frame.fieldOutputs.keys()
    deleted, total, max_peeq = 0, 0, 0.0

    if "STATUS" in fields:
        inst = wafer_inst(odb.rootAssembly)
        try:
            sub = last_frame.fieldOutputs["STATUS"].getSubset(region=inst)
            for v in sub.values:
                total += 1
                if v.data == 0.0: deleted += 1
        except Exception as e:
            print("  STATUS error: " + str(e))

    if "PEEQ" in fields:
        inst = wafer_inst(odb.rootAssembly)
        try:
            sub = last_frame.fieldOutputs["PEEQ"].getSubset(region=inst)
            for v in sub.values:
                if v.data > max_peeq: max_peeq = v.data
        except Exception as e:
            print("  PEEQ error: " + str(e))

    odb.close()

    del_frac = deleted / total if total > 0 else 0.0
    row = {
        "material":          job_info.get("material", "4HSiC"),
        "cut_depth_um":      job_info["cut_depth_um"],
        "blade_W_um":        job_info["blade_W_um"],
        "job":               job,
        "deletion_fraction": round(del_frac, 6),
        "deleted_elements":  deleted,
        "total_elements":    total,
        "max_RF1_N":         round(rf1_max, 4),
        "max_RF2_N":         round(rf2_max, 4),
        "max_PEEQ":          round(max_peeq, 6),
    }
    summary.append(row)
    print("  chip=%.4f (%d/%d)  RF1=%.2fN  RF2=%.2fN  PEEQ=%.4f" % (
        del_frac, deleted, total, rf1_max, rf2_max, max_peeq))

os.makedirs(RESULTS, exist_ok=True)
out = os.path.join(RESULTS, "parametric_summary_extended.csv")
if summary:
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    print("\n[OK] -> " + out + "  (%d rows)" % len(summary))
