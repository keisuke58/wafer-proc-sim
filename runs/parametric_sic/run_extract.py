"""Extraction runner — reads config to avoid ABAQUS sys.argv stripping."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'fem'))

# Hardcode paths for this run
MANIFEST  = os.path.join(os.path.dirname(__file__), 'jobs_partial.json')
ODB_DIR   = os.path.dirname(__file__)
RESULTS   = os.path.join(os.path.dirname(__file__), '..', '..', 'results')

# Inline extract logic (avoids argparse issue)
from odbAccess import openOdb
import numpy as np, csv

os.makedirs(RESULTS, exist_ok=True)
jobs = json.load(open(MANIFEST))
summary = []

for job_info in jobs:
    job  = job_info["job"]
    odb_path = os.path.join(ODB_DIR, job + ".odb")
    if not os.path.exists(odb_path):
        print("[!] Missing: " + odb_path)
        continue

    print("[->] " + job)
    odb = openOdb(path=odb_path, readOnly=True)

    # ── Cutting forces from history ──────────────────────────────────────────
    rf1_max, rf2_max = 0.0, 0.0
    for step_name in odb.steps.keys():
        step = odb.steps[step_name]
        for rname in step.historyRegions.keys():
            reg = step.historyRegions[rname]
            ho  = reg.historyOutputs
            if "RF1" in ho.keys():
                vals1 = [abs(v[1]) for v in ho["RF1"].data]
                vals2 = [abs(v[1]) for v in ho["RF2"].data] if "RF2" in ho.keys() else [0]
                if vals1: rf1_max = max(rf1_max, max(vals1))
                if vals2: rf2_max = max(rf2_max, max(vals2))

    # ── Chipping from STATUS in last frame ───────────────────────────────────
    last_step  = odb.steps[list(odb.steps.keys())[-1]]
    last_frame = last_step.frames[-1]
    deleted, total = 0, 0
    max_peeq = 0.0

    fo_keys   = last_frame.fieldOutputs.keys()
    status_fo = last_frame.fieldOutputs["STATUS"] if "STATUS" in fo_keys else None
    peeq_fo   = last_frame.fieldOutputs["PEEQ"]   if "PEEQ"   in fo_keys else None

    # ODB stores instance names as uppercase (WAFER-1, BLADE-1).
    # Always get wafer instance explicitly; fall back to first deformable instance.
    def _wafer_inst(assembly):
        for key in ("WAFER-1", "Wafer-1"):
            if key in assembly.instances.keys():
                return assembly.instances[key]
        for inst in assembly.instances.values():
            if inst.type != "RIGID_BODY":
                return inst
        return list(assembly.instances.values())[0]

    if status_fo:
        inst = _wafer_inst(odb.rootAssembly)
        try:
            vals = status_fo.getSubset(region=inst).values
            arr  = [v.data for v in vals]
            total   = len(arr)
            deleted = sum(1 for v in arr if v == 0.0)
        except Exception as e:
            print("  STATUS error: " + str(e))

    if peeq_fo:
        try:
            inst = _wafer_inst(odb.rootAssembly)
            vals = peeq_fo.getSubset(region=inst).values
            peeq_arr = [v.data for v in vals]
            if peeq_arr:
                max_peeq = float(max(peeq_arr))
        except Exception as e:
            print("  PEEQ error: " + str(e))

    odb.close()

    del_frac = deleted / total if total > 0 else 0.0
    row = {
        "material":              job_info.get("material", "4HSiC"),
        "cut_depth_um":          job_info["cut_depth_um"],
        "blade_W_um":            job_info["blade_W_um"],
        "job":                   job,
        "deletion_fraction":     round(del_frac, 6),
        "deleted_elements":      deleted,
        "total_elements":        total,
        "max_RF1_N":             round(rf1_max, 4),
        "max_RF2_N":             round(rf2_max, 4),
        "max_PEEQ":              round(max_peeq, 6),
    }
    summary.append(row)
    print("  chip=%.4f  RF1=%.2fN  RF2=%.2fN  PEEQ=%.4f" % (
        del_frac, rf1_max, rf2_max, max_peeq))

# Write CSV
out_csv = os.path.join(RESULTS, "parametric_summary.csv")
if summary:
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    print("\n[OK] parametric_summary.csv -> " + out_csv)
    print("Rows: " + str(len(summary)))
