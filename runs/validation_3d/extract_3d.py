"""
Extract results from 3D validation FEM jobs.
Compare with 2D surrogate predictions.

Run with: abaqus python extract_3d.py
"""
import os, sys, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from odbAccess import openOdb

DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, '..', '..', 'results')

JOBS = [
    {"job": "dicing_3d_d150_bw23", "cut_depth_um": 150, "blade_W_um": 23},
    {"job": "dicing_3d_d300_bw23", "cut_depth_um": 300, "blade_W_um": 23},
]

# Load 2D GP for comparison
sys.path.insert(0, os.path.join(DIR, '..', '..'))
try:
    from ml.train_from_experimental import ExperimentalGPSurrogate, MODEL_PATH
    import numpy as np
    gp = ExperimentalGPSurrogate.load(MODEL_PATH)
    def gp_pred(d, bw, feed=0.5, spindle=30000):
        x = np.array([[d, bw, feed, spindle]])
        mu, sig = gp.predict(x, return_std=True)
        return float(mu[0]), float(sig[0])
    HAS_GP = True
except Exception as e:
    print(f"[Warn] GP not loaded: {e}")
    HAS_GP = False

summary = []
for job_info in JOBS:
    job = job_info["job"]
    odb_path = os.path.join(DIR, job + "_run.odb")
    if not os.path.exists(odb_path):
        print(f"[!] Missing ODB: {odb_path}")
        continue

    print(f"[->] {job}")
    odb = openOdb(path=odb_path, readOnly=True)

    last_step = odb.steps[list(odb.steps.keys())[-1]]
    last_frame = last_step.frames[-1]
    fouts = last_frame.fieldOutputs.keys()

    deleted = total = 0
    rf1_max = rf2_max = 0.0

    if "STATUS" in fouts:
        for inst in odb.rootAssembly.instances.values():
            if inst.type == "RIGID_BODY":
                continue
            try:
                for v in last_frame.fieldOutputs["STATUS"].getSubset(region=inst).values:
                    total += 1
                    if v.data == 0.0:
                        deleted += 1
            except Exception:
                pass

    for sname in odb.steps.keys():
        step = odb.steps[sname]
        for rname in step.historyRegions.keys():
            reg = step.historyRegions[rname]
            ho  = reg.historyOutputs
            for key, attr in (("RF1", "rf1_max"), ("RF2", "rf2_max")):
                if key in ho.keys():
                    for v in ho[key].data:
                        val = abs(v[1])
                        if key == "RF1" and val > rf1_max: rf1_max = val
                        if key == "RF2" and val > rf2_max: rf2_max = val

    odb.close()
    del_frac = deleted / total if total > 0 else 0.0
    row = {
        "model": "3D",
        "cut_depth_um": job_info["cut_depth_um"],
        "blade_W_um":   job_info["blade_W_um"],
        "deletion_fraction": round(del_frac, 6),
        "deleted_elements":  deleted,
        "total_elements":    total,
        "max_RF2_N":         round(rf2_max, 4),
    }
    if HAS_GP:
        mu, sig = gp_pred(job_info["cut_depth_um"], job_info["blade_W_um"])
        row["gp_chip_pred_um"] = round(mu, 3)
        row["gp_chip_std_um"]  = round(sig, 3)
    summary.append(row)
    print(f"  del_frac={del_frac:.4f} ({deleted}/{total})  RF2={rf2_max:.1f}N")
    if HAS_GP:
        print(f"  GP chipping pred: {mu:.1f}±{sig:.1f} µm")

if summary:
    out = os.path.join(RESULTS, "parametric_summary_3d.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader(); w.writerows(summary)
    print(f"\n[OK] -> {out}  ({len(summary)} rows)")

    # Compare 2D vs 3D
    print("\n=== 2D vs 3D Comparison ===")
    try:
        import pandas as pd
        df2d = pd.read_csv(os.path.join(RESULTS, "parametric_summary_extended.csv"))
        print(f"  {'depth':>6}  {'2D del_frac':>12}  {'3D del_frac':>12}  {'2D RF2(kN)':>11}  {'3D RF2(kN)':>11}")
        for row in summary:
            d = row["cut_depth_um"]
            r2d = df2d[df2d["cut_depth_um"]==d]
            if not r2d.empty:
                df2d_row = r2d.iloc[0]
                print(f"  {d:>6}µm  {df2d_row['deletion_fraction']:>12.4f}  "
                      f"{row['deletion_fraction']:>12.4f}  "
                      f"{df2d_row['max_RF2_N']/1e3:>11.1f}  "
                      f"{row['max_RF2_N']/1e3:>11.1f}")
    except Exception as e:
        print(f"  Comparison error: {e}")
