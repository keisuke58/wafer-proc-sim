"""
Post-processor for dicing FEM results (Sunday use).
Reads ABAQUS .odb files, extracts:
  - Cutting force (RF1) and thrust force (RF2) vs time
  - Max principal stress field at final frame
  - Chipping size estimate from deleted elements (STATUS=0)

Usage (inside ABAQUS Python):
    abaqus python extract_results.py -- --job dicing_4HSiC_d050_bw30
    abaqus python extract_results.py -- --manifest jobs_4HSiC.json

Output:
    results/<job_name>/forces.csv
    results/<job_name>/chipping_stats.csv
    results/<job_name>/stress_field.npy
"""

import sys
import os
import json
import argparse
import numpy as np
import csv

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def extract_job(job_name: str, odb_dir: str = "."):
    """Extract forces, chipping stats, and stress field from one .odb file."""
    from odbAccess import openOdb
    from abaqusConstants import INTEGRATION_POINT, ELEMENT_NODAL

    odb_path = os.path.join(odb_dir, job_name + ".odb")
    if not os.path.exists(odb_path):
        print(f"[!] ODB not found: {odb_path}")
        return None

    out_dir = os.path.join(RESULTS_DIR, job_name)
    os.makedirs(out_dir, exist_ok=True)

    odb = openOdb(path=odb_path, readOnly=True)
    assembly = odb.rootAssembly

    # ── Blade forces from history output ─────────────────────────────────────
    force_rows = []
    for step_name in odb.steps.keys():
        step = odb.steps[step_name]
        for region_name in step.historyRegions.keys():
            region = step.historyRegions[region_name]
            rf1_data = region.historyOutputs.get("RF1")
            rf2_data = region.historyOutputs.get("RF2")
            if rf1_data is None:
                continue
            for (t, f1), (_, f2) in zip(rf1_data.data, rf2_data.data):
                force_rows.append({
                    "step": step_name, "time": t,
                    "RF1_N": f1, "RF2_N": f2})

    forces_path = os.path.join(out_dir, "forces.csv")
    if force_rows:
        with open(forces_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=force_rows[0].keys())
            writer.writeheader()
            writer.writerows(force_rows)
        print(f"  [✓] Forces → {forces_path}")
    else:
        print(f"  [!] No history force data found in {job_name}")

    # ── Chipping estimate from last frame: count STATUS=0 elements ────────────
    last_step = odb.steps[list(odb.steps.keys())[-1]]
    last_frame = last_step.frames[-1]

    # STATUS field (0 = deleted, 1 = active)
    status_field = None
    stress_field = None
    for fo in last_frame.fieldOutputs.values():
        if fo.name == "STATUS":
            status_field = fo
        if fo.name == "S":
            stress_field = fo

    chip_stats = {"job": job_name, "deleted_elements": 0,
                  "total_elements": 0, "deletion_fraction": 0.0,
                  "max_principal_stress_Pa": float("nan")}

    if status_field is not None:
        inst = assembly.instances.get("WAFER-1") or \
               list(assembly.instances.values())[0]
        status_on_inst = status_field.getSubset(region=inst)
        values = np.array([v.data for v in status_on_inst.values])
        total   = len(values)
        deleted = int(np.sum(values == 0.0))
        chip_stats["total_elements"]    = total
        chip_stats["deleted_elements"]  = deleted
        chip_stats["deletion_fraction"] = deleted / total if total > 0 else 0.0
        print(f"  [✓] Deleted elements: {deleted}/{total} "
              f"({chip_stats['deletion_fraction']*100:.2f}%)")

    # ── Max principal stress in last frame ────────────────────────────────────
    if stress_field is not None:
        # S.maxPrincipal for each integration point
        stress_values = []
        for v in stress_field.values:
            # v.maxPrincipal available if stress invariants computed
            if hasattr(v, "maxPrincipal"):
                stress_values.append(v.maxPrincipal)
            else:
                # Fallback: compute from S11, S22, S12
                s11, s22, s12 = v.data[0], v.data[1], v.data[3]
                sm = (s11 + s22) / 2
                r  = np.sqrt(((s11 - s22) / 2) ** 2 + s12 ** 2)
                stress_values.append(sm + r)
        if stress_values:
            max_sp = float(np.max(stress_values))
            chip_stats["max_principal_stress_Pa"] = max_sp
            print(f"  [✓] Max principal stress: {max_sp/1e9:.3f} GPa")

            # Save stress field array for surrogate model (Sunday ML)
            sp_arr = np.array(stress_values)
            np.save(os.path.join(out_dir, "stress_maxprincipal.npy"), sp_arr)

    # Save chipping stats
    stats_path = os.path.join(out_dir, "chipping_stats.csv")
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=chip_stats.keys())
        writer.writeheader()
        writer.writerow(chip_stats)
    print(f"  [✓] Stats → {stats_path}")

    odb.close()
    return chip_stats


def extract_all_from_manifest(manifest_path: str, odb_dir: str = "."):
    """Process all jobs listed in a JSON manifest (output of parametric_study)."""
    with open(manifest_path) as f:
        jobs = json.load(f)

    summary = []
    for job_info in jobs:
        job_name = job_info["job"]
        print(f"\n[→] Extracting: {job_name}")
        stats = extract_job(job_name, odb_dir=odb_dir)
        if stats is not None:
            row = {**job_info, **stats}
            summary.append(row)

    # Master summary CSV (input for surrogate model)
    summary_path = os.path.join(RESULTS_DIR, "parametric_summary.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if summary:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
        print(f"\n[✓] Master summary → {summary_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job",      help="Single job name (no .odb extension)")
    group.add_argument("--manifest", help="Path to jobs_*.json manifest")
    parser.add_argument("--odb-dir", default=".",
                        help="Directory containing .odb files")
    args = parser.parse_args()

    if args.job:
        extract_job(args.job, odb_dir=args.odb_dir)
    else:
        extract_all_from_manifest(args.manifest, odb_dir=args.odb_dir)
