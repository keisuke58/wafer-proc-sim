"""
Post-processor for grinding FEM results.
Reads ABAQUS .odb files from grinding_warpage_2d.py, extracts:
  - warpage_um    : peak-to-valley of U2 on top surface after Release step
  - max_residual_stress_MPa : max S22 (axial stress) after Release step

Usage (inside ABAQUS Python):
    abaqus python extract_grinding_results.py -- --job grind_Si_ws4000_f300_d020
    abaqus python extract_grinding_results.py -- --manifest jobs_grind_Si.json

Output:
    results/<job_name>/grinding_stats.csv
    results/parametric_summary_grinding.csv  (master summary for surrogate)
"""

import sys
import os
import json
import argparse
import numpy as np
import csv

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def extract_job(job_name: str, odb_dir: str = "."):
    """Extract warpage and residual stress from one grinding .odb file."""
    from odbAccess import openOdb
    from abaqusConstants import NODAL

    odb_path = os.path.join(odb_dir, job_name + ".odb")
    if not os.path.exists(odb_path):
        print("[!] ODB not found: %s" % odb_path)
        return None

    out_dir = os.path.join(RESULTS_DIR, job_name)
    os.makedirs(out_dir, exist_ok=True)

    odb  = openOdb(path=odb_path, readOnly=True)
    step_names = list(odb.steps.keys())
    print("  [*] Steps: %s" % step_names)

    stats = {
        "job":                       job_name,
        "warpage_um":                float("nan"),
        "max_residual_stress_MPa":   float("nan"),
    }

    # ── Warpage from Release step (last frame) ────────────────────────────────
    release_step = None
    for sn in ("Release", "release"):
        if sn in odb.steps:
            release_step = odb.steps[sn]
            break
    if release_step is None:
        release_step = odb.steps[step_names[-1]]
        print("  [!] 'Release' step not found — using last step: %s" % step_names[-1])

    last_frame = release_step.frames[-1]
    u_field    = last_frame.fieldOutputs.get("U")
    s_field    = last_frame.fieldOutputs.get("S")

    if u_field is not None:
        inst = None
        for k in odb.rootAssembly.instances.keys():
            if "WAFER" in k.upper():
                inst = odb.rootAssembly.instances[k]
                break
        if inst is None:
            inst = list(odb.rootAssembly.instances.values())[0]

        # Extract U2 (axial displacement) at all nodes on top surface
        # Top surface = nodes with max z-coordinate
        all_nodes   = inst.nodes
        z_coords    = np.array([n.coordinates[1] for n in all_nodes])
        z_top       = z_coords.max()
        top_mask    = np.abs(z_coords - z_top) < (z_top * 0.005 + 1e-7)

        u_on_inst = u_field.getSubset(region=inst, position=NODAL)
        u2_all    = np.array([v.data[1] for v in u_on_inst.values])
        u2_top    = u2_all[top_mask]

        if len(u2_top) > 0:
            warpage_m  = float(u2_top.max() - u2_top.min())
            stats["warpage_um"] = warpage_m * 1e6
            print("  [✓] Warpage: %.2f µm  (top-surface U2 peak-to-valley)" % stats["warpage_um"])
            np.save(os.path.join(out_dir, "u2_top.npy"), u2_top)
        else:
            print("  [!] No top-surface nodes found for warpage extraction")

    # ── Max axial residual stress S22 after release ───────────────────────────
    if s_field is not None:
        s22_vals = []
        for v in s_field.values:
            if len(v.data) >= 2:
                s22_vals.append(v.data[1])  # S22 = axial component in axisymmetric
        if s22_vals:
            max_s22_pa = float(np.max(np.abs(s22_vals)))
            stats["max_residual_stress_MPa"] = max_s22_pa / 1e6
            print("  [✓] Max |S22|: %.1f MPa" % stats["max_residual_stress_MPa"])

    # ── Write per-job CSV ──────────────────────────────────────────────────────
    stats_path = os.path.join(out_dir, "grinding_stats.csv")
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        writer.writeheader()
        writer.writerow(stats)
    print("  [✓] Stats → %s" % stats_path)

    odb.close()
    return stats


def extract_all_from_manifest(manifest_path: str, odb_dir: str = "."):
    """Process all jobs in a jobs_grind_*.json manifest."""
    with open(manifest_path) as f:
        jobs = json.load(f)

    summary = []
    for job_info in jobs:
        job_name = job_info["job"]
        print("\n[→] Extracting: %s" % job_name)
        stats = extract_job(job_name, odb_dir=odb_dir)
        if stats is not None:
            row = {}
            row.update(job_info)
            row.update(stats)
            summary.append(row)

    summary_path = os.path.join(RESULTS_DIR, "parametric_summary_grinding.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if summary:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
        print("\n[✓] Master summary → %s" % summary_path)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job",      help="Single job name (no .odb extension)")
    group.add_argument("--manifest", help="Path to jobs_grind_*.json manifest")
    parser.add_argument("--odb-dir", default=".",
                        help="Directory containing .odb files")
    args = parser.parse_args()

    if args.job:
        extract_job(args.job, odb_dir=args.odb_dir)
    else:
        extract_all_from_manifest(args.manifest, odb_dir=args.odb_dir)
