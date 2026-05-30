"""
Visualize d360 ODB: extract STATUS field frames and save PNG images.
Run with: abaqus python visualize_d360.py

Output: results/d360_frames/frame_NNN.png  (one per field output frame)
"""

import os, sys
import numpy as np

# ── ABAQUS odbAccess ──────────────────────────────────────────────────────────
try:
    from odbAccess import openOdb
except ImportError:
    print("[!] Run via: abaqus python visualize_d360.py")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ODB_PATH   = os.path.join(SCRIPT_DIR, "dicing_4HSiC_d360_bw23.odb")
OUT_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "results", "d360_frames")
os.makedirs(OUT_DIR, exist_ok=True)

# Blade geometry (µm): blade_W=23, starts at x=250 (W/2=500/2), plunges then feeds
WAFER_W_UM  = 500.0
BLADE_W_UM  = 23.0
BLADE_H_UM  = 60.0   # visual height above wafer top
WAFER_H_UM  = 450.0
PLUNGE_V    = -0.5e3  # µm/s downward (STEP 1)
FEED_V      =  0.5e3  # µm/s rightward (STEP 2)
# Step durations
T_PLUNGE    = 0.000724  # s
T_FEED      = 0.0008    # s


def wafer_instance(odb):
    asm = odb.rootAssembly
    for key in ("WAFER-1", "Wafer-1"):
        if key in asm.instances.keys():
            return asm.instances[key]
    for inst in asm.instances.values():
        if inst.type != "RIGID_BODY":
            return inst
    return list(asm.instances.values())[0]


def build_mesh_arrays(inst):
    """Return node_xy (dict label→[x,y] in µm) and elem_conn list."""
    M = 1e6  # m → µm
    node_xy = {}
    for n in inst.nodes:
        node_xy[n.label] = np.array(n.coordinates[:2]) * M

    elem_conn = []
    for e in inst.elements:
        elem_conn.append((e.label, list(e.connectivity)))
    return node_xy, elem_conn


def blade_rect(t_global, t_plunge_end, origin_x_um=250.0, wafer_top_um=WAFER_H_UM):
    """Return (x0, y0, width, height) of blade in µm at global time t_global."""
    if t_global <= t_plunge_end:
        # Plunge: blade moves down
        plunge_depth = (t_global * abs(PLUNGE_V))  # µm from top
        y_bottom = wafer_top_um - plunge_depth
    else:
        # Feed: blade at full depth, moves right
        plunge_depth = T_PLUNGE * abs(PLUNGE_V)
        y_bottom = wafer_top_um - plunge_depth
        feed_dx = (t_global - t_plunge_end) * FEED_V
        origin_x_um = origin_x_um + feed_dx

    x0 = origin_x_um - BLADE_W_UM / 2.0
    y0 = y_bottom
    return x0, y0, BLADE_W_UM, wafer_top_um + BLADE_H_UM - y0


def render_frame(node_xy, elem_conn, status_map, peeq_map,
                 t_global, t_plunge_end, frame_idx, step_name, step_time):
    fig, ax = plt.subplots(figsize=(7, 5))

    # Build polygon lists for alive and deleted elements
    polys_alive, polys_dead = [], []
    peeq_vals = []

    for label, conn in elem_conn:
        try:
            pts = np.array([node_xy[n] for n in conn])
        except KeyError:
            continue
        s = status_map.get(label, 1.0)
        if s < 0.5:
            polys_dead.append(pts)
        else:
            polys_alive.append(pts)
            peeq_vals.append(peeq_map.get(label, 0.0))

    # Alive elements colored by PEEQ
    if polys_alive:
        peeq_arr = np.array(peeq_vals)
        col_alive = PolyCollection(polys_alive, array=peeq_arr,
                                   cmap="plasma", clim=(0, 0.3),
                                   edgecolors="none", linewidths=0)
        ax.add_collection(col_alive)
        plt.colorbar(col_alive, ax=ax, label="PEEQ", fraction=0.03, pad=0.02)

    # Deleted elements in red
    if polys_dead:
        col_dead = PolyCollection(polys_dead, facecolors="#d62728",
                                  edgecolors="none", linewidths=0, alpha=0.7)
        ax.add_collection(col_dead)

    # Blade rectangle
    bx, by, bw, bh = blade_rect(t_global, t_plunge_end)
    blade_patch = mpatches.Rectangle((bx, by), bw, bh,
                                     facecolor="#636363", edgecolor="k",
                                     linewidth=0.8, zorder=5)
    ax.add_patch(blade_patch)

    ax.set_xlim(0, WAFER_W_UM)
    ax.set_ylim(-20, WAFER_H_UM + BLADE_H_UM + 10)
    ax.set_aspect("equal")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title(f"d360 — {step_name}  t={step_time*1e6:.0f} µs\n"
                 f"Deleted: {len(polys_dead)}/{len(elem_conn)} elements "
                 f"({100*len(polys_dead)/max(len(elem_conn),1):.1f}%)")

    dead_patch  = mpatches.Patch(color="#d62728", label="Deleted (STATUS=0)")
    alive_patch = mpatches.Patch(color="#9ecae1", label="Alive (PEEQ-colored)")
    blade_p     = mpatches.Patch(color="#636363", label="Blade")
    ax.legend(handles=[dead_patch, alive_patch, blade_p],
              loc="upper right", fontsize=8)

    out = os.path.join(OUT_DIR, f"frame_{frame_idx:04d}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"[->] Opening {ODB_PATH}")
odb  = openOdb(path=ODB_PATH, readOnly=True)
inst = wafer_instance(odb)

print("[->] Building mesh arrays ...")
node_xy, elem_conn = build_mesh_arrays(inst)
print(f"     {len(node_xy)} nodes, {len(elem_conn)} elements")

# Determine step origins for blade position tracking
step_names  = list(odb.steps.keys())
step_origin = {}
t_acc = 0.0
for sn in step_names:
    step_origin[sn] = t_acc
    t_acc += odb.steps[sn].timePeriod

# Identify plunge step (first step)
plunge_step_name = step_names[0]
t_plunge_end = step_origin[plunge_step_name] + odb.steps[plunge_step_name].timePeriod

frame_idx = 0
# Subsample: take every Nth frame to keep ~30 images total
total_frames = sum(len(odb.steps[s].frames) for s in step_names)
stride = max(1, total_frames // 30)
print(f"     Total frames: {total_frames}, stride: {stride}")

for sname in step_names:
    step = odb.steps[sname]
    for fi, frame in enumerate(step.frames):
        if fi % stride != 0 and fi != len(step.frames) - 1:
            continue

        t_global = step_origin[sname] + frame.frameValue
        fouts = frame.fieldOutputs.keys()

        # STATUS
        status_map = {}
        if "STATUS" in fouts:
            try:
                for v in frame.fieldOutputs["STATUS"].getSubset(region=inst).values:
                    status_map[v.elementLabel] = v.data
            except Exception:
                pass

        # PEEQ
        peeq_map = {}
        if "PEEQ" in fouts:
            try:
                for v in frame.fieldOutputs["PEEQ"].getSubset(region=inst).values:
                    peeq_map[v.elementLabel] = v.data
            except Exception:
                pass

        out_path = render_frame(node_xy, elem_conn, status_map, peeq_map,
                                t_global, t_plunge_end,
                                frame_idx, sname, frame.frameValue)
        print(f"  [{frame_idx:03d}] {sname} t={frame.frameValue*1e6:.1f}µs → {out_path}")
        frame_idx += 1

odb.close()
print(f"\n[OK] {frame_idx} frames saved to {OUT_DIR}")
print("     Combine with: ffmpeg -r 5 -i frame_%04d.png -vcodec libx264 d360_cutting.mp4")
