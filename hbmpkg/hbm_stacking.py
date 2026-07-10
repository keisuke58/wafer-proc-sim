#!/usr/bin/env python3
"""hbm_stacking.py — why a 16-Hi HBM stack bows more than a 4-Hi one.

Nobody publishes a real HBM stack-up (layer pitch, adhesive modulus, TBDB
bond conditions are proprietary), so this module builds the physics from a
textbook laminate-beam formulation instead of a leaked datasheet.

  WARPAGE     each die and each microbump/underfill interconnect layer has a
              different Young's modulus E and thermal expansion coefficient
              alpha. Bonded together and cooled from the reflow/anneal
              temperature to room temperature, the stack CANNOT contract
              uniformly (the low-alpha Si dies and the high-alpha polymer
              interconnect fight each other) — it bows instead. Classical
              laminated-beam theory reduces this to two equilibrium
              conditions (net axial force = 0, net bending moment = 0)
              across the whole cross-section, solved here as an exact 2x2
              linear system for the uniform strain eps0 and curvature kappa.

  CLOSED LOOP the same 2x2 system is re-derived from a completely
              independent numerical method — slicing the stack into ~20,000
              thin z-slices and solving the discretized normal equations by
              direct summation — and the two agree to a fraction of a
              percent. A single homogeneous layer (or N identical layers) is
              verified analytically to give exactly zero curvature, the
              baseline sanity check for any laminate solver.

  HBM STORY   two DISTINCT questions, kept deliberately separate because
              conflating them gives a misleading answer:
                (a) package-level: base substrate + N repeat units of
                    (thinned die + microbump/underfill) for N = 4/8/12/16 —
                    the generational "-Hi" ladder. In this substrate-
                    dominated geometry the bow actually SHRINKS as N grows
                    (self-stiffening: added high-modulus Si outpaces the
                    added thermal-mismatch moment) — the naive "more layers
                    = more warpage" intuition is wrong here, and the honest
                    surprising result is reported as such.
                (b) interlayer-level, isolated: a single (die + interconnect)
                    repeat unit's OWN intrinsic curvature, with no substrate
                    confound, comparing a microbump+underfill interlayer
                    against a hybrid (Cu-Cu) bond interlayer. This isolates
                    what stack designers actually mean when they say hybrid
                    bonding "reduces warpage risk" — and it is a large
                    effect (two orders of magnitude smaller local curvature),
                    unlike the confounded package-level number in (a).

Pure numpy + matplotlib.  Run:  python3 packaging/hbm_stacking.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "hbm_stacking_results.json")

# ── material library (E [GPa], alpha [ppm/K]) ───────────────────────────────
SI = dict(E_gpa=130.0, alpha_ppmK=2.6)          # thinned DRAM die
BUMP_UF = dict(E_gpa=6.0, alpha_ppmK=45.0)      # Cu/Sn microbump + underfill
HYBRID = dict(E_gpa=70.0, alpha_ppmK=3.0)       # Cu-Cu hybrid bond interface
                                                 # (oxide-dominated bondline,
                                                 # CTE close to Si/SiO2)
SUBSTRATE = dict(E_gpa=25.0, alpha_ppmK=16.0)   # organic build-up substrate

T_DIE_UM = 30.0
T_BUMP_UM = 10.0
T_HYBRID_UM = 0.5                                # near-zero bond line
T_SUB_UM = 300.0
DELTA_T_K = -225.0                               # 250C reflow -> 25C room


def _layers_from_spec(spec):
    """spec: list of (material_dict, thickness_um) -> list of
    (E_Pa, alpha_perK, t_m) bottom-to-top."""
    out = []
    for mat, t_um in spec:
        out.append((mat["E_gpa"] * 1e9, mat["alpha_ppmK"] * 1e-6, t_um * 1e-6))
    return out


def stack_curvature(layers, delta_t):
    """Exact 2x2 force+moment balance for a laminated beam. layers: list of
    (E_Pa, alpha_perK, t_m) stacked bottom (z=0) to top. Returns (kappa
    [1/m], eps0)."""
    z = 0.0
    A11 = A12 = A22 = b1 = b2 = 0.0
    for E, alpha, t in layers:
        zbar = z + t / 2.0
        A11 += E * t
        A12 += E * t * zbar
        A22 += E * (t * zbar**2 + t**3 / 12.0)
        b1 += E * t * alpha * delta_t
        b2 += E * t * zbar * alpha * delta_t
        z += t
    A = np.array([[A11, A12], [A12, A22]])
    b = np.array([b1, b2])
    eps0, kappa = np.linalg.solve(A, b)
    return float(kappa), float(eps0)


def stack_curvature_fine_slice(layers, delta_t, n_slice=20000):
    """Independent cross-check: discretize the stack into n_slice thin
    z-slices (material properties looked up per slice) and solve the same
    normal equations by direct Riemann-sum integration instead of the exact
    per-layer polynomial integral."""
    total_t = sum(t for _, _, t in layers)
    edges = np.linspace(0.0, total_t, n_slice + 1)
    zc = 0.5 * (edges[:-1] + edges[1:])
    dz = edges[1] - edges[0]
    E_of_z = np.empty(n_slice)
    a_of_z = np.empty(n_slice)
    z0 = 0.0
    for E, alpha, t in layers:
        m = (zc >= z0) & (zc < z0 + t)
        E_of_z[m] = E
        a_of_z[m] = alpha
        z0 += t
    A11 = np.sum(E_of_z * dz)
    A12 = np.sum(E_of_z * zc * dz)
    A22 = np.sum(E_of_z * zc**2 * dz)
    b1 = np.sum(E_of_z * a_of_z * delta_t * dz)
    b2 = np.sum(E_of_z * a_of_z * delta_t * zc * dz)
    A = np.array([[A11, A12], [A12, A22]])
    b = np.array([b1, b2])
    eps0, kappa = np.linalg.solve(A, b)
    return float(kappa), float(eps0)


def bow_um(kappa, width_m):
    """Circular-arc sag over a die width, in micrometers."""
    if abs(kappa) < 1e-9:
        return 0.0
    r = 1.0 / kappa
    half = width_m / 2.0
    disc = r * r - half * half
    if disc < 0:
        return float("nan")
    sag = abs(r) - np.sqrt(disc)
    return float(sag * 1e6) * np.sign(kappa)


def hi_stack(n_hi, bump=True, base=True):
    """base substrate + n_hi x (die [+ bump/hybrid layer])."""
    spec = []
    if base:
        spec.append((SUBSTRATE, T_SUB_UM))
    inter = BUMP_UF if bump else HYBRID
    t_inter = T_BUMP_UM if bump else T_HYBRID_UM
    for _ in range(n_hi):
        spec.append((SI, T_DIE_UM))
        spec.append((inter, t_inter))
    return _layers_from_spec(spec)


def unit_cell(bump=True):
    """A single (die + interconnect) repeat unit, NO substrate — isolates
    the interlayer's own intrinsic curvature from package-level effects."""
    inter = BUMP_UF if bump else HYBRID
    t_inter = T_BUMP_UM if bump else T_HYBRID_UM
    return _layers_from_spec([(SI, T_DIE_UM), (inter, t_inter)])


DIE_WIDTH_M = 5.0e-3     # 5 mm die width, bow reported over this span


def run():
    # ── validation 1: single layer -> exactly zero curvature ──
    single = _layers_from_spec([(SI, 30.0)])
    k_single, _ = stack_curvature(single, DELTA_T_K)

    # ── validation 2: N identical layers -> exactly zero curvature ──
    ident = _layers_from_spec([(SI, 30.0)] * 6)
    k_ident, _ = stack_curvature(ident, DELTA_T_K)

    # ── validation 3: closed-form vs fine-slice cross-check ──
    ref = hi_stack(8, bump=True)
    k_exact, _ = stack_curvature(ref, DELTA_T_K)
    k_fine, _ = stack_curvature_fine_slice(ref, DELTA_T_K)
    cross_err_pct = 100.0 * abs(k_fine - k_exact) / abs(k_exact)

    # ── story (a): package-level bow vs stack height (substrate-dominated,
    #    microbump interconnect) — reported honestly even though the trend
    #    (shrinking bow) is the opposite of the naive "taller = worse"
    #    intuition, because in THIS geometry the substrate-vs-Si mismatch
    #    dominates and the growing Si fraction self-stiffens the stack.
    ns = [4, 8, 12, 16]
    package_rows = {}
    for n in ns:
        layers = hi_stack(n, bump=True)
        k, _ = stack_curvature(layers, DELTA_T_K)
        package_rows[f"{n}-Hi"] = {
            "kappa_per_m": round(k, 4),
            "bow_um": round(bow_um(k, DIE_WIDTH_M), 3),
        }

    # ── story (b): isolated interlayer comparison, no substrate confound —
    #    this is what "hybrid bonding reduces warpage risk" actually means.
    k_bump_unit, _ = stack_curvature(unit_cell(bump=True), DELTA_T_K)
    k_hyb_unit, _ = stack_curvature(unit_cell(bump=False), DELTA_T_K)
    unit_bump = {"kappa_per_m": round(k_bump_unit, 3),
                 "bow_um": round(bow_um(k_bump_unit, DIE_WIDTH_M), 3)}
    unit_hybrid = {"kappa_per_m": round(k_hyb_unit, 5),
                   "bow_um": round(bow_um(k_hyb_unit, DIE_WIDTH_M), 5)}

    return {
        "config": {"delta_T_K": DELTA_T_K, "die_width_mm": DIE_WIDTH_M * 1e3,
                   "die_um": T_DIE_UM, "bump_underfill_um": T_BUMP_UM,
                   "hybrid_bondline_um": T_HYBRID_UM},
        "validation": {
            "single_layer_kappa_per_m": k_single,
            "identical_layers_kappa_per_m": k_ident,
            "fine_slice_vs_exact_pct_err": round(cross_err_pct, 5),
        },
        "package_level_bow_vs_height": package_rows,
        "isolated_unit_cell": {
            "microbump_underfill": unit_bump,
            "hybrid_Cu_Cu_bond": unit_hybrid,
            "curvature_ratio_bump_over_hybrid": round(
                abs(k_bump_unit / k_hyb_unit), 1),
        },
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [4, 8, 12, 16]
    bow_pkg = [res["package_level_bow_vs_height"][f"{n}-Hi"]["bow_um"]
               for n in ns]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax = axes[0]
    ax.plot(ns, bow_pkg, "o-", lw=2, color="#0b3d91")
    ax.set_xlabel("stack height (N-Hi)")
    ax.set_ylabel("package bow over 5 mm width [um]")
    ax.set_title("Package-level bow SHRINKS with stack height here (base-\n"
                 "substrate mismatch dominates; adding high-modulus Si\n"
                 "self-stiffens the stack) — the naive intuition is wrong",
                 fontsize=9.5)
    ax.grid(alpha=0.3); ax.set_xticks(ns)

    ax = axes[1]
    u = res["isolated_unit_cell"]
    labels = ["microbump\n+ underfill", "hybrid\n(Cu-Cu) bond"]
    vals = [abs(u["microbump_underfill"]["kappa_per_m"]),
            abs(u["hybrid_Cu_Cu_bond"]["kappa_per_m"])]
    bars = ax.bar(labels, vals, color=["#c0392b", "#1a7a4a"])
    ax.set_yscale("log")
    ax.set_ylabel("|curvature| of an isolated die+interconnect\nrepeat unit [1/m], no substrate")
    ratio = u["curvature_ratio_bump_over_hybrid"]
    ax.set_title(f"Isolated repeat-unit curvature: hybrid bonding is\n"
                 f"{ratio:.0f}x smaller — this is what \"reduces warpage\n"
                 f"risk\" means at the interconnect level", fontsize=9.5)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3g}", (b.get_x() + b.get_width()/2, v),
                    textcoords="offset points", xytext=(0, 4), fontsize=8,
                    ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "hbm_warpage.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
