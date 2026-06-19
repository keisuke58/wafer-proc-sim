"""
sic.hybrid.dataset — material-sweep dataset generator for hybrid 2nm dicing.

Sweeps {Si, SiC, GaN, Ga2O3, Diamond} × {laser+stealth, laser+plasma,
laser+blade}, Latin-Hypercube-samples the recipe space in each cell, runs the
full hybrid model, and emits a tidy ML-ready table (one row per sampled recipe)
plus a per-cell summary.

Target uses:
    - regression  : predict die_strength_MPa / total_chipping_um / throughput_wph
    - classification : predict `feasible` (2nm constraints satisfied)

IMPORTANT (data card): the laser groove/stealth OPTICAL model is SiC-calibrated
(refractive index, ablation thresholds). The material-specific signal therefore
lives in the FRACTURE/THERMAL KPIs (die strength, chipping, subsurface damage,
residual stress), driven by each material's K_Ic, H, E and thermal properties.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from sic.hybrid.base import Recipe
from sic.hybrid.orchestrator import HybridDicer, ROUTES
from sic.hybrid.optimize import _space_for, _lhs, _decode
from data.materials.material_properties import ALL_MATERIALS

#: single-crystal semiconductors carried through the fracture/thermal model
SEMICONDUCTORS: List[str] = ["Si", "SiC", "GaN", "Ga2O3", "Diamond"]

#: full recipe feature schema (constant columns where a knob is not route-varied)
FEATURE_COLS = [
    "groove_power_W", "groove_speed_mm_s", "groove_passes",
    "stealth_power_W", "stealth_speed_mm_s", "stealth_focal_depth_um", "stealth_layers",
    "plasma_pressure_mTorr", "plasma_etch_rate_um_min",
    "blade_feed_mm_s", "blade_W_um",
]

#: wafer geometry columns swept alongside the recipe
GEOMETRY_COLS = ["street_width_um", "wafer_thickness_um", "lowk_stack_um"]

#: material-descriptor columns (enable OOD generalization to unseen materials)
MATERIAL_COLS = ["mat_K_Ic_MPa", "mat_H_GPa", "mat_E_GPa",
                 "mat_k_thermal", "mat_sigma_flex_MPa"]

#: (street_um, thickness_um) grid for the geometry sweep — advanced-node-class
DEFAULT_GEOMETRIES = [
    (20.0, 80.0), (25.0, 110.0), (30.0, 140.0), (40.0, 200.0),
]


def _material_features(material: str) -> dict:
    m = ALL_MATERIALS.get(material, ALL_MATERIALS["SiC"])
    return {
        "mat_K_Ic_MPa": m["K_Ic"] / 1e6,
        "mat_H_GPa": m["H_v"] / 1e9,
        "mat_E_GPa": m["E"] / 1e9,
        "mat_k_thermal": m["k_thermal"],
        "mat_sigma_flex_MPa": m["sigma_flex"] / 1e6,
    }


def build_dataset(materials: Optional[List[str]] = None, n_per_cell: int = 120,
                  seed: int = 0, base: Optional[Recipe] = None,
                  dicer: Optional[HybridDicer] = None,
                  geometries: Optional[List[tuple]] = None) -> pd.DataFrame:
    """Generate the tidy material×route×geometry×recipe dataset."""
    materials = materials or SEMICONDUCTORS
    base = base or Recipe()
    dicer = dicer or HybridDicer()
    geometries = geometries or DEFAULT_GEOMETRIES

    rows = []
    for mi, material in enumerate(materials):
        matfeat = _material_features(material)
        for gi, (street, thick) in enumerate(geometries):
            for ri, route in enumerate(ROUTES):
                space = _space_for(route)
                names = list(space)
                rng = np.random.default_rng(seed + 1009 * mi + 131 * gi + 31 * ri)
                U = _lhs(n_per_cell, len(names), rng)
                mbase = base.copy(material=material, route=route,
                                  street_width_um=street, wafer_thickness_um=thick)
                for u in U:
                    rec = _decode(route, u, mbase).copy(
                        material=material, street_width_um=street,
                        wafer_thickness_um=thick)
                    res = dicer.run(rec)
                    a, b, f = res.stage_a, res.stage_b, res.feasibility
                    row = {"material": material, "route": route}
                    row.update({c: getattr(rec, c) for c in GEOMETRY_COLS})
                    row.update({c: getattr(rec, c) for c in FEATURE_COLS})
                    row.update(matfeat)
                    row.update(
                        # stage A (laser groove) outputs
                        a_kerf_um=a.kerf_um, a_chip_um=a.chipping_um, a_haz_um=a.haz_um,
                        a_lowk_cleared_um=a.lowk_cleared_um, a_wph=a.throughput_wph,
                        # stage B (singulation) outputs
                        b_method=b.method, b_kerf_um=b.kerf_um, b_chip_um=b.chipping_um,
                        b_haz_um=b.haz_um, b_ssd_um=b.subsurface_crack_um,
                        b_residual_MPa=b.residual_stress_MPa, b_wph=b.throughput_wph,
                        # combined KPIs / labels
                        feasible=int(f.feasible), n_violations=len(f.violations),
                        die_strength_MPa=f.die_strength_MPa,
                        total_chipping_um=f.total_chipping_um,
                        max_haz_um=f.max_haz_um, governing_crack_um=f.governing_crack_um,
                        throughput_wph=res.throughput_wph,
                        residual_stress_MPa=res.residual_stress_MPa,
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per material×route aggregate (feasible rate + KPI medians)."""
    g = df.groupby(["material", "route"])
    out = g.agg(
        n=("feasible", "size"),
        feasible_rate=("feasible", "mean"),
        die_strength_med=("die_strength_MPa", "median"),
        chipping_med=("total_chipping_um", "median"),
        residual_med=("residual_stress_MPa", "median"),
        throughput_med=("throughput_wph", "median"),
        governing_crack_med=("governing_crack_um", "median"),
    ).reset_index()
    return out


def write_dataset(df: pd.DataFrame, out_dir: str) -> dict:
    """Write full + summary CSVs and return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    full = os.path.join(out_dir, "hybrid_2nm_dataset.csv")
    summ = os.path.join(out_dir, "hybrid_2nm_summary.csv")
    df.to_csv(full, index=False)
    summarize(df).to_csv(summ, index=False)
    return {"full": full, "summary": summ, "n_rows": len(df)}
