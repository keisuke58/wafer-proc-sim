"""
Smoke + schema tests for the material-sweep dataset generator.
"""
from __future__ import annotations

from sic.hybrid import Recipe
from sic.hybrid.dataset import (
    build_dataset, summarize, SEMICONDUCTORS, FEATURE_COLS,
)
from sic.hybrid.orchestrator import ROUTES

LABELS = ["feasible", "die_strength_MPa", "total_chipping_um",
          "throughput_wph", "residual_stress_MPa"]


def test_dataset_shape_and_schema():
    mats = ["SiC", "GaN", "Ga2O3"]
    n = 8
    geoms = [(25.0, 110.0), (40.0, 200.0)]
    df = build_dataset(materials=mats, n_per_cell=n, seed=0, geometries=geoms)
    assert len(df) == len(mats) * len(ROUTES) * len(geoms) * n
    from sic.hybrid.dataset import GEOMETRY_COLS, MATERIAL_COLS
    for col in ["material", "route", *FEATURE_COLS, *GEOMETRY_COLS, *MATERIAL_COLS, *LABELS]:
        assert col in df.columns, f"missing column {col}"
    # geometry sweep actually varied
    assert df.street_width_um.nunique() == 2
    assert set(df.material) == set(mats)
    assert set(df.route) == set(ROUTES)
    assert df.feasible.isin([0, 1]).all()
    assert (df.die_strength_MPa > 0).all()


def test_die_strength_orders_by_material():
    # Higher toughness → higher (capped) die strength: Ga2O3 < SiC < Diamond.
    df = build_dataset(materials=["Ga2O3", "SiC", "Diamond"], n_per_cell=6, seed=1)
    med = df.groupby("material").die_strength_MPa.median()
    assert med["Ga2O3"] < med["SiC"] < med["Diamond"]


def test_summarize_one_row_per_cell():
    df = build_dataset(materials=["SiC", "Si"], n_per_cell=5, seed=2)
    s = summarize(df)
    assert len(s) == 2 * len(ROUTES)
    assert (s.feasible_rate.between(0.0, 1.0)).all()
    assert "Si" in SEMICONDUCTORS  # brittle reference present
