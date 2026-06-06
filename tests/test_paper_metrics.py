"""Regression tests for published paper metrics (README Key Results table)."""

import pytest

from ml.train_from_experimental import loo_heteroscedastic


@pytest.mark.slow
def test_loo_all_grades_matches_paper():
    m = loo_heteroscedastic("all")
    assert m["n"] >= 25
    assert m["rmse"] < 3.5
    assert m["r2"] > 0.25


@pytest.mark.slow
def test_loo_grade_ab_matches_paper():
    m = loo_heteroscedastic("AB")
    assert m["n"] >= 11
    assert m["rmse"] < 1.8
    assert m["r2"] > 0.75