"""Tests for hitachi_infra/bearing_rul_uq.py — infrastructure RUL prognostics
with calibrated Bayesian uncertainty."""
import numpy as np
import pytest

rul = pytest.importorskip("hitachi_infra.bearing_rul_uq")


class TestModel:
    def test_unit_reaches_failure(self):
        a, b, T = rul.simulate_unit(np.random.default_rng(0))
        assert T > 0 and b > 0

    def test_bayes_interval_ordered(self):
        rng = np.random.default_rng(1)
        a, b, T = rul.simulate_unit(rng)
        m0, S0 = rul.fit_prior()
        ts, y = rul.observe(a, b, 0.6 * T, rng)
        med, lo, hi, _ = rul.bayes_rul(ts, y, 0.6 * T, m0, S0, rng=rng)
        assert lo < med < hi


class TestCalibrationUQ:
    def test_90ci_is_calibrated(self):
        res = rul.run()
        cov = res["by_life_fraction"]["life_60pct"]["coverage_90ci"]
        # empirical coverage of the 90% credible interval sits near nominal
        assert 0.83 <= cov <= 0.97

    def test_interval_sharpens_toward_failure(self):
        res = rul.run()
        w = [res["by_life_fraction"][f"life_{f}pct"]["mean_ci_width"]
             for f in (40, 60, 80)]
        assert w[0] > w[1] > w[2]

    def test_accuracy_improves_with_more_data(self):
        res = rul.run()
        e = [res["by_life_fraction"][f"life_{f}pct"]["bayes_rmse"]
             for f in (40, 60, 80)]
        assert e[0] > e[1] > e[2]

    def test_bayes_not_worse_than_naive_point(self):
        res = rul.run()
        for f in (40, 60, 80):
            d = res["by_life_fraction"][f"life_{f}pct"]
            assert d["bayes_rmse"] <= d["naive_rmse"] + 0.3
