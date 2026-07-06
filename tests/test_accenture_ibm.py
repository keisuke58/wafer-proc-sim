"""Tests for accenture/ (applied AI: uplift, drift) and ibm/ (trustworthy
AI: fairness, Shapley)."""
import numpy as np
import pytest

um = pytest.importorskip("accenture.uplift_modeling")
dm = pytest.importorskip("accenture.drift_monitoring")
fm = pytest.importorskip("ibm.fairness_mitigation")
sx = pytest.importorskip("ibm.shapley_explain")


class TestUplift:
    def test_t_learner_recovers_true_uplift(self):
        x, t, y, tau = um.make_trial(seed=0)
        up, _ = um.t_learner(x, t, y)
        assert np.corrcoef(up, tau)[0, 1] > 0.7

    def test_uplift_targeting_beats_response_and_random(self):
        x, t, y, tau = um.make_trial(seed=0)
        fr, q_up, q_re, q_rand, _ = um._eval_split(x, t, y, tau)
        assert um.qini_area(fr, q_up) > um.qini_area(fr, q_re)
        assert um.qini_area(fr, q_re) >= um.qini_area(fr, q_rand) - 5

    def test_aa_test_has_no_uplift_signal(self):
        x, t, y, _ = um.make_trial(seed=2, aa=True)
        fr, q_up, _, _, _ = um._eval_split(x, t, y)
        # held-out Qini area on a zero-effect trial is ~0
        assert abs(um.qini_area(fr, q_up)) < 15


class TestDrift:
    def test_psi_zero_for_identical(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=5000)
        assert dm.psi(v, v.copy()) < 0.01

    def test_psi_monotone_in_shift(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(size=5000)
        prev = -1.0
        for s in (0.0, 0.3, 0.6, 1.0):
            live = rng.normal(s, 1.0, size=5000)
            cur = dm.psi(ref, live)
            assert cur >= prev - 1e-9
            prev = cur

    def test_alarm_leads_accuracy_drop(self):
        res = dm.run()
        assert res["psi_identical_dist"] < 0.01
        assert res["alarm_window_psi"] is not None
        assert res["acc_drop_window"] is not None
        # unsupervised alarm fires strictly before the labelled accuracy drop
        assert res["alarm_window_psi"] < res["acc_drop_window"]


class TestFairness:
    def test_reweighing_equalizes_weighted_positive_rate(self):
        res = fm.run()
        assert res["reweigh_closed_loop"]["abs_diff"] < 1e-9

    def test_baseline_is_unfair_group_thresholds_reach_parity(self):
        res = fm.run()
        assert res["baseline_thresh0.5"]["DIR"] < 0.8      # unfair
        assert res["group_thresholds"]["DIR"] > 0.95       # parity
        assert res["group_thresholds"]["dem_parity_diff"] < 0.01

    def test_parity_costs_accuracy(self):
        res = fm.run()
        assert res["parity_accuracy_cost_pts"] > 0.0


class TestShapley:
    def test_efficiency_axiom_holds_exactly(self):
        f, bg, w = sx.make_model()
        rng = np.random.default_rng(3)
        x = rng.normal(size=sx.N_FEAT)
        phi = sx.exact_shapley(f, bg, x)
        base = float(f(bg).mean())
        pred = float(f(x)[0])
        assert abs(phi.sum() - (pred - base)) < 1e-10

    def test_null_feature_gets_zero(self):
        """A feature the model ignores has ~0 Shapley value."""
        rng = np.random.default_rng(0)
        bg = rng.normal(size=(120, 4))
        w = np.array([1.5, 0.0, -1.0, 0.7])   # feature 1 is a null player

        def f(X):
            X = np.atleast_2d(X)
            return sx._sigmoid(X @ w)
        x = rng.normal(size=4)
        phi = sx.exact_shapley(f, bg, x)
        assert abs(phi[1]) < 1e-9

    def test_permutation_converges_to_exact(self):
        res = sx.run()
        errs = res["permutation_max_err_vs_exact"]
        assert errs["1600"] < errs["25"]
        assert errs["1600"] < 0.01
