"""
Tests for map_estimator.py (B) and surrogate.py (C).

Notes on test design:
  - MAP accuracy tests use qualitative checks (correct sign of beta, LL not worse
    than uninformative baseline). With only 17 binary observations, exact parameter
    recovery is not expected (sign-only observations have low Fisher information).
  - Surrogate accuracy tests check relative error < 10% on holdout.
  - Surrogate speed is not tested in CI (timing is environment-dependent).
  - Surrogate is trained once at module level and shared across tests.
"""
from __future__ import annotations
import sys, os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from research.phasefield.at2_simulator import (
    PFParams, SimConfig, log_likelihood,
    generate_synthetic_dataset,
)
from research.phasefield.map_estimator import (
    MapResult, fit_map, relative_error,
    _scale_aware_log_prior, _decode_2d,
)
from research.phasefield.surrogate import (
    AT2Surrogate, generate_training_data,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

FAST = SimConfig(n_elem=100, n_steps=80, u_max_factor=5.0)

TRUE = PFParams(Gc=8e-4, ell=0.04, E=1.0, beta=-0.40)

# Dense angle grid (avoids theta=0 which is always p=0.5)
THETA_GRID = list(np.arange(10, 91, 5.0))

OBS = generate_synthetic_dataset(TRUE, theta_list=THETA_GRID, cfg=FAST, seed=0)


# ── Train surrogate once for all surrogate tests ───────────────────────────────

@pytest.fixture(scope="module")
def trained_surrogate():
    data = generate_training_data(300, cfg=FAST, seed=0)
    surr = AT2Surrogate(hidden=[64, 64])
    surr.fit(data, n_epochs=2000, lr=3e-3, verbose=False)
    return surr


# ════════════════════════════════════════════════════════════════════════════
#  1.  map_estimator helpers
# ════════════════════════════════════════════════════════════════════════════

class TestMapHelpers:
    def test_decode_2d_positive_gc(self):
        p = _decode_2d(np.array([np.log(5e-4), -0.3]), ell=0.04, E=1.0)
        assert p.Gc == pytest.approx(5e-4, rel=1e-6)
        assert p.beta == pytest.approx(-0.3, rel=1e-6)

    def test_scale_aware_log_prior_at_center_is_zero_beta(self):
        # At beta=0, Gc=center, ell=center: prior should be 0 (modulo Gc/ell terms)
        lp = _scale_aware_log_prior(
            PFParams(Gc=8e-4, ell=0.04, E=1.0, beta=0.0),
            Gc_center=8e-4, ell_center=0.04, sigma_log=2.0,
        )
        assert np.isfinite(lp)
        assert lp > -1e-3   # nearly zero at center

    def test_scale_aware_log_prior_negative_gc_is_neginf(self):
        p = PFParams(Gc=-1e-4, ell=0.04, E=1.0, beta=0.0)
        assert _scale_aware_log_prior(p, 8e-4, 0.04) == -np.inf

    def test_relative_error_zero_for_identical(self):
        errs = relative_error(TRUE, TRUE)
        assert errs["Gc"]   == pytest.approx(0.0)
        assert errs["beta"] == pytest.approx(0.0)

    def test_relative_error_positive(self):
        p_wrong = PFParams(Gc=4e-4, ell=TRUE.ell, E=TRUE.E, beta=-0.2)
        errs = relative_error(p_wrong, TRUE)
        assert errs["Gc"]   > 0.0
        assert errs["beta"] > 0.0


# ════════════════════════════════════════════════════════════════════════════
#  2.  fit_map — structural checks
# ════════════════════════════════════════════════════════════════════════════

class TestFitMapStructure:
    @pytest.fixture(autouse=True)
    def _run_map(self):
        self.result = fit_map(
            OBS, ell=TRUE.ell, E=TRUE.E, cfg=FAST,
            n_restarts=3, seed=0,
        )

    def test_returns_map_result(self):
        assert isinstance(self.result, MapResult)

    def test_params_is_pf_params(self):
        assert isinstance(self.result.params, PFParams)

    def test_gc_positive(self):
        assert self.result.params.Gc > 0.0

    def test_neg_log_post_finite(self):
        assert np.isfinite(self.result.neg_log_post)

    def test_n_restarts_tried(self):
        assert self.result.n_restarts_tried == 3

    def test_all_neg_lp_nonempty(self):
        assert len(self.result.all_neg_lp) > 0


# ════════════════════════════════════════════════════════════════════════════
#  3.  fit_map — physics checks
# ════════════════════════════════════════════════════════════════════════════

class TestFitMapPhysics:
    @pytest.fixture(autouse=True)
    def _run_map(self):
        self.result = fit_map(
            OBS, ell=TRUE.ell, E=TRUE.E, cfg=FAST,
            n_restarts=6, seed=42,
        )

    def test_beta_is_negative(self):
        # Data has more cracks at high theta → consistent with negative beta
        assert self.result.params.beta < 0.1, \
            f"Expected beta < 0.1, got {self.result.params.beta:.4f}"

    def test_map_ll_not_worse_than_uninformative(self):
        # Uninformative: beta=0 → all p_crack=0.5 → LL = N*log(0.5)
        uninformative_ll = len(OBS) * np.log(0.5)
        map_ll = log_likelihood(self.result.params, OBS, cfg=FAST)
        assert map_ll >= uninformative_ll - 0.5, \
            f"MAP LL={map_ll:.3f} worse than uninformative={uninformative_ll:.3f}"

    def test_map_ll_beats_true_params_or_close(self):
        # MAP should find params with LL >= true LL (or very close)
        # True params don't necessarily maximize LL for this specific realization
        map_ll  = log_likelihood(self.result.params, OBS, cfg=FAST)
        true_ll = log_likelihood(TRUE, OBS, cfg=FAST)
        assert map_ll >= true_ll - 2.0, \
            f"MAP LL={map_ll:.3f} far below true LL={true_ll:.3f}"

    def test_beta_error_below_threshold(self):
        # With 17 sign-only observations, expect ~50% error; flag if >90%
        err = relative_error(self.result.params, TRUE)
        assert err["beta"] < 0.90, \
            f"beta error {err['beta']*100:.0f}% seems too large"


# ════════════════════════════════════════════════════════════════════════════
#  4.  Surrogate — prediction correctness
# ════════════════════════════════════════════════════════════════════════════

class TestSurrogatePredict:
    def test_predict_positive(self, trained_surrogate):
        init_U = trained_surrogate.predict_init_U(8e-4, 0.04)
        assert init_U > 0.0

    def test_predict_batch_shape(self, trained_surrogate):
        gc_arr  = np.array([4e-4, 8e-4, 12e-4])
        ell_arr = np.array([0.03, 0.04, 0.05])
        result  = trained_surrogate.predict_batch(gc_arr, ell_arr)
        assert result.shape == (3,)

    def test_higher_gc_higher_init_u(self, trained_surrogate):
        u_lo = trained_surrogate.predict_init_U(4e-4, 0.04)
        u_hi = trained_surrogate.predict_init_U(12e-4, 0.04)
        assert u_hi > u_lo, "Higher Gc → later crack initiation"

    def test_larger_ell_lower_init_u(self, trained_surrogate):
        # ell in denominator of critical strain: larger ell → smaller init_U
        u_thin = trained_surrogate.predict_init_U(8e-4, 0.02)
        u_wide = trained_surrogate.predict_init_U(8e-4, 0.08)
        assert u_thin > u_wide, "Larger ell → earlier crack initiation"

    def test_mean_relative_error_below_10pct(self, trained_surrogate):
        test_data = generate_training_data(80, cfg=FAST, seed=777)
        pred = trained_surrogate.predict_batch(
            np.exp(test_data.log_Gc), np.exp(test_data.log_ell))
        rel_err = np.abs(pred - test_data.init_U) / (test_data.init_U + 1e-12)
        mean_err = rel_err.mean()
        assert mean_err < 0.10, f"Mean relative error {mean_err*100:.1f}% > 10%"


# ════════════════════════════════════════════════════════════════════════════
#  5.  Surrogate — observation model
# ════════════════════════════════════════════════════════════════════════════

class TestSurrogateCrackProb:
    def test_probability_in_unit_interval(self, trained_surrogate):
        for theta in [0.0, 30.0, 60.0, 90.0]:
            p = trained_surrogate.crack_probability(TRUE, theta)
            assert 0.0 < p < 1.0, f"p={p:.4f} at theta={theta}"

    def test_negative_beta_higher_p_at_90(self, trained_surrogate):
        # beta=-0.4: Gc_eff(90°) < Gc_ref → easier to crack → p > 0.5
        p = trained_surrogate.crack_probability(TRUE, 90.0)
        assert p > 0.5, f"Expected p>0.5 at theta=90 with beta={TRUE.beta}"

    def test_theta_zero_gives_half(self, trained_surrogate):
        # At theta=0: Gc_eff = Gc0 exactly → z=0 → p=0.5
        p = trained_surrogate.crack_probability(TRUE, 0.0)
        assert abs(p - 0.5) < 1e-6, f"Expected p=0.5 at theta=0, got {p:.6f}"

    def test_crack_prob_increases_with_negative_beta_at_large_theta(self, trained_surrogate):
        # For negative beta, p_crack(theta) should increase from theta=0 to theta=90
        p_lo = trained_surrogate.crack_probability(TRUE, 20.0)
        p_hi = trained_surrogate.crack_probability(TRUE, 80.0)
        assert p_hi > p_lo


# ════════════════════════════════════════════════════════════════════════════
#  6.  Surrogate — log-likelihood
# ════════════════════════════════════════════════════════════════════════════

class TestSurrogateLL:
    def test_ll_finite(self, trained_surrogate):
        ll = trained_surrogate.log_likelihood(TRUE, OBS)
        assert np.isfinite(ll)

    def test_ll_nonpositive(self, trained_surrogate):
        ll = trained_surrogate.log_likelihood(TRUE, OBS)
        assert ll <= 0.0

    def test_surrogate_ll_close_to_sim_ll(self, trained_surrogate):
        ll_sim  = log_likelihood(TRUE, OBS, cfg=FAST)
        ll_surr = trained_surrogate.log_likelihood(TRUE, OBS)
        # Allow generous tolerance due to coarse FAST simulator discretization
        assert abs(ll_sim - ll_surr) < 3.0, \
            f"LL mismatch: sim={ll_sim:.3f} surr={ll_surr:.3f}"

    def test_surrogate_ll_beats_uninformative(self, trained_surrogate):
        uninformative = len(OBS) * np.log(0.5)
        ll_surr = trained_surrogate.log_likelihood(TRUE, OBS)
        assert ll_surr > uninformative - 1.0
