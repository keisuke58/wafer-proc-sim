"""Tests for demos/decc_sim2real.py — DECC-style sim-to-real Bayesian calibration.

Fast settings: short chains, few shots, fixed seeds.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demos.decc_sim2real import (
    TRUE_PARAMS,
    aim,
    hit_error,
    make_observations,
    run_mcmc,
    simulate_bounces,
)


class TestForwardModel:
    def test_deterministic(self):
        a = simulate_bounces(45.0, 6.0, *TRUE_PARAMS)
        b = simulate_bounces(45.0, 6.0, *TRUE_PARAMS)
        np.testing.assert_array_equal(a, b)

    def test_drag_shortens_range(self):
        no_drag = simulate_bounces(45.0, 6.0, 0.0, 0.82, 0.15)
        with_drag = simulate_bounces(45.0, 6.0, 0.47, 0.82, 0.15)
        assert with_drag[-1] < no_drag[-1]

    def test_higher_restitution_longer_range(self):
        low_e = simulate_bounces(45.0, 6.0, 0.47, 0.60, 0.15)
        high_e = simulate_bounces(45.0, 6.0, 0.47, 0.90, 0.15)
        assert high_e[-1] > low_e[-1]


class TestCalibration:
    @pytest.fixture(scope="class")
    def posterior(self):
        obs, thetas, v0s = make_observations(n_shots=10, seed=0)
        samples, _ = run_mcmc(obs, thetas, v0s, n_chains=4, n_steps=400,
                              n_burn=150, seed=1)
        return samples

    def test_posterior_mean_near_truth(self, posterior):
        mean = posterior.mean(axis=0)
        rel_err = np.abs(mean - TRUE_PARAMS) / TRUE_PARAMS
        assert np.all(rel_err < 0.30), f"rel err {rel_err}"

    def test_calibration_beats_ideal_physics(self, posterior):
        mean = posterior.mean(axis=0)
        theta_cal, v0_cal = aim(mean)
        theta_ideal, v0_ideal = aim(np.array([0.0, 1.0, 0.0]))
        assert hit_error(theta_cal, v0_cal) < hit_error(theta_ideal, v0_ideal)
