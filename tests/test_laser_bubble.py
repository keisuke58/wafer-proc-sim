"""Tests for demos/laser_bubble.py — SD thermal runaway + R-P cavitation."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demos.laser_bubble import (
    T_MELT,
    T_ROOM,
    absorption_coeff,
    focal_temperature,
    max_radius,
    runaway_threshold,
    simulate_bubble,
)


class TestThermalRunaway:
    def test_absorption_increases_with_temperature(self):
        T = np.array([300.0, 800.0, 1400.0])
        a = absorption_coeff(T)
        assert np.all(np.diff(a) > 0)

    def test_below_threshold_no_melt(self):
        i_th = runaway_threshold()
        _, T = focal_temperature(0.5 * i_th)
        assert T.max() < T_MELT

    def test_above_threshold_melts(self):
        i_th = runaway_threshold()
        _, T = focal_temperature(1.2 * i_th)
        assert T.max() >= T_MELT

    def test_temperature_starts_at_room(self):
        _, T = focal_temperature(1e13)
        assert T[0] == pytest.approx(T_ROOM)


class TestCavitationBubble:
    def test_bubble_grows_then_collapses(self):
        t, r = simulate_bubble(p_gas0=5e6)
        i_max = int(np.argmax(r))
        assert r[i_max] > 3.0 * r[0]          # significant growth
        assert 0 < i_max < len(r) - 1         # interior maximum
        assert r[i_max:].min() < 0.5 * r[i_max]  # collapse afterwards

    def test_max_radius_increases_with_pressure(self):
        radii = [max_radius(p) for p in (2e6, 5e6, 10e6)]
        assert radii[0] < radii[1] < radii[2]

    def test_radius_stays_positive(self):
        _, r = simulate_bubble(p_gas0=10e6)
        assert np.all(r > 0)
