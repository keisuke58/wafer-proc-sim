"""Tests for accretech/surface_roundness — roughness + roundness metrology."""
import numpy as np
import pytest

sr = pytest.importorskip("accretech.surface_roundness")


class TestRoughness:
    def test_gaussian_filter_is_50pct_at_cutoff(self):
        """ISO 16610-21 definition: amplitude transmission = 50% at λc."""
        assert abs(sr.filter_transmission(sr.LC_MM) - 0.5) < 0.01

    def test_transmission_monotone(self):
        t_short = sr.filter_transmission(sr.LC_MM / 4)
        t_long = sr.filter_transmission(sr.LC_MM * 4)
        assert t_short > 0.95 and t_long < 0.1

    def test_ra_of_pure_sine_is_2A_over_pi(self):
        """Analytic closed loop: Ra(sine, amplitude A) = 2A/π."""
        n = int(5 * sr.LC_MM / sr.DX_MM)
        x = np.arange(n) * sr.DX_MM
        A = 0.5
        z = A * np.sin(2 * np.pi * x / (sr.LC_MM / 8.0))
        ra, rq, rz = sr.roughness_params(z)
        assert abs(ra - 2 * A / np.pi) / (2 * A / np.pi) < 0.02
        assert abs(rz - 2 * A) / (2 * A) < 0.05


class TestRoundness:
    def test_lsc_center_recovers_eccentricity(self):
        th, r = sr.synth_round(ecc_um=8.0, noise_um=0.0, seed=3)
        _, (cx, cy) = sr.roundness_lsc(th, r)
        assert abs(np.hypot(cx, cy) - 8.0) < 0.05

    def test_mzc_never_worse_than_lsc(self):
        th, r = sr.synth_round()
        lsc, _ = sr.roundness_lsc(th, r)
        mzc, _ = sr.roundness_mzc(th, r)
        assert mzc <= lsc + 1e-9

    def test_harmonics_recover_injected_amplitudes(self):
        th, r = sr.synth_round(ecc_um=8.0, oval_um=3.0, lobe3_um=2.0,
                               noise_um=0.0, seed=4)
        h = sr.harmonics(th, r)
        assert abs(h[0] - 8.0) < 0.05     # k=1 eccentricity
        assert abs(h[1] - 3.0) < 0.05     # k=2 ovality
        assert abs(h[2] - 2.0) < 0.05     # k=3 lobing
        assert h[4] < 0.05                # nothing injected at k=5
