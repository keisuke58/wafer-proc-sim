"""Tests for the Kioxia (NAND) and TEL (etch/ALD) modules."""
import numpy as np
import pytest

ns = pytest.importorskip("memdev.nand_sim")
ea = pytest.importorskip("telproc.etch_ald")


class TestNand:
    def test_ispp_distribution_width_tracks_step(self):
        """ISPP theory: programmed-state spread ~ uniform(step) + noise."""
        dists = ns.ispp_distribution(n=50_000, seed=1)
        sd = np.std(dists[3])
        expect = np.sqrt(ns.SIGMA0 ** 2 + ns.DV_PGM ** 2 / 12.0)
        assert abs(sd - expect) / expect < 0.05

    def test_rber_monotonic_in_cycles_and_time(self):
        assert ns.rber(0, 0) < ns.rber(1000, 0) < ns.rber(10000, 0)
        assert ns.rber(3000, 0) < ns.rber(3000, 720) < ns.rber(3000, 8760)

    def test_temperature_accelerates_retention(self):
        assert ns.rber(3000, 8760, 85) > ns.rber(3000, 8760, 55)

    def test_ecc_cliff(self):
        """BCH-72: astronomically safe at 1e-3 raw, dead at 2e-2."""
        assert ns.uber_bch(1e-3) < 1e-30
        assert ns.uber_bch(2e-2) > 0.9

    def test_endurance_rating_in_tlc_regime(self):
        e = ns.endurance_rating()
        assert 1000 < e < 50000
        assert ns.endurance_rating(temp_C=85) < e


class TestEtchAld:
    def test_arde_narrow_trench_lags(self):
        _, d20 = ea.arde_depth(20.0)
        _, d100 = ea.arde_depth(100.0)
        assert d20[-1] < d100[-1] * 0.75

    def test_passivation_sweeps_bow_to_taper(self):
        z, h_low, d = ea.etch_profile(passivation=0.2)
        z, h_high, d2 = ea.etch_profile(passivation=0.85)
        m = z < min(d, d2)
        # low passivation: max width well beyond the mask opening (bow)
        assert h_low[m].max() > 25.0 * 1.5
        # heavy passivation: bottom narrower than the top (taper)
        zz = z[m]
        top = h_high[m][zz < 0.2 * zz.max()].mean()
        bot = h_high[m][zz > 0.8 * zz.max()].mean()
        assert bot < top

    def test_ald_self_limiting_and_front_like(self):
        z, th = ea.ald_coverage(40.0, dose=10.0)
        assert th.max() <= 1.0
        assert th[0] > 0.99          # mouth saturates first
        assert th[-1] < 0.5          # bottom still open at this dose

    def test_ald_dose_scales_as_ar_squared(self):
        d10 = ea.dose_for_bottom_coverage(10.0)
        d40 = ea.dose_for_bottom_coverage(40.0)
        slope = np.log(d40 / d10) / np.log(4.0)
        assert 1.6 < slope < 2.4
