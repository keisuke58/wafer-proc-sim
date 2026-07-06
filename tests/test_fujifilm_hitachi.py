"""Tests for fujifilm/ (resist physics) and hitachi/ (CD-SEM metrology)."""
import numpy as np
import pytest

cs = pytest.importorskip("fujifilm.car_stochastics")
rd = pytest.importorskip("fujifilm.resist_develop")
ht = pytest.importorskip("hitachi.cdsem")


class TestCarStochastics:
    def test_ler_follows_shot_noise_law(self):
        """LER ~ dose^p with p near the counting-statistics -1/2."""
        p, _ = cs.dose_exponent()
        assert -0.65 < p < -0.30

    def test_more_dose_less_roughness(self):
        lo = np.mean([cs.ler_3sigma(0.5, seed=s) for s in range(3)])
        hi = np.mean([cs.ler_3sigma(4.0, seed=s) for s in range(3)])
        assert hi < lo

    def test_diffusion_has_interior_optimum(self):
        """Too little blur: shot noise; too much: edge slope dies."""
        l_small = np.mean([cs.ler_3sigma(1.0, 2.0, s) for s in range(3)])
        l_mid = np.mean([cs.ler_3sigma(1.0, 16.0, s) for s in range(3)])
        l_big = np.mean([cs.ler_3sigma(1.0, 40.0, s) for s in range(3)])
        assert l_mid < l_small and l_mid < l_big


class TestResistDevelop:
    def test_clear_time_closed_loop(self):
        """At E0 the bright column clears in exactly T_DEV."""
        e0 = rd.dose_to_clear(8.0)
        _, _, t = rd.clear_time(e0, 8.0)
        assert abs(t[-1, rd.NX // 2] - rd.T_DEV) / rd.T_DEV < 0.01

    def test_selectivity_raises_contrast(self):
        gammas = [rd.characteristic_curve(n)[2] for n in (2.0, 8.0, 16.0)]
        assert gammas[0] < gammas[1] < gammas[2]

    def test_selectivity_steepens_sidewall(self):
        angs = []
        for n in (2.0, 16.0):
            e0 = rd.dose_to_clear(n)
            _, _, ang = rd.profile(1.35 * e0, n)
            angs.append(ang)
        assert angs[1] > angs[0] > 60.0


class TestCdSem:
    def test_beam_size_biases_cd_outward(self):
        """Bigger beam -> wider white bands -> larger raw CD."""
        biases = [ht.detect_cd(*ht.se_profile(beam_nm=b)) - ht.CD_TRUE
                  for b in (2.0, 5.0, 8.0)]
        assert biases[0] < biases[1] < biases[2]
        assert biases[0] > 0.0

    def test_ler_naive_overestimates_and_unbias_recovers(self):
        rng = np.random.default_rng(7)
        naive, unb = ht.ler_unbias(rng, frames=1)
        assert naive > ht.LER_TRUE_NM * 1.1          # bias visible
        assert abs(unb - ht.LER_TRUE_NM) < 0.35      # recovered
        assert abs(unb - ht.LER_TRUE_NM) < abs(naive - ht.LER_TRUE_NM)

    def test_more_frames_less_noise_bias(self):
        rng = np.random.default_rng(3)
        naive1, _ = ht.ler_unbias(rng, frames=1)
        naive8, _ = ht.ler_unbias(rng, frames=8)
        assert naive8 < naive1

    def test_shrink_extrapolation_beats_naive_average(self):
        rng = np.random.default_rng(7)
        cds = ht.cd_vs_frames(rng)
        ref = float(np.nanmean(
            [ht.detect_cd(*ht.noisy_profile(rng, ht.CD_TRUE, 1))
             for _ in range(200)]))
        err_naive = abs(float(np.mean(cds)) - ref)
        err_ext = abs(ht.shrink_extrapolate(cds) - ref)
        assert err_ext < 0.8
        assert err_naive > 2.0 * err_ext
