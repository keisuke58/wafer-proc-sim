"""Tests for the Lasertec deepening pair: EUV printability + photon budget."""
import numpy as np
import pytest

ep = pytest.importorskip("inspection.euv_printability")
pb = pytest.importorskip("inspection.photon_budget")


class TestPrintability:
    def test_reference_prints_nominal_cd(self):
        """Threshold calibration: the defect-free aerial image must print
        the nominal 16 nm line within half a nanometer."""
        I0 = ep.aerial_image(ep.mask_field(None))
        cd = ep.measure_cd(I0, ep.PITCH / 4)
        assert abs(cd - ep.PITCH / 2) < 0.5

    def test_dcd_grows_with_particle_size(self):
        d = [abs(ep.delta_cd(("particle", ep.PITCH * 0.75, 0.0, s)))
             for s in (5.0, 8.0, 12.0)]
        assert d[0] < d[1] < d[2]

    def test_phase_defect_prints_worse_out_of_focus(self):
        """The classic ML-bump signature: through-focus asymmetric, worst
        at negative defocus, mild at best focus."""
        b = ("bump", ep.PITCH * 0.75, 0.0, 16.0)
        d0 = abs(ep.delta_cd(b, 0.0))
        dm = abs(ep.delta_cd(b, -25.0))
        assert dm > d0

    def test_seeing_is_not_printing(self):
        """A small particle can be MORE visible than a phase bump that
        prints WORSE — the disposition argument in one assertion."""
        small_particle = ("particle", ep.PITCH * 0.75, 0.0, 6.0)
        bump = ("bump", ep.PITCH * 0.75, 0.0, 16.0)
        det_p = ep.detection_score(small_particle)
        det_b = ep.detection_score(bump)
        dcd_p = max(abs(ep.delta_cd(small_particle, z)) for z in (0, -25))
        dcd_b = max(abs(ep.delta_cd(bump, z)) for z in (0, -25))
        assert dcd_b > 10.0 > dcd_p          # bump prints, particle doesn't
        assert det_p > 0.5 * det_b           # yet visibility is comparable


class TestPhotonBudget:
    def test_threshold_matches_false_budget(self):
        from scipy.stats import norm
        k = pb.threshold_sigma()
        assert abs(pb.N_PIX * norm.sf(k) - pb.FALSE_PER_MASK) < 1e-6 \
            * pb.N_PIX * norm.sf(k) + 1e-9 or \
            np.isclose(pb.N_PIX * norm.sf(k), pb.FALSE_PER_MASK, rtol=1e-6)

    def test_capture_improves_with_dwell(self):
        dws = [pb.dwell_for_masks_per_day(m) for m in (4.0, 2.0, 1.0)]
        caps = [pb.capture_prob(16.0, d) for d in dws]
        assert caps[0] < caps[1] < caps[2]

    def test_source_power_buys_sensitivity(self):
        dw = pb.dwell_for_masks_per_day(2.0)
        m1 = pb.min_detectable_size(dw, pb.PHI_1X)
        m2 = pb.min_detectable_size(dw, 2 * pb.PHI_1X)
        assert m2 < m1
        # SNR ∝ √N ⇒ size² ∝ 1/√phi ⇒ size ratio = 2^(-1/4)
        assert np.isclose(m2 / m1, 2 ** -0.25, rtol=0.02)
