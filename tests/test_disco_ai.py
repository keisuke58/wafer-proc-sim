"""Tests for the DISCO AI pair: AE waveform anomaly + deep active learning."""
import numpy as np
import pytest

pytest.importorskip("torch")

ae = pytest.importorskip("ml.cutting_ae_anomaly")
al = pytest.importorskip("vision.adc_active_learning")


class TestAeAnomaly:
    def test_gain_drift_is_normalized_out(self):
        """The spectrum features must be insensitive to sensor gain — the
        nuisance that kills raw-RMS monitoring."""
        rng = np.random.default_rng(0)
        w = ae.synth_cut("normal", rng=rng)
        s1 = ae.log_spectrum(w)
        s2 = ae.log_spectrum(3.0 * w)
        assert np.allclose(s1, s2, atol=1e-5)

    def test_self_supervised_detector_beats_rms_baseline(self):
        """Small-budget end-to-end run: the normal-only autoencoder must
        separate every failure mode far better than the RMS chart."""
        res, _ = ae.run(seed=1, n_train=120, n_eval=50, epochs=80)
        for kind in ("chipping", "glazing", "crack"):
            assert res["auroc_ae"][kind] > 0.9
            assert res["auroc_ae"][kind] > res["auroc_rms"][kind] + 0.15

    def test_progressive_glazing_alarm_is_prompt(self):
        res, _ = ae.run(seed=2, n_train=120, n_eval=50, epochs=80)
        d = res["glazing_stream"]["delay_ae_cuts"]
        assert d is not None and d <= 30


class TestActiveLearning:
    def test_uncertainty_selection_prefers_ambiguous_images(self):
        """After a short warm-up training, entropy selection must pick
        samples with higher predictive entropy than a random draw."""
        (X, y), (X_te, y_te) = al.prepare_data(n_per_class=60, seed=1)
        rng = np.random.default_rng(0)
        model = al._build(0)
        model = al._train(model, X[:30], y[:30], epochs=10, seed=0)
        pool = np.arange(30, len(X))
        picked = al._select("entropy", model, X, pool, 10, rng)
        p = al._probs(model, X)
        ent = -(p * np.log(p + 1e-12)).sum(axis=1)
        assert ent[picked].mean() > ent[pool].mean()
        assert len(np.unique(picked)) == 10

    def test_learning_improves_with_budget(self):
        (X, y), (X_te, y_te) = al.prepare_data(n_per_class=60, seed=2)
        import vision.adc_active_learning as m
        saved = (m.INIT_LABELS, m.BATCH, m.ROUNDS)
        m.INIT_LABELS, m.BATCH, m.ROUNDS = 20, 20, 2
        try:
            b, a = al.run_strategy("entropy", X, y, X_te, y_te, seed=5)
        finally:
            m.INIT_LABELS, m.BATCH, m.ROUNDS = saved
        assert a[-1] > a[0] - 0.02      # no collapse, and normally improves
        assert a[-1] > 0.6
        assert b[-1] == 60
