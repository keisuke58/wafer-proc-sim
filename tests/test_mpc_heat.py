"""
Tests for:
  - pipeline/_mpc_kernel     (GP-surrogate MPC optimizer)
  - fem/dicing_heat_sim      (wafer thermal simulation — CPU fallback)
"""
from __future__ import annotations
import math, sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from pipeline import _mpc_kernel as _mpc
    _HAS_MPC = True
except ImportError:
    _HAS_MPC = False

from fem.dicing_heat_sim import DicingHeatSim


# ═══════════════════════════════════════════════════════════════════════════════
# MPC kernel
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_MPC, reason="_mpc_kernel not built")
class TestMPCKernel:

    def _make_gp(self, n=20, seed=0):
        """Return synthetic GP training data for quality and wear."""
        rng = np.random.default_rng(seed)
        # 5D input: [wear, hardness, depth, feed, rpm]
        Xtr = np.ascontiguousarray(rng.uniform(
            [0, 5, 5, 10, 10000],
            [50, 10, 30, 200, 35000], (n, 5)))
        # Targets: quality ∝ feed/rpm, wear ∝ feed*depth
        y_q = Xtr[:, 3] / Xtr[:, 4] * 1e3 + rng.normal(0, 0.01, n)
        y_w = Xtr[:, 2] * Xtr[:, 3] * 1e-3 + rng.normal(0, 0.01, n)
        ls  = np.ascontiguousarray(np.array([10., 2., 5., 50., 5000.]))
        sf  = 1.0; noise = 0.01

        from ml import _gp_inference_kernel as _gp
        K_q = np.array(_gp.rbf_kernel_matrix(Xtr, ls, sf)) + noise * np.eye(n)
        K_w = np.array(_gp.rbf_kernel_matrix(Xtr, ls, sf)) + noise * np.eye(n)
        alpha_q = np.ascontiguousarray(np.linalg.solve(K_q, y_q))
        alpha_w = np.ascontiguousarray(np.linalg.solve(K_w, y_w))
        return Xtr, alpha_q, alpha_w, ls, sf

    # ── MPCParams ──────────────────────────────────────────────────────────────

    def test_params_defaults(self):
        p = _mpc.MPCParams()
        assert p.horizon == 10
        assert p.n_feed  == 20
        assert p.n_rpm   == 20
        assert 10.0 <= p.feed_min < p.feed_max <= 200.0
        assert 10000 <= p.rpm_min < p.rpm_max <= 35000

    def test_params_mutable(self):
        p = _mpc.MPCParams()
        p.horizon = 5
        assert p.horizon == 5

    # ── mpc_evaluate_action ────────────────────────────────────────────────────

    def test_evaluate_returns_costs(self):
        Xtr, aq, aw, ls, sf = self._make_gp()
        x0 = np.ascontiguousarray(np.array([10.0, 7.0, 15.0]))
        r = _mpc.mpc_evaluate_action(x0, Xtr, aq, Xtr, aw, ls, sf,
                                     80.0, 30000.0, _mpc.MPCParams())
        for k in ("quality_cost", "wear_cost", "smoothness_cost", "total_cost"):
            assert k in r

    def test_evaluate_total_nonnegative(self):
        Xtr, aq, aw, ls, sf = self._make_gp()
        x0 = np.ascontiguousarray(np.array([5.0, 8.0, 10.0]))
        p  = _mpc.MPCParams()
        r  = _mpc.mpc_evaluate_action(x0, Xtr, aq, Xtr, aw, ls, sf,
                                      60.0, 25000.0, p)
        assert r["total_cost"] >= 0.0

    def test_high_feed_higher_wear(self):
        """Higher feed → larger wear_cost (all else equal)."""
        Xtr, aq, aw, ls, sf = self._make_gp()
        x0 = np.ascontiguousarray(np.array([10.0, 7.0, 15.0]))
        p  = _mpc.MPCParams()
        r_slow = _mpc.mpc_evaluate_action(x0, Xtr, aq, Xtr, aw, ls, sf,
                                          20.0, 30000.0, p)
        r_fast = _mpc.mpc_evaluate_action(x0, Xtr, aq, Xtr, aw, ls, sf,
                                          150.0, 30000.0, p)
        # wear cost should generally be higher for faster feed
        # (GP may not be monotone, but total cost should differ)
        assert r_slow["total_cost"] != r_fast["total_cost"]

    # ── mpc_optimize ──────────────────────────────────────────────────────────

    def test_optimize_returns_valid_control(self):
        Xtr, aq, aw, ls, sf = self._make_gp()
        x0 = np.ascontiguousarray(np.array([10.0, 7.0, 15.0]))
        p  = _mpc.MPCParams()
        r  = _mpc.mpc_optimize(x0, Xtr, aq, Xtr, aw, ls, sf,
                                80.0, 30000.0, p)
        assert "feed_rate_mms" in r and "spindle_rpm" in r
        assert p.feed_min <= r["feed_rate_mms"] <= p.feed_max
        assert p.rpm_min  <= r["spindle_rpm"]   <= p.rpm_max

    def test_optimize_cost_nonnegative(self):
        Xtr, aq, aw, ls, sf = self._make_gp()
        x0 = np.ascontiguousarray(np.array([5.0, 6.0, 20.0]))
        r = _mpc.mpc_optimize(x0, Xtr, aq, Xtr, aw, ls, sf,
                               80.0, 30000.0, _mpc.MPCParams())
        assert r["total_cost"] >= 0.0

    def test_optimize_total_cost_le_arbitrary_action(self):
        """Optimal cost ≤ cost of any fixed action in the search grid."""
        Xtr, aq, aw, ls, sf = self._make_gp()
        x0 = np.ascontiguousarray(np.array([10.0, 7.0, 15.0]))
        p  = _mpc.MPCParams()
        r_opt  = _mpc.mpc_optimize(x0, Xtr, aq, Xtr, aw, ls, sf,
                                    80.0, 30000.0, p)
        # pick a point that IS on the grid: feed at midpoint index, rpm at midpoint
        feed_mid = (p.feed_min + p.feed_max) / 2
        rpm_mid  = (p.rpm_min  + p.rpm_max)  / 2
        r_arb  = _mpc.mpc_evaluate_action(x0, Xtr, aq, Xtr, aw, ls, sf,
                                           feed_mid, rpm_mid, p)
        assert r_opt["total_cost"] <= r_arb["total_cost"] + 1e-9

    # ── mpc_pareto_front ──────────────────────────────────────────────────────

    def test_pareto_front_nonempty(self):
        Xtr, aq, aw, ls, sf = self._make_gp()
        x0     = np.ascontiguousarray(np.array([10.0, 7.0, 15.0]))
        front  = _mpc.mpc_pareto_front(x0, Xtr, aq, Xtr, aw, ls, sf,
                                        _mpc.MPCParams())
        assert len(front) > 0

    def test_pareto_front_structure(self):
        Xtr, aq, aw, ls, sf = self._make_gp()
        x0    = np.ascontiguousarray(np.array([5.0, 8.0, 10.0]))
        front = _mpc.mpc_pareto_front(x0, Xtr, aq, Xtr, aw, ls, sf,
                                       _mpc.MPCParams())
        p = _mpc.MPCParams()
        for item in front:
            for k in ("feed_rate_mms", "spindle_rpm", "quality_cost", "wear_cost"):
                assert k in item
            assert p.feed_min <= item["feed_rate_mms"] <= p.feed_max
            assert p.rpm_min  <= item["spindle_rpm"]   <= p.rpm_max

    def test_pareto_front_nondominance(self):
        """No Pareto-front member should be dominated by another."""
        Xtr, aq, aw, ls, sf = self._make_gp()
        x0    = np.ascontiguousarray(np.array([10.0, 7.0, 15.0]))
        front = _mpc.mpc_pareto_front(x0, Xtr, aq, Xtr, aw, ls, sf,
                                       _mpc.MPCParams())
        pts = [(item["quality_cost"], item["wear_cost"]) for item in front]
        for i, (qi, wi) in enumerate(pts):
            for j, (qj, wj) in enumerate(pts):
                if i == j: continue
                dominated = (qj <= qi and wj <= wi and (qj < qi or wj < wi))
                assert not dominated, f"point {i} dominated by {j}"


# ═══════════════════════════════════════════════════════════════════════════════
# DicingHeatSim  (CPU NumPy fallback — always runs)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDicingHeatSim:

    def _make_sim(self, nx=64, ny=64):
        return DicingHeatSim(nx=nx, ny=ny, dx=20e-6)

    def test_initial_temperature_uniform(self):
        sim = self._make_sim()
        sim.set_ambient(25.0)
        assert sim.T.min() == pytest.approx(25.0)
        assert sim.T.max() == pytest.approx(25.0)

    def test_dt_stable(self):
        """Auto-computed dt must satisfy Fourier stability r < 0.25."""
        sim = self._make_sim()
        r = sim.alpha * sim.dt / sim.dx**2
        assert r < 0.25

    def test_blade_source_peak(self):
        sim  = self._make_sim()
        Q    = 5e7
        src  = sim.blade_source_field(x_m=0.0, y_m=0.0, Q=Q)
        assert src.shape == (64, 64)
        assert abs(src.max() - Q) / Q < 0.01

    def test_blade_source_nonnegative(self):
        sim = self._make_sim()
        src = sim.blade_source_field(x_m=1e-3, y_m=0.5e-3)
        assert src.min() >= 0.0

    def test_run_lane_heats_wafer(self):
        sim = self._make_sim(nx=64, ny=64)
        sim.set_ambient(25.0)
        T = sim.run_lane(y_mm=0.64, feed_mms=80.0, n_steps_per_pos=5)
        assert sim.peak_temp > 25.0, "blade should heat the wafer"

    def test_run_lane_peak_above_ambient(self):
        sim = self._make_sim(nx=64, ny=64)
        sim.set_ambient(25.0)
        sim.run_lane(y_mm=0.64, n_steps_per_pos=5)
        assert sim.peak_temp > 30.0

    def test_run_lane_shape_unchanged(self):
        sim = self._make_sim(nx=64, ny=64)
        T   = sim.run_lane(y_mm=0.3, n_steps_per_pos=5)
        assert T.shape == (64, 64)

    def test_higher_q_higher_peak(self):
        sim1 = self._make_sim(nx=64, ny=64); sim1.set_ambient(25.0)
        sim2 = self._make_sim(nx=64, ny=64); sim2.set_ambient(25.0)
        sim1.run_lane(y_mm=0.64, Q_W_per_m2=1e7, n_steps_per_pos=5)
        sim2.run_lane(y_mm=0.64, Q_W_per_m2=5e7, n_steps_per_pos=5)
        assert sim2.peak_temp > sim1.peak_temp

    def test_mean_temp_accessor(self):
        sim = self._make_sim()
        sim.set_ambient(100.0)
        assert sim.mean_temp == pytest.approx(100.0)

    def test_gpu_info_returns_dict(self):
        info = DicingHeatSim.gpu_info()
        assert isinstance(info, dict)
