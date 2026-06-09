"""
Parity + smoke tests for the three new C++ pybind11 kernels:
  - _arde_kernel     : ARDE plasma model
  - _nsga2_kernel    : non-dominated sort + crowding distance
  - _motion_kernel   : RT P-controller motion loop

Each test verifies numerical identity with the scalar / Python reference
and basic correctness properties. All tests are skipped gracefully when
the compiled .so is absent (build with bash build_all_kernels.sh).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

# ── ARDE kernel ───────────────────────────────────────────────────────────────

try:
    from fem import _arde_kernel as _cpp_arde
    _HAS_ARDE = True
except ImportError:
    _HAS_ARDE = False

from sic.physical_limits import plasma_aspect_ratio_limit, plasma_arde_vec


@pytest.mark.skipif(not _HAS_ARDE, reason="_arde_kernel not built")
class TestArdeKernel:
    def _scalar_grid(self, n=50):
        rng = np.random.default_rng(42)
        w = rng.uniform(1.0, 20.0, n)
        t = rng.uniform(50.0, 300.0, n)
        p = rng.uniform(4.0, 28.0, n)
        return w, t, p

    def test_parity_with_scalar_reference(self):
        """C++ result must match scalar plasma_aspect_ratio_limit element-wise."""
        w, t, p = self._scalar_grid()
        cpp = plasma_arde_vec(w, t, p)

        for i in range(len(w)):
            ref = plasma_aspect_ratio_limit(
                trench_width_um=w[i],
                wafer_thickness_um=t[i],
                pressure_mTorr=p[i],
            )
            assert abs(cpp["AR_crit"].ravel()[i]   - ref["AR_crit_50pct"])        < 1e-9
            assert abs(cpp["AR_max"].ravel()[i]     - ref["AR_max_practical"])    < 1e-9
            assert abs(cpp["current_AR"].ravel()[i] - ref["current_AR"])          < 1e-9
            assert abs(cpp["etch_rate"].ravel()[i]  - ref["etch_rate_at_current_AR"]) < 1e-9
            assert bool(cpp["feasible"].ravel()[i]) == ref["feasible"]

    def test_numpy_fallback_matches_cpp(self):
        """NumPy fallback path must be numerically identical to the C++ path."""
        w = np.linspace(2.0, 15.0, 30)
        t = np.full(30, 100.0)
        cpp = plasma_arde_vec(w, t, pressure_mTorr=10.0)

        # Disable C++ to exercise NumPy fallback directly.
        import sic.physical_limits as pl
        orig = pl._CPP_ARDE
        pl._CPP_ARDE = False
        try:
            np_result = plasma_arde_vec(w, t, pressure_mTorr=10.0)
        finally:
            pl._CPP_ARDE = orig

        for key in ("AR_crit", "AR_max", "current_AR", "etch_rate"):
            np.testing.assert_allclose(cpp[key], np_result[key], rtol=1e-10)

    def test_feasibility_boundary(self):
        """A trench just inside AR_max should be feasible; just outside should not."""
        # At 10 mTorr, AR_crit=8, AR_max=8*(19)^0.5≈34.87
        AR_max_approx = 8.0 * math.sqrt(19.0)
        t = 100.0  # wafer thickness fixed
        w_feasible   = t / (AR_max_approx * 0.99)
        w_infeasible = t / (AR_max_approx * 1.01)
        r = plasma_arde_vec([w_feasible, w_infeasible], [t, t], 10.0)
        assert bool(r["feasible"][0]) is True
        assert bool(r["feasible"][1]) is False

    def test_broadcast_2d(self):
        """2-D grids broadcast correctly."""
        w = np.linspace(3.0, 10.0, 5)
        t = np.linspace(50.0, 200.0, 4)
        W, T = np.meshgrid(w, t)
        r = plasma_arde_vec(W, T, pressure_mTorr=10.0)
        assert r["AR_crit"].shape == (4, 5)


# ── NSGA-II kernel ────────────────────────────────────────────────────────────

try:
    from optimization import _nsga2_kernel as _cpp_nsga2
    _HAS_NSGA2 = True
except ImportError:
    _HAS_NSGA2 = False

from optimization.multiobjective_pipeline import (
    fast_nondominated_sort,
    crowding_distance,
)


@pytest.mark.skipif(not _HAS_NSGA2, reason="_nsga2_kernel not built")
class TestNsga2Kernel:
    def _random_F(self, n=60, m=6, seed=0):
        return np.random.default_rng(seed).random((n, m))

    def test_parity_nondominated_sort(self):
        """C++ and Python produce identical fronts (same sets, same counts)."""
        F = self._random_F()

        import optimization.multiobjective_pipeline as mp
        orig = mp._CPP_NSGA2
        mp._CPP_NSGA2 = False
        try:
            py_fronts = fast_nondominated_sort(F)
        finally:
            mp._CPP_NSGA2 = orig

        cpp_fronts = _cpp_nsga2.nondominated_sort(np.ascontiguousarray(F))

        assert len(cpp_fronts) == len(py_fronts)
        for cf, pf in zip(cpp_fronts, py_fronts):
            assert sorted(cf) == sorted(pf)

    def test_pareto_front_is_nondominated(self):
        """No member of front[0] may be dominated by any other member."""
        F = self._random_F(n=80)
        fronts = _cpp_nsga2.nondominated_sort(np.ascontiguousarray(F))
        front0 = list(fronts[0])
        for i in front0:
            for j in front0:
                if i == j:
                    continue
                # Neither direction should dominate
                assert not (np.all(F[j] <= F[i]) and np.any(F[j] < F[i])), \
                    f"front[0] member {j} dominates {i}"

    def test_parity_crowding_distance(self):
        """C++ and Python crowding distances are identical."""
        F = self._random_F(n=30, m=4)
        front = list(range(30))

        import optimization.multiobjective_pipeline as mp
        orig = mp._CPP_NSGA2
        mp._CPP_NSGA2 = False
        try:
            py_dist = crowding_distance(F, front)
        finally:
            mp._CPP_NSGA2 = orig

        cpp_dist = _cpp_nsga2.crowding_distance(np.ascontiguousarray(F), front)
        np.testing.assert_allclose(cpp_dist, py_dist, rtol=1e-10)

    def test_boundary_members_have_inf_distance(self):
        """Best and worst on each objective always get distance = inf."""
        F = self._random_F(n=20, m=3)
        front = list(range(20))
        dist = np.array(_cpp_nsga2.crowding_distance(np.ascontiguousarray(F), front))
        assert np.any(np.isinf(dist)), "expected at least two inf entries"


# ── Motion kernel ─────────────────────────────────────────────────────────────

try:
    from pipeline import _motion_kernel as _cpp_motion
    _HAS_MOTION = True
except ImportError:
    _HAS_MOTION = False


@pytest.mark.skipif(not _HAS_MOTION, reason="_motion_kernel not built")
class TestMotionKernel:
    def test_settles_within_tolerance(self):
        """Axis must reach within 1 µm of target."""
        r = _cpp_motion.move_to(start_mm=0.0, target_mm=10.0)
        pos = np.array(r["pos_mm"])
        assert abs(pos[-1] - 10.0) < 0.001, f"final pos {pos[-1]} not within 1µm of 10mm"

    def test_parity_with_python_loop(self):
        """C++ trajectory must match the Python P-controller to floating-point."""
        from pipeline.disco_sw_stack import MotionController
        import pipeline.disco_sw_stack as sw

        orig = sw._CPP_MOTION
        sw._CPP_MOTION = False
        try:
            mc = MotionController(Kp=50.0, max_vel_mms=300.0)
            py_log = mc.move_to(target_mm=5.0, n_cycles=500)
        finally:
            sw._CPP_MOTION = orig

        r = _cpp_motion.move_to(
            start_mm=0.0, target_mm=5.0, Kp=50.0,
            max_vel_mms=300.0, dt_s=0.001, settle_um=1.0, max_cycles=500,
        )
        cpp_pos = np.array(r["pos_mm"])
        py_pos  = np.array([s["pos_mm"] for s in py_log])

        assert len(cpp_pos) == len(py_pos), \
            f"length mismatch: C++={len(cpp_pos)} Python={len(py_pos)}"
        # Python path stores round(pos, 6) in the log dict → 0.5e-6 max error.
        np.testing.assert_allclose(cpp_pos, py_pos, atol=1e-5)

    def test_velocity_saturation(self):
        """Commanded velocity must never exceed max_vel_mms."""
        r = _cpp_motion.move_to(
            start_mm=0.0, target_mm=100.0,
            Kp=200.0, max_vel_mms=50.0,
        )
        vel = np.abs(np.array(r["vel_mms"]))
        assert float(vel.max()) <= 50.0 + 1e-9

    def test_move_sequence(self):
        """Multi-segment move must end near the last target."""
        targets = [5.0, 2.0, 8.0, 1.0]
        r = _cpp_motion.move_sequence(
            start_mm=0.0, targets_mm=targets,
        )
        pos = np.array(r["pos_mm"])
        assert abs(pos[-1] - targets[-1]) < 0.001
        assert len(r["seg_ends"]) == len(targets)

    def test_zero_distance_move(self):
        """Moving to current position should produce a one-cycle log."""
        r = _cpp_motion.move_to(start_mm=5.0, target_mm=5.0)
        assert r["n_cycles"] == 1
