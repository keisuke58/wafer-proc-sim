"""Tests for AP試験-coverage modules:
  embedded/rtos_scheduler.cpp   → RTOS / 割り込み / タスク管理
  pipeline/spc_monitor.cpp      → 品質管理 / SPC / Cpk
  machine/reliability_kernel.cpp→ 信頼性 / RASIS / OEE
  data/process_db.py            → データベース / SQL
"""
from __future__ import annotations
import sys, os
import pytest
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ══════════════════════════════════════════════════════════════════════════════
# RTOS Scheduler
# ══════════════════════════════════════════════════════════════════════════════

class TestRtosScheduler:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("embedded._rtos_scheduler_kernel",
                            reason="RTOS kernel not built; run build_all_kernels.sh rtos")
        from embedded import _rtos_scheduler_kernel as rtos
        self.rtos = rtos

    def test_disco_tasks_schedulable(self):
        """DISCO task set must satisfy RM utilization bound."""
        r = self.rtos.simulate_disco_tasks(200000)
        assert r["rm_schedulable"] is True, (
            f"U={r['total_utilization']:.3f} > RM bound {r['rm_bound']:.3f}"
        )

    def test_disco_utilization_range(self):
        """Total utilization must be approximately 0.151 (within 1%)."""
        r = self.rtos.simulate_disco_tasks(200000)
        assert abs(r["total_utilization"] - 0.151) < 0.002

    def test_four_tasks_returned(self):
        r = self.rtos.simulate_disco_tasks(100000)
        assert len(r["tasks"]) == 4

    def test_ctrl_most_activations(self):
        """CTRL (10kHz) should activate ~10× more than COMM (1kHz)."""
        r = self.rtos.simulate_disco_tasks(100000)
        tasks = {t["name"]: t for t in r["tasks"]}
        ratio = tasks["CTRL"]["activations"] / max(1, tasks["COMM"]["activations"])
        assert 9.0 <= ratio <= 11.0, f"CTRL/COMM activation ratio {ratio:.1f} expected ~10"

    def test_no_deadline_misses_nominal(self):
        """Nominal DISCO task set must have zero deadline misses."""
        r = self.rtos.simulate_disco_tasks(200000)
        for t in r["tasks"]:
            assert t["deadline_misses"] == 0, (
                f"{t['name']} had {t['deadline_misses']} deadline misses"
            )

    def test_overloaded_task_causes_misses(self):
        """A task with WCET > period must produce deadline misses."""
        tc = self.rtos.TaskConfig("OVERLOAD", 10, 15, 0)  # C=15 > T=10
        r = self.rtos.run_scheduler([tc], 200)
        tasks = {t["name"]: t for t in r["tasks"]}
        assert tasks["OVERLOAD"]["deadline_misses"] > 0

    def test_rm_bound_formula(self):
        """RM bound = n·(2^(1/n)−1) for n=4 ≈ 0.757."""
        r = self.rtos.simulate_disco_tasks(100000)
        expected = 4.0 * (2.0 ** (1.0 / 4.0) - 1.0)
        assert abs(r["rm_bound"] - expected) < 1e-9

    def test_context_switches_positive(self):
        r = self.rtos.simulate_disco_tasks(10000)
        assert r["context_switches"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# SPC Monitor
# ══════════════════════════════════════════════════════════════════════════════

class TestSpcMonitor:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("pipeline._spc_monitor_kernel",
                            reason="SPC kernel not built; run build_all_kernels.sh spc")
        from pipeline import _spc_monitor_kernel as spc
        self.spc = spc

    def _in_control(self, n=50, mean=0.150, std=0.003, seed=0):
        """Generate n measurements centred on nominal kerf width 0.150 mm."""
        import random
        random.seed(seed)
        return [random.gauss(mean, std) for _ in range(n)]

    def test_cpk_in_control(self):
        """Cpk must be > 1.33 for a well-centred process within ±0.02 mm spec."""
        data = self._in_control(100, mean=0.150, std=0.003)
        r = self.spc.spc_update(data, usl=0.170, lsl=0.130, subgroup_size=5)
        assert r["cpk"] > 1.33, f"Cpk = {r['cpk']:.3f} expected > 1.33"

    def test_cpk_out_of_spec(self):
        """Cpk must be < 1.0 when σ is large relative to the spec window."""
        data = self._in_control(50, mean=0.150, std=0.015)
        r = self.spc.spc_update(data, usl=0.170, lsl=0.130, subgroup_size=5)
        assert r["cpk"] < 1.0, f"Cpk = {r['cpk']:.3f} should be < 1.0 for wide σ"

    def test_ucl_lcl_relationship(self):
        """UCL_x > xbar_bar > LCL_x always."""
        data = self._in_control(50)
        r = self.spc.spc_update(data, usl=0.170, lsl=0.130)
        assert r["ucl_x"] > r["xbar_bar"] > r["lcl_x"]

    def test_cp_le_cpk_when_shifted(self):
        """Cp ≥ Cpk when process mean is off-centre."""
        # shift mean towards USL
        data = self._in_control(50, mean=0.165, std=0.003)
        r = self.spc.spc_update(data, usl=0.170, lsl=0.130)
        assert r["cp"] >= r["cpk"] - 1e-9

    def test_weco_rule1_fires_on_outlier(self):
        """A single 4σ outlier must trigger WECO rule 1."""
        data = [0.150] * 49 + [0.200]  # 4σ outlier in last point
        r = self.spc.spc_update(data, usl=0.170, lsl=0.130, subgroup_size=5)
        assert r["weco_alarm"] is True

    def test_in_control_no_weco(self):
        """Tightly in-control data should not trigger WECO alarms."""
        import random
        random.seed(7)
        data = [random.gauss(0.150, 0.001) for _ in range(50)]
        r = self.spc.spc_update(data, usl=0.180, lsl=0.120, subgroup_size=5)
        assert r["weco_alarm"] is False

    def test_ewma_length(self):
        """EWMA list length must equal number of measurements."""
        data = self._in_control(40)
        r = self.spc.spc_update(data, usl=0.170, lsl=0.130)
        assert len(r["ewma"]) == 40

    def test_invalid_subgroup_raises(self):
        with pytest.raises(Exception):
            self.spc.spc_update([0.15] * 20, 0.17, 0.13, subgroup_size=1)


# ══════════════════════════════════════════════════════════════════════════════
# Reliability Kernel
# ══════════════════════════════════════════════════════════════════════════════

class TestReliabilityKernel:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("machine._reliability_kernel",
                            reason="Reliability kernel not built; run build_all_kernels.sh reliability")
        from machine import _reliability_kernel as rel
        self.rel = rel

    def test_series_two_equal_components(self):
        """Two identical components in series: MTBF_sys = MTBF/2."""
        r = self.rel.series_system([100.0, 100.0])
        assert abs(r["mtbf_hours"] - 50.0) < 1e-9

    def test_series_three_components(self):
        """Series system MTBF: 1/Σ(1/MTBFi)."""
        mtbfs = [1000.0, 500.0, 2000.0]  # λ = 0.001+0.002+0.0005 = 0.0035
        expected = 1.0 / (1/1000 + 1/500 + 1/2000)
        r = self.rel.series_system(mtbfs)
        assert abs(r["mtbf_hours"] - expected) < 0.01

    def test_parallel_improves_single(self):
        """Parallel system MTTF must be > single-component MTTF."""
        r = self.rel.parallel_system([100.0, 100.0])
        assert r["mttf_hours"] > 100.0

    def test_availability_formula(self):
        """Availability = MTBF / (MTBF + MTTR)."""
        r = self.rel.availability_oee(200.0, 8.0)
        expected = 200.0 / 208.0
        assert abs(r["availability"] - expected) < 1e-9

    def test_oee_product(self):
        """OEE = Availability × Performance × Quality."""
        r = self.rel.availability_oee(200.0, 8.0, performance_rate=0.9, quality_rate=0.98)
        A = 200.0 / 208.0
        assert abs(r["oee"] - A * 0.9 * 0.98) < 1e-9

    def test_weibull_beta1_exponential(self):
        """Weibull with β=1 reduces to exponential: R(t)=exp(-t/η)."""
        t_vals = [0.0, 50.0, 100.0, 200.0]
        r = self.rel.weibull_analysis(t_vals, beta=1.0, eta=100.0)
        for i, t in enumerate(t_vals):
            expected_R = math.exp(-t / 100.0)
            assert abs(r["reliability"][i] - expected_R) < 1e-6

    def test_weibull_mttf_beta1(self):
        """MTTF for Weibull β=1 must equal η (scale parameter)."""
        r = self.rel.weibull_analysis([0.0], beta=1.0, eta=72.0)
        assert abs(r["mttf_hours"] - 72.0) < 0.01

    def test_weibull_failure_phase_labels(self):
        """Phase labels must match beta range."""
        r1 = self.rel.weibull_analysis([10.0], beta=0.5, eta=100.0)
        r2 = self.rel.weibull_analysis([10.0], beta=1.0, eta=100.0)
        r3 = self.rel.weibull_analysis([10.0], beta=2.5, eta=100.0)
        assert r1["failure_phase"] == "infant_mortality"
        assert r2["failure_phase"] == "random_failure"
        assert r3["failure_phase"] == "wear_out"

    def test_blade_sim_mttf_accuracy(self):
        """Monte Carlo MTTF must match analytic within 5%."""
        r = self.rel.simulate_blade_life(n_blades=5000, beta=2.0, eta=72.0, seed=1)
        assert r["mttf_error_pct"] < 5.0, (
            f"Monte Carlo MTTF error {r['mttf_error_pct']:.1f}% too large"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Process DB (SQLite)
# ══════════════════════════════════════════════════════════════════════════════

class TestProcessDB:
    @pytest.fixture
    def db(self):
        from data.process_db import ProcessDB
        with ProcessDB(":memory:") as db:
            db.ensure_lot("LOT001", product_code="Si300")
            db.ensure_recipe("R01", feed_rate_mm_s=80.0, spindle_rpm=30000.0)
            yield db

    def test_insert_and_query(self, db):
        wid = db.insert_wafer("LOT001", position_in_lot=1)
        rid = db.start_run(wid, "R01")
        db.insert_measurement(rid, line_idx=0, kerf_width_mm=0.152,
                              chipping_um=1.5, blade_wear_pct=10.0, cpk=1.6)
        db.finish_run(rid)

        rows = db.execute_sql("SELECT kerf_width_mm FROM measurement WHERE run_id=?", (rid,))
        assert len(rows) == 1
        assert abs(rows[0]["kerf_width_mm"] - 0.152) < 1e-9

    def test_lot_summary_view(self, db):
        """v_lot_summary VIEW must aggregate measurements correctly via JOIN."""
        wid = db.insert_wafer("LOT001")
        rid = db.start_run(wid, "R01")
        for i in range(10):
            db.insert_measurement(rid, line_idx=i,
                                  kerf_width_mm=0.150 + 0.001 * i,
                                  cpk=1.5)
        rows = db.lot_summary("LOT001")
        assert len(rows) == 1
        assert rows[0]["n_measurements"] == 10

    def test_out_of_spec_query(self, db):
        """out_of_spec must return only measurements outside [lsl, usl]."""
        wid = db.insert_wafer("LOT001")
        rid = db.start_run(wid, "R01")
        db.insert_measurement(rid, line_idx=0, kerf_width_mm=0.150)
        db.insert_measurement(rid, line_idx=1, kerf_width_mm=0.180)  # OOS
        db.insert_measurement(rid, line_idx=2, kerf_width_mm=0.110)  # OOS
        oos = db.out_of_spec(usl=0.170, lsl=0.130)
        assert len(oos) == 2

    def test_recipe_performance_groupby(self, db):
        """recipe_performance must GROUP BY recipe_id."""
        db.ensure_recipe("R02", feed_rate_mm_s=100.0)
        wid = db.insert_wafer("LOT001")
        rid1 = db.start_run(wid, "R01")
        rid2 = db.start_run(wid, "R02")
        for r in [rid1, rid2]:
            db.insert_measurement(r, kerf_width_mm=0.150, cpk=1.5)
        rows = db.recipe_performance()
        recipe_ids = {r["recipe_id"] for r in rows}
        assert "R01" in recipe_ids and "R02" in recipe_ids

    def test_alarm_insert_and_filter(self, db):
        wid = db.insert_wafer("LOT001")
        rid = db.start_run(wid, "R01")
        db.insert_alarm(rid, "SPC_ALARM", "Cpk < 1.33", severity="WARNING")
        db.insert_alarm(rid, "ESTOP",     "Blade depth fault", severity="ERROR")
        warnings = db.alarm_history(severity="WARNING")
        errors   = db.alarm_history(severity="ERROR")
        assert len(warnings) == 1
        assert len(errors)   == 1

    def test_foreign_key_enforced(self, db):
        """Insert measurement referencing non-existent run_id must fail."""
        with pytest.raises(Exception):
            db._conn.execute(
                "INSERT INTO measurement(run_id, kerf_width_mm) VALUES (9999, 0.15)"
            )
            db._conn.commit()

    def test_context_manager(self):
        from data.process_db import ProcessDB
        with ProcessDB(":memory:") as db:
            db.ensure_lot("LOT_CM")
            rows = db.execute_sql("SELECT lot_id FROM lot")
        assert rows[0]["lot_id"] == "LOT_CM"

    def test_arbitrary_sql_select(self, db):
        """execute_sql must return list-of-dicts for any SELECT."""
        rows = db.execute_sql("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = {r["name"] for r in rows}
        assert "measurement" in table_names
        assert "alarm" in table_names
