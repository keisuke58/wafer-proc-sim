"""Tests for disco_tsdb/equipment_tsdb.py — equipment time-series foundation."""
import pytest

tsdb = pytest.importorskip("disco_tsdb.equipment_tsdb")


class TestStore:
    def test_ingest_count(self):
        s = tsdb.TimeSeriesStore()
        rows, _ = tsdb.simulate_stream(["A", "B"], "R1", 10, 60, seed=1)
        s.ingest_batch(rows)
        # 2 tools * 3 sensors * 10 samples
        assert s.count() == 2 * len(tsdb.SENSORS) * 10

    def test_downsample_bucket_mean_is_correct(self):
        s = tsdb.TimeSeriesStore()
        # time buckets align to absolute epoch multiples (like time_bucket),
        # so anchor to a bucket boundary for a deterministic check.
        base = (tsdb.BASE_TS // 900) * 900
        # two samples in bucket 0 (values 1,3 -> mean 2), one in next bucket
        s.ingest_batch([
            (base + 0,   "equipment", "A", "R", "vibration_rms", 1.0),
            (base + 100, "equipment", "A", "R", "vibration_rms", 3.0),
            (base + 900, "equipment", "A", "R", "vibration_rms", 9.0),
        ])
        ds = s.query_downsample("vibration_rms", "A", 900)
        assert len(ds) == 2
        assert ds[0][1] == pytest.approx(2.0)   # mean of bucket 0
        assert ds[0][4] == 2                     # count
        assert ds[1][1] == pytest.approx(9.0)

    def test_tag_filter_isolates_tool(self):
        s = tsdb.TimeSeriesStore()
        rows, _ = tsdb.simulate_stream(["A", "B"], "R1", 5, 60, seed=2)
        s.ingest_batch(rows)
        assert len(s.raw("vibration_rms", "A")) == 5
        assert all(True for _ in s.raw("vibration_rms", "B"))
        assert len(s.query_last("spindle_current")) == 2


class TestMonitoring:
    def _store(self):
        s = tsdb.TimeSeriesStore()
        rows, _ = tsdb.simulate_stream(["T1", "T3"], "R2", 240, 60, seed=7,
                                       anomaly_tool="T3")
        s.ingest_batch(rows)
        return s

    def test_fault_tool_flagged_clean_tool_not(self):
        s = self._store()
        clean = s.query_anomaly_windows("vibration_rms", "T1", 900, 0.5)
        bad = s.query_anomaly_windows("vibration_rms", "T3", 900, 0.5)
        assert len(clean) == 0
        assert len(bad) >= 3

    def test_alarm_starts_in_last_third(self):
        s = self._store()
        bad = s.query_anomaly_windows("vibration_rms", "T3", 900, 0.5)
        first_offset_min = (bad[0][0] - tsdb.BASE_TS) / 60
        # ramp starts at 2/3 of 240 min = 160 min; alarm shortly after
        assert first_offset_min >= 150


class TestLineProtocol:
    def test_line_protocol_format(self):
        pts = [(tsdb.BASE_TS, "T1", "R2",
                {"spindle_current": 12.0, "vibration_rms": 0.3})]
        lp = tsdb.to_line_protocol(pts)
        assert lp.startswith("equipment,tool=T1,recipe=R2 ")
        assert "spindle_current=12.0000" in lp
        assert lp.strip().endswith(str(tsdb.BASE_TS * 1_000_000_000))


def test_run_smoke():
    res, _, _, _ = tsdb.run()
    assert res["ingest"]["rows"] == 3 * 3 * 240
    assert res["anomaly"]["fault_tool_T3_alarm_buckets"] >= 3
    assert res["anomaly"]["clean_tool_T1_alarm_buckets"] == 0
