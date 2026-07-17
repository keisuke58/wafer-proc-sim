#!/usr/bin/env python3
"""equipment_tsdb.py — a compact time-series data-collection & analysis
foundation for semiconductor equipment sensor streams.

The DISCO "AI algorithm engineer" role asks for "real-time collection" and
"design/development of a collection & analysis foundation" with time-series
databases (InfluxDB, PostgreSQL). This module builds that foundation end to
end with the Python standard library's SQLite (always available), while
deliberately mirroring the concepts those TSDBs are built on:

  DATA MODEL     points carry a measurement, a set of tags (tool, recipe),
      one or more fields (sensor values) and a nanosecond timestamp — the
      InfluxDB line-protocol model. `to_line_protocol()` emits exactly that
      wire format, so the same ingest maps onto a real InfluxDB unchanged.

  INGESTION      a simulated multi-tool equipment stream (spindle current,
      coolant temperature, vibration RMS) is written with an index on
      (measurement, ts) — the shape a TSDB needs for range scans.

  TIME-BUCKET    downsampling via GROUP BY on a time bucket is the portable
      form of InfluxDB `aggregateWindow()` / TimescaleDB `time_bucket()` —
      the core query for dashboards, retention rollups and monitoring.

  MONITORING     an anomaly-window query flags buckets whose mean vibration
      exceeds a limit — the "abnormality / early-warning" hook the APC work
      (see disco_apc_csharp/) consumes.

numpy + sqlite3 + matplotlib.  Run:  python3 disco_tsdb/equipment_tsdb.py
"""
import json
import os
import sqlite3

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "equipment_tsdb_results.json")

BASE_TS = 1_700_000_000          # fixed epoch base (reproducible, no wall clock)
SENSORS = ("spindle_current", "coolant_temp", "vibration_rms")


class TimeSeriesStore:
    """Minimal time-series store over SQLite with an InfluxDB-shaped data
    model (measurement + tags + fields + ts) and time-bucket queries."""

    def __init__(self, path=":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS samples(
                ts       INTEGER NOT NULL,   -- epoch seconds
                measurement TEXT NOT NULL,
                tool     TEXT NOT NULL,      -- tag
                recipe   TEXT NOT NULL,      -- tag
                sensor   TEXT NOT NULL,      -- field key
                value    REAL NOT NULL       -- field value
            )""")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_meas_ts ON samples(measurement, ts)")
        self.conn.commit()

    def ingest_batch(self, rows):
        """rows: iterable of (ts, measurement, tool, recipe, sensor, value)."""
        self.conn.executemany(
            "INSERT INTO samples VALUES (?,?,?,?,?,?)", rows)
        self.conn.commit()

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

    def query_downsample(self, sensor, tool, bucket_s):
        """Time-bucket mean/max/min — the portable aggregateWindow/time_bucket.
        Returns list of (bucket_start_ts, mean, max, min, n)."""
        cur = self.conn.execute(
            """SELECT (ts/?)*? AS b, AVG(value), MAX(value), MIN(value), COUNT(*)
               FROM samples WHERE sensor=? AND tool=?
               GROUP BY b ORDER BY b""",
            (bucket_s, bucket_s, sensor, tool))
        return cur.fetchall()

    def query_last(self, sensor):
        """Last value per tool for a sensor (InfluxDB last())."""
        cur = self.conn.execute(
            """SELECT tool, value FROM samples s WHERE sensor=? AND ts=(
                   SELECT MAX(ts) FROM samples s2
                   WHERE s2.tool=s.tool AND s2.sensor=s.sensor)
               ORDER BY tool""", (sensor,))
        return cur.fetchall()

    def query_anomaly_windows(self, sensor, tool, bucket_s, limit):
        """Buckets whose mean exceeds `limit` — early-warning hook."""
        return [(b, m) for (b, m, _mx, _mn, _n)
                in self.query_downsample(sensor, tool, bucket_s) if m > limit]

    def raw(self, sensor, tool):
        cur = self.conn.execute(
            "SELECT ts, value FROM samples WHERE sensor=? AND tool=? ORDER BY ts",
            (sensor, tool))
        return cur.fetchall()


def to_line_protocol(rows_by_point):
    """Emit InfluxDB line protocol from grouped points.
    rows_by_point: (ts, tool, recipe, {sensor: value}) -> one line each."""
    lines = []
    for ts, tool, recipe, fields in rows_by_point:
        fstr = ",".join("%s=%s" % (k, ("%.4f" % v)) for k, v in fields.items())
        lines.append("equipment,tool=%s,recipe=%s %s %d"
                     % (tool, recipe, fstr, ts * 1_000_000_000))
    return "\n".join(lines)


def simulate_stream(tools, recipe, n, period_s, seed, anomaly_tool=None):
    """Generate a multi-tool sensor stream; optionally ramp vibration on one
    tool in the last third to emulate a developing fault. Returns (rows,
    points) where rows feed ingest_batch and points feed line protocol."""
    rng = np.random.default_rng(seed)
    rows, points = [], []
    for k in range(n):
        ts = BASE_TS + k * period_s
        for tool in tools:
            spindle = 12.0 + 0.5 * np.sin(k / 9.0) + rng.normal(0, 0.15)
            coolant = 20.0 + 0.02 * k * 0 + rng.normal(0, 0.1)  # ~flat + noise
            vib = 0.30 + rng.normal(0, 0.03)
            if anomaly_tool is not None and tool == anomaly_tool and k > 2 * n // 3:
                vib += 0.010 * (k - 2 * n // 3)   # ramp = bearing/blade fault
            fields = {"spindle_current": spindle,
                      "coolant_temp": coolant, "vibration_rms": vib}
            points.append((ts, tool, recipe, fields))
            for s, v in fields.items():
                rows.append((ts, "equipment", tool, recipe, s, float(v)))
    return rows, points


def run():
    tools = ["T1", "T2", "T3"]
    n, period = 240, 60          # 240 samples @ 60 s = 4 h per tool
    store = TimeSeriesStore()
    rows, points = simulate_stream(tools, "R2", n, period, seed=7,
                                   anomaly_tool="T3")
    store.ingest_batch(rows)

    bucket = 900                 # 15-min rollup
    limit = 0.5                  # vibration alarm level
    ds_clean = store.query_downsample("vibration_rms", "T1", bucket)
    ds_bad = store.query_downsample("vibration_rms", "T3", bucket)
    an_clean = store.query_anomaly_windows("vibration_rms", "T1", bucket, limit)
    an_bad = store.query_anomaly_windows("vibration_rms", "T3", bucket, limit)
    last = store.query_last("spindle_current")

    res = {
        "model": "SQLite TSDB, InfluxDB-shaped (measurement/tags/fields/ts)",
        "ingest": {"rows": store.count(), "tools": len(tools),
                   "sensors": len(SENSORS), "samples_per_tool": n,
                   "period_s": period},
        "downsample": {"bucket_s": bucket,
                       "buckets_per_tool": len(ds_clean)},
        "anomaly": {"alarm_level": limit,
                    "clean_tool_T1_alarm_buckets": len(an_clean),
                    "fault_tool_T3_alarm_buckets": len(an_bad),
                    "first_alarm_ts_offset_min":
                        int((an_bad[0][0] - BASE_TS) / 60) if an_bad else None},
        "last_spindle_current": {t: round(v, 3) for t, v in last},
        "line_protocol_sample": to_line_protocol(points[:2]).split("\n"),
    }
    return res, store, bucket, limit


def _plot(path, store, bucket, limit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))

    # left: raw vs downsample for the faulty tool T3
    ax = axes[0]
    raw = store.raw("vibration_rms", "T3")
    rt = [(t - BASE_TS) / 3600 for t, _ in raw]
    rv = [v for _, v in raw]
    ax.plot(rt, rv, color="#9db4c8", lw=0.9, label="raw (60 s)")
    ds = store.query_downsample("vibration_rms", "T3", bucket)
    dt = [(b - BASE_TS) / 3600 for b, _, _, _, _ in ds]
    dv = [m for _, m, _, _, _ in ds]
    ax.plot(dt, dv, color="#0e6ba8", lw=2, marker="o", ms=3,
            label="time-bucket mean (15 min)")
    ax.axhline(limit, color="#c0392b", ls="--", lw=1.2, label="alarm level")
    for b, m in store.query_anomaly_windows("vibration_rms", "T3", bucket, limit):
        ax.scatter([(b - BASE_TS) / 3600], [m], s=70, facecolors="none",
                   edgecolors="#e08a00", linewidths=1.6, zorder=5)
    ax.set_title("Tool T3 — developing fault\n(raw → 15-min rollup → alarm)",
                 fontsize=10)
    ax.set_xlabel("time [h]"); ax.set_ylabel("vibration RMS")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # right: three tools' rollups — only T3 crosses
    ax = axes[1]
    for tool, col in zip(["T1", "T2", "T3"], ["#2e7d57", "#b4740a", "#c0392b"]):
        ds = store.query_downsample("vibration_rms", tool, bucket)
        ax.plot([(b - BASE_TS) / 3600 for b, *_ in ds],
                [m for _, m, *_ in ds], color=col, lw=1.8, marker=".",
                label=tool + (" (fault)" if tool == "T3" else ""))
    ax.axhline(limit, color="#555", ls="--", lw=1)
    ax.set_title("Fleet view — one query per tool\n(tag filter = tool)",
                 fontsize=10)
    ax.set_xlabel("time [h]"); ax.set_ylabel("vibration RMS (15-min mean)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Equipment time-series foundation: ingest → time-bucket "
                 "rollup → anomaly windows", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=115)
    plt.close(fig)


def main():
    res, store, bucket, limit = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "disco_tsdb.png"), store, bucket, limit)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
