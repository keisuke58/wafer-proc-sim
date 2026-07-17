# DISCO TSDB — equipment time-series collection & analysis foundation

A compact **time-series data-collection and analysis foundation** for
semiconductor equipment sensor streams, built for the DISCO "AI algorithm
engineer" role ("real-time collection", "design/development of a collection &
analysis foundation", time-series DBs such as **InfluxDB / PostgreSQL**).

Implemented with the Python standard library's **SQLite** (always available)
while deliberately mirroring the concepts real TSDBs are built on, so the same
ingest maps onto InfluxDB/TimescaleDB unchanged.

## What it does

- **InfluxDB-shaped data model** — every point is `measurement` + tags
  (`tool`, `recipe`) + fields (sensor values) + nanosecond `ts`.
  `to_line_protocol()` emits the exact InfluxDB line-protocol wire format.
- **Ingestion** — a simulated multi-tool stream (spindle current, coolant
  temperature, vibration RMS) written with an index on `(measurement, ts)` —
  the shape a TSDB needs for range scans.
- **Time-bucket downsampling** — `GROUP BY` on a time bucket, the portable form
  of InfluxDB `aggregateWindow()` / TimescaleDB `time_bucket()`; the core query
  for dashboards and retention rollups.
- **Monitoring** — an anomaly-window query flags buckets whose mean vibration
  exceeds a limit; this is the early-warning hook the APC work
  (`disco_apc_csharp/`) consumes.

## Result (committed run)

3 tools × 3 sensors × 240 samples/tool @ 60 s = **2,160 rows**; 15-min rollup →
17 buckets/tool. A developing fault is injected on **T3**: the anomaly query
flags **4 buckets on T3, 0 on the clean tool**, first alarm ~181 min in.

## Run / test

```bash
python3 disco_tsdb/equipment_tsdb.py        # ingest → query → figure + JSON
python3 -m pytest tests/test_disco_tsdb.py  # 7 tests
```

## How this maps to SEMI equipment-data standards (GEM / EDA)

This foundation is transport-agnostic; in a real fab the samples arrive over
the SEMI standards below. Knowing where each fits is the point:

- **SECS-II (SEMI E5)** — the message *content* (data item structures) exchanged
  between equipment and host.
- **HSMS (SEMI E37)** — SECS messages over TCP/IP (the modern transport).
- **GEM (SEMI E30)** — the *behavioural* standard on top of SECS/HSMS: the
  equipment **state model**, **Status Variables (SV) / Equipment Constants (EC) /
  Data Variables (DV)**, **Collection Events (CE)**, alarms, and remote commands.
  In this module, a "collection event → report" maps to a point insert; SVs map
  to the `sensor`/`value` fields; equipment/recipe identity maps to tags.
- **EDA / Interface A (SEMI E120/E125/E132/E134/E164)** — the high-frequency
  **data-collection** framework: the equipment self-describes its structure
  (E120/E125 metadata model), a client authenticates/subscribes (E132), and
  **Data Collection Plans (E134)** stream trace data at high rates — exactly the
  "real-time collection → analysis foundation" this module stands in for. E134's
  trace/rollup semantics correspond to the time-bucket queries here.

So: **GEM** for equipment control + event-driven reports, **EDA (Interface A)**
for high-rate trace collection → land both in a time-series store → time-bucket
rollups + anomaly windows → feed APC. This module is the store-and-analyze half,
written to slot behind either interface.
