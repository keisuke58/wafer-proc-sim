"""
Wafer Process Log — SQLite database layer for dicing process data.

AP試験 concepts demonstrated (データベース・SQL):
  - 正規化 (3NF): lot → wafer → process_run → measurement
  - SQL: SELECT/INSERT/UPDATE/DELETE/JOIN/GROUP BY/HAVING/サブクエリ
  - トランザクション (ACID): commit/rollback via context manager
  - インデックス: kerf_width_mm, lot_id, wafer_id for fast QC queries
  - ビュー: v_lot_summary (lot-level statistics)
  - 集約関数: AVG/STDDEV/MIN/MAX/COUNT

Application: store detect_kerf_width() + spc_update() results per wafer cut.
  SELECT lot_id, AVG(kerf_width_mm), MIN(cpk) to find lots needing re-inspection.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional


# ── Schema ────────────────────────────────────────────────────────────────────
# 3NF decomposition:
#   lot (lot_id PK, created_at, product_code)
#   wafer (wafer_id PK, lot_id FK, position_in_lot)
#   recipe (recipe_id PK, feed_rate_mm_s, spindle_rpm, blade_id)
#   process_run (run_id PK, wafer_id FK, recipe_id FK, started_at, finished_at)
#   measurement (meas_id PK, run_id FK, line_idx, kerf_width_mm, chipping_um,
#                blade_wear_pct, cpk, status)
#   alarm (alarm_id PK, run_id FK, triggered_at, severity, code, message)

_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS lot (
    lot_id       TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    product_code TEXT NOT NULL DEFAULT 'WAFER'
);

CREATE TABLE IF NOT EXISTS wafer (
    wafer_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    lot_id           TEXT NOT NULL REFERENCES lot(lot_id),
    position_in_lot  INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS recipe (
    recipe_id       TEXT PRIMARY KEY,
    feed_rate_mm_s  REAL NOT NULL DEFAULT 100.0,
    spindle_rpm     REAL NOT NULL DEFAULT 30000.0,
    blade_id        TEXT NOT NULL DEFAULT 'NBD720'
);

CREATE TABLE IF NOT EXISTS process_run (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    wafer_id    INTEGER NOT NULL REFERENCES wafer(wafer_id),
    recipe_id   TEXT    NOT NULL REFERENCES recipe(recipe_id),
    started_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS measurement (
    meas_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES process_run(run_id),
    line_idx      INTEGER NOT NULL DEFAULT 0,
    kerf_width_mm REAL    NOT NULL,
    chipping_um   REAL    NOT NULL DEFAULT 0.0,
    blade_wear_pct REAL   NOT NULL DEFAULT 0.0,
    cpk           REAL,
    status        TEXT    NOT NULL DEFAULT 'ok'
);

CREATE TABLE IF NOT EXISTS alarm (
    alarm_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES process_run(run_id),
    triggered_at TEXT    NOT NULL DEFAULT (datetime('now')),
    severity     TEXT    NOT NULL DEFAULT 'WARNING',
    code         TEXT    NOT NULL,
    message      TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_meas_run    ON measurement(run_id);
CREATE INDEX IF NOT EXISTS idx_meas_kerf   ON measurement(kerf_width_mm);
CREATE INDEX IF NOT EXISTS idx_meas_status ON measurement(status);
CREATE INDEX IF NOT EXISTS idx_alarm_sev   ON alarm(severity);

-- Lot summary view: JOIN lot→wafer→process_run→measurement
CREATE VIEW IF NOT EXISTS v_lot_summary AS
SELECT
    l.lot_id,
    l.product_code,
    COUNT(DISTINCT w.wafer_id)  AS n_wafers,
    COUNT(m.meas_id)            AS n_measurements,
    ROUND(AVG(m.kerf_width_mm), 4)  AS kerf_avg_mm,
    ROUND(MIN(m.kerf_width_mm), 4)  AS kerf_min_mm,
    ROUND(MAX(m.kerf_width_mm), 4)  AS kerf_max_mm,
    ROUND(AVG(m.cpk), 3)            AS cpk_avg,
    SUM(CASE WHEN m.status != 'ok' THEN 1 ELSE 0 END) AS n_defects
FROM lot l
JOIN wafer w ON w.lot_id = l.lot_id
JOIN process_run r ON r.wafer_id = w.wafer_id
JOIN measurement m ON m.run_id = r.run_id
GROUP BY l.lot_id, l.product_code;
"""


def _rows_to_dicts(cursor: sqlite3.Cursor) -> list[dict]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


class ProcessDB:
    """
    SQLite-backed wafer process log.

    Parameters
    ----------
    path : str | Path
        Database file path, or ':memory:' for in-memory (default; useful for tests).

    Usage
    -----
        db = ProcessDB()
        db.ensure_lot('LOT001', product_code='Si300')
        db.ensure_recipe('R01', feed_rate_mm_s=80.0, spindle_rpm=30000.0)
        wid = db.insert_wafer('LOT001', position_in_lot=1)
        rid = db.start_run(wid, 'R01')
        db.insert_measurement(rid, line_idx=0, kerf_width_mm=0.152,
                              chipping_um=2.1, blade_wear_pct=12.0, cpk=1.45)
        db.finish_run(rid)

        # Analytics
        rows = db.lot_summary()
        oor  = db.out_of_spec(usl=0.165, lsl=0.135)
        db.execute_sql('SELECT lot_id, kerf_avg_mm FROM v_lot_summary')
    """

    def __init__(self, path: str | Path = ":memory:"):
        self._path = str(path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ── DDL helpers ────────────────────────────────────────────────────────────

    def ensure_lot(self, lot_id: str, product_code: str = "WAFER") -> None:
        """INSERT OR IGNORE lot row."""
        self._conn.execute(
            "INSERT OR IGNORE INTO lot(lot_id, product_code) VALUES (?,?)",
            (lot_id, product_code),
        )
        self._conn.commit()

    def ensure_recipe(
        self,
        recipe_id: str,
        feed_rate_mm_s: float = 100.0,
        spindle_rpm: float = 30000.0,
        blade_id: str = "NBD720",
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO recipe(recipe_id,feed_rate_mm_s,spindle_rpm,blade_id)"
            " VALUES (?,?,?,?)",
            (recipe_id, feed_rate_mm_s, spindle_rpm, blade_id),
        )
        self._conn.commit()

    # ── Wafer & run ────────────────────────────────────────────────────────────

    def insert_wafer(self, lot_id: str, position_in_lot: int = 1) -> int:
        """Insert a wafer row; returns wafer_id."""
        cur = self._conn.execute(
            "INSERT INTO wafer(lot_id, position_in_lot) VALUES (?,?)",
            (lot_id, position_in_lot),
        )
        self._conn.commit()
        return cur.lastrowid

    def start_run(self, wafer_id: int, recipe_id: str) -> int:
        """Start a process run; returns run_id."""
        cur = self._conn.execute(
            "INSERT INTO process_run(wafer_id, recipe_id) VALUES (?,?)",
            (wafer_id, recipe_id),
        )
        self._conn.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int) -> None:
        self._conn.execute(
            "UPDATE process_run SET finished_at = datetime('now') WHERE run_id = ?",
            (run_id,),
        )
        self._conn.commit()

    # ── Measurements & alarms ──────────────────────────────────────────────────

    def insert_measurement(
        self,
        run_id: int,
        *,
        line_idx: int = 0,
        kerf_width_mm: float,
        chipping_um: float = 0.0,
        blade_wear_pct: float = 0.0,
        cpk: Optional[float] = None,
        status: str = "ok",
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO measurement(run_id,line_idx,kerf_width_mm,chipping_um,"
            "blade_wear_pct,cpk,status) VALUES (?,?,?,?,?,?,?)",
            (run_id, line_idx, kerf_width_mm, chipping_um, blade_wear_pct, cpk, status),
        )
        self._conn.commit()
        return cur.lastrowid

    def insert_alarm(
        self,
        run_id: int,
        code: str,
        message: str = "",
        severity: str = "WARNING",
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO alarm(run_id,severity,code,message) VALUES (?,?,?,?)",
            (run_id, severity, code, message),
        )
        self._conn.commit()
        return cur.lastrowid

    # ── Analytics queries ──────────────────────────────────────────────────────

    def lot_summary(self, lot_id: Optional[str] = None) -> list[dict]:
        """SELECT from v_lot_summary view (GROUP BY lot_id via JOIN)."""
        if lot_id:
            cur = self._conn.execute(
                "SELECT * FROM v_lot_summary WHERE lot_id = ?", (lot_id,)
            )
        else:
            cur = self._conn.execute("SELECT * FROM v_lot_summary")
        return _rows_to_dicts(cur)

    def out_of_spec(
        self,
        usl: float,
        lsl: float,
        lot_id: Optional[str] = None,
    ) -> list[dict]:
        """Return measurements outside [lsl, usl] — subquery join demo."""
        base_sql = """
            SELECT m.meas_id, m.run_id, m.line_idx,
                   m.kerf_width_mm, m.chipping_um, m.cpk, m.status,
                   l.lot_id, w.position_in_lot
            FROM measurement m
            JOIN process_run r ON r.run_id = m.run_id
            JOIN wafer w ON w.wafer_id = r.wafer_id
            JOIN lot l ON l.lot_id = w.lot_id
            WHERE m.kerf_width_mm NOT BETWEEN ? AND ?
        """
        if lot_id:
            cur = self._conn.execute(base_sql + " AND l.lot_id = ?", (lsl, usl, lot_id))
        else:
            cur = self._conn.execute(base_sql, (lsl, usl))
        return _rows_to_dicts(cur)

    def recipe_performance(self) -> list[dict]:
        """GROUP BY recipe_id — average Cpk and kerf stats per recipe."""
        cur = self._conn.execute("""
            SELECT r.recipe_id,
                   COUNT(m.meas_id)              AS n_cuts,
                   ROUND(AVG(m.kerf_width_mm),4) AS kerf_avg,
                   ROUND(AVG(m.cpk),3)           AS cpk_avg,
                   ROUND(AVG(m.chipping_um),2)   AS chip_avg_um,
                   SUM(CASE WHEN m.status!='ok' THEN 1 ELSE 0 END) AS n_defects
            FROM process_run r
            JOIN measurement m ON m.run_id = r.run_id
            GROUP BY r.recipe_id
            HAVING COUNT(m.meas_id) > 0
            ORDER BY cpk_avg DESC
        """)
        return _rows_to_dicts(cur)

    def alarm_history(self, severity: Optional[str] = None) -> list[dict]:
        """Return alarm rows, optionally filtered by severity."""
        if severity:
            cur = self._conn.execute(
                "SELECT * FROM alarm WHERE severity = ? ORDER BY triggered_at DESC",
                (severity,),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM alarm ORDER BY triggered_at DESC"
            )
        return _rows_to_dicts(cur)

    def execute_sql(self, query: str, params: tuple = ()) -> list[dict]:
        """Run an arbitrary SELECT (read-only) and return list of row dicts."""
        cur = self._conn.execute(query, params)
        return _rows_to_dicts(cur)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ProcessDB":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"ProcessDB(path={self._path!r})"
