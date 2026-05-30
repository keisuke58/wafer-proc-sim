"""
Thread-safe live sensor simulator — singleton shared by FastAPI workers.

Produces one WaferReading every interval_s seconds via a daemon thread.
Configuration (anomaly_rate, interval_s, n_life) can be changed at runtime.
"""
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.sensor_fusion import NOMINAL, SENSOR_CONFIG, strategy_A_weighted, strategy_D_majority

MAX_HISTORY = 300


@dataclass
class WaferReading:
    wafer_id:             int
    spindle_current_A:    float
    vibration_rms_g:      float
    coolant_temp_C:       float
    acoustic_emission_dB: float
    camera_chip_um:       Optional[float]
    true_chip_um:         float
    is_anomaly:           bool
    fusion_z:             float
    fusion_flag:          bool
    majority_votes:       int
    majority_flag:        bool
    timestamp:            float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k, v in d.items():
            if isinstance(v, (np.bool_, np.integer, np.floating)):
                d[k] = v.item()
        return d


class LiveSimulator:

    def __init__(self):
        self._lock    = threading.Lock()
        self._history: deque[WaferReading] = deque(maxlen=MAX_HISTORY)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._wafer_id = 0
        self._rng      = np.random.default_rng(42)

        # Mutable config
        self.anomaly_rate = 0.15
        self.interval_s   = 1.0
        self.n_life       = 200
        self._rebuild_wear()

    def _rebuild_wear(self):
        from optimization.blade_wear import BladeWearModel
        self._wear = BladeWearModel(
            chip_0=NOMINAL["camera_chip_um"],
            n_life=self.n_life,
            alpha=2.0,
            usl=15.0,
        )

    def configure(self, anomaly_rate: float = None, interval_s: float = None,
                  n_life: int = None):
        with self._lock:
            if anomaly_rate is not None:
                self.anomaly_rate = float(anomaly_rate)
            if interval_s is not None:
                self.interval_s = float(interval_s)
            if n_life is not None and int(n_life) != self.n_life:
                self.n_life = int(n_life)
                self._rebuild_wear()

    def reset(self):
        with self._lock:
            self._wafer_id = 0
            self._history.clear()
            self._rng = np.random.default_rng(self._rng.integers(1 << 31))
            self._rebuild_wear()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _make_reading(self) -> WaferReading:
        n         = self._wafer_id
        true_chip = self._wear.predict_chip(n % self.n_life)
        scale     = true_chip / NOMINAL["camera_chip_um"]
        is_anom   = bool(self._rng.random() < self.anomaly_rate)
        amp       = float(self._rng.uniform(1.5, 2.5)) if is_anom else 1.0

        row: dict = {}
        for sensor, cfg in SENSOR_CONFIG.items():
            base  = NOMINAL[sensor] * (1 + (scale - 1) * 0.5) * amp
            noise = self._rng.normal(0, cfg["std"])
            if "available_prob" in cfg and self._rng.random() > cfg["available_prob"]:
                row[sensor] = np.nan
            else:
                row[sensor] = base + noise

        series = pd.Series(row)
        flag_A, z_A     = strategy_A_weighted(series)
        flag_D, votes_D = strategy_D_majority(series)

        cam = row.get("camera_chip_um", np.nan)
        return WaferReading(
            wafer_id             = n,
            spindle_current_A    = float(row.get("spindle_current_A",    NOMINAL["spindle_current_A"])),
            vibration_rms_g      = float(row.get("vibration_rms_g",      NOMINAL["vibration_rms_g"])),
            coolant_temp_C       = float(row.get("coolant_temp_C",       NOMINAL["coolant_temp_C"])),
            acoustic_emission_dB = float(row.get("acoustic_emission_dB", NOMINAL["acoustic_emission_dB"])),
            camera_chip_um       = float(cam) if not np.isnan(cam) else None,
            true_chip_um         = float(true_chip),
            is_anomaly           = is_anom,
            fusion_z             = float(z_A),
            fusion_flag          = bool(flag_A),
            majority_votes       = int(votes_D),
            majority_flag        = bool(flag_D),
        )

    def _loop(self):
        while self._running:
            with self._lock:
                reading = self._make_reading()
                self._history.append(reading)
                self._wafer_id += 1
                sleep_t = self.interval_s
            time.sleep(sleep_t)

    def get_history(self, n: int = 60) -> list[dict]:
        with self._lock:
            items = list(self._history)
        return [r.to_dict() for r in items[-n:]]

    def get_latest(self) -> Optional[dict]:
        with self._lock:
            return self._history[-1].to_dict() if self._history else None


_instance: Optional[LiveSimulator] = None
_instance_lock = threading.Lock()


def get_simulator() -> LiveSimulator:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = LiveSimulator()
    return _instance
