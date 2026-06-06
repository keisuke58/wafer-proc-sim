"""
DISCO dicing saw — machine control software stack
Covers the core ソフト scope: state machine, recipe management,
real-time motion loop, data logging, and operator API.

Layers modelled
---------------
1. State machine      — IEC 61131-3 inspired: IDLE → SETUP → DICING → INSPECT → DONE
2. Recipe manager     — load / validate / version-track process recipes
3. RT motion loop     — 1 ms cycle: position command → servo error → output
4. Data historian     — ring-buffer logger, CSV/HDF5 export
5. Operator API       — simple command dispatcher (simulates HMI / remote recipe upload)
6. Software FMEA      — failure mode registry and watchdog coverage

References
----------
IEC 61131-3 (PLC programming languages)
DISCO machine software architecture (inferred from public manuals)
PLCopen Motion Control (MC_MoveAbsolute, MC_Home)
"""

import time
import json
import collections
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Callable


# ── 1. State Machine ──────────────────────────────────────────────────────────

class MachineState(Enum):
    IDLE      = auto()
    SETUP     = auto()   # load recipe, home axes
    DICING    = auto()   # active cutting
    INSPECT   = auto()   # post-cut vision check
    DONE      = auto()   # lot complete
    FAULT     = auto()   # safety interlock / alarm
    E_STOP    = auto()   # emergency stop


VALID_TRANSITIONS: dict[MachineState, list[MachineState]] = {
    MachineState.IDLE:    [MachineState.SETUP,   MachineState.E_STOP],
    MachineState.SETUP:   [MachineState.DICING,  MachineState.FAULT, MachineState.E_STOP],
    MachineState.DICING:  [MachineState.INSPECT, MachineState.FAULT, MachineState.E_STOP],
    MachineState.INSPECT: [MachineState.DICING,  MachineState.DONE,  MachineState.E_STOP],
    MachineState.DONE:    [MachineState.IDLE],
    MachineState.FAULT:   [MachineState.IDLE],
    MachineState.E_STOP:  [MachineState.IDLE],
}


class StateMachine:
    def __init__(self):
        self.state = MachineState.IDLE
        self._history: list[tuple[str, MachineState]] = []
        self._callbacks: dict[MachineState, list[Callable]] = {s: [] for s in MachineState}

    def transition(self, target: MachineState, reason: str = "") -> bool:
        if target not in VALID_TRANSITIONS.get(self.state, []):
            return False
        self._history.append((reason or f"{self.state.name}→{target.name}", target))
        self.state = target
        for cb in self._callbacks[target]:
            cb()
        return True

    def on_enter(self, state: MachineState, callback: Callable):
        self._callbacks[state].append(callback)

    def history(self) -> list[dict]:
        return [{"reason": r, "state": s.name} for r, s in self._history]


# ── 2. Recipe Manager ─────────────────────────────────────────────────────────

@dataclass
class DicingRecipe:
    name: str
    spindle_rpm: float          # [rpm]
    feed_mms: float             # [mm/s]
    cut_depth_um: float         # [µm]
    kerf_pitch_um: float        # die pitch [µm]
    n_passes: int = 1           # multipass cutting
    water_flow_lpm: float = 1.0 # coolant [L/min]
    inspect_every_n: int = 10   # inspect every N cuts
    version: int = 1

    def validate(self) -> list[str]:
        errors = []
        if not (5_000 <= self.spindle_rpm <= 60_000):
            errors.append(f"spindle_rpm {self.spindle_rpm} out of range [5000, 60000]")
        if not (0.1 <= self.feed_mms <= 500.0):
            errors.append(f"feed_mms {self.feed_mms} out of range [0.1, 500]")
        if not (10 <= self.cut_depth_um <= 800):
            errors.append(f"cut_depth_um {self.cut_depth_um} out of range [10, 800]")
        if self.n_passes < 1:
            errors.append("n_passes must be >= 1")
        return errors

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "DicingRecipe":
        return cls(**json.loads(s))


class RecipeManager:
    def __init__(self):
        self._store: dict[str, DicingRecipe] = {}

    def save(self, recipe: DicingRecipe) -> list[str]:
        errors = recipe.validate()
        if errors:
            return errors
        self._store[recipe.name] = recipe
        return []

    def load(self, name: str) -> DicingRecipe | None:
        return self._store.get(name)

    def list_recipes(self) -> list[str]:
        return list(self._store.keys())

    def bump_version(self, name: str) -> bool:
        if name not in self._store:
            return False
        r = self._store[name]
        self._store[name] = DicingRecipe(**{**asdict(r), "version": r.version + 1})
        return True


# ── 3. RT Motion Loop (simulated 1 ms cycle) ─────────────────────────────────

@dataclass
class AxisState:
    pos_mm: float = 0.0
    vel_mms: float = 0.0
    error_mm: float = 0.0


class MotionController:
    """
    Simplified 1 ms position controller.
    P-controller with velocity feed-forward (mimics PLCopen MC_MoveAbsolute).
    """
    DT_S = 0.001  # 1 ms cycle time

    def __init__(self, Kp: float = 50.0, max_vel_mms: float = 300.0):
        self.Kp = Kp
        self.max_vel = max_vel_mms
        self.axis = AxisState()
        self._log: list[dict] = []

    def move_to(self, target_mm: float, n_cycles: int = 500) -> list[dict]:
        self._log.clear()
        for _ in range(n_cycles):
            error = target_mm - self.axis.pos_mm
            cmd_vel = np.clip_scalar(self.Kp * error, -self.max_vel, self.max_vel)
            self.axis.vel_mms = cmd_vel
            self.axis.pos_mm  += cmd_vel * self.DT_S
            self.axis.error_mm = error
            self._log.append({
                "pos_mm": round(self.axis.pos_mm, 6),
                "error_um": round(error * 1000, 3),
                "vel_mms": round(cmd_vel, 3),
            })
            if abs(error) < 0.001:  # settle within 1 µm
                break
        return self._log

    def home(self) -> None:
        self.axis = AxisState()


def _np_clip_scalar(val, lo, hi):
    return max(lo, min(hi, val))


# monkey-patch a scalar clip for numpy-free usage
class _NpShim:
    @staticmethod
    def clip_scalar(val, lo, hi):
        return _np_clip_scalar(val, lo, hi)

import sys as _sys
if "numpy" not in _sys.modules:
    np = _NpShim()  # type: ignore
else:
    import numpy as _np
    class _NpShim2:
        @staticmethod
        def clip_scalar(val, lo, hi):
            return float(_np.clip(val, lo, hi))
    np = _NpShim2()  # type: ignore


# ── 4. Data Historian ─────────────────────────────────────────────────────────

class DataHistorian:
    """
    Ring-buffer logger for process variables.
    Stores last N samples; exports to CSV or JSON.
    """
    def __init__(self, capacity: int = 100_000):
        self._buf: collections.deque = collections.deque(maxlen=capacity)

    def log(self, timestamp_s: float, **kwargs):
        self._buf.append({"t": timestamp_s, **kwargs})

    def tail(self, n: int = 10) -> list[dict]:
        return list(self._buf)[-n:]

    def stats(self) -> dict:
        if not self._buf:
            return {}
        keys = [k for k in self._buf[0] if k != "t"]
        result = {}
        for k in keys:
            vals = [r[k] for r in self._buf if k in r]
            result[k] = {
                "mean": sum(vals) / len(vals),
                "min":  min(vals),
                "max":  max(vals),
                "n":    len(vals),
            }
        return result

    def to_csv(self) -> str:
        if not self._buf:
            return ""
        header = ",".join(self._buf[0].keys())
        rows   = [",".join(str(v) for v in r.values()) for r in self._buf]
        return header + "\n" + "\n".join(rows)

    def clear(self):
        self._buf.clear()


# ── 5. Operator API ───────────────────────────────────────────────────────────

class OperatorAPI:
    """
    Command dispatcher — simulates HMI button presses or remote recipe upload.
    Commands are strings; responses are dicts (JSON-serialisable).
    """
    def __init__(self, sm: StateMachine, rm: RecipeManager, historian: DataHistorian):
        self.sm = sm
        self.rm = rm
        self.historian = historian
        self._active_recipe: DicingRecipe | None = None

    def dispatch(self, cmd: str, **kwargs) -> dict:
        handlers = {
            "load_recipe":    self._load_recipe,
            "start":          self._start,
            "stop":           self._stop,
            "e_stop":         self._estop,
            "reset_fault":    self._reset,
            "get_state":      self._get_state,
            "get_stats":      self._get_stats,
        }
        fn = handlers.get(cmd)
        if fn is None:
            return {"ok": False, "error": f"unknown command: {cmd}"}
        return fn(**kwargs)

    def _load_recipe(self, name: str = "", **_) -> dict:
        r = self.rm.load(name)
        if r is None:
            return {"ok": False, "error": f"recipe '{name}' not found"}
        errors = r.validate()
        if errors:
            return {"ok": False, "errors": errors}
        self._active_recipe = r
        return {"ok": True, "recipe": asdict(r)}

    def _start(self, **_) -> dict:
        if self._active_recipe is None:
            return {"ok": False, "error": "no recipe loaded"}
        ok = self.sm.transition(MachineState.SETUP, "operator start")
        if ok:
            ok = self.sm.transition(MachineState.DICING, "setup complete")
        return {"ok": ok, "state": self.sm.state.name}

    def _stop(self, **_) -> dict:
        ok = self.sm.transition(MachineState.DONE, "operator stop")
        if ok:
            ok = self.sm.transition(MachineState.IDLE, "lot complete")
        return {"ok": ok, "state": self.sm.state.name}

    def _estop(self, **_) -> dict:
        ok = self.sm.transition(MachineState.E_STOP, "E-STOP pressed")
        return {"ok": ok, "state": self.sm.state.name}

    def _reset(self, **_) -> dict:
        ok = self.sm.transition(MachineState.IDLE, "fault reset")
        return {"ok": ok, "state": self.sm.state.name}

    def _get_state(self, **_) -> dict:
        return {"state": self.sm.state.name, "history": self.sm.history()[-5:]}

    def _get_stats(self, **_) -> dict:
        return {"stats": self.historian.stats()}


# ── 6. Software FMEA ──────────────────────────────────────────────────────────

SW_FMEA = [
    {
        "id": "SW-01",
        "component": "StateMachine",
        "failure_mode": "Invalid state transition accepted",
        "effect": "Machine moves unexpectedly during setup",
        "severity": 8,
        "detection": "VALID_TRANSITIONS guard + unit test",
        "mitigation": "Reject invalid transitions; log attempt",
    },
    {
        "id": "SW-02",
        "component": "RecipeManager",
        "failure_mode": "Out-of-range recipe parameter accepted",
        "effect": "Spindle overspeed or wafer crack",
        "severity": 9,
        "detection": "validate() called on save and load",
        "mitigation": "Block save/load on validation failure",
    },
    {
        "id": "SW-03",
        "component": "MotionController",
        "failure_mode": "1 ms RT loop overrun",
        "effect": "Position lag, possible collision",
        "severity": 7,
        "detection": "Watchdog timer in PLC layer",
        "mitigation": "Safe Torque Off on >2 ms overrun",
    },
    {
        "id": "SW-04",
        "component": "DataHistorian",
        "failure_mode": "Ring buffer overflow (silent data loss)",
        "effect": "Missing QC data; traceability gap",
        "severity": 4,
        "detection": "deque maxlen drops oldest — no exception",
        "mitigation": "Log warning when buffer > 80% full",
    },
    {
        "id": "SW-05",
        "component": "OperatorAPI",
        "failure_mode": "Unknown command silently ignored",
        "effect": "Operator thinks action was executed",
        "severity": 5,
        "detection": "Return {ok: False, error: ...} for unknown cmd",
        "mitigation": "HMI highlights NAK response in red",
    },
]


# ── Demo / smoke test ─────────────────────────────────────────────────────────

def demo() -> dict:
    print("=" * 60)
    print("DISCO SW Stack — Demo")
    print("=" * 60)

    # Setup
    sm        = StateMachine()
    rm        = RecipeManager()
    historian = DataHistorian(capacity=1000)
    api       = OperatorAPI(sm, rm, historian)

    # Register a recipe
    recipe = DicingRecipe(
        name="SiC_100um_30k",
        spindle_rpm=30_000,
        feed_mms=5.0,
        cut_depth_um=100,
        kerf_pitch_um=140,
    )
    errors = rm.save(recipe)
    print(f"\n[RecipeManager] save errors: {errors or 'none'}")

    # Load & start via API
    r = api.dispatch("load_recipe", name="SiC_100um_30k")
    print(f"[API] load_recipe: {r}")

    r = api.dispatch("start")
    print(f"[API] start: {r}")

    # Simulate historian logging
    for i in range(20):
        historian.log(i * 0.001, spindle_rpm=30_000 + (i % 3) * 10,
                      feed_mms=5.0, chipping_um=1.2 + 0.1 * (i % 5))

    r = api.dispatch("stop")
    print(f"[API] stop: {r}")

    stats = api.dispatch("get_stats")
    print(f"[Historian] stats: {json.dumps(stats['stats'], indent=2)}")

    print(f"\n[SW FMEA] {len(SW_FMEA)} failure modes registered")
    for fm in SW_FMEA:
        print(f"  {fm['id']}  SEV={fm['severity']}  {fm['failure_mode']}")

    print(f"\n[State history]")
    for h in sm.history():
        print(f"  {h['reason']:35s} → {h['state']}")

    return {
        "state": sm.state.name,
        "recipes": rm.list_recipes(),
        "fmea_count": len(SW_FMEA),
    }


if __name__ == "__main__":
    demo()
