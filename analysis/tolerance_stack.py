"""
tolerance_stack.py

Dimensional tolerance stack-up analysis for DISCO dicing saw assembly.

Three methods:
  WC   — worst-case (arithmetic sum of tolerances)
  RSS  — root-sum-of-squares (statistical, assumes 3-sigma = tol)
  MC   — Monte Carlo (arbitrary distributions, yields Cpk)

Typical use case: spindle runout budget, blade mounting, chuck flatness,
stage positioning error budget.

Example
-------
>>> chain = ToleranceChain("spindle_runout_budget")
>>> chain.add("chuck_flatness",    nominal=0.0, tol=2.0, units="um")
>>> chain.add("blade_flange",      nominal=0.0, tol=1.5, units="um")
>>> chain.add("spindle_bearing",   nominal=0.0, tol=0.5, units="um")
>>> chain.add("thermal_expansion", nominal=0.0, tol=1.0, units="um", dist="uniform")
>>> r = chain.analyze()
>>> print(r)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Literal, Dict, Any

DistType = Literal["normal", "uniform", "triangular"]


@dataclass
class Dimension:
    name: str
    nominal: float
    tol: float          # half-range (±tol); for normal dist tol = 3σ
    units: str = "mm"
    dist: DistType = "normal"
    bilateral: bool = True  # True: ±tol, False: +0/-tol (unilateral, shifts mean)


@dataclass
class AnalysisResult:
    method: str
    nominal_sum: float
    tol_total: float
    upper: float
    lower: float
    units: str
    # MC-only
    mean: float = 0.0
    std: float = 0.0
    cpk: float = float("nan")
    yield_pct: float = float("nan")
    hist: np.ndarray = field(default_factory=lambda: np.array([]))
    hist_edges: np.ndarray = field(default_factory=lambda: np.array([]))

    def __repr__(self) -> str:
        base = (f"[{self.method}] nominal={self.nominal_sum:.4g} {self.units}  "
                f"tol=±{self.tol_total:.4g}  "
                f"[{self.lower:.4g}, {self.upper:.4g}]")
        if not math.isnan(self.cpk):
            base += f"  Cpk={self.cpk:.3f}  yield={self.yield_pct:.2f}%"
        return base


class ToleranceChain:
    """Stack-up analysis for a chain of dimensions.

    Parameters
    ----------
    name : str
        Descriptive name for the assembly / error budget.
    usl / lsl : float | None
        Upper / lower spec limits for Cpk calculation (MC only).
        If None, derived from WC bounds.
    """

    def __init__(self, name: str = "",
                 usl: float | None = None,
                 lsl: float | None = None) -> None:
        self.name = name
        self._dims: List[Dimension] = []
        self._usl = usl
        self._lsl = lsl

    # ── API ──────────────────────────────────────────────────────────────────

    def add(self, name: str, nominal: float, tol: float,
            units: str = "mm", dist: DistType = "normal",
            bilateral: bool = True) -> "ToleranceChain":
        self._dims.append(Dimension(name, nominal, tol, units, dist, bilateral))
        return self  # fluent

    def analyze(self, n_mc: int = 200_000) -> Dict[str, AnalysisResult]:
        if not self._dims:
            raise ValueError("no dimensions added")
        return {
            "WC":  self._wc(),
            "RSS": self._rss(),
            "MC":  self._mc(n_mc),
        }

    def summary(self, n_mc: int = 200_000) -> str:
        results = self.analyze(n_mc)
        lines = [f"=== Tolerance stack-up: {self.name} ===",
                 f"  Dimensions: {len(self._dims)}"]
        for method, r in results.items():
            lines.append(f"  {r}")
        return "\n".join(lines)

    # ── internal methods ──────────────────────────────────────────────────────

    def _nominal_sum(self) -> float:
        return sum(d.nominal for d in self._dims)

    def _units(self) -> str:
        # All dims should have same units; return first
        return self._dims[0].units if self._dims else ""

    def _wc(self) -> AnalysisResult:
        nom = self._nominal_sum()
        tol = sum(d.tol for d in self._dims)
        return AnalysisResult("WC", nom, tol,
                              upper=nom + tol, lower=nom - tol,
                              units=self._units())

    def _rss(self) -> AnalysisResult:
        nom = self._nominal_sum()
        tol = math.sqrt(sum(d.tol ** 2 for d in self._dims))
        return AnalysisResult("RSS", nom, tol,
                              upper=nom + tol, lower=nom - tol,
                              units=self._units())

    def _mc(self, n: int) -> AnalysisResult:
        rng = np.random.default_rng(42)
        samples = np.zeros(n)

        for d in self._dims:
            if d.dist == "normal":
                sigma = d.tol / 3.0
                s = rng.normal(d.nominal, sigma, n)
            elif d.dist == "uniform":
                lo = d.nominal - d.tol
                hi = d.nominal + d.tol
                s = rng.uniform(lo, hi, n)
            elif d.dist == "triangular":
                lo = d.nominal - d.tol
                hi = d.nominal + d.tol
                s = rng.triangular(lo, d.nominal, hi, n)
            else:
                raise ValueError(f"unknown dist '{d.dist}'")

            if not d.bilateral:
                # Unilateral: nominal is at +tol end, all values ≤ nominal
                s = d.nominal - np.abs(s - d.nominal)

            samples += s

        mean = float(np.mean(samples))
        std  = float(np.std(samples, ddof=1))

        # Spec limits for Cpk
        wc = self._wc()
        usl = self._usl if self._usl is not None else wc.upper
        lsl = self._lsl if self._lsl is not None else wc.lower

        if std > 0:
            cpu = (usl - mean) / (3.0 * std)
            cpl = (mean - lsl) / (3.0 * std)
            cpk = min(cpu, cpl)
        else:
            cpk = float("inf")

        in_spec = float(np.sum((samples >= lsl) & (samples <= usl))) / n * 100.0
        tol_3s  = 3.0 * std

        hist, edges = np.histogram(samples, bins=60)

        return AnalysisResult("MC", wc.nominal_sum, tol_3s,
                              upper=mean + tol_3s, lower=mean - tol_3s,
                              units=self._units(),
                              mean=mean, std=std,
                              cpk=cpk, yield_pct=in_spec,
                              hist=hist, hist_edges=edges)


# ── Preset assemblies ──────────────────────────────────────────────────────────

def spindle_runout_budget() -> ToleranceChain:
    """Typical DISCO dicing saw spindle total indicated runout (TIR) budget."""
    chain = ToleranceChain("spindle_TIR_budget", usl=5.0, lsl=-5.0)
    chain.add("spindle_bearing_runout", nominal=0.0, tol=0.5, units="um")
    chain.add("blade_flange_flatness",  nominal=0.0, tol=1.5, units="um")
    chain.add("blade_bore_clearance",   nominal=0.0, tol=1.0, units="um", dist="uniform")
    chain.add("chuck_flatness",         nominal=0.0, tol=2.0, units="um")
    chain.add("thermal_drift",          nominal=0.0, tol=1.0, units="um", dist="uniform")
    return chain


def stage_positioning_budget() -> ToleranceChain:
    """XY stage positioning error budget for 200mm wafer dicing."""
    chain = ToleranceChain("stage_positioning_budget", usl=3.0, lsl=-3.0)
    chain.add("encoder_resolution",    nominal=0.0, tol=0.1, units="um")
    chain.add("linear_scale_error",    nominal=0.0, tol=0.5, units="um")
    chain.add("abbe_error",            nominal=0.0, tol=1.0, units="um", dist="uniform")
    chain.add("thermal_expansion",     nominal=0.0, tol=0.8, units="um", dist="triangular")
    chain.add("vibration_floor",       nominal=0.0, tol=0.3, units="um")
    return chain


def blade_height_budget() -> ToleranceChain:
    """Z-axis blade height (cut depth) error budget."""
    chain = ToleranceChain("blade_height_budget", usl=5.0, lsl=-5.0)
    chain.add("z_encoder",             nominal=0.0, tol=0.2, units="um")
    chain.add("spindle_thermal_grow",  nominal=0.0, tol=2.0, units="um", dist="triangular")
    chain.add("blade_wear",            nominal=0.0, tol=3.0, units="um", dist="uniform")
    chain.add("wafer_thickness_var",   nominal=0.0, tol=1.5, units="um")
    return chain
