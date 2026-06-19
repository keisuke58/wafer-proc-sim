"""
sic.hybrid.optimize — constrained multi-objective recipe search per route.

Objectives (both maximized): die break strength [MPa] and throughput [WPH],
subject to the 2nm feasibility constraints (Constraints2nm). For each route we
Latin-Hypercube sample the recipe space, evaluate via HybridDicer, keep the
feasible points, and extract the non-dominated (Pareto) front.

Reuses optimization._nsga2_kernel.nondominated_sort when the compiled kernel is
present; otherwise a NumPy fallback gives identical fronts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from sic.hybrid.base import Recipe
from sic.hybrid.orchestrator import HybridDicer, HybridResult, ROUTES

# Optional compiled non-dominated sort
try:
    from optimization import _nsga2_kernel as _nsga2
    _HAS_NSGA2 = True
except Exception:  # pragma: no cover - kernel optional
    _HAS_NSGA2 = False


# ── search space ───────────────────────────────────────────────────────────────
# name -> (lo, hi, is_int).  Groove knobs are shared by every route.
_GROOVE_SPACE = {
    "groove_power_W":   (12.0, 40.0, False),
    "groove_speed_mm_s": (80.0, 400.0, False),
    "groove_passes":    (2, 6, True),
}
_ROUTE_SPACE: Dict[str, dict] = {
    "laser+stealth": {
        "stealth_power_W":       (2.0, 8.0, False),
        "stealth_speed_mm_s":    (150.0, 500.0, False),
        "stealth_focal_depth_um": (40.0, 90.0, False),
        "stealth_layers":        (1, 3, True),
    },
    "laser+plasma": {
        "plasma_pressure_mTorr":  (5.0, 40.0, False),
        "plasma_etch_rate_um_min": (4.0, 15.0, False),
    },
    "laser+blade": {
        "blade_feed_mm_s": (40.0, 120.0, False),
        "blade_W_um":      (15.0, 30.0, False),
    },
}


#: objective columns (all framed as MAXIMIZATION)
OBJECTIVES = ("die_strength_MPa", "throughput_wph", "neg_residual_MPa")


@dataclass
class ParetoPoint:
    recipe: Recipe
    result: HybridResult
    die_strength_MPa: float
    throughput_wph: float
    residual_stress_MPa: float

    def obj_vector(self) -> list:
        """Maximization vector [strength, throughput, -residual]."""
        return [self.die_strength_MPa, self.throughput_wph, -self.residual_stress_MPa]


def _space_for(route: str) -> dict:
    space = dict(_GROOVE_SPACE)
    space.update(_ROUTE_SPACE[route])
    return space


def _lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin-Hypercube sample in the unit cube [0,1]^d."""
    cut = np.linspace(0, 1, n + 1)
    u = rng.random((n, d))
    pts = cut[:n, None] + u * (1.0 / n)
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def _decode(route: str, u: np.ndarray, base: Recipe) -> Recipe:
    space = _space_for(route)
    over = {"route": route}
    for val, (name, (lo, hi, is_int)) in zip(u, space.items()):
        x = lo + val * (hi - lo)
        over[name] = int(round(x)) if is_int else float(x)
    return base.copy(**over)


def nondominated_max(F: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated front for MAXIMIZATION of all columns."""
    if len(F) == 0:
        return np.array([], dtype=int)
    if _HAS_NSGA2:
        # kernel minimizes → negate
        fronts = _nsga2.nondominated_sort(np.ascontiguousarray(-F, dtype=np.float64))
        return np.asarray(fronts[0], dtype=int)
    n = len(F)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(F[j] >= F[i]) and np.any(F[j] > F[i]):
                dominated[i] = True
                break
    return np.where(~dominated)[0]


@dataclass
class RouteSearch:
    route: str
    n_eval: int
    n_feasible: int
    pareto: List[ParetoPoint]
    best_strength: Optional[ParetoPoint]      # max die strength among feasible
    best_throughput: Optional[ParetoPoint]    # max WPH among feasible
    best_lowstress: Optional[ParetoPoint]     # min residual stress among feasible


def optimize_route(route: str, base: Recipe | None = None, n_samples: int = 200,
                   seed: int = 0, dicer: HybridDicer | None = None) -> RouteSearch:
    """Sample + evaluate + Pareto-filter one route over 3 objectives."""
    if route not in ROUTES:
        raise ValueError(f"unknown route {route!r}")
    base = base or Recipe()
    dicer = dicer or HybridDicer()
    rng = np.random.default_rng(seed)
    space = _space_for(route)
    U = _lhs(n_samples, len(space), rng)

    feas: List[ParetoPoint] = []
    for u in U:
        rec = _decode(route, u, base)
        res = dicer.run(rec)
        if res.feasibility.feasible:
            feas.append(ParetoPoint(
                recipe=rec, result=res,
                die_strength_MPa=res.feasibility.die_strength_MPa,
                throughput_wph=res.throughput_wph,
                residual_stress_MPa=res.residual_stress_MPa))

    pareto: List[ParetoPoint] = []
    best_s = best_t = best_r = None
    if feas:
        F = np.array([p.obj_vector() for p in feas])
        idx = nondominated_max(F)
        pareto = [feas[i] for i in idx]
        best_s = max(feas, key=lambda p: p.die_strength_MPa)
        best_t = max(feas, key=lambda p: p.throughput_wph)
        best_r = min(feas, key=lambda p: p.residual_stress_MPa)

    return RouteSearch(
        route=route, n_eval=n_samples, n_feasible=len(feas),
        pareto=pareto, best_strength=best_s, best_throughput=best_t,
        best_lowstress=best_r)


def compare_routes(base: Recipe | None = None, n_samples: int = 200,
                   seed: int = 0, dicer: HybridDicer | None = None
                   ) -> Dict[str, RouteSearch]:
    """Optimize every route on the same base recipe → 'strongest 2nm' comparison."""
    base = base or Recipe()
    dicer = dicer or HybridDicer()
    return {
        route: optimize_route(route, base, n_samples, seed + k, dicer)
        for k, route in enumerate(ROUTES)
    }


def _normalized(F: np.ndarray) -> np.ndarray:
    """Min-max normalize columns to [0,1]; flat columns (zero span) → 0."""
    span = F.max(0) - F.min(0)
    out = np.zeros_like(F)
    nz = span > 0
    out[:, nz] = (F[:, nz] - F.min(0)[nz]) / span[nz]
    return out


def knee_point(search: RouteSearch, weights=None) -> Optional[ParetoPoint]:
    """Pareto knee = max weighted sum of min-max-normalized objectives.

    Robust to flat objectives (e.g. SiC die strength pinned at the pristine cap):
    a zero-span objective contributes 0 and the choice falls back to the others.
    """
    if not search.pareto:
        return None
    F = np.array([p.obj_vector() for p in search.pareto])
    norm = _normalized(F)
    w = np.ones(F.shape[1]) if weights is None else np.asarray(weights, float)
    return search.pareto[int(np.argmax(norm @ w))]


def strongest_route(searches: Dict[str, "RouteSearch"], weights=None):
    """Pick the overall 'strongest 2nm' route by comparing route knees across the
    same three objectives. Returns (route_name, ParetoPoint) or (None, None)."""
    knees = [(name, knee_point(s, weights)) for name, s in searches.items()]
    knees = [(n, k) for n, k in knees if k is not None]
    if not knees:
        return None, None
    F = np.array([k.obj_vector() for _, k in knees])
    norm = _normalized(F)
    w = np.ones(F.shape[1]) if weights is None else np.asarray(weights, float)
    return knees[int(np.argmax(norm @ w))]
