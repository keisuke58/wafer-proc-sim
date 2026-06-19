"""
sic.hybrid.sensitivity — variable-importance ranking for hybrid 2nm dicing.

Which process knobs actually move the outcomes? For each route we sample the
recipe space, run the full hybrid model, and rank every variable by its
influence on:

    feasible        (2nm constraints satisfied, 0/1)
    die_strength_MPa
    throughput_wph
    residual_stress_MPa

Importance = |Spearman rank correlation| (monotonic, robust, sign-aware), with
standardized OLS β reported alongside as a linear cross-check. No SciPy/sklearn
dependency — rank stats are computed in NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from sic.hybrid.base import Recipe
from sic.hybrid.orchestrator import HybridDicer, ROUTES
from sic.hybrid.optimize import _space_for, _lhs, _decode

TARGETS = ("feasible", "die_strength_MPa", "throughput_wph", "residual_stress_MPa")


# ── rank statistics (NumPy-only) ────────────────────────────────────────────────
def _avg_rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (1..n), ties shared — like scipy.stats.rankdata('average')."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0   # average of [i..j] (1-based)
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(); y = y - y.mean()
    dx = np.sqrt((x * x).sum()); dy = np.sqrt((y * y).sum())
    if dx == 0 or dy == 0:
        return 0.0
    return float((x * y).sum() / (dx * dy))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_avg_rank(x), _avg_rank(y))


def standardized_betas(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Standardized OLS coefficients (z-scored X and y)."""
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
    yz = (y - y.mean()) / (y.std() + 1e-12)
    beta, *_ = np.linalg.lstsq(Xz, yz, rcond=None)
    return beta


# ── analysis ────────────────────────────────────────────────────────────────────
@dataclass
class SensitivityResult:
    route: str
    var_names: List[str]
    X: np.ndarray                       # (n, d) raw design
    targets: Dict[str, np.ndarray]      # name -> (n,) outcome
    spearman: Dict[str, np.ndarray]     # target -> (d,) signed ρ
    betas: Dict[str, np.ndarray]        # target -> (d,) standardized β
    n_feasible: int

    def ranking(self, target: str) -> List[tuple]:
        """[(var, rho, beta)] sorted by |rho| desc."""
        rho = self.spearman[target]
        beta = self.betas[target]
        idx = np.argsort(-np.abs(rho))
        return [(self.var_names[i], float(rho[i]), float(beta[i])) for i in idx]

    def overall_ranking(self) -> List[tuple]:
        """[(var, mean|rho| over the 3 continuous KPIs + feasibility)] desc."""
        M = np.abs(np.array([self.spearman[t] for t in TARGETS]))   # (4, d)
        score = M.mean(0)
        idx = np.argsort(-score)
        return [(self.var_names[i], float(score[i])) for i in idx]


def analyze(route: str, base: Recipe | None = None, n_samples: int = 400,
            seed: int = 0, dicer: HybridDicer | None = None) -> SensitivityResult:
    if route not in ROUTES:
        raise ValueError(f"unknown route {route!r}")
    base = base or Recipe()
    dicer = dicer or HybridDicer()
    space = _space_for(route)
    names = list(space)
    rng = np.random.default_rng(seed)
    U = _lhs(n_samples, len(names), rng)

    X = np.zeros((n_samples, len(names)))
    cols = {t: np.zeros(n_samples) for t in TARGETS}
    for k, u in enumerate(U):
        rec = _decode(route, u, base)
        res = dicer.run(rec)
        for j, name in enumerate(names):
            X[k, j] = getattr(rec, name)
        cols["feasible"][k] = float(res.feasibility.feasible)
        cols["die_strength_MPa"][k] = res.feasibility.die_strength_MPa
        cols["throughput_wph"][k] = res.throughput_wph
        cols["residual_stress_MPa"][k] = res.residual_stress_MPa

    feas_mask = cols["feasible"] > 0.5
    sp: Dict[str, np.ndarray] = {}
    bt: Dict[str, np.ndarray] = {}
    for t in TARGETS:
        # feasibility uses the full design; continuous KPIs use the feasible
        # subset (importance among shippable recipes), falling back to all.
        if t == "feasible":
            mask = np.ones(n_samples, bool)
        else:
            mask = feas_mask if feas_mask.sum() >= max(8, len(names) + 2) else np.ones(n_samples, bool)
        Xm, ym = X[mask], cols[t][mask]
        sp[t] = np.array([spearman(Xm[:, j], ym) for j in range(len(names))])
        bt[t] = standardized_betas(Xm, ym)

    return SensitivityResult(
        route=route, var_names=names, X=X, targets=cols,
        spearman=sp, betas=bt, n_feasible=int(feas_mask.sum()))


def analyze_all(base: Recipe | None = None, n_samples: int = 400, seed: int = 0,
                dicer: HybridDicer | None = None) -> Dict[str, SensitivityResult]:
    base = base or Recipe()
    dicer = dicer or HybridDicer()
    return {r: analyze(r, base, n_samples, seed + k, dicer)
            for k, r in enumerate(ROUTES)}
