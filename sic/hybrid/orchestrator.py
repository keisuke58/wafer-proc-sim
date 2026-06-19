"""
sic.hybrid.orchestrator — chain Stage A (laser groove) → Stage B (singulation).

The orchestrator is the "合致" point: it runs the laser groove, passes its
outcome downstream, runs the selected singulation method, evaluates the 2nm
constraints, and reports the combined KPI (series throughput, die strength,
feasibility).

Routes:
    laser+stealth : groove → stealth dicing      (kerfless, low stress)
    laser+plasma  : groove → plasma/DRIE         (stress-free, mask-defined)
    laser+blade   : groove → blade singulation   (baseline, chipping-prone)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from sic.hybrid.base import Recipe, StageOutcome, DicingStage
from sic.hybrid.adapters import LaserGrooveStage, StealthStage, BladeStage
from sic.hybrid.plasma_stage import PlasmaStage
from sic.hybrid.constraints_2nm import Constraints2nm, FeasibilityReport


#: route name → Stage B class
ROUTES: Dict[str, Type[DicingStage]] = {
    "laser+stealth": StealthStage,
    "laser+plasma":  PlasmaStage,
    "laser+blade":   BladeStage,
}


@dataclass
class HybridResult:
    route: str
    recipe: Recipe
    stage_a: StageOutcome
    stage_b: StageOutcome
    feasibility: FeasibilityReport
    throughput_wph: float          # series (per-wafer time adds)

    @property
    def residual_stress_MPa(self) -> float:
        """Process-induced residual stress at the *die sidewall*.

        The groove sits in the (removed) street and only grazes the top die
        corner, so the through-thickness sidewall — and thus die-edge residual
        stress — is defined mainly by the Stage-B singulation, with a small
        groove contribution at the corner.
        """
        return (0.85 * self.stage_b.residual_stress_MPa
                + 0.15 * self.stage_a.residual_stress_MPa)

    # ── objectives (orchestrator's contract with the optimizer) ──
    @property
    def objectives(self) -> Dict[str, float]:
        """Maximization objectives. Residual stress enters as its negation."""
        return {
            "die_strength_MPa": self.feasibility.die_strength_MPa,
            "throughput_wph":   self.throughput_wph,
            "neg_residual_MPa": -self.residual_stress_MPa,
        }

    def summary(self) -> Dict[str, float]:
        f = self.feasibility
        return {
            "route":             self.route,
            "feasible":          f.feasible,
            "die_strength_MPa":  round(f.die_strength_MPa, 1),
            "throughput_wph":    round(self.throughput_wph, 2),
            "residual_stress_MPa": round(self.residual_stress_MPa, 1),
            "total_chipping_um": round(f.total_chipping_um, 3),
            "max_haz_um":        round(f.max_haz_um, 3),
            "governing_crack_um": round(f.governing_crack_um, 3),
            "n_violations":      len(f.violations),
        }


class HybridDicer:
    """Runs a hybrid route end-to-end and scores it against 2nm constraints."""

    def __init__(self, constraints: Constraints2nm | None = None,
                 stage_a: DicingStage | None = None):
        self.constraints = constraints or Constraints2nm()
        self.stage_a = stage_a or LaserGrooveStage()

    def run(self, recipe: Recipe) -> HybridResult:
        if recipe.route not in ROUTES:
            raise ValueError(
                f"unknown route {recipe.route!r}; choose from {sorted(ROUTES)}")

        out_a = self.stage_a.simulate(recipe, upstream=None)
        stage_b = ROUTES[recipe.route]()
        out_b = stage_b.simulate(recipe, upstream=out_a)

        feas = self.constraints.evaluate(recipe, out_a, out_b)

        # Series throughput: per-wafer times add (Stage A then Stage B).
        wph_a = max(out_a.throughput_wph, 1e-9)
        wph_b = max(out_b.throughput_wph, 1e-9)
        series_wph = 1.0 / (1.0 / wph_a + 1.0 / wph_b)

        return HybridResult(
            route=recipe.route, recipe=recipe,
            stage_a=out_a, stage_b=out_b,
            feasibility=feas, throughput_wph=series_wph,
        )

    def run_all_routes(self, recipe: Recipe) -> Dict[str, HybridResult]:
        """Run every route on the same base recipe (groove params shared)."""
        return {name: self.run(recipe.copy(route=name)) for name in ROUTES}
