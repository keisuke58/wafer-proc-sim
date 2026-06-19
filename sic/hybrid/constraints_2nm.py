"""
sic.hybrid.constraints_2nm — 2nm-node feasibility constraints + die-strength model.

A hybrid route is "feasible" for an advanced (2nm-class) node only if it:
  1. clears the BEOL low-k/metal/TEG stack in the street (else blade/plasma fail),
  2. keeps the HAZ below the low-k thermal-damage budget (no delamination),
  3. keeps total chipping inside the street margin (no neighbour-die damage),
  4. leaves enough die break strength (subsurface crack → Griffith strength).

Die strength is estimated from the governing (deepest) subsurface crack across
all stages via LEFM:  σ_f = K_Ic / (Y·√(π·c)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from sic.hybrid.base import Recipe, StageOutcome
from data.materials.material_properties import ALL_MATERIALS


@dataclass
class FeasibilityReport:
    feasible: bool
    die_strength_MPa: float
    total_chipping_um: float
    max_haz_um: float
    governing_crack_um: float
    violations: List[str] = field(default_factory=list)
    penalty: float = 0.0          # 0 when feasible; >0 sum of normalized excesses

    def __bool__(self) -> bool:
        return self.feasible


class Constraints2nm:
    """Feasibility checker + die-strength model for a hybrid dicing route."""

    def __init__(
        self,
        lowk_haz_max_um: float = 2.0,        # thermal budget before ULK delamination
        chipping_max_um: float = 3.0,        # absolute front-side chipping ceiling
        die_strength_min_MPa: float = 250.0, # absolute min die break strength
        strength_min_frac: float | None = None,  # if set, threshold = frac·σ_flex
        Y_geom: float = 1.12,                # surface-crack geometry factor
        floor_crack_um: float = 0.05,        # min crack so strength stays finite
    ):
        self.lowk_haz_max_um = lowk_haz_max_um
        self.chipping_max_um = chipping_max_um
        self.die_strength_min_MPa = die_strength_min_MPa
        self.strength_min_frac = strength_min_frac
        self.Y_geom = Y_geom
        self.floor_crack_um = floor_crack_um

    # ── per-material die-strength requirement ──
    def min_strength(self, material: str) -> float:
        """Required die strength. Absolute by default; if ``strength_min_frac`` is
        set, the requirement scales with the material's flexural strength
        (a brittle material is held to a brittle-material spec)."""
        if self.strength_min_frac is not None:
            m = ALL_MATERIALS.get(material, ALL_MATERIALS["SiC"])
            return self.strength_min_frac * m["sigma_flex"] / 1e6
        return self.die_strength_min_MPa

    # ── die strength from governing subsurface crack ──
    def die_strength_MPa(self, material: str, crack_um: float) -> float:
        m = ALL_MATERIALS.get(material, ALL_MATERIALS["SiC"])
        c = max(crack_um, self.floor_crack_um) * 1e-6           # m
        sigma = m["K_Ic"] / (self.Y_geom * math.sqrt(math.pi * c))   # Pa
        return min(sigma, m["sigma_flex"]) / 1e6                # MPa, capped

    # ── full evaluation ──
    def evaluate(self, recipe: Recipe,
                 stage_a: StageOutcome, stage_b: StageOutcome) -> FeasibilityReport:
        viols: List[str] = []
        penalty = 0.0

        # (1) low-k stack cleared by the groove?
        if stage_a.lowk_cleared_um < recipe.lowk_stack_um:
            deficit = recipe.lowk_stack_um - stage_a.lowk_cleared_um
            viols.append(
                f"low-k not cleared: groove {stage_a.lowk_cleared_um:.2f} < "
                f"stack {recipe.lowk_stack_um:.2f} µm")
            penalty += deficit / max(recipe.lowk_stack_um, 1e-6)

        # plasma route additionally needs the street physically open
        if stage_b.method == "plasma" and not stage_b.extra.get("street_open", False):
            viols.append("plasma street blocked (upstream groove too shallow)")
            penalty += 1.0
        if stage_b.method == "plasma" and not stage_b.extra.get("ar_feasible", True):
            viols.append(
                f"plasma AR stall: AR {stage_b.extra.get('current_AR', 0):.1f} "
                f"> AR_max {stage_b.extra.get('AR_max', 0):.1f}")
            penalty += 1.0

        # (2) HAZ within low-k thermal budget (both stages)
        max_haz = max(stage_a.haz_um, stage_b.haz_um)
        if max_haz > self.lowk_haz_max_um:
            viols.append(
                f"HAZ {max_haz:.2f} > low-k budget {self.lowk_haz_max_um:.2f} µm "
                "(delamination risk)")
            penalty += (max_haz - self.lowk_haz_max_um) / self.lowk_haz_max_um

        # (3) chipping inside street margin AND below absolute ceiling
        total_chip = max(stage_a.chipping_um, stage_b.chipping_um)
        kerf = max(stage_a.kerf_um, stage_b.kerf_um)
        street_margin = max(0.0, (recipe.street_width_um - kerf) / 2.0)
        if total_chip > street_margin:
            viols.append(
                f"chipping {total_chip:.2f} > street margin {street_margin:.2f} µm")
            penalty += (total_chip - street_margin) / max(street_margin, 0.5)
        if total_chip > self.chipping_max_um:
            viols.append(
                f"chipping {total_chip:.2f} > ceiling {self.chipping_max_um:.2f} µm")
            penalty += (total_chip - self.chipping_max_um) / self.chipping_max_um

        # (4) die break strength
        governing_crack = max(stage_a.subsurface_crack_um, stage_b.subsurface_crack_um)
        strength = self.die_strength_MPa(recipe.material, governing_crack)
        thr = self.min_strength(recipe.material)
        if strength < thr:
            viols.append(f"die strength {strength:.0f} < min {thr:.0f} MPa")
            penalty += (thr - strength) / thr

        return FeasibilityReport(
            feasible=(len(viols) == 0),
            die_strength_MPa=strength,
            total_chipping_um=total_chip,
            max_haz_um=max_haz,
            governing_crack_um=governing_crack,
            violations=viols,
            penalty=penalty,
        )
