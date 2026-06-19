"""
sic.hybrid.plasma_stage — plasma (DRIE) singulation behind DicingStage.

Wraps the existing ARDE model (sic.physical_limits.plasma_arde_vec). Plasma
dicing is stress-free and chip-free, but the plasma cannot etch metal/low-k, so
it REQUIRES an upstream laser groove to clear the street first — enforced here
by reading ``upstream.lowk_cleared_um``.

Throughput is set by Aspect-Ratio-Dependent Etching: deep/narrow streets slow
the etch (etch_rate → 0 as AR → AR_max), which directly lowers WPH.
"""
from __future__ import annotations

from typing import Optional

from sic.hybrid.base import Recipe, StageOutcome, DicingStage
from sic.physical_limits import plasma_arde_vec


class PlasmaStage(DicingStage):
    """DRIE/Bosch through-wafer singulation (stress-free, mask-defined)."""

    name = "plasma"

    #: street must be cleared at least this deep upstream before plasma can start
    MIN_GROOVE_CLEAR_UM = 1.0
    #: mask/handling overhead per wafer [s]
    OVERHEAD_S = 120.0
    #: photoresist mask alignment margin per side [µm] → trench < street
    MASK_MARGIN_UM = 2.0

    def simulate(self, recipe: Recipe,
                 upstream: Optional[StageOutcome] = None) -> StageOutcome:
        r = plasma_arde_vec(
            trench_width_um=recipe.street_width_um,
            wafer_thickness_um=recipe.wafer_thickness_um,
            pressure_mTorr=recipe.plasma_pressure_mTorr,
        )
        etch_rate_frac = float(r["etch_rate"])     # 0..1, ARDE de-rating
        feasible_ar = bool(r["feasible"])

        # Effective vertical etch rate and through-wafer time.
        eff_rate_um_min = max(recipe.plasma_etch_rate_um_min * etch_rate_frac, 1e-6)
        etch_time_s = recipe.wafer_thickness_um / eff_rate_um_min * 60.0
        throughput = 3600.0 / (etch_time_s + self.OVERHEAD_S)

        # Was the street opened by the upstream groove?
        cleared = upstream.lowk_cleared_um if upstream is not None else 0.0
        street_open = cleared >= self.MIN_GROOVE_CLEAR_UM

        # Mask-defined trench is narrower than the street (alignment margin each
        # side), so a real die-edge margin exists.
        trench_um = max(recipe.street_width_um - 2.0 * self.MASK_MARGIN_UM,
                        0.6 * recipe.street_width_um)

        # Plasma is stress-free and essentially chip-free when the street is open.
        chipping = 0.05 if street_open else recipe.street_width_um   # blocked → wall
        subsurface = 0.0
        residual = 5.0      # near-zero, slight passivation-film stress

        return StageOutcome(
            stage="B", method=self.name,
            kerf_um=trench_um,
            chipping_um=chipping,
            haz_um=0.0,
            subsurface_crack_um=subsurface,
            residual_stress_MPa=residual,
            throughput_wph=throughput,
            extra={
                "etch_rate_frac": etch_rate_frac,
                "current_AR": float(r["current_AR"]),
                "AR_max": float(r["AR_max"]),
                "ar_feasible": feasible_ar,
                "street_open": street_open,
                "etch_time_s": etch_time_s,
            },
        )
