"""
sic.hybrid.base — unified process contract for hybrid 2nm dicing.

Every dicing method (laser grooving, stealth, plasma, blade) is wrapped as a
``DicingStage`` that consumes a ``Recipe`` (plus the upstream stage outcome,
if any) and returns a ``StageOutcome`` with a common set of physical KPIs.

This is the "全ダイシングの合致" layer: heterogeneous physics solvers
(fem.laser_groove_thermal_2d, fem.dicing_heat_sim, sic.physical_limits) are
made interchangeable behind one interface so the orchestrator and optimizer
can treat any route uniformly.

See sic/hybrid_dicing_2nm_plan.md for the strategy.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Process recipe ────────────────────────────────────────────────────────────
@dataclass
class Recipe:
    """All knobs for a hybrid dicing run (Stage A laser groove + Stage B singulation)."""

    # Wafer / die geometry (2nm-node-class defaults)
    material: str = "SiC"
    wafer_diameter_mm: float = 300.0
    wafer_thickness_um: float = 110.0      # thinned die
    street_width_um: float = 25.0          # tight advanced-node street
    die_size_mm: float = 5.0
    lowk_stack_um: float = 2.0             # BEOL ULK + cap thickness that MUST be cleared

    # ── Stage A: laser grooving (clears low-k/metal/TEG from the street) ──
    # ps cold ablation is the physically correct default for 2nm low-k streets
    # (ns leaves a >15µm HAZ → low-k delamination; fs is cleaner still).
    groove_power_W: float = 20.0
    groove_speed_mm_s: float = 100.0
    groove_freq_kHz: float = 100.0
    groove_beam_radius_um: float = 5.0
    groove_passes: int = 5
    groove_pulse_regime: str = "ps"        # ns / ps / fs
    groove_absorptivity: float = 0.85

    # ── Stage B: singulation method selection ──
    route: str = "laser+stealth"           # laser+stealth | laser+plasma | laser+blade

    # Stage B — stealth dicing knobs
    stealth_power_W: float = 4.0
    stealth_speed_mm_s: float = 300.0
    stealth_freq_kHz: float = 90.0
    stealth_focal_depth_um: float = 60.0
    stealth_NA: float = 0.85
    stealth_layers: int = 2
    stealth_regime: str = "ps"
    stealth_bessel: bool = False           # use Bessel beam variant

    # Stage B — plasma (DRIE) knobs
    plasma_pressure_mTorr: float = 20.0
    plasma_etch_rate_um_min: float = 8.0   # nominal vertical etch rate at AR=0

    # Stage B — blade knobs
    blade_W_um: float = 23.0
    blade_feed_mm_s: float = 80.0
    blade_spindle_rpm: float = 30000.0
    blade_cut_depth_um: float = 120.0
    blade_Q_W_per_m2: float = 5e7

    def copy(self, **overrides) -> "Recipe":
        d = asdict(self)
        d.update(overrides)
        return Recipe(**d)


# ── Stage outcome ─────────────────────────────────────────────────────────────
@dataclass
class StageOutcome:
    """Common KPI vector returned by every stage."""

    stage: str                 # "A" (groove) or "B" (singulation)
    method: str                # e.g. "laser_groove", "stealth", "plasma", "blade"
    kerf_um: float = 0.0               # cut / modified-layer width
    chipping_um: float = 0.0           # surface chipping (front/back side max)
    haz_um: float = 0.0                # heat-affected-zone depth
    subsurface_crack_um: float = 0.0   # latent crack depth under the kerf
    residual_stress_MPa: float = 0.0   # process-induced residual stress
    lowk_cleared_um: float = 0.0       # depth of street stack removed (Stage A only)
    throughput_wph: float = 0.0        # wafers per hour for this stage
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        return d


# ── Stage interface ───────────────────────────────────────────────────────────
class DicingStage(ABC):
    """A single dicing operation behind the unified contract."""

    #: short stable identifier
    name: str = "stage"

    @abstractmethod
    def simulate(self, recipe: Recipe,
                 upstream: Optional[StageOutcome] = None) -> StageOutcome:
        """Run the physics model and return a StageOutcome."""
        raise NotImplementedError


# ── Shared helpers ────────────────────────────────────────────────────────────
def wafer_lane_length_m(wafer_diameter_mm: float, die_size_mm: float) -> float:
    """
    Total dicing-lane length on a round wafer for a square die pitch.

    n_lanes per direction ≈ D / die_size; each lane ≈ D long; two orthogonal
    directions. Circular-fill factor ~π/4 accounts for the round wafer.
    """
    D = wafer_diameter_mm
    pitch = max(die_size_mm, 1e-3)
    n_per_dir = D / pitch
    total_mm = 2.0 * n_per_dir * D * (math.pi / 4.0)
    return total_mm * 1e-3


def scan_throughput_wph(recipe: Recipe, speed_mm_s: float, n_passes: int = 1,
                        index_overhead_s: float = 45.0) -> float:
    """Wafers/hour for a scan-based stage (laser groove, stealth, blade)."""
    lane_m = wafer_lane_length_m(recipe.wafer_diameter_mm, recipe.die_size_mm)
    lane_mm = lane_m * 1e3
    speed = max(speed_mm_s, 1e-6)
    scan_s = n_passes * lane_mm / speed
    per_wafer_s = scan_s + index_overhead_s
    return 3600.0 / max(per_wafer_s, 1e-6)
