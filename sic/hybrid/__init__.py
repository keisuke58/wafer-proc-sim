"""
sic.hybrid — unified hybrid 2nm dicing process model.

Combines laser grooving (Stage A) with a selectable singulation method
(Stage B: stealth / plasma / blade) behind one ``DicingStage`` contract,
then evaluates 2nm feasibility constraints and optimizes the recipe.

Public API:
    from sic.hybrid import (
        Recipe, StageOutcome, DicingStage,
        LaserGrooveStage, StealthStage, BladeStage, PlasmaStage,
        Constraints2nm, HybridDicer, ROUTES,
    )
"""
from __future__ import annotations

import os
import sys

# Make the repo root importable so `fem.*`, `data.*`, `sic.*` resolve even
# when this package is imported as a standalone path.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sic.hybrid.base import (          # noqa: E402
    Recipe, StageOutcome, DicingStage,
    wafer_lane_length_m, scan_throughput_wph,
)
from sic.hybrid.adapters import (      # noqa: E402
    LaserGrooveStage, StealthStage, BladeStage,
)
from sic.hybrid.plasma_stage import PlasmaStage          # noqa: E402
from sic.hybrid.constraints_2nm import (                 # noqa: E402
    Constraints2nm, FeasibilityReport,
)
from sic.hybrid.orchestrator import HybridDicer, ROUTES  # noqa: E402

__all__ = [
    "Recipe", "StageOutcome", "DicingStage",
    "wafer_lane_length_m", "scan_throughput_wph",
    "LaserGrooveStage", "StealthStage", "BladeStage", "PlasmaStage",
    "Constraints2nm", "FeasibilityReport",
    "HybridDicer", "ROUTES",
]
