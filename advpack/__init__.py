"""
advpack —  connected advanced-packaging process line.

Chains the four advanced-packaging operations behind one contract so the whole
"thin → warp → bond → singulate" flow can be simulated end-to-end:

    ThinGrind  → backgrinding to a thin die (+ subsurface damage, grind stress)
    Warpage    → Stoney bow from stress mismatch (explodes as the wafer thins)
    Bond       → hybrid bonding (overlay error & voids driven by warp)
    Singulate  → dice the bonded stack (chipping / interface delamination)

This is the "切る → 積んで切る" shift: the centre of gravity moves from pure
dicing to grind + bond + singulation of stacked/brittle substrates.

Public API:
    from advpack import (StackConfig, StackState, PackagingStage,
                           ThinGrindStage, WarpageStage, BondStage,
                           SingulateStage, PackagingLine)
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from advpack.line import (   # noqa: E402
    StackConfig, StackState, PackagingStage,
    ThinGrindStage, WarpageStage, BondStage, SingulateStage,
    PackagingLine, LineResult, stoney_bow_um, min_manufacturable_thickness,
)

__all__ = [
    "StackConfig", "StackState", "PackagingStage",
    "ThinGrindStage", "WarpageStage", "BondStage", "SingulateStage",
    "PackagingLine", "LineResult", "stoney_bow_um", "min_manufacturable_thickness",
]
