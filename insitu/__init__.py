"""
insitu — process-embedded SHM for dicing ("結晶の聴診器").

Treat the wafer being diced as a structure under structural health monitoring:
listen to its acoustic emission, fit a healthy baseline on the first windows of
the cut, and flag incipient subsurface cracks by Mahalanobis novelty detection —
the Stage-0 detector from structure-agnostic SHM, ported from guided-wave/COPV
monitoring to in-process dicing.

Public API:
    from insitu import (simulate_ae, window_features, MahaDetector,
                        monitor, monitor_all, FEATURE_NAMES)
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from insitu.listen import (   # noqa: E402
    AETrace, AEEvent, simulate_ae, window_features, FEATURE_NAMES,
    MahaDetector, MonitorResult, monitor, monitor_all,
)

__all__ = [
    "AETrace", "AEEvent", "simulate_ae", "window_features", "FEATURE_NAMES",
    "MahaDetector", "MonitorResult", "monitor", "monitor_all",
]
