"""
pytest suite for all 4 Keyence modules.

  fem/keyence_metrology_model  — VK-X confocal metrology
  fem/keyence_lj_profiler      — LJ-X laser triangulation profiler
  fem/keyence_iv3_vision       — IV3 inline machine vision
  pipeline/keyence_apc         — EnKF R2R APC with VK-X feedback
"""

import sys
sys.path.insert(0, '/home/nishioka/git/wafer-proc-sim')

import numpy as np
import numpy.testing as npt
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# fem.keyence_metrology_model
# ─────────────────────────────────────────────────────────────────────────────
from fem.keyence_metrology_model import (
    sic_reflectance,
    simulate_chip_measurement,
    batch_measurement,
    compute_roi,
    ROI_SCENARIOS,
)


def test_sic_reflectance():
    """4H-SiC reflectance at 405 nm: Sellmeier model gives ~33 % (n≈3.75)."""
    R = sic_reflectance(405)
    assert 0.18 <= R <= 0.40


def test_chip_measurement_grade_a():
    """VK-X 100× (σ=0.28 µm) should yield Grade A."""
    result = simulate_chip_measurement(5.0, 'vkx_100x')
    assert result.grade == 'A'


def test_chip_measurement_grade_d():
    """Visual measurement (σ=4.50 µm) should yield Grade D."""
    result = simulate_chip_measurement(5.0, 'visual')
    assert result.grade == 'D'


def test_roi_vkx_positive():
    """VK-X Grade-A scenario should show positive 5-year net benefit."""
    roi = compute_roi(ROI_SCENARIOS[0])
    assert roi['net_5yr'] > 0


def test_batch_measurement_count():
    """batch_measurement should return exactly one result per input width."""
    results = batch_measurement(np.ones(10) * 5.0, 'vkx_100x')
    assert len(results) == 10


# ─────────────────────────────────────────────────────────────────────────────
# fem.keyence_lj_profiler  (skip entire group if module absent)
# ─────────────────────────────────────────────────────────────────────────────
lj = pytest.importorskip('fem.keyence_lj_profiler')


def test_kerf_profile_created():
    """Simulated kerf profile must report a positive measured kerf width."""
    profile = lj.kerf_profile_scan(50.0, 150, 20)
    assert profile.kerf_width_measured_um > 0


def test_wafer_warp_shape():
    """wafer_warp_map(10, 10, ...) must return a (10, 10) array."""
    Z = lj.wafer_warp_map(10, 10, 40)
    assert Z.shape == (10, 10)


# ─────────────────────────────────────────────────────────────────────────────
# fem.keyence_iv3_vision  (skip entire group if module absent)
# ─────────────────────────────────────────────────────────────────────────────
iv3 = pytest.importorskip('fem.keyence_iv3_vision')


def test_inspection_results_length():
    """simulate_inline_inspection should return exactly n_dies results."""
    results = iv3.simulate_inline_inspection(
        np.full(50, 4.0), np.full(50, 1.0), 50
    )
    assert len(results) == 50


@pytest.mark.parametrize("key", ['n_pass', 'n_fail', 'yield_pct'])
def test_yield_dict_keys(key):
    """yield_from_inspection must contain the expected summary keys."""
    stats = iv3.yield_from_inspection([])
    assert key in stats


# ─────────────────────────────────────────────────────────────────────────────
# pipeline.keyence_apc
# ─────────────────────────────────────────────────────────────────────────────
from pipeline.keyence_apc import run_apc


@pytest.fixture(scope='module')
def apc_result():
    return run_apc(n_wafers=30, seed=0)


def test_apc_vkx_beats_open(apc_result):
    """VK-X EnKF control RMSE must be lower than open-loop RMSE."""
    assert apc_result['rmse_vkx'] < apc_result['rmse_open']


def test_apc_vkx_beats_opt(apc_result):
    """VK-X EnKF control RMSE must be lower than optical-scope EnKF RMSE."""
    assert apc_result['rmse_vkx'] < apc_result['rmse_opt']
