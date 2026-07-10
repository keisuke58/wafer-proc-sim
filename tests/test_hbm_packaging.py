"""Tests for packaging/ (HBM stack warpage, TSV thermal stress) — the
back-end 3D-stacking physics fill-in (19th portfolio entry)."""
import numpy as np
import pytest

hs = pytest.importorskip("hbmpkg.hbm_stacking")
tsv = pytest.importorskip("hbmpkg.tsv_thermal_stress")


class TestHbmStacking:
    def test_single_layer_gives_zero_curvature(self):
        layers = hs._layers_from_spec([(hs.SI, 30.0)])
        kappa, _ = hs.stack_curvature(layers, hs.DELTA_T_K)
        assert abs(kappa) < 1e-9

    def test_identical_layers_give_zero_curvature(self):
        layers = hs._layers_from_spec([(hs.SI, 30.0)] * 6)
        kappa, _ = hs.stack_curvature(layers, hs.DELTA_T_K)
        assert abs(kappa) < 1e-9

    def test_fine_slice_matches_closed_form(self):
        layers = hs.hi_stack(8, bump=True)
        k_exact, _ = hs.stack_curvature(layers, hs.DELTA_T_K)
        k_fine, _ = hs.stack_curvature_fine_slice(layers, hs.DELTA_T_K)
        assert abs(k_fine - k_exact) / abs(k_exact) < 0.001

    def test_package_bow_monotone_in_stack_height(self):
        res = hs.run()
        bows = [res["package_level_bow_vs_height"][f"{n}-Hi"]["bow_um"]
                for n in (4, 8, 12, 16)]
        assert all(bows[i] > bows[i + 1] for i in range(len(bows) - 1))

    def test_hybrid_unit_cell_curvature_much_smaller_than_bump(self):
        k_bump, _ = hs.stack_curvature(hs.unit_cell(bump=True), hs.DELTA_T_K)
        k_hyb, _ = hs.stack_curvature(hs.unit_cell(bump=False), hs.DELTA_T_K)
        assert abs(k_bump / k_hyb) > 50


class TestTsvThermalStress:
    def test_identical_materials_zero_stress(self):
        sol = tsv.solve_via(2.5e-6, 22.57e-6, core=tsv.SI, shell=tsv.SI)
        assert abs(sol["P_core"]) < 1e-6
        assert abs(sol["P_shell"]) < 1e-6
        assert abs(sol["Q_shell"]) < 1e-6

    def test_cu_core_and_si_interface_are_nonzero_and_opposite_sign(self):
        sol = tsv.solve_via(2.5e-6, 22.57e-6)
        hoop_si = sol["P_shell"] + sol["Q_shell"] / (2.5e-6) ** 2
        assert sol["P_core"] > 0
        assert hoop_si < 0

    def test_dilute_limit_converges(self):
        b1 = tsv.unit_cell_radius(200e-6)
        b2 = tsv.unit_cell_radius(500e-6)
        s1 = tsv.solve_via(2.5e-6, b1)
        s2 = tsv.solve_via(2.5e-6, b2)
        h1 = s1["P_shell"] + s1["Q_shell"] / (2.5e-6) ** 2
        h2 = s2["P_shell"] + s2["Q_shell"] / (2.5e-6) ** 2
        assert abs(h2 - h1) / abs(h1) < 0.001

    def test_tighter_pitch_increases_stress_magnitude(self):
        res = tsv.run()
        rows = res["pitch_sweep"]
        vals = [abs(rows[f"{p}um"]["si_interface_hoop_MPa"])
                for p in (20, 30, 40, 60, 80, 200, 500)]
        assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
