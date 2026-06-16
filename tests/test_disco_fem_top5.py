"""Tests for the DISCO FEM Top-5 unified driver / registry."""
import os

import pytest

from fem import disco_fem_top5 as top5


def test_registry_is_ranked_1_to_5():
    ranks = [e.rank for e in top5.TOP5]
    assert ranks == [1, 2, 3, 4, 5]
    # priority is strictly decreasing with rank (it IS the ranking axis)
    prios = [e.priority for e in top5.TOP5]
    assert prios == sorted(prios, reverse=True)


def test_every_entry_is_complete():
    for e in top5.TOP5:
        assert e.title and e.disco_product and e.physics
        assert e.solver in ("ABAQUS/Standard", "ABAQUS/Explicit", "analytic")
        assert e.owner
        assert e.run_cmd, f"#{e.rank} missing run_cmd"


def test_get_and_bad_rank():
    assert top5.get(1).key == "warpage"
    with pytest.raises(KeyError):
        top5.get(99)


def test_analytic_demos_return_numbers():
    demos = top5.run_demos()
    # die-strength proxy: larger chipping → lower fracture strength
    ds = demos["die_strength"]
    assert ds["chip_2um"]["fracture_MPa"] > ds["chip_10um"]["fracture_MPa"]
    # modal proxy: design rule wants f1 >> f_blade
    assert demos["modal"]["separation_ratio"] > 1.0
    # TAIKO ring reduces sag vs uniform thinning
    assert demos["warpage"]["uniform_vs_taiko_sag_x"] > 1.0
    # #2 chipping proxy: feed↑ → chipping↑, and fit to real data is tight
    cw = demos["chipping"]["sweep_depth80um_30krpm"]
    chips = [cw[k]["chipping_um"] for k in
             ("feed_2mm_s", "feed_3.5mm_s", "feed_5mm_s")]
    assert chips == sorted(chips)               # monotone increasing in feed
    assert demos["chipping"]["real_data_fit_MAE_um"] < 1.5
    assert demos["chipping"]["strength_loss_MPa_per_um_at_5um"] < 0


def test_empirical_chipping_matches_experiment():
    """Owner-side chipping law is anchored to the 11 Grade-A points."""
    import numpy as np
    from fem.die_strength_model import empirical_chipping_um
    from ml.surrogate_transformer import make_experimental_data
    X, y = make_experimental_data()
    pred = np.array([empirical_chipping_um(d, f, s) for d, f, s in X[:, :3]])
    assert float(np.mean(np.abs(pred - y))) < 1.5   # MAE < 1.5 µm


def test_report_writes_markdown(tmp_path):
    out = top5.write_report(out_md=str(tmp_path / "r.md"), with_figure=False)
    assert os.path.exists(out)
    txt = open(out).read()
    assert "Top-5" in txt and "TAIKO" in txt


def test_table_renders():
    t = top5.render_table()
    for e in top5.TOP5:
        assert e.title in t
