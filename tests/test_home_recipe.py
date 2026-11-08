"""Unit tests required by decision_rules.md §2.4:
 (i)   the cell-mean formula recovers planted residuals;
 (ii)  the sum-coded model term equals the formula on complete balanced tables;
 (iii) both paths agree when no cell is missing.
Plus: H3 classification precedence and B1/B2 labels on constructed cases.
Run:  python -m pytest tests/ -q   (from the repository root)
"""
import numpy as np
import pandas as pd
import pytest
from analysis.common import ARCHS, RECIPES, HOME
from analysis import home_recipe as hr
from analysis import h3_classify as h3


def planted_table(rng, G=1.5):
    """4x4 means with known main effects and a double-centred interaction."""
    alpha = np.array([0.0, 1.0, 2.0, 3.0])
    rho = np.array([0.0, 1.0, 0.5, -0.5])
    inter = np.zeros((4, 4))
    inter[0, 1], inter[0, 0] = +G, -G          # EEGNet likes R2, dislikes R1
    inter[3, 0], inter[3, 1] = +G, -G          # CSOANet-v2 likes R1, dislikes R2
    assert np.allclose(inter.sum(0), 0) and np.allclose(inter.sum(1), 0)
    return 70 + alpha[:, None] + rho[None, :] + inter, inter


def make_cube(rng, S=60, G=1.5, noise=0.0):
    means, inter = planted_table(rng, G)
    u = rng.normal(0, 5, S)
    y = means[None] + u[:, None, None] + rng.normal(0, noise, (S, 4, 4))
    return y, inter


def cube_to_long(y):
    rows = []
    for s in range(y.shape[0]):
        for i, a in enumerate(ARCHS):
            for j, r in enumerate(RECIPES):
                rows.append(dict(subject=f"S{s:03d}", dataset="D1" if s % 2 else "D2",
                                 architecture=a, recipe=r, y=y[s, i, j]))
    return pd.DataFrame(rows)


# (i) ---------------------------------------------------------------------
def test_cellmean_formula_recovers_planted_residuals():
    rng = np.random.default_rng(0)
    means, inter = planted_table(rng, G=1.5)
    res = hr.interaction_residuals(means)
    assert np.allclose(res, inter, atol=1e-10)
    h = hr.home_advantage_cellmeans(means)
    assert h["EEGNet"] == pytest.approx(+1.5) and h["CSOANet-v2"] == pytest.approx(+1.5)
    assert h["MI-BMInet"] == pytest.approx(0.0)


def test_simple_difference_is_contaminated_by_main_effect():
    """Documents why 'home minus mean of others' is NOT used."""
    rng = np.random.default_rng(0)
    means, _ = planted_table(rng, G=0.0)         # no interaction at all
    v2 = ARCHS.index("CSOANet-v2")
    simple = means[v2, 0] - means[v2, 1:].mean()  # R1 minus mean of others
    assert abs(simple) > 0.1                       # non-zero although no home effect exists
    assert hr.home_advantage_cellmeans(means)["CSOANet-v2"] == pytest.approx(0.0)


# (ii) --------------------------------------------------------------------
def test_sum_coded_model_term_matches_formula_on_complete_table():
    rng = np.random.default_rng(1)
    y, inter = make_cube(rng, S=80, G=1.5, noise=0.3)
    long = cube_to_long(y)
    h_model, inter_model = hr.home_advantage_model(long)
    h_cell = hr.home_advantage_cellmeans(y.mean(0))
    for a in HOME:
        assert h_model[a] == pytest.approx(h_cell[a], abs=0.05)
    assert np.allclose(inter_model, hr.interaction_residuals(y.mean(0)), atol=0.05)


# (iii) -------------------------------------------------------------------
def test_two_paths_agree_when_nothing_missing():
    rng = np.random.default_rng(2)
    y, _ = make_cube(rng, S=60, G=1.0, noise=0.5)
    long = cube_to_long(y)
    out = hr.run(y, avg_long=long, n_boot=50)
    for a in HOME:
        assert out["h_model"][a] == pytest.approx(out["h_cellmeans"][a], abs=0.05)
    assert out["n_dropped"] == 0


def test_subject_drop_path_reports_dropped():
    rng = np.random.default_rng(3)
    y, _ = make_cube(rng, S=40, G=1.0, noise=0.5)
    y[0, 2, 3] = np.nan                            # one missing cell
    ps = hr.per_subject_home(y)
    assert ps.attrs["n_dropped"] == 1
    assert ps["subject_idx"].nunique() == 39


def test_drop_R4_recomputes_on_4x3():
    rng = np.random.default_rng(4)
    y, _ = make_cube(rng, S=40, G=1.0, noise=0.2)
    out = hr.run(y, drop_R4=True, n_boot=20)
    assert set(out["h_cellmeans"]) == set(HOME)
    # with the planted pattern, dropping R4 changes the centring, so values differ
    full = hr.run(y, n_boot=20)["h_cellmeans"]
    assert any(abs(out["h_cellmeans"][a] - full[a]) > 1e-6 for a in HOME)


# H3 classification -------------------------------------------------------
def test_classification_precedence():
    assert h3.classify_interval(0.3, 1.8) == "A"                     # better wins
    assert h3.classify_interval(0.3, 1.8, equiv_first=True) == "E"
    assert h3.classify_interval(-0.5, 1.5) == "E"
    assert h3.classify_interval(-0.5, 2.5) == "I"
    assert h3.classify_interval(-3.0, -0.2) == "B"


def test_change_counting_and_outcomes():
    def fake(cats):
        rows = [dict(pair=p, recipe=r, category=c, A="", B="", n=1, mean=0, half=0, lo=0, hi=0)
                for p, row in cats.items() for r, c in zip(RECIPES, row)]
        return pd.DataFrame(rows)
    # sign reversal -> A
    iv = fake({"p1": "ABAA", "p2": "EEEE", "p3": "IIII", "p4": "IIII", "p5": "IIII", "p6": "IIII"})
    out = h3.h3_outcome(h3.count_changes(iv))
    assert out["outcome"] == "A" and out["sign_reversals"] == 3
    # equivalent<->better only -> A'
    iv = fake({"p1": "AEAA", "p2": "EEEE", "p3": "IIII", "p4": "IIII", "p5": "IIII", "p6": "IIII"})
    assert h3.h3_outcome(h3.count_changes(iv))["outcome"] == "A'"
    # stable -> B1 (>=16 definitive, no definitive change)
    iv = fake({"p1": "AAAA", "p2": "EEEE", "p3": "BBBB", "p4": "AAAA", "p5": "IIII", "p6": "IIII"})
    assert h3.h3_outcome(h3.count_changes(iv))["outcome"] == "B1"
    # inconclusive -> B2 (<8 definitive)
    iv = fake({"p1": "AAII", "p2": "IIII", "p3": "IIII", "p4": "IIII", "p5": "IIII", "p6": "IIII"})
    assert h3.h3_outcome(h3.count_changes(iv))["outcome"] == "B2"
    # intermediate
    iv = fake({"p1": "AAAA", "p2": "EEEE", "p3": "IIII", "p4": "IIII", "p5": "IIII", "p6": "IIII"})
    assert h3.h3_outcome(h3.count_changes(iv))["outcome"] == "intermediate"
