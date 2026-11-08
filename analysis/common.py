"""Shared constants and loaders for recipe-x-arch confirmatory analysis.

Input table (long format, one row per subject x architecture x recipe x seed,
or seed-averaged with seed = -1):

    subject, dataset, fold, architecture, recipe, seed, bacc_f32, bacc_int8,
    train_acc, val_acc, training_failure

Accuracies in PERCENT. `training_failure` is 0/1 from the training-side rule.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

ARCHS = ["EEGNet", "ShallowConvNet", "MI-BMInet", "CSOANet-v2"]
RECIPES = ["R1", "R2", "R3", "R4"]
HOME = {"EEGNet": "R2", "MI-BMInet": "R3", "CSOANet-v2": "R1"}   # ShallowConvNet: none
SESOI = 2.0
ALPHA = 0.05
PAIRS = list(itertools.combinations(ARCHS, 2))                   # 6
RECIPE_PAIRS = list(itertools.combinations(RECIPES, 2))           # 6
N_INTERVALS = len(PAIRS) * len(RECIPES)                           # 24
BONF_LEVEL = 1 - ALPHA / N_INTERVALS                              # 0.99792


def load_long(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["architecture"] = pd.Categorical(df["architecture"], ARCHS)
    df["recipe"] = pd.Categorical(df["recipe"], RECIPES)
    assert not df["architecture"].isna().any(), "unknown architecture label"
    assert not df["recipe"].isna().any(), "unknown recipe label"
    return df


def seed_average(df: pd.DataFrame, metric: str = "bacc_f32",
                 drop_failures: bool = False) -> pd.DataFrame:
    """Average over seeds -> one row per subject x architecture x recipe.
    If drop_failures, rows flagged as training failures are removed first
    (sensitivity analysis only; primary analysis keeps them)."""
    d = df.copy()
    if drop_failures and "training_failure" in d:
        d = d[d["training_failure"] == 0]
    keys = ["subject", "dataset", "architecture", "recipe"]
    out = (d.groupby(keys, observed=True)[metric].mean()
             .reset_index().rename(columns={metric: "y"}))
    return out


def to_cube(avg: pd.DataFrame):
    """Long seed-averaged table -> (y[s,a,r], subjects, dataset_of_subject).
    Missing cells become NaN."""
    Y = avg.pivot_table(index="subject", columns=["architecture", "recipe"],
                        values="y", observed=False)
    Y = Y.reindex(columns=pd.MultiIndex.from_product([ARCHS, RECIPES]))
    y = Y.to_numpy().reshape(len(Y), len(ARCHS), len(RECIPES))
    ds = avg.drop_duplicates("subject").set_index("subject")["dataset"].reindex(Y.index)
    return y, list(Y.index), ds.to_numpy()


def holm(pvals):
    """Holm step-down adjusted p-values (array-like) -> np.ndarray."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj
