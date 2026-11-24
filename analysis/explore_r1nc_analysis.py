#!/usr/bin/env python3
"""Exploratory analysis of F1 under R1 without cropping ("R1nc"; S4 row 15).
Compares R1nc with R1 and R4 per architecture, and the EEGNet - CSOANet-v2 gap under
R1nc against R1 (the H2b-type contrast), subject-level and with a random intercept per
dataset x fold cluster. Reads result files only.

Usage: python -m analysis.explore_r1nc_analysis --res results/confirmatory --nc results/exploratory_r1nocrop --out docs/exploratory_r1nocrop
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from scipy import stats
from .common import ARCHS
from .revision_v11 import cluster_mean_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/confirmatory")
    ap.add_argument("--nc", default="results/exploratory_r1nocrop")
    ap.add_argument("--out", default="docs/exploratory_r1nocrop")
    o = ap.parse_args(); os.makedirs(o.out, exist_ok=True)
    units = ("F1a", "F1b")
    d = pd.concat([pd.read_csv(f"{o.res}/{u}_long.csv") for u in units]
                  + [pd.read_csv(f"{o.nc}/{u}_long.csv") for u in units])
    d = d[d.recipe.isin(["R1", "R1nc", "R4"])]
    runs = d.drop_duplicates(["dataset", "architecture", "recipe", "fold", "seed"])
    n_runs = runs[runs.recipe == "R1nc"].groupby("architecture").size().to_dict()
    avg = d.groupby(["subject", "dataset", "architecture", "recipe"]).bacc_f32.mean().rename("y").reset_index()
    fold = d.drop_duplicates("subject").set_index("subject").fold
    y = avg.pivot_table(index="subject", columns=["architecture", "recipe"], values="y")
    cl = (d.drop_duplicates("subject").set_index("subject").dataset.astype(str) + ":" + fold.astype(str)).reindex(y.index).to_numpy()

    def est(x):
        x = np.asarray(x, float); ok = ~np.isnan(x)
        ci = stats.t.interval(.95, ok.sum() - 1, x[ok].mean(), stats.sem(x[ok]))
        m, lo, hi, _ = cluster_mean_ci(x[ok], cl[ok], .95)
        return dict(mean=float(x[ok].mean()), lo=float(ci[0]), hi=float(ci[1]), lo_cluster=lo, hi_cluster=hi, n=int(ok.sum()))

    V = dict(n_runs_R1nc=n_runs, cell_means={}, contrasts={}, gap_EEGNet_minus_v2={}, did_EEGNet_v2={})
    for a in ARCHS:
        V["cell_means"][a] = {r: float(y[(a, r)].mean()) for r in ("R1", "R1nc", "R4")}
        V["contrasts"][a] = {"R1nc-R1": est(y[(a, "R1nc")] - y[(a, "R1")]),
                             "R4-R1nc": est(y[(a, "R4")] - y[(a, "R1nc")])}
    for r in ("R1", "R1nc", "R4"):
        V["gap_EEGNet_minus_v2"][r] = est(y[("EEGNet", r)] - y[("CSOANet-v2", r)])
    g = {r: y[("EEGNet", r)] - y[("CSOANet-v2", r)] for r in ("R1", "R1nc", "R4")}
    V["did_EEGNet_v2"]["R1nc-R1"] = est(g["R1nc"] - g["R1"])
    V["did_EEGNet_v2"]["R4-R1nc"] = est(g["R4"] - g["R1nc"])
    V["by_dataset"] = {ds: {a: float((y[(a, "R1nc")] - y[(a, "R1")])[
        avg.drop_duplicates("subject").set_index("subject").dataset.reindex(y.index) == ds].mean()) for a in ARCHS}
        for ds in sorted(avg.dataset.unique())}
    diag = runs[runs.recipe.isin(["R1", "R1nc", "R4"])].groupby(["architecture", "recipe"])[
        ["train_acc", "val_acc", "best_epoch", "training_failure"]].mean().round(2)
    diag.to_csv(f"{o.out}/R1nc_run_diagnostics.csv")
    json.dump(V, open(f"{o.out}/R1nc_analysis.json", "w"), indent=2)
    print(json.dumps({"n_runs": n_runs, "cell_means": V["cell_means"], "did_EEGNet_v2": V["did_EEGNet_v2"]}, indent=1))


if __name__ == "__main__":
    main()
