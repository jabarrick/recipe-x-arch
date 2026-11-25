#!/usr/bin/env python3
"""Revision V1.5 (post hoc, exploratory; S4 row 17). Reads result files only.
1. F2 descriptive ranking: 24 subject-level intervals (each F2 subject has its own trained
   networks, so subject = training-run cluster), ranks per recipe, and the same with flagged runs excluded.
2. Interaction residuals among {R1nc, R2, R4}: per-subject residuals with a random intercept per
   dataset x fold cluster, 95 % CI and TOST (90 % CI within +/-2).
3. Cho2017 half of the R1nc runs (run after the analysis script was written) as a replication.
4. Share of condition-T configurations with cropping on.
Usage: python -m analysis.revision_v15 --res results/confirmatory --nc docs/exploratory_r1nocrop --ana docs/confirmatory_analysis_V1.1 --cfg docs/Tprime/tprime_configs.json --out <dir>
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd
from scipy import stats
from .common import ARCHS, RECIPES, SESOI
from .revision_v11 import cluster_mean_ci

REC = ["R1nc", "R2", "R4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/confirmatory"); ap.add_argument("--nc", default="docs/exploratory_r1nocrop")
    ap.add_argument("--nc-names", nargs=2, default=["F1a_R1nc_long.csv", "F1b_R1nc_long.csv"])
    ap.add_argument("--ana", default="docs/confirmatory_analysis_V1.1"); ap.add_argument("--cfg", default="docs/Tprime/tprime_configs.json")
    ap.add_argument("--out", required=True)
    o = ap.parse_args(); os.makedirs(o.out, exist_ok=True)
    V = {}
    # 1 F2
    f2 = pd.read_csv(f"{o.res}/F2_long.csv")
    cm = f2.groupby(["architecture", "recipe"]).bacc_f32.mean().unstack().loc[ARCHS, RECIPES]
    cmx = f2[f2.training_failure == 0].groupby(["architecture", "recipe"]).bacc_f32.mean().unstack().loc[ARCHS, RECIPES]
    fl = f2.groupby(["architecture", "recipe"]).training_failure.sum().unstack().loc[ARCHS, RECIPES]
    iv = pd.read_csv(f"{o.ana}/F2_h3_intervals_f32.csv")
    V["F2"] = dict(cell_means=cm.round(2).to_dict(), cell_means_failures_excluded=cmx.round(2).to_dict(),
                   flagged_runs=fl.astype(int).to_dict(),
                   order={r: list(cm[r].sort_values(ascending=False).index) for r in RECIPES},
                   intervals=iv[["pair", "recipe", "mean", "lo", "hi", "category"]].round(2).to_dict(orient="records"))
    # F1 order for contrast
    f1 = pd.concat([pd.read_csv(f"{o.res}/{u}_long.csv") for u in ("F1a", "F1b")])
    c1 = f1.groupby(["architecture", "recipe"]).bacc_f32.mean().unstack().loc[ARCHS, RECIPES]
    V["F1_order"] = {r: list(c1[r].sort_values(ascending=False).index) for r in RECIPES}
    # 2 residual CIs + TOST
    d = pd.concat([f1] + [pd.read_csv(f"{o.nc}/{n}") for n in o.nc_names])
    d = d[d.recipe.isin(REC + ["R1"])]
    avg = d.groupby(["subject", "dataset", "fold", "architecture", "recipe"]).bacc_f32.mean().rename("y").reset_index()
    y = avg.pivot_table(index="subject", columns=["architecture", "recipe"], values="y")
    meta = avg.drop_duplicates("subject").set_index("subject").reindex(y.index)
    cl = (meta.dataset + ":" + meta.fold.astype(str)).to_numpy()
    Y = np.array([[y[(a, r)].to_numpy() for r in REC] for a in ARCHS]).transpose(2, 0, 1)
    R = Y - Y.mean(2, keepdims=True) - Y.mean(1, keepdims=True) + Y.mean((1, 2), keepdims=True)
    rows = []
    for i, a in enumerate(ARCHS):
        for j, r in enumerate(REC):
            m, lo, hi, _ = cluster_mean_ci(R[:, i, j], cl, .95)
            _, lo90, hi90, _ = cluster_mean_ci(R[:, i, j], cl, .90)
            rows.append(dict(architecture=a, recipe=r, residual=m, lo=lo, hi=hi, tost90_lo=lo90, tost90_hi=hi90,
                             equivalent=bool(lo90 > -SESOI and hi90 < SESOI)))
    rt = pd.DataFrame(rows); rt.to_csv(f"{o.out}/F1_residuals_R1nc_R2_R4.csv", index=False)
    V["residuals_tost"] = dict(n_equivalent=int(rt.equivalent.sum()), n=len(rt), table=rt.round(2).to_dict(orient="records"))
    # 3 Cho2017 replication
    cho = meta.dataset.to_numpy() == "Cho2017"
    rep = {}
    for a in ARCHS:
        x = (y[(a, "R1nc")] - y[(a, "R1")]).to_numpy()[cho]
        ci = stats.t.interval(.95, len(x) - 1, x.mean(), stats.sem(x))
        m, lo, hi, _ = cluster_mean_ci(x, cl[cho], .95)
        rep[a] = dict(mean=float(x.mean()), lo=float(ci[0]), hi=float(ci[1]), lo_cluster=lo, hi_cluster=hi, n=int(len(x)))
    g = lambda r: (y[("EEGNet", r)] - y[("CSOANet-v2", r)]).to_numpy()[cho]
    x = g("R1nc") - g("R1"); ci = stats.t.interval(.95, len(x) - 1, x.mean(), stats.sem(x))
    m, lo, hi, _ = cluster_mean_ci(x, cl[cho], .95)
    rep["DiD EEGNet-CSOANet-v2"] = dict(mean=float(x.mean()), lo=float(ci[0]), hi=float(ci[1]), lo_cluster=lo, hi_cluster=hi, n=int(len(x)))
    V["cho2017_replication"] = rep
    # 4 crop share in the search
    cfg = pd.DataFrame(json.load(open(o.cfg)))
    V["search_crop_share"] = dict(n_crop=int(cfg.crop.sum()), n=int(len(cfg)))
    json.dump(V, open(f"{o.out}/revision_v15.json", "w"), indent=2, default=float)
    print(json.dumps({k: V[k] for k in ("F1_order", "cho2017_replication", "search_crop_share")}, indent=1, default=float))
    print(json.dumps({k: V["F2"][k] for k in ("cell_means", "cell_means_failures_excluded", "flagged_runs", "order")}, indent=1))
    print(rt.round(2).to_string())
    print(pd.DataFrame(V["F2"]["intervals"]).to_string())


if __name__ == "__main__":
    main()
