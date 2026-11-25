#!/usr/bin/env python3
"""Revision V1.4 (post hoc, exploratory; S4 row 16). Reads result files only.
1. Additivity among {R1nc, R2, R4}: interaction residuals, recipe ranges, LRT for the interaction
   (subject random intercept, and crossed subject + training run).
2. R1nc - R1 on the fold-1 test subjects only (not used by the search-space regression that
   suggested the cropping hypothesis).
3. F2 cell means (for S3).
Usage: python -m analysis.revision_v14 --res results/confirmatory --nc docs/exploratory_r1nocrop --out <dir>
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from .common import ARCHS
from .revision_v11 import cluster_mean_ci

REC = ["R1nc", "R2", "R4"]


def fit(f, d, **kw):
    return smf.mixedlm(f, d, **kw).fit(reml=False, method=["lbfgs", "powell"], maxiter=3000)


def lrt(a, b, df):
    c = 2 * (a.llf - b.llf); return dict(chi2=float(c), df=df, p=float(stats.chi2.sf(max(c, 0), df)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/confirmatory"); ap.add_argument("--nc", default="docs/exploratory_r1nocrop")
    ap.add_argument("--nc-names", nargs=2, default=["F1a_R1nc_long.csv", "F1b_R1nc_long.csv"]); ap.add_argument("--out", required=True)
    o = ap.parse_args(); os.makedirs(o.out, exist_ok=True)
    d = pd.concat([pd.read_csv(f"{o.res}/{u}_long.csv") for u in ("F1a", "F1b")] + [pd.read_csv(f"{o.nc}/{n}") for n in o.nc_names])
    avg = d.groupby(["subject", "dataset", "fold", "architecture", "recipe"]).bacc_f32.mean().rename("y").reset_index()
    avg["cluster"] = avg.dataset + ":" + avg.fold.astype(str)
    avg["run"] = avg.cluster + ":" + avg.architecture + ":" + avg.recipe
    V = {}
    # 1 additivity
    a3 = avg[avg.recipe.isin(REC)].copy().reset_index(drop=True)
    a3["architecture"] = pd.Categorical(a3.architecture, ARCHS); a3["recipe"] = pd.Categorical(a3.recipe, REC)
    cell = a3.groupby(["architecture", "recipe"], observed=True).y.mean().unstack().loc[ARCHS, REC]
    t = cell.to_numpy(); res = t - t.mean(1, keepdims=True) - t.mean(0, keepdims=True) + t.mean()
    V["additivity"] = dict(cell_means=cell.round(2).to_dict(), residuals=pd.DataFrame(res, index=ARCHS, columns=REC).round(2).to_dict(),
                           max_abs_residual=float(np.abs(res).max()),
                           recipe_range={a: float(t[i].max() - t[i].min()) for i, a in enumerate(ARCHS)})
    full = fit("y ~ C(dataset) + C(architecture, Sum) * C(recipe, Sum)", a3, groups=a3["subject"])
    add = fit("y ~ C(dataset) + C(architecture, Sum) + C(recipe, Sum)", a3, groups=a3["subject"])
    V["additivity"]["H2_like_subject"] = lrt(full, add, 6)
    a3["one"] = 1; vc = {"subject": "0 + C(subject)", "run": "0 + C(run)"}
    fullc = fit("y ~ C(dataset) + C(architecture, Sum) * C(recipe, Sum)", a3, groups=a3["one"], vc_formula=vc, re_formula="0")
    addc = fit("y ~ C(dataset) + C(architecture, Sum) + C(recipe, Sum)", a3, groups=a3["one"], vc_formula=vc, re_formula="0")
    V["additivity"]["H2_like_crossed"] = lrt(fullc, addc, 6)
    # per-subject double-centred residual of CSOANet-v2 and EEGNet in their R1nc / R2 cells
    y = a3.pivot_table(index="subject", columns=["architecture", "recipe"], values="y", observed=False)
    Y = np.array([[y[(ar, r)].to_numpy() for r in REC] for ar in ARCHS]).transpose(2, 0, 1)
    R = Y - Y.mean(2, keepdims=True) - Y.mean(1, keepdims=True) + Y.mean((1, 2), keepdims=True)
    cl = a3.drop_duplicates("subject").set_index("subject").cluster.reindex(y.index).to_numpy()
    for ar, r in (("CSOANet-v2", "R1nc"), ("EEGNet", "R2")):
        m, lo, hi, _ = cluster_mean_ci(R[:, ARCHS.index(ar), REC.index(r)], cl, .95)
        V["additivity"][f"residual {ar}@{r}"] = dict(mean=m, lo_cluster=lo, hi_cluster=hi)
    # H2b analogue with R1nc in place of R1
    g = {r: y[("EEGNet", r)] - y[("CSOANet-v2", r)] for r in REC}
    x = (g["R2"] - g["R1nc"]).to_numpy(); ci = stats.t.interval(.95, len(x) - 1, x.mean(), stats.sem(x))
    m, lo, hi, _ = cluster_mean_ci(x, cl, .95)
    V["H2b_with_R1nc"] = dict(mean=float(x.mean()), lo=float(ci[0]), hi=float(ci[1]), lo_cluster=lo, hi_cluster=hi)
    # 2 fold-1 test subjects only
    yy = avg.pivot_table(index="subject", columns=["architecture", "recipe"], values="y")
    f1 = avg[avg.fold == 0].subject.unique()
    out = {}
    for ar in ARCHS:
        x = (yy[(ar, "R1nc")] - yy[(ar, "R1")]).loc[f1].dropna()
        ci = stats.t.interval(.95, len(x) - 1, x.mean(), stats.sem(x))
        out[ar] = dict(mean=float(x.mean()), lo=float(ci[0]), hi=float(ci[1]), n=int(len(x)))
    gg = lambda r: (yy[("EEGNet", r)] - yy[("CSOANet-v2", r)]).loc[f1]
    x = (gg("R1nc") - gg("R1")).dropna(); ci = stats.t.interval(.95, len(x) - 1, x.mean(), stats.sem(x))
    out["DiD EEGNet-CSOANet-v2"] = dict(mean=float(x.mean()), lo=float(ci[0]), hi=float(ci[1]), n=int(len(x)))
    V["fold1_only"] = out
    # 3 R3 DiD (registered cells)
    x = ((yy[("EEGNet", "R3")] - yy[("CSOANet-v2", "R3")]) - (yy[("EEGNet", "R1")] - yy[("CSOANet-v2", "R1")])).dropna()
    V["DiD_EEGNet_v2_R3_R1"] = float(x.mean())
    json.dump(V, open(f"{o.out}/revision_v14.json", "w"), indent=2, default=float)
    print(json.dumps(V, indent=1, default=float))


if __name__ == "__main__":
    main()
