#!/usr/bin/env python3
"""Revision V1.17 / JNE V1.1 (sixth review; post hoc, S4 row 19). Reads result files only.
1. Logit-scale re-analysis: per subject x cell, empirical logit of seed-averaged correct counts.
   H1/H2 (subject RI and crossed subject+run), H2b, the {R1nc,R2,R4} omnibus and 18 DiDs.
2. Wild cluster bootstrap (Webb 6-point weights, 9,999 draws, 10 clusters) for the key contrasts:
   restricted (WCR) p-values and unrestricted bootstrap-t (WCU) 95 % CIs.
3. Bonferroni count of the 18 DiDs (percentage scale) excluding zero.
Usage: python -m analysis.revision_v17 --res data --nc nc --out rev17
"""
from __future__ import annotations
import argparse, itertools, json, os
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from .common import ARCHS
from .revision_v11 import cluster_mean_ci

R44 = ["R1", "R2", "R3", "R4"]
R3S = ["R1nc", "R2", "R4"]
RNG = np.random.default_rng(20261125)
WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])


def load(o):
    f = pd.concat([pd.read_csv(f"{o.res}/{u}_long.csv") for u in ("F1a", "F1b")] +
                  [pd.read_csv(f"{o.nc}/{n}") for n in ("F1a_R1nc_long.csv", "F1b_R1nc_long.csv")])
    f["k"] = f.bacc_f32 / 100 * f.n_trials
    g = f.groupby(["subject", "dataset", "fold", "architecture", "recipe"]).agg(
        y=("bacc_f32", "mean"), k=("k", "mean"), n=("n_trials", "first")).reset_index()
    g["yl"] = np.log((g.k + .5) / (g.n - g.k + .5))
    g["cluster"] = g.dataset + ":" + g.fold.astype(str)
    g["run"] = g.cluster + ":" + g.architecture + ":" + g.recipe
    return g


def fit(formula, d, crossed):
    d = d.reset_index(drop=True).copy(); d["one"] = 1
    if crossed:
        return smf.mixedlm(formula, d, groups=d["one"], vc_formula={"s": "0 + C(subject)", "r": "0 + C(run)"},
                           re_formula="0").fit(reml=False, method=["lbfgs", "powell"], maxiter=3000)
    return smf.mixedlm(formula, d, groups=d["subject"]).fit(reml=False, method=["lbfgs", "powell"], maxiter=3000)


def lrt(a, b, df):
    c = 2 * (a.llf - b.llf); return dict(chi2=round(float(c), 2), df=df, p=float(stats.chi2.sf(max(c, 0), df)))


def omnibus(g, recipes, yvar, crossed):
    d = g[g.recipe.isin(recipes)].copy(); d["yv"] = d[yvar]
    full = fit("yv ~ C(dataset) + C(architecture, Sum) * C(recipe, Sum)", d, crossed)
    noi = fit("yv ~ C(dataset) + C(architecture, Sum) + C(recipe, Sum)", d, crossed)
    nor = fit("yv ~ C(dataset) + C(architecture, Sum)", d, crossed)
    k = len(recipes)
    return dict(H1=lrt(noi, nor, k - 1), H2=lrt(full, noi, 3 * (k - 1)))


def wide(g, yvar):
    y = g.pivot_table(index="subject", columns=["architecture", "recipe"], values=yvar)
    cl = g.drop_duplicates("subject").set_index("subject").cluster.reindex(y.index).to_numpy()
    return y, cl


def did_vec(y, a, b, r, s):  # (a-b) under s minus (a-b) under r
    return ((y[(a, s)] - y[(b, s)]) - (y[(a, r)] - y[(b, r)])).to_numpy()


def cr1_t(x, cl, mu0=0.0):
    """Mean with CR1 cluster-robust SE; t statistic against mu0."""
    m = x.mean(); u = x - m; G = len(np.unique(cl)); n = len(x)
    s = np.array([u[cl == c].sum() for c in np.unique(cl)])
    se = np.sqrt(G / (G - 1) * (s ** 2).sum()) / n
    return m, se, (m - mu0) / se


def wild(x, cl, B=9999):
    x = np.asarray(x, float); ok = ~np.isnan(x); x, cl = x[ok], cl[ok]
    cls = np.unique(cl); idx = np.searchsorted(cls, cl); G = len(cls)
    m, se, t = cr1_t(x, cl)
    # WCR: impose null mean 0
    tb = np.empty(B)
    for b in range(B):  # under H0 (mean 0) the restricted residuals are x itself
        w = RNG.choice(WEBB, G)[idx]
        tb[b] = cr1_t(w * x, cl)[2]
    p = float((np.abs(tb) >= abs(t)).mean())
    # WCU bootstrap-t CI
    u = x - m; tu = np.empty(B)
    for b in range(B):
        w = RNG.choice(WEBB, G)[idx]
        xb = m + w * u
        mb, seb, _ = cr1_t(xb, cl); tu[b] = (mb - m) / seb
    lo, hi = m - np.quantile(tu, .975) * se, m - np.quantile(tu, .025) * se
    return dict(mean=round(float(m), 2), lo=round(float(lo), 2), hi=round(float(hi), 2), p_wcr=round(p, 4))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="results/confirmatory")
    ap.add_argument("--nc", default="docs/exploratory_r1nocrop"); ap.add_argument("--out", required=True)
    o = ap.parse_args(); os.makedirs(o.out, exist_ok=True)
    g = load(o); V = {}
    # 1 logit scale
    V["logit"] = dict(
        registered_subject=omnibus(g, R44, "yl", False), registered_crossed=omnibus(g, R44, "yl", True),
        no_crop_subject=omnibus(g, R3S, "yl", False), no_crop_crossed=omnibus(g, R3S, "yl", True),
        no_crop_pct_crossed=omnibus(g, R3S, "y", True))
    yl, cl = wide(g, "yl"); yp, _ = wide(g, "y")
    h2b = lambda y, r: did_vec(y, "EEGNet", "CSOANet-v2", r, "R2") if r != "R2" else None
    V["logit"]["H2b"] = {r: dict(zip(["mean", "lo", "hi", "icc"], np.round(cluster_mean_ci(did_vec(yl, "EEGNet", "CSOANet-v2", "R1", r), cl, .95), 3)))
                         for r in ("R2", "R4", "R1nc")}
    # 18 DiDs on both scales; Bonferroni counts
    k = 18; lv = 1 - .05 / k; rows = []
    for a, b in itertools.combinations(ARCHS, 2):
        for r, s in itertools.combinations(R3S, 2):
            for nm, y in (("pct", yp), ("logit", yl)):
                x = did_vec(y, a, b, r, s)
                m, lo, hi, _ = cluster_mean_ci(x, cl, .95); _, lb, hb, _ = cluster_mean_ci(x, cl, lv)
                rows.append(dict(scale=nm, pair=f"{a} - {b}", contrast=f"{s} vs {r}", did=m, lo95=lo, hi95=hi,
                                 loB=lb, hiB=hb, ex95=bool(lo > 0 or hi < 0), exB=bool(lb > 0 or hb < 0)))
    T = pd.DataFrame(rows); T.round(3).to_csv(f"{o.out}/F1_did18_pct_logit.csv", index=False)
    V["did18"] = {s: dict(ex95=int(T[T.scale == s].ex95.sum()), exBonf=int(T[T.scale == s].exB.sum()),
                          exBonf_list=T[(T.scale == s) & T.exB][["pair", "contrast"]].values.tolist()) for s in ("pct", "logit")}
    # 2 wild cluster bootstrap, percentage scale
    W = {}
    W["H2b R2 vs R1"] = wild(did_vec(yp, "EEGNet", "CSOANet-v2", "R1", "R2"), cl)
    W["H2b R4 vs R1"] = wild(did_vec(yp, "EEGNet", "CSOANet-v2", "R1", "R4"), cl)
    W["H2b R2 vs R1nc"] = wild(did_vec(yp, "EEGNet", "CSOANet-v2", "R1nc", "R2"), cl)
    W["gap change R1nc vs R1"] = wild(did_vec(yp, "EEGNet", "CSOANet-v2", "R1", "R1nc"), cl)
    W["Shallow-CSOA R4 vs R2"] = wild(did_vec(yp, "ShallowConvNet", "CSOANet-v2", "R2", "R4"), cl)
    W["Shallow-MI R4 vs R2"] = wild(did_vec(yp, "ShallowConvNet", "MI-BMInet", "R2", "R4"), cl)
    W["EEG-MI R2 vs R1"] = wild(did_vec(yp, "EEGNet", "MI-BMInet", "R1", "R2"), cl)
    for a in ARCHS:
        W[f"R1nc-R1 {a}"] = wild((yp[(a, "R1nc")] - yp[(a, "R1")]).to_numpy(), cl)
    V["wild_cluster_bootstrap"] = W
    json.dump(V, open(f"{o.out}/revision_v17.json", "w"), indent=2, default=float)
    print(json.dumps(V, indent=1, default=float))


if __name__ == "__main__":
    main()
