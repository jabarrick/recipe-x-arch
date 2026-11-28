#!/usr/bin/env python3
"""Preregistered analysis for the independent test (OSF 8st47, Amendment 3, V1.1). Frozen before download.

Input: <res>/IND_long.csv from train/independent_dreyer.py.
Unit: per-subject seed-averaged float32 balanced accuracy. Cluster: test fold (10 clusters).

Primary hypothesis (one test, alpha = 0.05):
  HA  crop effect on the gap   d_A = (EEGNet - CSOANet-v2)[R1nc] - (EEGNet - CSOANet-v2)[R1]      predicted > 0
  Test: mean of d_A with a cluster-robust (CR1) t statistic; p from a wild cluster bootstrap with the null
  imposed (Webb six-point weights, 9,999 draws, seed 20261126), two-sided.
  Supported: p < 0.05 and mean > 0. Contradicted: p < 0.05 and mean < 0. Otherwise inconclusive; if the
  90 % bootstrap-t CI lies within (-2, 2) the effect is reported as smaller than the SESOI.

Secondary (pre-specified, reported, no decision), on the corrected design {R1nc, R2, R4} x
{EEGNet, ShallowConvNet-MN, MI-BMInet, CSOANet-v2}:
  S1  recipe main effect and recipe x architecture interaction: likelihood-ratio tests with crossed random
      effects for subject and training run (fold x architecture x recipe), ML, best converged fit over four
      optimisers; percentage and empirical-logit scales. Expectation from F1: neither significant.
  S2  pairwise categories as registered for H3 (Bonferroni over 6 pairs x 3 recipes, SESOI 2 points) and the
      number of definitive category changes. Expectation from F1: none.
  S3  R1nc - R1 per architecture; HA on the empirical-logit scale; number of flagged runs.
Usage: python -m analysis.independent_analysis --res results/independent --out docs/independent
"""
from __future__ import annotations
import argparse, json, os, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from . import h3_classify as H3

warnings.filterwarnings("ignore")
WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
B = 9999
ARCHS = ["EEGNet", "ShallowConvNet-MN", "MI-BMInet", "CSOANet-v2"]
CORR = ["R1nc", "R2", "R4"]


def cr1(x, cl):
    m = x.mean(); u = x - m; G = len(np.unique(cl)); n = len(x)
    s = np.array([u[cl == c].sum() for c in np.unique(cl)])
    se = np.sqrt(G / (G - 1) * (s ** 2).sum()) / n
    return m, se


def test(x, cl, rng):
    ok = ~np.isnan(x); x, cl = x[ok], cl[ok]
    idx = np.searchsorted(np.unique(cl), cl); G = idx.max() + 1
    m, se = cr1(x, cl); t = m / se
    tb = np.array([(lambda xb: cr1(xb, cl)[0] / cr1(xb, cl)[1])(rng.choice(WEBB, G)[idx] * x) for _ in range(B)])
    p = float((np.abs(tb) >= abs(t)).mean())
    u = x - m
    tu = np.array([(lambda xb: (cr1(xb, cl)[0] - m) / cr1(xb, cl)[1])(m + rng.choice(WEBB, G)[idx] * u) for _ in range(B)])
    ci = lambda a: (m - np.quantile(tu, 1 - a / 2) * se, m - np.quantile(tu, a / 2) * se)
    (l95, h95), (l90, h90) = ci(.05), ci(.10)
    return dict(mean=float(m), se=float(se), t=float(t), p=p, ci95=[float(l95), float(h95)], ci90=[float(l90), float(h90)],
                n_subjects=int(len(x)), n_clusters=int(G))


def fit_best(formula, d):
    d = d.reset_index(drop=True).copy(); d["one"] = 1; best = None
    for meth in (["bfgs"], ["nm", "lbfgs"], ["powell"], ["lbfgs"]):
        try:
            r = smf.mixedlm(formula, d, groups=d["one"], vc_formula={"s": "0 + C(subject)", "r": "0 + C(run)"},
                            re_formula="0").fit(reml=False, method=meth, maxiter=5000)
        except Exception:
            continue
        if r.converged and (best is None or r.llf > best.llf):
            best = r
    return best


def omnibus(d, yvar):
    d = d.copy(); d["yv"] = d[yvar]; k = d.recipe.nunique()
    full = fit_best("yv ~ C(architecture, Sum) * C(recipe, Sum)", d)
    noi = fit_best("yv ~ C(architecture, Sum) + C(recipe, Sum)", d)
    nor = fit_best("yv ~ C(architecture, Sum)", d)
    if None in (full, noi, nor):
        return dict(error="no converged fit")
    L = lambda a, b, df: dict(chi2=float(2 * (a.llf - b.llf)), df=df, p=float(stats.chi2.sf(max(2 * (a.llf - b.llf), 0), df)))
    return dict(recipe=L(noi, nor, k - 1), interaction=L(full, noi, 3 * (k - 1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--unit", default="IND"); o = ap.parse_args(); os.makedirs(o.out, exist_ok=True)
    rng = np.random.default_rng(20261126)
    d = pd.read_csv(f"{o.res}/{o.unit}_long.csv")
    d["k"] = d.bacc_f32 / 100 * d.n_trials
    a = d.groupby(["subject", "fold", "architecture", "recipe"]).agg(y=("bacc_f32", "mean"), k=("k", "mean"),
                                                                     n=("n_trials", "first")).reset_index()
    a["yl"] = np.log((a.k + .5) / (a.n - a.k + .5))
    V = {"flagged_runs": int(d.groupby(["architecture", "recipe", "fold", "seed"]).training_failure.max().sum())}
    for scale in ("y", "yl"):
        w = a.pivot_table(index="subject", columns=["architecture", "recipe"], values=scale)
        cl = a.drop_duplicates("subject").set_index("subject").fold.reindex(w.index).to_numpy()
        gap = lambda p, q, r: w[(p, r)] - w[(q, r)]
        C = {"HA": (gap("EEGNet", "CSOANet-v2", "R1nc") - gap("EEGNet", "CSOANet-v2", "R1")).to_numpy()}
        for arch in ARCHS:
            C[f"R1nc-R1 {arch}"] = (w[(arch, "R1nc")] - w[(arch, "R1")]).to_numpy()
        V[scale] = {k: test(v, cl, rng) for k, v in C.items()}
    r = V["y"]["HA"]
    if r["p"] < .05 and r["mean"] > 0: dec = "supported"
    elif r["p"] < .05 and r["mean"] < 0: dec = "contradicted"
    elif -2 < r["ci90"][0] and r["ci90"][1] < 2: dec = "inconclusive; effect smaller than the 2-point SESOI"
    else: dec = "inconclusive"
    V["decision"] = {"HA": dec}
    # secondary: corrected design {R1nc, R2, R4}
    c = a[a.recipe.isin(CORR) & a.architecture.isin(ARCHS)].copy()
    c["run"] = c.fold.astype(str) + ":" + c.architecture + ":" + c.recipe
    V["S1_omnibus"] = {"percentage": omnibus(c, "y"), "logit": omnibus(c, "yl")}
    w = c.pivot_table(index="subject", columns=["architecture", "recipe"], values="y")
    # positional: H3 labels index 1 as "ShallowConvNet"; here it is ShallowConvNet-MN
    cube = np.array([[w[(ar, r)].to_numpy() for r in CORR] for ar in ARCHS]).transpose(2, 0, 1)
    h3 = H3.run(cube, recipes=CORR)
    V["S2_categories"] = {k: h3[k] for k in ("outcome", "n_definitive", "n_intervals", "sign_reversals",
                                             "equiv_better_changes", "inconclusive_definitive_changes")}
    V["S2_categories"]["matrix"] = h3["category_matrix"].to_dict()
    json.dump(V, open(f"{o.out}/independent_results.json", "w"), indent=2, default=str)
    print(json.dumps({"HA": V["decision"]["HA"], "S1": V["S1_omnibus"], "S2": V["S2_categories"]["outcome"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
