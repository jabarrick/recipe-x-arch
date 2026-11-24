#!/usr/bin/env python3
"""Analyses added in revision V1.1 in response to an internal review (post hoc; S4 row 14).
None changes a registered decision. Reads the result files only.

1. Training-run clustering. In the cross-subject families every test subject of a fold is
   scored by the same trained networks, so subject-level errors are correlated within a
   dataset x fold cluster (10 clusters in F1). We re-estimate
   (a) H1/H2 with crossed random effects for subject and training run
       (dataset:fold:architecture:recipe, seed-averaged), and
   (b) H2b and the 24 H3 intervals with a random intercept per dataset x fold cluster and
       t on G - 1 = 9 degrees of freedom, Bonferroni-adjusted as registered.
2. Exploratory: H1/H2 and variance shares without R3.
3. Difference-in-differences for EEGNet - MI-BMInet (R1 against each other recipe).
4. Condition T against the best fixed recipe (R4), per architecture.
5. Per-cell training diagnostics (selected epoch, training accuracy).
6. Condition-T search: which configuration dimensions predict validation accuracy.

Usage: python -m analysis.revision_v11 --res results/confirmatory --out <dir>
"""
from __future__ import annotations
import argparse, itertools, json, os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from .common import ARCHS, RECIPES, PAIRS, ALPHA, SESOI, load_long, seed_average, to_cube
from .h3_classify import classify_interval

RNG = np.random.default_rng(20261124)


def _fit(formula, d, groups, **kw):
    return smf.mixedlm(formula, d, groups=groups, **kw).fit(reml=False, method=["lbfgs", "powell"], maxiter=3000)


def _lrt(f, r, df):
    c = 2 * (f.llf - r.llf)
    return dict(chi2=float(c), df=df, p=float(stats.chi2.sf(max(c, 0), df)))


def avg_with_cluster(df):
    a = seed_average(df)
    fold = df.drop_duplicates("subject").set_index("subject")["fold"]
    a["cluster"] = a["dataset"].astype(str) + ":" + a["subject"].map(fold).astype(str)
    a["run"] = a["cluster"] + ":" + a["architecture"].astype(str) + ":" + a["recipe"].astype(str)
    a["architecture"] = pd.Categorical(a["architecture"].astype(str), ARCHS)
    a["recipe"] = pd.Categorical(a["recipe"].astype(str), RECIPES)
    return a.reset_index(drop=True)


# ------------------------------------------------------------------ 1a crossed subject + run
def crossed_h1_h2(a, recipes=RECIPES):
    d = a[a.recipe.astype(str).isin(recipes)].copy()
    d["recipe"] = pd.Categorical(d["recipe"].astype(str), recipes)
    d["one"] = 1
    vc = {"subject": "0 + C(subject)", "run": "0 + C(run)"}
    base = "y ~ C(dataset) + " if d.dataset.nunique() > 1 else "y ~ "
    full = _fit(base + "C(architecture, Sum) * C(recipe, Sum)", d, d["one"], vc_formula=vc, re_formula="0")
    noi = _fit(base + "C(architecture, Sum) + C(recipe, Sum)", d, d["one"], vc_formula=vc, re_formula="0")
    nor = _fit(base + "C(architecture, Sum)", d, d["one"], vc_formula=vc, re_formula="0")
    k = len(recipes)
    return dict(H1=_lrt(noi, nor, k - 1), H2=_lrt(full, noi, (len(ARCHS) - 1) * (k - 1)),
                var_subject=float(full.vcomp[list(full.model.exog_vc.names).index("subject")]),
                var_run=float(full.vcomp[list(full.model.exog_vc.names).index("run")]),
                var_resid=float(full.scale), converged=bool(full.converged and noi.converged and nor.converged))


# ------------------------------------------------------------------ 1b cluster-robust intervals
def cluster_mean_ci(x, cl, level):
    """Mean of x with a random intercept per cluster; t on G-1 df."""
    d = pd.DataFrame(dict(x=x, cl=cl)).dropna()
    G = d.cl.nunique()
    try:
        m = smf.mixedlm("x ~ 1", d, groups=d["cl"]).fit(reml=True)
        est, se = float(m.fe_params["Intercept"]), float(m.bse_fe["Intercept"])
        icc = float(m.cov_re.iloc[0, 0] / (m.cov_re.iloc[0, 0] + m.scale))
    except Exception:  # fall back to cluster means
        cm = d.groupby("cl").x.mean(); est, se, icc = cm.mean(), cm.std(ddof=1) / np.sqrt(G), np.nan
    tc = stats.t.ppf(1 - (1 - level) / 2, G - 1)
    return est, est - tc * se, est + tc * se, icc


def h3_clustered(a):
    y = a.pivot_table(index="subject", columns=["architecture", "recipe"], values="y", observed=False)
    cl = a.drop_duplicates("subject").set_index("subject")["cluster"].reindex(y.index)
    level = 1 - ALPHA / (len(PAIRS) * len(RECIPES))
    rows = []
    for p, q in PAIRS:
        for r in RECIPES:
            x = (y[(p, r)] - y[(q, r)]).to_numpy()
            m, lo, hi, icc = cluster_mean_ci(x, cl.to_numpy(), level)
            rows.append(dict(pair=f"{p} - {q}", recipe=r, mean=m, lo=lo, hi=hi, half=(hi - lo) / 2, icc=icc,
                             category=classify_interval(lo, hi)))
    return pd.DataFrame(rows)


def h2b_clustered(a):
    y = a.pivot_table(index="subject", columns=["architecture", "recipe"], values="y", observed=False)
    cl = a.drop_duplicates("subject").set_index("subject")["cluster"].reindex(y.index)
    c = (y[("EEGNet", "R2")] - y[("EEGNet", "R1")]) - (y[("CSOANet-v2", "R2")] - y[("CSOANet-v2", "R1")])
    m, lo, hi, icc = cluster_mean_ci(c.to_numpy(), cl.to_numpy(), 0.95)
    G = cl.nunique()
    t = m / ((hi - lo) / 2 / stats.t.ppf(.975, G - 1))
    return dict(mean=m, lo=lo, hi=hi, icc=icc, p=float(2 * stats.t.sf(abs(t), G - 1)), G=int(G))


def home_clustered(a):
    """Per-subject home residuals (cell-mean formula on each subject's 4 x 4 table), cluster-robust CI."""
    from .common import HOME
    y = a.pivot_table(index="subject", columns=["architecture", "recipe"], values="y", observed=False)
    cl = a.drop_duplicates("subject").set_index("subject")["cluster"].reindex(y.index)
    Y = np.array([[y[(ar, r)].to_numpy() for r in RECIPES] for ar in ARCHS]).transpose(2, 0, 1)   # S, A, R
    res = Y - Y.mean(2, keepdims=True) - Y.mean(1, keepdims=True) + Y.mean((1, 2), keepdims=True)
    out = {}
    for ar, r in HOME.items():
        m, lo, hi, icc = cluster_mean_ci(res[:, ARCHS.index(ar), RECIPES.index(r)], cl.to_numpy(), .95)
        out[ar] = dict(h=m, lo=lo, hi=hi, icc=icc)
    m, lo, hi, _ = cluster_mean_ci(res[:, ARCHS.index("EEGNet"), RECIPES.index("R3")], cl.to_numpy(), .95)
    out["EEGNet@R3 (not home)"] = dict(h=m, lo=lo, hi=hi)
    return out


def compare_categories(orig, new):
    o = orig.set_index(["pair", "recipe"]).category
    n = new.set_index(["pair", "recipe"]).category
    j = pd.DataFrame(dict(orig=o, new=n))
    trans = j.groupby(["orig", "new"]).size().to_dict()
    defin = {"A", "B", "E"}
    nd = int(n.isin(defin).sum())
    # changes across recipes within pairs under the clustered intervals
    sign = eb = inc = 0
    for pair, row in new.pivot(index="pair", columns="recipe", values="category").iterrows():
        for r1, r2 in itertools.combinations(RECIPES, 2):
            c1, c2 = row[r1], row[r2]
            if c1 == c2:
                continue
            if c1 in defin and c2 in defin:
                if {c1, c2} == {"A", "B"}: sign += 1
                else: eb += 1
            else:
                inc += 1
    return dict(transitions={f"{k[0]}->{k[1]}": v for k, v in trans.items()}, n_definitive=nd,
                kept=int((j.orig == j.new).sum()), sign_reversals=sign, equiv_better=eb, inc_def=inc,
                H3_supported=bool(sign + eb > 0))


# ------------------------------------------------------------------ 2 without R3
def no_r3(a):
    d = a[a.recipe.astype(str) != "R3"].copy()
    d["recipe"] = pd.Categorical(d["recipe"].astype(str), ["R1", "R2", "R4"])
    base = "y ~ C(dataset) + "
    full = _fit(base + "C(architecture, Sum) * C(recipe, Sum)", d, d["subject"])
    noi = _fit(base + "C(architecture, Sum) + C(recipe, Sum)", d, d["subject"])
    nor = _fit(base + "C(architecture, Sum)", d, d["subject"])
    cell = d.groupby(["architecture", "recipe"], observed=True).y.mean().unstack().loc[ARCHS, ["R1", "R2", "R4"]].to_numpy()
    g = cell.mean(); ae = cell.mean(1) - g; re_ = cell.mean(0) - g
    inter = cell - cell.mean(1, keepdims=True) - cell.mean(0, keepdims=True) + g
    ssA, ssR, ssAR = (ae ** 2).sum() * 3, (re_ ** 2).sum() * 4, (inter ** 2).sum()
    tot = ssA + ssR + ssAR
    return dict(H1=_lrt(noi, nor, 2), H2=_lrt(full, noi, 6),
                fixed_shares=dict(A=ssA / tot, R=ssR / tot, AR=ssAR / tot),
                recipe_range={a_: float(cell[i].max() - cell[i].min()) for i, a_ in enumerate(ARCHS)},
                crossed=crossed_h1_h2(a, ["R1", "R2", "R4"]))


# ------------------------------------------------------------------ 3 DiD EEGNet - MI-BMInet
def did(a):
    y = a.pivot_table(index="subject", columns=["architecture", "recipe"], values="y", observed=False)
    cl = a.drop_duplicates("subject").set_index("subject")["cluster"].reindex(y.index)
    out = {}
    diff = {r: y[("EEGNet", r)] - y[("MI-BMInet", r)] for r in RECIPES}
    for r in ("R2", "R3", "R4"):
        x = (diff[r] - diff["R1"]).to_numpy()
        ci = stats.t.interval(.95, len(x) - 1, x.mean(), stats.sem(x))
        m, lo, hi, _ = cluster_mean_ci(x, cl.to_numpy(), .95)
        out[f"{r}-R1"] = dict(mean=float(x.mean()), lo_subject=float(ci[0]), hi_subject=float(ci[1]),
                              p_subject=float(stats.ttest_1samp(x, 0).pvalue), lo_cluster=lo, hi_cluster=hi)
    return out


# ------------------------------------------------------------------ 4 T against R4
def t_vs(a, res, units, ref="R4"):
    T = pd.concat([pd.read_csv(f"{res}/{u}_T_long.csv") for u in units])
    Tp = pd.concat([pd.read_csv(f"{res}/{u}_Tprime_long.csv") for u in units])
    y = a.pivot_table(index="subject", columns=["architecture", "recipe"], values="y", observed=False)
    cl = a.drop_duplicates("subject").set_index("subject")["cluster"]
    out = {}
    for name, D in (("T", T), ("Tprime", Tp)):
        t = seed_average(D).pivot_table(index="subject", columns="architecture", values="y", observed=False)
        for arch in ARCHS:
            x = (t[arch] - y[(arch, ref)].reindex(t.index)).dropna()
            ci = stats.t.interval(.95, len(x) - 1, x.mean(), stats.sem(x))
            m, lo, hi, _ = cluster_mean_ci(x.to_numpy(), cl.reindex(x.index).to_numpy(), .95)
            out[f"{name}-{ref}/{arch}"] = dict(mean=float(x.mean()), lo=float(ci[0]), hi=float(ci[1]),
                                               lo_cluster=lo, hi_cluster=hi)
    return out


# ------------------------------------------------------------------ 5 diagnostics
def diagnostics(df):
    runs = df.drop_duplicates(["dataset", "architecture", "recipe", "fold", "seed"])
    g = runs.groupby(["architecture", "recipe"], observed=True)
    return pd.DataFrame(dict(best_epoch=g.best_epoch.median(), epochs_run=g.epochs.median(),
                             train_acc=g.train_acc.mean(), val_acc=g.val_acc.mean(),
                             test=df.groupby(["architecture", "recipe"], observed=True).bacc_f32.mean())).reset_index()


# ------------------------------------------------------------------ 6 T-search regression
def search_regression(res, units, cfgfile):
    cfg = pd.DataFrame(json.load(open(cfgfile))).rename(columns={"id": "config_id"})
    S = pd.concat([pd.read_csv(f"{res}/{u}_T_search.csv").assign(unit=u) for u in units])
    d = S.merge(cfg, on="config_id")
    d["loglr"] = np.log10(d["lr"])
    for c in ("noise", "shift", "crop", "chdrop", "mixup"):
        d[c] = d[c].astype(int)
    d["n_aug"] = d[["noise", "shift", "crop", "chdrop", "mixup"]].sum(1)
    d["ls"] = (d["label_smoothing"] > 0).astype(int)
    m = smf.ols("val_bacc ~ C(unit) * C(architecture) + loglr + C(optimizer) + C(schedule) + C(batch) + noise + shift"
                " + crop + chdrop + mixup + ls + C(select)", d).fit()
    keep = [k for k in m.params.index if not k.startswith(("C(unit)", "C(architecture)", "Intercept"))]
    return pd.DataFrame(dict(coef=m.params[keep], lo=m.conf_int().loc[keep, 0], hi=m.conf_int().loc[keep, 1],
                             p=m.pvalues[keep])).reset_index().rename(columns={"index": "term"}), float(m.rsquared), int(m.nobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/confirmatory")
    ap.add_argument("--cfg", default="docs/Tprime/tprime_configs.json")
    ap.add_argument("--ana", default="docs/confirmatory_analysis_V1.1")
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args(); os.makedirs(a_.out, exist_ok=True)
    df = pd.concat([load_long(f"{a_.res}/F1a_long.csv"), load_long(f"{a_.res}/F1b_long.csv")])
    a = avg_with_cluster(df)
    V = {}
    V["crossed"] = crossed_h1_h2(a)
    h3c = h3_clustered(a); h3c.to_csv(f"{a_.out}/F1_h3_intervals_clustered.csv", index=False)
    orig = pd.read_csv(f"{a_.ana}/F1_h3_intervals_f32.csv")
    V["h3_clustered"] = compare_categories(orig, h3c)
    V["h3_clustered"]["median_halfwidth"] = float(h3c.half.median())
    V["h3_clustered"]["median_icc"] = float(h3c.icc.median())
    V["h2b_clustered"] = h2b_clustered(a)
    V["home_clustered"] = home_clustered(a)
    V["no_R3"] = no_r3(a)
    V["did_EEGNet_MI"] = did(a)
    V["T_vs_R4"] = t_vs(a, a_.res, ("F1a", "F1b"))
    dg = diagnostics(df); dg.to_csv(f"{a_.out}/F1_cell_diagnostics.csv", index=False)
    V["diagnostics"] = dg.round(2).to_dict(orient="records")
    sr, r2, n = search_regression(a_.res, ("F1a", "F1b"), a_.cfg)
    sr.to_csv(f"{a_.out}/F1_T_search_regression.csv", index=False)
    V["search_regression"] = dict(r2=r2, n=n, terms=sr.round(3).to_dict(orient="records"))
    json.dump(V, open(f"{a_.out}/revision_v11.json", "w"), indent=2, default=float)
    print(json.dumps({k: V[k] for k in ("crossed", "h3_clustered", "h2b_clustered")}, indent=1, default=float))


if __name__ == "__main__":
    main()
