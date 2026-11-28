#!/usr/bin/env python3
"""Corrected design (seventh review; post hoc, S4 row 23). Reads result files only.
Corrected 4x4 on F1: R1nc in place of R1 (crop removed) and ShallowConvNet-MN in place of ShallowConvNet
(reference max-norm constraints). Registered analyses re-run on it and, for comparison, on the registered 4x4:
H1/H2 (crossed subject + training-run LRT, best converged fit over optimisers; percentage and logit),
H2b, H3 (24 Bonferroni intervals, registered rule), home residuals. Wild cluster bootstrap with a fixed
per-contrast seed, so every number is reproducible regardless of call order.
Also every exploratory contrast reported in V1.15 (crop variants, R2va, ShallowConvNet-MN, the CSOANet-v2 x R3
diagnosis), with Holm-adjusted p over the whole exploratory set.
Usage: python -m analysis.revision_v19 --res data --nc nc --rev6 rev6 --out rev19
"""
from __future__ import annotations
import argparse, json, os, warnings, zlib
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from . import h3_classify as H3
from .home_recipe import interaction_residuals
from .revision_v17 import cr1_t, WEBB

warnings.filterwarnings("ignore")
A = ["EEGNet", "ShallowConvNet", "MI-BMInet", "CSOANet-v2"]
HOME = {"EEGNet": "R2", "MI-BMInet": "R3", "CSOANet-v2": "R1"}
BASE_SEED = 20261128
DESIGNS = {
    "registered": dict(recipes=["R1", "R2", "R3", "R4"], shallow="ShallowConvNet"),
    "corrected": dict(recipes=["R1nc", "R2", "R3", "R4"], shallow="ShallowConvNet-MN"),
    "corrected_noR3": dict(recipes=["R1nc", "R2", "R4"], shallow="ShallowConvNet-MN"),
}


def wild(x, cl, label, B=9999):
    """Wild cluster bootstrap (Webb). WCR p and WCU bootstrap-t CI from the same weight draws; seed fixed by the data."""
    x = np.asarray(x, float); ok = ~np.isnan(x); x, cl = x[ok], cl[ok]
    # seed from the data, not the label: an identical contrast gives an identical interval wherever it is reported
    rng = np.random.default_rng(BASE_SEED + zlib.crc32(np.round(x, 6).tobytes()) % 100000)
    cls = np.unique(cl); idx = np.searchsorted(cls, cl); G = len(cls)
    m, se, t = cr1_t(x, cl); u = x - m
    W = rng.choice(WEBB, (B, G))[:, idx]
    tr = np.array([cr1_t(w * x, cl)[2] for w in W])
    tu = np.array([(lambda r: (r[0] - m) / r[1])(cr1_t(m + w * u, cl)) for w in W])
    p = float((np.abs(tr) >= abs(t)).mean())
    lo, hi = m - np.quantile(tu, .975) * se, m - np.quantile(tu, .025) * se
    return dict(mean=round(float(m), 2), lo=round(float(lo), 2), hi=round(float(hi), 2), p=round(p, 4))


def load(o):
    f = pd.concat([pd.read_csv(f"{o.res}/{u}_long.csv") for u in ("F1a", "F1b")] +
                  [pd.read_csv(f"{o.nc}/{u}_R1nc_long.csv") for u in ("F1a", "F1b")] +
                  [pd.read_csv(f"{o.rev6}/{u}_long.csv") for u in ("F1a", "F1b")])
    f["k"] = f.bacc_f32 / 100 * f.n_trials
    g = f.groupby(["subject", "dataset", "fold", "architecture", "recipe"]).agg(
        y=("bacc_f32", "mean"), k=("k", "mean"), n=("n_trials", "first")).reset_index()
    g["yl"] = np.log((g.k + .5) / (g.n - g.k + .5))
    g["cluster"] = g.dataset + ":" + g.fold.astype(str)
    return g


def design(g, recipes, shallow):
    d = g[g.recipe.isin(recipes) & g.architecture.isin([a if a != "ShallowConvNet" else shallow for a in A])].copy()
    d.loc[d.architecture == shallow, "architecture"] = "ShallowConvNet"
    d["recipe"] = d.recipe.replace({"R1nc": "R1"})
    d["run"] = d.cluster + ":" + d.architecture + ":" + d.recipe
    return d


def fit_best(formula, d):
    d = d.reset_index(drop=True).copy(); d["one"] = 1
    best = None
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
    full = fit_best("yv ~ C(dataset) + C(architecture, Sum) * C(recipe, Sum)", d)
    noi = fit_best("yv ~ C(dataset) + C(architecture, Sum) + C(recipe, Sum)", d)
    nor = fit_best("yv ~ C(dataset) + C(architecture, Sum)", d)
    L = lambda a, b, df: dict(chi2=round(float(2 * (a.llf - b.llf)), 2), df=df,
                              p=round(float(stats.chi2.sf(max(2 * (a.llf - b.llf), 0), df)), 4))
    return dict(H1=L(noi, nor, k - 1), H2=L(full, noi, 3 * (k - 1)))


def exploratory(g):
    y = g.pivot_table(index="subject", columns=["architecture", "recipe"], values="y")
    yl = g.pivot_table(index="subject", columns=["architecture", "recipe"], values="yl")
    cl = g.drop_duplicates("subject").set_index("subject").cluster.reindex(y.index).to_numpy()
    E, S, C = {}, "ShallowConvNet-MN", "CSOANet-v2"
    gap = lambda Y, a, b, r: Y[(a, r)] - Y[(b, r)]
    def add(k, x): E[k] = wild(np.asarray(x, float), cl, "exp:" + k)
    for a in A:
        add(f"R1nc-R1 | {a}", y[(a, "R1nc")] - y[(a, "R1")])
        add(f"R1ct-R1 | {a}", y[(a, "R1ct")] - y[(a, "R1")])
        add(f"R1ns-R1 | {a}", y[(a, "R1ns")] - y[(a, "R1")])
        add(f"R1nc-R1ct | {a}", y[(a, "R1nc")] - y[(a, "R1ct")])
        add(f"R1nc-R1ns | {a}", y[(a, "R1nc")] - y[(a, "R1ns")])
        add(f"R2va-R2 | {a}", y[(a, "R2va")] - y[(a, "R2")])
    add("R2va-R2 | mean of four", sum(y[(a, "R2va")] - y[(a, "R2")] for a in A) / 4)
    for r, s in (("R1", "R1nc"), ("R1", "R1ct"), ("R1ct", "R1nc"), ("R1", "R1ns"), ("R1", "R2"), ("R1", "R4"), ("R1nc", "R2"), ("R1nc", "R2va")):
        add(f"EEGNet-{C} gap: {s} minus {r}", gap(y, "EEGNet", C, s) - gap(y, "EEGNet", C, r))
    add(f"EEGNet-{C} gap: R1nc minus R1 (logit)", gap(yl, "EEGNet", C, "R1nc") - gap(yl, "EEGNet", C, "R1"))
    add("MI-BMInet-EEGNet gap: R2 minus R1", gap(y, "MI-BMInet", "EEGNet", "R2") - gap(y, "MI-BMInet", "EEGNet", "R1"))
    for b in (C, "MI-BMInet"):
        for sh in ("ShallowConvNet", S):
            add(f"{sh}-{b} gap: R4 minus R2", gap(y, sh, b, "R4") - gap(y, sh, b, "R2"))
            add(f"{sh}-{b} gap: R4 minus R3", gap(y, sh, b, "R4") - gap(y, sh, b, "R3"))
        add(f"ShallowConvNet-{b} gap: R4 minus R2va", gap(y, "ShallowConvNet", b, "R4") - gap(y, "ShallowConvNet", b, "R2va"))
    for r in ("R1", "R1nc", "R2", "R3", "R4"):
        add(f"{S} - ShallowConvNet | {r}", y[(S, r)] - y[("ShallowConvNet", r)])
    add(f"R1nc-R1 | {S}", y[(S, "R1nc")] - y[(S, "R1")])
    from .common import holm
    keys = list(E); ph = holm([E[k]["p"] for k in keys])
    for k, q in zip(keys, ph): E[k]["p_holm"] = round(float(q), 4)
    return E


def r3_diagnosis(o):
    d = pd.concat([pd.read_csv(f"{o.res}/{u}_long.csv") for u in ("F1a", "F1b")])
    c = d[d.architecture == "CSOANet-v2"]
    run = c.groupby(["dataset", "fold", "seed", "recipe"]).agg(bacc=("bacc_f32", "mean"), tr=("train_acc", "first"),
                                                                be=("best_epoch", "first"), fl=("training_failure", "first")).reset_index()
    r3 = run[run.recipe == "R3"].copy(); r3["learned"] = r3.tr > 65
    m = r3.merge(run[run.recipe == "R4"][["dataset", "fold", "seed", "bacc"]], on=["dataset", "fold", "seed"], suffixes=("", "_R4"))
    out = {"n_runs": int(len(r3)), "n_learned": int(r3.learned.sum()), "n_flagged": int(r3.fl.sum()),
           "folds_with_mixed_outcome": int((r3.groupby(["dataset", "fold"]).learned.nunique() > 1).sum()),
           "n_folds": int(r3.groupby(["dataset", "fold"]).ngroups)}
    for k, gg in m.groupby("learned"):
        out["learned" if k else "not_learned"] = dict(median_bacc=round(float(gg.bacc.median()), 1), median_train_acc=round(float(gg.tr.median()), 1),
                                                     median_best_epoch=float(gg.be.median()), mean_minus_R4=round(float((gg.bacc - gg.bacc_R4).mean()), 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("res", "nc", "rev6", "out"): ap.add_argument("--" + k, required=True)
    o = ap.parse_args(); os.makedirs(o.out, exist_ok=True)
    g = load(o); V = {}
    for name, spec in DESIGNS.items():
        d = design(g, **spec); R = ["R1" if r == "R1nc" else r for r in spec["recipes"]]
        v = V[name] = {}
        y = d.pivot_table(index="subject", columns=["architecture", "recipe"], values="y")
        yl = d.pivot_table(index="subject", columns=["architecture", "recipe"], values="yl")
        cl = d.drop_duplicates("subject").set_index("subject").cluster.reindex(y.index).to_numpy()
        v["cell_means"] = {f"{a}|{r}": round(float(y[(a, r)].mean()), 2) for a in A for r in R}
        v["omnibus_pct"] = omnibus(d, "y"); v["omnibus_logit"] = omnibus(d, "yl")
        did = lambda Y, r, s: ((Y[("EEGNet", s)] - Y[("CSOANet-v2", s)]) - (Y[("EEGNet", r)] - Y[("CSOANet-v2", r)])).to_numpy()
        v["H2b"] = wild(did(y, "R1", "R2"), cl, f"{name}:H2b")
        v["H2b_logit"] = wild(did(yl, "R1", "R2"), cl, f"{name}:H2b_logit")
        cube = np.array([[y[(a, r)].to_numpy() for r in R] for a in A]).transpose(2, 0, 1)  # S, A, R
        h3 = H3.run(cube, recipes=R)
        v["H3"] = {k: h3[k] for k in ("outcome", "n_definitive", "n_intervals", "sign_reversals",
                                     "equiv_better_changes", "inconclusive_definitive_changes")}
        v["H3"]["categories"] = h3["category_matrix"].to_dict()
        v["H3"]["means"] = {f"{r['pair']}|{r['recipe']}": round(float(r['mean']), 2) for _, r in h3["intervals"].iterrows()}
        res = interaction_residuals(cube)  # per subject
        v["home_residual"] = {}
        for a, hr in HOME.items():
            if hr in R:
                v["home_residual"][a] = wild(res[:, A.index(a), R.index(hr)], cl, f"{name}:home:{a}")
        v["all_residuals"] = {f"{a}|{r}": round(float(res[:, i, j].mean()), 2) for i, a in enumerate(A) for j, r in enumerate(R)}
        print(name, v["omnibus_pct"], v["omnibus_logit"], v["H2b"], v["H2b_logit"], v["H3"]["outcome"], v["H3"]["n_definitive"], flush=True)
    V["exploratory"] = exploratory(g)
    V["csoanet_R3_diagnosis"] = r3_diagnosis(o)
    print(json.dumps(V["csoanet_R3_diagnosis"]))
    json.dump(V, open(f"{o.out}/revision_v19.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
