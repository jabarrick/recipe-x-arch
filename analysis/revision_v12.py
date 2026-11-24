#!/usr/bin/env python3
"""Analyses added in revision V1.2 (post hoc; S4 row 15). None changes a registered decision.

1. Sensitivity of the clustered H3 intervals to the degrees of freedom and to the model:
   t(9) (V1.1), t(8) with a dataset fixed effect, normal critical value (design effect only),
   a cluster bootstrap over the 10 dataset x fold clusters, and leave-one-fold-out of the
   registered (subject-level) H3.
2. Difference-in-differences for EEGNet - CSOANet-v2 (H2b is the R2-vs-R1 case) and for the
   other recipes against R1, subject-level and clustered.
3. Condition-T search regression with standard errors clustered by configuration.
4. Run-level training accuracy of CSOANet-v2 under R3.
5. The F2 int8 category change.

Usage: python -m analysis.revision_v12 --res results/confirmatory --ana docs/confirmatory_analysis_V1.1 --out <dir>
"""
from __future__ import annotations
import argparse, itertools, json, os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from .common import ARCHS, RECIPES, PAIRS, ALPHA, load_long, seed_average
from .h3_classify import classify_interval, run as h3_run
from .revision_v11 import avg_with_cluster, cluster_mean_ci

LEVEL = 1 - ALPHA / (len(PAIRS) * len(RECIPES))
DEF = {"A", "B", "E"}
RNG = np.random.default_rng(20261124)


def summarise(cats: pd.DataFrame) -> dict:
    m = cats.pivot(index="pair", columns="recipe", values="category")[RECIPES]
    sign = eb = inc = 0
    for _, row in m.iterrows():
        for r1, r2 in itertools.combinations(RECIPES, 2):
            c1, c2 = row[r1], row[r2]
            if c1 == c2: continue
            if c1 in DEF and c2 in DEF:
                if {c1, c2} == {"A", "B"}: sign += 1
                else: eb += 1
            else: inc += 1
    nd = int(m.isin(DEF).to_numpy().sum())
    out = "A" if sign else ("A'" if eb else ("B1" if nd >= 16 else ("B2" if nd < 8 else "intermediate")))
    return dict(n_definitive=nd, sign_reversals=sign, equiv_better=eb, inc_def=inc, outcome=out,
                median_halfwidth=float(cats.half.median()))


def wide(a):
    y = a.pivot_table(index="subject", columns=["architecture", "recipe"], values="y", observed=False)
    meta = a.drop_duplicates("subject").set_index("subject")[["cluster", "dataset"]].reindex(y.index)
    return y, meta


def intervals(a, method):
    y, meta = wide(a)
    rows = []
    for p, q in PAIRS:
        for r in RECIPES:
            d = pd.DataFrame(dict(x=(y[(p, r)] - y[(q, r)]).to_numpy(), cl=meta.cluster.to_numpy(),
                                  ds=meta.dataset.astype(str).to_numpy())).dropna()
            G = d.cl.nunique()
            if method == "t9":
                m, lo, hi, _ = cluster_mean_ci(d.x.to_numpy(), d.cl.to_numpy(), LEVEL)
            elif method in ("z", "t8_dataset"):
                if method == "z":
                    fit = smf.mixedlm("x ~ 1", d, groups=d["cl"]).fit(reml=True)
                    m, se = float(fit.fe_params["Intercept"]), float(fit.bse_fe["Intercept"]); crit = stats.norm.ppf(1 - (1 - LEVEL) / 2)
                else:   # dataset fixed effect; estimate = mean of the two dataset means, df = G - 2
                    fit = smf.mixedlm("x ~ 0 + C(ds)", d, groups=d["cl"]).fit(reml=True)
                    b, V = fit.fe_params.to_numpy(), fit.cov_params().iloc[:2, :2].to_numpy()
                    w = np.array([.5, .5]); m, se = float(w @ b), float(np.sqrt(w @ V @ w)); crit = stats.t.ppf(1 - (1 - LEVEL) / 2, G - 2)
                lo, hi = m - crit * se, m + crit * se
            elif method == "boot":
                cls = d.cl.unique(); cm = d.groupby("cl").x.agg(["sum", "count"])
                bs = []
                for _ in range(2000):
                    k = RNG.choice(cls, len(cls), replace=True)
                    s_ = cm.loc[k]; bs.append(s_["sum"].sum() / s_["count"].sum())
                m = d.x.mean(); se = float(np.std(bs, ddof=1)); crit = stats.norm.ppf(1 - (1 - LEVEL) / 2)
                lo, hi = m - crit * se, m + crit * se
            rows.append(dict(pair=f"{p} - {q}", recipe=r, mean=m, lo=lo, hi=hi, half=(hi - lo) / 2,
                             category=classify_interval(lo, hi)))
    return pd.DataFrame(rows)


def leave_one_fold_out(df):
    out = {}
    for f in sorted(df.fold.unique()):
        d = df[df.fold != f]
        y = d.pipe(seed_average).pivot_table(index="subject", columns=["architecture", "recipe"], values="y", observed=False)
        cube = np.stack([[y[(a, r)].to_numpy() for r in RECIPES] for a in ARCHS]).transpose(2, 0, 1)
        H = h3_run(cube)
        out[f"drop_fold_{f + 1}"] = dict(N=int(cube.shape[0]), outcome=H["outcome"], n_definitive=H["n_definitive"],
                                         sign_reversals=H["sign_reversals"], equiv_better=H["equiv_better_changes"])
    return out


def did_eeg_v2(a):
    y, meta = wide(a)
    g = {r: y[("EEGNet", r)] - y[("CSOANet-v2", r)] for r in RECIPES}
    out = {}
    for r in ("R2", "R3", "R4"):
        x = (g[r] - g["R1"]).to_numpy()
        ci = stats.t.interval(.95, len(x) - 1, x.mean(), stats.sem(x))
        m, lo, hi, _ = cluster_mean_ci(x, meta.cluster.to_numpy(), .95)
        out[f"{r}-R1"] = dict(mean=float(x.mean()), lo=float(ci[0]), hi=float(ci[1]), lo_cluster=lo, hi_cluster=hi)
    gs = {r: float(g[r].mean()) for r in RECIPES}
    return dict(did=out, gap_EEGNet_minus_v2=gs)


def search_regression_clustered(res, units, cfgfile):
    cfg = pd.DataFrame(json.load(open(cfgfile))).rename(columns={"id": "config_id"})
    S = pd.concat([pd.read_csv(f"{res}/{u}_T_search.csv").assign(unit=u) for u in units]).merge(cfg, on="config_id")
    S["loglr"] = np.log10(S.lr)
    for c in ("noise", "shift", "crop", "chdrop", "mixup"): S[c] = S[c].astype(int)
    S["ls"] = (S.label_smoothing > 0).astype(int)
    f = ("val_bacc ~ C(unit) * C(architecture) + loglr + C(optimizer) + C(schedule) + C(batch) + noise + shift"
         " + crop + chdrop + mixup + ls + C(select)")
    m = smf.ols(f, S).fit(cov_type="cluster", cov_kwds=dict(groups=S["config_id"]))
    keep = [k for k in m.params.index if not k.startswith(("C(unit)", "C(architecture)", "Intercept"))]
    ci = m.conf_int()
    # per-architecture crop effect
    per = {}
    for a in ARCHS:
        mm = smf.ols(f.replace("C(unit) * C(architecture)", "C(unit)"), S[S.architecture == a]).fit(
            cov_type="cluster", cov_kwds=dict(groups=S[S.architecture == a]["config_id"]))
        per[a] = dict(coef=float(mm.params["crop"]), lo=float(mm.conf_int().loc["crop", 0]), hi=float(mm.conf_int().loc["crop", 1]))
    return dict(terms=pd.DataFrame(dict(coef=m.params[keep], lo=ci.loc[keep, 0], hi=ci.loc[keep, 1], p=m.pvalues[keep]))
                .reset_index().rename(columns={"index": "term"}).round(3).to_dict(orient="records"),
                n_configs=int(S.config_id.nunique()), n=int(len(S)), crop_by_arch=per)


def csoanet_r3(df):
    r = df[(df.architecture == "CSOANet-v2") & (df.recipe == "R3")].drop_duplicates(["dataset", "fold", "seed"])
    t = df[(df.architecture == "CSOANet-v2") & (df.recipe == "R3")].groupby(["dataset", "fold", "seed"]).bacc_f32.mean()
    r = r.set_index(["dataset", "fold", "seed"]).assign(test=t).reset_index()
    bins = pd.cut(r.train_acc, [0, 55, 60, 65, 70, 100], right=True)
    return dict(n_runs=int(len(r)), flagged=int(r.training_failure.sum()),
                train_acc_bins={str(k): int(v) for k, v in bins.value_counts().sort_index().items()},
                runs=r[["dataset", "fold", "seed", "train_acc", "val_acc", "test", "training_failure"]]
                .sort_values("train_acc").round(1).to_dict(orient="records"))


def f2_int8_change(ana):
    a = pd.read_csv(f"{ana}/F2_h3_intervals_f32.csv"); b = pd.read_csv(f"{ana}/F2_h3_intervals_int8.csv")
    j = a.assign(int8=b.category, mean8=b["mean"])
    ch = j[j.category != j.int8]
    return ch[["pair", "recipe", "mean", "category", "mean8", "int8"]].round(2).to_dict(orient="records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/confirmatory")
    ap.add_argument("--ana", default="docs/confirmatory_analysis_V1.1")
    ap.add_argument("--cfg", default="docs/Tprime/tprime_configs.json")
    ap.add_argument("--out", required=True)
    o = ap.parse_args(); os.makedirs(o.out, exist_ok=True)
    df = pd.concat([load_long(f"{o.res}/F1a_long.csv"), load_long(f"{o.res}/F1b_long.csv")])
    a = avg_with_cluster(df)
    V = {"h3_sensitivity": {}}
    tabs = {}
    for meth in ("t9", "t8_dataset", "z", "boot"):
        iv = intervals(a, meth); tabs[meth] = iv
        V["h3_sensitivity"][meth] = summarise(iv)
    pd.concat([t.assign(method=k) for k, t in tabs.items()]).to_csv(f"{o.out}/F1_h3_intervals_cluster_sensitivity.csv", index=False)
    V["h3_leave_one_fold_out"] = leave_one_fold_out(df)
    V["did_EEGNet_CSOANet"] = did_eeg_v2(a)
    V["search_regression_clustered"] = search_regression_clustered(o.res, ("F1a", "F1b"), o.cfg)
    V["csoanet_r3"] = csoanet_r3(df)
    V["f2_int8_change"] = f2_int8_change(o.ana)
    json.dump(V, open(f"{o.out}/revision_v12.json", "w"), indent=2, default=float)
    print(json.dumps({k: V[k] for k in ("h3_sensitivity", "h3_leave_one_fold_out", "did_EEGNet_CSOANet", "f2_int8_change")}, indent=1, default=float))
    print(json.dumps(V["search_regression_clustered"]["crop_by_arch"], indent=1))
    print({k: v for k, v in V["csoanet_r3"].items() if k != "runs"})


if __name__ == "__main__":
    main()
