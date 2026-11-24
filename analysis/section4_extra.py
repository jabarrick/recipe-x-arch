#!/usr/bin/env python3
"""Descriptive and sensitivity values for Section 4 that the frozen driver does not
compute (registered sensitivity analyses and table entries). Added after the
confirmatory runs (S4 row 13). Reads the result files only; no decision is made here.

Usage:
  python -m analysis.section4_extra --res results/confirmatory --out <dir>
"""
from __future__ import annotations
import argparse, itertools, json, os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from .common import ARCHS, RECIPES, HOME, load_long, seed_average, to_cube
from . import mixed_model as mm
from . import h3_classify as h3

RNG = np.random.default_rng(20261124)


def _lrt(full, red, df):
    chi2 = 2 * (full.llf - red.llf)
    return dict(chi2=float(chi2), df=df, p=float(stats.chi2.sf(max(chi2, 0), df)))


def _fit(formula, d, groups="subject", **kw):
    return smf.mixedlm(formula, d, groups=d[groups], **kw).fit(reml=False, method=["lbfgs", "powell"],
                                                                maxiter=2000)


def _prep(avg):
    d = avg.dropna(subset=["y"]).reset_index(drop=True).copy()
    d["architecture"] = pd.Categorical(d["architecture"].astype(str), ARCHS)
    d["recipe"] = pd.Categorical(d["recipe"].astype(str), RECIPES)
    return d


def table5(df, avg, n_boot=2000):
    """Per architecture: recipe range with subject bootstrap CI, best/worst recipe,
    mean train/val accuracy, failures, retried runs."""
    y, subj, _ = to_cube(avg)
    cell = np.nanmean(y, 0)
    rows = []
    boots = np.empty((n_boot, len(ARCHS)))
    for b in range(n_boot):
        cb = np.nanmean(y[RNG.integers(0, len(subj), len(subj))], 0)
        boots[b] = cb.max(1) - cb.min(1)
    # run-level counts: one row per subject in the long file, so count unique runs
    runs = df.drop_duplicates(["architecture", "recipe", "fold", "seed", "dataset"])
    for i, a in enumerate(ARCHS):
        ra = runs[runs.architecture == a]
        rows.append(dict(architecture=a, recipe_range=cell[i].max() - cell[i].min(),
                         range_lo=np.quantile(boots[:, i], .025), range_hi=np.quantile(boots[:, i], .975),
                         best=RECIPES[int(cell[i].argmax())], worst=RECIPES[int(cell[i].argmin())],
                         train_acc=float(df[df.architecture == a].train_acc.mean()),
                         val_acc=float(df[df.architecture == a].val_acc.mean()),
                         n_runs=len(ra), n_fail=int(ra.training_failure.sum()),
                         n_retried=int(ra.retried.sum()),
                         fail_by_recipe={r: int(ra[ra.recipe == r].training_failure.sum()) for r in RECIPES}))
    return pd.DataFrame(rows)


def low_train_cells(df, thr=60.0):
    m = df.groupby(["architecture", "recipe"], observed=True).train_acc.mean()
    return {f"{a}/{r}": round(float(v), 1) for (a, r), v in m.items() if v < thr}


def dataset_effect(avg):
    d = _prep(avg)
    m = _fit("y ~ C(dataset) + C(architecture, Sum) * C(recipe, Sum)", d)
    k = [p for p in m.params.index if p.startswith("C(dataset)")][0]
    return dict(term=k, coef=float(m.params[k]), lo=float(m.conf_int().loc[k, 0]), hi=float(m.conf_int().loc[k, 1]))


def dataset_interactions(avg):
    d = _prep(avg)
    base = "C(architecture, Sum) * C(recipe, Sum)"
    m0 = _fit(f"y ~ C(dataset) + {base}", d)
    mDA = _fit(f"y ~ C(dataset) + {base} + C(dataset):C(architecture, Sum)", d)
    mDR = _fit(f"y ~ C(dataset) + {base} + C(dataset):C(recipe, Sum)", d)
    mDAR2 = _fit(f"y ~ C(dataset) + {base} + C(dataset):C(architecture, Sum) + C(dataset):C(recipe, Sum)", d)
    m3 = _fit(f"y ~ C(dataset) * {base}", d)
    return dict(dataset_x_arch=_lrt(mDA, m0, 3), dataset_x_recipe=_lrt(mDR, m0, 3),
                three_way=_lrt(m3, mDAR2, 9))


def separate_datasets(avg):
    out = {}
    for ds in sorted(avg.dataset.unique()):
        a = avg[avg.dataset == ds]
        y, _, _ = to_cube(a)
        fam = mm.decide_family(a, y)
        H3 = h3.run(y)
        out[ds] = dict(N=int(a.subject.nunique()), H1_p=fam["H1"]["p"], H2_chi2=fam["H2"]["chi2"],
                       H2_p=fam["H2"]["p"], H2_supported=bool(fam["H2"]["supported"]),
                       H2b_mean=fam["H2b"]["mean"], H2b_p_holm=fam["H2b"]["p_holm"],
                       H3_outcome=H3["outcome"], n_definitive=H3["n_definitive"],
                       sign_reversals=H3["sign_reversals"], equiv_better=H3["equiv_better_changes"])
    return out


def fold_fixed(avg, df):
    fold = df.drop_duplicates("subject").set_index("subject")["fold"]
    d = _prep(avg); d["fold"] = d["subject"].map(fold).astype(str)
    base = "y ~ C(dataset) + C(fold) + "
    full = _fit(base + "C(architecture, Sum) * C(recipe, Sum)", d)
    noi = _fit(base + "C(architecture, Sum) + C(recipe, Sum)", d)
    nor = _fit(base + "C(architecture, Sum)", d)
    return dict(H1=_lrt(noi, nor, 3), H2=_lrt(full, noi, 9))


def seed_random(df):
    """Unaveraged runs; subject random intercept plus seed-within-subject variance component."""
    d = df.reset_index(drop=True).copy()
    d["y"] = d["bacc_f32"]; d["seed"] = d["seed"].astype(str)
    d["architecture"] = pd.Categorical(d["architecture"].astype(str), ARCHS)
    d["recipe"] = pd.Categorical(d["recipe"].astype(str), RECIPES)
    vc = {"seed": "0 + C(seed)"}
    base = "y ~ C(dataset) + " if d.dataset.nunique() > 1 else "y ~ "
    full = _fit(base + "C(architecture, Sum) * C(recipe, Sum)", d, vc_formula=vc)
    noi = _fit(base + "C(architecture, Sum) + C(recipe, Sum)", d, vc_formula=vc)
    nor = _fit(base + "C(architecture, Sum)", d, vc_formula=vc)
    return dict(H1=_lrt(noi, nor, 3), H2=_lrt(full, noi, 9), seed_var=float(full.vcomp[0]),
                converged=bool(full.converged and noi.converged and nor.converged))


def art_anova(avg):
    """Aligned rank transform (Wobbrock et al. 2011) for the recipe main effect and
    the interaction, each tested by LRT on the ranked aligned response with a subject
    random intercept."""
    d = _prep(avg)
    gm = d.y.mean()
    ma = d.groupby("architecture", observed=True).y.transform("mean")
    mr = d.groupby("recipe", observed=True).y.transform("mean")
    mc = d.groupby(["architecture", "recipe"], observed=True).y.transform("mean")
    resid = d.y - mc
    out = {}
    # recipe main effect: residual + estimated recipe effect
    d["r_R"] = (resid + (mr - gm)).rank()
    out["H1"] = _lrt(_fit("r_R ~ C(architecture, Sum) + C(recipe, Sum)", d),
                     _fit("r_R ~ C(architecture, Sum)", d), 3)
    d["r_AR"] = (resid + (mc - ma - mr + gm)).rank()
    out["H2"] = _lrt(_fit("r_AR ~ C(architecture, Sum) * C(recipe, Sum)", d),
                     _fit("r_AR ~ C(architecture, Sum) + C(recipe, Sum)", d), 9)
    return out


def seed_sd(df):
    g = df.groupby(["subject", "architecture", "recipe"], observed=True).bacc_f32.std(ddof=1)
    return dict(median=float(g.median()), mean=float(g.mean()))


def int8_summary(df, cov):
    agree = df.groupby("architecture", observed=True).int8_agreement.mean().reindex(ARCHS)
    c = cov.groupby("architecture").agg(share=("share", "mean"), q=("quantized_ops", "mean"),
                                        n=("compute_ops", "mean")).reindex(ARCHS)
    fl = cov.groupby("architecture").float_op_types.agg(lambda s: sorted(set(" ".join(s.dropna()).split())))
    delta = (df.bacc_f32 - df.bacc_int8).groupby(df.architecture, observed=True).mean().reindex(ARCHS)
    return dict(agreement_overall=float(df.int8_agreement.mean()),
                agreement=agree.round(2).to_dict(), f32_minus_int8=delta.round(2).to_dict(),
                coverage_share=c["share"].round(3).to_dict(),
                quantized_of_compute={a: f"{c.loc[a, 'q']:.0f}/{c.loc[a, 'n']:.0f}" for a in ARCHS},
                float_ops=fl.reindex(ARCHS).to_dict())


def failure_dependence(df, avg):
    """Which H3 category changes involve only cells containing failures."""
    y, _, _ = to_cube(avg)
    H = h3.run(y); cats = H["category_matrix"]
    failcells = {(a, r) for (a, r), v in
                 df.groupby(["architecture", "recipe"], observed=True).training_failure.sum().items() if v > 0}
    out = []
    for pair, row in cats.iterrows():
        a, b = pair.split(" - ")
        for r1, r2 in itertools.combinations(RECIPES, 2):
            c1, c2 = row[r1], row[r2]
            if c1 != c2 and c1 in h3.DEFINITIVE and c2 in h3.DEFINITIVE:
                cells = {(a, r1), (b, r1), (a, r2), (b, r2)}
                out.append(dict(pair=pair, recipes=f"{r1}-{r2}", change=f"{c1}->{c2}",
                                involves_failure_cells=sorted(f"{x}/{y_}" for x, y_ in cells & failcells)))
    return out


def h4_extra(units, res):
    """Kendall tau bootstrap CI, T gain, winner's-curse shrinkage, selected configurations."""
    from importlib import util
    spec = util.spec_from_file_location("cfg", f"{res}/configurable.py")
    cfgmod = util.module_from_spec(spec)
    try:
        spec.loader.exec_module(cfgmod); configs = {c["id"]: c for c in cfgmod.sample_configs()}
    except Exception:
        configs = {}
    rec = pd.concat([load_long(f"{res}/{u}_long.csv") for u in units])
    T = pd.concat([pd.read_csv(f"{res}/{u}_T_long.csv") for u in units])
    Tp = pd.concat([pd.read_csv(f"{res}/{u}_Tprime_long.csv") for u in units])
    m = lambda d: seed_average(d, "bacc_f32").pivot_table(index="subject", columns="architecture",
                                                          values="y", observed=False).reindex(columns=ARCHS)
    r1, t, tp = m(rec[rec.recipe == "R1"]), m(T), m(Tp)
    s = r1.index.intersection(t.index); a1, at = r1.loc[s].to_numpy(), t.loc[s].to_numpy()
    taus = []
    for _ in range(2000):
        i = RNG.integers(0, len(s), len(s))
        taus.append(stats.kendalltau(a1[i].mean(0), at[i].mean(0))[0])
    gain = (t.loc[s].mean() - r1.loc[s].mean()).reindex(ARCHS)
    gain_p = (tp.loc[s].mean() - r1.loc[s].mean()).reindex(ARCHS)
    sel, shrink = {}, {}
    for u in units:
        srch = pd.read_csv(f"{res}/{u}_T_search.csv")
        tl = pd.read_csv(f"{res}/{u}_T_long.csv")
        for a in ARCHS:
            sc = srch[srch.architecture == a].groupby("config_id").val_bacc.mean()
            best = int(sc.reset_index().sort_values(["val_bacc", "config_id"], ascending=[False, True]).iloc[0].config_id)
            test = float(tl[tl.architecture == a].bacc_f32.mean())
            sel[f"{u}/{a}"] = dict(config_id=best, search_score=float(sc.max()), test_mean=test,
                                   config=configs.get(best))
            shrink[f"{u}/{a}"] = float(sc.max() - test)
    return dict(tau_ci=[float(np.nanquantile(taus, .025)), float(np.nanquantile(taus, .975))],
                gain_T=gain.round(3).to_dict(), gain_Tprime=gain_p.round(3).to_dict(),
                selected=sel, shrinkage=shrink)


def exploratory_gap(df):
    """(b) recipe gain over R4 vs train-validation gap under R4, per architecture x recipe cell."""
    c = df.groupby(["architecture", "recipe"], observed=True).agg(test=("bacc_f32", "mean"),
                                                                 tr=("train_acc", "mean"),
                                                                 va=("val_acc", "mean")).reset_index()
    base = c[c.recipe == "R4"].set_index("architecture")
    c["gain_vs_R4"] = c.apply(lambda r: r.test - base.loc[r.architecture, "test"], axis=1)
    c["gap_R4"] = c.architecture.map(base.tr - base.va)
    sub = c[c.recipe != "R4"]
    rho = stats.spearmanr(sub.gap_R4, sub.gain_vs_R4)
    return dict(cells=c.round(2).to_dict(orient="records"), spearman=float(rho[0]), p=float(rho[1]), n=len(sub))


def exploratory_noR3(df):
    """Exploratory (post hoc, not registered): home residuals on the 4 x 3 table without R3,
    the recipe in which ShallowConvNet (all runs) and CSOANet-v2 (11/30 runs) were flagged."""
    from .home_recipe import tost
    y, _, _ = to_cube(seed_average(df, "bacc_f32")); k = [RECIPES.index(r) for r in ("R1", "R2", "R4")]
    y3 = y[:, :, k]
    res = y3 - y3.mean(2, keepdims=True) - y3.mean(1, keepdims=True) + y3.mean((1, 2), keepdims=True)
    out = {}
    for a, r in (("CSOANet-v2", "R1"), ("EEGNet", "R2")):
        h = res[:, ARCHS.index(a), ["R1", "R2", "R4"].index(r)]
        t, p = stats.ttest_1samp(h, 0.0)
        ci = stats.t.interval(0.95, len(h) - 1, loc=h.mean(), scale=stats.sem(h))
        eq = tost(h)
        out[a] = dict(h=float(h.mean()), lo=float(ci[0]), hi=float(ci[1]), p=float(p),
                      tost_lo=float(eq["ci90_lo"]), tost_hi=float(eq["ci90_hi"]), equivalent=bool(eq["equivalent"]))
    t = np.nanmean(y3, 0); r = t - t.mean(1, keepdims=True) - t.mean(0, keepdims=True) + t.mean()
    out["residuals"] = pd.DataFrame(r, index=ARCHS, columns=["R1", "R2", "R4"]).round(3).to_dict()
    tf = np.nanmean(y, 0); rf = tf - tf.mean(1, keepdims=True) - tf.mean(0, keepdims=True) + tf.mean()
    out["residuals_4x4"] = pd.DataFrame(rf, index=ARCHS, columns=RECIPES).round(3).to_dict()
    return out


def safe(f, *a):
    try:
        return f(*a)
    except Exception as e:  # reported, not hidden
        return dict(error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/confirmatory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    V = {}
    F1 = pd.concat([load_long(f"{a.res}/F1a_long.csv"), load_long(f"{a.res}/F1b_long.csv")])
    F2 = load_long(f"{a.res}/F2_long.csv")
    cov1 = pd.concat([pd.read_csv(f"{a.res}/{u}_int8_coverage.csv") for u in ("F1a", "F1b")])
    cov2 = pd.read_csv(f"{a.res}/F2_int8_coverage.csv")
    for tag, df, cov in (("F1", F1, cov1), ("F2", F2, cov2)):
        avg = seed_average(df, "bacc_f32")
        t5 = table5(df, avg); t5.to_csv(f"{a.out}/{tag}_table5.csv", index=False)
        V[tag] = dict(table5=t5.to_dict(orient="records"), low_train_cells=low_train_cells(df),
                      seed_sd=seed_sd(df), int8=int8_summary(df, cov), excluded_subjects=0,
                      fold_fixed=safe(fold_fixed, avg, df) if tag == "F1" else "not applicable (within-subject)",
                      seed_random=safe(seed_random, df), art=safe(art_anova, avg),
                      failure_dependence=failure_dependence(df, avg), exploratory_b=exploratory_gap(df))
        if tag == "F1":
            V[tag]["dataset_effect"] = dataset_effect(avg)
            V[tag]["dataset_interactions"] = dataset_interactions(avg)
            V[tag]["separate"] = separate_datasets(avg)
            V[tag]["h4"] = h4_extra(("F1a", "F1b"), a.res)
            V[tag]["exploratory_noR3"] = exploratory_noR3(df)
        else:
            V[tag]["h4"] = h4_extra(("F2",), a.res)
    with open(f"{a.out}/section4_extra.json", "w") as f:
        json.dump(V, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("written", f"{a.out}/section4_extra.json")


if __name__ == "__main__":
    main()
