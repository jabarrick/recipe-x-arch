"""Confirmatory mixed model for H1, H2 (LRT, ML fits) and H2b (per-subject contrast),
with Holm correction within a family. Registration §statistical model; Amendment 1 §3.5.

Model: y ~ dataset + architecture * recipe + (1 | subject), fitted by ML.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from .common import ARCHS, RECIPES, holm, ALPHA


def _fit(formula: str, d: pd.DataFrame, re_formula: str | None = None):
    m = smf.mixedlm(formula, d, groups=d["subject"], re_formula=re_formula)
    return m.fit(reml=False, method=["lbfgs", "powell"], maxiter=2000)


def lrt(full, reduced, df: int) -> tuple[float, float]:
    chi2 = 2 * (full.llf - reduced.llf)
    return float(chi2), float(stats.chi2.sf(max(chi2, 0), df))


def fit_family(avg_long: pd.DataFrame, random_slopes: bool = False) -> dict:
    """avg_long: seed-averaged long table with columns subject, dataset, architecture,
    recipe, y. Returns H1/H2 LRTs, variance shares, fitted models."""
    d = avg_long.dropna(subset=["y"]).copy()
    d["architecture"] = pd.Categorical(d["architecture"].astype(str), categories=ARCHS)
    d["recipe"] = pd.Categorical(d["recipe"].astype(str), categories=RECIPES)
    has_dataset = d["dataset"].nunique() > 1
    base = "y ~ " + ("C(dataset) + " if has_dataset else "")
    re = "~ C(architecture) + C(recipe)" if random_slopes else None

    full = _fit(base + "C(architecture, Sum) * C(recipe, Sum)", d, re)
    no_inter = _fit(base + "C(architecture, Sum) + C(recipe, Sum)", d, re)
    no_recipe = _fit(base + "C(architecture, Sum)", d, re)

    h2_chi2, h2_p = lrt(full, no_inter, df=(len(ARCHS) - 1) * (len(RECIPES) - 1))   # 9
    h1_chi2, h1_p = lrt(no_inter, no_recipe, df=len(RECIPES) - 1)                   # 3

    # variance shares from cell means (sum-of-squares decomposition of the 4x4 table
    # of fitted cell means, weighted equally), plus subject variance from the model
    cell = d.groupby(["architecture", "recipe"])["y"].mean().unstack().loc[ARCHS, RECIPES].to_numpy()
    grand = cell.mean(); a_eff = cell.mean(1) - grand; r_eff = cell.mean(0) - grand
    inter = cell - cell.mean(1, keepdims=True) - cell.mean(0, keepdims=True) + grand
    ss = dict(A=(a_eff ** 2).sum() * len(RECIPES), R=(r_eff ** 2).sum() * len(ARCHS),
              AR=(inter ** 2).sum())
    var_u = float(full.cov_re.iloc[0, 0]); var_e = float(full.scale)
    tot = sum(ss.values()) / (len(ARCHS) * len(RECIPES)) + var_u + var_e
    shares = {k: v / (len(ARCHS) * len(RECIPES)) / tot for k, v in ss.items()}
    shares["subject"] = var_u / tot; shares["residual"] = var_e / tot

    return dict(H1=dict(chi2=h1_chi2, df=3, p=h1_p), H2=dict(chi2=h2_chi2, df=9, p=h2_p),
                variance_shares=shares, cell_means=pd.DataFrame(cell, index=ARCHS, columns=RECIPES),
                recipe_range={a: float(cell[i].max() - cell[i].min()) for i, a in enumerate(ARCHS)},
                models=dict(full=full, no_interaction=no_inter, no_recipe=no_recipe),
                converged=all(m.converged for m in (full, no_inter, no_recipe)))


def h2b_contrast(y: np.ndarray, archs=ARCHS, recipes=RECIPES) -> dict:
    """c[s] = (y[s,EEGNet,R2] - y[s,EEGNet,R1]) - (y[s,v2,R2] - y[s,v2,R1]); two-sided t-test;
    supported only if significant AND positive (after Holm, applied by caller)."""
    e, v = archs.index("EEGNet"), archs.index("CSOANet-v2")
    r1, r2 = recipes.index("R1"), recipes.index("R2")
    c = (y[:, e, r2] - y[:, e, r1]) - (y[:, v, r2] - y[:, v, r1])
    c = c[~np.isnan(c)]
    t, p = stats.ttest_1samp(c, 0.0)
    ci = stats.t.interval(0.95, len(c) - 1, loc=c.mean(), scale=stats.sem(c))
    return dict(c=c, mean=float(c.mean()), ci_lo=ci[0], ci_hi=ci[1], t=float(t), df=len(c) - 1,
                p=float(p), pct_positive=float((c > 0).mean() * 100),
                delta_EEGNet=float(np.nanmean(y[:, e, r2] - y[:, e, r1])),
                delta_v2=float(np.nanmean(y[:, v, r2] - y[:, v, r1])))


def decide_family(avg_long: pd.DataFrame, y: np.ndarray) -> dict:
    """H1, H2, H2b with Holm over the three; returns decisions."""
    fam = fit_family(avg_long)
    h2b = h2b_contrast(y)
    p_raw = [fam["H1"]["p"], fam["H2"]["p"], h2b["p"]]
    p_adj = holm(p_raw)
    fam["H1"]["p_holm"], fam["H2"]["p_holm"], h2b["p_holm"] = map(float, p_adj)
    fam["H1"]["supported"] = p_adj[0] < ALPHA
    fam["H2"]["supported"] = p_adj[1] < ALPHA
    h2b["supported"] = bool(p_adj[2] < ALPHA and h2b["mean"] > 0)
    fam["H2b"] = h2b
    return fam


def cluster_bootstrap(avg_long: pd.DataFrame, y: np.ndarray, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Subject-level cluster bootstrap of: recipe variance share, mean recipe range,
    interaction variance share, H2b mean. Sensitivity only (Amendment 1 §3.5 item 1)."""
    rng = np.random.default_rng(seed)
    subjects = avg_long["subject"].unique(); S = len(subjects)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, S, S)
        yb = y[idx]
        cell = np.nanmean(yb, axis=0)
        grand = cell.mean(); a = cell.mean(1) - grand; r = cell.mean(0) - grand
        inter = cell - cell.mean(1, keepdims=True) - cell.mean(0, keepdims=True) + grand
        ssA, ssR, ssAR = (a**2).sum()*4, (r**2).sum()*4, (inter**2).sum()
        tot = ssA + ssR + ssAR
        rows.append(dict(share_R=ssR/tot, share_AR=ssAR/tot,
                         mean_recipe_range=float((cell.max(1)-cell.min(1)).mean()),
                         h2b_mean=h2b_contrast(yb)["mean"]))
    return pd.DataFrame(rows)
