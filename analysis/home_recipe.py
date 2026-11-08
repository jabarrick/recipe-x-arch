"""Home-recipe analysis (Amendment 1 §3.1; decision_rules.md §2).

Two estimation paths:
  * cell-mean path  : h_a from the double-centred 4x4 table (THE definition)
  * model path      : (alpha*rho)_{a,home(a)} from a sum-coded mixed model
                      (check on complete tables; fallback when >10 % of
                      subjects would be dropped in the failure-exclusion
                      sensitivity analysis)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from .common import ARCHS, RECIPES, HOME, SESOI, ALPHA, holm


# --------------------------------------------------------------------------
# cell-mean path
# --------------------------------------------------------------------------

def interaction_residuals(table: np.ndarray) -> np.ndarray:
    """Double-centred residuals of a (..., A, R) table. NaN-free input only."""
    t = np.asarray(table, float)
    row = t.mean(axis=-1, keepdims=True)
    col = t.mean(axis=-2, keepdims=True)
    grand = t.mean(axis=(-1, -2), keepdims=True)
    return t - row - col + grand


def home_advantage_cellmeans(table: np.ndarray, archs=ARCHS, recipes=RECIPES) -> dict:
    """h_a for each architecture with a home recipe, from one (A, R) table."""
    res = interaction_residuals(table)
    return {a: float(res[archs.index(a), recipes.index(HOME[a])]) for a in archs if a in HOME}


def per_subject_home(y: np.ndarray, archs=ARCHS, recipes=RECIPES) -> pd.DataFrame:
    """y: [S, A, R] seed-averaged, complete tables only (rows with NaN dropped).
    Returns DataFrame[subject_idx, architecture, h]."""
    complete = ~np.isnan(y).any(axis=(1, 2))
    res = interaction_residuals(y[complete])                   # [S', A, R]
    rows = []
    for s_idx, s in enumerate(np.where(complete)[0]):
        for a in archs:
            if a in HOME:
                rows.append(dict(subject_idx=s, architecture=a,
                                 h=res[s_idx, archs.index(a), recipes.index(HOME[a])]))
    out = pd.DataFrame(rows)
    out.attrs["n_dropped"] = int((~complete).sum())
    return out


def residual_ranges(y: np.ndarray, n_boot: int = 2000, seed: int = 0,
                    archs=ARCHS) -> pd.DataFrame:
    """Residual range (max-min of the four (alpha*rho)_{a,r}) and raw range
    (best-worst recipe mean) per architecture, from the cell means of y,
    with subject-level bootstrap CIs and the share of resamples in which each
    architecture has the smallest residual range."""
    complete = ~np.isnan(y).any(axis=(1, 2))
    yc = y[complete]
    rng = np.random.default_rng(seed)
    S = yc.shape[0]

    def ranges(sample):
        means = sample.mean(axis=0)                      # [A, R]
        res = interaction_residuals(means)
        return res.max(1) - res.min(1), means.max(1) - means.min(1)

    rr0, raw0 = ranges(yc)
    boot_rr = np.empty((n_boot, len(archs))); boot_raw = np.empty_like(boot_rr)
    for b in range(n_boot):
        idx = rng.integers(0, S, S)
        boot_rr[b], boot_raw[b] = ranges(yc[idx])
    smallest = (boot_rr.argmin(axis=1)[:, None] == np.arange(len(archs))).mean(0)
    return pd.DataFrame(dict(
        architecture=archs,
        residual_range=rr0,
        rr_lo=np.percentile(boot_rr, 2.5, axis=0), rr_hi=np.percentile(boot_rr, 97.5, axis=0),
        raw_range=raw0,
        raw_lo=np.percentile(boot_raw, 2.5, axis=0), raw_hi=np.percentile(boot_raw, 97.5, axis=0),
        share_smallest_residual_range=smallest))


def tost(x: np.ndarray, bound: float = SESOI, alpha: float = ALPHA) -> dict:
    """Two one-sided tests against ±bound; 90 % CI = 1-2*alpha."""
    x = np.asarray(x, float); n = len(x)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    t_lo = (m + bound) / se; t_hi = (m - bound) / se
    p_lo = 1 - stats.t.cdf(t_lo, n - 1); p_hi = stats.t.cdf(t_hi, n - 1)
    ci = stats.t.interval(1 - 2 * alpha, n - 1, loc=m, scale=se)
    return dict(mean=m, ci90_lo=ci[0], ci90_hi=ci[1], p_tost=max(p_lo, p_hi),
                equivalent=bool(max(p_lo, p_hi) < alpha))


def home_tests(per_subj: pd.DataFrame) -> pd.DataFrame:
    """One-sample t-tests of h[s,a] vs 0, Holm over the three home architectures;
    TOST added for CSOANet-v2."""
    rows = []
    for a, g in per_subj.groupby("architecture", observed=True):
        h = g["h"].to_numpy()
        t, p = stats.ttest_1samp(h, 0.0)
        ci = stats.t.interval(0.95, len(h) - 1, loc=h.mean(), scale=stats.sem(h))
        rows.append(dict(architecture=a, n=len(h), h_mean=h.mean(), ci_lo=ci[0], ci_hi=ci[1],
                         t=t, p_raw=p))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"])
    out["significant"] = out["p_holm"] < ALPHA
    v2 = per_subj.loc[per_subj.architecture == "CSOANet-v2", "h"].to_numpy()
    eq = tost(v2)
    out["tost_ci90_lo"] = np.where(out.architecture == "CSOANet-v2", eq["ci90_lo"], np.nan)
    out["tost_ci90_hi"] = np.where(out.architecture == "CSOANet-v2", eq["ci90_hi"], np.nan)
    out["tost_equivalent"] = np.where(out.architecture == "CSOANet-v2", eq["equivalent"], None)
    return out


# --------------------------------------------------------------------------
# model path (sum-coded mixed model)
# --------------------------------------------------------------------------

def home_advantage_model(avg_long: pd.DataFrame) -> dict:
    """Fit y ~ dataset + C(arch, Sum) * C(recipe, Sum) + (1|subject), ML, and
    return the sum-coded interaction coefficient for each home cell. Works on
    incomplete tables. Returns {arch: h_a}."""
    import statsmodels.formula.api as smf
    d = avg_long.dropna(subset=["y"]).copy()
    d["architecture"] = pd.Categorical(d["architecture"].astype(str), categories=ARCHS)
    d["recipe"] = pd.Categorical(d["recipe"].astype(str), categories=RECIPES)
    formula = ("y ~ " + ("C(dataset) + " if d["dataset"].nunique() > 1 else "")
               + "C(architecture, Sum) * C(recipe, Sum)")
    m = smf.mixedlm(formula, d, groups=d["subject"]).fit(reml=False)
    # Sum coding drops the last level; reconstruct full interaction matrix.
    A, R = len(ARCHS), len(RECIPES)
    inter = np.zeros((A, R))
    # statsmodels names: C(architecture, Sum)[S.EEGNet]:C(recipe, Sum)[S.R1]
    for i, a in enumerate(ARCHS[:-1]):
        for j, r in enumerate(RECIPES[:-1]):
            name = f"C(architecture, Sum)[S.{a}]:C(recipe, Sum)[S.{r}]"
            inter[i, j] = m.params[name]
    # fill dropped level by zero-sum constraints
    inter[:-1, -1] = -inter[:-1, :-1].sum(axis=1)
    inter[-1, :] = -inter[:-1, :].sum(axis=0)
    return {a: float(inter[ARCHS.index(a), RECIPES.index(HOME[a])]) for a in HOME}, inter


# --------------------------------------------------------------------------
# decision mapping (decision_rules.md §2.5)
# --------------------------------------------------------------------------

def mechanism_decision(tests: pd.DataFrame, ranges: pd.DataFrame, ranking_threshold=0.80) -> str:
    v2 = tests.set_index("architecture").loc["CSOANet-v2"]
    v2_pos_sig = bool(v2["significant"] and v2["h_mean"] > 0)
    v2_equiv = bool(v2["tost_equivalent"])
    v2_smallest = float(ranges.set_index("architecture").loc["CSOANet-v2", "share_smallest_residual_range"])
    if v2_pos_sig:
        return "i"
    if v2_equiv and v2_smallest >= ranking_threshold:
        return "ii"
    return "iii"


def run(y: np.ndarray, avg_long: pd.DataFrame | None = None, drop_R4: bool = False,
        n_boot: int = 2000) -> dict:
    """Full home-recipe analysis on a cube y[S,A,R]. If drop_R4, the R4 column
    is removed for all architectures (4x3 sensitivity)."""
    if drop_R4:
        keep = [RECIPES.index(r) for r in RECIPES if r != "R4"]
        y = y[:, :, keep]
        recipes = [r for r in RECIPES if r != "R4"]
    else:
        recipes = RECIPES
    ps = per_subject_home(y, recipes=recipes)
    tests = home_tests(ps)
    ranges = residual_ranges(y, n_boot=n_boot)
    out = dict(per_subject=ps, tests=tests, ranges=ranges, n_dropped=ps.attrs["n_dropped"],
               h_cellmeans=home_advantage_cellmeans(np.nanmean(y, axis=0), recipes=recipes),
               mechanism=mechanism_decision(tests, ranges))
    if avg_long is not None and not drop_R4:
        out["h_model"], out["interaction_model"] = home_advantage_model(avg_long)
    return out
