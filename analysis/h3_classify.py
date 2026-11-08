"""H3 pairwise classification and outcome categories (registration + Amendment 1
§3.2; decision_rules.md §1). Also used for H4 (R1 vs T) and for the int8
sensitivity analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from .common import ARCHS, RECIPES, PAIRS, RECIPE_PAIRS, SESOI, ALPHA, N_INTERVALS

DEFINITIVE = {"A", "B", "E"}


def classify_interval(lo: float, hi: float, sesoi: float = SESOI, equiv_first: bool = False) -> str:
    """A = first architecture better, B = second better, E = equivalent, I = inconclusive.
    Precedence (Amendment 1): 'better' wins over 'equivalent' unless equiv_first."""
    better = "A" if lo > 0 else ("B" if hi < 0 else None)
    equiv = lo > -sesoi and hi < sesoi
    if equiv_first and equiv:
        return "E"
    if better:
        return better
    return "E" if equiv else "I"


def pairwise_intervals(y: np.ndarray, recipes=RECIPES, archs=ARCHS, level: float | None = None,
                       equiv_first: bool = False) -> pd.DataFrame:
    """y: [S, A, R] seed-averaged (NaN rows dropped per pair). Bonferroni over
    len(PAIRS)*len(recipes) intervals unless `level` given."""
    n_int = len(PAIRS) * len(recipes)
    level = level if level is not None else 1 - ALPHA / n_int
    rows = []
    for a, b in PAIRS:
        ia, ib = archs.index(a), archs.index(b)
        for r_idx, r in enumerate(recipes):
            d = y[:, ia, r_idx] - y[:, ib, r_idx]
            d = d[~np.isnan(d)]
            n = len(d)
            m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
            tcrit = stats.t.ppf(1 - (1 - level) / 2, n - 1)
            lo, hi = m - tcrit * se, m + tcrit * se
            rows.append(dict(pair=f"{a} - {b}", A=a, B=b, recipe=r, n=n, mean=m,
                             half=tcrit * se, lo=lo, hi=hi,
                             category=classify_interval(lo, hi, equiv_first=equiv_first)))
    return pd.DataFrame(rows)


def count_changes(intervals: pd.DataFrame, recipes=RECIPES) -> dict:
    cats = intervals.pivot(index="pair", columns="recipe", values="category")[list(recipes)]
    sign_rev = equiv_better = inc_def = 0
    pairs_changed = set()
    import itertools
    for pair, row in cats.iterrows():
        for r1, r2 in itertools.combinations(recipes, 2):
            c1, c2 = row[r1], row[r2]
            if c1 == c2:
                continue
            if c1 in DEFINITIVE and c2 in DEFINITIVE:
                pairs_changed.add(pair)
                if {c1, c2} == {"A", "B"}:
                    sign_rev += 1
                else:
                    equiv_better += 1
            else:
                inc_def += 1
    return dict(category_matrix=cats, n_definitive=int(cats.isin(DEFINITIVE).to_numpy().sum()),
                n_intervals=cats.size, sign_reversals=sign_rev, equiv_better_changes=equiv_better,
                inconclusive_definitive_changes=inc_def, pairs_with_definitive_change=sorted(pairs_changed))


def h3_outcome(ch: dict) -> dict:
    """Registered H3 decision + Amendment-1 B1/B2 labels + A/A' reading."""
    def_def = ch["sign_reversals"] + ch["equiv_better_changes"]
    supported = def_def > 0
    n_def = ch["n_definitive"]; n_int = ch["n_intervals"]
    if supported:
        outcome = "A" if ch["sign_reversals"] > 0 else "A'"
    elif n_def >= np.ceil(2 / 3 * n_int):        # 16 of 24
        outcome = "B1"
    elif n_def < np.floor(1 / 3 * n_int):         # < 8 of 24
        outcome = "B2"
    else:
        outcome = "intermediate"
    return dict(H3_supported=supported, outcome=outcome, n_definitive=n_def, n_intervals=n_int,
                sign_reversals=ch["sign_reversals"], equiv_better_changes=ch["equiv_better_changes"],
                inconclusive_definitive_changes=ch["inconclusive_definitive_changes"])


def run(y: np.ndarray, recipes=RECIPES, equiv_first: bool = False) -> dict:
    iv = pairwise_intervals(y, recipes=recipes, equiv_first=equiv_first)
    ch = count_changes(iv, recipes=recipes)
    out = h3_outcome(ch)
    out["intervals"] = iv
    out["category_matrix"] = ch["category_matrix"]
    out["median_halfwidth"] = float(iv["half"].median())
    return out


def compare_conditions(y_r1: np.ndarray, y_t: np.ndarray) -> dict:
    """H4: classify 6 pairs under R1 and under T, count category changes, Kendall tau
    of the two architecture rankings (by mean accuracy). y_*: [S, A]."""
    y2 = np.stack([y_r1, y_t], axis=2)        # [S, A, 2]
    iv = pairwise_intervals(y2, recipes=["R1", "T"])
    cats = iv.pivot(index="pair", columns="recipe", values="category")
    changed = int((cats["R1"] != cats["T"]).sum())
    order_r1 = np.argsort(-np.nanmean(y_r1, 0)); order_t = np.argsort(-np.nanmean(y_t, 0))
    tau, p = stats.kendalltau(np.nanmean(y_r1, 0), np.nanmean(y_t, 0))
    return dict(intervals=iv, category_matrix=cats, n_category_changes=changed,
                order_R1=[ARCHS[i] for i in order_r1], order_T=[ARCHS[i] for i in order_t],
                kendall_tau=tau, kendall_p=p)
