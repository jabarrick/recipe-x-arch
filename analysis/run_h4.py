#!/usr/bin/env python3
"""H4 driver (secondary): R1 versus condition T, plus the Amendment-1 sensitivity
analyses (T' and fold-1-only T). Sequencing only: the decision logic is the frozen
h3_classify.compare_conditions / count_changes. Added after v1.1-frozen (S4 row 12).

Usage:
  python -m analysis.run_h4 --family F1 --units F1a F1b --res results/confirmatory --out <dir>
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from .common import ARCHS, ALPHA, PAIRS, load_long, seed_average
from . import h3_classify as h3


def matrix(df, recipe):
    """seed-averaged float32 -> [S, A] for one recipe/condition."""
    d = df[df.recipe.astype(str) == recipe]
    avg = seed_average(d, "bacc_f32")
    m = avg.pivot_table(index="subject", columns="architecture", values="y", observed=False)
    return m.reindex(columns=ARCHS)


def compare(r1, other, label, out, tag, level=None):
    subj = r1.index.intersection(other.index)
    y1, y2 = r1.loc[subj].to_numpy(), other.loc[subj].to_numpy()
    res = h3.compare_conditions(y1, y2)
    if level is not None:   # same comparison at an explicitly given interval level
        y = np.stack([y1, y2], axis=2)
        res["intervals"] = h3.pairwise_intervals(y, recipes=["R1", "T"], level=level)
        res["category_matrix"] = res["intervals"].pivot(index="pair", columns="recipe", values="category")
        res["n_category_changes"] = int((res["category_matrix"]["R1"] != res["category_matrix"]["T"]).sum())
    ch = h3.count_changes(res["intervals"], recipes=["R1", "T"])
    res["intervals"].assign(condition=label).to_csv(f"{out}/{tag}_h4_intervals_{label}.csv", index=False)
    return dict(condition=label, N=len(subj), level=float(level if level else 1 - ALPHA / (len(PAIRS) * 2)),
                n_category_changes=res["n_category_changes"],
                definitive_changes=ch["sign_reversals"] + ch["equiv_better_changes"],
                sign_reversals=ch["sign_reversals"], equiv_better_changes=ch["equiv_better_changes"],
                inconclusive_definitive_changes=ch["inconclusive_definitive_changes"],
                categories=res["category_matrix"].to_dict(orient="index"),
                order_R1=res["order_R1"], order_T=res["order_T"],
                kendall_tau=float(res["kendall_tau"]), kendall_p=float(res["kendall_p"]),
                mean_R1=dict(zip(ARCHS, np.nanmean(y1, 0).round(3).tolist())),
                mean_cond=dict(zip(ARCHS, np.nanmean(y2, 0).round(3).tolist())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="F1")
    ap.add_argument("--units", nargs="+", default=["F1a", "F1b"])
    ap.add_argument("--res", default="results/confirmatory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rec = pd.concat([load_long(f"{a.res}/{u}_long.csv") for u in a.units])
    T = pd.concat([pd.read_csv(f"{a.res}/{u}_T_long.csv") for u in a.units])
    Tp = pd.concat([pd.read_csv(f"{a.res}/{u}_Tprime_long.csv") for u in a.units])
    r1, t, tp = matrix(rec, "R1"), matrix(T, "T"), matrix(Tp, "Tprime")
    V = {"T_registered": compare(r1, t, "T", a.out, a.family),
         "T_level_99.79": compare(r1, t, "T_99.79", a.out, a.family, level=1 - ALPHA / 24),
         "Tprime": compare(r1, tp, "Tprime", a.out, a.family)}
    if a.family == "F1":    # fold-1 test subjects only (no tuning overlap); fold index 0 in the files
        f1 = rec[rec.fold == 0].subject.unique()
        V["T_fold1_only"] = compare(r1.loc[r1.index.isin(f1)], t, "T_fold1", a.out, a.family)
    V["n_fail"] = dict(T=int(T.training_failure.sum()), Tprime=int(Tp.training_failure.sum()),
                       runs_T=int(len(T)), runs_Tprime=int(len(Tp)))
    with open(f"{a.out}/h4_values_{a.family}.json", "w") as f:
        json.dump(V, f, indent=2, default=str)
    for k, v in V.items():
        if isinstance(v, dict) and "definitive_changes" in v:
            print(k, "N", v["N"], "def.changes", v["definitive_changes"], "sign rev", v["sign_reversals"],
                  "all changes", v["n_category_changes"], "tau", round(v["kendall_tau"], 2))


if __name__ == "__main__":
    main()
