#!/usr/bin/env python3
"""Revision V1.6 (correction of a V1.5 analysis; S4 row 18). Reads result files only.

V1.5 tested each double-centred interaction residual among {R1nc, R2, R4} against the +/-2-point
SESOI. The SESOI was defined for pairwise accuracy differences; an interaction is a 2 x 2
difference-in-differences (DiD) built from four residuals, so the residual test used the wrong scale.
This script replaces it: every 2 x 2 DiD (6 architecture pairs x 3 recipe pairs = 18 contrasts;
6 independent df) with a random intercept per dataset x fold cluster (t on G-1 df), TOST against
+/-2 at the unadjusted 90 % level and at a Bonferroni-adjusted level (1 - 2*0.05/18).
No new training runs and no new model fits beyond these contrasts.
Usage: python -m analysis.revision_v16 --res results/confirmatory --nc docs/exploratory_r1nocrop --out <dir>
"""
from __future__ import annotations
import argparse, itertools, json, os
import numpy as np, pandas as pd
from .common import ARCHS, SESOI
from .revision_v11 import cluster_mean_ci

REC = ["R1nc", "R2", "R4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/confirmatory"); ap.add_argument("--nc", default="docs/exploratory_r1nocrop")
    ap.add_argument("--nc-names", nargs=2, default=["F1a_R1nc_long.csv", "F1b_R1nc_long.csv"])
    ap.add_argument("--out", required=True)
    o = ap.parse_args(); os.makedirs(o.out, exist_ok=True)
    f1 = pd.concat([pd.read_csv(f"{o.res}/{u}_long.csv") for u in ("F1a", "F1b")])
    d = pd.concat([f1] + [pd.read_csv(f"{o.nc}/{n}") for n in o.nc_names])
    d = d[d.recipe.isin(REC)]
    avg = d.groupby(["subject", "dataset", "fold", "architecture", "recipe"]).bacc_f32.mean().rename("y").reset_index()
    y = avg.pivot_table(index="subject", columns=["architecture", "recipe"], values="y")
    meta = avg.drop_duplicates("subject").set_index("subject").reindex(y.index)
    cl = (meta.dataset + ":" + meta.fold.astype(str)).to_numpy()
    k = 18; lvl_b = 1 - 2 * 0.05 / k
    rows = []
    for a, b in itertools.combinations(ARCHS, 2):
        for r, s in itertools.combinations(REC, 2):
            x = ((y[(a, s)] - y[(b, s)]) - (y[(a, r)] - y[(b, r)])).to_numpy()
            m, lo, hi, _ = cluster_mean_ci(x, cl, .95)
            _, l90, h90, _ = cluster_mean_ci(x, cl, .90)
            _, lb, hb, _ = cluster_mean_ci(x, cl, lvl_b)
            rows.append(dict(pair=f"{a} - {b}", recipes=f"{s} vs {r}", did=m, lo95=lo, hi95=hi,
                             tost90_lo=l90, tost90_hi=h90, eq_unadj=bool(l90 > -SESOI and h90 < SESOI),
                             tostB_lo=lb, tostB_hi=hb, eq_bonf=bool(lb > -SESOI and hb < SESOI),
                             nonzero95=bool(lo > 0 or hi < 0)))
    t = pd.DataFrame(rows)
    t["absdid"] = t.did.abs(); t = t.sort_values("absdid", ascending=False).drop(columns="absdid")
    t.to_csv(f"{o.out}/F1_did_tost_R1nc_R2_R4.csv", index=False)
    V = dict(n=len(t), n_eq_unadj=int(t.eq_unadj.sum()), n_eq_bonf=int(t.eq_bonf.sum()),
             n_nonzero95=int(t.nonzero95.sum()), max_abs_did=float(t.did.abs().max()),
             bonferroni_level=lvl_b, table=t.round(2).to_dict(orient="records"))
    json.dump(V, open(f"{o.out}/revision_v16.json", "w"), indent=2, default=float)
    print(t.round(2).to_string()); print({k_: v for k_, v in V.items() if k_ != "table"})


if __name__ == "__main__":
    main()
