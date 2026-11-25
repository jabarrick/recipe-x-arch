#!/usr/bin/env python3
"""Preregistered analysis for the independent test (OSF 8st47, Amendment 3). Frozen before download.

Input: <res>/IND_long.csv from train/independent_dreyer.py.
Unit: per-subject seed-averaged float32 balanced accuracy. Cluster: test fold (10 clusters).
Contrasts (per subject, percentage points):
  HA  crop effect on the gap   d_A = (EEGNet - CSOANet-v2)[R1nc] - (EEGNet - CSOANet-v2)[R1]      predicted > 0
  HB  R2 interaction           d_B = (ShallowConvNet - CSOANet-v2)[R1nc] - (ShallowConvNet - CSOANet-v2)[R2]  predicted > 0
Test: mean of d with a cluster-robust (CR1) t statistic; p from a wild cluster bootstrap with the
null imposed (Webb six-point weights, 9,999 draws, seed 20261126); two-sided; Holm over {HA, HB} at 0.05.
Supported: Holm-adjusted p < 0.05 and mean > 0. Contradicted: Holm-adjusted p < 0.05 and mean < 0.
Otherwise inconclusive; if the 90 % bootstrap-t CI lies within (-2, 2) the effect is reported as
smaller than the SESOI.
Secondary (descriptive, no decision): R1nc - R1 per architecture; the same contrasts with R2va in
place of R2; the same on the empirical-logit scale.
Usage: python -m analysis.independent_analysis --res results/independent --out docs/independent
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd

WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
B = 9999


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
        C = {"HA": (gap("EEGNet", "CSOANet-v2", "R1nc") - gap("EEGNet", "CSOANet-v2", "R1")).to_numpy(),
             "HB": (gap("ShallowConvNet", "CSOANet-v2", "R1nc") - gap("ShallowConvNet", "CSOANet-v2", "R2")).to_numpy()}
        if ("ShallowConvNet", "R2va") in w:
            C["HB_with_R2va"] = (gap("ShallowConvNet", "CSOANet-v2", "R1nc") - gap("ShallowConvNet", "CSOANet-v2", "R2va")).to_numpy()
        for arch in ("EEGNet", "ShallowConvNet", "MI-BMInet", "CSOANet-v2"):
            C[f"R1nc-R1 {arch}"] = (w[(arch, "R1nc")] - w[(arch, "R1")]).to_numpy()
        V[scale] = {k: test(v, cl, rng) for k, v in C.items()}
    # Holm over the two registered hypotheses (percentage scale)
    ps = sorted((V["y"][h]["p"], h) for h in ("HA", "HB"))
    adj = {}; run = 0.0
    for i, (p, h) in enumerate(ps):
        run = max(run, min(1.0, p * (2 - i))); adj[h] = run
    V["decision"] = {}
    for h in ("HA", "HB"):
        r = V["y"][h]; pa = adj[h]
        if pa < .05 and r["mean"] > 0: dec = "supported"
        elif pa < .05 and r["mean"] < 0: dec = "contradicted"
        elif -2 < r["ci90"][0] and r["ci90"][1] < 2: dec = "inconclusive; effect smaller than the 2-point SESOI"
        else: dec = "inconclusive"
        V["decision"][h] = dict(p_holm=pa, decision=dec)
    json.dump(V, open(f"{o.out}/independent_results.json", "w"), indent=2)
    print(json.dumps(V["decision"], indent=1))


if __name__ == "__main__":
    main()
