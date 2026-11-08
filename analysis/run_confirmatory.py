#!/usr/bin/env python3
"""Confirmatory analysis driver. Reads the long results table, runs the registered
and Amendment-1 analyses for one family, writes results/section4_values.json and
CSV tables. Decision logic lives in the imported modules; this file only sequences.

Usage:
  python -m analysis.run_confirmatory results/F1_long.csv --family F1 --out results/
  python -m analysis.run_confirmatory --demo --out results_demo/
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from .common import ARCHS, RECIPES, load_long, seed_average, to_cube
from . import mixed_model as mm
from . import home_recipe as hr
from . import h3_classify as h3


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def analyse_family(df: pd.DataFrame, out_dir: str, tag: str, n_boot: int = 2000) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    V = {}

    # ---- primary: seed-averaged float32, failures kept -------------------
    avg = seed_average(df, "bacc_f32")
    y, subjects, ds = to_cube(avg)
    V["4.1"] = dict(N=len(subjects), N_by_dataset=pd.Series(ds).value_counts().to_dict(),
                    N_runs=int(len(df)), n_fail=int(df.get("training_failure", pd.Series(0)).sum()),
                    mean=float(np.nanmean(y)), min=float(np.nanmin(np.nanmean(y, 0))),
                    max=float(np.nanmax(np.nanmean(y, 0))))
    if "train_acc" in df:
        V["4.1"].update(tr_min=float(df.train_acc.min()), tr_max=float(df.train_acc.max()),
                        va_min=float(df.val_acc.min()), va_max=float(df.val_acc.max()))

    fam = mm.decide_family(avg, y)
    V["4.2"] = dict(H1=fam["H1"], H2=fam["H2"], variance_shares=fam["variance_shares"],
                    recipe_range=fam["recipe_range"], converged=fam["converged"])
    fam["cell_means"].to_csv(f"{out_dir}/{tag}_cell_means.csv")
    h2b = fam["H2b"]; V["4.3"] = {k: v for k, v in h2b.items() if k != "c"}

    home = hr.run(y, avg_long=avg, n_boot=n_boot)
    home_noR4 = hr.run(y, drop_R4=True, n_boot=n_boot)
    V["4.3b"] = dict(h_cellmeans=home["h_cellmeans"], h_model=home.get("h_model"),
                     tests=home["tests"], ranges=home["ranges"], mechanism=home["mechanism"],
                     h_noR4=home_noR4["h_cellmeans"], mechanism_noR4=home_noR4["mechanism"],
                     n_dropped=home["n_dropped"])
    home["tests"].to_csv(f"{out_dir}/{tag}_home_tests.csv", index=False)
    home["ranges"].to_csv(f"{out_dir}/{tag}_residual_ranges.csv", index=False)

    H3 = h3.run(y)
    V["4.4"] = {k: v for k, v in H3.items() if k not in ("intervals", "category_matrix")}
    H3["intervals"].to_csv(f"{out_dir}/{tag}_h3_intervals_f32.csv", index=False)
    H3["category_matrix"].to_csv(f"{out_dir}/{tag}_h3_categories_f32.csv")

    # ---- int8 sensitivity ------------------------------------------------
    if "bacc_int8" in df:
        avg8 = seed_average(df, "bacc_int8"); y8, _, _ = to_cube(avg8)
        fam8 = mm.decide_family(avg8, y8); H3_8 = h3.run(y8)
        same = (fam8["H1"]["supported"] == fam["H1"]["supported"] and
                fam8["H2"]["supported"] == fam["H2"]["supported"] and
                fam8["H2b"]["supported"] == fam["H2b"]["supported"])
        cat_changes = int((H3_8["category_matrix"].to_numpy() != H3["category_matrix"].to_numpy()).sum())
        V["4.6_int8"] = dict(same_decisions=bool(same), H1=fam8["H1"], H2=fam8["H2"],
                             H2b={k: v for k, v in fam8["H2b"].items() if k != "c"},
                             h3_outcome=H3_8["outcome"], category_changes_f32_to_int8=cat_changes,
                             deployment_module="D1" if cat_changes == 0 else "D2")
        H3_8["intervals"].to_csv(f"{out_dir}/{tag}_h3_intervals_int8.csv", index=False)

    # ---- training-failure exclusion sensitivity --------------------------
    if "training_failure" in df and df["training_failure"].any():
        avg_x = seed_average(df, "bacc_f32", drop_failures=True); yx, subj_x, _ = to_cube(avg_x)
        frac_incomplete = float(np.isnan(yx).any(axis=(1, 2)).mean())
        home_x = hr.run(yx, avg_long=avg_x, n_boot=n_boot)
        path = "model" if frac_incomplete > 0.10 else "subject_drop"
        V["4.6_failures_excluded"] = dict(frac_subjects_incomplete=frac_incomplete, path_used=path,
                                          h=home_x["h_model"] if path == "model" else home_x["h_cellmeans"],
                                          h3_outcome=h3.run(yx)["outcome"])

    # ---- random slopes / bootstrap --------------------------------------
    try:
        rs = mm.fit_family(avg, random_slopes=True)
        V["4.6_random_slopes"] = dict(H1=rs["H1"], H2=rs["H2"], converged=rs["converged"])
    except Exception as e:  # convergence failures are reported, not hidden
        V["4.6_random_slopes"] = dict(error=str(e))
    bs = mm.cluster_bootstrap(avg, y, n_boot=min(n_boot, 500))
    V["4.6_bootstrap"] = {c: dict(lo=float(bs[c].quantile(.025)), hi=float(bs[c].quantile(.975)))
                          for c in bs.columns}

    with open(f"{out_dir}/section4_values_{tag}.json", "w") as f:
        json.dump(_json_safe(V), f, indent=2)
    return V


def make_demo_long(path: str, S=106, seeds=(0, 1, 2), seed=0):
    rng = np.random.default_rng(seed)
    alpha = np.array([0, 1.0, 2.0, 3.0]); rho = np.array([0, 1.0, 0.5, -0.5])
    inter = np.zeros((4, 4)); inter[0, 1] = inter[3, 0] = 1.0; inter[0, 0] = inter[3, 1] = -1.0
    u = rng.normal(0, 7, S); rows = []
    for s in range(S):
        for i, a in enumerate(ARCHS):
            for j, r in enumerate(RECIPES):
                for sd in seeds:
                    mu = 68 + alpha[i] + rho[j] + inter[i, j] + u[s]
                    f32 = mu + rng.normal(0, 3.5)
                    rows.append(dict(subject=f"S{s:03d}", dataset="Lee2019" if s < 54 else "Cho2017",
                                     fold=s % 5, architecture=a, recipe=r, seed=sd,
                                     bacc_f32=f32, bacc_int8=f32 - abs(rng.normal(0.3, 0.3)),
                                     train_acc=mu + 15, val_acc=mu + 2,
                                     training_failure=int(rng.random() < 0.003)))
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("long_csv", nargs="?")
    ap.add_argument("--family", default="F1")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.demo:
        a.long_csv = f"{a.out}/demo_long.csv"; make_demo_long(a.long_csv)
    df = load_long(a.long_csv)
    V = analyse_family(df, a.out, a.family, n_boot=a.n_boot)
    print(json.dumps({"H1": V["4.2"]["H1"], "H2": V["4.2"]["H2"],
                      "H2b": {k: V["4.3"][k] for k in ("mean", "p_holm", "supported")},
                      "mechanism": V["4.3b"]["mechanism"], "H3": V["4.4"]["outcome"],
                      "int8": V.get("4.6_int8", {}).get("deployment_module")}, indent=2, default=str))


if __name__ == "__main__":
    main()
