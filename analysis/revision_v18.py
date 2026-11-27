#!/usr/bin/env python3
"""Follow-up runs requested by the sixth review (post hoc; S4 row 21): R1ct, R1ns, R2va, ShallowConvNet-MN.
Contrasts per subject (percentage points), wild cluster bootstrap (Webb, 9,999 draws, 10 dataset x fold clusters).
Usage: python -m analysis.revision_v18 --res results/confirmatory --nc docs/exploratory_r1nocrop --rev6 results/exploratory_rev6 --out <dir>
"""
import argparse, json, os
import numpy as np, pandas as pd
from .revision_v17 import wild

A = ["EEGNet", "ShallowConvNet", "MI-BMInet", "CSOANet-v2"]


def main():
    ap = argparse.ArgumentParser()
    for k in ("res", "nc", "rev6", "out"): ap.add_argument("--" + k, required=True)
    o = ap.parse_args(); os.makedirs(o.out, exist_ok=True)
    d = pd.concat([pd.read_csv(f"{o.res}/{u}_long.csv") for u in ("F1a", "F1b")] +
                  [pd.read_csv(f"{o.nc}/{u}_R1nc_long.csv") for u in ("F1a", "F1b")] +
                  [pd.read_csv(f"{o.rev6}/{u}_long.csv") for u in ("F1a", "F1b")])
    d.loc[d.architecture == "ShallowConvNet-MN", "architecture"] = "ShallowConvNet-MN"
    runs = d.drop_duplicates(["dataset", "architecture", "recipe", "fold", "seed"])
    a = d.groupby(["subject", "dataset", "fold", "architecture", "recipe"]).bacc_f32.mean().reset_index()
    a["cl"] = a.dataset + ":" + a.fold.astype(str)
    w = a.pivot_table(index="subject", columns=["architecture", "recipe"], values="bacc_f32")
    cl = a.drop_duplicates("subject").set_index("subject").cl.reindex(w.index).to_numpy()
    W = lambda x: wild(np.asarray(x, float), cl)
    g = lambda p, q, r: w[(p, r)] - w[(q, r)]
    V = {"cell_means": {f"{k[0]}|{k[1]}": round(float(v), 2) for k, v in w.mean().items()},
         "flagged_runs_rev6": int(runs[runs.recipe.isin(["R1ct", "R1ns", "R2va"]) | (runs.architecture == "ShallowConvNet-MN")].training_failure.sum()),
         "median_best_epoch": {f"{k[0]}|{k[1]}": float(v) for k, v in runs.groupby(["architecture", "recipe"]).best_epoch.median().items()},
         "contrasts": {}}
    C = V["contrasts"]
    for ar in A:
        C[f"R1ct-R1 {ar}"] = W(w[(ar, "R1ct")] - w[(ar, "R1")])
        C[f"R1ns-R1 {ar}"] = W(w[(ar, "R1ns")] - w[(ar, "R1")])
        C[f"R1nc-R1ct {ar}"] = W(w[(ar, "R1nc")] - w[(ar, "R1ct")])
        C[f"R2va-R2 {ar}"] = W(w[(ar, "R2va")] - w[(ar, "R2")])
    C["gap EEGNet-CSOANet-v2: R1ct vs R1"] = W(g("EEGNet", "CSOANet-v2", "R1ct") - g("EEGNet", "CSOANet-v2", "R1"))
    C["gap EEGNet-CSOANet-v2: R1nc vs R1ct"] = W(g("EEGNet", "CSOANet-v2", "R1nc") - g("EEGNet", "CSOANet-v2", "R1ct"))
    C["gap EEGNet-CSOANet-v2: R1ns vs R1"] = W(g("EEGNet", "CSOANet-v2", "R1ns") - g("EEGNet", "CSOANet-v2", "R1"))
    for b in ("CSOANet-v2", "MI-BMInet"):
        C[f"ShallowConvNet-{b}: R4 vs R2"] = W(g("ShallowConvNet", b, "R4") - g("ShallowConvNet", b, "R2"))
        C[f"ShallowConvNet-{b}: R4 vs R2va"] = W(g("ShallowConvNet", b, "R4") - g("ShallowConvNet", b, "R2va"))
        C[f"ShallowConvNet-MN-{b}: R4 vs R2"] = W(g("ShallowConvNet-MN", b, "R4") - g("ShallowConvNet-MN", b, "R2"))
    for r in ("R2", "R4"):
        C[f"ShallowConvNet-MN - ShallowConvNet under {r}"] = W(w[("ShallowConvNet-MN", r)] - w[("ShallowConvNet", r)])
    C["gap EEGNet-CSOANet-v2: R2va vs R1nc"] = W(g("EEGNet", "CSOANet-v2", "R2va") - g("EEGNet", "CSOANet-v2", "R1nc"))
    json.dump(V, open(f"{o.out}/revision_v18.json", "w"), indent=2)
    print(json.dumps(C, indent=1))


if __name__ == "__main__":
    main()
