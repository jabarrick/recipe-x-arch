#!/usr/bin/env python3
"""Condition T' search (Amendment 1 §3.4; Supplementary S1).

For each architecture, the same 24 configurations (train/configurable.py, CONFIG_SEED) are
trained on the training subjects of fold 1 of the PhysioNet-MI left/right split and scored on
fold 1's validation subjects as the mean validation balanced accuracy over seeds 0 and 1.
The highest-scoring configuration per architecture is T' and is written into Amendment 1 §3.10;
it is then applied unchanged to every confirmatory cell.  No confirmatory data are touched.

Fold 1 here is the first fold of tp.make_folds(..., "cross-subject", 5, 0.1, split_seed=0),
the same split as the binary pilot.  Fold 1's test subjects are not used.

    python -m train.tprime_search --csoanet-root ..\\csoanet2026_v2_deploy --out results\\tprime
    python -m train.tprime_search --csoanet-root ... --synthetic --epochs 2 --max-subjects 10 --n-configs 3   # smoke test

Resumable; one row per (architecture, config, seed) in tprime_runs.csv.
Tie-break: higher score, then lower config id.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from train.pilot_binary import ARCHS, load_deploy, build_model, constrain
from train import configurable as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csoanet-root", required=True)
    ap.add_argument("--out", default="results/tprime")
    ap.add_argument("--cache", default="results/pilot_binary/cache_lr_1-40Hz_0-4s.npz",
                    help="PhysioNet left/right cache written by the binary pilot")
    ap.add_argument("--data-root", default="~/mne_data")
    ap.add_argument("--archs", nargs="+", default=ARCHS, choices=ARCHS)
    ap.add_argument("--n-configs", type=int, default=C.N_CONFIGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(C.SCORING_SEEDS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    deps = load_deploy(a.csoanet_root); tp = deps[0]
    import train.pilot_binary as PB
    PB.FS = tp.SFREQ
    dev = torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    nc = 2

    if a.synthetic:
        X, y, g, _ = tp.load_synthetic(nc, n_subjects=a.max_subjects or 20)
    elif Path(a.cache).exists():
        d = np.load(a.cache); X, y, g = d["X"], d["y"], d["g"]
    else:
        X, y, g = tp.load_physionet(Path(a.data_root).expanduser(), "lr", 1.0, 40.0, 0.0, 4.0, a.max_subjects)
    sess = np.zeros(len(y), dtype=np.int64)
    tr, va, te, te_s = next(iter(tp.make_folds(g, sess, "cross-subject", 5, a.val_frac, a.split_seed)))
    n_tr, n_va = len(np.unique(g[tr])), len(np.unique(g[va]))
    print(f"T' search: fold 1 -> {n_tr} training subjects, {n_va} validation subjects ({len(va)} trials); "
          f"test subjects of fold 1 not used | {dev}")

    configs = C.sample_configs(a.n_configs)
    (out / "tprime_configs.json").write_text(json.dumps(configs, indent=1))
    (out / "settings.json").write_text(json.dumps(vars(a), indent=1, default=str))

    Xn = torch.from_numpy(tp.normalise(X, tr, "fixed")).unsqueeze(1)
    Y = torch.from_numpy(y)
    runs_csv = out / "tprime_runs.csv"
    rows = pd.read_csv(runs_csv).to_dict("records") if runs_csv.exists() else []
    done = {(r["architecture"], r["config_id"], r["seed"]) for r in rows}

    ch, t = X.shape[1], X.shape[2]
    for arch in a.archs:
        for cfg in configs:
            for seed in a.seeds:
                if (arch, cfg["id"], seed) in done:
                    continue
                t0 = time.time()
                model = build_model(arch, ch, t, nc, tp.SFREQ, deps)
                model, info = C.train_cfg(model, lambda m: constrain(arch, m, deps), cfg,
                                          Xn[tr], Y[tr], Xn[va], Y[va], nc, a.epochs, dev,
                                          seed=seed * 1000, amp=not a.no_amp)
                rows.append(dict(architecture=arch, config_id=cfg["id"], seed=seed, **{k: info[k] for k in
                                 ("val_bacc", "val_loss", "best_epoch", "epochs_run", "train_bacc", "nonfinite")}))
                print(f"{arch:<14} cfg {cfg['id']:>2} seed {seed}: val bacc {info['val_bacc']:.1f} "
                      f"train {info['train_bacc']:.1f} | best ep {info['best_epoch']}/{info['epochs_run']} "
                      f"{time.time() - t0:.0f}s")
                pd.DataFrame(rows).to_csv(runs_csv, index=False)

    df = pd.DataFrame(rows)
    score = (df.groupby(["architecture", "config_id"], as_index=False)
               .agg(score=("val_bacc", "mean"), n_seeds=("seed", "nunique"), any_nonfinite=("nonfinite", "max")))
    score.to_csv(out / "tprime_scores.csv", index=False)
    chosen = {}
    for arch in a.archs:
        s = score[score.architecture == arch].copy()
        s["score"] = s["score"].fillna(-math.inf)
        best = s.sort_values(["score", "config_id"], ascending=[False, True]).iloc[0]
        cfg = next(c for c in configs if c["id"] == int(best.config_id))
        chosen[arch] = dict(config=cfg, score=float(best.score),
                            runner_up=float(s.sort_values("score", ascending=False).iloc[1].score) if len(s) > 1 else None)
    (out / "tprime_selected.json").write_text(json.dumps(chosen, indent=1))
    print("\nSelected T' configuration per architecture:")
    for arch, c in chosen.items():
        cf = c["config"]
        augs = "+".join(k for k in C.AUGS if cf[k]) or "none"
        print(f"  {arch:<14} cfg {cf['id']:>2}  score {c['score']:.2f}  | {cf['optimizer']} lr {cf['lr']:.2e} "
              f"{cf['schedule']} batch {cf['batch']} aug {augs} ls {cf['label_smoothing']} {cf['select']}")


if __name__ == "__main__":
    main()
