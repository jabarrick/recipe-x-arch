#!/usr/bin/env python3
"""Baseline reproduction on BCI IV 2a (decision_rules.md §5; Amendment 1 §2).

Each baseline is our study implementation, trained with the protocol of its reference figure,
and compared with that figure under a ±3-point tolerance fixed before any run.

  EEGNet-8,2      ref 60.46 %  MOABB benchmark, BNCI2014_001, 4 classes, within-session 5-fold
  ShallowConvNet  ref 72.47 %  same
  MI-BMInet       ref 76.03 %  Wang et al. 2024, session 1 -> session 2

MOABB protocol, reproduced from the v1.1.2 pipeline files (pipelines/Keras_EEGNet_8_2.yml,
Keras_ShallowConvNet.yml) and moabb.pipelines.{features,deep_learning}:
  - MotorImagery(n_classes=4) defaults: band-pass 8-32 Hz, dataset interval [2, 6] s -> 0-4 s after cue
  - Resampler_Epoch: 128 Hz for EEGNet, 250 Hz for ShallowConvNet
  - StandardScaler_Epoch: sklearn StandardScaler().fit_transform(X[i]) per trial, i.e. each TIME
    SAMPLE z-scored across channels (ddof 0) -- reproduced as is
  - WithinSessionEvaluation: StratifiedKFold(5, shuffle=True, random_state=42) per subject x session;
    folds averaged within session; score = mean over subject x session
  - Keras fit: Adam lr 1e-3 (eps 1e-7), batch 64, <= 300 epochs, validation_split 0.2 = LAST 20 %
    of the training fold in index order; EarlyStopping(val_loss, patience 75), restore_best_weights
    False -> final weights; ReduceLROnPlateau(val_loss, patience 75, factor 0.5, min_delta 1e-4)
  - metric: accuracy
Wang et al. 2024 protocol (Tables II-III): session 1 train, session 2 test; 0-3 s after cue; data as
provided (0.5-100 Hz); 250 Hz; N_k = 32, N_f = 64; Adam lr 1e-3 eps 1e-7, 500 epochs, batch 32, CE;
final-epoch weights; dropout 0.5 (authors' within-subject reference configuration). The paper removes
artefact-marked trials; MOABB does not expose those marks, so all 288 trials are used (listed in S7).
Normalisation for MI-BMInet: fixed per-channel gain from session 1 (our pipeline; not stated in the paper).

Differences of model definition from the reference implementations (constraints, kernel rounding) are
NOT changed: the point is to verify the implementation used in the study.  They are listed in S7.

    python -m train.baseline_repro --csoanet-root ..\\csoanet2026_v2_deploy --out results\\baseline_repro
    python -m train.baseline_repro --csoanet-root ... --fake --epochs 3 --seeds 0      # smoke test (MOABB FakeDataset)
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from train.pilot_binary import load_deploy, constrain

REFERENCE = {"EEGNet": 60.46, "ShallowConvNet": 72.47, "MI-BMInet": 76.03}
TOLERANCE = 3.0
CLASSES = ["left_hand", "right_hand", "feet", "tongue"]


# ------------------------------------------------------------------ data
def get_dataset(fake):
    if fake:
        from moabb.datasets.fake import FakeDataset
        return FakeDataset(event_list=CLASSES, n_sessions=2, n_runs=1, n_subjects=2, paradigm="imagery")
    try:
        from moabb.datasets import BNCI2014_001 as D
    except ImportError:
        from moabb.datasets import BNCI2014001 as D
    return D()


def load(dataset, fmin, fmax, tmin, tmax, resample=None):
    import mne
    from moabb.paradigms import MotorImagery
    mne.set_log_level("ERROR")
    kw = dict(n_classes=4, events=CLASSES, tmin=tmin, tmax=tmax)
    if fmin is not None:
        kw.update(fmin=fmin, fmax=fmax)
    if resample:
        kw.update(resample=resample)
    X, labels, meta = MotorImagery(**kw).get_data(dataset=dataset, subjects=list(dataset.subject_list))
    y = np.array([CLASSES.index(l) for l in labels], dtype=np.int64)
    sess_names = sorted(meta["session"].unique())
    sess = meta["session"].map({n: i for i, n in enumerate(sess_names)}).to_numpy()
    return X.astype(np.float32), y, meta["subject"].to_numpy(), sess


def scaler_epoch(X):
    """moabb StandardScaler_Epoch: StandardScaler().fit_transform(X[i]) -> per time sample across channels."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd).astype(np.float32)


# ------------------------------------------------------------------ training loops
@torch.no_grad()
def predict(model, X, dev, bs=256):
    model.eval()
    return torch.cat([model(X[i:i + bs].to(dev)).argmax(1).cpu() for i in range(0, len(X), bs)]).numpy()


@torch.no_grad()
def loss_of(model, X, Y, dev, bs=256):
    model.eval()
    lg = torch.cat([model(X[i:i + bs].to(dev)).float().cpu() for i in range(0, len(X), bs)])
    return float(F.cross_entropy(lg, Y))


def fit_keras_like(model, cfn, Xtr, Ytr, Xva, Yva, dev, epochs=300, batch=64, lr=1e-3, patience=75):
    """Keras fit with EarlyStopping(val_loss, patience, no restore) + ReduceLROnPlateau(val_loss,
    patience, factor 0.5, min_delta 1e-4); final weights returned."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    es_best, es_wait = np.inf, 0
    rl_best, rl_wait = np.inf, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Ytr))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx].to(dev), Ytr[idx].to(dev)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb), yb).backward()
            opt.step(); cfn(model)
        vl = loss_of(model, Xva, Yva, dev)
        if vl < rl_best - 1e-4:
            rl_best, rl_wait = vl, 0
        else:
            rl_wait += 1
            if rl_wait >= patience:
                for gp in opt.param_groups:
                    gp["lr"] *= 0.5
                rl_wait = 0
        if vl < es_best:
            es_best, es_wait = vl, 0
        else:
            es_wait += 1
            if es_wait >= patience:
                return ep + 1
    return epochs


def fit_fixed(model, cfn, Xtr, Ytr, dev, epochs=500, batch=32, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Ytr))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            xb, yb = Xtr[idx].to(dev), Ytr[idx].to(dev)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb), yb).backward()
            opt.step(); cfn(model)
    return epochs


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csoanet-root", required=True)
    ap.add_argument("--out", default="results/baseline_repro")
    ap.add_argument("--archs", nargs="+", default=list(REFERENCE), choices=list(REFERENCE))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fake", action="store_true", help="MOABB FakeDataset, for a smoke test")
    a = ap.parse_args()

    deps = load_deploy(a.csoanet_root); tp, bl, mb, v2 = deps
    from train.shallow import ShallowConvNet
    dev = torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ds = get_dataset(a.fake)
    csv = out / "baseline_repro_per_unit.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["architecture"], r["seed"], r["subject"], r["session"], r["fold"]) for r in rows}

    for arch in a.archs:
        if arch in ("EEGNet", "ShallowConvNet"):
            fs = 128 if arch == "EEGNet" else 250
            X, y, g, sess = load(ds, None, None, 0.0, None, resample=fs)        # MOABB defaults: 8-32 Hz, 0-4 s
            X = scaler_epoch(X)
            from sklearn.model_selection import StratifiedKFold
            for seed in a.seeds:
                for s in np.unique(g):
                    for se in np.unique(sess[g == s]):
                        idx = np.flatnonzero((g == s) & (sess == se))
                        skf = StratifiedKFold(5, shuffle=True, random_state=42)
                        for fold, (tr_i, te_i) in enumerate(skf.split(idx, y[idx])):
                            if (arch, seed, int(s), int(se), fold) in done:
                                continue
                            tr, te = idx[tr_i], idx[te_i]                  # tr ascending (index order)
                            n_va = int(round(0.2 * len(tr)))
                            fit_i, va_i = tr[:-n_va], tr[-n_va:]          # Keras validation_split: last 20 %
                            Xt = torch.from_numpy(X).unsqueeze(1)
                            Yt = torch.from_numpy(y)
                            torch.manual_seed(seed * 1000 + fold); np.random.seed(seed * 1000 + fold)
                            ch, t = X.shape[1], X.shape[2]
                            model = (bl.make_eegnet(ch, t, 4, fs=fs, drop=0.5) if arch == "EEGNet"
                                     else ShallowConvNet(ch, t, 4)).to(dev)
                            t0 = time.time()
                            n_ep = fit_keras_like(model, lambda m: constrain(arch, m, deps), Xt[fit_i], Yt[fit_i],
                                                  Xt[va_i], Yt[va_i], dev, epochs=a.epochs or 300)
                            acc = 100.0 * float((predict(model, Xt[te], dev) == y[te]).mean())
                            rows.append(dict(architecture=arch, seed=seed, subject=int(s), session=int(se), fold=fold,
                                             acc=acc, n_test=len(te), epochs=n_ep, protocol="MOABB within-session 5-fold"))
                            print(f"{arch:<14} seed {seed} S{int(s):02d} sess {int(se)} fold {fold}: {acc:.1f} "
                                  f"({n_ep} ep, {time.time() - t0:.0f}s)")
                            pd.DataFrame(rows).to_csv(csv, index=False)
        else:   # MI-BMInet, Wang et al. protocol
            X, y, g, sess = load(ds, 0.5, 40.0 if a.fake else 100.0, 0.0, 3.0)   # FakeDataset is 128 Hz
            for seed in a.seeds:
                for s in np.unique(g):
                    if (arch, seed, int(s), 1, 0) in done:
                        continue
                    tr = np.flatnonzero((g == s) & (sess == 0)); te = np.flatnonzero((g == s) & (sess == 1))
                    gain = X[tr].std(axis=(0, 2), keepdims=True) + 1e-8
                    Xt = torch.from_numpy(X / gain).unsqueeze(1); Yt = torch.from_numpy(y)
                    torch.manual_seed(seed * 1000 + int(s)); np.random.seed(seed * 1000 + int(s))
                    model = mb.MIBMInet(X.shape[1], X.shape[2], 4, fs=250.0, nk=32, nf=64, p_dropout=0.5).to(dev)
                    t0 = time.time()
                    n_ep = fit_fixed(model, lambda m: None, Xt[tr], Yt[tr], dev, epochs=a.epochs or 500)
                    acc = 100.0 * float((predict(model, Xt[te], dev) == y[te]).mean())
                    rows.append(dict(architecture=arch, seed=seed, subject=int(s), session=1, fold=0, acc=acc,
                                     n_test=len(te), epochs=n_ep, protocol="session 1 -> session 2"))
                    print(f"{arch:<14} seed {seed} S{int(s):02d}: {acc:.1f} ({time.time() - t0:.0f}s)")
                    pd.DataFrame(rows).to_csv(csv, index=False)

    df = pd.DataFrame(rows)
    summary = {}
    for arch, d in df.groupby("architecture"):
        unit = d.groupby(["seed", "subject", "session"]).acc.mean()        # folds averaged within session
        per_seed = unit.groupby("seed").mean()                             # mean over subject x session
        m = float(per_seed.mean())
        summary[arch] = dict(reproduced=round(m, 2), sd_across_units=round(float(unit.groupby(["subject", "session"]).mean().std()), 2),
                             per_seed=[round(float(v), 2) for v in per_seed], reference=REFERENCE[arch],
                             difference=round(m - REFERENCE[arch], 2),
                             within_tolerance=bool(abs(m - REFERENCE[arch]) <= TOLERANCE), protocol=d.protocol.iloc[0])
    (out / "baseline_repro_summary.json").write_text(json.dumps(summary, indent=1))
    print("\n" + json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
