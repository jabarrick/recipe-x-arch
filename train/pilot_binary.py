#!/usr/bin/env python3
"""Binary pilot for recipe-x-arch (decision_rules.md §6) and, with other flags, the
4 x 4 training driver reused for the confirmatory runs.

Architecture and recipe are independent dimensions here; the pilot code in
CSOANet2026/csoanet-v2-deploy attached the recipe to the model.  Data loading,
folds, the R1 augmentation, weight constraints and the models are imported from
that folder unchanged (it is a dependency, not copied).

    python -m train.pilot_binary --csoanet-root ../csoanet2026_v2_deploy
    python -m train.pilot_binary --csoanet-root ../csoanet2026_v2_deploy --synthetic --epochs 3   # smoke test

Pilot settings (fixed in decision_rules §6): PhysioNet-MI left vs right (runs 4/8/12,
S088/S092/S100/S104 excluded), 4 architectures x 4 recipes, cross-subject 5-fold
(seed 0, 10 % of training subjects for validation), one seed (0), 1-40 Hz, 0-4 s,
fixed per-channel gain from training subjects.  Output:

    <out>/pilot_binary_long.csv      subject, dataset, fold, architecture, recipe, seed,
                                     bacc_f32, acc_f32, n_trials, train_acc, val_acc, epochs,
                                     training_failure, retried
    <out>/pilot_binary_s8.csv        subject, architecture, recipe, bacc, n_trials   (S8 input)

Recipes (Amendment 1, Table 3):
    R1  AdamW 1e-3, wd 1e-3, cosine to 1e-5; mixup 0.2 + label smoothing 0.1 + noise +
        shift + crop + channel dropout; early stopping on val accuracy, patience 15
    R2  Adam 1e-3 constant; CE; no augmentation; all epochs; lowest val loss
    R3  Adam 1e-2 (eps 1e-7), x0.1 at 40 % and 80 %; batch 16; CE; no augmentation;
        all epochs; lowest val loss
    R4  R1 without augmentation (and without mixup / label smoothing): AdamW cosine,
        CE, early stopping on val accuracy, patience 15
Weight constraints belong to the architecture, not the recipe: EEGNet max-norm 1 / 0.25,
CSOANet-v2 max-norm 0.25 on all filters, MI-BMInet none, ShallowConvNet none.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = ["EEGNet", "ShallowConvNet", "MI-BMInet", "CSOANet-v2"]
RECIPES = ["R1", "R2", "R3", "R4"]
TRAIN_FAIL_BACC = 55.0          # decision_rules §3


# ----------------------------------------------------------------------------- dependency
def load_deploy(root: str):
    root = Path(root).expanduser().resolve()
    if not (root / "deploy" / "train_physionet_mi.py").exists():
        sys.exit(f"--csoanet-root {root} has no deploy/train_physionet_mi.py")
    sys.path.insert(0, str(root))
    import importlib
    tp = importlib.import_module("deploy.train_physionet_mi")
    bl = importlib.import_module("deploy.baselines")
    mb = importlib.import_module("deploy.mibminet")
    v2 = importlib.import_module("eeg_project.scripts.models.csoanet2026_v2")
    return tp, bl, mb, v2


# ----------------------------------------------------------------------------- models
class ShallowWrap(nn.Module):
    """braindecode ShallowFBCSPNet on (B, 1, C, T) input, logits out."""
    def __init__(self, ch, t, nc):
        super().__init__()
        from braindecode.models import ShallowFBCSPNet
        import inspect
        sig = inspect.signature(ShallowFBCSPNet.__init__).parameters
        kw = dict(n_chans=ch, n_outputs=nc, n_times=t, final_conv_length="auto")
        if "add_log_softmax" in sig:                 # braindecode 0.8-0.9: logits only if False
            kw["add_log_softmax"] = False
        self.net = ShallowFBCSPNet(**kw)
        for name in ("softmax", "logsoftmax"):       # defensive: strip any trailing log-softmax
            if hasattr(self.net, name):
                setattr(self.net, name, nn.Identity())

    def forward(self, x):
        y = self.net(x.squeeze(1))
        return y.flatten(1) if y.dim() > 2 else y


def build_model(arch, ch, t, nc, fs, deps):
    tp, bl, mb, v2 = deps
    if arch == "EEGNet":
        return bl.make_eegnet(ch, t, nc, fs=fs, drop=0.25)             # cross-subject dropout
    if arch == "ShallowConvNet":
        return ShallowWrap(ch, t, nc)
    if arch == "MI-BMInet":
        return mb.MIBMInet(ch, t, nc, fs=fs, nk=32, nf=64, p_dropout=0.2)   # registered N_k / N_f
    if arch == "CSOANet-v2":
        return v2.CSOANetV2(ch, t, nc, kt=tp._long(fs))                   # long kernel (0.5 s)
    raise ValueError(arch)


def constrain(arch, model, deps):
    tp, bl, mb, v2 = deps
    if arch == "EEGNet":
        bl.eegnet_max_norm(model)
    elif arch == "CSOANet-v2":
        v2.CSOANetV2.max_norm(model)
    # MI-BMInet, ShallowConvNet: none


# ----------------------------------------------------------------------------- recipes
def make_recipe(name, model, epochs, lr=1e-3):
    """Returns dict(opt, sched, batch, augment, smooth, select, patience)."""
    p = model.parameters()
    if name == "R1":
        opt = torch.optim.AdamW(p, lr=lr, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
        return dict(opt=opt, sched=sch, batch=None, augment=True, select="val_acc", patience=15)
    if name == "R4":
        opt = torch.optim.AdamW(p, lr=lr, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
        return dict(opt=opt, sched=sch, batch=None, augment=False, select="val_acc", patience=15)
    if name == "R2":
        return dict(opt=torch.optim.Adam(p, lr=1e-3), sched=None, batch=None, augment=False,
                    select="val_loss", patience=None)
    if name == "R3":
        opt = torch.optim.Adam(p, lr=1e-2, eps=1e-7)
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.4 * epochs), int(0.8 * epochs)], 0.1)
        return dict(opt=opt, sched=sch, batch=16, augment=False, select="val_loss", patience=None)
    raise ValueError(name)


# ----------------------------------------------------------------------------- training
@torch.no_grad()
def logits_of(model, X, dev, bs=256):
    model.eval()
    return torch.cat([model(X[i:i + bs].to(dev)).float().cpu() for i in range(0, len(X), bs)])


def bacc(y_true, y_pred, nc):
    rec = [(y_pred[y_true == c] == c).mean() for c in range(nc) if (y_true == c).any()]
    return 100.0 * float(np.mean(rec))


def train_one(arch, recipe_name, Xtr, Ytr, Xva, Yva, nc, epochs, batch_default, dev, seed, deps, amp=True):
    """Returns (model, info). info: val_acc, val_loss, epochs_run, train_bacc_unaug, nonfinite."""
    torch.manual_seed(seed); np.random.seed(seed % 2**32)
    ch, t = Xtr.shape[2], Xtr.shape[3]
    model = build_model(arch, ch, t, nc, FS, deps).to(dev)
    rc = make_recipe(recipe_name, model, epochs)
    batch = rc["batch"] or batch_default
    use_amp = dev.type == "cuda" and amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_metric = -math.inf if rc["select"] == "val_acc" else math.inf
    best_state, wait, best_ep, nonfinite = None, 0, 0, False
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Ytr))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            xb, yb = Xtr[idx].to(dev), Ytr[idx].to(dev)
            if rc["augment"]:
                xb, yb = deps[0].augment(xb, yb, nc)                 # mixup + smoothing inside
            else:
                yb = F.one_hot(yb, nc).float()
            rc["opt"].zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = -(yb * F.log_softmax(model(xb).float(), 1)).sum(1).mean()
            if not torch.isfinite(loss):
                nonfinite = True
                break
            scaler.scale(loss).backward(); scaler.step(rc["opt"]); scaler.update()
            constrain(arch, model, deps)
        if nonfinite:
            break
        if rc["sched"] is not None:
            rc["sched"].step()
        lg = logits_of(model, Xva, dev)
        v_loss = float(F.cross_entropy(lg, Yva)); v_acc = float((lg.argmax(1) == Yva).float().mean())
        metric = v_acc if rc["select"] == "val_acc" else v_loss
        better = metric > best_metric if rc["select"] == "val_acc" else metric < best_metric
        if better:
            best_metric, best_ep, wait = metric, ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_val = (v_acc, v_loss)
        else:
            wait += 1
            if rc["patience"] is not None and wait >= rc["patience"]:
                break
    if best_state is None:                                           # diverged before the first epoch finished
        return model, dict(val_acc=float("nan"), val_loss=float("nan"), epochs_run=ep + 1,
                           train_bacc=float("nan"), nonfinite=True)
    model.load_state_dict(best_state)
    step = max(1, len(Ytr) // 2000)
    pred_tr = logits_of(model, Xtr[::step], dev).argmax(1).numpy()
    tb = bacc(Ytr[::step].numpy(), pred_tr, nc)                      # un-augmented, selected checkpoint
    return model, dict(val_acc=best_val[0], val_loss=best_val[1], epochs_run=ep + 1, train_bacc=tb,
                       nonfinite=nonfinite)


# ----------------------------------------------------------------------------- main
FS = 160.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csoanet-root", required=True, help="folder containing deploy/ and eeg_project/")
    ap.add_argument("--out", default="results/pilot_binary")
    ap.add_argument("--data-root", default="~/mne_data")
    ap.add_argument("--archs", nargs="+", default=ARCHS, choices=ARCHS)
    ap.add_argument("--recipes", nargs="+", default=RECIPES, choices=RECIPES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    deps = load_deploy(a.csoanet_root); tp = deps[0]
    global FS; FS = tp.SFREQ
    dev = torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    nc = 2

    # data (same loader and cache layout as the pilot script)
    cache = out / "cache_lr_1-40Hz_0-4s.npz"
    if a.synthetic:
        X, y, g, _ = tp.load_synthetic(nc, n_subjects=a.max_subjects or 20)
    elif cache.exists():
        d = np.load(cache); X, y, g = d["X"], d["y"], d["g"]
    else:
        print("Loading PhysioNet-MI left/right (runs 4, 8, 12) ...")
        X, y, g = tp.load_physionet(Path(a.data_root).expanduser(), "lr", 1.0, 40.0, 0.0, 4.0, a.max_subjects)
        np.savez(cache, X=X, y=y, g=g)
    sess = np.zeros(len(y), dtype=np.int64)
    print(f"{len(y)} trials, {len(np.unique(g))} subjects, {X.shape[1]} ch x {X.shape[2]} samples | {dev}")

    long_csv = out / "pilot_binary_long.csv"
    rows = [] if (a.fresh or not long_csv.exists()) else pd.read_csv(long_csv).to_dict("records")
    done = {(r["architecture"], r["recipe"], r["fold"], r["seed"]) for r in rows}
    (out / "settings.json").write_text(json.dumps(vars(a), indent=1, default=str))
    Y = torch.from_numpy(y)

    for fold, (tr, va, te, te_s) in enumerate(tp.make_folds(g, sess, "cross-subject", a.folds, a.val_frac, a.split_seed)):
        Xn = torch.from_numpy(tp.normalise(X, tr, "fixed")).unsqueeze(1)
        for arch in a.archs:
            for rec in a.recipes:
                for seed in a.seeds:
                    if (arch, rec, fold, seed) in done:
                        continue
                    t0 = time.time()
                    run_seed = seed * 1000 + fold
                    model, info = train_one(arch, rec, Xn[tr], Y[tr], Xn[va], Y[va], nc, a.epochs, a.batch,
                                            dev, run_seed, deps, amp=not a.no_amp)
                    retried = False
                    if info["nonfinite"]:                            # registered retry: seed + 100
                        model, info = train_one(arch, rec, Xn[tr], Y[tr], Xn[va], Y[va], nc, a.epochs, a.batch,
                                                dev, run_seed + 100, deps, amp=not a.no_amp)
                        retried = True
                    failure = int(info["nonfinite"] or (not math.isnan(info["train_bacc"]) and info["train_bacc"] <= TRAIN_FAIL_BACC))
                    pred = logits_of(model.cpu(), Xn[te], torch.device("cpu")).argmax(1).numpy()
                    yt, gt = y[te], g[te]
                    for s in te_s:
                        k = gt == s
                        rows.append(dict(subject=f"S{int(s):03d}", dataset="PhysioNet-MI", fold=fold,
                                         architecture=arch, recipe=rec, seed=seed,
                                         bacc_f32=bacc(yt[k], pred[k], nc), acc_f32=100.0 * float((pred[k] == yt[k]).mean()),
                                         n_trials=int(k.sum()), train_acc=info["train_bacc"], val_acc=100 * info["val_acc"],
                                         epochs=info["epochs_run"], training_failure=failure, retried=int(retried)))
                    print(f"fold {fold + 1}/{a.folds} {arch:<14} {rec} seed {seed}: train {info['train_bacc']:.1f} "
                          f"val {100 * info['val_acc']:.1f} test {bacc(yt, pred, nc):.1f} | {info['epochs_run']} ep"
                          f"{' FAIL' if failure else ''}{' retried' if retried else ''} {time.time() - t0:.0f}s")
                    pd.DataFrame(rows).to_csv(long_csv, index=False)

    df = pd.DataFrame(rows)
    s8 = (df.groupby(["subject", "architecture", "recipe"], as_index=False)
            .agg(bacc=("bacc_f32", "mean"), n_trials=("n_trials", "first")))
    s8.to_csv(out / "pilot_binary_s8.csv", index=False)
    cell = df.pivot_table(index="architecture", columns="recipe", values="bacc_f32", aggfunc="mean").loc[a.archs, a.recipes]
    print("\nCell means (subject-level balanced accuracy, %):\n", cell.round(1).to_string())
    print(f"training failures: {int(df.training_failure.sum())} / {len(df)} rows; S8 input -> {out / 'pilot_binary_s8.csv'}")


if __name__ == "__main__":
    main()
