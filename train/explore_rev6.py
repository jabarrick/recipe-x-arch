#!/usr/bin/env python3
"""Exploratory runs requested by the sixth internal review (post hoc; S4 row 20). F1 only.

The frozen confirmatory driver (train/confirmatory.py) is used unchanged; this file registers
extra recipes and one extra architecture variant by monkey-patching, points the data cache at
the confirmatory cache, and SAVES EVERY TRAINED MODEL (state_dict) under <out>/models/.

Recipes added (everything not listed is identical to the parent recipe):
  R1ct  R1, and validation + test windows are cropped too, with the same transform as in training
        at the fixed middle offset (keep 90 %, shift to start, zero-fill the end). Tests whether the
        harm is the train/test mismatch. (int8 calibration still uses uncropped training trials.)
  R1ns  R1, but the crop is applied in place: the 90 % segment stays where it is and the samples
        outside it are zeroed (no temporal shift). Separates the shift from the zero padding.
  R2va  R2, but the checkpoint is the epoch with the highest validation balanced accuracy over all
        epochs (no early stopping) instead of the lowest validation loss. Isolates R2's checkpoint rule.
Architecture added:
  ShallowConvNet-MN  the study's ShallowConvNet plus the MOABB/Keras max-norm constraints
        (conv filters 2.0, dense 0.5), as in the S7 baseline diagnostic.

Usage (repository root), e.g.:
  python -m train.explore_rev6 --csoanet-root ..\\csoanet2026_v2_deploy --unit F1a --out results\\exploratory_rev6 ^
         --archs EEGNet ShallowConvNet MI-BMInet CSOANet-v2 --recipes R2va
  python -m train.explore_rev6 ... --archs ShallowConvNet-MN --recipes R2 R4
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from train import configurable as C
from train import confirmatory as K
from train.baseline_repro import moabb_maxnorm

CROP = 0.9

# ------------------------------------------------------------------ recipes
C.REGISTERED["R1nc"] = dict(C.REGISTERED["R1"], crop=False)
C.REGISTERED["R1ct"] = dict(C.REGISTERED["R1"], test_crop=True)
C.REGISTERED["R1ns"] = dict(C.REGISTERED["R1"], crop_mode="inplace")
C.REGISTERED["R2va"] = dict(C.REGISTERED["R2"], select="val_acc_all")
K.RECIPES = ["R1ct", "R1ns", "R2va", "R1nc", "R1", "R2", "R3", "R4"]
K.ARCHS = list(K.ARCHS) + ["ShallowConvNet-MN"]

# ------------------------------------------------------------------ augmentation with an in-place crop option
_aug = C.augment


def augment(X, Y, nc, cfg, noise=0.02, shift=5, crop=CROP, ch_drop=0.1, mixup_alpha=0.2):
    if not (cfg.get("crop") and cfg.get("crop_mode") == "inplace"):
        return _aug(X, Y, nc, cfg, noise, shift, crop, ch_drop, mixup_alpha)
    B, _, Cc, T = X.shape
    if cfg["noise"]:
        X = X + torch.randn_like(X) * noise
    if cfg["shift"]:
        s = int(torch.randint(-shift, shift + 1, (1,)))
        if s:
            X = torch.roll(X, s, dims=-1)
    cl = int(T * crop)
    st = int(torch.randint(0, T - cl + 1, (1,)))
    out = torch.zeros_like(X)
    out[..., st:st + cl] = X[..., st:st + cl]           # same segment, kept in place
    X = out
    if cfg["chdrop"]:
        X = X * (torch.rand(B, 1, Cc, 1, device=X.device) > ch_drop).float()
    ls = cfg["label_smoothing"]
    Y1 = F.one_hot(Y, nc).float() * (1 - ls) + ls / nc
    if cfg["mixup"]:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().to(X.device)
        perm = torch.randperm(B, device=X.device)
        X, Y1 = lam * X + (1 - lam) * X[perm], lam * Y1 + (1 - lam) * Y1[perm]
    return X, Y1


C.augment = augment


def crop_fixed(X, crop=CROP):
    """The training crop at the fixed middle offset: keep 90 %, shift to the start, zero-fill the end."""
    T = X.shape[-1]; cl = int(T * crop); st = (T - cl) // 2
    out = torch.zeros_like(X); out[..., :cl] = X[..., st:st + cl]
    return out


# ------------------------------------------------------------------ checkpoint rule "val_acc_all" + model saving
_train = C.train_cfg
STATE = {"name": None, "dir": None}


def train_cfg(model, constrain_fn, cfg, Xtr, Ytr, Xva, Yva, nc, epochs, dev, seed, amp=True, patience=15):
    if cfg.get("select") == "val_acc_all":
        cfg = dict(cfg, select="val_acc_es"); patience = 10 ** 9
    model, info = _train(model, constrain_fn, cfg, Xtr, Ytr, Xva, Yva, nc, epochs, dev, seed, amp, patience)
    if STATE["name"] and STATE["dir"]:
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, STATE["dir"] / f"{STATE['name']}.pt")
    return model, info


C.train_cfg = train_cfg

# ------------------------------------------------------------------ architecture variant
_build, _constrain = K.build, K.constrain


def build(arch, ch, t, within, deps):
    if arch == "ShallowConvNet-MN":
        from train.shallow import ShallowConvNet
        return ShallowConvNet(ch, t, 2)
    return _build(arch, ch, t, within, deps)


def constrain(arch, m, deps):
    if arch == "ShallowConvNet-MN":
        return moabb_maxnorm(m)
    return _constrain(arch, m, deps)


K.build, K.constrain = build, constrain

# ------------------------------------------------------------------ test-time crop + model names
_run = K.run_one


def run_one(arch, cfg, Xn, Y, y, g, tr, va, te, te_s, within, epochs, patience, dev, seed, deps, calib_idx,
            onnx_dir, name, amp):
    STATE["name"] = name
    STATE["dir"] = Path(onnx_dir).parent / "models"; STATE["dir"].mkdir(exist_ok=True)
    if cfg.get("test_crop"):
        Xn = Xn.clone()
        idx = torch.from_numpy(np.concatenate([va, te]))
        Xn[idx] = crop_fixed(Xn[idx])
    return _run(arch, cfg, Xn, Y, y, g, tr, va, te, te_s, within, epochs, patience, dev, seed, deps, calib_idx,
                onnx_dir, name, amp)


K.run_one = run_one

# ------------------------------------------------------------------ data cache = confirmatory cache
_load = K.load_unit
K.load_unit = lambda name, out, max_subjects=None: _load(name, Path("results/confirmatory"), max_subjects)

if __name__ == "__main__":
    import sys
    if "--stage" not in sys.argv:
        sys.argv += ["--stage", "recipes"]
    K.main()
