"""Configurable training recipe and the condition-T / T' search space (Supplementary S1).

A recipe is a dict with keys
    optimizer  : "adam" | "adamw"          (AdamW: weight decay 1e-3)
    lr         : float
    schedule   : "constant" | "cosine" | "step"     (cosine to lr*0.01; step x0.1 at 40 % and 80 % of epochs)
    batch      : int
    noise, shift, crop, chdrop, mixup : bool   (augmentations, each on/off)
    label_smoothing : float
    select     : "val_acc_es" | "val_loss_all"
                 val_acc_es  : early stopping on validation accuracy, patience 15, keep best-accuracy checkpoint
                 val_loss_all: train all epochs, keep lowest-validation-loss checkpoint

The four registered recipes are points of this space (R3's Adam eps 1e-7 aside):
    R1 = adamw, 1e-3, cosine, 64, all five augmentations, ls 0.1, val_acc_es
    R2 = adam,  1e-3, constant, 64, none, ls 0, val_loss_all
    R3 = adam,  1e-2, step, 16, none, ls 0, val_loss_all
    R4 = adamw, 1e-3, cosine, 64, none, ls 0, val_acc_es
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F

AUGS = ("noise", "shift", "crop", "chdrop", "mixup")

# ------------------------------------------------------------------ search space (S1)
SPACE = dict(
    optimizer=["adam", "adamw"],
    lr=("loguniform", 1e-4, 1e-2),
    schedule=["constant", "cosine", "step"],
    batch=[16, 32, 64],
    noise=[False, True], shift=[False, True], crop=[False, True], chdrop=[False, True], mixup=[False, True],
    label_smoothing=[0.0, 0.1],
    select=["val_acc_es", "val_loss_all"],
)
N_CONFIGS = 24
CONFIG_SEED = 20261101          # registration date; fixes the 24 draws, identical for every architecture
SCORING_SEEDS = (0, 1)


def sample_configs(n: int = N_CONFIGS, seed: int = CONFIG_SEED) -> list[dict]:
    """Draw n configurations independently and uniformly from SPACE (lr log-uniform).
    The same list is used for every architecture, so the budget is equal and paired."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        c = {"id": i}
        for k, v in SPACE.items():
            if isinstance(v, tuple) and v[0] == "loguniform":
                c[k] = float(np.exp(rng.uniform(np.log(v[1]), np.log(v[2]))))
            else:
                c[k] = v[int(rng.integers(len(v)))]
                if isinstance(c[k], np.generic):
                    c[k] = c[k].item()
        out.append(c)
    return out


REGISTERED = {
    "R1": dict(optimizer="adamw", lr=1e-3, schedule="cosine", batch=64, noise=True, shift=True, crop=True,
               chdrop=True, mixup=True, label_smoothing=0.1, select="val_acc_es"),
    "R2": dict(optimizer="adam", lr=1e-3, schedule="constant", batch=64, noise=False, shift=False, crop=False,
               chdrop=False, mixup=False, label_smoothing=0.0, select="val_loss_all"),
    "R3": dict(optimizer="adam", lr=1e-2, schedule="step", batch=16, noise=False, shift=False, crop=False,
               chdrop=False, mixup=False, label_smoothing=0.0, select="val_loss_all", adam_eps=1e-7),
    "R4": dict(optimizer="adamw", lr=1e-3, schedule="cosine", batch=64, noise=False, shift=False, crop=False,
               chdrop=False, mixup=False, label_smoothing=0.0, select="val_acc_es"),
}


# ------------------------------------------------------------------ augmentation (switchable copy of the R1 package)
def augment(X, Y, nc, cfg, noise=0.02, shift=5, crop=0.9, ch_drop=0.1, mixup_alpha=0.2):
    """Same operations and order as deploy.train_physionet_mi.augment, each switchable."""
    B, _, C, T = X.shape
    if cfg["noise"]:
        X = X + torch.randn_like(X) * noise
    if cfg["shift"]:
        s = int(torch.randint(-shift, shift + 1, (1,)))
        if s:
            X = torch.roll(X, s, dims=-1)
    if cfg["crop"]:
        cl = int(T * crop)
        st = int(torch.randint(0, T - cl + 1, (1,)))
        out = torch.zeros_like(X)
        out[..., :cl] = X[..., st:st + cl]
        X = out
    if cfg["chdrop"]:
        X = X * (torch.rand(B, 1, C, 1, device=X.device) > ch_drop).float()
    ls = cfg["label_smoothing"]
    Y1 = F.one_hot(Y, nc).float() * (1 - ls) + ls / nc
    if cfg["mixup"]:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().to(X.device)
        perm = torch.randperm(B, device=X.device)
        X, Y1 = lam * X + (1 - lam) * X[perm], lam * Y1 + (1 - lam) * Y1[perm]
    return X, Y1


# ------------------------------------------------------------------ training
@torch.no_grad()
def logits_of(model, X, dev, bs=256):
    model.eval()
    return torch.cat([model(X[i:i + bs].to(dev)).float().cpu() for i in range(0, len(X), bs)])


def bacc(y_true, y_pred, nc):
    rec = [(y_pred[y_true == c] == c).mean() for c in range(nc) if (y_true == c).any()]
    return 100.0 * float(np.mean(rec))


def make_optim(cfg, params, epochs):
    eps = cfg.get("adam_eps", 1e-8)
    if cfg["optimizer"] == "adamw":
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-3, eps=eps)
    else:
        opt = torch.optim.Adam(params, lr=cfg["lr"], eps=eps)
    if cfg["schedule"] == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg["lr"] * 0.01)
    elif cfg["schedule"] == "step":
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.4 * epochs), int(0.8 * epochs)], 0.1)
    else:
        sch = None
    return opt, sch


def train_cfg(model, constrain_fn, cfg, Xtr, Ytr, Xva, Yva, nc, epochs, dev, seed, amp=True, patience=15):
    """Train `model` with recipe `cfg`. Returns (model, info) with the selected checkpoint loaded.
    info: val_bacc, val_loss, best_epoch, epochs_run, train_bacc (un-augmented, selected checkpoint), nonfinite."""
    torch.manual_seed(seed); np.random.seed(seed % 2**32)
    model = model.to(dev)
    opt, sch = make_optim(cfg, model.parameters(), epochs)
    any_aug = any(cfg[a] for a in AUGS) or cfg["label_smoothing"] > 0
    use_amp = dev.type == "cuda" and amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    by_acc = cfg["select"] == "val_acc_es"
    best_metric = -math.inf if by_acc else math.inf
    best_state, best_info, wait, nonfinite = None, None, 0, False
    yva = Yva.numpy()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Ytr))
        for i in range(0, len(perm), cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            if len(idx) < 2:
                continue
            xb, yb = Xtr[idx].to(dev), Ytr[idx].to(dev)
            if any_aug:
                xb, yb = augment(xb, yb, nc, cfg)
            else:
                yb = F.one_hot(yb, nc).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = -(yb * F.log_softmax(model(xb).float(), 1)).sum(1).mean()
            if not torch.isfinite(loss):
                nonfinite = True
                break
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            constrain_fn(model)
        if nonfinite:
            break
        if sch is not None:
            sch.step()
        lg = logits_of(model, Xva, dev)
        v_loss = float(F.cross_entropy(lg, Yva))
        v_bacc = bacc(yva, lg.argmax(1).numpy(), nc)
        improved = (v_bacc > best_metric) if by_acc else (v_loss < best_metric)
        if improved:
            best_metric, wait = (v_bacc if by_acc else v_loss), 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_info = dict(val_bacc=v_bacc, val_loss=v_loss, best_epoch=ep + 1)
        else:
            wait += 1
            if by_acc and wait >= patience:
                break
    if best_state is None:
        return model, dict(val_bacc=float("nan"), val_loss=float("nan"), best_epoch=0, epochs_run=ep + 1,
                           train_bacc=float("nan"), nonfinite=True)
    model.load_state_dict(best_state)
    step = max(1, len(Ytr) // 2000)
    tb = bacc(Ytr[::step].numpy(), logits_of(model, Xtr[::step], dev).argmax(1).numpy(), nc)
    return model, dict(**best_info, epochs_run=ep + 1, train_bacc=tb, nonfinite=nonfinite)
