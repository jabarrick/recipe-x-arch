# S1 — Condition T / T′ search space (fixed 2026-11-10, before the T′ search was run)

The registration (OSF 8st47) and the outline name the dimensions of the condition-T search — *24 configurations drawn at random from a fixed space: optimiser, learning rate log-uniform [1e-4, 1e-2], schedule, batch size, each augmentation on/off, label smoothing, model-selection rule* — but do not enumerate the levels of each dimension. They are enumerated here, committed before any T′ run, and implemented in `train/configurable.py`. Recorded in S4 as a specification, not a change.

## Levels

| Dimension | Levels | Sampling |
|---|---|---|
| Optimiser | Adam; AdamW (weight decay 1e-3) | uniform |
| Learning rate | [1e-4, 1e-2] | log-uniform (registered) |
| Schedule | constant; cosine to 0.01 × lr; step ×0.1 at 40 % and 80 % of epochs | uniform |
| Batch size | 16; 32; 64 | uniform |
| Gaussian noise (σ 0.02) | off; on | uniform |
| Circular time shift (±5 samples) | off; on | uniform |
| Random crop (90 %, zero-padded) | off; on | uniform |
| Channel dropout (10 %) | off; on | uniform |
| Mixup (α 0.2) | off; on | uniform |
| Label smoothing | 0; 0.1 | uniform |
| Model selection | early stopping on validation balanced accuracy, patience 15, best checkpoint; all epochs, lowest validation loss | uniform |

Epochs follow the protocol (cross-subject 100, within-subject 500). Augmentation magnitudes are those of R1. Weight constraints belong to the architecture and are not searched.

The four registered recipes are points of this space (R3's Adam ε = 1e-7 aside): R1 = AdamW, 1e-3, cosine, 64, all five augmentations, LS 0.1, early stopping; R2 = Adam, 1e-3, constant, 64, none, LS 0, lowest validation loss; R3 = Adam, 1e-2, step, 16, none, LS 0, lowest validation loss; R4 = AdamW, 1e-3, cosine, 64, none, LS 0, early stopping.

## The 24 draws

Drawn once with `numpy.random.default_rng(20261101)` (the registration date), independently per dimension, and **the same 24 configurations are used for every architecture**, so the budget is equal and the search is paired across architectures. The registered recipes are not among the draws.

| id | optimiser | lr | schedule | batch | augmentations | ls | selection |
|---|---|---|---|---|---|---|---|
| 0 | adamw | 5.20e-04 | cosine | 32 | crop | 0.1 | val_acc_es |
| 1 | adam | 3.40e-04 | cosine | 32 | none | 0.0 | val_acc_es |
| 2 | adamw | 1.83e-03 | constant | 32 | noise+crop+chdrop+mixup | 0.1 | val_acc_es |
| 3 | adamw | 3.91e-03 | cosine | 16 | noise+shift+crop+mixup | 0.0 | val_acc_es |
| 4 | adamw | 8.22e-04 | step | 64 | noise+shift+mixup | 0.0 | val_acc_es |
| 5 | adam | 6.74e-04 | cosine | 32 | shift+crop | 0.0 | val_acc_es |
| 6 | adam | 8.28e-03 | cosine | 64 | shift+crop+mixup | 0.1 | val_acc_es |
| 7 | adam | 4.56e-03 | cosine | 64 | noise+shift+crop+chdrop+mixup | 0.0 | val_acc_es |
| 8 | adamw | 1.66e-04 | step | 32 | noise+crop+chdrop | 0.1 | val_loss_all |
| 9 | adamw | 1.85e-03 | cosine | 64 | noise | 0.0 | val_loss_all |
| 10 | adam | 3.06e-03 | step | 64 | noise+shift+chdrop | 0.0 | val_loss_all |
| 11 | adam | 6.95e-03 | step | 16 | noise+shift+crop+mixup | 0.0 | val_loss_all |
| 12 | adam | 1.02e-04 | constant | 32 | noise+shift+chdrop+mixup | 0.1 | val_loss_all |
| 13 | adamw | 2.53e-03 | step | 32 | noise | 0.0 | val_acc_es |
| 14 | adamw | 1.10e-04 | constant | 32 | noise+shift+mixup | 0.0 | val_acc_es |
| 15 | adamw | 5.00e-04 | cosine | 32 | shift+crop | 0.1 | val_acc_es |
| 16 | adam | 4.19e-04 | step | 32 | shift+chdrop+mixup | 0.0 | val_loss_all |
| 17 | adamw | 1.35e-04 | cosine | 16 | noise+shift+mixup | 0.1 | val_acc_es |
| 18 | adam | 6.71e-03 | constant | 16 | noise+shift+crop | 0.1 | val_acc_es |
| 19 | adamw | 9.88e-03 | step | 64 | noise+shift+chdrop | 0.1 | val_acc_es |
| 20 | adam | 3.53e-04 | constant | 64 | mixup | 0.0 | val_acc_es |
| 21 | adamw | 4.18e-03 | step | 16 | shift+crop | 0.1 | val_acc_es |
| 22 | adamw | 2.00e-04 | cosine | 64 | noise | 0.1 | val_loss_all |
| 23 | adam | 1.38e-04 | cosine | 16 | noise+shift+mixup | 0.0 | val_acc_es |

## Scoring and selection

- **T′** (Amendment 1 §3.4): PhysioNet-MI left vs right (runs 4/8/12; S088, S092, S100, S104 excluded), fold 1 of the pilot split (`make_folds`, split seed 0, 10 % validation subjects). Each configuration is trained on fold 1's training subjects and scored on fold 1's validation subjects as the mean over seeds 0 and 1 of validation balanced accuracy at the selected checkpoint. Fold 1's test subjects are not used. The best configuration per architecture (ties: lower id) is T′ and is applied unchanged to every confirmatory cell.
- **T** (registration): the same 24 configurations and scoring rule on the training and validation subjects of fold 1 of each confirmatory cross-subject cell, or on session 1 of the 10 lowest-ID subjects in the within-subject cell.
- Disclosed: the validation subjects both stop training (under the early-stopping rule) and score the configuration, so the selected score is optimistic (winner's curse). This is the registered procedure.
