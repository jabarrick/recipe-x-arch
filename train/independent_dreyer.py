#!/usr/bin/env python3
"""Independent preregistered test (OSF 8st47, Amendment 3, V1.1): Dreyer2023 (MOABB 1.7.2), left vs right hand MI.

Frozen together with analysis/independent_analysis.py BEFORE the dataset is downloaded.
Uses the frozen confirmatory driver unchanged, with:
  - unit "IND" = Dreyer2023, cross-subject, 10 folds (split seed 0), 10 % validation subjects;
  - nominal 240 trials per subject for the registered 80 % exclusion rule;
  - architectures EEGNet, ShallowConvNet-MN (with the reference max-norm constraints), MI-BMInet, CSOANet-v2;
  - recipes R1, R1nc (R1 without cropping), R2 and R4;
  - every trained model saved under <out>/models/.
Dry run before download (pilot cache, no confirmatory data):
  python -m train.independent_dreyer --csoanet-root ..\\csoanet2026_v2_deploy --unit IND --out results\\ind_dryrun ^
         --pilot-dryrun --max-subjects 20 --epochs 2 --archs EEGNet ShallowConvNet-MN MI-BMInet CSOANet-v2 --recipes R1 R1nc R2 R4
Confirmatory run:
  python -m train.independent_dreyer --csoanet-root ..\\csoanet2026_v2_deploy --unit IND --out results\\independent ^
         --archs EEGNet ShallowConvNet-MN MI-BMInet CSOANet-v2 --recipes R1 R1nc R2 R4
"""
import numpy as np
from train import explore_rev6 as E          # registers R1nc and ShallowConvNet-MN, model saving, cache redirect
from train import confirmatory as K

N_FOLDS = 10
K.UNITS["IND"] = ("Dreyer2023", "cross-subject")
K.NOMINAL_TRIALS = 240
K.load_unit = E._load                        # IND has its own cache in --out (no confirmatory cache)


def folds(g, sess, protocol, val_frac, split_seed=0):
    subs = np.unique(g); rng = np.random.default_rng(split_seed); rng.shuffle(subs)
    for k, test in enumerate(np.array_split(subs, N_FOLDS)):
        rest = np.setdiff1d(subs, test); rest = rest[rng.permutation(len(rest))]
        n_val = max(1, int(round(len(rest) * val_frac)))
        tr_s, va_s = rest[n_val:], rest[:n_val]
        yield k, *(np.flatnonzero(np.isin(g, x)) for x in (tr_s, va_s, test)), test


K.folds = folds

if __name__ == "__main__":
    import sys
    if "--stage" not in sys.argv:
        sys.argv += ["--stage", "recipes"]
    K.main()
