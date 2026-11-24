#!/usr/bin/env python3
"""Exploratory (post hoc, revision V1.2; S4 row 15): F1 under R1 with the random-crop
augmentation switched off ("R1nc"), everything else identical to R1.

Why: in the condition-T search, random cropping was the only configuration dimension with
a large effect (-5.9 points). In the R1 implementation the crop keeps 90 % of the window at a
random offset, moves it to the start of the window and zero-fills the last 10 %, once per
batch; test windows are never cropped. The question is how much of R1's deficit this one
component explains, for each architecture.

The frozen confirmatory driver is used unchanged; this file only registers one extra recipe
and points the data cache at the confirmatory cache. Results go to a separate folder.

Usage (repository root):
  python -m train.explore_r1_nocrop --csoanet-root ..\\csoanet2026_v2_deploy --unit F1a --out results\\exploratory_r1nocrop
  python -m train.explore_r1_nocrop --csoanet-root ..\\csoanet2026_v2_deploy --unit F1b --out results\\exploratory_r1nocrop
"""
import sys
from pathlib import Path
from train import configurable as C
from train import confirmatory as K

C.REGISTERED["R1nc"] = dict(C.REGISTERED["R1"], crop=False)
K.RECIPES = ["R1nc"]

_load = K.load_unit
K.load_unit = lambda name, out, max_subjects=None: _load(name, Path("results/confirmatory"), max_subjects)

if "--recipes" not in sys.argv:
    sys.argv += ["--recipes", "R1nc"]
if "--stage" not in sys.argv:
    sys.argv += ["--stage", "recipes"]

if __name__ == "__main__":
    K.main()
