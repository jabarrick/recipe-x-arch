# train/ — training-side driver

`train/pilot_binary.py` runs the 4 × 4 binary pilot (decision_rules §6) and writes the S8 input.
It imports data loading, folds, the R1 augmentation, the models and the weight constraints from the
CSOANet2026 `csoanet-v2-deploy` folder, which is a dependency (pass its path with `--csoanet-root`).

```powershell
# in recipe-x-arch, with the conda env that has torch + mne + braindecode + moabb
pip install braindecode
python -m train.pilot_binary --csoanet-root ..\csoanet2026_v2_deploy --synthetic --epochs 3 --max-subjects 10 --device cpu --out results\smoke   # smoke test
python -m train.pilot_binary --csoanet-root ..\csoanet2026_v2_deploy --out results\pilot_binary                                                 # the pilot (GPU)
python analysis\s8_interval_simulation.py results\pilot_binary\pilot_binary_s8.csv --out results\S8 --n-subjects 106 193                          # S8
```

Resumable: finished (architecture, recipe, fold, seed) rows are skipped on rerun; `--fresh` discards them.
First real run downloads PhysioNet-MI through MNE into `~/mne_data` (≈ 1.3 GB for runs 4/8/12 of 105 subjects).

Notes recorded on 2026-11-09:
- MI-BMInet is built with `nk=32, nf=64` explicitly (the registered configuration); the dependency's default at 160 Hz is 16/128.
- ShallowConvNet is braindecode's `ShallowFBCSPNet` (final_conv_length="auto", logits output), no weight constraint.
- R4 = R1 without augmentation, mixup and label smoothing (plain CE), same optimiser / schedule / early stopping.
- Training failure flag = non-finite loss after the seed+100 retry, or un-augmented training balanced accuracy ≤ 55 % at the selected checkpoint.
- Smoke test on synthetic data (3 epochs, CPU) ran 80/80 cells; CSOANet-v2 did not converge in 3 epochs under lr 1e-3 recipes and was correctly flagged — expected at 3 epochs, not a bug.
