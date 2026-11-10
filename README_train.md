# train/ — training-side driver

`train/pilot_binary.py` runs the 4 × 4 binary pilot (decision_rules §6) and writes the S8 input.
It imports data loading, folds, the R1 augmentation, the models and the weight constraints from the
CSOANet2026 `csoanet-v2-deploy` folder, which is a dependency (pass its path with `--csoanet-root`).

```powershell
# in recipe-x-arch, with the conda env that has torch + mne + braindecode + moabb
# braindecode is NOT needed: ShallowConvNet is implemented in train/shallow.py (tests/test_shallow.py checks it equals braindecode when braindecode is present)
python -m train.pilot_binary --csoanet-root ..\csoanet2026_v2_deploy --synthetic --epochs 3 --max-subjects 10 --device cpu --out results\smoke   # smoke test
python -m train.pilot_binary --csoanet-root ..\csoanet2026_v2_deploy --out results\pilot_binary                                                 # the pilot (GPU)
python analysis\s8_interval_simulation.py results\pilot_binary\pilot_binary_s8.csv --out results\S8 --n-subjects 106 193                          # S8
```

Resumable: finished (architecture, recipe, fold, seed) rows are skipped on rerun; `--fresh` discards them.
First real run downloads PhysioNet-MI through MNE into `~/mne_data` (≈ 1.3 GB for runs 4/8/12 of 105 subjects).

Notes recorded on 2026-11-09:
- MI-BMInet is built with `nk=32, nf=64` explicitly (the registered configuration); the dependency's default at 160 Hz is 16/128.
- ShallowConvNet is `train/shallow.py`, layer-for-layer equal to braindecode 2.0 `ShallowFBCSPNet` (final_conv_length="auto"), no weight constraint. Implemented in-repo on 2026-11-09 after braindecode → torchaudio DLL failure on Windows; equivalence test passes on 3 input shapes.
- R4 = R1 without augmentation, mixup and label smoothing (plain CE), same optimiser / schedule / early stopping.
- Training failure flag = non-finite loss after the seed+100 retry, or un-augmented training balanced accuracy ≤ 55 % at the selected checkpoint.
- Smoke test on synthetic data (3 epochs, CPU) ran 80/80 cells; CSOANet-v2 did not converge in 3 epochs under lr 1e-3 recipes and was correctly flagged — expected at 3 epochs, not a bug.

## T′ search and baseline reproduction (added 2026-11-10)

```powershell
# T′: 4 architectures x 24 configurations x 2 seeds = 192 runs on PhysioNet left/right fold 1 (uses the pilot cache)
python -m train.tprime_search --csoanet-root ..\csoanet2026_v2_deploy --out results\tprime
# baseline reproduction on BCI IV 2a (MOABB download on first run); 3 seeds
python -m train.baseline_repro --csoanet-root ..\csoanet2026_v2_deploy --out results\baseline_repro
```

- `train/configurable.py`: the S1 search space (`docs/S1_search_space.md`), the 24 fixed draws, a switchable copy of the R1 augmentation, and a generic training loop.
- `train/tprime_search.py` writes `tprime_selected.json` (→ Amendment 1 §3.10), `tprime_scores.csv`, `tprime_runs.csv`.
- `train/baseline_repro.py` writes `baseline_repro_summary.json` (reproduced vs reference, ±3 verdict) and per-unit rows. EEGNet / ShallowConvNet follow the MOABB v1.1.2 Keras pipelines (8–32 Hz, 128 / 250 Hz, per-time-sample channel z-score, within-session stratified 5-fold with random_state 42, last-20 % validation, EarlyStopping + ReduceLROnPlateau on val_loss with patience 75, final weights); MI-BMInet follows Wang et al. 2024 (session 1 → 2, 0–3 s, 0.5–100 Hz, Adam 1e-3, 500 epochs, batch 32, final weights).
- Both resumable. Smoke tests: `--synthetic --epochs 2 --max-subjects 10 --n-configs 3` and `--fake --epochs 3 --seeds 0` (needs `pip install moabb`).
