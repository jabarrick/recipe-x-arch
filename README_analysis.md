# recipe-x-arch — analysis bundle V0.2 (2026-11-08; V0.1 had a patsy namespace bug)

Drop into the repository root. Commit `docs/decision_rules.md` and `analysis/` **before** the binary pilot.

```
docs/decision_rules.md            frozen decision rules (Amendment 1)
analysis/common.py                constants, loaders, Holm
analysis/s8_interval_simulation.py   S8 (decision_rules §7); input: binary-pilot subject-level CSV
analysis/mixed_model.py           H1/H2 LRT (ML), H2b contrast, variance shares, cluster bootstrap
analysis/home_recipe.py           home advantage: cell-mean path + sum-coded model path, TOST, residual ranges, mapping
analysis/h3_classify.py           24 intervals, precedence, change counts, A/A'/B1/B2, H4 comparison
analysis/run_confirmatory.py      driver → results/section4_values_<family>.json + CSV tables
tests/test_home_recipe.py         unit tests required by decision_rules §2.4 (+ H3 precedence/outcomes)
```

Run tests: `python -m pytest tests/ -q` (8 tests; all pass on 2026-11-08).
Synthetic end-to-end check: `python -m analysis.run_confirmatory --demo --out results_demo/`.
S8 on the binary pilot: `python analysis/s8_interval_simulation.py pilot.csv --out S8/ --n-subjects 106 193`.

Input format for `run_confirmatory`: long CSV with columns
`subject, dataset, fold, architecture, recipe, seed, bacc_f32, bacc_int8, train_acc, val_acc, training_failure`
(accuracies in percent; architecture ∈ {EEGNet, ShallowConvNet, MI-BMInet, CSOANet-v2}; recipe ∈ {R1..R4}).

Dependencies: numpy, pandas, scipy, statsmodels ≥ 0.14, matplotlib (S8 figure), pytest.

Not yet in this bundle (belongs to the training side): MOABB loaders, the four recipes, the T search,
ONNX int8 export, operator-coverage report, calibration-set sampler (decision_rules §8).
