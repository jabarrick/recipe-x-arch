---
title: "Supplementary material — Is a shared training recipe neutral?"
subtitle: "Supplementary S1–S8 — V1.0 (24 November 2026)"
---

# S1 Recipes and the condition-T search space

Registered recipes R1–R4 are given in the main text (Table 3). The search space and the 24 configurations below were fixed on 10 November 2026, before the T′ search was run, and are shared by all architectures.

The registration (OSF 8st47) and the outline name the dimensions of the condition-T search — *24 configurations drawn at random from a fixed space: optimiser, learning rate log-uniform [1e-4, 1e-2], schedule, batch size, each augmentation on/off, label smoothing, model-selection rule* — but do not enumerate the levels of each dimension. They are enumerated here, committed before any T′ run, and implemented in `train/configurable.py`. Recorded in S4 as a specification, not a change.

#### Levels

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

#### The 24 draws

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

#### Scoring and selection


- **T′** (Amendment 1 §3.4): PhysioNet-MI left vs right (runs 4/8/12; S088, S092, S100, S104 excluded), fold 1 of the pilot split (`make_folds`, split seed 0, 10 % validation subjects). Each configuration is trained on fold 1's training subjects and scored on fold 1's validation subjects as the mean over seeds 0 and 1 of validation balanced accuracy at the selected checkpoint. Fold 1's test subjects are not used. The best configuration per architecture (ties: lower id) is T′ and is applied unchanged to every confirmatory cell.
- **T** (registration): the same 24 configurations and scoring rule on the training and validation subjects of fold 1 of each confirmatory cross-subject cell, or on session 1 of the 10 lowest-ID subjects in the within-subject cell.
- Disclosed: the validation subjects both stop training (under the early-stopping rule) and score the configuration, so the selected score is optimistic (winner's curse). This is the registered procedure.

### Configurations selected by condition T (confirmatory) and T′ (pilot data)

| unit   | architecture   |   config |   search_score |   test_mean |
|:-------|:---------------|---------:|---------------:|------------:|
| F1a    | EEGNet         |       13 |          86.69 |       80.30 |
| F1a    | ShallowConvNet |       10 |          83.25 |       76.25 |
| F1a    | MI-BMInet      |       19 |          85.25 |       80.10 |
| F1a    | CSOANet-v2     |       20 |          72.50 |       71.55 |
| F1b    | EEGNet         |       19 |          65.18 |       72.26 |
| F1b    | ShallowConvNet |       19 |          64.23 |       70.35 |
| F1b    | MI-BMInet      |       19 |          63.69 |       72.99 |
| F1b    | CSOANet-v2     |        4 |          58.21 |       68.63 |
| F2     | EEGNet         |       21 |          83.67 |       62.52 |
| F2     | ShallowConvNet |       23 |          83.45 |       63.71 |
| F2     | MI-BMInet      |        3 |          84.71 |       64.77 |
| F2     | CSOANet-v2     |        3 |          91.53 |       66.44 |

*Search score: mean validation balanced accuracy over scoring seeds 0 and 1 on the unit's tuning split. Test mean: mean test balanced accuracy of the selected configuration over all subjects and seeds. The two are computed on different subjects; their difference is not an estimate of the winner's curse (main text 4.5).*

T′ selections (PhysioNet-MI, before the code freeze): EEGNet #10; ShallowConvNet #1; MI-BMInet #13; CSOANet-v2 #10.

# S2 Subject exclusions, fold assignment and flagged runs

**Exclusions.** No subject was excluded in any unit (80 %-of-200-trials rule; MOABB 1.7.2). Lee2019-MI: 54 subjects × 200 trials (100 per session). Cho2017: 52 subjects, 49 with 200 trials and 3 with 240.

**Fold assignment, F1 (cross-subject, 5 folds; subject numbers within each dataset).** Fold 1 is the tuning fold of condition T.

| dataset    |   fold | subject                                               |
|:-----------|-------:|:------------------------------------------------------|
| Cho2017    |      1 | 005, 011, 012, 019, 024, 028, 029, 035, 036, 037, 039 |
| Cho2017    |      2 | 002, 003, 004, 007, 021, 022, 023, 025, 041, 044, 050 |
| Cho2017    |      3 | 001, 009, 017, 018, 020, 027, 031, 038, 046, 048      |
| Cho2017    |      4 | 010, 013, 014, 026, 033, 043, 045, 047, 049, 051      |
| Cho2017    |      5 | 006, 008, 015, 016, 030, 032, 034, 040, 042, 052      |
| Lee2019_MI |      1 | 005, 011, 012, 019, 024, 028, 029, 035, 036, 037, 046 |
| Lee2019_MI |      2 | 002, 003, 004, 007, 021, 022, 023, 025, 043, 051, 052 |
| Lee2019_MI |      3 | 001, 009, 017, 018, 020, 027, 031, 038, 044, 048, 050 |
| Lee2019_MI |      4 | 010, 013, 014, 026, 033, 039, 041, 045, 047, 049, 053 |
| Lee2019_MI |      5 | 006, 008, 015, 016, 030, 032, 034, 040, 042, 054      |

**Flagged runs, F1.** One row per (dataset, architecture, recipe, reason); 41 of 480 runs in total.

| dataset    | architecture   | recipe   | reason                               |   runs |
|:-----------|:---------------|:---------|:-------------------------------------|-------:|
| Cho2017    | ShallowConvNet | R3       | non-finite loss after seed+100 retry |     15 |
| Cho2017    | CSOANet-v2     | R3       | training bacc ≤ 55 %                 |      2 |
| Lee2019_MI | ShallowConvNet | R3       | non-finite loss after seed+100 retry |     15 |
| Lee2019_MI | CSOANet-v2     | R3       | training bacc ≤ 55 %                 |      9 |

**Flagged runs, F2 (subject-level runs; 124 of 2,592).**

| architecture   |   R1 |   R2 |   R3 |   R4 |
|:---------------|-----:|-----:|-----:|-----:|
| EEGNet         |   42 |    6 |    0 |    2 |
| ShallowConvNet |    7 |    0 |    3 |    1 |
| MI-BMInet      |   16 |   10 |   21 |    8 |
| CSOANet-v2     |    1 |    4 |    2 |    1 |

# S3 Results files and seed variability

Per-subject × architecture × recipe × seed results (float32 and int8 balanced accuracy, float32–int8 agreement, training and validation accuracy at the selected checkpoint, epochs, failure flag) are released as CSV files in the OSF project and the repository: `F1a_long.csv`, `F1b_long.csv`, `F2_long.csv`, and for conditions T and T′ `*_T_long.csv`, `*_Tprime_long.csv`, `*_T_search.csv`.

**Median subject-level seed SD per cell (points), F1.** The median exceeds the 2-point SESOI in 13 of 16 cells.

| architecture   |   R1 |   R2 |   R3 |   R4 |
|:---------------|-----:|-----:|-----:|-----:|
| EEGNet         | 2.29 | 1.80 | 2.27 | 1.76 |
| ShallowConvNet | 2.50 | 2.47 | 2.75 | 2.02 |
| MI-BMInet      | 2.48 | 2.54 | 2.57 | 1.89 |
| CSOANet-v2     | 2.36 | 3.25 | 3.94 | 2.75 |

**Median subject-level seed SD per cell (points), F2.** All 16 cells exceed 2 points.

| architecture   |   R1 |   R2 |   R3 |   R4 |
|:---------------|-----:|-----:|-----:|-----:|
| EEGNet         | 5.11 | 3.79 | 2.94 | 3.51 |
| ShallowConvNet | 4.02 | 3.61 | 3.51 | 3.89 |
| MI-BMInet      | 3.91 | 3.61 | 3.89 | 4.04 |
| CSOANet-v2     | 4.15 | 3.91 | 3.65 | 4.65 |

# S4 Preregistration deviations and post-freeze changes

Registration: OSF 8st47 (1 Nov 2026). Rules and analysis code frozen at f3fc1449 (2026-11-08T09:09:39Z).
Every change to `docs/decision_rules.md` or to the analysis code after f3fc1449, and every post-download fix to data-loading code, is one row here. "Before results" = made before any confirmatory run.

| # | Date (UTC) | Commit | File(s) | Registered / frozen text | Change | Reason | Before results? | Affects a decision rule? |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-08 | 2312659 | docs/decision_rules.md | change log row `[TO FILL]` | freeze commit hash and time recorded | self-reference only | yes | no |
| 2 | 2026-11-08 | 558ac92 | docs/decision_rules.md §5 | reference figures marked `[TO VERIFY]`; MI-BMInet recipe described only as "published recipe" | reference figures filled: EEGNet-8,2 60.46 % and ShallowConvNet 72.47 % from the MOABB benchmark (within-session 5-fold, BNCI2014-001); MI-BMInet 76.03 % from Wang et al. 2024 (session 1→2). Note added that R3 is MI-BMInet's cross-subject recipe, not its BCI IV 2a recipe | Lawhern 2018 gives no numeric within-subject figure; Schirrmeister 2017 figures are for cropped training; Wang 2024 uses different recipes for 2a and PhysioNet | yes | no — tolerance (±3) and handling rule unchanged; home mapping unchanged (defined on F1) |
| 3 | 2026-11-10 | 531c7fa | docs/S1_search_space.md, train/configurable.py | registration names the condition-T search dimensions (optimiser, lr log-uniform [1e-4, 1e-2], schedule, batch, each augmentation on/off, label smoothing, selection rule) without enumerating levels | levels enumerated; 24 configurations drawn once with seed 20261101 and shared across architectures; scoring = mean val balanced accuracy over seeds 0 and 1 | specification needed to run T′; made before any T′ or T run | yes | no — fills an unenumerated registered space; does not touch H1–H3 rules |
| 4 | 2026-11-10 | 531c7fa | train/baseline_repro.py | decision_rules §5: "protocol of the reference figure matched as closely as the pipeline allows" | MOABB reference protocol taken from the v1.1.2 Keras pipeline files (8–32 Hz, per-time-sample channel z-score, last-20 % validation, EarlyStopping/ReduceLROnPlateau patience 75, final weights); model definitions kept as in the study | makes "matched protocol" concrete before any baseline run | yes | no |
| 5 | 2026-11-11 | 588d023 | train/baseline_repro.py | decision_rules §5: if outside tolerance, "the implementation is checked once against the reference and the check is documented" | diagnostic option `ShallowConvNet+MOABB-maxnorm` added (study ShallowConvNet + MOABB Keras max-norm 2.0 / 0.5), reported descriptively in S7; study model unchanged | ShallowConvNet reproduced 66.62 vs 72.47 (−5.85), outside ±3; this is the documented check | yes (before any confirmatory run) | no — model, tolerance and handling unchanged |
| 6 | 2026-11-13 | 1a7ba7b | train/confirmatory.py (new) | Amendment 1 §2 states that tag v1.0-frozen (a44e5933) is the frozen confirmatory code | v1.0-frozen contained the analysis code, the recipes, models, constraints, training loop (train/configurable.py), quantisation and the pilot/T′/baseline drivers, but not the orchestration that runs the confirmatory units (MOABB loading of Lee2019-MI and Cho2017, F1/F2 folds and splits, seeds 0–2, T search per unit, T′ application, int8 export and operator coverage). That driver was written after v1.0-frozen, dry-run only on synthetic data, and frozen at tag v1.1-frozen before any confirmatory download; recorded on OSF as Amendment 2. Early stopping under R1/R4 uses validation balanced accuracy (pilot driver: plain accuracy); classes are balanced, and S1 already specifies balanced accuracy. | omission in the freezing order, noticed before any confirmatory data access | yes (no confirmatory data downloaded) | no — no rule, threshold or analysis changed |
| 7 | 2026-11-17 | c804e98 | train/confirmatory.py `load_unit` (data-loading layer, post-download) | registration: MOABB 1.7.2 | The local environment had MOABB 1.5.0, which returns only one of the two Lee2019-MI sessions (100 trials per subject), so every subject fell below the 80 % trial rule and the first Lee load excluded all 54. Fix: `load_unit` now refuses to run unless MOABB is 1.7.2, refuses caches not built with 1.7.2, prints the reason for every excluded subject, and stops with an explicit error if all subjects are excluded. The Cho2017 cache built under 1.5.0 is discarded and rebuilt under 1.7.2; no model was trained on it. | post-download fix confined to data loading (registration §3.9 of Amendment 1) | yes — no confirmatory training run had started | no |
| 8 | 2026-11-17 | — | environment record | registration: MOABB 1.7.2 | The BCI IV 2a baseline reproduction (S7) loaded BNCI2014_001 with MOABB 1.5.0, not 1.7.2. That check is descriptive and decides nothing; it is reported with this version. The PhysioNet-MI pilot and the T′ search used MNE's own loader and are unaffected. | discovered with row 7 | yes | no |
| 9 | 2026-11-18 | 5a6b37e | environment; README_confirmatory.md; tools/run_confirmatory.ps1 | registration: MOABB 1.7.2 | After the switch to MOABB 1.7.2 (row 7) the environment for the confirmatory runs is recorded in docs/environment_confirmatory.txt (numpy 2.5.3, MNE 1.11.0, torch 2.6.0+cu124, onnxruntime 1.30.0); numpy and MNE were upgraded as MOABB 1.7.2 dependencies after v1.1-frozen. Raw data are read with MOABB_DOWNLOAD_PROVIDER=upstream so that MOABB 1.7.2 reads the local GigaDB files instead of prefetching the NEMAR mirror (one subject was prefetched once; SHA-256 identical). A shell script runs the stages in the registered order; it calls the frozen driver unchanged. | environment and data source record | yes — before any confirmatory training run | no |
| 10 | 2026-11-18 | 6af6271 | tools/run_confirmatory.ps1 (wrapper only) | — | The wrapper set $ErrorActionPreference = "Stop". At 15:46Z, during F1a fold 0, a torch.onnx constant-folding UserWarning written to stderr was turned by Windows PowerShell into a terminating error and stopped the run after 24 completed runs (EEGNet and ShallowConvNet, R1–R4, seeds 0–2). The wrapper now uses "Continue", converts stderr lines to plain text, and still stops on a non-zero exit code. The frozen driver is unchanged; the stage resumes from its result files, so completed runs are kept and not repeated. No results were inspected. | operational fix to the run wrapper | n/a — during the confirmatory runs | no |
| 11 | 2026-11-21 | 1b5434c | analysis/run_confirmatory.py (failure-excluded sensitivity block only) | decision_rules.md §2.4 | First confirmatory run of the analysis (F1, 18:31Z) stopped in the training-failure sensitivity: ShallowConvNet × R3 is flagged as a training failure in every run (non-finite loss after the seed+100 retry, §3), so excluding failures leaves every subject with an incomplete table. §2.4 then prescribes the model path (>10 % incomplete), but the code still built the subject-drop tables, which were empty, and crashed. Fix: when the model path applies, h is taken from the sum-coded mixed model only; missing cells per architecture × recipe are reported; an H3 failure on incomplete tables is reported instead of raised. Primary analyses, rules and thresholds unchanged; the crashed output is kept in results/confirmatory/analysis_V1.0, the complete output in analysis_V1.1. | bug in a sensitivity branch exposed by the data | no (found when running the analysis) | no |
| 12 | 2026-11-23 | b11fe27 | analysis/run_h4.py (new) | Methods §3.6.3 H4; Amendment 1 §3.4 | The frozen tag contains the H4 decision function (`h3_classify.compare_conditions`, `count_changes`) but no driver that assembles R1, T, T′ and the fold-1-only subset from the result files. The driver was written after the T and T′ runs for F1 had finished and only sequences frozen functions; it adds no rule. Interval level: the frozen function uses Bonferroni over the 12 intervals of the R1-versus-T comparison (99.58 %); the result at 99.79 % (24 intervals, as for H3) is reported beside it and gives the same categories. Fold 1 of the registration is fold index 0 in the result files. | omission of a driver (sequencing only) | no — written after the F1 T/T′ runs, before running the H4 analysis | no |
| 13 | 2026-11-24 | 0c158fe, a370118 | analysis/section4_extra.py, analysis/figures_section4.py, tools/static_cost.py (new) | Section 4 template (2026-11-08); Amendment 1 §3.5 sensitivity list | The frozen driver does not compute several registered sensitivity analyses and table entries: dataset interactions and per-dataset analyses, fold as fixed effect, seed as random effect, ART-ANOVA, Table 5 entries, int8 agreement and operator coverage summaries, the Kendall τ interval and the T configurations. section4_extra.py computes them from the result files; figures_section4.py draws Figures 1–4 to the specifications fixed in the template. static_cost.py fills Table 9 with the CSOANet2026 repository's own ONNX profiler (deploy/profile_model.py); the first table (docs/static_cost_V1.0.csv) was computed in a separate environment (torch 2.15, onnx 1.23) and can be regenerated with the command in the script. ART is implemented in Python (aligned rank transform with an LRT on ranks with a subject random intercept), not with the R package ARTool. One analysis was added after the results were seen and is labelled post hoc exploratory: home residuals on the 4 × 3 table without R3 (4.7 d). | analyses registered but not implemented in the frozen code; one post hoc exploratory analysis | no — after the results | no — no registered decision depends on these scripts |

# S5 Quantisation and static cost

**Procedure.** Post-training static quantisation with ONNX Runtime (QDQ format, per-channel symmetric int8 weights, MinMax calibration). The calibration set is 256 un-augmented training trials drawn without replacement with seed = fold index (cross-subject) or subject index (within-subject), the same for every architecture, recipe and seed in a fold or subject; the indices are saved with the models (`*_calibration_indices.json`). Operator coverage counts compute nodes wrapped in Q/DQ.

**Operator coverage and agreement.**

| architecture   | quantised_of_compute   | float32_ops      |   agreement_F1 |   f32_minus_int8_F1 |   agreement_F2 |
|:---------------|:-----------------------|:-----------------|---------------:|--------------------:|---------------:|
| EEGNet         | 5/9                    | AveragePool, Elu |          96.10 |                0.86 |          98.91 |
| ShallowConvNet | 6/6                    | —                |          94.05 |                0.94 |          96.70 |
| MI-BMInet      | 7/7                    | —                |          97.78 |                0.06 |          99.05 |
| CSOANet-v2     | 27/27                  | —                |          94.24 |                0.49 |          97.76 |

**Mean float32 − int8 loss per cell, F1 (points).**

| architecture   |    R1 |   R2 |    R3 |    R4 |
|:---------------|------:|-----:|------:|------:|
| EEGNet         |  0.06 | 0.11 |  3.21 |  0.07 |
| ShallowConvNet |  0.83 | 0.79 |  1.25 |  0.88 |
| MI-BMInet      |  0.16 | 0.06 | -0.00 |  0.03 |
| CSOANet-v2     | -0.07 | 0.54 |  1.51 | -0.03 |

**Static cost (Table 11 of the main text, full).** Read off the exported ONNX graph with `deploy/profile_model.py` of the CSOANet2026 repository (`tools/static_cost.py`).

| dataset              | architecture   |   channels |   samples |   params_train |   params_deployed |   flash_kB |   MMACs |   peak_kB |   stream_kB |
|:---------------------|:---------------|-----------:|----------:|---------------:|------------------:|-----------:|--------:|----------:|------------:|
| Lee2019_MI (F1a, F2) | EEGNet         |         62 |      1000 |           3818 |              3762 |        3.8 |  63.181 |     544.9 |       500   |
| Lee2019_MI (F1a, F2) | ShallowConvNet |         62 |      1000 |         105202 |            105162 |      102.9 | 157.375 |    2424.3 |      2401.9 |
| Lee2019_MI (F1a, F2) | MI-BMInet      |         62 |      1000 |           6722 |              6562 |        6.5 |   1.985 |      91.9 |        31.3 |
| Lee2019_MI (F1a, F2) | CSOANet-v2     |         62 |      1000 |           3800 |              3634 |        3.6 |   2.082 |      76.3 |        23.5 |
| Cho2017 (F1b)        | EEGNet         |         64 |       750 |           3594 |              3538 |        3.5 |  48.909 |     421.9 |       386.7 |
| Cho2017 (F1b)        | ShallowConvNet |         64 |       750 |         107042 |            107002 |      104.7 | 120.839 |    1861.9 |      1843.4 |
| Cho2017 (F1b)        | MI-BMInet      |         64 |       750 |           6530 |              6370 |        6.3 |   1.537 |      70.4 |        23.5 |
| Cho2017 (F1b)        | CSOANet-v2     |         64 |       750 |           3704 |              3538 |        3.5 |   1.573 |      58.7 |        17.7 |

# S6 Minimal reporting checklist for comparisons of EEG decoders

1. **Recipe in full.** Optimiser and its parameters, learning rate and schedule, batch size, epochs, early stopping and model-selection rule, every augmentation with its magnitude, label smoothing, weight constraints. State which elements belong to the model definition and which to the recipe.
2. **Whose recipe.** State whether the shared recipe was developed with one of the compared models, and by whom.
3. **More than one recipe.** Report at least two recipes (one plain, one with augmentation) or an equal-budget search per model, with the budget stated.
4. **Intervals, not ranks.** Report every pairwise difference with an adjusted confidence interval at the subject level, and a margin below which differences are treated as equivalent.
5. **Seeds.** Report the number of seeds and the seed-to-seed SD per cell next to the pairwise differences.
6. **Failures.** Report runs that diverged or failed to learn, the rule used to flag them, and whether they were kept.
7. **Tuning data.** State which subjects or sessions were used for any hyperparameter choice and whether they are also test data.
8. **Deployment claims.** Make them on the quantised graph, under the recipe that will be deployed, with operator coverage and the calibration set reported.
9. **Cost.** Report parameters, MACs and peak activation memory of the deployed graph for each input shape used.
10. **Preregistration and deviations.** If preregistered, link the registration and list every deviation with its reason and timing.

# S7 Pilot results

The pilots were run on data the authors had used before and are disclosed as motivation, not evidence.

## S7.1 Binary pilot (PhysioNet-MI, left vs right hand)

PhysioNet-MI, left vs right hand imagery (runs 4/8/12; S088, S092, S100, S104 excluded), 105 subjects, 44.8 scored trials per subject (42–45).
4 architectures × 4 recipes, cross-subject 5-fold (split seed 0, 10 % validation subjects), one training seed (0), 100 epochs max, 1–40 Hz, 0–4 s, fixed per-channel gain.
Driver `train/pilot_binary.py` at commit 0eff2c8; run 9–10 Nov 2026, after the rules were frozen at f3fc1449. Raw rows: `pilot_binary_long.csv`; settings: `pilot_binary_settings.json`.

Training failures: 0 of 1,680 subject rows (0 of 80 runs); retries: 0.

#### Subject-level balanced accuracy, cell means (%)

| architecture   |    R1 |    R2 |    R3 |    R4 |
|:---------------|------:|------:|------:|------:|
| EEGNet         | 76.95 | 81.17 | 80.55 | 80.8  |
| ShallowConvNet | 78.88 | 81.81 | 80.26 | 81.08 |
| MI-BMInet      | 80.16 | 81.66 | 80.73 | 80.44 |
| CSOANet-v2     | 75.49 | 79.42 | 76.42 | 77.48 |

#### Interaction residuals (points; double-centred cell means)

| | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| EEGNet | −1.21 | **−0.13** | 0.77 | 0.56 |
| ShallowConvNet | 0.09 | −0.13 | −0.16 | 0.21 |
| MI-BMInet | 1.12 | −0.52 | **0.08** | −0.67 |
| CSOANet-v2 | **−0.00** | 0.79 | −0.69 | −0.09 |

Home cells in bold. H2b contrast (EEGNet R2−R1 minus CSOANet-v2 R2−R1): +0.29 points (SD 10.83, n = 105); four-class pilot was +3.95.

#### Training diagnostics (mean over folds)

|                          |   train_acc |   val_acc |   epochs |
|:-------------------------|------------:|----------:|---------:|
| ('CSOANet-v2', 'R1')     |        79.3 |      80.4 |     37.8 |
| ('CSOANet-v2', 'R2')     |        88.6 |      81.2 |    100   |
| ('CSOANet-v2', 'R3')     |        82.8 |      77.3 |    100   |
| ('CSOANet-v2', 'R4')     |        85.1 |      81.8 |     42   |
| ('EEGNet', 'R1')         |        78.8 |      78.4 |     30.4 |
| ('EEGNet', 'R2')         |        86   |      83   |    100   |
| ('EEGNet', 'R3')         |        85.6 |      82.2 |    100   |
| ('EEGNet', 'R4')         |        84.1 |      83.7 |     31.4 |
| ('MI-BMInet', 'R1')      |        87.9 |      83.3 |     52.8 |
| ('MI-BMInet', 'R2')      |        89.8 |      84.1 |    100   |
| ('MI-BMInet', 'R3')      |        86.7 |      83.7 |    100   |
| ('MI-BMInet', 'R4')      |        88.6 |      85.2 |     30   |
| ('ShallowConvNet', 'R1') |        90   |      82.6 |     30.6 |
| ('ShallowConvNet', 'R2') |        95.4 |      84.3 |    100   |
| ('ShallowConvNet', 'R3') |        86.4 |      82.9 |    100   |
| ('ShallowConvNet', 'R4') |        96.5 |      85.7 |     34.8 |

R1 and R4 stop early on validation accuracy (patience 15) at about 30–50 epochs; R2 and R3 run all 100 epochs and keep the lowest-validation-loss checkpoint.

#### Reading

Descriptive only. R1 (the authors' recipe) was the lowest recipe for every architecture; CSOANet-v2 (the authors' model) was the lowest architecture under every recipe. Both lower the authors' prior for H2b and for any CSOANet-v2 advantage. PhysioNet-MI is the development dataset of CSOANet-v2 and R1, so even this pattern is not independent evidence. No rule was changed after these results.

## S7.2 Baseline reproduction (BCI Competition IV 2a)

Driver `train/baseline_repro.py` at commit 531c7fa; reference figures and ±3-point tolerance fixed in decision_rules §5 (558ac92) before any run; protocols made concrete in S4 row 4 before any run. Three seeds per baseline.

| Baseline | Reference | Reproduced (mean of 3 seeds) | Per seed | Difference | Within ±3 | Protocol |
|---|---|---|---|---|---|---|
| EEGNet-8,2 | 60.46 (MOABB benchmark) | 61.8 | 62.14, 61.45, 61.83 | +1.34 | **yes** | MOABB within-session 5-fold, 8–32 Hz, 128 Hz |
| ShallowConvNet | 72.47 (MOABB benchmark) | 66.62 | 66.37, 66.9, 66.6 | -5.85 | **no** | MOABB within-session 5-fold, 8–32 Hz, 250 Hz |
| MI-BMInet | 76.03 (Wang et al. 2024) | 74.05 | 73.69, 74.65, 73.8 | -1.98 | **yes** | session 1 → 2, 0–3 s, 0.5–100 Hz |

Mean epochs run: EEGNet 249 of 300, ShallowConvNet 138 of 300 (early stopping on validation loss, patience 75), MI-BMInet 500 of 500.

#### Remaining protocol and definition differences


- EEGNet: our temporal kernel 65 / separable 17 at 128 Hz (MOABB Keras 64 / 16); max-norm 1 (depthwise) and 0.25 (classifier) as in Lawhern et al. (MOABB: 0.5 on the classifier); PyTorch default initialisation (Keras: Glorot uniform).
- ShallowConvNet: **no max-norm** (braindecode / study definition); MOABB's Keras version uses max-norm 2.0 per filter on both convolutions and 0.5 on the classifier. SafeLog clamps at 1e-6 (MOABB: clip to [1e-7, 1e4]). PyTorch default initialisation (Keras: Glorot uniform).
- MI-BMInet: all 288 trials per session used (the paper removes artefact-marked trials; MOABB does not expose the marks); 3 seeds (paper: 25 repetitions); fixed per-channel gain from session 1; dropout 0.5.

#### ShallowConvNet outside tolerance — pre-specified handling (decision_rules §5)

1. **Implementation check (done, documented).** `tests/test_shallow.py` shows the study's ShallowConvNet is numerically identical to braindecode 2.0 `ShallowFBCSPNet` (outputs equal to 1e-4 on three input shapes, same parameter count). No implementation error was found.
2. **Implementation kept.** The study's ShallowConvNet is not changed.
3. **Gap reported here.** −5.85 points against the MOABB reference.
4. **Home-residual flag:** not applicable — ShallowConvNet has no home recipe in the set. Because ShallowConvNet serves as descriptive side-evidence in mechanism module (i), its residual pattern will be reported with this gap attached.
5. **Diagnostic (descriptive, does not change the model):** the same protocol with MOABB's max-norm constraints added (`--archs ShallowConvNet+MOABB-maxnorm`, S4 row 5; seed 0 only; run 12–13 Nov 2026). Result: **74.15 %** (+1.68 against 72.47, within ±3), against 66.37 % for the study definition with the same seed. The constraint raised accuracy in all 18 subject × session units (by 3.1 to 17.1 points; median 6.9) and lengthened training (mean 243 vs 138 epochs before early stopping). The gap to the MOABB reference is therefore explained by the max-norm constraints of the Keras implementation that produced the reference figure, not by an implementation error.

**Implication recorded before the confirmatory runs.** The study's ShallowConvNet follows braindecode's definition (no max-norm); on BCI IV 2a within-session under the MOABB protocol this definition is about 8 points below the constrained variant. In the cross-subject binary pilot on PhysioNet-MI, the study definition was among the two best architectures under every recipe (S7_pilot_binary.md), so there is no sign of a comparable handicap in the confirmatory setting, but readers should know the size of the difference in this one.

# S8 Interval-width simulation (attached to Amendment 1)

Pilot: 105 subjects, median 45 trials/cell; confirmatory n_trials = 200; SESOI = ±2.0; 2000 simulations per scenario; category precedence: better first.

#### Per-pair SD decomposition (points)

| pair                        | recipe   |   mean_diff_pilot |   sd_pilot |   sd_binom_pilot |   sd_between | truncated   |   sd_binom_conf |   sd_conf |
|:----------------------------|:---------|------------------:|-----------:|-----------------:|-------------:|:------------|----------------:|----------:|
| EEGNet - ShallowConvNet     | R1       |             -1.93 |      11.19 |             8.34 |         7.46 | False       |            3.95 |      8.44 |
| EEGNet - ShallowConvNet     | R2       |             -0.64 |       9.91 |             7.81 |         6.1  | False       |            3.69 |      7.13 |
| EEGNet - ShallowConvNet     | R3       |              0.29 |       7.68 |             7.99 |         0    | True        |            3.78 |      3.78 |
| EEGNet - ShallowConvNet     | R4       |             -0.28 |       9.25 |             7.88 |         4.84 | False       |            3.73 |      6.11 |
| EEGNet - MI-BMInet          | R1       |             -3.2  |       9.2  |             8.24 |         4.11 | False       |            3.9  |      5.66 |
| EEGNet - MI-BMInet          | R2       |             -0.48 |       5.88 |             7.83 |         0    | True        |            3.71 |      3.71 |
| EEGNet - MI-BMInet          | R3       |             -0.18 |       5.98 |             7.98 |         0    | True        |            3.77 |      3.77 |
| EEGNet - MI-BMInet          | R4       |              0.36 |       5.61 |             7.91 |         0    | True        |            3.74 |      3.74 |
| EEGNet - CSOANet-v2         | R1       |              1.46 |      10.52 |             8.6  |         6.05 | False       |            4.07 |      7.29 |
| EEGNet - CSOANet-v2         | R2       |              1.75 |      10.06 |             8.01 |         6.09 | False       |            3.79 |      7.18 |
| EEGNet - CSOANet-v2         | R3       |              4.13 |      10.38 |             8.28 |         6.25 | False       |            3.92 |      7.38 |
| EEGNet - CSOANet-v2         | R4       |              3.32 |      10.9  |             8.16 |         7.23 | False       |            3.86 |      8.19 |
| ShallowConvNet - MI-BMInet  | R1       |             -1.27 |       8.53 |             8.1  |         2.68 | False       |            3.83 |      4.67 |
| ShallowConvNet - MI-BMInet  | R2       |              0.15 |       8.99 |             7.78 |         4.5  | False       |            3.68 |      5.82 |
| ShallowConvNet - MI-BMInet  | R3       |             -0.48 |       7.53 |             7.99 |         0    | True        |            3.78 |      3.78 |
| ShallowConvNet - MI-BMInet  | R4       |              0.64 |       9.34 |             7.89 |         5.01 | False       |            3.73 |      6.25 |
| ShallowConvNet - CSOANet-v2 | R1       |              3.39 |       8.55 |             8.47 |         1.2  | False       |            4.01 |      4.18 |
| ShallowConvNet - CSOANet-v2 | R2       |              2.39 |       9.76 |             7.95 |         5.65 | False       |            3.76 |      6.79 |
| ShallowConvNet - CSOANet-v2 | R3       |              3.84 |       9.85 |             8.3  |         5.31 | False       |            3.92 |      6.6  |
| ShallowConvNet - CSOANet-v2 | R4       |              3.61 |       8.69 |             8.13 |         3.07 | False       |            3.85 |      4.92 |
| MI-BMInet - CSOANet-v2      | R1       |              4.67 |       8.89 |             8.37 |         3    | False       |            3.96 |      4.96 |
| MI-BMInet - CSOANet-v2      | R2       |              2.23 |       9.43 |             7.98 |         5.03 | False       |            3.78 |      6.29 |
| MI-BMInet - CSOANet-v2      | R3       |              4.32 |       9.98 |             8.28 |         5.58 | False       |            3.92 |      6.82 |
| MI-BMInet - CSOANet-v2      | R4       |              2.97 |      11.37 |             8.16 |         7.92 | False       |            3.86 |      8.81 |

#### Scenario grid

|   D |   G |   n |   P_A |   P_A_sign |   P_A_equiv_only |   P_B1 |   P_B2 |   mean_n_definitive |   mean_changes_def_def |   mean_changes_sign_rev |   mean_changes_inc_def |   median_halfwidth |
|----:|----:|----:|------:|-----------:|-----------------:|-------:|-------:|--------------------:|-----------------------:|------------------------:|-----------------------:|-------------------:|
|   1 |   0 | 106 | 0.028 |      0     |            0.028 |  0.001 |  0.768 |               5.74  |                  0.038 |                   0     |                  9.959 |              1.783 |
|   1 |   1 | 106 | 0.75  |      0.234 |            0.516 |  0     |  0.212 |               7.442 |                  1.731 |                   0.236 |                 15.02  |              1.784 |
|   1 |   2 | 106 | 1     |      0.996 |            0.003 |  0     |  0     |              10.34  |                  5.074 |                   1.917 |                 17.818 |              1.782 |
|   1 |   3 | 106 | 1     |      1     |            0     |  0     |  0     |              12.616 |                  8.11  |                   4.195 |                 18.612 |              1.786 |
|   2 |   0 | 106 | 0.034 |      0     |            0.034 |  0.003 |  0.614 |               6.692 |                  0.044 |                   0     |                  9.98  |              1.786 |
|   2 |   1 | 106 | 0.758 |      0.144 |            0.614 |  0     |  0.104 |               9.234 |                  1.716 |                   0.144 |                 16.794 |              1.783 |
|   2 |   2 | 106 | 1     |      0.996 |            0.004 |  0     |  0     |              10.57  |                  3.557 |                   1.399 |                 18.016 |              1.783 |
|   2 |   3 | 106 | 1     |      1     |            0     |  0     |  0     |              11.634 |                  4.938 |                   3.12  |                 18.634 |              1.784 |
|   3 |   0 | 106 | 0.033 |      0     |            0.033 |  0.052 |  0.108 |              10.981 |                  0.039 |                   0     |                  8.39  |              1.784 |
|   3 |   1 | 106 | 0.686 |      0.068 |            0.617 |  0.016 |  0.012 |              12.252 |                  1.294 |                   0.068 |                 15.301 |              1.783 |
|   3 |   2 | 106 | 0.989 |      0.975 |            0.014 |  0     |  0     |              12.906 |                  3.532 |                   1.238 |                 16.622 |              1.784 |
|   3 |   3 | 106 | 1     |      1     |            0     |  0     |  0     |              13.098 |                  3.66  |                   2.356 |                 17.495 |              1.785 |
|   1 |   0 | 193 | 0.234 |      0     |            0.234 |  0.403 |  0     |              16.052 |                  0.498 |                   0     |                  9.93  |              1.312 |
|   1 |   1 | 193 | 1     |      0.725 |            0.275 |  0     |  0     |              17.809 |                  9.676 |                   0.764 |                 12.905 |              1.312 |
|   1 |   2 | 193 | 1     |      1     |            0     |  0     |  0     |              18.257 |                 14.489 |                   3.232 |                 10.657 |              1.31  |
|   1 |   3 | 193 | 1     |      1     |            0     |  0     |  0     |              19.658 |                 17.076 |                   5.386 |                  9.962 |              1.311 |
|   2 |   0 | 193 | 0.4   |      0     |            0.4   |  0.284 |  0     |              15.994 |                  0.916 |                   0     |                 11.233 |              1.311 |
|   2 |   1 | 193 | 1     |      0.46  |            0.54  |  0     |  0     |              17.488 |                  8.264 |                   0.471 |                 13.582 |              1.311 |
|   2 |   2 | 193 | 1     |      1     |            0     |  0     |  0     |              17.946 |                 11.575 |                   2     |                 11.966 |              1.312 |
|   2 |   3 | 193 | 1     |      1     |            0     |  0     |  0     |              18.183 |                 11.809 |                   5.743 |                 12.825 |              1.312 |
|   3 |   0 | 193 | 0.413 |      0     |            0.413 |  0.475 |  0     |              17.824 |                  0.89  |                   0     |                  8.536 |              1.311 |
|   3 |   1 | 193 | 0.997 |      0.198 |            0.799 |  0.003 |  0     |              18.108 |                  4.998 |                   0.213 |                 11.468 |              1.311 |
|   3 |   2 | 193 | 1     |      1     |            0     |  0     |  0     |              18.698 |                 10.398 |                   1.86  |                 10.808 |              1.31  |
|   3 |   3 | 193 | 1     |      1     |            0     |  0     |  0     |              18.622 |                 10.459 |                   4.346 |                 11.625 |              1.31  |

#### Decision quantity (decision_rules.md §4)

P(B2 | G = 0, D ∈ {1,2,3}, n = 106) = **0.496** → do NOT add

P(A | G = 0) = 0.032 (family-wise false-positive rate of the registered H3 rule)

P(A_sign | G = 0) = 0.000 (sign reversals only)

P(A from equivalent ↔ better only | G = 0) = 0.032

Reading rule (Amendment 1 §3.2): if H3 support comes only from equivalent ↔ better changes with no sign reversal, it is reported as 'difference below SESOI with stable direction', not as a change in ranking.

![S8 outcome probabilities](/mnt/user-data/uploads/recipe-x-arch/docs/S8/S8_outcome_probabilities.png)
