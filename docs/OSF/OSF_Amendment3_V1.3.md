---
title: "OSF 8st47 — Amendment 3: independent test of the cropping effect and of the corrected design"
subtitle: "V1.3, 28 November 2026 — made public on OSF before Dreyer2023 is downloaded"
---

# 1. Why this amendment

The confirmatory study (registration and Amendments 1–2) found that the authors' recipe R1 narrowed EEGNet's lead over CSOANet-v2, and that the recipe effect differed across architectures. Post hoc analyses traced these results to two defects in the authors' implementation and to one optimisation failure:

- R1's random crop, applied in training but not at test, harmed EEGNet, ShallowConvNet and MI-BMInet more than CSOANet-v2;
- the study's ShallowConvNet lacked the reference max-norm constraints, which mattered under the loss-selected recipes R2 and R3;
- CSOANet-v2 failed to train in 20 of 30 runs under R3.

With the crop removed (R1nc) and the constraints added (ShallowConvNet-MN), no pair of architectures changed definitive pairwise category across recipes, and without R3 neither a recipe effect nor an interaction remained. All of this is post hoc and comes from F1 (Lee2019-MI and Cho2017) only.

This amendment registers one directional test of the crop effect, and pre-specified secondary analyses of the corrected design, on a dataset that neither author has downloaded or analysed.

**Changes from drafts V1.0 (25 November 2026), V1.1 and V1.2 (28 November 2026), none made public.** Hypothesis HB (the R2 effect on the ShallowConvNet − CSOANet-v2 gap) is withdrawn, because the follow-up runs traced that effect to the missing max-norm constraints. ShallowConvNet is replaced by ShallowConvNet-MN, and the secondary recipe R2va by R4, so that the corrected design {R1nc, R2, R4} can be examined. V1.2 adds the data identifiers and the download source; V1.3, the posted version, adds the freeze commit and the dry-run date.

# 2. Data

- **Dataset.** Dreyer2023 (Dreyer et al., 2023, *Sci. Data* 10:580, doi:10.1038/s41597-023-02445-z; data at Zenodo, doi:10.5281/zenodo.8089820), loaded through MOABB 1.7.2 (`moabb.datasets.Dreyer2023`, the union of Dreyer2023A, B and C) with left- and right-hand motor-imagery trials only. MOABB 1.7.2 downloads the files from its OSF mirror of the Zenodo record (osf.io/8tdk5). The dataset has 87 participants (A 60, B 21, C 6) and 27 EEG electrodes sampled at 512 Hz (plus 3 EOG and 2 EMG channels, not used), with 240 trials (120 per class) in one session.
- **Authors' statement.** Neither author has downloaded, opened or analysed this dataset. Both authors confirmed this on 28 November 2026.
- **Exclusions.** As registered: subjects that cannot be loaded, or that keep fewer than 80 % of 240 trials, are excluded before training.
- **Preprocessing.** As in F1: common-average reference, band-pass 1–40 Hz, resampling to 250 Hz, and the MOABB default trial window for the dataset. Gain is estimated per channel on the training subjects.
- **Evaluation.** Cross-subject, 10 folds (split seed 0), with 10 % of the training subjects used for validation. Ten folds give ten clusters for inference on shared training runs.
- **Seeds.** 0, 1 and 2 for every run.

# 3. Factors

- **Architectures.** EEGNet-8,2, ShallowConvNet-MN (the study's ShallowConvNet with the Keras reference max-norm constraints: convolutional filters 2.0, dense layer 0.5), MI-BMInet and CSOANet-v2, otherwise exactly as in the confirmatory study.
- **Recipes.** R1; R1nc (R1 without the crop, otherwise identical); R2; R4. All as in the confirmatory study.
- **Run count.** 4 architectures × 4 recipes × 3 seeds × 10 folds = 480 runs. Every trained model is retained.

# 4. Primary hypothesis

Contrasts are computed per subject on seed-averaged float32 balanced accuracy, in percentage points.

**HA — the crop narrows EEGNet's lead over CSOANet-v2.**

d_A = (EEGNet − CSOANet-v2) under R1nc − (EEGNet − CSOANet-v2) under R1.

Predicted: d_A > 0. In F1, d_A was +4.78 points (95 % CI 2.66 to 6.88, wild cluster bootstrap).

**Test and decision rule.**

- **Test statistic.** The mean of d_A, with a cluster-robust (CR1) standard error over the 10 folds.
- **p-value.** From a wild cluster bootstrap with the null imposed (Webb six-point weights, 9,999 draws, seed 20261126), two-sided, α = 0.05. There is one primary hypothesis, so no multiplicity correction is applied.
- **Supported:** p < 0.05 and a positive mean.
- **Contradicted:** p < 0.05 and a negative mean.
- **Inconclusive:** any other result. If the 90 % bootstrap-t interval also lies within ±2 points (the study's SESOI), the effect is reported as smaller than the SESOI.

# 5. Secondary analyses (pre-specified, reported, no decision)

On the corrected design {R1nc, R2, R4} × {EEGNet, ShallowConvNet-MN, MI-BMInet, CSOANet-v2}:

- **S1.** Recipe main effect and recipe × architecture interaction: likelihood-ratio tests with crossed random effects for subject and training run (fold × architecture × recipe), ML, best converged fit over four optimisers; percentage and empirical-logit scales. Expectation from F1: neither significant.
- **S2.** Pairwise categories by the registered H3 rule (Bonferroni over 6 pairs × 3 recipes; SESOI 2 points) and the number of definitive category changes across recipes. Expectation from F1: none.

Also reported: R1nc − R1 for each architecture; HA on the empirical-logit scale; the number of flagged runs under the training-failure rule of Amendment 1.

# 6. Code

- **Driver:** `train/independent_dreyer.py`. It calls the frozen confirmatory driver, using the recipe and architecture registrations of `train/explore_rev6.py`.
- **Analysis:** `analysis/independent_analysis.py` (tested on synthetic data).
- **Freeze.** Both files, and the files they import, are frozen at tag `v2.0-independent-frozen`, commit 79f8e5a5d5d37680815d3d546d5ae5f651001e27 (github.com/jabarrick/recipe-x-arch), committed and pushed on 28 November 2026 before this amendment was made public.
- **Dry run.** The driver was tested before any download, using the PhysioNet pilot cache and the driver's `--pilot-dryrun` option, on 28 November 2026: all 480 runs (20 pilot subjects, 2 epochs) completed without error, and the analysis script ran on its output. No Dreyer2023 file was downloaded.

# 7. Timing

1. Dry run, then freeze the code.
2. Make this amendment public.
3. Download Dreyer2023.
4. Run and analyse.

Deviations are listed in the manuscript's supplementary material S4, as before.
