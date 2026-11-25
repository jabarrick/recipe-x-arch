---
title: "OSF 8st47 — Amendment 3: independent confirmatory test of the cropping and R2 effects"
subtitle: "Draft V1.0, 25 November 2026 — to be made public on OSF before Dreyer2023 is downloaded"
---

# 1. Why this amendment

The confirmatory study (registration and Amendments 1–2) found that the authors' recipe R1 narrowed EEGNet's lead over CSOANet-v2. Post hoc analyses attributed most of this to R1's random crop. In R1, the crop is applied in training but not at test. They also found that a smaller interaction remained between R2 and the other recipes, mainly for ShallowConvNet. Both findings are post hoc, come from F1 only, and partly share subjects with the analysis that generated them.

This amendment registers a test of both findings on a dataset that neither author has downloaded or analysed.

# 2. Data

- **Dataset.** Dreyer2023 (Dreyer et al., 2023, *Sci. Data* 10:580), loaded through MOABB 1.7.2 with left- and right-hand motor-imagery trials only. It has 87 participants and 27 electrodes sampled at 512 Hz, with 240 trials (120 per class) in one session.
- **Authors' statement.** Neither author has downloaded, opened or analysed this dataset. [Both authors to confirm before posting.]
- **Exclusions.** As registered: subjects that cannot be loaded, or that keep fewer than 80 % of 240 trials, are excluded before training.
- **Preprocessing.** As in F1: common-average reference, band-pass 1–40 Hz, resampling to 250 Hz, and the MOABB default trial window for the dataset. Gain is estimated per channel on the training subjects.
- **Evaluation.** Cross-subject, 10 folds (split seed 0), with 10 % of the training subjects used for validation. Ten folds are used, instead of five, to give ten clusters for inference on shared training runs.
- **Seeds.** 0, 1 and 2 for every run.

# 3. Factors

- **Architectures.** EEGNet-8,2, ShallowConvNet, MI-BMInet and CSOANet-v2, exactly as in the confirmatory study.
- **Recipes:**
  - R1;
  - R1nc, which is R1 without the crop;
  - R2;
  - R2va, which is R2 with the checkpoint chosen by the highest validation balanced accuracy over all epochs (secondary).
- **Run count.** 4 architectures × 4 recipes × 3 seeds × 10 folds = 480 runs. Every trained model is retained.

# 4. Hypotheses

Contrasts are computed per subject on seed-averaged balanced accuracy, in percentage points.

**HA — the crop narrows EEGNet's lead.**

d_A = (EEGNet − CSOANet-v2) under R1nc − (EEGNet − CSOANet-v2) under R1.

Predicted: d_A > 0. In F1, d_A was +4.78 points (Lee2019-MI and Cho2017), and +3.71 points in the Cho2017 replication.

**HB — R2 interacts with ShallowConvNet.**

d_B = (ShallowConvNet − CSOANet-v2) under R1nc − (ShallowConvNet − CSOANet-v2) under R2.

Predicted: d_B > 0. In F1, d_B was +2.65 points.

# 5. Test and decision rules

- **Test statistic.** The mean of d, with a cluster-robust (CR1) standard error over the 10 folds.
- **p-value.** From a wild cluster bootstrap with the null imposed (Webb six-point weights, 9,999 draws, seed 20261126), two-sided. Holm correction over {HA, HB} at α = 0.05.
- **Supported:** Holm-adjusted p < 0.05 and a positive mean.
- **Contradicted:** Holm-adjusted p < 0.05 and a negative mean.
- **Inconclusive:** any other result. If the 90 % bootstrap-t interval also lies within ±2 points (the study's SESOI), the effect is reported as smaller than the SESOI.

# 6. Secondary analyses (reported, no decision)

- R1nc − R1 for each architecture.
- HB with R2va in place of R2.
- All contrasts on the empirical-logit scale.
- The number of flagged runs, using the training-failure rule of Amendment 1.

# 7. Code

- **Driver:** `train/independent_dreyer.py`. It calls the frozen confirmatory driver, using the recipe registrations of `train/explore_rev6.py`.
- **Analysis:** `analysis/independent_analysis.py`.
- **Freeze.** Both files are frozen at tag `v2.0-independent-frozen`, committed before this amendment is made public. The commit hash will be recorded here.
- **Dry run.** The driver is tested before any download, using the PhysioNet pilot cache and the driver's `--pilot-dryrun` option.

# 8. Timing

The order is:
1. Freeze the code.
2. Make this amendment public.
3. Download Dreyer2023.
4. Run.

Deviations are listed in the manuscript's supplementary material S4, as before.
