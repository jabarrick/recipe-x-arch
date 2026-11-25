---
title: "A train/test-inconsistent augmentation biased a shared-recipe comparison: a preregistered case study in compact motor-imagery EEG decoders"
---

Yu Gao^1,2,\*^ and Yimin Zhang^1^

^1^ Doctoral Programme in Electrical and Computer Engineering (PDEEC), Faculty of Engineering, University of Porto, Rua Dr. Roberto Frias, 4200-465 Porto, Portugal

^2^ Doctoral Programme in Health Data Science (HEADS), Faculty of Medicine, University of Porto, Porto, Portugal

^\*^ Author to whom any correspondence should be addressed.

E-mail: Yu_Gao@alumni.baylor.edu

ORCID iDs: Yu Gao [to add]; Yimin Zhang [to add]

# Abstract

*Objective.* Compact motor-imagery EEG decoders are often compared under one shared training recipe. We tested, in a case where the shared recipe had been developed with one of the compared models, whether the recipe changed the comparison. *Approach.* In a preregistered 4 × 4 design (OSF 8st47), EEGNet, ShallowConvNet, MI-BMInet and the first author's CSOANet-v2 were each trained under the authors' heavy-augmentation recipe, two published plain recipes and the heavy recipe without augmentation, on Lee2019-MI and Cho2017 (106 subjects, cross-subject, three seeds). Because test subjects in a fold share trained networks, intervals were also computed with dataset × fold clustering. *Main results.* As registered, the recipe effect differed across architectures, and the authors' recipe narrowed EEGNet's lead over CSOANet-v2 by 3.5 percentage points of balanced accuracy (cluster-robust 95% CI 1.4–5.7). No pair of architectures reversed order, and the one registered change in a pairwise conclusion did not survive the cluster correction. Exploratory analyses attributed most of the narrowing to a random crop applied in training but not at test: removing it raised the baselines by 3.7–5.9 points and CSOANet-v2 by 1.1, and the narrowing disappeared. Among the remaining recipes, the interaction was not significant once training runs were modelled (p = 0.13). *Significance.* A shared recipe can bias a comparison through a single implementation detail. Augmentations should be checked for train/test consistency, the recipe's origin stated, and uncertainty computed over training runs as well as subjects.

**Keywords:** motor imagery, EEG decoding, training recipe, data augmentation, benchmarking, preregistration, compact neural networks

# 1. Introduction

EEG decoders are compared in one of two ways. Either each model is trained with its published recipe, so that architecture and recipe change together, or all models share one recipe chosen by whoever runs the comparison [1]. A recipe here means everything that is not the model or the data: optimiser, schedule, augmentation and model-selection rule. The shared recipe is presented as the fairer choice, because it removes the proposer's tuning advantage. It assumes, however, that the recipe treats all architectures alike. If it does not, a shared recipe does not remove a bias; it chooses one.

There are reasons to doubt the assumption. Pipeline choices interact with models in EEG decoding, and equal tuning budgets have shrunk reported architectural progress elsewhere in machine learning (Section 2). The question is sharpest when the shared recipe was developed together with one of the compared models. That is our situation: the first author developed both the CSOANet decoder [2] and the heavy-augmentation recipe R1 used to train it. In our four-class pilots, changing only the recipe moved EEGNet by 6.6–8.7 points and other architectures by less. A later binary pilot showed no such asymmetry (Section 3.3), which argued against our own prediction. We therefore preregistered a test on two datasets neither author had analysed, with the direction of the main contrast and the decision rules set against our own model.

We ask three questions:

- **RQ1.** Does the recipe change accuracy?
- **RQ2.** Does the change depend on the architecture, and does a recipe developed with one compared model favour that model?
- **RQ3.** Do pairwise conclusions change with the recipe?

The study is a case study. It covers one shared recipe developed with one of the compared models, four compact convolutional architectures and two datasets. Its lesson concerns how such a recipe can bias a comparison, not how often shared recipes do so. Secondary analyses (an equal-budget search, within-subject evaluation and int8 quantisation) are reported briefly.

# 2. Related work

EEGNet [3], ShallowConvNet [4], MI-BMInet [5] and CSOANet [2] are compact convolutional decoders. Each was published with its own recipe, and they are usually compared under the proposer's recipe for the proposed model. MOABB [6–8] and braindecode [9] standardised data, splits and implementations, but not the recipe.

Pipeline choices interact with models. Preprocessing shifts decoding performance differently across classifiers [10], and can flip up to 42 % of single-trial predictions [11]. Minimal preprocessing often beats heavy pipelines [12]. The benefit of EEG augmentation depends on architecture and data size [13].

Outside EEG, equal tuning budgets have shrunk or erased apparent architectural progress [14–18]. A modern recipe lifted a 2015 ResNet to the level of later architectures [19]. Seed variance rivals many reported differences [20, 21], and methods tend to do best under the conditions they were developed in [22, 23].

Quantised deployments report int8 accuracy for one model under its own recipe [5, 24].

# 3. Methods

## 3.1. Design, data and training

Four architectures are each trained under four recipes, with three seeds per cell. Hypotheses, decision rules and analysis code were fixed before download (Section 3.3). Two public datasets, loaded through MOABB 1.7.2 with left- and right-hand trials only, form the confirmatory family F1. F1 pools Lee2019-MI (54 subjects; [25]) and Cho2017 (52 subjects; [26]), each evaluated cross-subject with 5 folds. The secondary family F2 uses Lee2019-MI within subjects (session 1 → session 2) and is an independent Holm [27] family.

Preprocessing is identical for every cell:

- common-average reference;
- band-pass 1–40 Hz;
- resampling to 250 Hz;
- epoch 0–4 s after the cue (0–3 s for Cho2017);
- per-channel standardisation estimated on training data.

**Splits.** Cross-subject, subjects are split into 5 folds with seed 0, and 10 % of each fold's training subjects (about four per dataset) form the validation set used for checkpoint selection. Within-subject, 20 % of session 1 is used for validation and session 2 is the test set. One split seed was used, so split variance is not estimated.

**Failures.** A run whose loss becomes non-finite is retried once with seed + 100. It is flagged as a training failure, for a sensitivity analysis only, if the loss is still non-finite or its training accuracy at the selected checkpoint is ≤ 55 %.

**Architectures** are EEGNet-8,2 (3.8 k parameters), ShallowConvNet (105 k), MI-BMInet (6.7 k) and CSOANet-v2 (3.8 k). Each keeps its own weight constraints under every recipe (supplementary material table S10.1). Before the confirmatory runs, each baseline was checked against a pre-named reference figure on BCI IV 2a, with a tolerance of ±3 points. EEGNet and MI-BMInet passed. ShallowConvNet missed by 5.9 points and was kept, as pre-specified, after verification against braindecode. With the max-norm constraints of the Keras reference added, it reached the reference (supplementary material S7). The study's ShallowConvNet therefore has no max-norm constraints, which bears on the results for that architecture (Section 5.5).

**Recipes** are applied identically to every architecture (Table 1). Epochs and batch size are fixed per protocol (cross-subject: 100 epochs, batch 64; within-subject: 500 epochs, batch 32) unless a recipe sets its own batch size.

- R1 is the authors' recipe. Its crop keeps 90 % of the window at one random offset per batch, moves it to the start and zero-fills the rest. Test windows are never cropped.
- R2 and R3 are the published recipes of EEGNet and MI-BMInet.
- R4 is R1 without augmentation.

The recipes differ in several components at once. Only R1 against R4 (augmentation) and, post hoc, R1 against R1nc (the crop) isolate a single component.

**Home pairs.** These are EEGNet + R2, MI-BMInet + R3 (cross-subject only) and CSOANet-v2 + R1. Only the last is exact. Protocol unification shortened EEGNet's published 500 epochs to 100. R3 is MI-BMInet's cross-subject recipe, published for a different network configuration (supplementary material S1).

**Table 1.** Training recipes (constant across architectures).

| Recipe | Optimiser / schedule | Loss / augmentation | Model selection | Origin |
|---|---|---|---|---|
| R1 — heavy augmentation | AdamW, lr 1e-3, weight decay 1e-3, cosine to 1e-5 | CE with mixup 0.2 and label smoothing 0.1; Gaussian noise 0.02; circular time shift ±5 samples; random crop 90 % (one offset per batch, shifted to the window start, last 10 % zero-filled; not applied at test); channel dropout 10 % | early stopping on validation accuracy, patience 15 (cross-subject) / 80 (within-subject) | [2] |
| R2 — plain, constant lr | Adam, lr 1e-3, constant | CE, no augmentation | lowest validation loss | EEGNet [3] |
| R3 — plain, step schedule | Adam, lr 1e-2, ×0.1 at 40 % and 80 % of epochs; batch 16 | CE, no augmentation | lowest validation loss | MI-BMInet [5] |
| R4 — R1 without augmentation | as R1 | CE, no augmentation | as R1 | control |

**Condition T** gives each architecture the same search budget instead of a shared recipe. For each architecture and dataset, 24 configurations from a fixed space (supplementary material S1) are scored on the fold-1 training and validation subjects, with two seeds, and the best is retrained on all folds. Because those subjects are tested in folds 2–5, T is also evaluated on the 22 fold-1 test subjects only. A leakage-free variant, T′, applies configurations selected the same way on the pilot dataset PhysioNet-MI.

**Quantisation and outcomes.** Every model is evaluated in float32 and after int8 post-training quantisation (ONNX Runtime QDQ, MinMax calibration on 256 training trials shared within a fold). The primary outcome is seed-averaged subject-level float32 balanced accuracy. The frozen confirmatory driver did not retain trained models.

## 3.2. Statistical analysis

**Model.** Accuracy is modelled with fixed effects for dataset, architecture, recipe and architecture × recipe, and a random intercept per subject. Fixed effects are tested by likelihood-ratio tests. Interaction terms are sum-coded, so each is the double-centred residual of its cell.

**Smallest effect size of interest (SESOI).** It is 2 points, fixed at registration as roughly the change a new seed produces in a single configuration. Seed variability turned out larger: the median subject-level seed SD exceeded 2 points in 13 of 16 F1 cells (supplementary material S3). The SESOI is therefore a convention fixed in advance, not an estimate of noise.

**Hypotheses.** H1, H2 and H2b are Holm-corrected within each family.

- **H1:** recipe main effect.
- **H2:** recipe × architecture interaction.
- **H2b:** per subject, (EEGNet under R2 − under R1) − (CSOANet-v2 under R2 − under R1). Supported only if significant and positive.
- **H3:** each of the 6 architecture pairs under each recipe gets a Bonferroni-adjusted 99.79 % interval, classified by a fixed precedence. An interval that excludes zero is *better* (for one side), even if it lies within ±2. An interval that includes zero and lies within ±2 is *equivalent*. Any other interval is *inconclusive*. The categories are therefore mutually exclusive. H3 is supported if a pair falls into two different definitive categories under two recipes.
  - The outcome labels were fixed in Amendment 1. **A:** support including a sign reversal. **A′:** support only from *equivalent* ↔ *better* changes, read as differences below the margin whose direction is stable, not as a change of ranking. **B1:** no support with ≥ 16 of 24 intervals definitive (stable). **B2:** fewer than 8 definitive (uninformative). **Intermediate:** anything else.
  - A stable outcome was close to unreachable at this sample size (supplementary material S8). Because the categories dichotomise continuous differences, we also report the underlying differences-in-differences.
- **H4 (secondary):** the same classification for R1 against T.

**Home-recipe analysis (secondary).** H2b and co-adaptation predict the same sign. To separate them, the home advantage of an architecture is the interaction residual of its home cell. The rules are asymmetric against our own model:

- **(i)** If CSOANet-v2's home residual is significantly positive, the result is consistent with co-adaptation, and H2b is not read as robustness.
- **(ii)** If it is equivalent to zero (TOST against ±2; [28]) and its residual range is the smallest in ≥ 80 % of bootstrap resamples, robustness is supported.
- **(iii)** Otherwise, the mechanism is unresolved.

**Registered sensitivity analyses** repeat the tests on int8 accuracy, on each dataset separately, without flagged runs, and under alternative specifications, including an aligned-rank-transform ANOVA [29].

**Post hoc analyses** were added after the results were known. They are labelled wherever they appear and are listed in supplementary material S9 and S4:

- **Training runs as clusters.** All test subjects of a cross-subject fold are scored by the same networks, so their errors are correlated within each dataset × fold cluster (10 clusters in F1). H1 and H2 are refitted with crossed random effects for subject and training run, taking the best of several optimisers. Single contrasts are re-estimated with a random intercept per cluster and t on 9 degrees of freedom. The key contrasts are also tested with a wild cluster bootstrap (Webb weights, 9,999 draws). This bootstrap is better suited to few clusters than a pairs cluster bootstrap with a normal critical value [30].
- **Logit scale.** Interactions on bounded accuracies depend on where each architecture sits on the scale. The main tests are therefore repeated on the empirical logit of each subject's seed-averaged correct count.
- **A search-space regression** of the condition-T validation accuracies.
- **A rerun of F1 under R1 without cropping (R1nc).**
- **Tests of all 2 × 2 interaction contrasts among R1nc, R2 and R4.**

## 3.3. Preregistration, pilots and deviations

The registration (1 November 2026) and Amendments 1 and 2 (13 and 15 November) were public before the datasets were downloaded on 15 November. The runs took about 126 GPU-hours on one laptop GPU.

Two pilots preceded the confirmatory study (supplementary material S7). A four-class pilot motivated the hypotheses. A binary pilot (PhysioNet-MI, 105 subjects) was run after the decision rules were committed. In it, moving from R1 to R2 improved EEGNet by 4.2 points and CSOANet-v2 by 3.9, giving an H2b contrast of +0.29 points. It changed no rule, but it lowered our prior for H2b. PhysioNet-MI is the dataset on which CSOANet and R1 were developed.

Supplementary material S4 lists every change after the code freeze. None alters a hypothesis or threshold. In particular, data first loaded under MOABB 1.5.0 lacked one Lee2019-MI session, and were reloaded under the registered 1.7.2 before any training.

# 4. Results

## 4.1. Overview and registered tests

F1 comprised 106 subjects and 480 runs. Forty-one runs, all under R3, were flagged and, as registered, kept:

- **ShallowConvNet, all 30 runs.** All diverged after 12–34 epochs. Their last checkpoints before divergence performed in the range of the architecture's other cells.
- **CSOANet-v2, 11 of 30 runs.** These failed to learn, and eight more runs of that cell were near chance.

Two diagnostics matter later (supplementary material S3):
- Under R1, training accuracy was 3–10 points below R4 for every architecture.
- Under R2, ShallowConvNet's selected checkpoint came very early (median epoch 6.5). For the other architectures it came at epoch 20 to 63.5.

![**Figure 1.** Subject-level float32 balanced accuracy in each architecture × recipe cell of F1 (○ Lee2019-MI, △ Cho2017; N = 106). Black: cell mean with 95 % CI; dashed line: chance; framed panels: home cells; filled markers: the subject's cell contains a flagged run.](/home/claude/work/s4/figs/fig1_cells_V0.2.png)

The recipe changed accuracy (H1: χ²(3) = 160.5, Holm p = 2.9 × 10⁻³⁴), and the change depended on the architecture (H2: χ²(9) = 414.7, p = 3.0 × 10⁻⁸³). These registered tests treat subjects as independent. With crossed subject and training-run effects (post hoc), the statistics fall to χ²(3) = 29.1 (p = 2 × 10⁻⁶) and χ²(9) = 95.3 (p = 10⁻¹⁶). On the logit scale they are χ²(3) = 31.9 and χ²(9) = 75.4 (both p < 10⁻⁶).

R1 was the worst recipe for every baseline; R3 was the worst for CSOANet-v2 (Table 2). Without the R3 column, the interaction remained significant (crossed χ²(6) = 21.0, p = 0.002), but its share of the fixed-effect variance fell from 30 % to 10 %. Much of H2 therefore reflects R3's numerical instability.

**Table 2.** Cell means, F1, float32 (%). In parentheses: interaction residual of the registered 4 × 4 table R1–R4 (points); home cells in bold. R1nc is R1 without cropping (post hoc, Section 4.5) and is not part of the registered table.

| | R1 | R2 | R3 | R4 | R1nc |
|:------------------|:-----------|:-----------|:-----------|:-----------|:------|
| EEGNet | 70.6 (−2.58) | **75.9** (−0.28) | 76.6 (+3.00) | 76.8 (−0.14) | 76.5 |
| ShallowConvNet | 69.8 (−0.30) | 71.9 (−1.26) | 71.6 (+1.09) | 74.4 (+0.47) | 73.8 |
| MI-BMInet | 73.2 (−0.68) | 76.1 (−0.82) | **76.9** (+2.60) | 76.6 (−1.11) | 77.0 |
| CSOANet-v2 | **69.1** (+3.55) | 70.9 (+2.35) | 59.3 (−6.69) | 70.1 (+0.78) | 70.2 |

![**Figure 2.** Left: cell means with 95 % CIs by recipe (points offset horizontally by architecture). Right: interaction residuals of the 4 × 4 table of cell means (points); framed cells are home cells.](/home/claude/work/s4/figs/fig2_interaction_V0.4.png)

H2b was supported. From R1 to R2, EEGNet gained 5.29 points and CSOANet-v2 1.79, so c̄ = 3.50 (95 % CI 2.29, 4.71; Holm p = 9.8 × 10⁻⁸).
- **With shared training runs accounted for,** the interval is (1.39, 5.67) with a cluster intercept and t(9), and (1.44, 5.51) with the wild cluster bootstrap (p = 0.002).
- **On the logit scale,** c̄ is 0.22 (0.12, 0.32).
- **By dataset,** the contrast was significant in each.
- **With R4 in place of R2,** it was 5.20 (cluster-robust 2.18, 8.35).

![**Figure 3.** Per-subject H2b contrast c[s] = (R2 − R1) for EEGNet minus (R2 − R1) for CSOANet-v2, by dataset and pooled, with mean and 95 % CI. Shaded band: ±2-point SESOI.](/home/claude/work/s4/figs/fig3_h2b_V0.4.png)

## 4.2. Home-recipe analysis

CSOANet-v2's home residual was 3.55 (3.03, 4.07). It was significant and not equivalent to zero, and its residual range was the largest of the four (supplementary material table S10.3). The registered outcome is therefore (i), consistent with co-adaptation. MI-BMInet's home residual was also significant (2.60); EEGNet's was not (−0.28).

Three features of the table limit what (i) can mean:

- **R3 inflates it.** CSOANet-v2's R3 residual (−6.69) raises its other residuals, because residuals in a row sum to zero. Without R3 (post hoc), its home residual was 1.32, inside the margin.
- **Non-home cells are as large.** EEGNet's residual under R3, which is not its home recipe, was +3.00.
- **The raw means point the other way.** CSOANet-v2's best recipe was R2, not R1. The residual says that R1 hurt the baselines more, not that CSOANet-v2 did best under R1.

## 4.3. Pairwise conclusions (H3) and search (H4)

Twenty of 24 intervals were definitive (median half-width 1.80; Table 3, Figure 4). H3 was supported, with outcome A′. MI-BMInet was *better* than EEGNet under R1 (−2.62; −4.13, −1.11) and *equivalent* under the other recipes. No pair reversed. Relative to R1, every other recipe narrowed MI-BMInet's lead by 2.3–2.9 points (R2: 2.44, wild cluster bootstrap 1.13, 3.72). Under R1, CSOANet-v2 could not be distinguished from EEGNet or ShallowConvNet; under the other recipes, EEGNet was clearly ahead of it.

**Table 3.** Category matrix, F1, float32 (code: mean difference; A = first better, B = second better, E = equivalent, I = inconclusive). Second block: int8.

| Pair | R1 | R2 | R3 | R4 | Sign rev. | E ↔ better | I ↔ def. |
|---|---|---|---|---|---|---|---|
| EEGNet − ShallowConvNet | I +0.79 | A +4.05 | A +4.98 | A +2.46 | 0 | 0 | 3 |
| EEGNet − MI-BMInet | B −2.62 | E −0.18 | E −0.33 | E +0.25 | 0 | 3 | 0 |
| EEGNet − CSOANet-v2 | I +1.49 | A +4.99 | A +17.30 | A +6.70 | 0 | 0 | 3 |
| ShallowConvNet − MI-BMInet | B −3.41 | B −4.23 | B −5.30 | B −2.21 | 0 | 0 | 0 |
| ShallowConvNet − CSOANet-v2 | I +0.70 | I +0.94 | A +12.33 | A +4.24 | 0 | 0 | 4 |
| MI-BMInet − CSOANet-v2 | A +4.12 | A +5.17 | A +17.63 | A +6.45 | 0 | 0 | 0 |
| **Totals (of 36)** | | | | | **0** | **3** | **10** |
| *int8:* EEGNet − MI-BMInet | B | E | **B** | E | 0 | 4 | 0 |

*Outcome label: A′ (H3 supported by equivalent ↔ better changes only). In int8, one category differs from float32 (EEGNet − MI-BMInet under R3: E → B); all other int8 categories are identical to float32.*

![**Figure 4.** The 24 pairwise differences (first minus second architecture) with Bonferroni-adjusted 99.79 % CIs (registered, subjects independent), by recipe, on a common axis. Arrows: intervals beyond the axis, with their mean and category. Light bars under each interval: the clustered interval (post hoc; random intercept per dataset × fold cluster, t(9)), coloured by its category. Shaded band: ±2-point equivalence margin. Labels: A first better, B second better, E equivalent, I inconclusive.](/home/claude/work/s4/figs/fig4_h3_intervals_V0.3.png)

**Clustering (post hoc).** The H3 result depends on how the ten clusters are treated:

- With a normal critical value, or with a pairs cluster bootstrap (which also uses one), 20 intervals stay definitive and the outcome is A′. Both approaches are expected to over-reject with ten clusters [30].
- With t(9), only 13 stay definitive and the outcome is intermediate, so H3 would not be supported.

We take t(9) as the reference. Under it, the registered change in a pairwise conclusion does not hold.

**H4.** Condition T kept the R1 order, and its only definitive change was the same EEGNet − MI-BMInet change. H4 was therefore supported only in the A′ sense, and T′ and the fold-1-only analysis agreed. T beat R1 by 1.0–5.7 points but gained nothing against R4 (−1.01 to +0.03; supplementary material table S10.4). Eleven of the 24 search configurations used the defective crop, and the search was scored on about four validation subjects. H4 therefore says little about equal-budget search in general.

## 4.4. Secondary family, int8 and sensitivity

**F2 (within subjects).** No hypothesis was supported (H1 p = 0.11; H2 p = 0.0501; H2b c̄ = 0.78, −3.05 to 4.62), and as registered these results are inconclusive. Accuracies were 58–66 %, and 124 of 2,592 runs were flagged (supplementary material table S10.5). We do not interpret F2 further.

**int8.** The decisions were unchanged, and 95.5 % of predictions agreed with float32. EEGNet was only partly quantised (5 of 9 operators), so int8 comparisons involving it are not on equal terms. The single category change, EEGNet − MI-BMInet under R3, came from that architecture (supplementary material S5).

**Sensitivity.** Random slopes, fold as a fixed effect, seed as a random effect and the aligned-rank transform left the decisions unchanged. Excluding flagged runs empties the whole ShallowConvNet × R3 cell, so that analysis used the pre-specified model path; its decisions were also unchanged. The two datasets agreed on H1, H2 and H2b but not on the H3 label (Lee2019-MI B1, Cho2017 A′), so the pooled H3 result is dataset-dependent.

## 4.5. Exploratory analyses (post hoc)

**Cropping.** In the search data, cropping was the dimension with the largest effect. It lowered validation accuracy by 9.5 points for EEGNet, 6.8 for MI-BMInet, 5.8 for ShallowConvNet and 1.4 (−0.5, 3.4) for CSOANet-v2. Checkpoint selection on validation loss (−1.8) and batch 16 (−1.6) also mattered.

Switching off only the crop (R1nc − R1) raised accuracy for every baseline. The intervals are wild cluster bootstrap 95 % CIs:

- EEGNet: +5.86 (3.75, 8.04);
- ShallowConvNet: +3.98 (2.59, 5.42);
- MI-BMInet: +3.74 (2.01, 5.45);
- CSOANet-v2: +1.07 (−0.07, 2.21), not significant.

R1nc matched R4 for every architecture (Table 2), and training accuracy rose to the R4 level. The R1 deficit was therefore a train/test mismatch, not under-fitting. Removing the crop widened EEGNet's lead over CSOANet-v2 by 4.78 points (2.74, 6.87), which is most of the 5.2-point narrowing that R1 produced relative to R4. On the logit scale the widening was 0.26 (0.12, 0.39), so it is not an artefact of CSOANet-v2 being the least accurate architecture. How the crop does harm (the mismatch itself, the shift or the zero padding) was not tested (Section 5.6).

Three checks bear on the overlap between the analysis that suggested the hypothesis and its test:

- **Fold-1 test subjects.** On the 22 fold-1 test subjects, who were not used by the search regression, the effects were smaller for EEGNet (3.52 against 5.86) and MI-BMInet (2.40 against 3.74), but larger for ShallowConvNet (4.63 against 3.98) and CSOANet-v2 (1.33 against 1.07). The gap change was 2.19 (0.12, 4.26) against 4.78. These subjects form two clusters only, so their intervals omit training-run variance.
- **Cho2017 half.** This half was run after the analysis script was written. It comes from the same set of runs, and the Lee2019-MI half had already been inspected, so it is a check with a fixed analysis rather than an independent replication. Its gap change was 3.71 (1.93, 5.48).
- **Dataset heterogeneity.** The crop effect was about twice as large in Lee2019-MI as in Cho2017 (EEGNet 7.9 against 3.7; CSOANet-v2 2.1 against 0.0). The longer Lee2019-MI window (4 s against 3 s) is one candidate explanation; it was not tested.

**Remaining interaction.** On the table {R1nc, R2, R4} (supplementary material S9.9), every residual was within ±1 point.
- **Omnibus test.** It was significant with subjects as the unit (χ²(6) = 23.7). It was not significant with crossed subject and training-run effects (χ²(6) = 9.9, p = 0.13) or on the logit scale (χ²(6) = 4.9, p = 0.56). An earlier version reported χ²(6) = 4.6 for the crossed test; that came from a fit that had not converged.
- **The 18 2 × 2 contrasts (supplementary material S9.11).** Six had 95 % intervals excluding zero. After Bonferroni adjustment only two did, both ShallowConvNet between R4 and R2: against CSOANet-v2, 3.29 (2.08, 4.50); against MI-BMInet, 2.01 (0.95, 3.07). On the logit scale, four excluded zero before adjustment and none after.
- **Equivalence.** No contrast was equivalent to zero after adjustment.

The remaining interaction is therefore uncertain. If it is real, it concerns ShallowConvNet under R2. With R1nc in place of R1, the H2b contrast was −1.29 (wild cluster bootstrap −2.25, −0.35); on the logit scale it was −0.04 (−0.11, 0.04).

# 5. Discussion

## 5.1. Was the shared recipe neutral?

As registered, no. The recipe developed with our model narrowed that model's gap to EEGNet by 3.5 points relative to a published recipe, and the interval stays clear of zero when shared training runs are taken into account. A comparison run under our recipe alone would have shown our model at its closest to the others.

The exploratory analyses attribute most of this to one implementation detail: a crop applied in training but not at test, which hurt the baselines far more than our model. Without it, R1 behaved like R4 and H2b no longer favoured our model. The R1 − R4 contrast is therefore an effect of one implementation, not of heavy augmentation.

What remains among the other recipes is uncertain. It is not significant as an omnibus once training runs are modelled. The individual contrasts that survive adjustment all involve ShallowConvNet under R2, where two candidate causes are confounded: R2 selects an early checkpoint by validation loss, and our ShallowConvNet lacks max-norm constraints.

The study thus shows how a shared recipe can bias a comparison: through one augmentation implemented inconsistently between training and test, in the recipe one of the compared models was developed with. It does not estimate how often shared recipes are non-neutral. We suggest, as a hypothesis, that such defects are least likely to be noticed in the recipe a model was developed with, because that model is the one least hurt by them.

## 5.2. Robustness or co-adaptation?

The home-recipe analysis ruled out the reading favourable to our model, namely that CSOANet-v2 is robust to the recipe. As an operationalisation of co-adaptation, however, it did not work as intended, for three reasons:
- **Approximate home recipes.** Two of the three home recipes were approximations.
- **Residuals cannot separate the explanations.** A double-centred residual cannot tell "this model does well under this recipe" from "this recipe does badly for the others", and here the second reading fits better (Section 4.2).
- **The equivalence branch was on the wrong scale.** It compared one residual with a margin set for pairwise differences.

CSOANet-v2 was the architecture least affected by the crop. It was also the least accurate architecture under every recipe, so part of its smaller response could reflect its position on the accuracy scale. On the logit scale, however, the crop still widened its gap to EEGNet. Whether the architecture was shaped during development under this augmentation cannot be decided here.

## 5.3. The training run as a unit of uncertainty

The registered H3 result was fragile. It survived when subjects were treated as independent and was lost under the small-sample correction appropriate for ten clusters. The reason is general. In a cross-subject pairwise difference the subject variance cancels, and what remains is the training-run component (5.4 points²), which does not average out over the subjects of a fold. Intervals that treat subjects as independent are therefore too narrow, and so were the registered power calculations (supplementary material S8), which did not include this component. Pairwise conclusions near a margin should treat the training run as a unit, and should rest on differences-in-differences rather than on category changes.

## 5.4. Conflict of interest

The first author developed CSOANet and R1. The second author was not involved in either.

To limit the influence of this interest, we registered the direction of H2b and all thresholds in advance, included two recipes published by others, gave every architecture the same search budget, and made the home-recipe rules asymmetric against our model. The registered result is unfavourable to our choice of shared recipe. The post hoc analyses explain that result, but they do not reverse it: CSOANet-v2 remained the least accurate architecture under every recipe in F1.

The crop implementation comes from the code of the original CSOANet study [2]. Its baselines were trained without the crop, so the defect did not handicap them there. That comparison did, however, give the proposed model a heavier recipe than the baselines, which is the confound examined here. [Authors to decide and state here: how this will be reported to the journal that published Gao and Diniz [2], e.g. a correction or a note.] A note describing the crop and this study's findings has been prepared for the CSOANet2026 repository.

## 5.5. Limitations

- **Scope.** Two-class motor imagery, two datasets and four compact convolutional architectures, one of them ours and the least accurate. There are no FBCSP or Riemannian baselines, and no more recent architectures such as EEG-TCNet [24].
- **Recipes are packages.** R1, R2 and R3 differ in several components at once. Only R1 against R4 and R1 against R1nc isolate one.
- **ShallowConvNet.** It was used without max-norm constraints, which explain its gap to the reference figure (Section 3.1) and may contribute to its early R2 checkpoints and its divergence under R3.
- **R3.** It was numerically unstable for two architectures, so much of the registered interaction reflects stability rather than recipe preference.
- **Splits and validation.** One split seed; about four validation subjects per fold for checkpoint selection and for the condition-T search.
- **Seeds and SESOI.** Three seeds per cell, with seed variability larger than the 2-point SESOI in most cells.
- **Models.** The confirmatory driver did not retain trained models, so mechanism tests require retraining.
- **Post hoc analyses.** They were added after the results were known (supplementary material S9). One, a residual equivalence test, was on the wrong scale and was withdrawn, and one omnibus statistic was corrected after a convergence check. We read all post hoc results as hypotheses.

## 5.6. Further tests under way

Three sets of runs address the open questions. All models are retained.

- **Location of the crop's harm.** R1 with the crop applied at validation and test as well, and R1 with the crop applied in place without a shift.
- **R2's checkpoint rule.** R2 with the checkpoint chosen by validation accuracy instead of validation loss.
- **ShallowConvNet's constraints.** ShallowConvNet with max-norm constraints under every recipe.

An independent test on Dreyer2023, a left/right-hand dataset that neither author has analysed, is being preregistered (OSF 8st47, Amendment 3). It has two directional hypotheses: the crop effect on the EEGNet − CSOANet-v2 gap, and the R2 effect on the ShallowConvNet − CSOANet-v2 gap. It uses ten folds, so that inference over training runs rests on ten clusters.

# 6. Conclusion

In this preregistered case study, a shared training recipe developed with one of the compared models favoured that model. Most of the effect came from a single augmentation, a crop applied in training but not at test. Beyond it, any remaining recipe × architecture interaction was small and uncertain. No pair of architectures reversed, and the one registered change in a pairwise conclusion did not survive accounting for shared training runs.

Comparisons of compact decoders should:

- check every augmentation for train/test consistency;
- state who designed the shared recipe, and include at least one recipe not developed with any compared model;
- treat the training run, not only the subject, as a unit of uncertainty, and base pairwise conclusions on effect sizes with intervals rather than on category changes.

# Acknowledgments

[Funding: state any institutional support (FEUP / FMUP) or 'No specific funding was received for this work.']

# Conflict of interest

The first author developed CSOANet [2] and the R1 training recipe evaluated in this study; the second author was not involved in either. The direction of the main contrast and all decision thresholds were preregistered before data access, and the home-recipe rules were set against the first author's model (sections 3.2 and 5.4). No other competing interests.

# Data availability statement

The data that support the findings of this study are openly available. Lee2019-MI [25] and Cho2017 [26] were accessed through MOABB 1.7.2 [7]. All per-subject, per-seed results (float32 and int8) are available in the OSF project at https://osf.io/hfnkm. The confirmatory code is available at https://github.com/jabarrick/recipe-x-arch (tag v1.0-frozen, commit a44e5933; confirmatory driver at tag v1.1-frozen, commit 1a7ba7b8) and archived at Zenodo [DOI to be minted at submission]. Trained models from the confirmatory runs were not retained; they can be regenerated with the frozen code and the registered seeds, up to GPU non-determinism. Models from all runs added after the sixth revision are retained and will be archived with the code.

# Preregistration

The study was preregistered on OSF (8st47; https://doi.org/10.17605/OSF.IO/8ST47) on 1 November 2026. Amendment 1 (code freeze and analysis-plan addendum) was public on 13 November 2026 and Amendment 2 (freeze of the confirmatory driver) on 15 November 2026, both before the first confirmatory download. Deviations are listed in supplementary material S4.

# Ethical statement

This study used only publicly available, anonymised datasets [25, 26], collected by their original investigators under their own ethical approvals. No new data were collected and no further approval was required.

# Author contributions

Yu Gao: conceptualisation, methodology, software, investigation, formal analysis, writing – original draft. Yimin Zhang: validation, writing – review and editing. [Adjust to actual contributions before submission.]

# References

[1] Wimpff M, Gizzi L, Zerfowski J and Yang B 2024 EEG motor imagery decoding: a framework for comparative analysis with channel attention mechanisms *J. Neural Eng.* **21** 036020 (https://doi.org/10.1088/1741-2552/ad48b9)

[2] Gao Y and Diniz J M 2026 From perception to imagination: a robust deep learning architecture for real-time EEG decoding and neurophysiological validation *Comput. Methods Biomech. Biomed. Eng. Imaging Vis.* **14** 2684109 (https://doi.org/10.1080/21681163.2026.2684109)

[3] Lawhern V J, Solon A J, Waytowich N R, Gordon S M, Hung C P and Lance B J 2018 EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces *J. Neural Eng.* **15** 056013 (https://doi.org/10.1088/1741-2552/aace8c)

[4] Schirrmeister R T, Springenberg J T, Fiederer L D J, Glasstetter M, Eggensperger K, Tangermann M, Hutter F, Burgard W and Ball T 2017 Deep learning with convolutional neural networks for EEG decoding and visualization *Hum. Brain Mapp.* **38** 5391–420 (https://doi.org/10.1002/hbm.23730)

[5] Wang X, Hersche M, Magno M and Benini L 2024 MI-BMInet: an efficient convolutional neural network for motor imagery brain–machine interfaces with EEG channel selection *IEEE Sens. J.* **24** 8835–47 (https://doi.org/10.1109/JSEN.2024.3353146)

[6] Jayaram V and Barachant A 2018 MOABB: trustworthy algorithm benchmarking for BCIs *J. Neural Eng.* **15** 066011 (https://doi.org/10.1088/1741-2552/aadea0)

[7] Aristimunha B, Carrara I, Guetschel P, Sedlar S *et al* 2026 Mother of all BCI Benchmarks (version 1.7.2) (Zenodo) (https://doi.org/10.5281/zenodo.10034223)

[8] Chevallier S, Carrara I, Aristimunha B, Guetschel P, Sedlar S, Lopes B, Velut S, Khazem S and Moreau T 2024 The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark arXiv:2404.15319

[9] Aristimunha B *et al* 2025 Braindecode: toolbox for decoding raw electrophysiological brain data with deep learning models (Zenodo) (https://doi.org/10.5281/zenodo.17699192)

[10] Kessler R, Enge A and Skeide M A 2025 How EEG preprocessing shapes decoding performance *Commun. Biol.* **8** 1039 (https://doi.org/10.1038/s42003-025-08464-3)

[11] Hou D, Wu Z, Jiang L, Li Z, Lin F and Yamada K D 2026 Same brain, different prediction: how preprocessing choices undermine EEG decoding reliability arXiv:2605.07212

[12] Del Pup F, Zanola A, Tshimanga L F, Bertoldo A and Atzori M 2025 The more, the better? Evaluating the role of EEG preprocessing for deep learning applications *IEEE Trans. Neural Syst. Rehabil. Eng.* **33** 1061–70 (https://doi.org/10.1109/TNSRE.2025.3547616)

[13] Rommel C, Paillard J, Moreau T and Gramfort A 2022 Data augmentation for learning predictive models on EEG: a systematic comparison *J. Neural Eng.* **19** 066020 (https://doi.org/10.1088/1741-2552/aca220)

[14] Melis G, Dyer C and Blunsom P 2018 On the state of the art of evaluation in neural language models *Int. Conf. on Learning Representations (ICLR)*

[15] Lucic M, Kurach K, Michalski M, Gelly S and Bousquet O 2018 Are GANs created equal? A large-scale study *Advances in Neural Information Processing Systems* **31**

[16] Henderson P, Islam R, Bachman P, Pineau J, Precup D and Meger D 2018 Deep reinforcement learning that matters *Proc. AAAI Conf. on Artificial Intelligence* **32** (https://doi.org/10.1609/aaai.v32i1.11694)

[17] Musgrave K, Belongie S and Lim S-N 2020 A metric learning reality check *Computer Vision – ECCV 2020 (Lecture Notes in Computer Science vol 12370)* (Cham: Springer) pp 681–99 (https://doi.org/10.1007/978-3-030-58595-2_41)

[18] Ferrari Dacrema M, Cremonesi P and Jannach D 2019 Are we really making much progress? A worrying analysis of recent neural recommendation approaches *Proc. 13th ACM Conf. on Recommender Systems (RecSys '19)* (New York: ACM) (https://doi.org/10.1145/3298689.3347058)

[19] Wightman R, Touvron H and Jégou H 2021 ResNet strikes back: an improved training procedure in timm arXiv:2110.00476

[20] Bouthillier X *et al* 2021 Accounting for variance in machine learning benchmarks *Proc. Mach. Learn. Syst.* **3** 747–69

[21] Picard D 2021 torch.manual_seed(3407) is all you need: on the influence of random seeds in deep learning architectures for computer vision arXiv:2109.08203

[22] Dehghani M, Tay Y, Gritsenko A A, Zhao Z, Houlsby N, Diaz F, Metzler D and Vinyals O 2021 The benchmark lottery arXiv:2107.07002

[23] Sculley D, Snoek J, Wiltschko A and Rahimi A 2018 Winner's curse? On pace, progress, and empirical rigor *Int. Conf. on Learning Representations (ICLR), Workshop Track*

[24] Ingolfsson T M, Hersche M, Wang X, Kobayashi N, Cavigelli L and Benini L 2020 EEG-TCNet: an accurate temporal convolutional network for embedded motor-imagery brain–machine interfaces *Proc. 2020 IEEE Int. Conf. on Systems, Man, and Cybernetics (SMC)* pp 2958–65 (https://doi.org/10.1109/SMC42975.2020.9283028)

[25] Lee M-H, Kwon O-Y, Kim Y-J, Kim H-K, Lee Y-E, Williamson J, Fazli S and Lee S-W 2019 EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy *GigaScience* **8** giz002 (https://doi.org/10.1093/gigascience/giz002)

[26] Cho H, Ahn M, Ahn S, Kwon M and Jun S C 2017 EEG datasets for motor imagery brain–computer interface *GigaScience* **6** gix034 (https://doi.org/10.1093/gigascience/gix034)

[27] Holm S 1979 A simple sequentially rejective multiple test procedure *Scand. J. Stat.* **6** 65–70

[28] Schuirmann D J 1987 A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability *J. Pharmacokinet. Biopharm.* **15** 657–80 (https://doi.org/10.1007/BF01068419)

[29] Wobbrock J O, Findlater L, Gergle D and Higgins J J 2011 The aligned rank transform for nonparametric factorial analyses using only ANOVA procedures *Proc. SIGCHI Conf. on Human Factors in Computing Systems* (New York: ACM) pp 143–6 (https://doi.org/10.1145/1978942.1978963)

[30] Cameron A C, Gelbach J B and Miller D L 2008 Bootstrap-based improvements for inference with clustered errors *Rev. Econ. Stat.* **90** 414–27 (https://doi.org/10.1162/rest.90.3.414)
