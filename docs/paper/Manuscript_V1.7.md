---
title: "Is a shared training recipe neutral? Recipe × architecture interactions in compact motor-imagery EEG decoders"
author:

  - Yu Gao (Faculty of Engineering, University of Porto; CISTER Research Centre, ISEP)
  - Yimin Zhang
date: "Manuscript V1.7 — 25 November 2026 (condensed and reorganised; analyses unchanged from V1.6)"
---

# Abstract

Compact motor-imagery EEG decoders are usually compared under one shared training recipe, assumed not to affect their ranking. We tested this in a preregistered 4 × 4 design (OSF 8st47). EEGNet, ShallowConvNet, MI-BMInet and CSOANet-v2 were each trained under four recipes: the authors' heavy-augmentation recipe, developed with CSOANet-v2; two published plain recipes; and the heavy recipe without augmentation. Data were Lee2019-MI and Cho2017 (106 subjects, cross-subject, three seeds), and every decision rule was fixed before download. As registered, the shared recipe was not neutral. The recipe effect differed across architectures, and moving from a published plain recipe to the authors' recipe narrowed EEGNet's lead over CSOANet-v2 by 3.5 points [2.3, 4.7]. No pair of architectures reversed at the interval level; pairwise conclusions changed only between "better" and "equivalent", and that change was fragile to how subjects sharing trained networks were treated. Exploratory analyses traced the directional effect to one implementation defect in the authors' recipe: a random crop applied in training but not at test, which cost the baselines 3.7–5.9 points and CSOANet-v2 1.1. Removing it did not make the recipes interchangeable; one published recipe still shifted some pairwise gaps by 2–3 points. Shared-recipe comparisons should check each augmentation for train/test consistency, state who designed the recipe, and report at least one recipe not developed with any of the compared models.

**Keywords.** motor imagery; EEG decoding; training recipe; benchmark; preregistration; quantisation; compact neural networks.

# 1 Introduction

EEG decoders are compared in one of two ways. Either each model is trained with the recipe its authors published, so that architecture and recipe change together, or every model is trained with one shared recipe chosen by whoever runs the comparison (Wimpff et al., 2024). A recipe here means everything that is not the model or the data: optimiser, schedule, augmentation and model-selection rule. The shared recipe is usually presented as the fairer choice, because it removes the proposer's tuning advantage. But it assumes something that has not been tested: that a different shared recipe would rank the architectures the same way. If the ranking depends on the recipe, a shared recipe does not remove a bias; it chooses one.

There are reasons to expect such dependence. Preprocessing choices change EEG decoding performance differently for different models, and augmentation helps or hurts depending on architecture and data size (Section 2). In other fields of machine learning, equalising the tuning budget has repeatedly shrunk reported architectural progress. The question matters most for compact decoders, the models that run on wearable hardware, because their published recipes differ as much as their topologies, and because deployment papers report quantised accuracy for one model under its own recipe only.

Our own pilot work made the question concrete: changing only the recipe moved EEGNet by 6.6–8.7 points, and other architectures by less. Because that pilot was built on the data on which our own model and recipe were developed, we preregistered a confirmatory study on two datasets neither author had opened, with every hypothesis and threshold fixed before download.

We ask three questions. **RQ1:** with the architecture fixed, does the training recipe change accuracy? **RQ2:** does the change depend on the architecture, and does a shared recipe developed with one of the compared models favour that model? **RQ3:** do conclusions about which architecture is better change with the recipe? Four compact architectures — EEGNet, ShallowConvNet, MI-BMInet and our own CSOANet-v2 — are crossed with four recipes on two public left- versus right-hand motor-imagery datasets, cross-subject, with three seeds per cell. Because both CSOANet-v2 and one of the recipes are ours, the design includes a directional test of whether our recipe favours our model, and a home-recipe analysis whose thresholds were set against our own model.

The contributions are:

- a preregistered 4 × 4 test of recipe main effects, recipe × architecture interactions and their consequences for pairwise conclusions;
- a directional test of a shared recipe against the model it was developed with, and an exploratory identification of the mechanism: one augmentation, implemented with a train/test mismatch;
- an equal-budget recipe search per architecture, compared with fixed recipes;
- the same analyses after int8 quantisation, with operator coverage reported;
- a demonstration that, in cross-subject benchmarks, the training run and not only the subject is a unit of uncertainty for pairwise conclusions.

# 2 Related work

**Compact decoders for motor imagery.** EEGNet (Lawhern et al., 2018) stacks temporal, depthwise spatial and separable convolutions into a few thousand parameters. ShallowConvNet (Schirrmeister et al., 2017) is a learned filter-bank analogue of common spatial patterns with a square–pool–log non-linearity. MI-BMInet (Wang et al., 2024) is a separable-convolution design co-optimised for microcontroller inference. CSOANet (Gao and Diniz, 2026) adds a static gating path to a long-kernel temporal front end and was developed by the present authors for the same target. Each was published with its own recipe, and they have almost always been compared under the proposer's recipe for the proposed model.

**Benchmarks and shared recipes.** MOABB (Jayaram and Barachant, 2018; Aristimunha et al., 2023; Chevallier et al., 2024) standardised datasets, splits and metrics, and its reproducibility study showed how much reported differences shrink once data and evaluation are held fixed. Braindecode provides reference implementations (Aristimunha et al., 2025). Within this ecosystem, comparisons either use each model's published recipe or one shared recipe for all (Wimpff et al., 2024). Whether holding the recipe fixed is neutral has not been tested in either practice.

**Pipeline choices interact with models.** Kessler, Enge and Skeide (2025) varied preprocessing in a multiverse design and found that decoding performance shifted considerably and differently across classifiers. Hou et al. (2026) reported that up to 42 % of single-trial predictions flip when only the preprocessing changes. Del Pup et al. (2025) found that minimal preprocessing often beats heavy pipelines for deep decoders. On the training side, the benefit of EEG data augmentation depends on the task, the architecture and the amount of data (Rommel et al., 2022). Outside EEG, equal tuning budgets have shrunk or erased apparent architectural progress in language modelling, GANs, reinforcement learning, metric learning and recommendation (Melis et al., 2018; Lucic et al., 2018; Henderson et al., 2018; Musgrave et al., 2020; Dacrema et al., 2019). A modern recipe lifted a 2015 ResNet to the level of later architectures (Wightman et al., 2021). Seed variance is of the same order as many reported differences (Bouthillier et al., 2021; Picard, 2021), and methods tend to do best under the conditions they were developed in (Dehghani et al., 2021; Sculley et al., 2018). The training recipe as a package has not been crossed with architecture in a factorial, preregistered design for EEG.

**Quantised deployment.** MI-BMInet and EEG-TCNet (Ingolfsson et al., 2020) report 8-bit implementations on low-power microcontrollers with negligible loss. Whether int8 quantisation preserves pairwise conclusions *between* architectures, and whether that depends on the recipe, has not been asked. Nor is operator coverage, the share of a network that actually runs in int8, usually reported.

# 3 Methods

## 3.1 Design and data

The study is a preregistered two-factor experiment: four architectures (factor A) are each trained under four recipes (factor R), with three seeds per cell. The primary outcome is subject-level balanced accuracy. All hypotheses, decision rules and analysis code were fixed before either confirmatory dataset was downloaded (Section 3.8).

Two public datasets, loaded through MOABB 1.7.2 with the left- and right-hand classes only, form the confirmatory family F1; one also forms the secondary family F2 (Table 1). F1 decides the hypotheses. F2 is an independent Holm (1979) family and cannot change the F1 conclusion. Neither author had downloaded or analysed either dataset before registration.

**Table 1.** Evaluation units.

| Unit | Dataset | Subjects | Protocol | Role |
|---|---|---|---|---|
| F1a | Lee2019-MI (OpenBMI; Lee et al., 2019) | 54 | cross-subject, 5-fold | primary confirmatory |
| F1b | Cho2017 (Cho et al., 2017) | 52 | cross-subject, 5-fold | primary confirmatory |
| F2 | Lee2019-MI | 54 | within-subject, session 1 → session 2 | secondary confirmatory |

Lee2019-MI has two sessions per subject on different days, 62 channels at 1000 Hz and 100 trials per session. Cho2017 has one session, 64 channels at 512 Hz and 100 trials per class.

**Preprocessing** is identical for every cell: common-average reference, band-pass 1–40 Hz, resampling to 250 Hz, epoch 0–4 s after the cue (0–3 s for Cho2017), all channels, no artefact rejection beyond MOABB defaults. Per-channel standardisation is estimated on the training subjects of each fold (cross-subject) or on the subject's session-1 training trials (within-subject).

**Splits.** Cross-subject: subjects are shuffled with seed 0 into 5 folds; 10 % of the training subjects of each fold form the validation set. Within-subject: 20 % of a subject's session-1 trials form the validation set and session 2 is the test set.

**Exclusions and failures.** Subjects that cannot be loaded or keep fewer than 80 % of their trials are excluded before training. A run whose loss becomes non-finite is retried once with seed + 100. A run is flagged as a *training failure*, for a sensitivity analysis only, if the loss is still non-finite after the retry or if its training balanced accuracy at the selected checkpoint is ≤ 55 %. Test accuracy never enters any inclusion decision.

## 3.2 Architectures

Each architecture keeps its own weight constraints under every recipe; we treat them as part of the model (Table 2). Before any confirmatory run, each baseline was trained with its published recipe on BCI IV 2a and compared with a reference figure named in advance, with a tolerance of ±3 points. EEGNet-8,2 (+1.3) and MI-BMInet (−2.0) were within tolerance. ShallowConvNet (−5.9) was not; as pre-specified, the implementation was verified against braindecode and kept, and a diagnostic showed that the gap is due to the max-norm constraints of the Keras reference (Supplementary S7).

**Table 2.** Architectures (input at 250 Hz). Parameters (trainable) and MACs per inference of the deployed graph, per dataset input shape (Supplementary S5).

| Architecture | Key settings | Params / MACs (Lee2019-MI) | Params / MACs (Cho2017) | Source |
|---|---|---|---|---|
| EEGNet-8,2 | temporal kernel 125; separable kernel 31; dropout 0.25 / 0.5; max-norm 1 / 0.25 | 3,818 / 63.2 M | 3,594 / 48.9 M | Lawhern et al., 2018 |
| ShallowConvNet | 40 temporal filters (kernel 25), 40 spatial filters, square → mean-pool 75/15 → log; dropout 0.5 | 105,202 / 157.4 M | 107,042 / 120.8 M | Schirrmeister et al., 2017 |
| MI-BMInet | N_k = 32, N_f = 64 (the authors' BCI IV 2a configuration); dropout 0.2 cross-subject / 0.5 within-subject (authors' reference configurations; not stated in the paper) | 6,722 / 1.99 M | 6,530 / 1.54 M | Wang et al., 2024 |
| CSOANet-v2 (long kernel) | temporal kernel 125; static gate with bounded term; ReLU; max-norm 0.25 | 3,800 / 2.08 M | 3,704 / 1.57 M | this work (v2 of CSOANet; Gao and Diniz, 2026) |

CSOANet-v2 and recipe R1 were developed together by the authors on the pilot datasets. This is the conflict of interest the design is built around.

## 3.3 Recipes

The four recipes are applied identically to every architecture (Table 3). R1 is the authors' heavy-augmentation recipe. R2 and R3 are the published recipes of EEGNet and MI-BMInet. R4 is R1 without augmentation, so R1 against R4 isolates the augmentation package. Epochs and batch size are fixed per protocol (cross-subject 100 epochs, batch 64; within-subject 500 epochs, batch 32) unless a recipe specifies its own batch size.

**Table 3.** Training recipes (constant across architectures).

| Recipe | Optimiser / schedule | Loss / augmentation | Model selection | Origin |
|---|---|---|---|---|
| R1 — heavy augmentation | AdamW, lr 1e-3, weight decay 1e-3, cosine to 1e-5 | CE with mixup 0.2 and label smoothing 0.1; Gaussian noise 0.02; circular time shift ±5 samples; random crop 90 % (one offset per batch, shifted to the window start, last 10 % zero-filled; not applied at test); channel dropout 10 % | early stopping on validation accuracy, patience 15 (cross-subject) / 80 (within-subject) | CSOANet 2026 |
| R2 — plain, constant lr | Adam, lr 1e-3, constant | CE, no augmentation | lowest validation loss | EEGNet (Lawhern et al., 2018) |
| R3 — plain, step schedule | Adam, lr 1e-2, ×0.1 at 40 % and 80 % of epochs; batch 16 | CE, no augmentation | lowest validation loss | MI-BMInet (Wang et al., 2024) |
| R4 — R1 without augmentation | as R1 | CE, no augmentation | as R1 | control |

R1's crop draws one random offset per batch, keeps 90 % of the window from that offset, moves it to the start and fills the last 10 % with zeros. Test windows are never cropped. This implementation is shared with the pilot code in which CSOANet-v2 and R1 were developed.

**Home pairs.** EEGNet + R2, MI-BMInet + R3 and CSOANet-v2 + R1 are *home* pairs. The coupling differs. CSOANet-v2 and R1 were developed together; EEGNet and R2 were published together on other data; R3 is MI-BMInet's published cross-subject recipe, so it is home only in F1. ShallowConvNet's published recipe is not in the set. Because epochs and batch size were unified, the EEGNet and MI-BMInet home recipes are approximations, whereas CSOANet-v2's is exact (Table 4).

**Table 4.** Fidelity of the home recipes.

| Home pair | Published recipe | Change in this study | Reason |
|---|---|---|---|
| EEGNet + R2 | Adam (default lr 1e-3), 500 epochs, lowest validation loss; batch size not stated (Lawhern et al., 2018) | cross-subject E = 100, B = 64; within-subject E = 500, B = 32 | protocol unification, fixed at registration |
| MI-BMInet + R3 | cross-subject recipe: Adam 1e-2 → 1e-3 → 1e-4 at epochs 0 / 40 / 80, 100 epochs, batch 16, with N_k = 16, N_f = 128 (Wang et al., 2024, PhysioNet); within-subject recipe on BCI IV 2a was Adam 1e-3 constant, 500 epochs, batch 32 | cross-subject E = 100 (equal to published), batch 16 kept, schedule at 40 % / 80 % of epochs; network kept at N_k = 32, N_f = 64 (the authors' BCI IV 2a configuration); within-subject E = 500 under R3 is not the published within-subject recipe | protocol unification, fixed at registration |
| CSOANet-v2 + R1 | CSOANet recipe | none | derives from the recipe used for CSOANet in Gao and Diniz (2026); finalised for CSOANet-v2 in this project's pilot work |

## 3.4 Condition T: equal-budget search

Condition T gives each architecture the same tuning budget instead of a shared recipe. For each architecture and evaluation unit, 24 configurations drawn once from a fixed space (optimiser, learning rate, schedule, batch size, each augmentation on or off, label smoothing, selection rule; Supplementary S1) are scored by mean validation accuracy over two seeds. Cross-subject, scoring uses the training and validation subjects of fold 1; within-subject, session 1 of ten subjects. The best configuration is then trained on all folds with the registered seeds.

T has two opposing biases. The fold-1 subjects it is tuned on are test subjects in folds 2–5, a hyperparameter leak that favours T; a sensitivity analysis therefore uses the 22 fold-1 test subjects only. Selecting the best of 24 noisy scores deflates T's test accuracy relative to its search score. A leakage-free variant, T′, applies unchanged the configurations selected by the same search on the pilot dataset PhysioNet-MI, recorded in Amendment 1 before the code freeze.

## 3.5 Seeds, quantisation and outcomes

Every cell is run with seeds 0, 1 and 2, and seed-averaged accuracy is the unit of analysis. Each model is evaluated in float32 and after post-training int8 quantisation (ONNX Runtime static QDQ, per-channel symmetric weights, MinMax calibration on 256 un-augmented training trials, the same set for every model within a fold or subject). We report the share of operators quantised per architecture.

The primary outcome is subject-level float32 balanced accuracy. Secondary outcomes are int8 balanced accuracy, float32–int8 agreement, training and validation accuracy at the selected checkpoint, and static cost (parameters, MACs, peak int8 activation memory). No latency is measured. The frozen driver deletes each exported model after scoring, so no trained models were retained.

## 3.6 Statistical analysis

**Model.** Subject-level accuracy is modelled as

y_{s,a,r} = μ + δ_{d(s)} + α_a + ρ_r + (αρ)_{a,r} + u_s + ε_{s,a,r},   u_s ~ N(0, σ_u²),

with dataset d(s) and a random intercept per subject. Fixed effects are tested by likelihood-ratio tests (LRT) on maximum-likelihood fits. Interaction terms are sum-coded, so (αρ)_{a,r} is the double-centred residual of cell (a, r).

**Smallest effect size of interest (SESOI).** Two points of balanced accuracy: about the difference a change of seed alone produces (median absolute difference between two seeds of one configuration in the T′ search: 1.1–2.1 points).

**Hypotheses.** H1, H2 and H2b are Holm-corrected within each family at α = 0.05.

- **H1:** recipe main effect (LRT, 3 df).
- **H2:** recipe × architecture interaction (LRT, 9 df).
- **H2b (directional):** per subject, c[s] = (EEGNet under R2 − EEGNet under R1) − (CSOANet-v2 under R2 − CSOANet-v2 under R1), tested against 0; supported only if significant and positive. R1 and R2 differ in optimiser, schedule, selection rule and augmentation, so H2b tests R1 as a package.
- **H3:** each of the 6 architecture pairs under each of the 4 recipes gets a Bonferroni-adjusted 99.79 % interval, classified as *A better*, *B better*, *equivalent* (within ±2 and including 0) or *inconclusive*. An interval that excludes 0 is *better* even if it lies within ±2. H3 is supported if any pair falls into two different definitive categories under two recipes. Amendment 1 fixed the outcome labels: B1 (stable; ≥ 16 of 24 definitive, no change), B2 (uninformative; < 8 definitive), and A′ for support that comes only from *equivalent* ↔ *better* changes. A′ is reported as differences below the margin whose direction is stable, not as a change in ranking.
- **H4 (secondary):** the same classification for R1 against condition T, with Kendall's τ.

**Home-recipe analysis (secondary; Amendment 1).** H2b and co-adaptation predict the same sign: if each model does best under its own recipe, moving from R1 to R2 helps EEGNet and hurts CSOANet-v2. To separate them, the home advantage of architecture a is the double-centred residual of its home cell,

h_a = ȳ_{a,home} − ȳ_{a,·} − ȳ_{·,home} + ȳ_{·,·},

tested per subject for EEGNet, MI-BMInet and CSOANet-v2 (Holm). For CSOANet-v2, equivalence is tested by TOST against ±2 points (Schuirmann, 1987). Each architecture's residual range has a bootstrap CI. The decision rules are asymmetric against our own model:

- (i) CSOANet-v2's home residual significantly positive: consistent with co-adaptation, and H2b is not read as recipe robustness;
- (ii) CSOANet-v2's home residual equivalent to zero and its residual range the smallest in ≥ 80 % of bootstrap resamples: robustness supported;
- (iii) otherwise: unresolved.

If excluding training failures leaves more than 10 % of subjects with an incomplete table, h_a is taken from the mixed model instead.

**Sensitivity analyses (registered, none decisive):** int8 accuracy; aligned-rank-transform ANOVA (Wobbrock et al., 2011); F1a and F1b separately, with the pooled conclusion called "dataset-dependent" if they disagree; random slopes; cluster bootstrap; fold as a fixed effect; seed as a random effect; exclusion of training failures.

**Power.** From the four-class pilot, power for H2b was 0.96 (full effect) and 0.37 (half). A simulation attached to Amendment 1 (Supplementary S8) gave a median Bonferroni half-width of 1.78 points at n = 106 and a false-positive rate of about 3 % for H3 under zero interaction, all from *equivalent* ↔ *better* changes. A stable outcome (B1) was therefore close to unreachable. The pre-specified rule for adding a third dataset was narrowly not met (0.496 against 0.5).

## 3.7 Post hoc analyses

Analyses added after the results, in response to internal reviews, are labelled post hoc wherever they appear and are listed with their status in Supplementary S9 and S4 (rows 14–18). None changes a registered decision. The main ones are:

- **Training-run clustering.** All test subjects of a cross-subject fold are scored by the same networks, so their errors are correlated within each dataset × fold cluster (10 in F1). H1 and H2 are refitted with crossed random effects for subject and training run. Single contrasts (pairwise differences, H2b, home residuals) are re-estimated with a random intercept per cluster and t on 9 df, and alternatively with a normal critical value, a cluster bootstrap and t(8) with a dataset effect.
- **A search-space regression** of the 24 condition-T configurations' validation accuracy on their dimensions.
- **An exploratory rerun of F1 under R1 without cropping (R1nc; 120 runs)**, run on 24 November 2026 because the regression pointed to cropping. The analysis script was written after the Lee2019-MI half had been inspected and before the Cho2017 half had run, so the Cho2017 half serves as a replication. The fold-1 subjects used by the regression are test subjects in folds 2–5; R1nc − R1 is therefore also reported for the 22 fold-1 test subjects.
- **Interaction among {R1nc, R2, R4}:** all 18 2 × 2 differences-in-differences (6 architecture pairs × 3 recipe pairs), with the cluster intercept and TOST against ±2, unadjusted and Bonferroni-adjusted. (V1.5 tested single residuals against ±2, which is the wrong scale; that test is withdrawn.)

No further analyses will be added before the independent test in Section 5.9.

## 3.8 Preregistration, pilots and deviations

The registration (OSF 8st47, 1 November 2026) fixed the design, hypotheses, SESOI, families and corrections. Amendment 1 (public 13 November) recorded the code freeze, baseline verification, home-recipe analysis, H3 outcome labels, condition-T disclosures and T′, the training-failure rule and the sensitivity analyses. Amendment 2 (public 15 November, 16:37 UTC) froze the confirmatory driver (tag v1.1-frozen). The datasets were downloaded after that, on 15 November (Cho2017 16:48–17:23, Lee2019-MI 17:25–20:28 UTC). The runs took about 126 GPU-hours on one laptop GPU (18–24 November 2026). No registered compute cut was needed.

Two pilots preceded the registration (Supplementary S7). A four-class pilot motivated the hypotheses. A binary pilot (PhysioNet-MI, 105 subjects), run after the decision rules were committed, fed the simulation in S8; it showed no H2b effect (+0.29 points) and lowered our prior. PhysioNet-MI is the dataset on which CSOANet-v2 and R1 were developed.

Supplementary S4 lists every change after the freeze. None changes a hypothesis, threshold or primary analysis. Two concern data loading: MOABB 1.5.0, first installed, returned one Lee2019-MI session only, and the data were reloaded under the registered 1.7.2 before any training; and MOABB joins Cho2017 trials into a continuous signal before filtering, so filter edge effects occur at trial boundaries, identically for every cell.

# 4 Results

## 4.1 Overview

F1 comprised 106 subjects and 480 runs; no subject was excluded (Supplementary S2). Forty-one runs (8.5 %) were flagged, all under R3:

- **All 30 ShallowConvNet runs under R3** reached a non-finite loss after 12–34 epochs, also after the retry. Their last checkpoint before divergence was used. Its test accuracy (cell mean 71.6 %) is within the range of the architecture's other cells, so these runs did learn.
- **Eleven CSOANet-v2 runs under R3** had training accuracy ≤ 55 %. Eight more had 55–60 % and test accuracy of 50–61 %, so the rule missed part of a cell that largely failed (Figure 1).

The primary analyses keep flagged runs, as registered. Cell means ranged from 59.3 to 76.9 % (mean 72.5 %; Table 6). Two diagnostics matter later:

- Under R1, training accuracy at the selected checkpoint was 3–10 points below R4 for every architecture, although runs lasted about as long.
- Under R2, which keeps the lowest validation loss over all epochs, the selected checkpoint came early: median epoch 6.5 for ShallowConvNet.

**Table 5.** Recipe dependence and training diagnostics, F1 (float32).

| Architecture | Recipe range [95 % CI] | Best | Worst | Train acc. | Val. acc. | Flagged runs |
|---|---|---|---|---|---|---|
| EEGNet | 6.2 [5.1, 7.4] | R4 | R1 | 80.7 | 75.8 | 0 / 120 |
| ShallowConvNet | 4.5 [3.8, 5.2] | R4 | R1 | 84.2 | 72.4 | 30 / 120 (R3, non-finite loss) |
| MI-BMInet | 3.7 [2.8, 4.6] | R3 | R1 | 84.5 | 76.1 | 0 / 120 |
| CSOANet-v2 | 11.6 [9.7, 13.5] | R2 | R3 | 72.3 | 67.4 | 11 / 120 (R3, training acc. ≤ 55 %) |

*Recipe range = best minus worst recipe mean, subject-level bootstrap (2,000 resamples). Train and validation accuracy are run means at the selected checkpoint. Excluded subjects: 0. Retried runs: 30 (all ShallowConvNet × R3).*

![**Figure 1.** Subject-level float32 balanced accuracy in each architecture × recipe cell of F1 (○ Lee2019-MI, △ Cho2017; N = 106). Black: cell mean with 95 % CI; dashed line: chance; framed panels: home cells; filled markers: the subject's cell contains a flagged run.](/home/claude/work/s4/figs/fig1_cells_V0.2.png)

## 4.2 Recipe main effect and interaction (H1, H2)

The recipe changed accuracy (H1: χ²(3) = 160.5, Holm p = 2.9 × 10⁻³⁴), and the change depended on the architecture (H2: χ²(9) = 414.7, p = 3.0 × 10⁻⁸³). Both remain supported with crossed subject and training-run effects (post hoc; χ²(3) = 29.1, p = 2 × 10⁻⁶; χ²(9) = 95.3, p = 10⁻¹⁶).

Of the total variance, architecture explained 6.9 %, recipe 1.7 % and their interaction 3.6 %; subjects explained 74.8 %. Among the fixed effects, the interaction's share was 29.8 % [25.3, 34.7]. R1 was the worst recipe for all three baselines; R3 was the worst for CSOANet-v2, whose recipe range (11.6 points) was the largest. The dataset main effect was 4.9 points [0.8, 9.1] in favour of Lee2019-MI.

**Table 6.** Cell means, F1, float32 (%) (top), and interaction residuals of the cell-mean table, in points (bottom).

| | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| EEGNet | 70.6 | 75.9 | 76.6 | 76.8 |
| ShallowConvNet | 69.8 | 71.9 | 71.6 | 74.4 |
| MI-BMInet | 73.2 | 76.1 | 76.9 | 76.6 |
| CSOANet-v2 | 69.1 | 70.9 | 59.3 | 70.1 |

![**Figure 2.** Left: cell means with 95 % CIs by recipe, one line per architecture. Right: interaction residuals of the 4 × 4 table of cell means (points); framed cells are home cells.](/home/claude/work/s4/figs/fig2_interaction_V0.2.png)

| | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| EEGNet | −2.58 | −0.28 (h) | 3.00 | −0.14 |
| ShallowConvNet | −0.30 | −1.26 | 1.09 | 0.47 |
| MI-BMInet | −0.68 | −0.82 | 2.60 (h) | −1.11 |
| CSOANet-v2 | 3.55 (h) | 2.35 | −6.69 | 0.78 |

*Home cells are marked (h).*

## 4.3 Directional interaction (H2b)

H2b was supported. Moving from R1 to R2 improved EEGNet by 5.29 points and CSOANet-v2 by 1.79, giving c̄ = 3.50 [2.29, 4.71] (t(105) = 5.73, Holm p = 9.8 × 10⁻⁸). It was significant in each dataset (3.95 and 3.03) and positive for 73.6 % of subjects. Both architectures gained under R2. The contrast is positive because EEGNet gained more, not because CSOANet-v2 lost. With a random intercept per cluster (post hoc), c̄ = 3.53 [1.39, 5.67]. The same contrast with R4 in place of R2 was 5.20 [3.84, 6.57] (clustered [2.18, 8.35]).

![**Figure 3.** Per-subject H2b contrast c[s] = (R2 − R1) for EEGNet minus (R2 − R1) for CSOANet-v2, by dataset and pooled, with mean and 95 % CI. Shaded band: ±2-point SESOI.](/home/claude/work/s4/figs/fig3_h2b_V0.2.png)

## 4.4 Home-recipe analysis (secondary confirmatory)

The home residuals were −0.28 [−0.71, 0.16] for EEGNet, 2.60 [2.14, 3.06] for MI-BMInet and 3.55 [3.03, 4.07] for CSOANet-v2. The last two were significant (Holm p = 1.9 × 10⁻¹⁹ and 3.6 × 10⁻²⁴), and CSOANet-v2's was not equivalent to zero (90 % CI [3.11, 3.99]). CSOANet-v2's residual range was the largest, not the smallest (0 % of resamples). By the preregistered rule the outcome is **(i), consistent with co-adaptation**. It was unchanged without the near-home R4 column (CSOANet-v2: 3.81) and with clustering (3.56 [2.46, 4.66]).

Three features of the table limit what (i) can mean:

- **R3 drives the pattern.** CSOANet-v2's R3 residual was −6.69. Residuals in a row sum to zero, so this one cell raises CSOANet-v2's other residuals, the home cell included. Without R3 (post hoc), its home residual fell to 1.32 [0.91, 1.73], inside the ±2 margin. Excluding flagged runs gave 2.42.
- **Non-home cells show residuals as large.** EEGNet's residual under R3, not its home recipe, was +3.00, larger than MI-BMInet's home residual in the same column.
- **The raw means point the other way.** CSOANet-v2's best recipe was R2 (70.9 %), not R1 (69.1 %). The positive home residual says that R1 hurt the baselines more than it hurt CSOANet-v2, not that CSOANet-v2 did best under R1.

**Table 7.** Home-recipe analysis, F1, float32. Without R4 (registered sensitivity), the home residuals were EEGNet −0.32, MI-BMInet 2.23 and CSOANet-v2 3.81, and the mechanism outcome was still (i).

| Architecture | Home | h_a [95 % CI] | Holm p | TOST 90 % CI | Residual range [CI] | Raw range [CI] | Smallest residual range |
|---|---|---|---|---|---|---|---|
| EEGNet | R2 | −0.28 [−0.71, 0.16] | 0.21 | — | 5.57 [4.72, 6.48] | 6.21 [5.12, 7.42] | 0.0 % |
| ShallowConvNet | none | — | — | — | 2.35 [1.58, 3.23] | 4.54 [3.85, 5.21] | 99.0 % |
| MI-BMInet | R3 | 2.60 [2.14, 3.06] | 1.9 × 10⁻¹⁹ | — | 3.71 [3.07, 4.35] | 3.66 [2.80, 4.59] | 1.0 % |
| CSOANet-v2 | R1 | 3.55 [3.03, 4.07] | 3.6 × 10⁻²⁴ | [3.11, 3.99], not equivalent | 10.24 [8.92, 11.57] | 11.65 [9.76, 13.56] | 0.0 % |

## 4.5 Stability of pairwise conclusions (H3)

The median adjusted half-width was 1.80 points, and 20 of 24 intervals were definitive (Table 8; Figure 4). H3 was supported, with outcome A′. One pair changed definitive category, without a sign reversal: MI-BMInet was *better* than EEGNet under R1 (−2.62 [−4.13, −1.11]) and *equivalent* under R2, R3 and R4. That gives 3 *equivalent* ↔ *better* changes among 36 opportunities, no sign reversals and 10 *inconclusive* ↔ definitive changes. The change is not only a classification artefact: relative to R1, every other recipe narrowed MI-BMInet's lead by 2.3–2.9 points (clustered lower bounds 0.87–1.22).

Seven of the ten *inconclusive* ↔ definitive changes involve CSOANet-v2. Under R1 it could not be distinguished from EEGNet or ShallowConvNet; under the other recipes EEGNet was clearly ahead of it. The registered H3 rule does not count such changes; H2b measures them.

**Table 8.** Category matrix, F1, float32 (code: mean difference; A = first better, B = second better, E = equivalent, I = inconclusive). Second block: int8.

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

**Sensitivity to training-run clustering (post hoc).** With a random intercept per dataset × fold cluster (median intraclass correlation 0.15), the result depends on the critical value:

- **Registered** (subjects independent): median half-width 1.80, 20 of 24 definitive, A′.
- **Design effect only** (normal critical value or cluster bootstrap): half-width 2.80 or 2.50, 20 definitive, A′.
- **Design effect with small-sample correction:** t(9), half-width 3.89, 13 definitive; t(8) with a dataset effect, 11 definitive. Both give an intermediate outcome.

With ten clusters, uncorrected intervals are expected to be too narrow (Cameron et al., 2008), so t(9) is the appropriate reference, and under it H3 would not be supported. Leaving out single folds of the registered analysis gave A′ every time, but that check does not address clustering. Under t(9), MI-BMInet remained ahead of CSOANet-v2 under every recipe and EEGNet ahead of it under R2–R4.

## 4.6 Shared recipe versus equal-budget search (H4)

T kept the R1 order (MI-BMInet > EEGNet > ShallowConvNet > CSOANet-v2; τ = 1.0, CI [0.33, 1.0]). Four of six categories changed between R1 and T (Table 10). The only definitive change was the EEGNet − MI-BMInet change seen in H3; the other three went from *inconclusive* to *better*. H4 is therefore supported, but only in the A′ sense. The fold-1-only evaluation showed no definitive change, and T′ gave the same categories as T.

T improved on R1 by 1.0 (CSOANet-v2) to 5.7 points (EEGNet), and T′ by slightly more. Against the best fixed recipe, R4, T gained nothing (post hoc): EEGNet −0.47 [−1.07, 0.13], ShallowConvNet −1.01 [−1.79, −0.23], MI-BMInet +0.03 [−0.75, 0.82], CSOANet-v2 −0.01 [−0.58, 0.55]. T′ was within 1 point of R4. Search scores differed from test accuracy by +1.0 to +7.0 points in Lee2019-MI and −6.1 to −10.4 in Cho2017. With four validation subjects per dataset, this mostly reflects which subjects were held out, so we do not report a winner's-curse size.

**Table 9.** R1, T and T′ (F1, float32, %), and the configurations selected.

| Architecture | R1 | T | T′ | Fold-1-only R1 → T (N = 22) | T selected, Lee2019-MI | T selected, Cho2017 | T′ selected |
|---|---|---|---|---|---|---|---|
| MI-BMInet | 73.2 | 76.6 | 77.5 | 73.7 → 76.3 | #19 | #19 | #13 |
| EEGNet | 70.6 | 76.4 | 77.1 | 72.9 → 76.4 | #13 | #19 | #10 |
| ShallowConvNet | 69.8 | 73.4 | 74.8 | 70.3 → 75.6 | #10 | #19 | #1 |
| CSOANet-v2 | 69.1 | 70.1 | 70.5 | 68.8 → 70.6 | #20 | #4 | #10 |

*Configurations (S1): #19 AdamW, lr 9.9 × 10⁻³, step schedule, batch 64, noise + shift + channel dropout, label smoothing 0.1, early stopping on validation accuracy. #13 AdamW, lr 2.5 × 10⁻³, step, batch 32, noise only, early stopping on validation accuracy. #10 Adam, lr 3.1 × 10⁻³, step, batch 64, noise + shift + channel dropout, best validation loss over all epochs. #20 Adam, lr 3.5 × 10⁻⁴, constant, batch 64, mixup only, early stopping on validation accuracy. #4 AdamW, lr 8.2 × 10⁻⁴, step, batch 64, noise + shift + mixup, early stopping on validation accuracy. #1 Adam, lr 3.4 × 10⁻⁴, cosine, batch 32, no augmentation, early stopping on validation accuracy. T is searched separately in each evaluation unit, so Lee2019-MI and Cho2017 can select different configurations. In Cho2017 one configuration (#19) won for three of the four architectures.*

**Table 10.** H4 categories, F1 (Bonferroni over the 12 intervals of this comparison, 99.58 %; the same categories at 99.79 %).

| Pair | R1 | T | T′ |
|---|---|---|---|
| EEGNet − ShallowConvNet | I | A | A |
| EEGNet − MI-BMInet | B | E | E |
| EEGNet − CSOANet-v2 | I | A | A |
| ShallowConvNet − MI-BMInet | B | B | B |
| ShallowConvNet − CSOANet-v2 | I | A | A |
| MI-BMInet − CSOANet-v2 | A | A | A |

## 4.7 Secondary family F2, int8 and sensitivity analyses

**F2 (within-subject, Lee2019-MI, N = 54).** No hypothesis was supported:

- H1: p = 0.11.
- H2: p = 0.0501.
- H2b: c̄ = 0.78 [−3.05, 4.62].
- H3: intermediate (9 of 24 definitive, median half-width 4.60).
- H4: no definitive change.

As registered, these are reported as inconclusive (registered power 0.70 and 0.18). Seed SDs were larger than in F1 (median 3.79 against 2.44 points), consistent with checkpoint selection on 20 validation trials. 124 of 2,592 runs were flagged, 42 of them EEGNet under R1.

Descriptively, the order differed from F1: CSOANet-v2 had the highest cell mean under R1, R3 and R4, and EEGNet the lowest under R1, R2 and R4 (Table 11). At the interval level the differences are fewer:

- CSOANet-v2 was ahead of EEGNet only under R1 (−7.52 [−12.41, −2.63]) and R4 (−7.48 [−13.76, −1.19]).
- It was ahead of MI-BMInet only under R1 and R3.
- EEGNet could not be distinguished from ShallowConvNet under any recipe.

CSOANet-v2's F2 accuracy was nearly flat across recipes (65.2–66.1 %).

**Table 11.** F2 (within-subject) cell means (%, float32) and rank of each architecture under R1, R2, R3 and R4. In F1 the order under R1–R3 was MI-BMInet, EEGNet, ShallowConvNet, CSOANet-v2.

| Architecture | R1 | R2 | R3 | R4 | Rank (R1, R2, R3, R4) |
|---|---|---|---|---|---|
| EEGNet | 57.7 | 58.6 | 59.7 | 58.0 | 4, 4, 3, 4 |
| ShallowConvNet | 61.3 | 62.1 | 58.3 | 61.4 | 3, 3, 4, 3 |
| MI-BMInet | 61.8 | 66.2 | 61.0 | 63.4 | 2, 1, 2, 2 |
| CSOANet-v2 | 65.2 | 65.4 | 66.1 | 65.4 | 1, 2, 1, 1 |

**int8.** Every decision for H1, H2, H2b and H3 was unchanged (H2b c̄ = 4.05 [2.90, 5.20]). Float32–int8 prediction agreement was 95.5 %. One of 24 categories changed: EEGNet − MI-BMInet under R3, from *equivalent* to *MI-BMInet better*. It comes from the EEGNet × R3 cell, which lost 3.21 points in int8, the largest loss of any cell. EEGNet was the only partly quantised architecture: 5 of 9 compute operators, with average pooling and ELU left in float32. The other three were fully quantised (Supplementary S5).

**Cost.** MI-BMInet was on the accuracy–MACs front in every dataset × recipe panel, in float32 and int8; CSOANet-v2, with 2–5 % more MACs and lower accuracy, never was. On the accuracy–peak-RAM front, CSOANet-v2 was always present, with about 17 % less peak int8 memory than MI-BMInet, and so was MI-BMInet (Supplementary S5).

**Sensitivity.**

- **Decisions unchanged** with random slopes, cluster bootstrap, fold as a fixed effect, seed as a random effect, the aligned-rank transform, and exclusion of flagged runs (home residual of CSOANet-v2: 2.42; H3 A′).
- **Datasets.** Both datasets showed strong interactions with architecture and recipe (three-way χ²(9) = 62.7). Analysed separately, they agreed on H1, H2 and H2b but not on the H3 label (Lee2019-MI B1, Cho2017 A′), so the pooled H3 conclusion is dataset-dependent as registered. Neither showed a sign reversal.

## 4.8 Exploratory analyses

**(a) Augmentation package.** Removing R1's augmentation (R4 − R1) helped every architecture: EEGNet +6.21 [5.03, 7.39], ShallowConvNet +4.54 [3.83, 5.25], MI-BMInet +3.34 [2.50, 4.18], CSOANet-v2 +1.00 [0.18, 1.83].

**(b) What in the search space mattered (post hoc).** Regressing the validation accuracy of the 24 configurations × 4 architectures on the fold-1 tuning split on the configuration dimensions (standard errors clustered by configuration) found three clear effects:

- Cropping lowered accuracy by 5.9 points [5.2, 6.6].
- Selecting the lowest validation loss lowered it by 1.8 [1.2, 2.4], relative to early stopping on validation accuracy.
- Batch 16 was about 1.6 points worse than larger batches.

The cropping effect differed by architecture: EEGNet 9.5 [8.8, 10.2], MI-BMInet 6.8 [6.2, 7.5], ShallowConvNet 5.8 [5.0, 6.6], CSOANet-v2 1.4 [−0.5, 3.4].

**(c) R1 without cropping (post hoc; hypothesis from b).** Switching off cropping alone, with everything else in R1 unchanged, raised accuracy for every architecture (R1nc − R1; subject-level CI, then clustered):

- EEGNet: +5.86 [4.83, 6.89], clustered [3.66, 8.05];
- ShallowConvNet: +3.98 [3.24, 4.72], clustered [2.57, 5.35];
- MI-BMInet: +3.74 [2.97, 4.51], clustered [2.02, 5.43];
- CSOANet-v2: +1.07 [0.33, 1.81], clustered [−0.13, 2.23].

R1nc matched R4 for every architecture (differences −0.40 to +0.56, all intervals including zero). The other four augmentations of R1 therefore had no net effect. Training accuracy rose to the R4 level (EEGNet 74.7 → 81.1, against 82.9 under R4), so the R1 deficit was a train/test mismatch, not under-fitting. Removing cropping widened EEGNet's lead over CSOANet-v2 by 4.78 [3.66, 5.90] (clustered [2.66, 6.97]), which is 4.8 of the 5.2-point narrowing that R1 produced relative to R4.

Two checks bear on the overlap between hypothesis and test:

- **Fold-1 test subjects.** On the 22 fold-1 test subjects, not used by the regression, the effects were about half as large (EEGNet +3.52 [1.78, 5.25]; gap change +2.19 [0.12, 4.26]). These subjects form two clusters only, so their intervals omit training-run variance.
- **Cho2017 replication.** The Cho2017 half, analysed with a script fixed in advance, gave EEGNet +3.71 [2.20, 5.23], CSOANet-v2 +0.01 [−1.08, 1.09], and a gap change of +3.71 [1.93, 5.48] (clustered).

The effect was larger in Lee2019-MI (4 s windows) than in Cho2017 (3 s).

**(d) Interaction once cropping and R3 are removed (post hoc).** On the table {R1nc, R2, R4} (Table 12), every residual was within ±1 point. Of the 18 2 × 2 contrasts (Supplementary S9.11):

- Six had 95 % intervals excluding zero, and all six involve R2. The largest was ShallowConvNet − CSOANet-v2, R4 against R2: 3.29 [2.08, 4.50], as large as the registered H2b. The next were the same pair, R2 against R1nc (−2.65 [−4.55, −0.75]), and ShallowConvNet − MI-BMInet, R4 against R2 (2.01 [0.95, 3.07]).
- No contrast between R1nc and R4 excluded zero.
- Seven of 18 were equivalent to zero at the unadjusted level, and none after Bonferroni adjustment.

With R1nc in place of R1, the H2b contrast was −1.29 [−2.29, −0.29] (clustered [−2.57, 0.00]). The omnibus interaction test on this table was significant with subjects as the unit (χ²(6) = 23.7) but not with crossed run effects (χ²(6) = 4.6, p = 0.60); a 6-df test has little power against an interaction concentrated in one recipe.

**Table 12.** Cell means (%) and interaction residuals (points), F1, recipes R1nc, R2 and R4 (post hoc). Residuals are descriptive; the SESOI applies to the 2 × 2 contrasts in Supplementary S9.11.

| | R1nc | R2 | R4 | Residual R1nc | Residual R2 | Residual R4 |
|---|---|---|---|---|---|---|
| EEGNet | 76.5 | 75.9 | 76.8 | −0.12 | −0.01 | +0.13 |
| ShallowConvNet | 73.8 | 71.9 | 74.4 | +0.28 | −1.00 | +0.72 |
| MI-BMInet | 77.0 | 76.1 | 76.6 | +0.24 | +0.02 | −0.27 |
| CSOANet-v2 | 70.2 | 70.9 | 70.1 | −0.40 | +0.99 | −0.58 |

# 5 Discussion

*Section 5 was drafted before data access as alternative modules with numerical placeholders, deposited with the preregistration. The modules used were selected by preregistered rules (outcome A′, mechanism i). The text has since been revised in light of the post hoc analyses, which are labelled as such; the deposited modules allow the original wording to be compared.*

## 5.1 Was the shared recipe neutral?

As registered, no. The recipe effect differed across architectures (H2), and the recipe developed with the authors' model narrowed that model's gap to EEGNet by 3.5 points relative to a published recipe (H2b), and by 5.2 relative to R4. A comparison run under our recipe alone would have shown our model at its closest to the others.

The exploratory analyses identify why. One augmentation in R1, a random crop applied in training but not at test, cost the baselines 3.7–5.9 points and CSOANet-v2 about one (4.8 c). Without it, R1 behaved like R4, and H2b no longer favoured our model. The R1 − R4 contrast is therefore an effect of one implementation, not of heavy augmentation.

Removing the defect did not make the recipes interchangeable. Among the remaining recipes, R2 still shifted some pairwise gaps by 2–3 points, most of all ShallowConvNet's, as much as the registered H2b (4.8 d). R2's checkpoint rule, which stopped ShallowConvNet around epoch 6, is the likeliest source, but it was not isolated.

Two different things therefore made the recipe matter here. One was a defect in the recipe our model was developed with, and it flattered that model. The other was an ordinary design choice in a published recipe. Neither reversed a pair of architectures at the interval level in F1; both changed the size of gaps near the margin. We put forward as a hypothesis, not a finding, that defects of the first kind are least likely to be noticed in the recipe a model was developed with. These analyses are post hoc, on F1 only, and were run after the two columns carrying most of the registered interaction (R1 and R3) had been set aside.

## 5.2 Robustness or co-adaptation?

Before the study, a positive H2b could have been read as "CSOANet-v2 is robust to the recipe". The home-recipe analysis was registered to rule this reading in or out, and it ruled it out: CSOANet-v2's home residual was large and significant, and its recipe range the largest of the four.

It could not, however, identify co-adaptation as such. A double-centred residual cannot tell "architecture a does well under recipe r" from "recipe r does badly for the others" (4.4). Here the second reading fits better: R1 was the worst recipe for every baseline, CSOANet-v2 did better under R2 than under R1, and much of the residual's size comes from CSOANet-v2's collapse under R3. This is a property of double-centred residuals in a 4 × 4 table with one unstable recipe, and we report it as a limitation of the registered method. The registered equivalence branch has a related weakness: it compares a single residual with a margin set for pairwise differences, so an "equivalent" home residual would have been weaker evidence than its label suggests.

The cropping result gives the asymmetry a concrete form. CSOANet-v2 was least affected by the crop, in the search data and in the rerun. Two readings fit and cannot be separated here. The architecture may have been shaped during development under this augmentation. Or it may simply be insensitive to recipe changes in general: its R4 − R1 difference, its gain from search and its recipe range without R3 were the smallest of the four. Architectural explanations, such as its gating, were not tested.

## 5.3 Pairwise conclusions and the statistical unit

The registered H3 was supported in its weakest form, A′: one pair moved between *better* and *equivalent*, and no pair reversed. The simulation in S8 showed that such changes can arise by sampling alone under zero interaction. Here H2 rejected a zero interaction, and the difference-in-differences for that pair excluded zero, so the change is not only noise. But it is fragile. It survives when subjects are treated as independent or with a design-effect correction, and is lost with the small-sample correction appropriate for ten clusters (4.5).

The reason is general. In cross-subject evaluation, all test subjects of a fold share the same trained networks. In a pairwise difference the subject variance cancels, and what remains is the training-run component (5.4 points² per cell) against the residual (15 points²). Unlike the residual, the run component does not average out over the roughly ten subjects of a fold. Registered intervals that treat subjects as independent are therefore too narrow for pairwise conclusions. Comparisons near a margin should not rest on one shared recipe, and they should treat the training run as a unit of uncertainty.

## 5.4 Equal-budget search

A per-architecture search with 24 configurations improved on R1 but not on R4, and it left the order unchanged. None of the configurations it selected in F1 used cropping (Table 9), so the search avoided the defect. Beyond cropping, only the checkpoint rule and batch 16 mattered, and R4 is already on the favourable side of both. A search had little left to gain. T′, tuned on another dataset, did as well as T, and search scores were poor predictors of test accuracy. With four validation subjects, a search of this size mostly selects noise. An equal budget is therefore not a better notion of fairness than a well-chosen fixed recipe here. What matters is that the fixed recipe does not penalise some architectures more than others.

## 5.5 Deployment

The recipe is the cheapest lever for a deployed model, because its cost is paid at training time. Quantisation changed one pairwise category, in one cell: EEGNet under R3, the recipe with the highest learning rate, on the only partly quantised architecture, under MinMax calibration. A percentile calibrator might remove the change, but testing that needs retraining, because no models were retained. Rankings meant for deployment should be checked on the quantised graphs, under the recipe that will be used, with operator coverage and calibration reported. Which architecture is cheapest does not depend on the recipe, but whether it is accurate enough depends on the cost measure: MI-BMInet dominates on MACs, CSOANet-v2 on memory.

## 5.6 Evaluation protocol

Within subjects (F2), CSOANet-v2 and EEGNet swapped ends of the order, although fewer differences were clear at the interval level (4.7). F2 differs from F1 in much more than the protocol:

- one dataset instead of two;
- about 80 training trials per subject instead of several thousand, with a transfer across sessions;
- 500 epochs with patience 80 instead of 100 with patience 15;
- checkpoint selection on 20 trials;
- a different standardisation;
- accuracies of 58–66 %, closer to chance, where orders are more easily reshuffled by noise.

The design cannot separate these. One untested reading is that CSOANet-v2's constraints (a static gate, max-norm 0.25) suit very small training sets, which would fit its flat F2 profile and its general insensitivity to recipe. No F2 hypothesis was supported, so we do not compare the size of this effect with that of the recipe. Where it is feasible, reporting a second protocol shows whether a ranking depends on it.

## 5.7 Conflict of interest and the original CSOANet paper

Both CSOANet-v2 and R1 are ours. We addressed this in four ways:

- Two of the recipes were published by others.
- The direction of H2b and every threshold were registered before data access.
- Condition T gave every architecture the same budget.
- The home-recipe rules were made asymmetric against our model.

The analysis returned the unfavourable reading, and the post hoc results remain unfavourable to our choice of shared recipe.

The cropping implementation analysed here comes from the code published with CSOANet (Gao and Diniz, 2026), a perception-versus-imagination study. There, the baselines were trained without cropping, so the defect did not handicap them; if anything it would have cost CSOANet slightly. That comparison did, however, give the proposed model a heavier recipe than the baselines, which is the confound this study examines. Our F2 results suggest that CSOANet-v2's standing depends on the evaluation protocol. They do not validate the original comparison, which differed in paradigm, used unequal recipes, and used a dataset with limitations unrelated to the recipe question. Those limitations are being documented in a note on the dataset's OpenNeuro page [reference to be added once the note is public], and their bearing on the original paper is being addressed separately by its authors. We have prepared a note for the CSOANet2026 repository describing the cropping implementation and this study's findings about it.

## 5.8 Limitations

- **Scope.** Two-class motor imagery, two datasets, four compact convolutional architectures and four recipes from a continuous space. Nothing here transfers to Transformer or pretrained decoders.
- **R3.** R3 was numerically unstable for two architectures, so much of the registered interaction concerns stability under a high learning rate rather than recipe preference. Without R3, the interaction's share of fixed-effect variance fell from 30 % to 10 %.
- **Implementations.** The architecture factor includes our implementation choices. ShallowConvNet was used without the Keras max-norm constraints, and MI-BMInet in its N_k = 32, N_f = 64 configuration. The EEGNet and MI-BMInet home recipes were approximations.
- **Post hoc results.** The cropping and remaining-interaction results are post hoc, on F1 only, and partly share subjects with the analysis that generated the hypothesis. Eleven groups of post hoc analyses were added over five internal revisions (Supplementary S9). One, a residual equivalence test, was found to be on the wrong scale and withdrawn. Together they form a garden of forking paths, and we read them as hypotheses for the test below.
- **Precision.** Three seeds per cell is few: the median seed SD exceeded the margin in 13 of 16 F1 cells (Supplementary S3). Power was planned on subjects only. Condition T had a 24-trial budget and four validation subjects. F2 was under-powered by design.

## 5.9 Future work

The first follow-up is an independent test, preregistered before download, on a left/right-hand dataset not used here (for example BNCI2014-004). It will compare R1, R1nc and R2 across the four architectures, with two directional hypotheses: the cropping effect, and the R2 interaction (ShallowConvNet − CSOANet-v2, R2 against R1nc). The design should then be repeated:

- with the training run as a registered random effect and more folds and seeds;
- with cropping applied at test time or left in place, to locate the harm;
- on new data and on visual imagery.

The multi-recipe protocol — several recipes, pairwise intervals with a margin, home-recipe residuals — could be offered to MOABB and braindecode so that new architectures are routinely tested this way.

# 6 Conclusion

As registered, no: the recipe effect differed across architectures, and the authors' recipe narrowed their model's gap to EEGNet. Exploratory analyses traced that directional effect to one implementation defect in the authors' recipe. Removing the defect did not make the remaining recipes interchangeable: a published plain recipe (R2) still changed some pairwise gaps by 2–3 points, most of all ShallowConvNet's.

The registered result stands on its own: a shared recipe developed with one of the compared models favoured that model. Most of that came from a crop applied in training but not at test. No pair of architectures reversed at the interval level in the cross-subject family, and the one registered change in a pairwise conclusion did not survive a small-sample analysis that accounts for shared training runs.

Comparisons of compact decoders should therefore:

- check each augmentation for train/test consistency, above all in a recipe developed with one of the compared models;
- state who designed the shared recipe, and report at least one recipe not developed with any of the compared models;
- treat the training run, not only the subject, as a unit of uncertainty for pairwise differences.

# Declarations

**Data availability.** Lee2019-MI and Cho2017 are public and were accessed through MOABB 1.7.2. All per-subject, per-seed results (float32 and int8) are released with the paper in the OSF project (osf.io/hfnkm). Trained models were not retained; they can be regenerated with the frozen code and the registered seeds, up to GPU non-determinism.

**Code availability.** Confirmatory code: github.com/jabarrick/recipe-x-arch, tag v1.0-frozen (commit a44e5933). The confirmatory driver is at tag v1.1-frozen (commit 1a7ba7b8; Amendment 2). The code is archived at Zenodo (DOI [to be minted at submission]). Pilot code: branch csoanet-v2-deploy of github.com/jabarrick/CSOANet2026.

**Preregistration.** OSF 8st47, DOI 10.17605/OSF.IO/8ST47, registered 1 November 2026:

- Amendment 1 (code freeze and analysis-plan addendum) was public on 13 November 2026.
- Amendment 2 (freeze of the confirmatory driver) was public on 15 November 2026.

- Both were public before the first confirmatory download, on 15 November 2026.

Deviations are listed in Supplementary S4.

**Author contributions (CRediT).** Yu Gao: conceptualisation, methodology, software, investigation, formal analysis, writing — original draft. Yimin Zhang: validation, writing — review and editing. [Adjust to actual contributions before submission.]

**Competing interests.** The authors proposed CSOANet and the R1 training recipe evaluated in this study. The study was preregistered with directional hypotheses and decision thresholds set before data access, to limit the influence of this interest (Sections 3.6 and 5.7). The preregistered home-recipe analysis returned the reading unfavourable to the authors' model. No other competing interests.

**Funding.** No specific funding. [Add institutional support statements if required by FEUP / CISTER.]

**Ethics.** The study uses only publicly available, anonymised datasets; no new data were collected.

# References

Aristimunha, B., Carrara, I., Guetschel, P., Sedlar, S., et al., & Chevallier, S. (2023). *Mother of All BCI Benchmarks (MOABB)* [Software; version 1.7.2 used]. Zenodo. https://doi.org/10.5281/zenodo.10034223

Aristimunha, B., Guetschel, P., Wimpff, M., Gemein, L., Rommel, C., Banville, H., et al., & Schirrmeister, R. T. (2025). *Braindecode: toolbox for decoding raw electrophysiological brain data with deep learning models* [Software]. Zenodo. https://doi.org/10.5281/zenodo.17699192

Bouthillier, X., Delaunay, P., Bronzi, M., et al. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems*, 3, 747–769.

Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *The Review of Economics and Statistics*, 90(3), 414–427. https://doi.org/10.1162/rest.90.3.414

Chevallier, S., Carrara, I., Aristimunha, B., Guetschel, P., Sedlar, S., Lopes, B., Velut, S., Khazem, S., & Moreau, T. (2024). The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark. arXiv:2404.15319.

Cho, H., Ahn, M., Ahn, S., Kwon, M., & Jun, S. C. (2017). EEG datasets for motor imagery brain–computer interface. *GigaScience*, 6(7), gix034. https://doi.org/10.1093/gigascience/gix034

Dacrema, M. F., Cremonesi, P., & Jannach, D. (2019). Are we really making much progress? A worrying analysis of recent neural recommendation approaches. In *Proceedings of the 13th ACM Conference on Recommender Systems* (pp. 101–109).

Dehghani, M., Tay, Y., Gritsenko, A. A., et al. (2021). The benchmark lottery. arXiv:2107.07002.

Del Pup, F., Zanola, A., Tshimanga, L. F., Bertoldo, A., & Atzori, M. (2025). The more, the better? Evaluating the role of EEG preprocessing for deep learning applications. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 33, 1061–1070. https://doi.org/10.1109/TNSRE.2025.3547616

Gao, Y., & Diniz, J. M. (2026). From perception to imagination: a robust deep learning architecture for real-time EEG decoding and neurophysiological validation. *Computer Methods in Biomechanics and Biomedical Engineering: Imaging & Visualization*. [volume, pages and DOI to be added]

Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep reinforcement learning that matters. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1).

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.

Hou, D., Wu, Z., Jiang, L., Li, Z., Lin, F., & Yamada, K. D. (2026). Same brain, different prediction: how preprocessing choices undermine EEG decoding reliability. arXiv:2605.07212.

Ingolfsson, T. M., Hersche, M., Wang, X., Kobayashi, N., Cavigelli, L., & Benini, L. (2020). EEG-TCNet: an accurate temporal convolutional network for embedded motor-imagery brain–machine interfaces. In *2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC)* (pp. 2958–2965). https://doi.org/10.1109/SMC42975.2020.9283028

Jayaram, V., & Barachant, A. (2018). MOABB: trustworthy algorithm benchmarking for BCIs. *Journal of Neural Engineering*, 15(6), 066011. https://doi.org/10.1088/1741-2552/aadea0

Kessler, R., Enge, A., & Skeide, M. A. (2025). How EEG preprocessing shapes decoding performance. *Communications Biology*, 8, 1039. https://doi.org/10.1038/s42003-025-08464-3

Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013. https://doi.org/10.1088/1741-2552/aace8c

Lee, M.-H., Kwon, O.-Y., Kim, Y.-J., Kim, H.-K., Lee, Y.-E., Williamson, J., Fazli, S., & Lee, S.-W. (2019). EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy. *GigaScience*, 8(5), giz002. https://doi.org/10.1093/gigascience/giz002

Lucic, M., Kurach, K., Michalski, M., Gelly, S., & Bousquet, O. (2018). Are GANs created equal? A large-scale study. In *Advances in Neural Information Processing Systems*, 31.

Melis, G., Dyer, C., & Blunsom, P. (2018). On the state of the art of evaluation in neural language models. In *International Conference on Learning Representations*.

Musgrave, K., Belongie, S., & Lim, S.-N. (2020). A metric learning reality check. In *European Conference on Computer Vision* (pp. 681–699).

Picard, D. (2021). torch.manual_seed(3407) is all you need: on the influence of random seeds in deep learning architectures for computer vision. arXiv:2109.08203.

Rommel, C., Paillard, J., Moreau, T., & Gramfort, A. (2022). Data augmentation for learning predictive models on EEG: a systematic comparison. *Journal of Neural Engineering*, 19(6), 066020. https://doi.org/10.1088/1741-2552/aca220

Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W., & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420. https://doi.org/10.1002/hbm.23730

Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.

Sculley, D., Snoek, J., Wiltschko, A., & Rahimi, A. (2018). Winner's curse? On pace, progress, and empirical rigor. *ICLR Workshop track*.

Wang, X., Hersche, M., Magno, M., & Benini, L. (2024). MI-BMInet: an efficient convolutional neural network for motor imagery brain–machine interfaces with EEG channel selection. *IEEE Sensors Journal*, 24(6), 8835–8847.

Wightman, R., Touvron, H., & Jégou, H. (2021). ResNet strikes back: an improved training procedure in timm. arXiv:2110.00476.

Wimpff, M., Gizzi, L., Zerfowski, J., & Yang, B. (2024). EEG motor imagery decoding: a framework for comparative analysis with channel attention mechanisms. *Journal of Neural Engineering*, 21(3), 036020.

Wobbrock, J. O., Findlater, L., Gergle, D., & Higgins, J. J. (2011). The aligned rank transform for nonparametric factorial analyses using only ANOVA procedures. In *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems* (pp. 143–146).

