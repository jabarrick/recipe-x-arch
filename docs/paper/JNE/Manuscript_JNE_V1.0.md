---
title: "Is a shared training recipe neutral? Recipe × architecture interactions in compact motor-imagery EEG decoders"
---

Yu Gao^1,2,\*^ and Yimin Zhang^1^

^1^ Doctoral Programme in Electrical and Computer Engineering (PDEEC), Faculty of Engineering, University of Porto, Rua Dr. Roberto Frias, 4200-465 Porto, Portugal

^2^ Doctoral Programme in Health Data Science (HEADS), Faculty of Medicine, University of Porto, Porto, Portugal

^\*^ Author to whom any correspondence should be addressed.

E-mail: Yu_Gao@alumni.baylor.edu

ORCID iDs: Yu Gao [to add]; Yimin Zhang [to add]

# Abstract

*Objective.* Compact motor-imagery EEG decoders are usually compared under one shared training recipe, on the assumption that the recipe does not affect their ranking. We tested this assumption. *Approach.* In a preregistered 4 × 4 design (OSF 8st47), EEGNet, ShallowConvNet, MI-BMInet and CSOANet-v2 were each trained under four recipes: the authors' heavy-augmentation recipe, developed with CSOANet-v2; two published plain recipes; and the heavy recipe without augmentation. Data were Lee2019-MI and Cho2017 (106 subjects, cross-subject evaluation, three seeds), and every decision rule was fixed before the data were downloaded. *Main results.* The recipe was not neutral. Its effect differed across architectures, and the authors' recipe narrowed EEGNet's lead over CSOANet-v2 by 3.5 percentage points of balanced accuracy (95% CI 2.3–4.7). No pair of architectures reversed order. One pairwise conclusion moved between 'better' and 'equivalent', and that change did not survive accounting for test subjects that share trained networks. Exploratory analyses traced the directional effect to a random crop applied in training but not at test, which cost the baselines 3.7–5.9 points and CSOANet-v2 1.1 points. Without the crop, one published recipe still shifted some pairwise gaps by 2–3 points. *Significance.* Shared-recipe comparisons should check every augmentation for train/test consistency, state who designed the recipe, and report at least one recipe that was not developed with any of the compared models.

**Keywords:** motor imagery, EEG decoding, training recipe, benchmarking, preregistration, quantisation, compact neural networks

# 1. Introduction

EEG decoders are compared in one of two ways. Either each model is trained with its published recipe, so that architecture and recipe change together, or all models share one recipe chosen by whoever runs the comparison [1]. A recipe here means everything that is not the model or the data: optimiser, schedule, augmentation and model-selection rule. The shared recipe is presented as the fairer choice because it removes the proposer's tuning advantage. It assumes, untested, that another shared recipe would rank the architectures the same way. If it would not, a shared recipe does not remove a bias; it chooses one.

There are reasons to doubt the assumption. Pipeline choices interact with models in EEG decoding, and equal tuning budgets have shrunk reported architectural progress elsewhere in machine learning (Section 2). The question matters most for compact decoders, which run on wearable hardware and whose published recipes differ as much as their topologies. In our own pilots, changing only the recipe moved EEGNet by 6.6–8.7 points and other architectures by less. Those pilots used the data on which our own model and recipe were developed. We therefore preregistered a test on two datasets neither author had opened.

We ask three questions:

- **RQ1.** Does the recipe change accuracy?
- **RQ2.** Does the change depend on the architecture, and does a shared recipe developed with one compared model favour that model?
- **RQ3.** Do pairwise conclusions change with the recipe?

Four compact architectures, including our own CSOANet-v2, are crossed with four recipes, including our own. The design therefore includes a directional test of whether our recipe favours our model, with decision thresholds set against our model. We also test an equal-budget search per architecture and repeat the analyses after int8 quantisation.

# 2. Related work

EEGNet [2], ShallowConvNet [3], MI-BMInet [4] and CSOANet [5] are compact convolutional decoders, each published with its own recipe and usually compared under the proposer's recipe for the proposed model. MOABB [6–8] and braindecode [9] standardised data, splits and implementations, but not the recipe.

Pipeline choices interact with models. Preprocessing shifts decoding performance differently across classifiers [10], and can flip up to 42 % of single-trial predictions [11]. Minimal preprocessing often beats heavy pipelines [12]. The benefit of EEG augmentation depends on architecture and data size [13].

Outside EEG, equal tuning budgets have shrunk or erased apparent architectural progress [14–18], and a modern recipe lifted a 2015 ResNet to the level of later architectures [19]. Seed variance rivals many reported differences [20, 21], and methods tend to do best under the conditions they were developed in [22, 23].

Quantised deployments report int8 accuracy for one model under its own recipe [4, 24]. Whether quantisation preserves conclusions between architectures has not been asked.

# 3. Methods

## 3.1. Design, data and training

Four architectures are each trained under four recipes, with three seeds per cell. Hypotheses, decision rules and analysis code were fixed before download (Section 3.3). Two public datasets, loaded through MOABB 1.7.2 with left- and right-hand trials only, form the confirmatory family F1. F1 pools Lee2019-MI (54 subjects; [25]) and Cho2017 (52 subjects; [26]), each evaluated cross-subject with 5 folds. The secondary family F2 uses Lee2019-MI within subjects (session 1 → session 2) and is an independent Holm [27] family.

Preprocessing is identical for every cell:

- common-average reference;
- band-pass 1–40 Hz;
- resampling to 250 Hz;
- epoch 0–4 s after the cue (0–3 s for Cho2017);
- per-channel standardisation estimated on training data.

**Splits.** Cross-subject, subjects are split into 5 folds with seed 0, and 10 % of each fold's training subjects form the validation set. Within-subject, 20 % of session 1 is used for validation, and session 2 is the test set.

**Failures.** A run whose loss becomes non-finite is retried once with seed + 100. It is flagged as a training failure, for a sensitivity analysis only, if the loss is still non-finite or its training accuracy at the selected checkpoint is ≤ 55 %.

**Architectures** are EEGNet-8,2 (3.8 k parameters), ShallowConvNet (105 k), MI-BMInet (6.7 k) and CSOANet-v2 (3.8 k); each keeps its own weight constraints under every recipe (supplementary material table S10.1). Before the confirmatory runs, each baseline was checked against a pre-named reference figure on BCI IV 2a (tolerance ±3 points). EEGNet and MI-BMInet passed. ShallowConvNet (−5.9) was kept after verification against braindecode; the gap lies in the Keras reference's max-norm constraints (supplementary material S7).

**Recipes** are applied identically to every architecture (Table 1). Epochs and batch size are fixed per protocol: cross-subject 100 epochs with batch 64, within-subject 500 with batch 32, unless a recipe sets its own batch size.

- R1 is the authors' recipe. Its crop keeps 90 % of the window at one random offset per batch, moves it to the start and zero-fills the rest; test windows are never cropped.
- R2 and R3 are the published recipes of EEGNet and MI-BMInet.
- R4 is R1 without augmentation.

The home pairs are EEGNet + R2, MI-BMInet + R3 (cross-subject only) and CSOANet-v2 + R1. The first two are approximations after protocol unification; the third is exact (supplementary material S1).

**Table 1.** Training recipes (constant across architectures).

| Recipe | Optimiser / schedule | Loss / augmentation | Model selection | Origin |
|---|---|---|---|---|
| R1 — heavy augmentation | AdamW, lr 1e-3, weight decay 1e-3, cosine to 1e-5 | CE with mixup 0.2 and label smoothing 0.1; Gaussian noise 0.02; circular time shift ±5 samples; random crop 90 % (one offset per batch, shifted to the window start, last 10 % zero-filled; not applied at test); channel dropout 10 % | early stopping on validation accuracy, patience 15 (cross-subject) / 80 (within-subject) | [5] |
| R2 — plain, constant lr | Adam, lr 1e-3, constant | CE, no augmentation | lowest validation loss | EEGNet [2] |
| R3 — plain, step schedule | Adam, lr 1e-2, ×0.1 at 40 % and 80 % of epochs; batch 16 | CE, no augmentation | lowest validation loss | MI-BMInet [4] |
| R4 — R1 without augmentation | as R1 | CE, no augmentation | as R1 | control |

**Condition T** gives each architecture the same search budget instead of a shared recipe. For each architecture and dataset, 24 configurations from a fixed space (supplementary material S1) are scored on the fold-1 training and validation subjects, with two seeds. The best is retrained on all folds. Because those subjects are tested in folds 2–5, T is also evaluated on the 22 fold-1 test subjects only. A leakage-free variant, T′, applies configurations selected the same way on the pilot dataset PhysioNet-MI.

**Quantisation and outcomes.** Every model is evaluated in float32 and after int8 post-training quantisation (ONNX Runtime QDQ, MinMax calibration on 256 training trials shared within a fold). The primary outcome is seed-averaged subject-level float32 balanced accuracy. No trained models were retained.

## 3.2. Statistical analysis

Accuracy is modelled with fixed effects for dataset, architecture, recipe and architecture × recipe, and a random intercept per subject. Fixed effects are tested by likelihood-ratio tests. Interaction terms are sum-coded, so each is the double-centred residual of its cell. The smallest effect size of interest (SESOI) is 2 points, about the difference a change of seed alone produces.

The hypotheses are listed below. H1, H2 and H2b are Holm-corrected within each family.

- **H1:** recipe main effect.
- **H2:** recipe × architecture interaction.
- **H2b:** per subject, (EEGNet under R2 − under R1) − (CSOANet-v2 under R2 − under R1). Supported only if significant and positive.
- **H3:** each of 6 architecture pairs under each recipe gets a Bonferroni-adjusted 99.79 % interval, classified as *better*, *equivalent* (within ±2, including 0) or *inconclusive*. H3 is supported if a pair falls into two definitive categories under two recipes. Amendment 1 labels support that comes only from *equivalent* ↔ *better* changes as A′: differences below the margin whose direction is stable, not a change of ranking. A stable outcome was close to unreachable at this sample size (supplementary material S8).
- **H4 (secondary):** the same classification for R1 against T.

**Home-recipe analysis (secondary).** H2b and co-adaptation predict the same sign. To separate them, the home advantage of an architecture is the interaction residual of its home cell. The rules are asymmetric against our own model:

- (i) If CSOANet-v2's home residual is significantly positive, the result is consistent with co-adaptation, and H2b is not read as robustness.
- (ii) If it is equivalent to zero (TOST against ±2; [28]) and its residual range is the smallest in ≥ 80 % of bootstrap resamples, robustness is supported.
- (iii) Otherwise, the mechanism is unresolved.

Registered sensitivity analyses repeat the tests on int8 accuracy, on each dataset separately, without flagged runs and under alternative specifications, including an aligned-rank-transform ANOVA [29].

**Post hoc analyses.** These were added after internal reviews, are labelled wherever they appear, and are listed with their status in supplementary material S9 and S4.

- **Training-run clustering.** All test subjects of a cross-subject fold are scored by the same networks, so pairwise contrasts are re-estimated with a random intercept per dataset × fold cluster (10 clusters). Critical values are t(9), normal, or a cluster bootstrap.
- **Search-space regression** of the condition-T validation accuracies.
- **A rerun of F1 under R1 without cropping (R1nc).** The Cho2017 half was analysed with a script fixed in advance and serves as a replication.
- **Equivalence tests** on all 2 × 2 interaction contrasts among R1nc, R2 and R4.

No further analyses will be added before the independent test in Section 5.6.

## 3.3. Preregistration and deviations

The registration (1 November 2026) and Amendments 1 and 2 (13 and 15 November) were public before the datasets were downloaded on 15 November. A binary pilot run after the rules were committed showed no H2b effect (+0.29 points) and changed no rule (supplementary material S7). The runs took about 126 GPU-hours on one laptop GPU.

Supplementary material S4 lists every post-freeze change; none alters a hypothesis or threshold. In particular, data first loaded under MOABB 1.5.0 lacked one Lee2019-MI session and were reloaded under the registered 1.7.2 before any training.

# 4. Results

## 4.1. Overview and registered tests

F1 comprised 106 subjects and 480 runs. Forty-one runs, all under R3, were flagged and, as registered, kept:

- **ShallowConvNet, 30 runs.** All diverged after 12–34 epochs, but their last checkpoints performed normally.
- **CSOANet-v2, 11 runs.** These failed to learn. Eight more runs of that cell were near chance.

Two diagnostics matter later. Under R1, training accuracy was 3–10 points below R4 for every architecture (supplementary material table S10.2). Under R2, the selected checkpoint came early (median epoch 6.5 for ShallowConvNet).


![**Figure 1.** Subject-level float32 balanced accuracy in each architecture × recipe cell of F1 (○ Lee2019-MI, △ Cho2017; N = 106). Black: cell mean with 95 % CI; dashed line: chance; framed panels: home cells; filled markers: the subject's cell contains a flagged run.](/home/claude/work/s4/figs/fig1_cells_V0.2.png)

The recipe changed accuracy (H1: χ²(3) = 160.5, Holm p = 2.9 × 10⁻³⁴), and the change depended on the architecture (H2: χ²(9) = 414.7, p = 3.0 × 10⁻⁸³). Both hold with crossed subject and training-run effects (post hoc; χ²(3) = 29.1, χ²(9) = 95.3). R1 was the worst recipe for every baseline; R3 was the worst for CSOANet-v2 (Table 2).

**Table 2.** Cell means, F1, float32 (%). In parentheses: interaction residual of the registered 4 × 4 table R1–R4 (points); home cells in bold. R1nc is R1 without cropping (post hoc, Section 4.5) and is not part of the registered table.

| | R1 | R2 | R3 | R4 | R1nc |
|---|---|---|---|---|---|
| EEGNet | 70.6 (−2.58) | **75.9** (−0.28) | 76.6 (+3.00) | 76.8 (−0.14) | 76.5 |
| ShallowConvNet | 69.8 (−0.30) | 71.9 (−1.26) | 71.6 (+1.09) | 74.4 (+0.47) | 73.8 |
| MI-BMInet | 73.2 (−0.68) | 76.1 (−0.82) | **76.9** (+2.60) | 76.6 (−1.11) | 77.0 |
| CSOANet-v2 | **69.1** (+3.55) | 70.9 (+2.35) | 59.3 (−6.69) | 70.1 (+0.78) | 70.2 |

![**Figure 2.** Left: cell means with 95 % CIs by recipe, one line per architecture. Right: interaction residuals of the 4 × 4 table of cell means (points); framed cells are home cells.](/home/claude/work/s4/figs/fig2_interaction_V0.2.png)

H2b was supported. From R1 to R2, EEGNet gained 5.29 points and CSOANet-v2 1.79: c̄ = 3.50 (2.29, 4.71) (Holm p = 9.8 × 10⁻⁸; clustered (1.39, 5.67)). The contrast was significant in each dataset. With R4 in place of R2 it was 5.20 (3.84, 6.57).

![**Figure 3.** Per-subject H2b contrast c[s] = (R2 − R1) for EEGNet minus (R2 − R1) for CSOANet-v2, by dataset and pooled, with mean and 95 % CI. Shaded band: ±2-point SESOI.](/home/claude/work/s4/figs/fig3_h2b_V0.2.png)

## 4.2. Home-recipe analysis

CSOANet-v2's home residual was 3.55 (3.03, 4.07). It was significant and not equivalent to zero, and its residual range was the largest of the four (supplementary material table S10.3). The registered outcome is therefore (i), consistent with co-adaptation. MI-BMInet's home residual was also significant (2.60); EEGNet's was not (−0.28).

The table limits what (i) can mean:

- CSOANet-v2's R3 residual (−6.69) raises its other residuals, because residuals in a row sum to zero. Without R3 (post hoc), its home residual was 1.32, inside the margin.
- EEGNet's residual under R3, which is not its home recipe, was +3.00.
- CSOANet-v2's raw best recipe was R2, not R1. The residual says that R1 hurt the baselines more, not that CSOANet-v2 did best under R1.


## 4.3. Pairwise conclusions (H3) and search (H4)

Twenty of 24 intervals were definitive (median half-width 1.80; Table 3, Figure 4). H3 was supported with outcome A′. MI-BMInet was *better* than EEGNet under R1 (−2.62 (−4.13, −1.11)) and *equivalent* under the other recipes. No pair reversed. Relative to R1, every other recipe narrowed MI-BMInet's lead by 2.3–2.9 points. Under R1, CSOANet-v2 could not be distinguished from EEGNet or ShallowConvNet; under the other recipes EEGNet was clearly ahead of it.

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

**Clustering (post hoc).** The result depends on how the ten clusters are treated:

- With a normal critical value or a cluster bootstrap, 20 intervals stay definitive (A′).
- With t(9), the appropriate small-sample reference [30], only 13 stay definitive, the outcome is intermediate, and H3 would not be supported.

**H4.** Condition T kept the R1 order. Its only definitive change was the same EEGNet − MI-BMInet change, so H4 was supported in the A′ sense; T′ and the fold-1-only analysis agreed. T beat R1 by 1.0–5.7 points, but against R4 it gained nothing (−1.01 to +0.03; supplementary material table S10.4).

## 4.4. Secondary family, int8 and sensitivity

**F2.** No hypothesis was supported in the within-subject family (H1 p = 0.11; H2 p = 0.0501; H2b c̄ = 0.78 (−3.05, 4.62)). As registered, these results are inconclusive. Descriptively, CSOANet-v2 had the highest cell mean under three recipes and EEGNet the lowest under three (supplementary material table S10.5). At the interval level, CSOANet-v2 was ahead of EEGNet only under R1 and R4, and EEGNet was never distinguishable from ShallowConvNet.


**int8.** The decisions were unchanged, and 95.5 % of predictions agreed with float32. One category changed: EEGNet − MI-BMInet under R3, driven by EEGNet, the only partly quantised architecture (5 of 9 operators). In cost terms, MI-BMInet dominated on MACs and CSOANet-v2 on peak memory (supplementary material S5).

**Sensitivity.** All registered alternatives left the decisions unchanged, including excluding flagged runs. The two datasets agreed on H1, H2 and H2b, but not on the H3 label (Lee2019-MI B1, Cho2017 A′), so the pooled H3 result is dataset-dependent.

## 4.5. Exploratory analyses (post hoc)

**Cropping.** In the search data, cropping was the dimension with the largest effect. It lowered validation accuracy by 9.5 points for EEGNet, 6.8 for MI-BMInet, 5.8 for ShallowConvNet and 1.4 (−0.5, 3.4) for CSOANet-v2. Checkpoint selection on validation loss (−1.8) and batch 16 (−1.6) also mattered.

Switching off only the crop (R1nc − R1) raised accuracy for every architecture:

- EEGNet +5.86 (clustered (3.66, 8.05));
- ShallowConvNet +3.98 ((2.57, 5.35));
- MI-BMInet +3.74 ((2.02, 5.43));
- CSOANet-v2 +1.07 ((−0.13, 2.23)).

R1nc matched R4 for every architecture (Table 2), and training accuracy rose to the R4 level, so the R1 deficit was a train/test mismatch, not under-fitting. Removing the crop widened EEGNet's lead over CSOANet-v2 by 4.78 points (clustered (2.66, 6.97)), which is most of the 5.2-point narrowing that R1 produced relative to R4.

Two checks bear on the overlap between hypothesis and test:

- On the 22 fold-1 test subjects, not used to form the hypothesis, the effects were about half as large.
- The Cho2017 half, analysed with a fixed script, replicated the gap change (+3.71 (1.93, 5.48)).

**Remaining interaction.** On {R1nc, R2, R4}, all residuals were within ±1 point (supplementary material S9.9). Of the 18 2 × 2 contrasts (supplementary material S9.11):

- Six excluded zero, all involving R2. The largest was ShallowConvNet − CSOANet-v2, R4 against R2: 3.29 (2.08, 4.50), as large as H2b.
- None between R1nc and R4 excluded zero.
- None was equivalent to zero after Bonferroni adjustment.

With R1nc in place of R1, H2b was −1.29 (clustered (−2.57, 0.00)).


# 5. Discussion

*Section 5 was drafted before data access as modules deposited with the preregistration; the modules used were selected by the registered rules and have since been revised. Post hoc results are labelled as such.*

## 5.1. Was the shared recipe neutral?

As registered, no. The recipe developed with our model narrowed that model's gap to EEGNet. A comparison run under our recipe alone would have shown our model at its closest to the others.

The exploratory analyses show why: one augmentation, implemented with a train/test mismatch, hurt the baselines far more than our model. Without it, R1 behaved like R4, and H2b no longer favoured our model. The R1 − R4 contrast is therefore an effect of one implementation, not of heavy augmentation.

Removing the defect did not make the recipes interchangeable. R2 still shifted ShallowConvNet's gaps by as much as the registered H2b, probably through its checkpoint rule, although that was not isolated.

So two things made the recipe matter: a defect in the recipe our model was developed with, which flattered it, and an ordinary choice in a published recipe. Neither reversed a pair; both changed gaps near the margin. We suggest, as a hypothesis, that defects are least likely to be noticed in the recipe a model was developed with.

## 5.2. Robustness or co-adaptation?

The home-recipe analysis ruled out the reading favourable to our model, that CSOANet-v2 is robust to the recipe. It could not identify co-adaptation as such. A double-centred residual cannot tell "this model does well under this recipe" from "this recipe does badly for the others", and here the second reading fits better (4.2). The registered equivalence branch had a related weakness: it compared one residual with a margin set for pairwise differences.

CSOANet-v2 was the architecture least affected by the crop. It may have been shaped during development under this augmentation, or it may be insensitive to recipes in general: its gains from removing augmentation and from search were also the smallest. The design cannot separate these readings.

## 5.3. The training run as a unit of uncertainty

The registered H3 result was fragile. It survived when subjects were treated as independent and was lost under the small-sample correction appropriate for ten clusters. The reason is general. In a cross-subject pairwise difference the subject variance cancels, and the training-run component (5.4 points²) does not average out over the subjects of a fold. Intervals that treat subjects as independent are therefore too narrow. Pairwise conclusions near a margin should not rest on one shared recipe, and they should treat the training run as a unit.

## 5.4. Search, deployment and protocol

**Search.** An equal-budget search avoided the crop (none of its F1 selections used it; supplementary material table S10.4) and then had little left to gain over R4. With four validation subjects, it mostly selected noise. A well-chosen fixed recipe was as fair as an equal budget.

**Deployment.** Quantisation changed one pairwise category, in a cell combining partial quantisation, MinMax calibration and the most aggressive recipe. Deployment rankings should be checked on the quantised graphs under the recipe that will be used.

**Protocol.** Within subjects, CSOANet-v2 and EEGNet swapped ends of the order. F2 differs from F1 in several ways the design cannot separate:

- one dataset instead of two;
- about 80 training trials per subject, with a session transfer;
- different epochs, early stopping and standardisation;
- checkpoint selection on 20 trials;
- accuracies close to chance.

CSOANet-v2's constraints (static gate, max-norm 0.25) might suit very small training sets; this was not tested. Where feasible, reporting a second protocol shows whether a ranking depends on it.

## 5.5. Conflict of interest and limitations

Both CSOANet-v2 and R1 are ours. We registered the direction of H2b and all thresholds in advance, included two recipes published by others, gave every architecture the same search budget, and made the home-recipe rules asymmetric against our model. The results are unfavourable to our choice of shared recipe.

The crop implementation comes from the code of the original CSOANet study [5]. Its baselines were trained without the crop, so the defect did not handicap them. That study did, however, give the proposed model a heavier recipe than the baselines, which is the confound examined here. Our F2 results do not validate it. That study also differs in paradigm, and its dataset has limitations, unrelated to the recipe, that are being documented on the dataset's OpenNeuro page [reference to be added once public] and addressed separately by its authors. We have prepared a note on the crop for the CSOANet2026 repository.

**Limitations.**

- **Scope.** Two-class motor imagery, two datasets, four compact convolutional architectures, four recipes.
- **R3.** It was numerically unstable for two architectures, so much of the registered interaction reflects stability rather than recipe preference.
- **Implementations.** The architecture factor includes our implementation choices.
- **Seeds.** Three seeds per cell is few.
- **Post hoc analyses.** These came from five rounds of internal review (supplementary material S9). One was found to be wrongly scaled and was withdrawn. We read the post hoc results as hypotheses.

## 5.6. Future work

The next step is an independent test, preregistered before download, on a dataset not used here (for example BNCI2014-004). It will compare R1, R1nc and R2 with two directional hypotheses: the crop effect, and the R2 interaction for ShallowConvNet − CSOANet-v2. The design should also be repeated with the training run as a registered random effect, with more seeds, and on other paradigms.

# 6. Conclusion

A shared training recipe was not neutral. The recipe developed with our model narrowed that model's gap to the strongest baseline, mostly because of a crop applied in training but not at test. Removing the crop left a smaller interaction from a published recipe. No pair of architectures reversed, and the one registered change in a pairwise conclusion did not survive accounting for shared training runs. Comparisons of compact decoders should:

- check every augmentation for train/test consistency;
- state who designed the shared recipe, and include at least one recipe not developed with any compared model;
- treat the training run, not only the subject, as a unit of uncertainty.

# Acknowledgments

[Funding: state any institutional support (FEUP / FMUP) or 'No specific funding was received for this work.'] On acceptance, the conflict-of-interest statement given in the covering letter will be added here: the authors proposed CSOANet and the R1 training recipe evaluated in this study.

# Data availability statement

The data that support the findings of this study are openly available. Lee2019-MI [25] and Cho2017 [26] were accessed through MOABB 1.7.2 [7]. All per-subject, per-seed results (float32 and int8) are available in the OSF project at https://osf.io/hfnkm. The confirmatory code is available at https://github.com/jabarrick/recipe-x-arch (tag v1.0-frozen, commit a44e5933; confirmatory driver at tag v1.1-frozen, commit 1a7ba7b8) and archived at Zenodo [DOI to be minted at submission]. Trained models were not retained; they can be regenerated with the frozen code and the registered seeds, up to GPU non-determinism.

# Preregistration

The study was preregistered on OSF (8st47; https://doi.org/10.17605/OSF.IO/8ST47) on 1 November 2026. Amendment 1 (code freeze and analysis-plan addendum) was public on 13 November 2026 and Amendment 2 (freeze of the confirmatory driver) on 15 November 2026, both before the first confirmatory download. Deviations are listed in supplementary material S4.

# Ethical statement

This study used only publicly available, anonymised datasets [25, 26], collected by their original investigators under their own ethical approvals. No new data were collected and no further approval was required.

# Author contributions

Yu Gao: conceptualisation, methodology, software, investigation, formal analysis, writing – original draft. Yimin Zhang: validation, writing – review and editing. [Adjust to actual contributions before submission.]

# References

[1] Wimpff M, Gizzi L, Zerfowski J and Yang B 2024 EEG motor imagery decoding: a framework for comparative analysis with channel attention mechanisms *J. Neural Eng.* **21** 036020 (https://doi.org/10.1088/1741-2552/ad48b9)

[2] Lawhern V J, Solon A J, Waytowich N R, Gordon S M, Hung C P and Lance B J 2018 EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces *J. Neural Eng.* **15** 056013 (https://doi.org/10.1088/1741-2552/aace8c)

[3] Schirrmeister R T, Springenberg J T, Fiederer L D J, Glasstetter M, Eggensperger K, Tangermann M, Hutter F, Burgard W and Ball T 2017 Deep learning with convolutional neural networks for EEG decoding and visualization *Hum. Brain Mapp.* **38** 5391–420 (https://doi.org/10.1002/hbm.23730)

[4] Wang X, Hersche M, Magno M and Benini L 2024 MI-BMInet: an efficient convolutional neural network for motor imagery brain–machine interfaces with EEG channel selection *IEEE Sens. J.* **24** 8835–47 (https://doi.org/10.1109/JSEN.2024.3353146)

[5] Gao Y and Diniz J M 2026 From perception to imagination: a robust deep learning architecture for real-time EEG decoding and neurophysiological validation *Comput. Methods Biomech. Biomed. Eng. Imaging Vis.* **14** 2684109 (https://doi.org/10.1080/21681163.2026.2684109)

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
