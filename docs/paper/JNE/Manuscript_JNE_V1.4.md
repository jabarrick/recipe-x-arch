---
title: "Is a shared training recipe neutral? Architecture rankings of compact motor-imagery EEG decoders under four recipes"
---

Yu Gao^1,2,\*^ and Yimin Zhang^1^

^1^ Doctoral Programme in Electrical and Computer Engineering (PDEEC), Faculty of Engineering, University of Porto, Rua Dr. Roberto Frias, 4200-465 Porto, Portugal

^2^ Doctoral Programme in Health Data Science (HEADS), Faculty of Medicine, University of Porto, Porto, Portugal

^\*^ Author to whom any correspondence should be addressed.

E-mail: Yu_Gao@alumni.baylor.edu

ORCID iDs: Yu Gao [to add]; Yimin Zhang [to add]

# Abstract

*Objective.* Compact motor-imagery EEG decoders are often compared under one shared training recipe. We tested whether the choice of shared recipe changes such a comparison, in a case where one candidate recipe had been developed with one of the compared models. *Approach.* EEGNet, ShallowConvNet, MI-BMInet and the first author's CSOANet-v2 were each trained under four shared recipes: (R1) the recipe developed with CSOANet, which combines several augmentations with label smoothing; (R2) the recipe published with EEGNet; (R3) the recipe published with MI-BMInet; and (R4) R1 without augmentation or label smoothing. Data were Lee2019-MI and Cho2017 (106 subjects, cross-subject, three seeds), and hypotheses and decision rules were preregistered (OSF 8st47). *Main results.* The registered analysis found that the recipe effect differed across architectures and that R1 narrowed EEGNet's lead over CSOANet-v2 by 3.5 points of balanced accuracy (95% CI 1.4–5.6, with training runs as clusters). Post hoc checks traced these results to two defects in our implementation, an augmentation in R1 applied in training but not at test and missing weight constraints in ShallowConvNet, and to CSOANet-v2 failing to train in 20 of 30 runs under R3. With the two defects corrected, no pair of architectures moved between definitive pairwise categories under any recipe, and without R3 neither a recipe effect (p = 0.58) nor an interaction (p = 0.11) remained. EEGNet and MI-BMInet were the most accurate under every recipe, and CSOANet-v2 the least. Treating shared training runs as clusters reduced the definitive pairwise conclusions of the registered analysis from 20 to 13 of 24. *Significance.* Before recipes are compared, augmentations and baseline implementations should be checked for every compared architecture, and the training run should be treated as a unit of uncertainty.

**Keywords:** motor imagery, EEG decoding, training recipe, data augmentation, benchmarking, preregistration, compact neural networks

# 1. Introduction

EEG decoders are compared in one of two ways. Either each model is trained with its published recipe, so that architecture and recipe change together, or all models share one recipe chosen by whoever runs the comparison [1]. A recipe here means everything that is not the model or the data: optimiser, schedule, augmentation and model-selection rule. The shared recipe is presented as the fairer choice, because it removes the proposer's tuning advantage. It assumes, however, that the recipe treats all architectures alike. If it does not, a shared recipe does not remove a bias; it chooses one.

There are reasons to doubt the assumption. Pipeline choices interact with models in EEG decoding, and equal tuning budgets have shrunk reported architectural progress elsewhere in machine learning (Section 2). The question is sharpest when the shared recipe was developed together with one of the compared models. That is our situation: the first author developed both the CSOANet decoder [2] and the recipe R1 used to train it. In our four-class pilots, changing only the recipe moved EEGNet by 6.6–8.7 points and other architectures by less; a later binary pilot showed no such asymmetry (Section 3.4). We therefore preregistered a test on two datasets neither author had analysed, with the direction of the main contrast and the decision rules set against our own model.

We ask three questions:

- **RQ1.** Does the recipe change accuracy?
- **RQ2.** Does the change depend on the architecture, and does a recipe developed with one compared model favour that model?
- **RQ3.** Do pairwise conclusions change with the recipe?

The registered analysis answered yes to all three. After it, we found two defects in our implementation: a random crop in R1 that was applied in training but not at test, and missing weight constraints in our ShallowConvNet. We report the registered analysis as it was run, show how the defects produced its positive results, and then repeat the registered tests on a corrected design that needed no new training runs beyond the follow-ups. The corrected design is post hoc. Its answers are the study's substantive result, but they are exploratory until the independent test (Section 5.6) is done.

# 2. Related work

EEGNet [3], ShallowConvNet [4], MI-BMInet [5] and CSOANet [2] are compact convolutional decoders. Each was published with its own recipe, and they are usually compared under the proposer's recipe for the proposed model. MOABB [6–8] and braindecode [9] standardised data, splits and implementations, but not the recipe.

Pipeline choices interact with models. Preprocessing shifts decoding performance differently across classifiers [10], and can flip up to 42 % of single-trial predictions [11]. Minimal preprocessing often beats heavy pipelines [12]. The benefit of EEG augmentation depends on architecture and data size [13].

Outside EEG, equal tuning budgets have shrunk or erased apparent architectural progress [14–18]. A modern recipe lifted a 2015 ResNet to the level of later architectures [19]. Seed variance rivals many reported differences [20, 21], and methods tend to do best under the conditions they were developed in [22, 23].

# 3. Methods

## 3.1. Design, data and training

Four architectures are each trained under four recipes, with three seeds per cell. Two public datasets, loaded through MOABB 1.7.2 with left- and right-hand trials only, form the confirmatory family F1: Lee2019-MI (54 subjects; [24]) and Cho2017 (52 subjects; [25]), each evaluated cross-subject with 5 folds. A secondary within-subject family (F2, Lee2019-MI session 1 → session 2) is reported in supplementary material S9.

Preprocessing is identical for every cell: common-average reference; band-pass 1–40 Hz; resampling to 250 Hz; epoch 0–4 s after the cue (0–3 s for Cho2017); per-channel standardisation estimated on training data.

**Splits.** Subjects are split into 5 folds with seed 0, and 10 % of each fold's training subjects (about four per dataset) form the validation set used for checkpoint selection. One split seed was used, so split variance is not estimated.

**Failures.** A run whose loss becomes non-finite is retried once with seed + 100. It is flagged as a training failure, for a sensitivity analysis only, if the loss is still non-finite or its training accuracy at the selected checkpoint is ≤ 55 %.

**Architectures** are EEGNet-8,2 (3.8 k parameters), ShallowConvNet (105 k), MI-BMInet (6.7 k) and CSOANet-v2 (3.8 k). Each keeps its own weight constraints under every recipe (supplementary material table S10.1). Before the confirmatory runs, each baseline was checked against a pre-named reference figure on BCI IV 2a, with a tolerance of ±3 points. EEGNet and MI-BMInet passed. ShallowConvNet missed by 5.9 points and was kept, as pre-specified, after verification against braindecode; with the max-norm constraints of the Keras reference it reached the reference (supplementary material S7). The registered runs therefore used a ShallowConvNet without max-norm constraints.

**Recipes** are applied identically to every architecture (Table 1): (R1) the recipe developed with CSOANet, which combines several augmentations with label smoothing; (R2) the recipe published with EEGNet; (R3) the recipe published with MI-BMInet; and (R4) R1 without augmentation or label smoothing. Cross-subject runs use 100 epochs and batch 64 unless a recipe sets its own batch size. R1's crop keeps 90 % of the window at one random offset per batch, moves it to the start and zero-fills the rest; test windows are never cropped. The recipes differ in several components at once; only R1 against R4 isolates one (augmentation).

**Home pairs.** These are EEGNet + R2, MI-BMInet + R3 and CSOANet-v2 + R1. Only the last is exact: protocol unification shortened EEGNet's published 500 epochs to 100, and R3 was published for a different MI-BMInet configuration (supplementary material S1).

**Table 1.** Training recipes (constant across architectures).

| Recipe | Optimiser / schedule | Loss / augmentation | Model selection | Origin |
|---|---|---|---|---|
| R1 — heavy augmentation | AdamW, lr 1e-3, weight decay 1e-3, cosine to 1e-5 | CE with mixup 0.2 and label smoothing 0.1; Gaussian noise 0.02; circular time shift ±5 samples; random crop 90 % (one offset per batch, shifted to the window start, last 10 % zero-filled; not applied at test); channel dropout 10 % | early stopping on validation accuracy, patience 15 (cross-subject) / 80 (within-subject) | [2] |
| R2 — plain, constant lr | Adam, lr 1e-3, constant | CE, no augmentation | lowest validation loss | EEGNet [3] |
| R3 — plain, step schedule | Adam, lr 1e-2, ×0.1 at 40 % and 80 % of epochs; batch 16 | CE, no augmentation | lowest validation loss | MI-BMInet [5] |
| R4 — R1 without augmentation | as R1 | CE, no augmentation | as R1 | control |

**Outcome.** The primary outcome is seed-averaged subject-level float32 balanced accuracy. Every model was also evaluated after int8 post-training quantisation, and an equal-budget search (condition T) was run; both are secondary and reported in supplementary material S5 and S9.

## 3.2. Registered and corrected designs

The **registered design** is the 4 × 4 table above. Two defects in it were found after the results were known (Section 4.2):

- R1's crop was applied in training but not at test, and the shift of the kept segment to the window start harmed some architectures even when training and test were consistent.
- ShallowConvNet lacked the reference max-norm constraints.

The **corrected design** replaces R1 by R1nc (R1 without the crop, otherwise identical) and ShallowConvNet by ShallowConvNet-MN (with the reference constraints: convolutional filters 2.0, dense layer 0.5). All other cells are the registered runs. The registered tests (H1–H3 and the home-recipe analysis) are repeated on it unchanged. Because the corrected design was defined after the results were known, its tests are post hoc.

**Follow-up runs** (F1, same folds and seeds, all models retained) supplied the corrected cells and three diagnostic variants:

- R1nc for every architecture, and ShallowConvNet-MN under R1, R1nc, R2, R3 and R4;
- R1ct: R1 with the same crop applied to validation and test windows, at the middle offset;
- R1ns: R1 with the crop applied in place, so the kept segment is not shifted and the samples outside it are zeroed;
- R2va: R2 with the checkpoint chosen by the highest validation accuracy instead of the lowest validation loss.

## 3.3. Statistical analysis

**Registered model.** Accuracy is modelled with fixed effects for dataset, architecture, recipe and architecture × recipe, and a random intercept per subject; fixed effects are tested by likelihood-ratio tests. Interaction terms are sum-coded, so each is the double-centred residual of its cell. The smallest effect size of interest (SESOI) is 2 points, fixed at registration; seed variability turned out larger (median subject-level seed SD above 2 points in 13 of 16 cells; supplementary material S3), so the SESOI is a convention, not an estimate of noise.

**Hypotheses.** H1, H2 and H2b are Holm-corrected within the family [26].

- **H1:** recipe main effect.
- **H2:** recipe × architecture interaction.
- **H2b:** per subject, (EEGNet under R2 − under R1) − (CSOANet-v2 under R2 − under R1). Supported only if significant and positive.
- **H3:** each of the 6 architecture pairs under each recipe gets a Bonferroni-adjusted 99.79 % interval. An interval that excludes zero is *better* (for one side); one that includes zero and lies within ±2 is *equivalent*; any other is *inconclusive*. H3 is supported if a pair falls into two different definitive categories under two recipes. Outcome labels (Amendment 1): **A**, support with a sign reversal; **A′**, support only from *equivalent* ↔ *better* changes; **B1**, no support with ≥ 16 of 24 intervals definitive (stable); **B2**, fewer than 8 definitive; otherwise **intermediate**.

**Home-recipe analysis (secondary).** The home advantage of an architecture is the interaction residual of its home cell. If CSOANet-v2's is significantly positive, the result is read as consistent with co-adaptation; if it is equivalent to zero (TOST against ±2; [27]) and its residual range is the smallest, as robustness; otherwise as unresolved.

**Training runs as clusters (post hoc).** All test subjects of a cross-subject fold are scored by the same networks, so their errors are correlated within each dataset × fold cluster (10 clusters). H1 and H2 are therefore also fitted with crossed random effects for subject and training run, taking the best converged fit over four optimisers. Single contrasts are tested with a wild cluster bootstrap (Webb weights, 9,999 draws; [28]): p from the restricted bootstrap and 95 % intervals from the unrestricted bootstrap-t, so a bound near zero and a p near 0.05 can disagree. Bootstrap seeds are fixed by the data, so an identical contrast gives an identical interval wherever it is reported, and the Monte Carlo error of a p near 0.05 is about ±0.004. All numbers in Sections 4.2–4.4 come from one script (supplementary material S4, row 23).

**Logit scale (post hoc).** Interactions on bounded accuracies depend on where each architecture sits on the scale, so the main tests are repeated on the empirical logit of each subject's seed-averaged correct count. The percentage scale is primary, as registered; we report both wherever they disagree.

**Multiplicity.** The 51 exploratory contrasts of Sections 4.2–4.4 are listed in supplementary material S9.14 with unadjusted and Holm-adjusted p. We give unadjusted intervals in the text and state where a conclusion does not survive adjustment.

## 3.4. Preregistration, pilots and deviations

The registration (1 November 2026) and Amendments 1 and 2 (13 and 15 November) were public before the datasets were downloaded on 15 November. Two pilots preceded the study (supplementary material S7). A four-class pilot motivated the hypotheses. A binary pilot on PhysioNet-MI, the dataset on which CSOANet and R1 were developed, was run after the decision rules were committed; in it, the H2b contrast was +0.29 points, which lowered our prior for H2b but changed no rule. Supplementary material S4 lists every change after the code freeze; none alters a hypothesis or threshold.

# 4. Results

## 4.1. Registered analysis

F1 comprised 106 subjects and 480 runs. Forty-one runs, all under R3, were flagged: all 30 ShallowConvNet runs, which diverged after 12–34 epochs, and 11 of 30 CSOANet-v2 runs, which failed to learn.

![**Figure 1.** Subject-level float32 balanced accuracy in each architecture × recipe cell of F1 (○ Lee2019-MI, △ Cho2017; N = 106). Black: cell mean with 95 % CI; dashed line: chance; framed panels: home cells; filled markers: the subject's cell contains a flagged run.](/home/claude/work/s4/figs/fig1_cells_V0.2.png)

**H1 and H2** were supported. The recipe changed accuracy (χ²(3) = 160.5, Holm p = 2.9 × 10⁻³⁴), and the change depended on the architecture (χ²(9) = 414.7, p = 3.0 × 10⁻⁸³). These registered tests treat subjects as independent; with crossed subject and training-run effects they fall to χ²(3) = 29.1 and χ²(9) = 95.3 (both p < 10⁻⁵), and on the logit scale to 31.9 and 75.4. R1 was the worst recipe for every baseline, and R3 for CSOANet-v2 (Table 2; Figure 2, top).

**H2b** was supported: from R1 to R2, EEGNet gained 5.29 points and CSOANet-v2 1.79, so c̄ = 3.50 (95 % CI 2.29, 4.71; Holm p = 9.8 × 10⁻⁸). With shared training runs accounted for, the interval is (1.38, 5.61) (wild cluster bootstrap, p = 0.002); on the logit scale c̄ = 0.22 (0.11, 0.33) (Figure 3, left).

**Home recipe.** CSOANet-v2's home residual was 3.55 (2.41, 4.67), so the registered reading is (i), consistent with co-adaptation. The raw means do not support co-adaptation, however: CSOANet-v2's best recipe was R2, not R1, and its residual was raised by its −6.69 residual under R3 (residuals in a row sum to zero).

**H3** was supported with outcome A′: 20 of 24 intervals were definitive, and the one definitive change was EEGNet − MI-BMInet, *better* for MI-BMInet under R1 (−2.62) and *equivalent* under the other recipes. No pair reversed. With t(9) critical values for the ten clusters (post hoc), only 13 intervals stayed definitive and the outcome became intermediate, so the registered change did not survive accounting for shared training runs (Table 3).

**Secondary results** (supplementary material S5 and S9). F2 supported no hypothesis (accuracies 58–66 %, 124 of 2,592 runs flagged). int8 left every decision unchanged. Condition T kept the R1 order and gained nothing against R4. Registered sensitivity analyses, including an aligned-rank-transform ANOVA [29], left the decisions unchanged.

**Table 2.** Cell means, F1, float32 (%). Registered recipes R1–R4 for the four registered architectures (home cells in bold). Post hoc: R1ct (crop also at validation and test), R1ns (crop without shifting the kept segment), R1nc (no crop), R2va (R2 with the checkpoint chosen by validation accuracy) and ShallowConvNet-MN (with max-norm constraints). The corrected design uses R1nc, R2, R3, R4 and ShallowConvNet-MN in place of ShallowConvNet. — = not run.

| | R1 | R1ct | R1ns | R1nc | R2 | R2va | R3 | R4 |
|:------------------|:------|:------|:------|:------|:------|:------|:------|:------|
| EEGNet | 70.6 | 73.9 | 75.1 | 76.5 | **75.9** | 77.4 | 76.6 | 76.8 |
| ShallowConvNet | 69.8 | 70.5 | 71.3 | 73.8 | 71.9 | 74.4 | 71.6 | 74.4 |
| ShallowConvNet-MN | 69.7 | — | — | 73.6 | 75.3 | — | 76.0 | 75.4 |
| MI-BMInet | 73.2 | 74.8 | 75.5 | 77.0 | 76.1 | 77.2 | **76.9** | 76.6 |
| CSOANet-v2 | **69.1** | 70.2 | 69.8 | 70.2 | 70.9 | 71.6 | 59.3 | 70.1 |

## 4.2. Two implementation defects

**The crop.** Switching off only the crop (R1nc − R1) raised every baseline: EEGNet by 5.86 (3.71, 8.08), ShallowConvNet by 3.98 (2.55, 5.33) and MI-BMInet by 3.74 (2.01, 5.45). CSOANet-v2 changed by 1.07 (−0.06, 2.18), not significantly. R1nc matched R4 for every architecture (Table 2). Removing the crop widened EEGNet's lead over CSOANet-v2 by 4.78 points (2.66, 6.88; logit 0.26, 0.12 to 0.39), most of the 5.20-point narrowing that R1 produced relative to R4. The three effects on the baselines and the gap change survive Holm adjustment.

Two variants show that the crop's harm was not only a train/test mismatch (Table 2). Cropping validation and test windows as well (R1ct) left EEGNet, ShallowConvNet and MI-BMInet 2.1–3.3 points below R1nc (all intervals excluding zero), while CSOANet-v2 under R1ct was indistinguishable from R1nc (+0.03; −0.46, 0.52). Not shifting the kept segment (R1ns) also left the baselines below R1nc, by 1.4–2.5 points. R1ct and R1ns each differ from R1 in more than one respect (R1ct crops test windows at one fixed offset; R1ns still zero-fills), so they do not decompose the harm into additive parts. What they do show is that consistency between training and test would not have removed the crop's bias: under R1ct the EEGNet − CSOANet-v2 gap was still 2.55 points (0.77, 4.33) narrower than under R1nc.

**ShallowConvNet without max-norm.** Adding the reference constraints changed nothing under R1 (−0.10; −0.66, 0.46) or R1nc (−0.20; −0.90, 0.48), but raised ShallowConvNet by 3.48 (1.95, 5.01) under R2 and by 4.39 (2.58, 6.26) under R3, where none of the 30 constrained runs diverged. Under R2, the constraints moved the selected checkpoint from a median epoch of 6.5 to 64: R2 selects the checkpoint with the lowest validation loss, and the unconstrained network reached it after a few epochs. These two cells produced the interactions involving ShallowConvNet in the registered table. They also changed its pairwise categories: against EEGNet and MI-BMInet it was clearly behind under R2–R4 in the registered table, and indistinguishable from them with the constraints (Table 3).

## 4.3. Corrected design

With R1nc in place of R1 and ShallowConvNet-MN in place of ShallowConvNet (Table 2; Figure 2, bottom):

- **H3:** outcome **B1**. Eighteen of 24 intervals were definitive, and no pair moved between definitive categories under any two recipes (Table 3). The only category changes were between *inconclusive* and *better* for ShallowConvNet-MN, which was behind EEGNet and MI-BMInet under R1nc (by 2.87 and 3.37 points) and indistinguishable from them under R2–R4. EEGNet and MI-BMInet were *equivalent* under every recipe, and CSOANet-v2 was behind every other architecture under every recipe.
- **H1 and H2** remained significant with crossed effects (χ²(3) = 14.6, p = 0.002; χ²(9) = 99.5, p = 2 × 10⁻¹⁷; logit 11.1 and 74.4). Almost all of the interaction is one cell: CSOANet-v2 under R3 (59.3 %, residual −6.59; Section 4.4). Without R3, neither a recipe effect (χ²(2) = 1.1, p = 0.58) nor an interaction (χ²(6) = 10.2, p = 0.11) remained, and on the logit scale both p values were above 0.2.
- **H2b** reversed sign on the percentage scale: −1.29 (−2.23, −0.32; p = 0.015). EEGNet gained slightly less than CSOANet-v2 from R1nc to R2. On the logit scale the contrast was −0.04 (−0.12, 0.05; p = 0.34), and it does not survive Holm adjustment over the exploratory set, so we read it as no difference rather than as a reversal.
- **Home recipe.** CSOANet-v2's home residual was 2.20 (1.15, 3.29), again carried by its R3 cell; without R3 it was 0.01 (−0.43, 0.46).

![**Figure 2.** F1 cell means (left; 95 % CIs, points offset by architecture) and interaction residuals of the 4 × 4 table (right; framed: home cells). Top: as registered. Bottom: corrected design, with R1 without the crop and ShallowConvNet with max-norm constraints; all other cells are the registered runs.](/home/claude/work/s4/figs/fig2_registered_vs_corrected_V0.5.png)

![**Figure 3.** Per-subject H2b contrast c[s] (circles Lee2019-MI, triangles Cho2017) with the registered R1 (left) and with R1 without the crop (right). Diamonds: mean with wild cluster bootstrap 95 % CI. Shaded band: ±2-point SESOI.](/home/claude/work/s4/figs/fig3_h2b_V0.5.png)

**Table 3.** Pairwise categories, F1, float32 (A = first better, B = second better, E = equivalent, I = inconclusive; registered Bonferroni 99.79 % intervals, subjects independent) with the mean difference (points). Top: registered design (outcome A′; 13 of 24 intervals definitive with t(9) critical values). Bottom: corrected design, in which R1 is R1 without the crop and ShallowConvNet has max-norm constraints (outcome B1; post hoc).

| Pair | R1 | R2 | R3 | R4 |
|:----------------------------|:-----------|:-----------|:-----------|:-----------|
| ***Registered*** | | | | |
| EEGNet − ShallowConvNet | I +0.79 | A +4.05 | A +4.98 | A +2.46 |
| EEGNet − MI-BMInet | B −2.62 | E −0.18 | E −0.33 | E +0.25 |
| EEGNet − CSOANet-v2 | I +1.49 | A +4.99 | A +17.30 | A +6.70 |
| ShallowConvNet − MI-BMInet | B −3.41 | B −4.23 | B −5.30 | B −2.21 |
| ShallowConvNet − CSOANet-v2 | I +0.70 | I +0.94 | A +12.33 | A +4.24 |
| MI-BMInet − CSOANet-v2 | A +4.12 | A +5.17 | A +17.63 | A +6.45 |
| ***Corrected*** | | | | |
| EEGNet − ShallowConvNet | A +2.87 | I +0.56 | I +0.58 | I +1.45 |
| EEGNet − MI-BMInet | E −0.50 | E −0.18 | E −0.33 | E +0.25 |
| EEGNet − CSOANet-v2 | A +6.28 | A +4.99 | A +17.30 | A +6.70 |
| ShallowConvNet − MI-BMInet | B −3.37 | I −0.74 | I −0.91 | I −1.21 |
| ShallowConvNet − CSOANet-v2 | A +3.41 | A +4.43 | A +16.72 | A +5.25 |
| MI-BMInet − CSOANet-v2 | A +6.78 | A +5.17 | A +17.63 | A +6.45 |

## 4.4. CSOANet-v2 under R3 and the R2 checkpoint rule

**CSOANet-v2 under R3.** R3 uses Adam with a learning rate of 10⁻², ten times that of the other recipes. Of the 30 CSOANet-v2 runs under R3, 10 learned normally: their median training accuracy was 73.8 %, the selected checkpoint came late (median epoch 95) and their test accuracy was 0.66 points below the same fold and seed under R4. The other 20 stayed near chance (median test accuracy 52.6 %, checkpoint at median epoch 12), 11 of them flagged. In 5 of the 10 folds, some seeds learned and others did not. The cell is therefore an optimisation failure at that learning rate, not a recipe preference, and we do not read the R3 column of the interaction as evidence about recipes.

**The R2 checkpoint rule.** Selecting R2's checkpoint by validation accuracy (R2va) raised all four architectures, by 1.45 points on average (0.27, 2.60): EEGNet 1.47, ShallowConvNet 2.57, MI-BMInet 1.09 and CSOANet-v2 0.66 (only ShallowConvNet's interval excludes zero). Choosing the checkpoint by validation loss on about four validation subjects thus cost every architecture something, and the unconstrained ShallowConvNet most. With R2va in place of R2, the ShallowConvNet − CSOANet-v2 contrast between R4 and R2 fell from 3.30 (2.05, 4.53) to 1.38 (−0.11, 2.89), which is no longer significant but is not zero. Neither the averaged effect nor any single R2va effect survives Holm adjustment.

# 5. Discussion

## 5.1. Was the shared recipe neutral?

As registered, no: the recipe developed with our model narrowed that model's gap to EEGNet by 3.5 points relative to a published recipe, and the interval stays clear of zero when shared training runs are taken into account. Every registered positive result, however, traces to three sources: a crop in that recipe, which harmed the other architectures more than ours; a ShallowConvNet without its reference constraints, which failed under the two loss-selected recipes; and CSOANet-v2's optimisation failure at R3's learning rate. With the two defects corrected, no pair of architectures moved between definitive categories under any of the four recipes, and without R3 no recipe effect or interaction remained. In this comparison, then, the recipes were neutral once the implementations were correct, apart from one model's optimisation failure under one recipe.

That is not a reassuring result for shared recipes in general, because the defects were not exotic. The crop was in the recipe developed with one of the compared models, and that model was the one least harmed by it, so the defect favoured its proposer without anything in that model's own results to reveal it. We suggest, as a hypothesis, that such defects are least likely to be noticed in the recipe a model was developed with. The study shows one way a shared recipe can bias a comparison; it does not estimate how often this happens.

## 5.2. What the crop shows about augmentation checks

A train/test consistency check would have caught only part of the crop's effect. Under R1ct, with training and test windows cropped alike, the baselines still lost 2–3 points and CSOANet-v2 none. Why CSOANet-v2 was insensitive is not resolved. A difference in temporal kernel length is not the reason, since EEGNet and CSOANet-v2 use the same 125-sample kernel (supplementary material table S10.1). Other candidates, such as underfitting (CSOANet-v2's mean training accuracy was 76 % outside R3, against 80–86 % for the others), zero padding entering batch-normalisation statistics, or one offset per batch, were not tested. The practical check is the one that found the defect here: turn each augmentation off, one at a time, for every compared architecture, not only for the proposed one.

## 5.3. The training run as a unit of uncertainty

The registered H3 result survived when subjects were treated as independent and was lost under the small-sample correction appropriate for ten clusters. In a cross-subject pairwise difference the subject variance cancels, and what remains is the training-run component, which does not average out over the subjects of a fold. Intervals that treat subjects as independent are therefore too narrow, and so were the registered power calculations (supplementary material S8). The ten clusters are also not independent: two folds of one dataset share about three quarters of their training subjects. Only repeated splits with different seeds would remove this, and we did not run them.

## 5.4. Conflict of interest

The first author developed CSOANet and R1; the second author was not involved in either. To limit the influence of this interest, we registered the direction of H2b and all thresholds in advance, included two recipes published by others, and made the home-recipe rules asymmetric against our model. CSOANet-v2 was the least accurate architecture under every recipe, registered or corrected.

The crop implementation comes from the code of the original CSOANet study [2]. Re-reading that code for this study, we found that its published Methods did not describe the training accurately. In its within-subject analysis, only CSOANet2026 was trained with AdamW, label smoothing and augmentation; the deep-learning baselines were trained with Adam, plain cross-entropy and no augmentation, and every model's checkpoint was selected on the evaluation fold. In its leave-one-subject-out analysis, the baselines received the same recipe except the random crop. We have notified the journal, and a correction is in preparation. That comparison therefore gave the proposed model a different recipe from the baselines, which is the kind of confound examined here.

## 5.5. Limitations

- **The corrected design is post hoc.** It was defined after the registered results and the defects were known, and its tests are exploratory. Its R2, R3 and R4 cells for EEGNet, MI-BMInet and CSOANet-v2 are the registered runs.
- **Scope.** Two-class motor imagery, two datasets and four compact convolutional architectures, one of them ours and the least accurate. There are no FBCSP or Riemannian baselines and no more recent architectures such as EEG-TCNet [30].
- **Recipes are packages.** R1, R2 and R3 differ in several components at once. Only R1 against R4, and the post hoc variants of R1 and R2, isolate one, and the R1 variants do so only approximately.
- **Baseline implementations.** ShallowConvNet's constraints were omitted in the registered runs. EEGNet's published 500 epochs were shortened to 100, and MI-BMInet's dropout is not stated in its paper (supplementary material S1).
- **R3.** CSOANet-v2's failure under R3 was diagnosed from training logs, not tested by rerunning at a lower learning rate.
- **Splits, validation and seeds.** One split seed; about four validation subjects per fold; three seeds per cell, with seed variability larger than the SESOI in most cells.
- **Models.** The confirmatory driver did not retain trained models; all models from the follow-up runs are retained.

## 5.6. Independent test

An independent test on Dreyer2023, a left/right-hand dataset that neither author has analysed, is being preregistered (OSF 8st47, Amendment 3). It will test the crop effect on the EEGNet − CSOANet-v2 gap with ten folds. Its second registered hypothesis, on R2 and ShallowConvNet, is withdrawn, because the follow-up runs traced that effect to our implementation.

# 6. Conclusion

In this preregistered comparison of four compact motor-imagery decoders under four shared recipes, the registered analysis found that the recipe changed the comparison and favoured the model it was developed with. That result came from two defects in our implementation, one of them a crop in that recipe, and from one model's optimisation failure under one recipe. With the defects corrected, no pairwise conclusion changed between recipes.

Comparisons of compact decoders should:

- switch each augmentation off, one at a time, for every compared architecture, and check that it is applied consistently at training and test;
- check baseline implementations against their references, including weight constraints, before comparing recipes;
- state who designed the shared recipe, and include at least one recipe not developed with any compared model;
- treat the training run, not only the subject, as a unit of uncertainty.

# Acknowledgments

[Funding: state any institutional support (FEUP / FMUP) or 'No specific funding was received for this work.']

# Conflict of interest

The first author developed CSOANet [2] and the R1 training recipe evaluated in this study; the second author was not involved in either. The direction of the main contrast and all decision thresholds were preregistered before data access, and the home-recipe rules were set against the first author's model (sections 3.3 and 5.4). No other competing interests.

# Data availability statement

The data that support the findings of this study are openly available. Lee2019-MI [24] and Cho2017 [25] were accessed through MOABB 1.7.2 [7]. All per-subject, per-seed results (float32 and int8) are available in the OSF project at https://osf.io/hfnkm. The confirmatory code is available at https://github.com/jabarrick/recipe-x-arch (tag v1.0-frozen, commit a44e5933; confirmatory driver at tag v1.1-frozen, commit 1a7ba7b8) and archived at Zenodo [DOI to be minted at submission]. Trained models from the confirmatory runs were not retained; they can be regenerated with the frozen code and the registered seeds, up to GPU non-determinism. Models from all follow-up runs are retained and will be archived with the code.

# Preregistration

The study was preregistered on OSF (8st47; https://doi.org/10.17605/OSF.IO/8ST47) on 1 November 2026. Amendment 1 (code freeze and analysis-plan addendum) was public on 13 November 2026 and Amendment 2 (freeze of the confirmatory driver) on 15 November 2026, both before the first confirmatory download. Deviations are listed in supplementary material S4.

# Ethical statement

This study used only publicly available, anonymised datasets [24, 25], collected by their original investigators under their own ethical approvals. No new data were collected and no further approval was required.

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

[24] Lee M-H, Kwon O-Y, Kim Y-J, Kim H-K, Lee Y-E, Williamson J, Fazli S and Lee S-W 2019 EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy *GigaScience* **8** giz002 (https://doi.org/10.1093/gigascience/giz002)

[25] Cho H, Ahn M, Ahn S, Kwon M and Jun S C 2017 EEG datasets for motor imagery brain–computer interface *GigaScience* **6** gix034 (https://doi.org/10.1093/gigascience/gix034)

[26] Holm S 1979 A simple sequentially rejective multiple test procedure *Scand. J. Stat.* **6** 65–70

[27] Schuirmann D J 1987 A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability *J. Pharmacokinet. Biopharm.* **15** 657–80 (https://doi.org/10.1007/BF01068419)

[28] Cameron A C, Gelbach J B and Miller D L 2008 Bootstrap-based improvements for inference with clustered errors *Rev. Econ. Stat.* **90** 414–27 (https://doi.org/10.1162/rest.90.3.414)

[29] Wobbrock J O, Findlater L, Gergle D and Higgins J J 2011 The aligned rank transform for nonparametric factorial analyses using only ANOVA procedures *Proc. SIGCHI Conf. on Human Factors in Computing Systems* (New York: ACM) pp 143–6 (https://doi.org/10.1145/1978942.1978963)

[30] Ingolfsson T M, Hersche M, Wang X, Kobayashi N, Cavigelli L and Benini L 2020 EEG-TCNet: an accurate temporal convolutional network for embedded motor-imagery brain–machine interfaces *Proc. 2020 IEEE Int. Conf. on Systems, Man, and Cybernetics (SMC)* pp 2958–65 (https://doi.org/10.1109/SMC42975.2020.9283028)
