---
title: "Is a shared training recipe neutral? Recipe × architecture interactions in compact motor-imagery EEG decoders"
author:

  - Yu Gao (Faculty of Engineering, University of Porto; CISTER Research Centre, ISEP)
  - Yimin Zhang
date: "Manuscript V1.5 — 25 November 2026 (fourth revision after internal review)"
---

# Abstract

Compact motor-imagery EEG decoders are usually compared under one shared training recipe, assumed not to affect their ranking. We tested this in a preregistered 4 × 4 design (OSF 8st47). EEGNet, ShallowConvNet, MI-BMInet and CSOANet-v2 were each trained under four recipes: the authors' heavy-augmentation recipe, developed with CSOANet-v2; two published plain recipes; and the heavy recipe without augmentation. Data were Lee2019-MI and Cho2017 (106 subjects, cross-subject, three seeds; a within-subject family as secondary), and every decision rule was fixed before download. As registered, the shared recipe was not neutral. The recipe effect differed across architectures, and moving from a published plain recipe to the authors' recipe narrowed EEGNet's lead over CSOANet-v2 by 3.5 points [2.3, 4.7]. Pairwise conclusions changed only between "better" and "equivalent", and that change was fragile to how subjects sharing trained networks were treated. Exploratory analyses traced the effect to one implementation defect: a random crop applied in training but not at test, which cost the baselines 3.7–5.9 points and CSOANet-v2 1.1. Without it, and without one numerically unstable recipe, all interaction residuals were within ±1 point and equivalent to zero. The evaluation protocol mattered more than any recipe. Cross-subject, CSOANet-v2 ranked last under every recipe; within subjects, it ranked first under three of four. Shared-recipe comparisons should check each augmentation for train/test consistency, state who designed the recipe, and report results under more than one evaluation protocol.

**Keywords.** motor imagery; EEG decoding; training recipe; benchmark; preregistration; quantisation; compact neural networks.

# 1 Introduction

Comparing EEG decoders is done in one of two ways. In the first, each model is trained with the recipe its authors published — their optimiser, schedule, augmentation and model-selection rule — so that architecture and recipe change together and the comparison cannot say which one made the difference. In the second, every model is trained with one shared recipe, chosen by whoever runs the comparison (Wimpff et al., 2024). The second practice is usually presented as the fairer one, and it does remove the proposer's tuning advantage. But it rests on an assumption that has not been tested: that a different shared recipe would have ranked the architectures the same way. If the ranking depends on the recipe, then a shared recipe does not remove a bias; it chooses one.

There is reason to expect that it does depend. Upstream of training, preprocessing choices already change decoding performance in model-dependent ways: varying filters, referencing and artefact correction in a multiverse design shifted performance considerably and differently across classifiers (Kessler, Enge and Skeide, 2025), and at the level of single trials up to 42 % of predictions can flip when only the preprocessing changes (Hou et al., 2026). Data augmentation for EEG helps or hurts depending on the architecture and the amount of data (Rommel et al., 2022). In other areas of machine learning, equalising the tuning budget has repeatedly shrunk or erased reported architectural progress — in language modelling, generative adversarial networks, reinforcement learning, metric learning and recommendation (Melis et al., 2018; Lucic et al., 2018; Henderson et al., 2018; Musgrave et al., 2020; Dacrema et al., 2019) — and in vision a modern training recipe lifted a 2015 ResNet to the level of architectures published years later (Wightman et al., 2021). The training recipe, as a package, has not been given the same scrutiny in EEG decoding.

The question matters most for compact decoders — EEGNet (Lawhern et al., 2018), ShallowConvNet (Schirrmeister et al., 2017), MI-BMInet (Wang et al., 2024) and the gated CSOANet family (Gao and Diniz, 2026) — because these are the models that actually run on wearable hardware, because their published recipes differ from one another as much as their topologies do, and because small models may have less capacity to absorb a recipe that does not suit them. Deployment papers report int8 accuracy for one model under its own recipe; whether quantisation preserves the conclusions *between* architectures, and whether that depends on the recipe, is not known.

Three things could be true. The recipe could matter (a main effect); it could matter differently for different architectures (an interaction); or it could change which architecture comes out ahead. The third is the one that would most directly undermine shared-recipe comparisons, but it is also the hardest to decide. Differences between compact decoders on public benchmarks are often of the order of a few points, and classifying a difference as *equivalent* needs an adjusted confidence interval inside ±2 points, which a sample of this size can reach only for some pairs (Section 3.6.6). We therefore designed the study so that its main deliverables do not depend on that classification. The first-order questions — how much a recipe moves each architecture, whether the movement differs across architectures, and whether what looks like robustness to the recipe is in fact each architecture preferring the recipe it was developed with — can be answered at this sample size, and they are what a reader choosing a comparison protocol or a deployment model needs to know. The ranking question is kept as registered and reported for what the sample can say.

Our own pilot work made the question concrete. On PhysioNet-MI and BCI Competition IV 2a, changing only the recipe moved EEGNet by 6.6–8.7 points, and other architectures by less. The pilot was not evidence — it was built up configuration by configuration on the dataset on which our own model and recipe were developed — but it was enough to preregister the study on two datasets that neither author had opened, with every hypothesis, threshold and analysis fixed before download (Section 3.8).

We ask three questions. *RQ1:* with the architecture fixed, does the training recipe change test accuracy? *RQ2:* does the size of that change depend on the architecture, and in particular, does a shared recipe developed with one of the compared models favour that model? *RQ3:* does the conclusion about which architecture is better change with the recipe, to the extent this sample can decide? Four compact architectures are crossed with four recipes — a heavy-augmentation recipe, two published plain recipes, and the heavy recipe with its augmentation removed — on two public left- versus right-hand motor-imagery datasets, cross-subject, with three seeds per cell. RQ1 and RQ2 are tested with a subject-level mixed model (H1, H2), a directional contrast (H2b), and a preregistered home-recipe analysis that measures each architecture's advantage under its own recipe as an interaction residual, with an equivalence test for the authors' own model (Section 3.6.4). RQ3 is tested as registered, by classifying all 24 pairwise differences with Bonferroni-adjusted intervals and a 2-point equivalence margin (H3), and is reported together with the full matrix of pairwise effect sizes and intervals so that readers can see the magnitudes rather than only the categories.

The contributions are:

- **(i) The design.** A preregistered 4 × 4 two-factor design on untouched data, giving controlled estimates of recipe main effects, recipe × architecture interactions and per-architecture recipe ranges for compact motor-imagery decoders.
- **(ii) A test of a shared recipe against its authors' own model.** A registered directional contrast, run on a design in which the model under comparison and the shared recipe come from the same group. It is reported together with a preregistered home-recipe analysis whose limits we make explicit. The analysis can rule out the reading favourable to the authors' model, but a double-centred residual cannot by itself identify co-adaptation (Section 5.3). A post hoc search-space analysis identified a concrete mechanism, one augmentation in the authors' recipe that harms the baselines far more than the authors' model, and an exploratory rerun was consistent with it.
- **(iii) An equal-budget comparison.** A per-architecture recipe search with an equal budget, compared with the shared recipes. At this budget it did not improve on the best fixed recipe.
- **(iv) Deployment.** The same analyses after int8 quantisation, with operator coverage reported.
- **(v) Statistical units.** An illustration, in an EEG benchmark, of a point known from nested designs (Bouthillier et al., 2021): in cross-subject evaluation the training run, not only the subject, is a unit of uncertainty for pairwise conclusions.

For the ranking question we report all 24 pairwise intervals, the number of conclusion changes out of 36 opportunities, and the precision the sample achieved. We stated in advance that a result in which most intervals are inconclusive is read as uninformative, not as support for a shared recipe.

# 2 Related work

## 2.1 Compact decoders for motor imagery

Four convolutional families account for most deployed MI decoders. EEGNet (Lawhern et al., 2018) stacks a temporal convolution, a depthwise spatial convolution and a separable convolution into a few thousand parameters; ShallowConvNet (Schirrmeister et al., 2017) is a learned filter-bank common-spatial-pattern analogue with a square–pool–log non-linearity; MI-BMInet (Wang et al., 2024) is a separable-convolution design co-optimised for microcontroller inference and 8-bit quantisation; CSOANet (Gao and Diniz, 2026) adds a static gating path to a long-kernel temporal front end and was developed by the present authors for the same deployment target. Each was published with its own training recipe, and the recipes differ in optimiser, schedule, augmentation and model-selection rule as much as the architectures differ in topology (Section 3.3). The architectures have been compared many times, almost always under the proposer's recipe for the proposed model and either the original or a convenient recipe for the baselines.

The four designs are competitive on public MI benchmarks and are the realistic options when inference must run on a wearable. They have not been compared under a common set of recipes in which every recipe is applied to every architecture.

## 2.2 Benchmarks and evaluation protocols

MOABB (Aristimunha et al., 2023; Chevallier et al., 2024) standardised datasets, splits and metrics for BCI, and its large reproducibility study showed how much reported differences shrink once data and evaluation are held fixed. Braindecode (Schirrmeister et al., 2017) provides reference implementations of the standard deep decoders. Within this ecosystem, two practices coexist. Either each model is trained with its published recipe, which confounds architecture with recipe; or all models are trained with one shared recipe, as in the comparative framework of Wimpff et al. (2024), where EEGNet, ShallowConvNet and attention variants were all trained with the same optimiser, learning rate, schedule and epoch budget. The shared recipe is usually presented as the fairer choice, and in one sense it is: it removes the proposer's tuning advantage. But it rests on an untested assumption — that the ranking of architectures does not depend on which shared recipe was chosen.

Holding data and splits fixed is necessary and is now routine. Whether holding the recipe fixed is neutral, i.e. whether a different shared recipe would have produced a different ranking, has not been tested in either practice.

## 2.3 Preprocessing and recipe choices change decoding conclusions

The closest evidence concerns preprocessing rather than training. Kessler, Enge and Skeide (2025) varied filtering, referencing, baseline correction, detrending and artefact correction in a multiverse design and found that preprocessing choices shifted decoding performance considerably and differently for different classifiers; their data are event-related potentials, but the design point carries over. Hou et al. (2026) went to the trial level and reported that across six datasets and four paradigms up to 42 % of single-trial predictions flip when only the preprocessing changes, with a near-additive structure across binary interventions but non-trivial interaction shares on some datasets. Del Pup et al. (2025) found across thousands of models that minimal preprocessing often beats heavy pipelines for deep decoders. On the training side, Rommel et al. (2022) compared EEG data-augmentation strategies systematically and found their benefit to depend on the task, the architecture and the amount of data — an architecture × augmentation interaction, measured without a factorial design.

Upstream choices are not neutral, their effect is model-dependent, and at the trial level it is large. The training recipe — optimiser, schedule, augmentation and selection rule as a package — has not been crossed with architecture in a factorial design with a preregistered test of whether architecture conclusions change.

## 2.4 Tuning asymmetry in other fields

Machine-learning subfields that compared architectures under equal tuning budgets have repeatedly found that apparent architectural progress shrinks or disappears. Melis, Dyer and Blunsom (2018) showed that a well-tuned LSTM matched or beat newer language-model architectures; Lucic et al. (2018) found no consistent winner among GAN variants once budgets were equalised; Henderson et al. (2018) showed that deep reinforcement-learning results depend on seeds and hyperparameters as much as on algorithms; Musgrave, Belongie and Lim (2020) and Dacrema, Cremonesi and Jannach (2019) reached the same conclusion for metric learning and neural recommendation. In vision, Wightman, Touvron and Jégou (2021) showed that a modern recipe lifts a 2015 ResNet-50 to the level of architectures published years later: a recipe upgrade erased what had been read as an architectural generation gap. Two further strands bear directly on our design. Seed variance is of the same order as reported architecture differences in many benchmarks (Bouthillier et al., 2021; Picard, 2021), which is why we report per-cell seed variability against the SESOI. And benchmark-dependence and co-adaptation — a method performing best under the conditions it was developed in — are documented as structural features of empirical ML (Dehghani et al., 2021; Sculley et al., 2018); they are the basis of the home-recipe analysis in Section 3.6.4, which asks whether each architecture simply prefers the recipe it was born with.

In several fields, tuning asymmetry can create or erase architecture differences, and co-adaptation is a recognised mechanism. EEG decoding has no equivalent controlled test, and compact decoders, whose small capacity may make them more sensitive to the recipe, are where such a test matters most for practice.

## 2.5 Microcontroller inference and int8 quantisation

Deploying MI decoders on microcontrollers is an active line of work: MI-BMInet (Wang et al., 2024) and EEG-TCNet (Ingolfsson et al., 2020) report 8-bit implementations on parallel ultra-low-power MCUs with negligible accuracy loss, and the TinyML toolchain (TensorFlow Lite Micro, ONNX Runtime, CMSIS-NN) makes post-training quantisation routine. Deployment papers, however, report quantised accuracy for one model under its own recipe. Whether int8 quantisation preserves the pairwise conclusions between architectures, and whether the answer depends on which recipe the models were trained with, has not been asked; nor has operator coverage — the fraction of a network that actually runs in int8 — been reported alongside, although architectures with square–log non-linearities or depthwise paths may be only partly quantised.

Compact MI decoders can run in int8 on MCUs with small accuracy loss under their own recipe. A comparison of architectures after quantisation across several recipes, with operator coverage reported so that toolchain effects are not mistaken for architecture effects.

# 3 Methods

The study is a preregistered two-factor experiment: four compact motor-imagery (MI) decoders (factor A, architecture) are each trained under four fixed training recipes (factor R, recipe), on two public left- versus right-hand MI datasets, with three seeds per cell. The primary outcome is subject-level balanced accuracy. All hypotheses, decision rules, and the analysis code were fixed before either confirmatory dataset was downloaded; the registration and Amendment 1 are public, and the confirmatory code is frozen at a tagged commit (Section 3.8).

## 3.1 Data, evaluation units and preprocessing

### 3.1.1 Datasets and evaluation units

Two public datasets form the confirmatory family F1, and one of them also forms the secondary family F2 (Table 1). Both are loaded through MOABB 1.7.2 (Jayaram and Barachant, 2018; Aristimunha et al., 2023; Chevallier et al., 2024) with the left- versus right-hand MI classes only.

**Table 1.** Evaluation units.

| Unit | Dataset | Subjects | Protocol | Role |
|---|---|---|---|---|
| F1a | Lee2019-MI (OpenBMI; Lee et al., 2019) | 54 | cross-subject, 5-fold | primary confirmatory |
| F1b | Cho2017 (Cho et al., 2017) | 52 | cross-subject, 5-fold | primary confirmatory |
| F2 | Lee2019-MI | 54 | within-subject, session 1 → session 2 | secondary confirmatory |

F1a and F1b are analysed together as family F1 (106 subjects), with dataset as a fixed effect. F1 decides H1, H2 and H2b. F2 is an independent Holm (1979) family whose result cannot change the F1 conclusion.

Lee2019-MI contains two sessions per subject recorded on different days, 62 EEG channels at 1000 Hz, and 100 labelled trials per session in the offline phase that MOABB loads (50 per class). Cho2017 contains a single session per subject, 64 channels at 512 Hz, and 100 trials per class (MOABB 1.7.2 documentation).

Neither author had downloaded, opened or analysed Lee2019-MI or Cho2017 before registration (confirmed 14 October 2026) or before Amendment 1 was public. The two pilot datasets (PhysioNet-MI, BCI IV 2a) were used for development and are disclosed in Section 3.7 and Supplementary S7; they play no role in the confirmatory analysis.

### 3.1.2 Preprocessing

Preprocessing is fixed and identical for all architectures and recipes: common-average reference; band-pass 1–40 Hz; resampling to 250 Hz; epoch 0–4 s after the cue (0–3 s for Cho2017, whose trials are 3 s long); all EEG channels; no artefact rejection beyond MOABB defaults.

Per-channel standardisation differs by protocol and is reported as such. In the cross-subject cells, one gain per channel is estimated on all trials of the training subjects of that fold and applied to validation and test subjects (a cross-subject scalar). In the within-subject cell, one gain per channel is estimated on the subject's session-1 training trials and applied to that subject's session 2. The two procedures are not interchangeable and are not compared.

Because window length and channel count differ across datasets, the parameter count and multiply–accumulate (MAC) count of a given architecture differ across datasets; Table 2 therefore reports them per dataset.

### 3.1.3 Splits

Cross-subject: subjects are shuffled with seed 0 and divided into 5 folds; within each fold, 10 % of the training subjects form the validation set used for early stopping and model selection. Within-subject: 20 % of a subject's session-1 trials form the validation set; session 2 is the test set. Test data are never used for training or model selection. The hyperparameter search of condition T uses the training and validation subjects of fold 1 (Section 3.4); those subjects are test subjects in folds 2–5, so we do not claim that test data are never used for condition T (Section 3.4 states the consequence).

### 3.1.4 Exclusions and retries

Subjects that MOABB cannot load, or who retain fewer than 80 % of their nominal trials after loading, are excluded before training and listed in Supplementary S2. A run whose loss becomes non-finite is retried once with seed + 100; if it fails again, the original result is kept. A run is additionally flagged as a *training failure*, for the sensitivity analysis only, if its training balanced accuracy on the un-augmented training trials at the finally selected checkpoint is ≤ 55 % (Amendment 1). The flag uses training-side signals only; test accuracy never enters any inclusion decision.

## 3.2 Factor A: architectures

Four compact convolutional decoders span the designs most often deployed for MI (Table 2). Each architecture keeps its own weight constraints under every recipe, because we treat those constraints as part of the model definition rather than of the recipe.

**Table 2.** Architectures (input at 250 Hz). Parameters (trainable) and MACs per inference of the deployed graph, per dataset input shape (tools/static_cost.py; Table 11).

| Architecture | Key settings | Params / MACs (Lee2019-MI) | Params / MACs (Cho2017) | Source |
|---|---|---|---|---|
| EEGNet-8,2 | temporal kernel 125; separable kernel 31; dropout 0.25 / 0.5; max-norm 1 / 0.25 | 3,818 / 63.2 M | 3,594 / 48.9 M | Lawhern et al., 2018 |
| ShallowConvNet | 40 temporal filters (kernel 25), 40 spatial filters, square → mean-pool 75/15 → log; dropout 0.5 | 105,202 / 157.4 M | 107,042 / 120.8 M | Schirrmeister et al., 2017 |
| MI-BMInet | N_k = 32, N_f = 64 (the authors' BCI IV 2a configuration); dropout 0.2 cross-subject / 0.5 within-subject (authors' reference configurations; not stated in the paper) | 6,722 / 1.99 M | 6,530 / 1.54 M | Wang et al., 2024 |
| CSOANet-v2 (long kernel) | temporal kernel 125; static gate with bounded term; ReLU; max-norm 0.25 | 3,800 / 2.08 M | 3,704 / 1.57 M | this work (v2 of CSOANet, [CSOANet 2026 citation]) |

**Implementation and verification of baselines.** EEGNet-8,2 is a PyTorch implementation following Lawhern et al. (2018) (kernels fs/2 and fs/8, max-norm 1 and 0.25); ShallowConvNet is numerically identical to braindecode 2.0 `ShallowFBCSPNet` (Aristimunha et al., 2025); MI-BMInet is our implementation of Wang et al. (2024) following the authors' reference configurations. Before any confirmatory run, each baseline was trained with its own published BCI IV 2a recipe on the pilot dataset BCI IV 2a and compared with one reference figure named in advance in Amendment 1: for EEGNet-8,2 and ShallowConvNet, the MOABB benchmark values on BNCI2014-001 (60.46 % and 72.47 % four-class accuracy, within-session 5-fold cross-validation; Chevallier et al., 2024), because Lawhern et al. (2018) report no numeric within-subject figure for this dataset and Schirrmeister et al. (2017) report theirs (73.7 % and 71.9 %) with cropped training; for MI-BMInet, 76.03 % (Wang et al., 2024; session 1 → session 2, 0–3 s, Adam 1e-3, 500 epochs, batch 32). The accepted tolerance, fixed before the runs, was ±3 percentage points; reproduced values, protocol differences and the descriptive comparison with the original papers' figures are in Supplementary S7. EEGNet-8,2 reproduced 61.8 % (+1.3) and MI-BMInet 74.1 % (−2.0), within tolerance; ShallowConvNet reproduced 66.6 % (−5.9), outside it. The pre-specified handling applied: the implementation was verified against braindecode and kept; a diagnostic adding the max-norm constraints of the Keras implementation behind the reference figure gave 74.2 % (+1.7), locating the gap in those constraints rather than in an implementation error (S7).

**Disclosure.** CSOANet-v2 and recipe R1 (Section 3.3) were developed by the authors, together, on the pilot datasets. This is the conflict of interest that the design is built around (Sections 3.3 and 3.6).

## 3.3 Factor R: training recipes

Four recipes are defined identically for all four architectures, so the 4 × 4 design is fully crossed (Table 3). "Recipe" here means everything that is not the model definition or the data: optimiser, learning-rate schedule, weight decay, loss, augmentation, and the model-selection rule.

**Table 3.** Training recipes (constant across architectures).

| Recipe | Optimiser / schedule | Loss / augmentation | Model selection | Origin |
|---|---|---|---|---|
| R1 — heavy augmentation | AdamW, lr 1e-3, weight decay 1e-3, cosine to 1e-5 | CE with mixup 0.2 and label smoothing 0.1; Gaussian noise 0.02; circular time shift ±5 samples; random crop 90 % (one offset per batch, shifted to the window start, last 10 % zero-filled; not applied at test); channel dropout 10 % | early stopping on validation accuracy, patience 15 (cross-subject) / 80 (within-subject) | CSOANet 2026 |
| R2 — plain, constant lr | Adam, lr 1e-3, constant | CE, no augmentation | lowest validation loss | EEGNet (Lawhern et al., 2018) |
| R3 — plain, step schedule | Adam, lr 1e-2, ×0.1 at 40 % and 80 % of epochs; batch 16 | CE, no augmentation | lowest validation loss | MI-BMInet (Wang et al., 2024) |
| R4 — R1 without augmentation | as R1 | CE, no augmentation | as R1 | control |

R1 versus R4 isolates the augmentation package while holding optimiser, schedule and selection rule fixed. The augmentations are applied in a fixed order to each training batch. Cropping draws one random offset per batch, keeps 90 % of the window from that offset, moves it to the start of the window and fills the last 10 % with zeros. Test windows are never cropped. This implementation is shared with the pilot code in which CSOANet-v2 and R1 were developed.† R2 and R3 are the published recipes of two of the architectures in the set. Epochs and batch size are fixed per protocol for all recipes except where a recipe specifies its own batch size: cross-subject E = 100, B = 64; within-subject E = 500, B = 32.

### 3.3.1 Home and away pairs

Three architecture–recipe pairs in the design are *home* pairs: EEGNet with R2, MI-BMInet with R3, and CSOANet-v2 with R1. The coupling is not equally tight. CSOANet-v2 and R1 were developed together by the authors on the pilot data. EEGNet and R2 were published together in 2018 on other data. R3 is the recipe MI-BMInet was published with for *cross-subject* decoding (PhysioNet, 100 epochs, with N_k = 16 and N_f = 128 in that setting; Wang et al., 2024); its within-subject recipe on BCI IV 2a was instead Adam 1e-3 with a constant learning rate, 500 epochs and batch 32, which resembles R2. R3 is therefore MI-BMInet's home recipe in the cross-subject family F1, on which the home-recipe analysis is defined, and not in F2. R4 (R1 without augmentation) is a *near-home* recipe for CSOANet-v2. ShallowConvNet's published recipe is not in the set; it shares Adam and validation-based model selection with R2 but used a two-phase early-stopping scheme and cropped training (Schirrmeister et al., 2017), so ShallowConvNet is the architecture with the weakest home advantage available to it and is reported descriptively on that basis.

Because the registration fixes epochs and batch size per protocol, the home recipes of EEGNet and MI-BMInet are approximations of their published recipes, whereas CSOANet-v2's home recipe is unchanged (Table 4). This asymmetry is taken into account when interpreting the home-recipe analysis (Section 3.6.4).

**Table 4.** Fidelity of the home recipes.

| Home pair | Published recipe | Change in this study | Reason |
|---|---|---|---|
| EEGNet + R2 | Adam (default lr 1e-3), 500 epochs, lowest validation loss; batch size not stated (Lawhern et al., 2018) | cross-subject E = 100, B = 64; within-subject E = 500, B = 32 | protocol unification, fixed at registration |
| MI-BMInet + R3 | cross-subject recipe: Adam 1e-2 → 1e-3 → 1e-4 at epochs 0 / 40 / 80, 100 epochs, batch 16, with N_k = 16, N_f = 128 (Wang et al., 2024, PhysioNet); within-subject recipe on BCI IV 2a was Adam 1e-3 constant, 500 epochs, batch 32 | cross-subject E = 100 (equal to published), batch 16 kept, schedule at 40 % / 80 % of epochs; network kept at N_k = 32, N_f = 64 (the authors' BCI IV 2a configuration); within-subject E = 500 under R3 is not the published within-subject recipe | protocol unification, fixed at registration |
| CSOANet-v2 + R1 | CSOANet recipe | none | derives from the recipe used for CSOANet in Gao and Diniz (2026); finalised for CSOANet-v2 in this project's pilot work |

## 3.4 Condition T: equal-budget recipe search (H4 only)

Condition T asks what each architecture would reach if, instead of a shared recipe, each were given the same small tuning budget. For each architecture and evaluation unit, 24 configurations are drawn at random from a fixed space (optimiser; learning rate log-uniform in [1e-4, 1e-2]; schedule; batch size; each augmentation on or off; label smoothing; model-selection rule; Supplementary S1). Each configuration is scored on validation data only — cross-subject, on the training and validation subjects of fold 1; within-subject, on session 1 of the 10 subjects with the lowest IDs — as the mean validation accuracy over seeds 0 and 1 (Amendment 1; one seed if the compute budget forces a cut, Section 3.8). The best configuration is then trained and evaluated on all folds or subjects with the registered seeds.

**Disclosed leakage.** In the cross-subject cells, the fold-1 training and validation subjects are test subjects in folds 2–5, so hyperparameters are chosen with data from subjects that are later tested. This is a hyperparameter-level leak that favours condition T. The registered procedure is kept unchanged for the H4 decision; a sensitivity analysis evaluates condition T on the fold-1 test subjects only (22 subjects across the two datasets), where no overlap exists. In the within-subject cell, the 10 tuning subjects are tested on session 2, which the search never saw; their results are flagged.

**Condition T′ (leakage-free, pre-specified).** Before the code was frozen, the same 24-configuration search with the same scoring rule was run on the left- versus right-hand runs of the pilot dataset PhysioNet-MI (runs 4, 8 and 12; subjects S088, S092, S100 and S104 excluded, as in the pilot). The configuration selected for each architecture was written into Amendment 1 and is applied unchanged to every confirmatory cell. T′ therefore touches no confirmatory data and is a recipe, not a procedure; it mirrors the engineering practice of tuning on data at hand and deploying on new data. Its cost is cross-dataset transfer, and its disclosure is that PhysioNet-MI is the dataset on which CSOANet-v2 and R1 were developed. T′ is reported alongside the fold-1-only analysis.

Condition T carries two opposing biases that are stated wherever its results appear: the hyperparameter leak favours T, while the winner's curse of selecting the best of 24 noisy scores, scored with only two seeds, deflates T's test performance relative to its validation score. A result in which T does not beat R1 is read as "no benefit at a budget of 24 trials", not as "search does not help".

## 3.5 Seeds, quantisation and outcome measures

Every architecture × recipe × fold (or subject) cell is run with seeds 0, 1 and 2. Seed-averaged accuracies are the unit of analysis, as registered; per-cell seed standard deviations are reported in full in Supplementary S3 and compared with the SESOI, and a model with seed as a random effect is run as a sensitivity analysis.

Each trained model is evaluated in float32 and after post-training int8 quantisation (ONNX Runtime static QDQ; per-channel symmetric weights; MinMax calibration, the simplest default; 256 un-augmented training trials for calibration). The calibration trials are drawn without replacement from the training subjects of the fold (seed = fold index) or from the subject's session-1 training trials (seed = subject index), and the same calibration set is used for every model within a fold or subject, so that int8 differences between models cannot come from different calibration draws (Amendment 1; indices in Supplementary S5). For each architecture we also report the share of operators that ONNX Runtime quantises to int8 and list any left in float32, because an architecture whose activations stay partly in float32 is not being compared on equal terms (Section 4.7).

**Primary outcome.** Subject-level float32 balanced accuracy y[s, a, r], computed as the mean of the two per-class recalls (not plain accuracy, because the exclusion rule can leave a subject with unequal class counts).

**Secondary outcomes.** int8 balanced accuracy; float32–int8 prediction agreement; training and validation accuracy at the selected checkpoint; static cost (parameters, MACs, peak int8 activation memory). No on-device latency is reported here; hardware measurements of CSOANet-v2 are the subject of a separate deployment paper that reuses these architectures and this code. The frozen driver saves no model checkpoints, and its export step deletes each exported ONNX file once the run is scored, so no trained models were retained. They can be regenerated with the frozen code and the registered seeds, up to GPU non-determinism.†

**Derived indices.** The per-subject H2b contrast c[s]; the six per-subject pairwise architecture differences under each recipe; each architecture's recipe range (best minus worst recipe mean); variance shares of A, R and A × R; Kendall's τ between the R1 and T rankings.

## 3.6 Statistical model and decision rules

### 3.6.1 Model

Subject-level balanced accuracy is modelled as

y_{s,a,r} = μ + δ_{d(s)} + α_a + ρ_r + (αρ)_{a,r} + u_s + ε_{s,a,r},   u_s ~ N(0, σ_u²),

with dataset d(s), architecture a, recipe r, and a random intercept per subject. Fixed effects are tested by likelihood-ratio tests on maximum-likelihood fits (not REML), as registered. All reported interaction terms use sum-to-zero coding, so that (αρ)_{a,r} is the double-centred interaction residual of cell (a, r); the home-recipe analysis is defined directly on cell means (Section 3.6.4) and uses the model term only as a check.

### 3.6.2 Smallest effect size of interest

The SESOI is 2 percentage points of balanced accuracy, fixed at registration. Its justification is substantive, not statistical: it is of the same order as the seed-to-seed variability of a single training configuration (median absolute difference between two seeds of the same configuration in the T′ search: 1.1–1.3 points for EEGNet, ShallowConvNet and MI-BMInet, 2.1 for CSOANet-v2), so a difference within ±2 points is one that a change of seed alone could produce or erase. The expected standard error of a seed-averaged cell mean (about 0.6–1 point) is reported separately as a detection floor and is not the reason for the SESOI.

### 3.6.3 Confirmatory hypotheses and tests

- **H1 (recipe main effect).** LRT of the recipe term (3 df).
- **H2 (recipe × architecture interaction).** LRT of the interaction (9 df).
- **H2b (directional).** Per subject, c[s] = (y_{s,EEGNet,R2} − y_{s,EEGNet,R1}) − (y_{s,v2,R2} − y_{s,v2,R1}); two-sided one-sample *t*-test against 0; H2b is supported only if significant *and* positive. Because R1 and R2 differ in optimiser, weight decay, schedule, selection rule and the whole augmentation package, H2b tests sensitivity to the R1 recipe as a package, not to augmentation; the augmentation-specific contrast (R1 vs R4) carries no registered direction and is exploratory.
- H1, H2 and H2b are Holm-corrected within each family at α = 0.05. F1 decides; F2 is an independent family whose negative results are reported as inconclusive (registered power 0.70 and 0.18 for the full and half effect).
- **H3 (stability of architecture conclusions).** For each of the 6 architecture pairs under each of the 4 recipes, the mean paired difference with a Bonferroni-adjusted 99.79 % confidence interval (24 intervals). Each interval is classified as *A better* (excludes 0, positive), *B better* (excludes 0, negative), *equivalent* (includes 0 and lies within ±2 points), or *inconclusive*. An interval that excludes 0 and also lies within ±2 is *better*, not *equivalent* (precedence fixed in Amendment 1). H3 is supported if any pair falls into two different definitive categories under two recipes.
- **H4 (secondary).** The same classification applied to R1 versus condition T, with Kendall's τ between the two rankings; with four architectures τ has a wide interval, so category changes carry the interpretation.

Effect sizes accompany every test: recipe ranges, variance shares, and all pairwise differences with intervals, in percentage points.

**Outcome categories for H3 (Amendment 1).** When H3 is not supported, the 24 intervals are classified as *stable* (B1: at least 16 of 24 definitive and no pair moves between two definitive categories) or *inconclusive* (B2: fewer than 8 definitive); intermediate cases are reported as counts. Only B1 is described as evidence that a shared recipe preserves architecture conclusions; B2 is described as uninformative. We also report the number of category changes among the 36 opportunities (6 pairs × 6 recipe pairs), split into sign reversals, equivalent ↔ better changes, and inconclusive ↔ definitive changes. If all definitive ↔ definitive changes are of the equivalent ↔ better kind with no sign reversal, the result is reported as "pairwise differences below the SESOI whose direction is stable across recipes", not as a change in ranking; a simulation attached to Amendment 1 (S8) showed that under zero interaction such changes are the only source of false-positive H3 support at the planned sample size.

### 3.6.4 Home-recipe analysis (secondary confirmatory; Amendment 1)

H2b and co-adaptation predict the same sign. If each architecture performs best under the recipe it was developed with, then moving from R1 to R2 helps EEGNet and hurts CSOANet-v2, which is exactly c[s] > 0. H2b alone therefore cannot separate "a gated compact model is robust to the recipe" from "every model prefers its own recipe". Two indices, fixed in Amendment 1, address this.

**Home advantage.** For architecture a with home recipe home(a), h_a is the interaction residual of the home cell after removing both main effects, computed from the cell means:

h_a = ȳ_{a,home} − ȳ_{a,·} − ȳ_{·,home} + ȳ_{·,·}.

A simple "home minus away" difference is not used because it contains the recipe main effect: if R1 were the best recipe for every architecture, CSOANet-v2's simple difference would be positive regardless of any preference for its own recipe, and if R2 were the strongest recipe overall, EEGNet's home advantage would be inflated. The pilot already showed this pattern (Section 3.7). The per-subject version h[s, a] applies the same formula to each subject's own 4 × 4 table; a two-sided one-sample *t*-test is run for EEGNet, MI-BMInet and CSOANet-v2, Holm-corrected over the three. For CSOANet-v2 an equivalence test is added (TOST; Schuirmann, 1987; bounds ±2 points, 90 % CI): the absence of a home advantage is claimed only if equivalence is shown, never from a non-significant test. The primary analysis uses the full 4 × 4 table; because R4 is near-home for CSOANet-v2, a sensitivity analysis recomputes all residuals on the 4 × 3 table without the R4 column.

**Recipe sensitivity.** For each architecture, the residual range (maximum minus minimum of its four interaction residuals) with a subject-level bootstrap 95 % CI (2,000 resamples). The raw range (best minus worst recipe mean) is reported but not used for decisions, because an architecture near ceiling or floor has a small raw range for reasons unrelated to recipe sensitivity. Residuals sum to zero over architectures within each recipe, so they are coupled: a small residual range means "closest to additive within this set of four", not "insensitive to recipes".

**Decision rules.** (i) CSOANet-v2's home residual is significantly positive, whatever the other architectures show: consistent with co-adaptation; a genuine fit between gating and heavy augmentation cannot be excluded, because the two are indistinguishable in the home cell; H2b is then not read as evidence of recipe robustness. ShallowConvNet, which has no home recipe in the set, serves as descriptive side-evidence: if its residual pattern resembles CSOANet-v2's, this leans toward genuine fit. (ii) CSOANet-v2's home residual is equivalent to zero and its residual range is the smallest of the four in at least 80 % of bootstrap resamples: a robustness interpretation is supported in the relative sense above; if EEGNet or MI-BMInet also show a significant home residual, this is the strongest form of support. (iii) Any other pattern: the mechanism behind H2b is reported as unresolved, with all effect sizes. The thresholds are deliberately asymmetric: one significant residual suffices for the reading unfavourable to the authors' own model, whereas the favourable reading requires an equivalence test and a ranking criterion. The ranking criterion is 80 % rather than 95 % because strict ranking of four coupled ranges at 95 % is reachable only for very large differences.

**Incomplete tables.** In the sensitivity analysis that excludes training failures, a subject whose 4 × 4 table loses a cell is dropped from the home-recipe analysis; if more than 10 % of subjects would be dropped, h_a is instead taken from the sum-coded interaction term of the mixed model fitted to all remaining cells.

### 3.6.5 Sensitivity analyses

None of the following decides a hypothesis; all are reported. Registered: all tests repeated on int8 accuracy; aligned-rank-transform ANOVA; F1a and F1b analysed separately. Added in Amendment 1: by-subject random slopes for architecture and recipe (uncorrelated slopes if the full model does not converge) and a subject-level cluster bootstrap (2,000 resamples) for the H1, H2 and H2b statistics; fold as a fixed effect; dataset × architecture, dataset × recipe and the three-way interaction, with the pooled F1 conclusion downgraded to "dataset-dependent" if F1a and F1b disagree on H2 or H3; seed as a random effect on unaveraged accuracies; exclusion of training failures, with any H3 category change that depends only on cells containing training failures reported as a recipe-robustness issue rather than an interaction; the same 24-interval classification on int8 accuracy with the number of float32 → int8 category changes.

### 3.6.6 Power

Power was estimated at registration from the four-class pilot H2b contrast (mean 3.95, SD 9.68): 0.96 for F1 under the full effect and 0.37 under half the effect. The pilot was a four-class task, so these are rough. Amendment 1 adds a simulation (S8) built from a binary pilot (PhysioNet-MI left vs right, 105 subjects, 4 × 4, one seed), in which each pairwise subject-level variance was split into a between-subject and a binomial component and recomposed at the confirmatory trial count. At n = 106 the median Bonferroni half-width is 1.78 points; under zero interaction the H3 rule has a false-positive rate of about 3 % (entirely from equivalent ↔ better changes, with no sign reversals), the stable outcome B1 has probability of at most 5 %, and the inconclusive outcome B2 has probability 0.61–0.77 when true pairwise differences are 1–2 points. The pre-specified rule for adding a third dataset (P(B2) > 0.5 averaged over true differences of 1–3 points) gave 0.496, 0.72 Monte Carlo standard errors below the threshold, so no dataset was added; Amendment 1 records this margin and the consequence that a stable ranking is close to unreachable in this sample.

### 3.6.7 Analyses added after the results (post hoc)

The following were added in revision V1.1, after an internal review of the first draft, and are labelled post hoc wherever they are reported (S4 row 14). None changes a registered decision.

- **Training-run clustering.** In the cross-subject families every test subject of a fold is scored by the same trained networks, so subject-level errors are correlated within a dataset × fold cluster (10 clusters in F1). H1 and H2 are refitted with crossed random effects for subject and training run (dataset × fold × architecture × recipe, seed-averaged). H2b, the home residuals and the 24 H3 intervals are re-estimated with a random intercept per cluster and t on 9 degrees of freedom, Bonferroni-adjusted as registered.
- **Without R3.** H1, H2, variance shares and home residuals on the 4 × 3 table without the R3 column, which holds every flagged run.
- **Difference-in-differences** for EEGNet − MI-BMInet, R1 against each other recipe.
- **Condition T against R4,** the best fixed recipe.
- **Search-space regression.** Validation accuracy of the 24 × 4 condition-T configurations on the fold-1 tuning split, regressed on the configuration dimensions with unit × architecture effects; in V1.2, with standard errors clustered by configuration and with the cropping effect estimated per architecture.

Added in revision V1.2 (S4 row 15):

- **Model choice for the clustered analyses.** For a single contrast between cells of the same cluster (a pairwise difference, H2b, a home residual), the random intercept per dataset × fold cluster is the contrast of the training-run effects. It is therefore the same model as the crossed subject + run model, restricted to that contrast. We use the crossed model for the likelihood-ratio tests and the cluster intercept for single contrasts.
- **Small-sample sensitivity.** With ten clusters, the critical value matters: cluster-based inference with few clusters is known to over-reject unless it is corrected (Cameron et al., 2008). The H3 intervals are therefore also reported with t(8) and a dataset fixed effect, with a normal critical value (design effect only), with a cluster bootstrap over the ten clusters (2,000 resamples), and leaving out each fold in turn (registered subject-level model).
- **A further difference-in-differences** for EEGNet − CSOANet-v2: H2b is its R2-versus-R1 case.
- **An exploratory run** of F1 under R1 without cropping ("R1nc"; 4 architectures × 3 seeds × 5 folds × 2 datasets = 120 runs; `train/explore_r1_nocrop.py`, which calls the frozen driver with one added recipe). The runs took place on 24 November 2026 (Lee2019-MI 12:00–17:16, Cho2017 20:12–23:45 UTC). They are analysed with `analysis/explore_r1nc_analysis.py`: R1nc against R1 and R4 per architecture, and the EEGNet − CSOANet-v2 gap, subject-level and clustered. Timing: the run driver was written on 24 November at 11:24 UTC, before any R1nc result existed. The analysis script was written at 20:11 UTC, after the Lee2019-MI half of the runs had been inspected and before the Cho2017 half had run. Both were committed with the results (6b1c447, 23:45 UTC). The cropping hypothesis came from the search-space regression, which used the training and validation subjects of fold 1. These subjects are test subjects in folds 2–5, so hypothesis and test overlap. We therefore also report R1nc − R1 on the 22 fold-1 test subjects, who were not used by the regression.

Added in revision V1.4 (S4 row 16): additivity among {R1nc, R2, R4}; the H2b contrast with R1nc in place of R1; the fold-1-only check; F2 cell means (S3).

## 3.7 Pilot work (disclosed; not evidence)

Two pilots preceded the registration and Amendment 1 and are disclosed in full in Supplementary S7. The registered four-class pilot (PhysioNet-MI; BCI IV 2a) motivated the hypotheses: changing only the recipe moved EEGNet by 6.6–8.7 points and CSOANet-v2 by 2.7 points (Supplementary S7.3). Under R2, EEGNet improved from 52.7 to 59.3 (PhysioNet-MI four-class, cross-subject), but CSOANet-v2 also improved (same cell 47.9 → 50.6; BCI IV 2a four-class within-subject 66.9 → 69.6), which is why home advantage is defined as an interaction residual rather than a simple difference. A binary pilot (PhysioNet-MI left vs right hand, 105 subjects, 4 × 4, 5-fold, one seed), run after all decision rules had been committed, provided the subject-level variability for S8 and an end-to-end test of the pipeline; it had no training failures. Its pattern differed from the four-class pilot: R1 was the lowest recipe for all four architectures, CSOANet-v2 the lowest architecture under every recipe, the H2b contrast +0.29 points, and all home-cell residuals within ±0.2 points of zero (Supplementary S7). We report it because it lowers our own prior for H2b and for any advantage of CSOANet-v2; it changed no rule. PhysioNet-MI is the dataset on which CSOANet-v2 and R1 were developed; the pilots were built up configuration by configuration, which is why the confirmatory study uses two datasets neither author had touched.

## 3.8 Preregistration, code freezing and deviations

The registration (1 November 2026) fixed the design, hypotheses, SESOI, families and corrections. Amendment 1, approved and public on 13 November 2026 (two days before the first download), recorded the frozen commit, the baseline verification, the home-recipe analysis, the H3 outcome categories and precedence rule, the condition-T disclosures and T′, the training-failure rule, the calibration sampling, the sensitivity analyses, the revised compute budget (150–200 GPU-hours) with an ordered list of cuts that never touch the runs deciding H1–H3, and the rule for adding a third dataset. All decision rules were committed to the repository in a dated file before the binary pilot was run.

The decision rules and analysis code were first frozen at commit f3fc1449 (8 November 2026), before the binary pilot. Confirmatory code, including the training components, is frozen at tag v1.0-frozen (commit a44e5933, 13 November 2026) of github.com/jabarrick/recipe-x-arch. The pilot code lives on a separate branch of the CSOANet2026 repository, which the new repository uses only as a dependency.

The frozen tag contained the analysis code and the training components, but not the driver that runs the confirmatory units. That driver was written afterwards, tested on synthetic data only, and frozen at tag v1.1-frozen (commit 1a7ba7b8, 15 November 2026). This was recorded on OSF as Amendment 2, approved and public at 16:37 UTC on 15 November, before any confirmatory data were downloaded.

The confirmatory datasets were downloaded after that point:

- Cho2017: 15 November 2026, 16:48–17:23 UTC.
- Lee2019-MI: 15 November 2026, 17:25–20:28 UTC.

These times are the file modification times. The data were first loaded with MOABB 1.5.0, which returns only one of the two Lee2019-MI sessions. Loading was therefore repeated under the registered MOABB 1.7.2, and every cache built under 1.5.0 was discarded before any model was trained. MOABB stores Cho2017 as epoched trials and, before filtering, demeans them and joins them with zero buffers into a continuous signal. Filter edge effects therefore occur at trial boundaries. The analysis window lies inside each stored trial, and the handling is identical for every architecture and recipe. It mainly shifts accuracies on Cho2017; an interaction with the time-shift and cropping augmentations of R1, through architectures with different temporal kernels, cannot be excluded.

Changes confined to data-loading and I/O code were permitted after the download without counting as deviations. Each one is a separate commit listed in Supplementary S4 with its hash and reason. Every other change is a deviation and is listed in S4 with what was registered, what was done, why, and whether it preceded the results.

The confirmatory runs took place from 18 to 24 November 2026 on one laptop GPU (NVIDIA RTX 4060, 8 GB):

- They took about 126 GPU-hours in total, within the revised budget of 150–200.
- No registered cut was needed.
- One interruption, caused by the run wrapper and not by the training code, cost no completed run.

S4 lists 15 entries. None changes a hypothesis, a decision rule, a threshold or a primary analysis. Three of them concern code written or fixed after the results began to come in:

- a crash in the training-failure sensitivity branch, fixed so that the incomplete-table rule (Section 3.6.4) is applied as written;
- a driver that assembles the H4 comparison from frozen functions;
- a script that computes registered sensitivity analyses and table entries that the frozen driver does not compute;
- the analyses added after an internal review of the first manuscript draft (Section 3.6.7).

All analyses not listed in Sections 3.6.3–3.6.5 are exploratory and are labelled as such in Section 4.8.

# 4 Results

## 4.1 Descriptive overview

After exclusions, F1 comprised 106 subjects (54 from Lee2019-MI, 52 from Cho2017) and 480 training runs (4 architectures × 4 recipes × 3 seeds × 5 folds × 2 datasets); no subject was excluded before training (Supplementary S2). Thirty runs were retried for non-finite loss, all of them ShallowConvNet under R3, and all 30 were again non-finite after the seed + 100 retry, so they were flagged as training failures; eleven further runs, all CSOANet-v2 under R3, were flagged because training balanced accuracy at the selected checkpoint was at or below 55 %. In total 41 of 480 runs (8.5 %) were flagged, every one of them in the R3 column. As registered, the primary analyses keep them.

The ShallowConvNet flag needs one sentence of context. In each of these runs the loss became non-finite after 12–34 epochs, the run stopped, and the checkpoint with the best validation score before that point was used; its test accuracy (cell mean 71.6 %) lies within the range of the architecture's other cells (69.8–74.4 %). The flag is therefore correct under the rule (non-finite loss after the retry) but does not mark a model that failed to learn. The CSOANet-v2 flags do mark such models: flagged runs had mean training accuracy 51.6 % and test accuracy 51.5 %.

Mean float32 balanced accuracy across the 16 cells was 72.5 % (cell means 59.3–76.9 %; Figure 1). CSOANet-v2's R3 cell largely failed. Eleven of its 30 runs were flagged (training accuracy ≤ 55 %), and eight more had training accuracy of 55–60 % with test accuracy of 50–61 % (Supplementary S3). The registered threshold therefore missed part of the failure: in Figure 1, many unflagged subjects of this cell also lie near chance.† Under R1, training accuracy at the selected checkpoint was 3–10 points below R4 for every architecture, although runs lasted about as long (Supplementary S3, cell diagnostics). Training accuracy is measured on un-augmented trials, so this is consistent with under-fitting and equally with the train/test mismatch introduced by cropping (Section 3.3). The exploratory runs without cropping resolve this in favour of the mismatch (4.8 g). Under R2, which keeps the lowest validation loss over all epochs, the selected checkpoint came early: median epoch 6.5 for ShallowConvNet and 20 for MI-BMInet.† Training accuracy at the selected checkpoint ranged from 50.0 to 99.8 % across runs and validation accuracy from 49.3 to 88.9 % (Table 5); no cell had a mean training accuracy below 60 %.

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

The recipe changed accuracy (H1: LRT χ²(3) = 160.5, Holm-adjusted p = 2.9 × 10⁻³⁴), and the size of the change depended on the architecture (H2: LRT χ²(9) = 414.7, Holm-adjusted p = 3.0 × 10⁻⁸³). These registered p-values treat subjects as independent. With crossed random effects for subject and training run (post hoc, 4.7), both remain supported (χ²(3) = 29.1, p = 2 × 10⁻⁶; χ²(9) = 95.3, p = 10⁻¹⁶).†

Of the total variance, 6.9 % was due to architecture, 1.7 % to recipe and 3.6 % to their interaction, against 74.8 % between subjects and 13.0 % residual (Figure 2). Among the fixed effects alone, the recipe accounted for 13.8 % (cluster bootstrap [11.3, 16.6]) and the interaction for 29.8 % ([25.3, 34.7]). Recipe ranges were 6.2, 4.5, 3.7 and 11.6 points for EEGNet, ShallowConvNet, MI-BMInet and CSOANet-v2 respectively (Table 5). For the three baselines the worst recipe was R1; for CSOANet-v2 it was R3. Without the R3 column (exploratory, 4.8), the ranges are 6.2, 4.5, 3.3 and 1.8 points, the interaction remains significant (χ²(6) = 59.0, p = 7 × 10⁻¹¹), and its share of the fixed-effect variance falls from 30 % to 10 %.† The dataset main effect was 4.9 points [0.8, 9.1] in favour of Lee2019-MI; dataset interactions are reported in 4.7.

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

The registered contrast was supported: moving from R1 to R2 changed EEGNet by +5.29 points and CSOANet-v2 by +1.79 points, giving c̄ = 3.50 points [2.29, 4.71] (t(105) = 5.73, Holm-adjusted p = 9.8 × 10⁻⁸).

The contrast was positive and significant, in the registered direction. Compared with the pilot estimate of +3.95 points, the confirmatory estimate is 0.45 points smaller. Per-subject contrasts were positive for 73.6 % of subjects. The distribution was similar across datasets: c̄ = 3.95 in Lee2019-MI and 3.03 in Cho2017, both significant on their own (Figure 3). Both architectures gained from R2. The contrast is positive because EEGNet gained more, not because CSOANet-v2 lost. With a random intercept per dataset × fold cluster (post hoc), c̄ = 3.53 [1.39, 5.67], p = 0.005.†

![**Figure 3.** Per-subject H2b contrast c[s] = (R2 − R1) for EEGNet minus (R2 − R1) for CSOANet-v2, by dataset and pooled, with mean and 95 % CI. Shaded band: ±2-point SESOI.](/home/claude/work/s4/figs/fig3_h2b_V0.2.png)

## 4.4 Home-recipe analysis (secondary confirmatory)

The home-cell interaction residuals were −0.28 [−0.71, 0.16], 2.60 [2.14, 3.06] and 3.55 [3.03, 4.07] points for EEGNet, MI-BMInet and CSOANet-v2. MI-BMInet's and CSOANet-v2's were significant after Holm correction (p = 1.9 × 10⁻¹⁹ and 3.6 × 10⁻²⁴); EEGNet's was not (p = 0.21). CSOANet-v2's residual was not equivalent to zero (TOST, 90 % CI [3.11, 3.99], outside ±2).

Residual ranges (Table 7) were 5.57, 2.35, 3.71 and 10.24 points. CSOANet-v2's was the smallest in 0 % of 2,000 bootstrap resamples, so the criterion was not met. ShallowConvNet's was the smallest in 99.0 %. Raw ranges ordered the architectures slightly differently: MI-BMInet had the smallest raw range (3.66) and ShallowConvNet the smallest residual range. Both measures put CSOANet-v2 last by a wide margin. Removing the R4 column changed CSOANet-v2's home residual to 3.81 points and did not change the decision. ShallowConvNet has no home recipe in the set, and its residual pattern was unlike CSOANet-v2's: small everywhere and of opposite sign in R3 (+1.09 against −6.69). By the preregistered mapping (Section 3.6.4), the mechanism outcome is **(i), consistent with co-adaptation**.

Three features of the residuals bear on how this decision can be read.†

- **The R3 cell dominates.** CSOANet-v2's residual pattern is dominated by its R3 cell (−6.69), the cell that contains its eleven flagged runs. Residuals in a row sum to zero, so a single depressed cell raises every other residual in that row, the home cell included. In a column they also sum to zero, so the same cell raises the R3 residuals of the other three architectures.
- **EEGNet's R3 residual exceeds MI-BMInet's home residual.** EEGNet's residual in R3, which is not its home recipe, was +3.00 (clustered 95 % CI [1.81, 4.14]). That is larger than MI-BMInet's home residual in the same column.
- **The raw means point the other way.** In the raw cell means, CSOANet-v2's best recipe was R2, not its home recipe R1. R1 was the worst recipe for each of the three baselines. The positive home residual therefore says that R1 penalised the baselines more than it penalised CSOANet-v2. It does not say that CSOANet-v2 did best under R1.

With a random intercept per dataset × fold cluster (post hoc), the home residuals were:

- CSOANet-v2: 3.56 [2.46, 4.66];
- MI-BMInet: 2.58 [1.39, 3.77];
- EEGNet: −0.28 [−1.06, 0.51].

In the training-failure sensitivity analysis (4.7), CSOANet-v2's home residual fell to 2.42 points. Without R3 (post hoc, 4.8), it fell to 1.32, inside the equivalence margin.

**Table 7.** Home-recipe analysis, F1, float32. Without R4 (registered sensitivity), the home residuals were EEGNet −0.32, MI-BMInet 2.23 and CSOANet-v2 3.81, and the mechanism outcome was still (i).

| Architecture | Home | h_a [95 % CI] | Holm p | TOST 90 % CI | Residual range [CI] | Raw range [CI] | Smallest residual range |
|---|---|---|---|---|---|---|---|
| EEGNet | R2 | −0.28 [−0.71, 0.16] | 0.21 | — | 5.57 [4.72, 6.48] | 6.21 [5.12, 7.42] | 0.0 % |
| ShallowConvNet | none | — | — | — | 2.35 [1.58, 3.23] | 4.54 [3.85, 5.21] | 99.0 % |
| MI-BMInet | R3 | 2.60 [2.14, 3.06] | 1.9 × 10⁻¹⁹ | — | 3.71 [3.07, 4.35] | 3.66 [2.80, 4.59] | 1.0 % |
| CSOANet-v2 | R1 | 3.55 [3.03, 4.07] | 3.6 × 10⁻²⁴ | [3.11, 3.99], not equivalent | 10.24 [8.92, 11.57] | 11.65 [9.76, 13.56] | 0.0 % |

## 4.5 Stability of architecture conclusions (H3)

The median adjusted half-width of the 24 pairwise intervals was 1.80 points, against the 2-point equivalence margin. The equivalent category was therefore reachable only for pairs whose observed difference was well below 1 point; 20 of 24 intervals were definitive.

H3 was supported. One pair changed definitive category across recipes, and the change was not a sign reversal: EEGNet versus MI-BMInet was *MI-BMInet better* under R1 (−2.62 [−4.13, −1.11]) and *equivalent* under R2, R3 and R4 (−0.18, −0.33 and +0.25 points, every interval within ±2). Of the 36 opportunities, there were 0 sign reversals, 3 equivalent ↔ better changes (all three are this pair, R1 against each other recipe) and 10 inconclusive ↔ definitive changes (Table 8).

All changes were between *equivalent* and *better*. By the preregistered reading, this indicates differences below the margin, not a change in ranking (Section 5.1). None of the cells involved contains a flagged run. The point estimate of EEGNet − MI-BMInet does change sign, from −0.33 under R3 to +0.25 under R4, but inside the equivalence margin.†

The change is not only a matter of interval classification.† The difference-in-differences of EEGNet − MI-BMInet, each other recipe against R1, was:

- R2: +2.44 [1.42, 3.47];
- R3: +2.30 [1.42, 3.17];
- R4: +2.87 [1.79, 3.94].

With a random intercept per dataset × fold cluster the intervals are [1.22, 3.66], [0.91, 3.70] and [0.87, 4.94] (Supplementary S9). R1 widened MI-BMInet's lead over EEGNet by 2.3–2.9 points, with clustered lower bounds of about 1 point.

The inconclusive ↔ definitive changes carry a pattern of their own.† Seven of the ten involve CSOANet-v2. Under R1 CSOANet-v2 could not be distinguished from EEGNet (+1.49) or ShallowConvNet (+0.70). Under R2, R3 and R4 EEGNet was clearly ahead of it, by 4.99, 17.30 and 6.70 points. Under R3 and R4 ShallowConvNet was clearly ahead too. R1 is the recipe developed with CSOANet-v2. The registered H3 definition, which counts only changes between definitive categories, does not register this pattern. The quantity that does is the registered H2b, a difference-in-differences for EEGNet − CSOANet-v2 (4.3). The same contrast with R4 in place of R2 was 5.20 [3.84, 6.57] (clustered [2.18, 8.35]).†

**Subjects sharing trained models (post hoc).**† All test subjects of a fold are scored by the same networks. With a random intercept per dataset × fold cluster (median intraclass correlation 0.15), the median adjusted half-width rose from 1.80 to 3.89 points. Only 13 of 24 intervals remained definitive, and no pair changed definitive category across recipes. Under this analysis H3 would not be supported (outcome: intermediate).

This result depends on how the ten clusters are treated (post hoc, V1.2–V1.4).† The variants amount to two treatments.

- **Design effect only.** A normal critical value: median half-width 2.80, 20 intervals definitive, outcome A′. A cluster bootstrap also uses a normal critical value and is the same treatment: 2.50, 20, A′.
- **Design effect with a small-sample correction.** t(9): 13 definitive, intermediate. t(8) with a dataset effect: 11 definitive, intermediate.

With ten clusters, the uncorrected treatment is expected to give intervals that are too narrow (Cameron et al., 2008), so the corrected one is the appropriate reference. Under it, H3 would not be supported.

Leaving out each fold in turn from the registered subject-level analysis gave A′ in all five cases. That shows the result does not rest on a single fold; it does not address the clustering. Under t(9), the 13 intervals that survived were (Supplementary S9):

- MI-BMInet ahead of CSOANet-v2 under every recipe;
- EEGNet ahead of CSOANet-v2 under R2–R4;
- EEGNet and MI-BMInet each ahead of ShallowConvNet under R2 and R3;
- ShallowConvNet ahead of CSOANet-v2 under R3;
- EEGNet and MI-BMInet equivalent under R4.

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

## 4.6 Shared recipe versus equal-budget search (H4)

Condition T carries two opposing biases: a cross-fold hyperparameter leak that favours it, and a winner's curse from selecting the best of 24 two-seed scores, which penalises it. The H4 result is therefore read together with the two sensitivity analyses. The fold-1-only evaluation (N = 22) showed no definitive category change, and the leakage-free T′ gave the same six categories as T.

Under R1 the architecture order was MI-BMInet > EEGNet > ShallowConvNet > CSOANet-v2, and under T it was the same (Kendall τ = 1.0, subject bootstrap 95 % CI [0.33, 1.0]; wide, as expected with four items). Four of the 6 pairwise categories changed between R1 and T (Table 10):

- EEGNet versus MI-BMInet went from *MI-BMInet better* to *equivalent* (−0.26 [−1.41, 0.90] under T). This is the only change between definitive categories, and the same change as in H3.
- The other three changes went from *inconclusive* to *better*: EEGNet, and ShallowConvNet, over CSOANet-v2 (+6.24 and +3.24), and EEGNet over ShallowConvNet (+3.00).

By the registered classification H4 is supported, but only through an equivalent ↔ better change; by the reading fixed for H3, this is not a change in ranking.

T improved on R1 by 3.4 points on average (from +0.99 for CSOANet-v2 to +5.74 for EEGNet). T′ improved by 4.3 points (+1.42 to +6.49), slightly more than T. R1, however, was the worst fixed recipe for three of the four architectures. Against the best fixed recipe, R4, T gained nothing (post hoc):†

- EEGNet −0.47 [−1.07, 0.13];
- ShallowConvNet −1.01 [−1.79, −0.23];
- MI-BMInet +0.03 [−0.75, 0.82];
- CSOANet-v2 −0.01 [−0.58, 0.55].

T′ was within about 1 point of R4 for every architecture (+0.28 to +0.95). The difference between each selected configuration's search score and its test accuracy was +1.0 to +7.0 points in Lee2019-MI and −6.1 to −10.4 points in Cho2017. The tuning validation set contained only the four validation subjects of fold 1 in each dataset, so these differences mostly reflect which subjects it held. They do not estimate the winner's curse, and we do not report a winner's-curse size. T did beat R1 here, but its gain over R1 is the gain of not using R1. At a budget of 24 configurations and a validation set of four subjects, it did not beat a well-chosen fixed recipe. The leakage-free T′, tuned on the pilot dataset, did at least as well as T, so the leak did not produce the gain.

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

## 4.7 Secondary family, int8 and sensitivity analyses

**F2 (within-subject, Lee2019-MI, N = 54).**

- H1: χ²(3) = 7.55, Holm-adjusted p = 0.11.
- H2: χ²(9) = 20.20, Holm-adjusted p = 0.0501.
- H2b: c̄ = 0.78 [−3.05, 4.62], Holm-adjusted p = 0.68.
- H3: 9 of 24 intervals definitive, median half-width 4.60, 0 sign reversals, 0 equivalent ↔ better changes; outcome intermediate (neither B1 nor B2).
- H4: no definitive change and identical order under R1, T and T′ (τ = 1.0).

None of these is supported. As registered, the negative results are reported as inconclusive (registered power 0.70 / 0.18). Median per-cell seed SDs were 3.79 points in F2 against 2.44 in F1 (Supplementary S3), consistent with model-selection noise on 20 validation trials per subject. 124 of 2,592 F2 runs (4.8 %) were flagged, mostly EEGNet under R1 (42) and MI-BMInet across recipes (55). The search scores of the F2 T configurations exceeded their session-2 test accuracies by 20–25 points. This mostly measures the session-1 → session-2 transfer gap and not over-selection.

**int8.** All tests repeated on int8 accuracy gave the same decisions for H1, H2 and H2b (F1: χ² = 201.3 and 384.1; H2b c̄ = 4.05 [2.90, 5.20]), and H3 remained A′. Float32–int8 prediction agreement was 95.5 % (EEGNet 96.1, ShallowConvNet 94.1, MI-BMInet 97.8, CSOANet-v2 94.2). One of the 24 pairwise categories changed from float32 to int8 (Table 8, second block): EEGNet versus MI-BMInet under R3, from *equivalent* to *MI-BMInet better*. This change comes from the EEGNet × R3 cell, which lost 3.21 points in int8 on average; that is the largest loss of any cell, and 4 of its 30 runs lost more than 5 points.

Operator coverage: EEGNet 5 of 9 compute operators in int8 (56 %; AveragePool and ELU stayed in float32); ShallowConvNet 6/6, MI-BMInet 7/7, CSOANet-v2 27/27 (100 %). Coverage differed, and the one category change involves the only partly quantised architecture under MinMax calibration, so part of the int8 difference may be a toolchain or calibration effect. A percentile-calibration control was not possible, because the exported models were not retained.† In F2, agreement was 98.1 %, decisions were unchanged, and one category changed: MI-BMInet - CSOANet-v2 under R3, from *CSOANet-v2 better* (−5.14) to *inconclusive* (−4.82).

**Sensitivity (all reported, none decisive).**

- *Training-run clustering (post hoc, Section 3.6.7):*†
  - Crossed subject + training-run model: H1 χ²(3) = 29.1, p = 2.1e-06; H2 χ²(9) = 95.3, p = 1.4e-16. Variance components: subject 117, training run 5.4, residual 15.0 points². In a pairwise difference the subject component cancels, so the relevant comparison is training run against residual.†
  - H2b: 3.53 [1.39, 5.67].
  - H3: 13 of 24 definitive, median half-width 3.89, no definitive ↔ definitive change; H3 would not be supported. With a normal critical value or a cluster bootstrap the outcome stays A′ (4.5).
  - Per-pair intraclass correlations are estimated from ten clusters and are unstable. The highest (0.5–0.6, CSOANet-v2 under R3) reflect failed runs concentrated in some folds.
- *Random slopes:* H1 χ²(3) = 88.6, H2 χ²(9) = 528.5; converged; decisions unchanged. In F2 the random-slopes model did not converge, and this is reported.
- *Cluster bootstrap:* intervals as in 4.2; H2b c̄ [2.10, 4.74].
- *Fold as fixed effect:* H1 and H2 unchanged (χ² = 160.5 and 414.7).
- *Seed as random effect, on the unaveraged runs:* H1 χ²(3) = 339.7, H2 χ²(9) = 829.4; decisions unchanged.
- *ART-ANOVA* (aligned rank transform, Wobbrock et al., 2011; LRT on ranks with a subject random intercept): H1 χ²(3) = 195.9, H2 χ²(9) = 435.0, both p < 10⁻⁴⁰. In F2 the ART gave H1 p = 0.025 and H2 p = 0.003, unlike the parametric F2 tests; we report this and do not read it as a decision.
- *Dataset interactions:*
  - Dataset × architecture: χ²(3) = 108.4, p = 2.5 × 10⁻²³.
  - Dataset × recipe: χ²(3) = 49.0, p = 1.3 × 10⁻¹⁰.
  - Three-way: χ²(9) = 62.7, p = 4.1 × 10⁻¹⁰.
- *F1a and F1b analysed separately:*
  - They agreed on H1, H2 and H2b, all supported in both: H2 χ² = 361.2 in Lee2019-MI and 121.5 in Cho2017; H2b c̄ = 3.95 and 3.03.
  - They disagreed on the H3 label. Lee2019-MI alone is B1 (18 definitive, no change of definitive category); Cho2017 alone is A′ (17 definitive, 2 equivalent ↔ better changes, no sign reversal).
  - Following the preregistered wording, the pooled H3 conclusion is reported as dataset-dependent. Neither dataset showed a sign reversal.
- *Excluding the 41 flagged runs:* every subject loses the ShallowConvNet × R3 cell, so the model path applies (Section 3.6.4). The home residuals were EEGNet −0.01, MI-BMInet 1.80, CSOANet-v2 2.42, and H3 stayed A′. No H3 category change depended on cells containing training failures.

**Table 11.** Static cost of the deployed graphs (`docs/static_cost_V1.0.csv`; read off the exported ONNX graph with the CSOANet2026 repository's profiler, BN folded). Theoretical quantities only; no latency or energy was measured. Costs do not depend on the recipe.

| Architecture | Dataset (input) | Params (train / deployed) | Flash int8 (kB) | MACs (M) | Peak int8 RAM (kB) | Streaming (kB) |
|---|---|---|---|---|---|---|
| EEGNet | Lee2019-MI (62 × 1000) | 3,818 / 3,762 | 3.8 | 63.18 | 544.9 | 500.0 |
| ShallowConvNet | Lee2019-MI | 105,202 / 105,162 | 102.9 | 157.38 | 2,424.3 | 2,401.9 |
| MI-BMInet | Lee2019-MI | 6,722 / 6,562 | 6.5 | 1.99 | 91.9 | 31.3 |
| CSOANet-v2 | Lee2019-MI | 3,800 / 3,634 | 3.6 | 2.08 | 76.3 | 23.5 |
| EEGNet | Cho2017 (64 × 750) | 3,594 / 3,538 | 3.5 | 48.91 | 421.9 | 386.7 |
| ShallowConvNet | Cho2017 | 107,042 / 107,002 | 104.7 | 120.84 | 1,861.9 | 1,843.4 |
| MI-BMInet | Cho2017 | 6,530 / 6,370 | 6.3 | 1.54 | 70.4 | 23.5 |
| CSOANet-v2 | Cho2017 | 3,704 / 3,538 | 3.5 | 1.57 | 58.7 | 17.7 |

*Peak RAM is the largest set of activations alive at once in graph execution order at 1 byte per value. Streaming excludes the raw input window, which applies when the first layer consumes samples as they arrive.*

## 4.8 Exploratory analyses (not confirmatory)

These analyses are labelled exploratory, as registered. Items (d) to (f) were added after the results were seen.

(a) *Augmentation components.* The per-component ablation from R1 toward R4 needs training runs beyond the registered set and was not run. The registered package contrast R4 − R1 (R1 without its augmentation) was positive for all four architectures: EEGNet +6.21 [5.03, 7.39], ShallowConvNet +4.54 [3.83, 5.25], MI-BMInet +3.34 [2.50, 4.18], CSOANet-v2 +1.00 [0.18, 1.83]. Removing the augmentation package helped every architecture, and it helped CSOANet-v2 least.

(b) *Training diagnostics per cell* are given in Supplementary S3.

(c) *Deployment frontier.* For each dataset and recipe, we found the architectures that no other architecture beats on both accuracy and cost.

*Accuracy against MACs.*

- MI-BMInet was on the front in every one of the 8 dataset × recipe panels, in float32 and in int8.
- EEGNet joined it in 3 float32 panels (Lee2019-MI under R2 and R4, Cho2017 under R3), by margins of 0.0–0.6 points. Despite its much higher cost, it reached the front only by matching MI-BMInet's accuracy.
- CSOANet-v2 was never on the front. MI-BMInet has 2–5 % fewer MACs and was more accurate in every panel.

*Accuracy against peak int8 RAM.* CSOANet-v2 is on the front in every panel, because it has the smallest memory footprint: about 17 % below MI-BMInet, and 25 % below it in streaming mode. MI-BMInet is also on that front in every panel. EEGNet joins it in the same 3 panels as above.

The front therefore moved with the recipe only through EEGNet's near-ties with MI-BMInet. Which architecture is cheapest is a property of the graph, so it does not move. Whether the cheapest one is also accurate enough depends on the cost measure: MACs or memory.

(d) *Home residuals without R3 (post hoc).* R3 holds every flagged run, and CSOANet-v2's R3 cell drives its residual pattern (4.4). We therefore recomputed the home residuals on the 4 × 3 table without R3; MI-BMInet's home cell is R3 and drops out. CSOANet-v2's home residual was +1.32 [0.91, 1.73], significantly positive but also equivalent to zero (90 % CI [0.98, 1.66], within ±2). EEGNet's was +0.72 [0.36, 1.08], also within ±2. This does not change the registered decision (i). It indicates that about two thirds of CSOANet-v2's registered home residual is the mechanical consequence of its R3 collapse, and that the remaining home preference is smaller than the SESOI. On the same 4 × 3 table:

- H1 and H2 remained supported (χ²(2) = 183.4; χ²(6) = 59.0, p = 7e-11; with crossed subject and run effects, χ²(6) = 21.0, p = 0.002).
- The fixed-effect variance shares were architecture 56 %, recipe 34 % and interaction 10 %.

(e) *What in the search space mattered.* We regressed the validation accuracy of the 24 configurations × 4 architectures on the fold-1 tuning split of both datasets (n = 384, 24 distinct configurations) on the configuration dimensions, with unit × architecture effects. Standard errors are clustered by configuration (V1.2), and the validation set holds only about four subjects per unit, so the estimates are indicative.†

Three dimensions mattered:

- Random cropping lowered validation accuracy by 5.9 points [5.2, 6.6].
- Selecting the lowest validation loss over all epochs lowered it by 1.8 points [1.2, 2.4], relative to early stopping on validation accuracy.
- Batch 16 was about 1.6 points worse than batches 32 and 64.

Optimiser, learning rate, schedule, label smoothing and the other augmentations had no clear effect.

The cropping effect differed by architecture:†

- EEGNet: 9.5 points [8.8, 10.2];
- MI-BMInet: 6.8 [6.2, 7.5];
- ShallowConvNet: 5.8 [5.0, 6.6];
- CSOANet-v2: 1.4 [−0.5, 3.4].

Random cropping is part of R1, and CSOANet-v2 was developed under it (Section 5.3).

(f) *Condition T against R4* is reported in 4.6.

(g) *R1 without cropping (exploratory, run after V1.2; hypothesis from (e)).*† R1nc is R1 with cropping switched off and everything else unchanged. Removing cropping alone raised accuracy for every architecture (R1nc − R1; subject-level 95 % CI, then clustered):

- EEGNet: +5.86 [4.83, 6.89], clustered [3.66, 8.05];
- ShallowConvNet: +3.98 [3.24, 4.72], clustered [2.57, 5.35];
- MI-BMInet: +3.74 [2.97, 4.51], clustered [2.02, 5.43];
- CSOANet-v2: +1.07 [0.33, 1.81], clustered [−0.13, 2.23].

R1nc then matched R4 for every architecture (R4 − R1nc between −0.40 and +0.56, every interval including zero). The other four augmentations of R1 therefore had no net effect, and essentially the whole R1 − R4 deficit is attributable to cropping. Training accuracy at the selected checkpoint also rose to the R4 level (for example EEGNet 74.7 → 81.1 against 82.9 under R4), so the deficit was the train/test mismatch, not under-fitting. No run was flagged.

The gap between EEGNet and CSOANet-v2 was 1.49 points under R1, 6.28 under R1nc and 6.70 under R4. Removing cropping widened it by 4.78 [3.66, 5.90], clustered [2.66, 6.97]; the remaining R4 − R1nc difference was +0.42 [−0.63, 1.48]. The effect of cropping was larger in Lee2019-MI than in Cho2017 (EEGNet +7.9 against +3.7; CSOANet-v2 +2.1 against +0.0). Two explanations are possible, and neither was tested. First, Lee2019-MI's longer window (4 s against 3 s) means that the same 10 % crop removes more signal. Second, it shifts the content by up to 0.4 s rather than 0.3 s, misaligning it further from the cue.

*Out-of-sample check.* On the 22 fold-1 test subjects, who were not used by the regression that suggested the hypothesis, R1nc − R1 was:

- EEGNet: +3.52 [1.78, 5.25];
- ShallowConvNet: +4.63 [2.84, 6.42];
- MI-BMInet: +2.40 [0.82, 3.98];
- CSOANet-v2: +1.33 [−0.39, 3.04].

The change in the EEGNet − CSOANet-v2 gap was +2.19 [0.12, 4.26].

The direction is the same as in the full sample, but the effects are roughly halved (EEGNet 3.5 against 5.9; gap change 2.2 against 4.8), as expected when the subjects that generated a hypothesis are excluded from its test. These 22 subjects come from only two training clusters, so their intervals cannot include training-run variance and are too narrow.†

*Replication on the Cho2017 half.*† The analysis script was written after the Lee2019-MI half of the R1nc runs had been inspected and before the Cho2017 half had run. The Cho2017 half is therefore a replication with a fixed analysis (52 subjects; clustered intervals on 5 clusters):

- R1nc − R1: EEGNet +3.71 [2.20, 5.23], ShallowConvNet +2.35 [1.09, 3.61], MI-BMInet +2.09 [−0.25, 4.44], CSOANet-v2 +0.01 [−1.08, 1.09].
- Change in the EEGNet − CSOANet-v2 gap: +3.71 [1.93, 5.48].

It still shares the fold-1 subjects with the search-space regression. A confirmatory test needs an independent dataset (Section 5.8).

(h) *Additivity once cropping and R3 are removed (post hoc, V1.4).*† On the 4 × 3 table {R1nc, R2, R4} (Table 12):

- every interaction residual was within ±1 point, the largest being ShallowConvNet under R2 (−0.99);
- all 12 residuals were equivalent to zero by TOST against the ±2-point SESOI, with a random intercept per dataset × fold cluster (90 % CIs within [−1.48, 1.37]; Supplementary S9.10);
- the recipe ranges were EEGNet 0.92, ShallowConvNet 2.51, MI-BMInet 0.89 and CSOANet-v2 0.79 points. ShallowConvNet's range comes from its R2 cell, whose checkpoint is selected around epoch 6 (4.1).

The interaction test was significant with subjects as the unit (χ²(6) = 23.7, p = 0.0006) but not with crossed subject and training-run effects (χ²(6) = 4.6, p = 0.60). With R1nc in place of R1, the H2b contrast was −1.29 [−2.29, −0.29] (clustered [−2.57, 0.00]). The point estimate has the opposite sign to the registered +3.50; the clustered interval only just reaches zero, so the contrast no longer favours the authors' model.

Two cautions apply. R1nc and R4 differ only by four light augmentations and gave nearly identical results. The table therefore compares two recipe families, R2 against R1nc/R4, not three independent recipes. And the two columns that carried the interaction were removed before additivity was assessed, so the result is partly a consequence of that choice.

**Table 12.** Cell means (%) and interaction residuals (points), F1, recipes R1nc, R2 and R4 (post hoc).

| | R1nc | R2 | R4 | Residual R1nc | Residual R2 | Residual R4 |
|---|---|---|---|---|---|---|
| EEGNet | 76.5 | 75.9 | 76.8 | −0.12 | −0.01 | +0.13 |
| ShallowConvNet | 73.8 | 71.9 | 74.4 | +0.28 | −1.00 | +0.72 |
| MI-BMInet | 77.0 | 76.1 | 76.6 | +0.24 | +0.02 | −0.27 |
| CSOANet-v2 | 70.2 | 70.9 | 70.1 | −0.40 | +0.99 | −0.58 |

# 5 Discussion

*This section was drafted before data access as alternative modules with numerical placeholders, deposited with the preregistration materials. The modules used here were selected by the preregistered rules (outcome A′, mechanism i, deployment D2) and their placeholders filled afterwards. After two rounds of internal review (manuscript V1.0 and V1.1), several readings were revised. Sentences and paragraphs that were added or changed after the results were known are marked †.*

## 5.0 Overview

This study asked whether a shared training recipe is a neutral instrument for comparing compact motor-imagery decoders, or whether it chooses the winner. The answer has two audiences:

- **People who compare models.** It decides whether "same recipe for all" is a defensible protocol, or whether a comparison must report several recipes or an equal-budget search.
- **People who choose a model for deployment.** It decides whether a published ranking obtained under one recipe can be trusted to hold under the recipe they will actually use, and whether it survives int8 quantisation.

We take the two in turn, then the mechanism, then the limits of what this design can say.

## 5.1 Was the shared recipe neutral?

As registered, no: the authors' recipe narrowed their model's gap to EEGNet. Exploratory analyses traced this to one implementation defect in that recipe; among the other recipe families tested, recipe and architecture were close to additive. The evaluation protocol mattered far more than the recipe: within subjects, the order of the architectures was largely reversed.†

The registered evidence is the directional contrast itself.† H2b is a difference-in-differences: how much the gap between EEGNet and CSOANet-v2 changes when the recipe changes from R1 to R2. Under R1, the authors' heavy-augmentation recipe, EEGNet led CSOANet-v2 by 1.5 points. Under the other three recipes EEGNet led by 5.0 (R2), 17.3 (R3) and 6.7 points (R4). R1 therefore narrowed the gap by 3.50 points [2.29, 4.71] relative to R2, which is the registered H2b. Relative to R4, which is R1 without its augmentation and the comparator we chose post hoc, it narrowed the gap by 5.20 points [3.84, 6.57]; relative to R3 the figure is 15.8 points, dominated by CSOANet-v2's collapse under R3. Both remain clear of zero when subjects sharing trained networks are accounted for: [1.39, 5.67] and [2.18, 8.35].† R1 was also the worst of the four recipes for each of the three baselines. A comparison run under the shared recipe chosen by the proposers of one model would thus have shown that model at its closest to the others.†

**What made the recipe non-neutral (exploratory).**† The non-neutrality has an identifiable source. One augmentation in R1, a random crop applied in training but not at test, cost the baselines 3.7–5.9 points and CSOANet-v2 1.1 (4.8 g). Removing it brought every architecture to its R4 level. With cropping removed and the numerically unstable R3 set aside, recipe and architecture were close to additive. Among the two remaining recipe families (R2, and R1nc/R4), every interaction residual was within ±1 point and equivalent to zero by TOST (Table 12). With R1nc in place of R1, the H2b contrast no longer favoured the authors' model (−1.29; clustered interval [−2.57, 0.00]).

The R1 − R4 contrast must therefore not be read as an effect of heavy augmentation; it is an effect of one implementation. Among the recipes that remain, the main non-additivity is ShallowConvNet under R2, which comes from R2's checkpoint rule.

The useful conclusion is narrower than "a shared recipe is not neutral". Among the two recipe families that remained, the shared recipe was close to neutral. It was not neutral when it contained an unchecked implementation detail. In this case the detail sat in the recipe the authors' model had been developed with. We put forward as a hypothesis, not a finding, that this is where such details are least likely to be noticed. These analyses are post hoc and on F1 only.†

Whether the registered result changed *architecture conclusions* depends on what kind of conclusion one means. We separate three, and add a comparison across evaluation protocols that the registration treated as secondary.†

**Order.** By mean accuracy, the order of the four architectures was the same under all four recipes, with one exception: under R4 the point estimates of EEGNet and MI-BMInet swapped, by 0.25 points, inside the equivalence margin. No pair reversed sign at the registered interval level, and condition T and T′ gave the same order as R1. In F1, MI-BMInet was never behind ShallowConvNet, and CSOANet-v2 was never ahead of any other architecture.†

**Order across evaluation protocols (descriptive).**† The largest change in architecture order came not from the recipe but from the evaluation protocol. In the within-subject family F2 (Lee2019-MI, session 1 → session 2; N = 54), the order was largely reversed (Table 13). CSOANet-v2 ranked first under R1, R3 and R4 and second under R2. EEGNet ranked last under R1, R2 and R4.

At the interval level (subject-level; in F2 every subject has its own trained networks, so subjects are the training-run clusters):

- CSOANet-v2 was clearly ahead of EEGNet under R1 (−7.52 [−12.41, −2.63]) and R4 (−7.48 [−13.76, −1.19]); under R2 and R3 the upper bounds touched zero.
- CSOANet-v2 was ahead of MI-BMInet under R1 and R3, and of ShallowConvNet under R3.
- MI-BMInet was ahead of EEGNet under R1, R2 and R4.

Two qualifications apply.

- EEGNet's position under R1 is partly training failure: 42 of its 162 F2 runs under R1 were flagged. Excluding them raises its cell mean from 57.7 to 60.2, still last.
- CSOANet-v2 was almost flat across recipes in F2 (65.2–66.1). That fits the reading of general insensitivity to recipe changes (5.3).

No F2 hypothesis was supported, so this is description, not a test. But a change of evaluation protocol reordered these architectures far more than any recipe did. This bears on the original CSOANet paper, whose main comparison was within subjects (5.6).

**Table 13.** F2 (within-subject) cell means (%, float32) and rank of each architecture under R1, R2, R3 and R4. In F1 the order under R1–R3 was MI-BMInet, EEGNet, ShallowConvNet, CSOANet-v2.

| Architecture | R1 | R2 | R3 | R4 | Rank (R1, R2, R3, R4) |
|---|---|---|---|---|---|
| EEGNet | 57.7 | 58.6 | 59.7 | 58.0 | 4, 4, 3, 4 |
| ShallowConvNet | 61.3 | 62.1 | 58.3 | 61.4 | 3, 3, 4, 3 |
| MI-BMInet | 61.8 | 66.2 | 61.0 | 63.4 | 2, 1, 2, 2 |
| CSOANet-v2 | 65.2 | 65.4 | 66.1 | 65.4 | 1, 2, 1, 1 |

**Margin-level calls.** H3 was supported as registered, and every conclusion change was between *equivalent* and *better*. EEGNet and MI-BMInet were *equivalent* under R2, R3 and R4. Under R1, MI-BMInet was better by 2.62 points. The difference-in-differences between R1 and each other recipe was 2.3–2.9 points. Its clustered lower bounds were about 1 point (0.87–1.22), so R1 widened this gap by an amount whose interval excludes zero but not values below the margin (4.5).†

The simulation attached to Amendment 1 showed that equivalent ↔ better changes can arise by sampling variation alone when the true interaction is zero. It cannot show that this particular change is noise, because H2 rejected a zero interaction.†

**Distinguishability.** Under R1, CSOANet-v2 could not be distinguished from EEGNet or ShallowConvNet; under the other recipes EEGNet was clearly ahead of it. "Could not be distinguished" is not "competitive":

- MI-BMInet remained clearly ahead of CSOANet-v2 under R1;
- the clustered interval for EEGNet − CSOANet-v2 under R1 does not exclude a 6.8-point deficit.

We therefore rest the claim on the difference-in-differences above, not on the categories.†

**How fragile the H3 result is.**† The registered intervals treat the 106 subjects as independent, but all test subjects of a fold are scored by the same trained networks. Whether the single registered conclusion change survives depends on how the ten dataset × fold clusters are treated:

- It survives the design effect alone (normal critical value or cluster bootstrap: 20 of 24 definitive, outcome A′).
- It is lost when a small-sample correction is added, which is the appropriate reference with ten clusters (Cameron et al., 2008): t(9) gives 13 of 24 definitive and t(8) gives 11 of 24, outcome intermediate.
- Leaving out single folds of the registered analysis does not change it, but that check does not address clustering.

H1, H2 and H2b remain supported in every variant. We report the registered decision and these variants side by side, and read the margin-level change as fragile.†

For practice the result is reassuring in one respect and cautionary in two. The order of these architectures did not depend on the recipe. But the size of the gaps did, and the recipe chosen by the proposers of one model was the one under which that model was closest to the others. Claims of equivalence or superiority near the margin should not rest on one shared recipe. A shared recipe developed with one of the compared models should be disclosed as such, and compared with at least one recipe that was not.

## 5.2 What the directional contrast does and does not test

H2b was supported: c̄ = 3.50 points [2.29, 4.71]; 3.53 [1.39, 5.67] when subjects sharing trained networks are accounted for.† Two cautions apply.

First, R1 and R2 differ in optimiser, weight decay, schedule, model-selection rule and the whole augmentation package. H2b therefore tests sensitivity to the R1 recipe as a package, not to augmentation. The augmentation-only contrast is R1 versus R4, which carried no registered direction and is reported as exploratory in Section 4.8. There, removing the R1 augmentation package improved every architecture, by 1.0 (CSOANet-v2) to 6.2 points (EEGNet).†

Second, the contrast shrank from the pilot's +3.95 points to +3.50. This is the expected direction when a pilot is built up on the data the model was developed on, and we report the shrinkage rather than the pilot. Both architectures gained under R2 (+5.29 and +1.79).

## 5.3 Robustness or co-adaptation?

H2b and co-adaptation predict the same sign. CSOANet-v2 was developed under R1 and EEGNet under R2. "Each model prefers the recipe it was born with" therefore predicts that moving from R1 to R2 helps EEGNet and hurts CSOANet-v2, which is exactly a positive H2b. A positive H2b cannot by itself distinguish "a gated compact model is robust to the recipe" from the plainer story that involves our own model and our own recipe. The home-recipe analysis was preregistered to separate them, with thresholds set deliberately against our own model.

CSOANet-v2's interaction residual in its home cell was 3.55 points [3.03, 4.07] and significant. By the preregistered rule (i), H2b is therefore not read as evidence that CSOANet-v2 is robust to the training recipe, and the robustness narrative is withdrawn.

What the residual measures needs to be stated more carefully than the module did.† A double-centred residual cannot tell "architecture *a* does well under recipe *r*" from "recipe *r* does badly for the other architectures". The raw cell means favour the second reading:

- R1 was the worst recipe for each of the three baselines.
- R4 was at least as good as R1 for every architecture.
- CSOANet-v2's own best recipe was R2 (70.9 %), not R1 (69.1 %).

The size of the residual is also largely a property of the R3 column.† CSOANet-v2 collapsed under R3 (cell mean 59.3 %; 11 of 30 runs flagged and 8 more with training accuracy of 55–60 %). Residuals in a row sum to zero, so this low cell raises CSOANet-v2's other residuals, its home cell included. The same cell lowers the R3 column mean, which raises the R3 residuals of the other architectures. EEGNet's residual in R3, not its home recipe, was +3.00, larger than MI-BMInet's home residual there (+2.60). Without R3 (post hoc, 4.8), CSOANet-v2's home residual was 1.32 [0.91, 1.73], positive but inside the 2-point margin.

The home-recipe analysis therefore did what it was registered to do: it ruled out the favourable robustness reading. It could not, in this design, identify co-adaptation as such.† That is a property of double-centred residuals in a 4 × 4 table with one unstable recipe, and we report it as a limitation of the registered method.

The search-space data pointed to a concrete candidate mechanism, identified post hoc.† In the condition-T search, random cropping was the only configuration dimension with a large effect, and its effect differed by architecture. With standard errors clustered by configuration, cropping lowered validation accuracy by:

- EEGNet: 9.5 points [8.8, 10.2];
- MI-BMInet: 6.8 points [6.2, 7.5];
- ShallowConvNet: 5.8 points [5.0, 6.6];
- CSOANet-v2: 1.4 points [−0.5, 3.4].

In our implementation (shared with the pilot code), cropping keeps 90 % of the window at one random offset per batch, moves it to the start and zero-fills the end. Test windows are never cropped, so training and test inputs differ systematically. CSOANet-v2 was developed under exactly this augmentation, and it is the architecture least affected by it.

An exploratory rerun of F1 without cropping was consistent with this (4.8 g). The effect was smaller, about half, on the fold-1 test subjects not used to form the hypothesis, and it replicated in the Cho2017 half of the runs, which was analysed with a fixed script.† Removing cropping alone raised the baselines by 3.7–5.9 points and CSOANet-v2 by 1.1. It brought every architecture to its R4 level. It widened EEGNet's lead over CSOANet-v2 by 4.8 points (clustered [2.66, 6.97]), which is 4.8 of the 5.2-point narrowing that R1 produced relative to R4.

CSOANet-v2 was the least affected by this augmentation. Two readings fit, and they cannot be separated here. The architecture may have been shaped during development under this augmentation. Or it may simply be insensitive to recipe changes in general: its R4 − R1 difference (1.0 point), its gain from search (1.0) and its recipe range without R3 (1.8) were all the smallest of the four. The second reading is the same caution we applied to the double-centred residuals above. Architectural explanations, for example its gating or its spatial projection before temporal filtering, are speculation and were not tested.†

What remains is a finding about shared-recipe comparisons, and we state it on ourselves. The recipe we chose as "shared" narrowed our model's gap to EEGNet by 3.5 points relative to R2 (registered), and by 5.2 relative to R4 (5.1). Whatever the mechanism, a comparison run under our recipe alone would have flattered our model.†

## 5.4 Deployment

Recipe gains are paid at training time and are free at inference. For a deployed model the recipe is therefore the cheapest lever available: it changes nothing on the device. The deployment question is whether the architecture conclusions drawn in float32 survive int8 post-training quantisation, and whether that depends on the recipe.

Quantisation changed 1 of the 24 pairwise categories, with float32–int8 agreement of 95.5 %. The change is concentrated in one cell: EEGNet under R3, the recipe with the highest learning rate. EEGNet lost 3.21 points there on average, and 4 of 30 runs lost more than 5 points.

Operator coverage differed across architectures. ShallowConvNet, MI-BMInet and CSOANet-v2 were fully quantised. EEGNet was quantised in 5 of 9 compute operators: its average pooling and ELU stayed in float32. The one change therefore coincides with partial quantisation, with MinMax calibration (the simplest default, and sensitive to outliers), and with the most aggressive recipe.

We read this narrowly.† The toolchain and calibration choices, together with the recipe, can change a pairwise conclusion after quantisation. Rankings intended for deployment should therefore be checked on the quantised graphs under the recipe that will be used, with operator coverage and the calibration method reported. A percentile calibrator might remove this change. No trained models were retained, because the frozen driver deletes each exported file after scoring. The check therefore needs retraining with the frozen code and is left to the deployment study.†

## 5.5 Equal-budget search (H4)

*This paragraph is new; no module was pre-written for H4.*†

Giving each architecture its own 24-configuration search (condition T) raised accuracy over R1 by 1.0 to 5.7 points and left the architecture order unchanged (Kendall τ = 1.0). Against the best fixed recipe, R4, T gained nothing: the differences ranged from −1.0 (ShallowConvNet) to +0.03 points (MI-BMInet). Its gain over R1 is therefore the gain of not using R1. This fits the cropping result. The best fixed recipe, R4, is in effect R1 without its defect (R1nc ≈ R4), and 11 of the 24 search configurations had cropping switched on. The search therefore spent almost half its budget on the defect it would have needed to avoid.†

The leakage-free T′, tuned on a different dataset, did at least as well as T. In Cho2017 the selected configurations' search scores were 6 to 10 points below their test accuracy. Both observations indicate that a search scored on about four validation subjects mostly selects noise. The one strong systematic signal in the search data is the cropping effect discussed in 5.3. Under R1, training accuracy on the un-augmented training trials was 3–10 points below R4, although the runs lasted about as long (median selected epoch 18–39 against 17–27). Without cropping, training accuracy rose to the R4 level (4.8 g), so the deficit was the train/test mismatch introduced by cropping, not under-fitting.†

An equal budget of 24 configurations is thus not a better notion of fairness than a well-chosen fixed recipe in this setting. What matters is that the fixed recipe is not one that penalises some architectures more than others.

## 5.6 Conflict of interest and self-restraint

Both CSOANet-v2 and R1 are ours. We addressed this in four ways:

- R2 and R3 are recipes published by others.
- The direction of H2b and every threshold were registered before the data were touched.
- Condition T gave each architecture the same search budget.
- The home-recipe analysis was built so that one significant residual suffices for the reading unfavourable to our model. The favourable reading requires an equivalence test and a ranking criterion (Section 3.6.4).

The analysis returned the unfavourable reading, and the revision in 5.3 shows that the more accurate statement is still unfavourable to our choice of shared recipe.† Readers should also weigh that the pilot data on which CSOANet-v2 and R1 were developed are the data on which condition T′ was tuned.

**The original CSOANet paper.**† The cropping implementation analysed here comes from the code published with CSOANet (Gao and Diniz, 2026), a perception-versus-imagination study on OpenNeuro ds005697. In that study CSOANet was trained with the full augmentation set, cropping included. The deep-learning baselines (EEGNet, ShallowConvNet, DeepConvNet) were trained without augmentation in the within-subject comparison, and without cropping in the cross-subject comparison. The cropping defect therefore did not handicap the baselines there. If our motor-imagery results transfer, it would if anything have cost CSOANet slightly.

That comparison did, however, give the proposed model a different and heavier recipe than the baselines. That is the confound this study was designed to examine. The original study's dataset also has limitations of its own that have since been confirmed by its providers: undocumented changes of paradigm, conditions that were not randomised, and no eyes-open/eyes-closed control (data providers, personal communication, 2026). These limit the scope of that study's conclusions independently of the recipe question. We have added a note to the CSOANet2026 repository describing the cropping implementation, its train/test mismatch, and this study's findings about it.

## 5.7 Limitations

**Scope.** The study covers two-class motor imagery, two datasets, and four compact spatio-temporal convolutional architectures. Nothing here transfers to Transformer or pretrained decoders, and we have not tried to make it. The four recipes are a sample of a continuous space, chosen to span plain to heavy; a different sample could give a different interaction.

**Statistical unit.**† The registered intervals treat subjects as independent, although subjects in the same fold share trained networks. In a pairwise difference the subject variance cancels. What matters there is the training-run component (5.4 points² per cell) against the residual (15 points²), and unlike the residual, the run component does not average out over the roughly ten subjects of a fold. With only ten clusters, the result also depends on the small-sample correction (5.1). Future designs of this kind should include the training run in the registered model and plan the sample size on folds and seeds as well as subjects.

**R3.**† One recipe was numerically unstable for two of the four architectures. Every ShallowConvNet run under R3 reached a non-finite loss after 12–34 epochs and was represented by its last checkpoint before divergence, which is effectively a different recipe. Eleven of 30 CSOANet-v2 runs were flagged by the training-accuracy rule (≤ 55 %), and eight more had training accuracy of 55–60 % with test accuracy of 50–61 %. Most of the cell therefore failed, and the registered threshold missed part of it. Without R3, the interaction remains significant (χ²(6) = 59.0) but its share of the fixed-effect variance falls from 30 % to 10 %. CSOANet-v2's recipe range falls from 11.6 to 1.8 points. Much of the interaction in F1 is therefore a statement about numerical stability under a high learning rate rather than about recipe preference.

**Model selection.**† The registered recipes differ in their checkpoint rule, and the rule matters. Under R2, which keeps the lowest validation loss over all epochs, ShallowConvNet's selected epoch had a median of 6.5 and MI-BMInet's of 20. In the search data, this rule cost 1.8 points [1.2, 2.4] relative to early stopping on validation accuracy. Part of the differences between R2/R3 and R4 is therefore a selection-rule effect.

**The architecture factor includes our implementation choices.**† ShallowConvNet was implemented as in braindecode, without the max-norm constraints of MOABB's Keras version. Those constraints account for an 8-point difference on BCI IV 2a (S7) and may contribute to its divergence under R3. MI-BMInet was used in its N_k = 32, N_f = 64 configuration, whereas its home recipe (R3) was published with N_k = 16, N_f = 128. Statements such as "ShallowConvNet trailed MI-BMInet under every recipe" are statements about these implementations.

**Cropping.**† R1 was registered as it is, including its cropping implementation. The rerun without cropping is exploratory and post hoc, it is on F1 only, and its hypothesis came partly from the same subjects. A test on an independent dataset would be needed to confirm it.

**Condition T.** T had a budget of 24 trials and a disclosed hyperparameter leak that favours it, and its validation set was about four subjects. "T did not beat R4" therefore means "not at this budget and this validation size".

**Seeds.** Three seeds per cell is few. Seed variance in these models is of the same order as architecture differences (Bouthillier et al., 2021; Picard, 2021). Supplementary S3 reports every cell's seed SD against the 2-point margin: in F1, the median subject-level seed SD exceeded it in 13 of 16 cells.

**Power.** The power analysis came from a four-class pilot. The binary pilot and S8 corrected the interval-width estimate, but S8 also assumed independent subjects (see *Statistical unit*).

**F2.** F2 was under-powered by design and, we think, mainly because of model-selection noise: selecting a checkpoint on 20 validation trials per subject is noisier than the subject count alone suggests. S3 bears this out: the median F2 seed SD was 3.79 points against 2.44 in F1.

**Datasets.** The two datasets differed in how strongly they showed the interaction (dataset × architecture × recipe, χ²(9) = 62.7). Analysed alone, Lee2019-MI gives H3 outcome B1 and Cho2017 gives A′.

**Home recipes.** The home recipes of EEGNet and MI-BMInet were approximations after protocol unification, whereas CSOANet-v2's was exact (Table 4).

## 5.8 Future work

Three extensions follow directly:

- **The statistical unit.**† The design should be repeated with the training run as a registered random effect and with more folds and seeds.
- **An independent test of the cropping effect.**† R1 against R1nc on a left/right-hand dataset not used here (for example BNCI2014-004 in MOABB). This is two recipes × four architectures and is the first follow-up we will run.
- **The cropping mechanism.**† R1 with cropping applied at test time as well, or with the cropped segment left in place, would show whether the harm comes from the zero padding, from the temporal shift, or from the mismatch as such.
- **New data.** The design should be repeated on data the authors collect themselves and on a visual-imagery paradigm, where decoders are less mature and recipe effects may be larger.
- **A shared protocol.** The multi-recipe evaluation — several recipes, pairwise intervals with a margin, home-recipe residuals — can be contributed to MOABB and braindecode as a protocol. The question asked here of four architectures could then be asked routinely of new ones.

# 6 Conclusion

As registered, no: the authors' recipe narrowed their model's gap to EEGNet. Exploratory analyses traced this to one implementation defect in that recipe; among the other recipe families tested, recipe and architecture were close to additive. The evaluation protocol mattered far more than the recipe: within subjects, the order of the architectures was largely reversed.

Under the registered analysis, the authors' recipe narrowed the gap between their model and EEGNet by 3.5 points relative to a published plain recipe. An exploratory rerun traced most of this to a random crop applied in training but not at test, which cost the baselines several points and the authors' model about one. Without it and without one numerically unstable recipe, all interaction residuals were within ±1 point. The single registered change in a pairwise conclusion was not robust to a small-sample analysis that accounts for shared training runs. By contrast, moving from cross-subject to within-subject evaluation moved CSOANet-v2 from last to first under three of four recipes.

Comparisons of compact decoders should therefore:

- check each augmentation for train/test consistency, above all in a recipe developed with one of the compared models;
- say who designed the shared recipe, and report at least one recipe that was not developed with any of the compared models;
- report results under more than one evaluation protocol;
- treat the training run, not only the subject, as a unit of uncertainty for every pairwise difference.

# Declarations

**Data availability.** Lee2019-MI and Cho2017 are public and were accessed through MOABB 1.7.2. All per-subject, per-seed results (float32 and int8) are released with the paper in the OSF project (osf.io/hfnkm). Trained models were not retained; they can be regenerated with the frozen code and the registered seeds, up to GPU non-determinism.

**Code availability.** Confirmatory code: github.com/jabarrick/recipe-x-arch, tag v1.0-frozen (commit a44e5933). The confirmatory driver is at tag v1.1-frozen (commit 1a7ba7b8; Amendment 2). The code is archived at Zenodo (DOI [to be minted at submission]). Pilot code: branch csoanet-v2-deploy of github.com/jabarrick/CSOANet2026.

**Preregistration.** OSF 8st47, DOI 10.17605/OSF.IO/8ST47, registered 1 November 2026:

- Amendment 1 (code freeze and analysis-plan addendum) was public on 13 November 2026.
- Amendment 2 (freeze of the confirmatory driver) was public on 15 November 2026.
- Both were public before the first confirmatory download, on 15 November 2026.

Deviations are listed in Supplementary S4.

**Author contributions (CRediT).** Yu Gao: conceptualisation, methodology, software, investigation, formal analysis, writing — original draft. Yimin Zhang: validation, writing — review and editing. [Adjust to actual contributions before submission.]

**Competing interests.** The authors proposed CSOANet and the R1 training recipe evaluated in this study. The study was preregistered with directional hypotheses and decision thresholds set before data access, to limit the influence of this interest (Section 3.6.4, Section 5.6). The preregistered home-recipe analysis returned the reading unfavourable to the authors' model. No other competing interests.

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
