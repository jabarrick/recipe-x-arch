---
title: "Response to the fifth internal review (manuscript V1.5 → V1.6)"
date: "25 November 2026"
---

The reviewer's diagnosis is right. In V1.5, each new analysis put a new strong claim into the abstract. V1.6 reverses that. The abstract now has three parts (registered result → one exploratory mechanism → recommendations). The F2 comparison and the protocol claim are out of the abstract and the conclusion. One V1.5 analysis was wrongly scaled; it is corrected, and the corrected test **overturns** the V1.5 additivity claim. No other analysis was added. Analyses are now frozen until the independent test in 5.8.

# Major points

## 1. "The evaluation protocol mattered more than any recipe"

Accepted in full. The sentence is removed from the abstract, 5.1 and the conclusion. 5.1 now states that this comparison was not tested and is not claimed.

- **Ranks versus intervals.** The F2 paragraph in 5.1 now leads with intervals (S6, item 4):
  - CSOANet-v2 is ahead of EEGNet only under R1 and R4.
  - It is ahead of MI-BMInet only under R1 and R3; under R2 and R4 the two are indistinguishable.
  - EEGNet is indistinguishable from ShallowConvNet under every recipe.
- **"Largely reversed"** is replaced by the reviewer's more accurate description: CSOANet-v2 and EEGNet swapped ends, and MI-BMInet and ShallowConvNet stayed in the middle.
- **Confounds.** All of the differences the reviewer listed between F1 and F2 are now set out in 5.1, followed by the statement that the design cannot separate them:
  - dataset;
  - training-set size and session transfer;
  - epochs and patience;
  - 20 validation trials;
  - standardisation;
  - accuracy range near chance.
- **Regularisation reading.** Added as an untested reading that is compatible with CSOANet-v2's flat F2 profile: a static gate and max-norm 0.25 suiting small training sets.
- **Recommendation.** "Report results under more than one evaluation protocol" is removed from the conclusion. It now appears in 5.1 as "where it is feasible".
- **Same-subject F1a vs F2 comparison.** Not run. It would remove only the dataset confound, and point 6 (freeze analyses) argues against adding it. It can be included in the 5.8 follow-up if wanted.

## 2. Scale error in the equivalence test

Accepted. The reviewer is right, and the correction goes further than the review expected.

`analysis/revision_v16.py` replaces the residual TOST. It tests all 18 2 × 2 differences-in-differences among {R1nc, R2, R4}, with a cluster intercept and t(9), unadjusted and Bonferroni-adjusted (Supplementary S9.11; S4 row 18). It recomputes from existing result files only; no runs were added.

| | Result |
|---|---|
| Contrasts whose 95 % CI excludes zero | 6 of 18, **all involving R2** |
| Largest | ShallowConvNet − CSOANet-v2, R4 vs R2: **3.29 [2.08, 4.50]** (registered H2b 3.50) |
| Next | ShallowConvNet − CSOANet-v2, R2 vs R1nc: −2.65 [−4.55, −0.75]; ShallowConvNet − MI-BMInet, R4 vs R2: 2.01 [0.95, 3.07] |
| R4 vs R1nc contrasts excluding zero | 0 of 6 |
| Equivalent, unadjusted 90 % | 7 of 18 |
| Equivalent, Bonferroni (18 contrasts) | **0 of 18** |

The reviewer's hand calculation from Table 12 is confirmed. We also found that the V1.5 table itself showed the problem: four of the twelve clustered residual CIs (ShallowConvNet and CSOANet-v2 under R2 and R4) excluded zero, yet all twelve were labelled "equivalent".

Consequences:

- "All interaction residuals ... equivalent to zero" is deleted everywhere.
- The answer to the title question changes. Removing the crop defect removed the interaction that favoured the authors' model, but it did not make the recipes interchangeable: R2 still shifts some pairwise gaps by 2–3 points, mostly ShallowConvNet's. "The only non-neutral element was the crop defect" is withdrawn.
- R2's checkpoint rule is named as the likeliest source, but **not** asserted. R2 also differs from R4 in optimiser, weight decay and schedule, and the rule was not isolated. This is also noted under *Model selection* in 5.7.
- The omnibus LRT on this table is not significant with crossed effects (p = 0.60) while single contrasts are. This is disclosed and explained (a 6-df omnibus test against a one-recipe interaction).
- The registered home-recipe rule also compares a single residual with ±2. It is kept as registered, but 5.7 (*Home recipes*) now says that its "equivalent" branch shares the scale problem.
- 5.8: the independent test will include R2, and the ShallowConvNet − CSOANet-v2 interaction between R2 and R1nc will be a second preregistered directional hypothesis.

## 3. Search reasoning in 5.5

Accepted; the reviewer is right. None of the eight configurations selected by T in F1, nor the four selected by T′, had cropping on. 5.5 now says that the search avoided the defect. Once cropping is avoided, only the checkpoint rule and batch 16 had clear effects, and R4 is already on the favourable side of both (early stopping on validation accuracy, batch 64), so a search had little left to gain over R4.

One detail, which we leave out of the text: in F2, three of the four selected configurations did have cropping on.

## 4. Dangling reference to the original CSOANet paper

Accepted. The sentence is deleted from 5.1. 5.6 now states both directions. The F2 results suggest that CSOANet-v2's standing depends on protocol; they do not validate the original comparison, which differed in paradigm, used unequal recipes, and used a dataset with documented limitations.

## 5. ds005697 "personal communication"

Accepted.

- The personal-communication citation and the sentence "limit the scope of that study's conclusions" are removed.
- 5.6 now says only that the dataset has limitations unrelated to the recipe question, which are being documented in a note on its OpenNeuro page `[reference to be added once the note is public]`, and that their bearing on the original paper is being addressed separately by its authors.
- "We have added a note to the CSOANet2026 repository" is changed to "We have prepared a note", until it is actually added.

Order before submission: OpenNeuro note public → cite it → agree the 5.6 wording with the original paper's co-author.

# Meta point: freeze

Accepted.

- **Analyses frozen.** The new *Analysis path* limitation in 5.7 says so explicitly. 3.6.7 records that V1.6 adds only the corrected test and that nothing further will be added before the 5.8 test.
- **S9 consolidated.** It is now one section, "Post hoc analyses (revisions V1.1–V1.6)". It opens with a summary table (part, revision, analysis, main result, status in V1.6), and the status column marks what is superseded or withdrawn. The parts are grouped into A (V1.1–V1.2) and B (V1.4–V1.6).
- **Abstract.** Now in three parts and 221 words. F2 is removed. The remaining interaction is one clause, needed because it changes the answer.

# Minor points

- **4.8(h) second caution carried into 5.1.** Done.
- **"Still last".** Replaced with "EEGNet could not be distinguished from ShallowConvNet under any recipe", giving 60.2 vs 61.7 as point estimates.
- **"Where feasible".** Applied, and the recommendation moved from the conclusion to 5.1.
- **Hou et al. 2026.** Verified in V1.4 (arXiv:2605.07212).
- **Funding, Zenodo DOI, Gao & Diniz DOI, CRediT.** Still pending from the authors.

# Not adopted

- **Tracked-changes .docx.** Not produced. Word's *Review → Compare* on Manuscript_V1.5.docx and Manuscript_V1.6.docx generates it.
