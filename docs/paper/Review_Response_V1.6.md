---
title: "Response to the sixth internal review (JNE V1.0 → Manuscript V1.12 / JNE V1.1)"
date: "25 November 2026"
---

We checked every point against the data and the text. Most of the review is correct, including four errors that we introduced while condensing the manuscript.

V1.12 / JNE V1.1 makes four kinds of change:

- It adopts the reviewer's case-study framing.
- It corrects the statistics.
- It adds two re-analyses of existing results: on the logit scale, and with a wild cluster bootstrap.
- It starts the requested experiments and an independent preregistered test. These are not yet reported.

New values come from `analysis/revision_v17.py` (S4 row 19; S9.12).

# 1. Fatal issues

**1.1 The title asks a general question but the answer is a bug.** *Accepted.*

- **New title:** "A train/test-inconsistent augmentation biased a shared-recipe comparison: a preregistered case study in compact motor-imagery EEG decoders".
- **Introduction and conclusion:** both now state that the study shows *how* a shared recipe can bias a comparison, not *how often*.
- **The registered result stands as registered:** a recipe developed with one of the compared models favoured that model.

**1.2 The remaining interaction is selectively reported.** *Accepted. It was our error, and the corrected value differs.*

- The omnibus test had been dropped from the main text when we condensed it. It is now in 4.5 and in the abstract.
- On re-checking, the χ²(6) = 4.6 (p = 0.60) in S9.9 came from a fit that had not converged. The best converged fit over several optimisers gives χ²(6) = 9.9, p = 0.13. On the logit scale it is χ²(6) = 4.9, p = 0.56.
- The conclusion, that the remaining interaction is not significant once training runs are modelled, is the same. The number is corrected in the text, and the error is recorded in S4 row 19.

**1.3 Asymmetric criteria.** *Accepted.*

- 4.5 now reports both counts: 6 of 18 contrasts exclude zero unadjusted, and 2 of 18 after Bonferroni adjustment, both involving ShallowConvNet between R4 and R2.
- On the logit scale the counts are 4 and 0.
- The abstract no longer says that a published recipe "shifted gaps by 2–3 points".

**1.4 H2 is mainly R3's instability; "flagged runs excluded, conclusions unchanged" is wrong.** *Accepted.*

- 4.1 now reports that without R3 the interaction's share of the fixed-effect variance fell from 30 % to 10 % (crossed χ²(6) = 21.0, p = 0.002), and says that much of H2 reflects R3.
- The sensitivity paragraph now says that excluding flagged runs empties the ShallowConvNet × R3 cell, so the pre-specified model path was used.

**1.5 Unit of analysis; the abstract uses the wrong interval; S8.** *Accepted.*

- The abstract now gives the cluster-robust interval (1.4–5.7).
- 4.1 reports the crossed and logit tests next to the registered ones.
- H2b is now reported with three intervals: subject-level, cluster t(9), and wild cluster bootstrap (1.44, 5.51; p = 0.002).
- S8 and 5.3 now state that the power simulation omitted the training-run variance, so its 3 % false-positive rate is optimistic.

**1.6 The mechanism is untested and no models were kept.** *Accepted.*

- Runs with the crop applied at validation and test, and with an in-place crop that has no shift, are coded (`train/explore_rev6.py`). They save every model.
- 4.5 and 5.6 now say the mechanism was not tested.
- The confirmatory driver predates this decision. That is stated as a limitation, and every new run retains its models.

**1.7 CSOANet-v2's "insensitivity" may be a floor effect.** *Partly accepted.*

- On the empirical-logit scale, the crop still widened the EEGNet − CSOANet-v2 gap: 0.26 logits (0.12, 0.39). H2b also holds: 0.22 (0.12, 0.32).
- 5.2 now says that part of CSOANet-v2's smaller response could reflect its position on the scale, and the "insensitive to recipes in general" reading is removed.
- A trial-level GLMM would need trial-level predictions, which the frozen driver did not store. The logit of the per-subject correct counts is the closest available approximation.

**1.8 Our own binary pilot contradicts the direction.** *Accepted.*

- The introduction and 3.3 now report the binary pilot's figures: EEGNet +4.2, CSOANet-v2 +3.9, H2b +0.29 points.
- Both say it argued against our prediction.

# 2. Design and statistics

- **Recipes are packages.** *Accepted.* This is now stated in 3.1 and in the limitations. R2's checkpoint rule is isolated in the new R2va runs (R2 with the checkpoint chosen by validation accuracy).
- **Delete "no further analyses".** *Done.*
- **ShallowConvNet has no max-norm.** *Accepted.* 3.1, 5.1 and 5.5 now name the missing max-norm as a candidate cause of the remaining interaction. Runs of ShallowConvNet with max-norm under every recipe are coded.
- **Home recipes are not really home.** *Accepted.* 5.2 now calls the home-recipe analysis an operationalisation that did not work as intended, and gives three reasons.
- **H3 logic.** *Partly disputed.* The categories cannot both hold: by the precedence rule fixed in Amendment 1, an interval that excludes zero is *better*. This rule is now spelled out in 3.2. We accept the point about dichotomisation: 3.2, 5.3 and the conclusion now give priority to the differences-in-differences.
- **SESOI lacks a basis.** *Accepted.* 3.2 now says that seed SD exceeded 2 points in 13 of 16 cells, so the SESOI is a convention fixed in advance.
- **Mixed model and bootstrap.**
  - *Random slopes:* these were already a registered sensitivity analysis and are now named in 4.4.
  - *Bootstrap:* accepted. The pairs cluster bootstrap with a normal critical value is now described as over-rejecting. Key contrasts use the wild cluster bootstrap (Webb weights).
- **One split seed.** Stated in 3.1 and 5.5. The independent test uses ten folds.
- **Four validation subjects; T and H4.** *Accepted.* 4.3 now reports that 11 of the 24 configurations used the defective crop and that the search was scored on about four subjects. The claim in 5.4 that "a fixed recipe is as fair as an equal budget" is deleted.
- **Cho2017 is not a replication.** *Accepted.* It is now described as a check with a fixed analysis drawn from the same set of runs, and the twofold dataset heterogeneity is reported.
- **"Fold-1 about half" is inaccurate.** *Accepted.* The text now gives all four comparisons.
- **F2 should support nothing.** *Accepted.* The F2 discussion and the small-sample speculation about CSOANet-v2 are deleted. 4.4 keeps three sentences.
- **Quantisation is thin.** *Accepted.* It is cut to one paragraph, and the unequal operator coverage is stated. "Quantisation" is removed from the keywords.
- **Narrow scope.** *Accepted.* The limitations now name the missing FBCSP and Riemannian baselines and the missing newer architectures.

# 3. Credibility and conflict of interest

- **Conflict-of-interest statement in the manuscript.** *Done.* JNE V1.1 has a "Conflict of interest" section in addition to the covering letter.
- **"The rules were set against our model" does not hold.** *Partly accepted.* 5.4 now says that the post hoc analyses explain the registered result but do not reverse it, and that CSOANet-v2 remained the least accurate architecture.
- **The original CSOANet paper needs a correction, not a repository note.** *Decision needed from the authors.*
  - 5.4 has a placeholder for how this will be reported to the journal.
  - The vague paragraph on ds005697 is deleted, because it is unrelated to this study.
- **Timeline.**
  - *Accepted:* the sentence "Discussion drafted before data access" and the phrase "five rounds of internal review" are deleted. S4 and S9 keep the dated record, with a note explaining the draft labels.
  - *Not removable:* the statement that neither author had opened the datasets cannot be verified by others. It is the registered declaration.
- **Author details.**
  - *Fixed:* "the authors proposed CSOANet" now reads "the first author developed CSOANet; the second author was not involved".
  - *For you to decide:* the GitHub user name (jabarrick) and the e-mail domain. See the covering message.

# 4. Manuscript issues

- **Placeholders:** these remain (ORCID, funding, Zenodo DOI, CRediT) and need information from the authors.
- **Stale draft references in the supplement:** fixed where they point to the current text. Historical rows in S4 are kept, with a note on what the draft labels mean.
- **Wrong table pointer for training accuracy:** fixed; it now points to S3.
- **Undefined terms:** A, A′, B1, B2 and intermediate are now defined in 3.2.
- **"R2 checkpoints came early" over-generalised:** now limited to ShallowConvNet (median epoch 6.5; the other architectures 20–63.5).
- **Formatting:**
  - The empty bullets before (i)–(iii) are fixed.
  - Table 2's column widths are set so that "ShallowConvNet" no longer wraps.
  - Figure 2's left panel now uses offset points without connecting lines, and the duplicated labels are removed.
- **References:** all 30 were verified against publisher, Crossref, arXiv or Zenodo records.
- **Over-compression:** partly reversed. The main text is now about 4,900 words.

# 5. Plan

| Item | Status |
|---|---|
| Case-study framing; statistics rewritten around training runs and differences-in-differences | done in this version |
| Crop at validation and test; in-place crop; R2va; ShallowConvNet with max-norm; models retained | code ready (`tools/run_rev6.ps1`, about 2 days of GPU time) |
| Independent preregistered test (Dreyer2023, 10 folds) | Amendment 3 drafted; driver and analysis code ready; to be frozen and posted before download |
| H4, F2, int8, home residuals | compressed |
