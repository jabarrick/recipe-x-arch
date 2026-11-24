---
title: "Response to the second internal review (Manuscript V1.1 → V1.2)"
subtitle: "24 November 2026"
---

| # | Comment | Response | Where |
|---|---|---|---|
| 1 | Post hoc results override registered ones in the abstract and conclusion | **Accepted.** The abstract now reports the registered results first: H1, H2 and H2b supported (with H2b's effect size), the home analysis ruling out the favourable reading, and H3 = A′. Post hoc results follow under a "Post hoc" label. The conclusion now says "not robust to a post hoc analysis … with a small-sample correction". | Abstract, 6 |
| 2 | "Could not be distinguished" ≠ "competitive" | **Accepted.** The central claim is re-anchored on the registered H2b difference-in-differences (3.50 [2.29, 4.71]; clustered [1.39, 5.67]) and its R4 analogue (5.20 [3.84, 6.57]; clustered [2.18, 8.35]). The category count is demoted to supporting evidence; we state explicitly that MI-BMInet stays ahead and that the clustered interval does not exclude a 6.8-point deficit. | 4.5, 5.1, abstract, 6 |
| 3 | Intro and contributions contradict the discussion | **Accepted.** RQ2 and contributions (ii) and (iii) are rewritten. (ii) now states that the home analysis can rule out the favourable reading but cannot identify co-adaptation in this design; (iii) states that T did not beat the best fixed recipe; (v) on statistical units is added. 4.4 keeps the registered statement of (i) and points to 5.3. | 1, 4.4, 5.3 |
| 4 | Random cropping may be an implementation artefact | **Accepted and confirmed in the code.** One random offset per batch; 90 % kept and moved to the start; the last 10 % zero-filled; no cropping at test time. Now described in 3.3. With configuration-clustered SEs, cropping costs EEGNet 9.5, MI-BMInet 6.8, ShallowConvNet 5.8 and CSOANet-v2 only 1.4 [−0.5, 3.4] points. This is presented in 5.3 as a concrete, post hoc, testable candidate mechanism. The training-accuracy argument is now stated not to separate under-fitting from the mismatch. **Exploratory F1 runs under R1 without cropping (120 runs) are prepared; results to be added in V1.3.** | 3.3, 4.1, 4.8(e), 5.3, 5.5, 5.7, S9.7 |
| 5 | Defend the clustered analysis | **Accepted.** The two models are shown to be the same for single contrasts (3.6.7). Added H3 under t(8) with a dataset effect, a normal critical value, a cluster bootstrap and leave-one-fold-out. A′ survives the normal critical value, the bootstrap and all five leave-one-fold-out analyses; it is lost under t(9) and t(8). The text now says this. The variance comparison is corrected to run (5.4) against residual (15). | 3.6.7, 4.5, 4.7, 5.1, 5.7, S9.5 |
| 6 | "Bimodal" contradicts Fig. 1; the failure threshold misses runs | **Accepted.** 11 runs were flagged and 8 more had training accuracy of 55–60 % with test accuracy near chance. The text now says the cell largely failed and the threshold missed part of it. A run-level table is added to S3. | 4.1, 5.3, 5.7, S3 |
| 7 | "Models not retained" vs 3.5 | **Accepted.** 3.5 and Data availability now say that the exported models were deleted after scoring and can be regenerated with the frozen code and seeds, up to GPU non-determinism. | 3.5, 5.4, Declarations |
| 8 | Lowest-validation-loss selection picks very early checkpoints | **Accepted.** ShallowConvNet × R2 median epoch 6.5 and MI-BMInet 20; the rule costs 1.8 [1.2, 2.4] points in the search data. Added to 4.1 and to the limitations. | 4.1, 5.7 |

| Wording / figures | Response |
|---|---|
| "by more than 2 points" | Now "2.3–2.9 points, clustered lower bounds about 1". |
| Unnumbered DiD table | Moved inline as a list; full table in S9.6. |
| Fig. 4 | Arrows carry their category; clustered intervals overlaid as light bars. |
| † on the 5.5 heading | Removed; a note under the heading marks the paragraph as new. |
| S7.3 numbers | 6.6–8.7 points (PhysioNet 52.7 → 59.3; BCI IV 2a 58.4 → 67.1). |
| S7.3 recipe identity | The pilot shared recipe uses the same augmentation function and the same optimiser, schedule, batch and patience as R1. "Relaxed" is a weight-constraint variant, not a recipe. |
| F2 int8 change | MI-BMInet − CSOANet-v2 under R3: CSOANet-v2 better (−5.14) in float32, inconclusive (−4.82) in int8. |
| ICC instability | Note added: ten clusters, and the high ICCs reflect failed runs concentrated in some folds. |
| S8 empty heading | The image has a proper caption (Figure S8.1) and embeds in the docx; the markdown alt text was what showed. |
| S4 row 1 `[TO FILL]` | It quoted the registered placeholder; reworded. |
| Hou et al. 2026 | Verified: arXiv:2605.07212, 8 May 2026. |
| Title missing in docx | The docx has a Title-style paragraph; please check which viewer showed it missing. |
| Still open | CSOANet 2026 citation, Zenodo DOI, CRediT, Funding; R1-without-crop results. |
