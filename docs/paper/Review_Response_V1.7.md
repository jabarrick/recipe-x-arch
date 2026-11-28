---
title: "Response to the seventh internal review and to supervisor comments"
subtitle: "Manuscript V1.15 / JNE V1.4 — 28 November 2026"
---

# Supervisor comments (Ricardo Cruz, 27 November 2026)

| Comment | Response |
|---|---|
| "heavy" and "plain" recipes are never defined in the abstract | Adopted verbatim: the abstract now defines R1–R4 as suggested. |
| Fix the crop in the code, re-run, and leave it out of the abstract | The main result is now the corrected design (R1 without the crop; ShallowConvNet with max-norm), which required no new runs beyond the follow-ups. Because the study is preregistered, the registered results cannot be replaced; they are reported as run (Section 4.1). The abstract mentions the defects in one sentence and does not describe the crop. |

# Seventh internal review

| Point | Response | Where |
|---|---|---|
| 1a. Recipe main effect without the crop under-reported | Corrected design reported with and without R3, both scales: without R3, H1 χ²(2) = 1.1, p = 0.58; H2 χ²(6) = 10.2, p = 0.11. | 4.3; S9.14 |
| 1b. H2b reversal with R1nc under-reported; selective use of scales | Reported directly: −1.29 (−2.23, −0.32), p = 0.015 on the percentage scale; −0.04, p = 0.34 on the logit scale; not significant after Holm. Section 3.3 states that the percentage scale is primary and that both are reported wherever they disagree. | 4.3; 3.3 |
| 1c. State that all registered positive results trace to defects | Stated in the abstract, 5.1 and the conclusion. | Abstract; 5.1; 6 |
| 1d. Title finding is a self-made example; "preregistered" in title | Title changed to the research question; "preregistered" removed from the title. 5.1 says the study shows one way a recipe can bias a comparison and does not estimate frequency. The original CSOANet paper's own comparison did use different recipes for proposed model and baselines (5.4; correction in preparation). | Title; 5.1; 5.4 |
| 1e. Broken "analyses frozen" promise; Amendment 3 HB pointless | S9 introduction now says the promise was not kept and is withdrawn. HB is withdrawn (5.6); Amendment 3 will be revised before posting. | S9; 5.6 |
| 2a. Recovery percentages are not additive; no intervals | Withdrawn. The text now says only that consistent cropping did not remove the bias (baselines −2.1 to −3.3 against R1nc, intervals excluding zero; gap 2.55, 0.77 to 4.33). | 4.2; S9.14 |
| 2b. CSOANet-v2 "full recovery" built on noise | Removed. The text reports R1nc − R1ct = +0.03 (−0.46, 0.52) and that CSOANet-v2's total crop effect was not significant. | 4.2 |
| 2c. Kernel-length speculation contradicts Table S10.1; untested alternatives | Speculation removed; 5.2 names underfitting, BN statistics and one offset per batch as untested candidates and gives the practical check. | 5.2 |
| 2d. 2 × 2 shift × consistency factorial, BN re-estimation, PhysioNet R1nc/R1ct | Not done in this revision (needs new runs). Listed as untested in 5.2. | — |
| 3a. Registered tables rest on a defective baseline | Corrected 4 × 4 given side by side (Table 2, Table 3, Figure 2). | 4.3 |
| 3b. Reference-implementation check of all baselines | ShallowConvNet's omission corrected; EEGNet's epoch change and MI-BMInet's unstated dropout listed in limitations. No other check was possible without the reference code. | 5.5 |
| 3c. Asymmetric diagnosis: CSOANet-v2 under R3 | Diagnosed from training logs: 10 of 30 runs learned and matched R4 (−0.66 points); 20 stayed near chance; mixed outcomes across seeds in 5 of 10 folds. Read as an optimisation failure at lr 10⁻², and R3 is not used as evidence about recipes. A rerun at a lower learning rate is listed as not done. | 4.4; S9.14 |
| 3d. R2va read too narrowly | Average effect over the four architectures reported: +1.45 (0.27, 2.60). "Removed" replaced by "no longer significant but not zero". The R2va + max-norm combination was not run. | 4.4 |
| 4a. Numbers differ between text and supplement; bootstrap not seeded | All exploratory numbers now come from one script with seeds fixed by the data (identical contrasts give identical intervals). Earlier intervals superseded. | 3.3; S9.14 |
| 4b. Multiplicity; p and CI from different procedures | 51 exploratory contrasts listed with Holm-adjusted p; the text says which conclusions do not survive. 3.3 explains that p is from the restricted and the CI from the unrestricted bootstrap. | 3.3; S9.14 |
| 4c. Clusters not independent | Stated in 5.3 (two folds share about three quarters of their training subjects); repeated splits not run. | 5.3 |
| 4d. Replace with one trial-level binomial GLMM | Not done. The logit-scale analyses approximate it; a GLMM with crossed subject and run effects is a reasonable next step. | — |
| 5a. 5.4 placeholder | Filled: the original study's code shows unequal recipes and evaluation-fold checkpoint selection; the journal is being notified and a correction prepared. ORCID, funding, Zenodo DOI and CRediT remain placeholders. | 5.4 |
| 5b. No figure for the main finding; Figure 3 used naive CI | New Figure 2 (registered vs corrected) and Figure 3 (H2b with and without the crop, wild bootstrap CIs). Former Figure 4 moved to S9.14. | 4.1; 4.3 |
| 5c. Section 4.5 too long; H4, F2, int8 to supplement | Results restructured (4.1–4.4); main text 4,400 words (from 5,500); F2, H4 and int8 in one paragraph pointing to the supplement. | 4 |
| 5d. Stale supplement content | S9 introduction, S4 row 18, S9.10 title, Table S10.2 note and the version-label note fixed. | Supplement |
