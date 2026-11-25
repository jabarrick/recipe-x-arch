---
title: "Response to the third internal review (Manuscript V1.3 → V1.4)"
subtitle: "25 November 2026"
---

| # | Comment | Response | Where |
|---|---|---|---|
| 1 | Without cropping and R3, recipe and architecture are nearly additive; reframe | **Accepted, and confirmed.** On {R1nc, R2, R4} every residual is within ±1.0 point. The interaction LRT gives χ²(6) = 23.7 with subjects as the unit and 4.6 (p = 0.60) with crossed subject and run effects. With R1nc in place of R1 the H2b contrast becomes −1.29 [−2.29, −0.29]. These results are now the core of the abstract (exploratory) and of a new 5.1 paragraph with Table 12. The "not heavy augmentation, one implementation" sentence moved to 5.1. The title is unchanged; an alternative is offered separately. | Abstract, 4.8(h), 5.1, Table 12, S9.9 |
| 2 | The original CSOANet paper | **Checked in the CSOANet2026 code.** CSOANet was trained with cropping. Baselines were trained without augmentation (5-fold) and with augmentation but no cropping (LOSO). The defect did not favour CSOANet there, but the recipes differed, which is the confound studied here. Disclosed in 5.6; reference added (Gao and Diniz, 2026); README note drafted for the CSOANet2026 repository. The Table 4 / 5.7 inconsistency about R1's origin is resolved. | 5.6, Table 3, Table 4, references |
| 3 | Circular hypothesis → test; "confirmed" | **Accepted.** "Confirmed" changed to "consistent with". Added R1nc − R1 on the 22 fold-1 test subjects, who were not used by the regression: EEGNet +3.5, ShallowConvNet +4.6, MI-BMInet +2.4, CSOANet-v2 +1.3; DiD +2.19 [0.12, 4.26]. Script timing and commits disclosed. A test on a third dataset is named as needed for confirmation. | 3.6.7, 4.8(g), 5.3, 5.7 |
| 4 | "Co-adaptation is specific" overreaches | **Accepted.** Replaced with "least affected; development under it or general insensitivity cannot be separated". The architectural explanations are marked as speculation. | 5.3 |
| 5 | F2 cell means missing; F2 differs | **Accepted.** The F2 table is in S3. "Never ahead" is restricted to F1, and the F2 difference is stated (CSOANet-v2 highest under R1, R3 and R4 within subjects). | 5.1, S3 |
| 6 | Sensitivity presentation | **Accepted.** Now two treatments (design effect only, with z or bootstrap; design effect plus small-sample correction). Leave-one-fold-out is separated and said not to address clustering. Cameron, Gelbach and Miller (2008) is cited for the few-cluster correction. Contribution (v) is now "an illustration", citing Bouthillier et al. (2021). | 3.6.7, 4.5, 5.1, contributions |
| 7 | "Deleted to bound disk use" | **Corrected:** the frozen driver saves no checkpoints and deletes exported ONNX files after scoring. | 3.5, 5.4 |

| Minor | Response |
|---|---|
| Abstract too dense | Rewritten (241 words): registered results, then the exploratory mechanism, then the recommendation. |
| 3.5–5.2 upper bound | The R4 comparator is now said to be chosen post hoc; the R3 DiD (15.8) is reported and explained. |
| Table 3, R1 row | The crop implementation is stated in the table. |
| Dataset difference | Both explanations (more signal removed; larger misalignment to the cue) are given as untested. |
| Leftovers | S4 row 1 no longer contains `[TO FILL]` (fixed in V1.2). The S8 figure embeds in the docx. Hou et al. 2026 was verified (arXiv:2605.07212). Still open: volume/pages/DOI of Gao and Diniz (2026), Zenodo DOI, CRediT, funding. |
