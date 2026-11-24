---
title: "Response to the internal review of Manuscript V1.0"
subtitle: "Changes made in Manuscript V1.1 and Supplementary V1.1 — 24 November 2026"
---

Every change below is either a correction or a post hoc analysis labelled as such (Methods 3.6.7; S4 row 14). No registered decision was changed.

## Major points

| # | Comment | Response | Where |
|---|---|---|---|
| 1 | Subjects treated as independent although they share trained networks | **Accepted.** Added training-run clustering. A crossed subject + training-run model keeps H1, H2 and H2b, which remain supported (H2 χ²(9) = 95.3). With a random intercept per dataset × fold cluster, the median H3 half-width rises from 1.80 to 3.89 points, only 13 of 24 intervals stay definitive, and no definitive ↔ definitive change remains, so H3 would not be supported. Reported beside the registered decision, in the abstract, in 5.1 and in the limitations; S8's independence assumption is noted. | 3.6.7, 4.2, 4.3, 4.5, 4.7, 5.1, 5.7, S9.1 |
| 2 | Home residual is mostly an R3 column effect; MI-BMInet's home residual mirrors CSOANet-v2's collapse; "our model prefers our recipe" is literally wrong | **Accepted.** EEGNet's non-home R3 residual (+3.00) exceeds MI-BMInet's home residual (+2.60); the sentence "co-adaptation is not specific to CSOANet-v2" is deleted. 5.3 is rewritten: R1 penalised the baselines more than CSOANet-v2, whose own best recipe is R2. The no-R3 value of 1.32 is now in the abstract and in 5.3. | 4.4, 5.3, abstract |
| 3 | The strongest finding is buried | **Accepted.** Under R1 CSOANet-v2 could not be distinguished from EEGNet or ShallowConvNet; under every other recipe EEGNet led by 5.0–17.3 points; 7 of the 10 inconclusive ↔ definitive changes involve CSOANet-v2. This is now in the abstract, 4.5, 5.1 and the conclusion. | abstract, 4.5, 5.1, 6 |
| 4 | "Stable direction / order did not depend on the recipe" overstated; S8 misused | **Accepted.** The EEGNet − MI-BMInet point estimate changes sign under R4, inside the margin; this is stated. Difference-in-differences R1 vs others: +2.30 to +2.87, intervals excluding 0 even when clustered. The S8 sentence now says what S8 can and cannot show. | 4.5, 5.1 |
| 5 | T's gain is measured against the worst recipe | **Accepted.** T − R4: −1.01 to +0.03 points; T′ − R4: +0.28 to +0.95. 5.5 and the abstract now say T did not beat the best fixed recipe. | 4.6, 5.5, abstract, S9.3 |
| 6 | R3 measures numerical stability | **Accepted.** No-R3 H1/H2 and variance shares added: H2 χ²(6) = 59.0; interaction share of fixed effects 30 % → 10 %; CSOANet-v2 range 11.6 → 1.8. The abstract gives both ranges. Figure 1 marks subjects whose cell contains a flagged run. Limitations explain that ShallowConvNet's R3 cell is effectively a different recipe. | 4.1, 4.2, 4.8(d), 5.7, Fig. 1 |
| 7 | Why R1 is bad: augmentation vs early stopping | **Accepted as exploratory.** Per-cell diagnostics added (S3): under R1, training accuracy at the selected checkpoint is 3–10 points below R4 with similar epochs. The T-search regression (no new runs) identifies random cropping as the harmful component (−5.9 [−6.9, −4.9] points); the selection rule and batch 16 also matter. | 4.8(e), 5.5, S3, S9.4 |
| 8 | Implementation choices inside the architecture factor | **Accepted.** New limitation paragraph covering ShallowConvNet without max-norm and MI-BMInet's N_k/N_f. | 5.7 |
| 9 | int8 conclusion too strong; MinMax "conservative" | **Accepted.** Reworded to "toolchain and calibration choices, together with the recipe, can change a conclusion"; MinMax is now called the simplest default. A percentile-calibration control could not be run because the exported models were not retained; this is stated. | 3.5, 4.7, 5.4 |
| 10 | Cho2017 edge artefacts "cannot create an interaction" | **Accepted.** Weakened: an interaction through R1's time-shift and crop augmentations cannot be excluded. | 3.8 |

## Structure and writing

| Comment | Response |
|---|---|
| Abstract too dense, uninformative p-values | Rewritten around effect sizes (241 words); p-values removed. |
| Intro paragraph 4 is methods | Shortened to two sentences, with a pointer to 3.6.6. |
| Four-class pilot missing from S7 | Added as S7.3 from the retained summary files. The claim that the pilot "reversed a significant model difference" is not supported by those files and was removed from 1 and 3.7. **Needs your check:** restore it only if per-subject pilot results showing the reversal exist. |
| "Discussion written before data" | Note rewritten: modules had numerical placeholders filled afterwards; revised sentences are marked †. |
| Related work Known / Missing labels | Labels removed. |
| Figure 4 axes | Common axis (−8 to 9) with arrows and values for off-scale intervals. |
| Residual table unnumbered; bold lost | Merged into Table 6 as its bottom block; home cells marked (h). |
| Table 7 last row broken | Moved to the caption. |
| Section "4.3b" | Results renumbered 4.1–4.8; cross-references updated. |
| 4.7(b) Spearman on 12 cells | Removed. |
| "p = 0.050. This is 0.0501" | Now p = 0.0501. |

## Consistency list

| # | Item | Response |
|---|---|---|
| 1 | Residual fragment in 3.1.1 | Removed. Lee 0–4 s and Cho 0–3 s are consistent with Table 11. |
| 2 | "about 40 validation trials" | Corrected to 20 (4.7, 5.7). |
| 3 | "about 21 subjects" | Corrected to 22. |
| 4 | "five or six validation subjects" | Corrected to four. |
| 5 | v1.1-frozen date | S4 row 6 dated 2026-11-15 (tag v1.1-frozen). |
| 6 | f3fc1449 vs a44e5933 | Both freezes explained in 3.8 and in the S4 header. |
| 7 | S4 row 13 "Table 9" | Corrected to Table 11. Row 7's reference is to Amendment 1's own section numbering and is kept. |
| 8 | 3.8 cites 3.6.4 for the failure rule | The bullet refers to the incomplete-table rule, now stated explicitly in 3.6.4. The training-failure rule itself is in 3.1.4. |
| 9 | Placeholders | S8 figure now embeds in the supplement. Still open: CSOANet citation, Zenodo DOI, CRediT, funding. S4 row 1 quotes the registered `[TO FILL]` text verbatim and is kept. |
| 10 | References | Added Jayaram & Barachant (2018) and the braindecode software citation. Hou et al. (2026), arXiv:2605.07212, verified to exist. Kessler et al. DOI added. |
