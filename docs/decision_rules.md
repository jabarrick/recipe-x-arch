# Frozen decision rules — recipe-x-arch

**Registration:** OSF 8st47 (DOI 10.17605/OSF.IO/8ST47), registered 1 Nov 2026.
**Purpose of this file:** fix, in a dated commit, every decision rule that Amendment 1 adds to the registration, *before* the binary pilot (Section 6) is run and before any confirmatory dataset (Lee2019-MI, Cho2017, and Dreyer2023 if added) is downloaded. The commit history of this file is the evidence that the rules preceded the data. Any later change to this file is a deviation and must be listed in supplementary table S4 with its hash and reason.

**Status at first commit (f3fc1449, 2026-11-08T09:09:39Z):** no confirmatory dataset downloaded, opened or analysed; binary pilot not started.

Where this file and the registration differ, the registration wins for everything it specifies. This file only fills gaps and adds secondary analyses.

---

## 1. Outcome categories for H3

H3 is decided exactly as registered (24 pairwise intervals = 6 architecture pairs × 4 recipes; Bonferroni-adjusted 99.79 % CIs; categories *A better* / *B better* / *equivalent* (CI within ±2 points) / *inconclusive*; H3 supported if any pair falls into two different definitive categories under two recipes).

When H3 is **not** supported, the 24 intervals are classified:

| Label | Rule | May be described as |
|---|---|---|
| **B1 — stable** | ≥ 16 of 24 intervals definitive (A better, B better or equivalent) **and** no pair moves between two different definitive categories across recipes | evidence that a shared recipe preserves architecture conclusions |
| **B2 — inconclusive** | < 8 of 24 intervals definitive | uninformative about the neutrality of a shared recipe; **never** as support for a shared recipe |
| intermediate | anything else | counts only, no overall label |

Moves between *inconclusive* and a definitive category do not block B1 but are counted and reported.

**Category precedence.** An interval that excludes 0 *and* lies within ±2 is *A better* / *B better*; *equivalent* is only for intervals that include 0 and lie within ±2. Same rule in S8 and in the confirmatory analysis.

**Reading of equivalent ↔ better changes.** H3 is decided as registered (any definitive ↔ definitive change). If all such changes are equivalent ↔ better and there is no sign reversal (A better ↔ B better), the result is reported as "pairwise differences below the SESOI with stable direction across recipes" and is **not** described as a change in architecture ranking. Number of sign reversals is always reported beside the H3 decision.

Always reported, regardless of outcome:

- number of category changes out of the 36 opportunities (6 pairs × C(4,2) recipe pairs), split into (a) definitive ↔ definitive (sign reversal or equivalent ↔ better) and (b) inconclusive ↔ definitive;
- the same 24 intervals and categories on int8 balanced accuracy and the number of float32 → int8 category changes (sensitivity only; decides nothing).

## 2. Home-recipe analysis (secondary confirmatory)

### 2.1 Home mapping (fixed)

| Architecture | Home recipe | Notes |
|---|---|---|
| EEGNet-8,2 | R2 | published 2018 on other data; epochs/batch unified by protocol → approximation |
| MI-BMInet | R3 | epochs unified by protocol, batch 16 kept → approximation |
| CSOANet-v2 | R1 | developed jointly with R1 by the authors on pilot data; unchanged |
| ShallowConvNet | none in set | reported descriptively as closest to R2 |

R4 (= R1 without augmentation) is treated as *near-home* for CSOANet-v2.

### 2.2 Definition of home advantage

For architecture *a* with home recipe *home(a)*, from the 4 × 4 table of cell means:

```
h_a = ȳ[a, home(a)] − ȳ[a, ·] − ȳ[·, home(a)] + ȳ[·, ·]
```

This cell-mean formula **is the definition** for all tests. The sum-to-zero-coded mixed-model term (αρ)[a, home(a)] is computed only as a check and must agree on complete balanced tables. Treatment-coded coefficients are not this quantity and are not used. "Home minus mean of other recipes" and "home minus best away" are **not** used.

Per-subject version `h[s, a]`: same formula applied to each subject's own 4 × 4 table (seed-averaged, float32, family F1).

### 2.3 Tests

- One-sample two-sided *t*-test of `h[s, a]` against 0 for EEGNet, MI-BMInet, CSOANet-v2; Holm correction over the three, α = 0.05.
- CSOANet-v2 additionally: TOST equivalence test, bounds ±2 points, 90 % CI. "No home advantage" may be claimed **only** if equivalence is shown; never from a non-significant *t*-test.
- Recipe sensitivity per architecture: **residual range** = max − min of its four interaction residuals (αρ)[a, r], with subject-level bootstrap 95 % CI (2,000 resamples, resampling subjects). Raw range (best − worst recipe mean) is reported but **not** used for decisions.
- Sensitivity: all residuals recomputed on the 4 × 3 table without the R4 column (whole column removed for all architectures).

### 2.4 Incomplete tables (two estimation paths, both implemented and unit-tested)

Primary analysis has no incomplete tables (training failures kept as registered). In the sensitivity analysis that excludes training failures:

1. **Subject-drop path (default):** a subject whose 4 × 4 table loses any cell is dropped from the home-recipe analysis; number dropped is reported.
2. **Model path (fallback):** if more than 10 % of subjects would be dropped, `h_a` is instead estimated as the sum-coded interaction term of the mixed model fitted to all remaining cells.

Unit tests (part of the frozen tag): on synthetic 4 × 4 tables with planted residuals, (i) the cell-mean formula recovers the planted residuals; (ii) the sum-coded model term equals the formula on complete balanced tables to numerical tolerance; (iii) both paths give the same `h_a` when no cell is missing.

### 2.5 Decision mapping (deliberately asymmetric)

| Pattern | Reading |
|---|---|
| **(i)** CSOANet-v2 home residual significantly positive (whatever the others show) | consistent with co-adaptation; cannot exclude genuine fit; H2b is **not** read as evidence of CSOANet-v2's recipe robustness. Descriptive side-evidence only: if ShallowConvNet's residual pattern resembles CSOANet-v2's, this leans toward genuine fit |
| **(ii)** CSOANet-v2 home residual equivalent to 0 (TOST) **and** its residual range is the smallest of the four in ≥ 80 % of bootstrap resamples | robustness interpretation supported, in the *relative* sense (closest to additive within this set of four). If EEGNet or MI-BMInet also show a significantly positive home residual → strongest form of support |
| **(iii)** anything else | mechanism behind H2b reported as unresolved; all effect sizes reported |

Asymmetry is intentional: one significant residual suffices for the reading unfavourable to the authors' own model; the favourable reading needs equivalence **plus** ranking. Ranking threshold is 80 %, not 95 %, because strict ranking of four coupled residual ranges at 95 % is reachable only for very large differences. Residuals sum to zero within each recipe column, so a small residual range means "closest to additive among these four", not "insensitive to recipes".

## 3. Training-failure rule (training-side signals only)

A run is a **training failure** if either:

- its loss is non-finite after the registered retry with seed + 100; or
- its training balanced accuracy, computed by one forward pass over the **un-augmented** training trials at the **finally selected checkpoint**, is ≤ 55 %.

Test accuracy is never used to flag or exclude a run. Primary analyses keep failed runs as registered; a sensitivity analysis excludes them. If an H3 category change depends only on cells containing training failures, it is reported as a recipe-robustness issue, not an architecture × recipe interaction.

## 4. Rule for adding a third dataset (decision M1′)

Input: S8 simulation (Section 7) run on the binary pilot of Section 6.

Deciding quantity: **P(B2) under the zero-interaction scenario with true pairwise differences of 1–3 points, at n = 106.**

- If **P(B2) > 0.5** → Dreyer2023 is added as cell F1c (MOABB `Dreyer2023A/B/C`; window 0–4 s after cue, fixed in Amendment 1, not after download). H1, H2, H2b remain decided on the registered F1 (Lee2019-MI + Cho2017) with the registered Holm family. B1/B2 for H3 are decided on F1 + F1c; H3 on registered F1 is reported alongside. Dataset × architecture and dataset × recipe interactions become required analyses for the extended set.
- Otherwise → no dataset added; H3 kept as registered; the expected low probability of a definitive outcome is stated in the amendment so that B2 is not read as evidence either way.

Pre-condition for adding: both authors confirm they have never downloaded, opened or analysed Dreyer2023.

This decision is made once, before any confirmatory download, and recorded in Amendment 1.

## 5. Baseline reproduction tolerance

Each baseline is trained with its published recipe on the pilot dataset BCI IV 2a (within-subject) and compared with **one** published figure, named here before any run:

| Baseline | Reference figure | Source |
|---|---|---|
| EEGNet-8,2 | within-subject 4-class accuracy, BCI IV 2a | Lawhern et al. 2018 — [TO VERIFY: table/figure, window, band-pass, session split] |
| ShallowConvNet | within-subject 4-class accuracy, BCI IV 2a | Schirrmeister et al. 2017 — [TO VERIFY: table, protocol] |
| MI-BMInet | within-subject accuracy, BCI IV 2a | Wang et al. 2024 — [TO VERIFY: table, protocol] |

**Tolerance:** ±3 percentage points, with the published protocol matched as closely as the pipeline allows; every remaining protocol difference listed in S7 beside the two numbers.

**If outside tolerance:** the implementation is checked once against the reference and the check is documented; if no error is found, the implementation is kept, the gap is reported in S7, and that architecture's home-recipe residual is flagged as resting on an imperfect home recipe. The tolerance is not widened after the fact.

The pilot value EEGNet/R2 within-subject BCI IV 2a = 67.1 % (already in the registration) is a result of this pipeline, not the published figure, and is reported alongside.

## 6. Binary pilot (to be run only after this file is committed)

- Data: PhysioNet-MI, left vs right hand, runs 4, 8, 12; subjects S088, S092, S100, S104 excluded. Registered preprocessing and evaluation.
- Design: 4 architectures × 4 recipes (R1–R4), cross-subject 5-fold, **one seed (seed 0)**.
- Uses: (a) subject-level balanced-accuracy table as input to S8; (b) end-to-end check of the frozen pipeline. Nothing else. Not evidence for or against any hypothesis.
- Disclosure: PhysioNet-MI is the development dataset of CSOANet-v2 and R1. Full results go to S7.
- The pilot does **not** feed back into any rule in this file.

## 7. S8 — interval-width simulation (procedure fixed here; numbers filled later)

Implemented in `analysis/s8_interval_simulation.py` (part of the frozen tag).

1. For each architecture pair *(a, b)* and recipe *r*, compute subject-level differences `d[s] = y[s,a,r] − y[s,b,r]` from the binary pilot.
2. Decompose the observed variance: `Var_obs = Var_between + Var_binom`.
   - `Var_binom` for a cell = `p(1−p)/n_trials` per subject, with *p* that subject's accuracy and *n_trials* that subject's scored trials (≈ 45 in the pilot); for a difference, the **sum** of the two cells' binomial variances (assumes uncorrelated trial-level errors; conservative). Average over subjects.
   - `Var_between = max(Var_obs − Var_binom, 0)`; report how often truncation occurs.
3. Recompose at the confirmatory trial count: `SD_conf = sqrt(Var_between + Var_binom(n = 200))` [TO VERIFY: 200 per subject per cell in Lee2019-MI / Cho2017 cross-subject].
4. Between-subject covariance over the 16 cells: pilot covariance of subject-centred tables minus the binomial diagonal, projected to PSD. If more than 8 of 16 eigenvalues are clipped, use the simplified subject × architecture + residual model (`--cov arch`) and state it.
5. Scenarios: D = largest true pairwise difference, main effects D × (0, 1/3, 2/3, 1), D ∈ {1, 2, 3}; G = largest |interaction residual|, ±G swap between two architectures across R1/R2, G ∈ {0, 1, 2, 3}. Simulate n = 106 (and n = 106 + Dreyer2023 subjects). Bonferroni 99.79 % intervals; classify with the precedence in Section 1; apply Section 1 rules.
6. Report per pair: pilot SD, binomial component, recomposed SD, median half-width; per scenario: P(A) (registered rule), P(A, sign reversals only), P(B1), P(B2). Report both P(A) values under G = 0 explicitly.
7. Hand P(B2) from the scenario named in Section 4 to decision M1′.

## 8. int8 calibration sample

- Cross-subject cells: 256 trials sampled without replacement, uniformly over all training trials of the fold (training subjects only; never validation or test subjects); RNG seed = fold index.
- Within-subject cell: 256 trials from the subject's session-1 training trials, same scheme, seed = subject index; all of them if fewer than 256.
- Same calibration set for every architecture × recipe × seed within a fold / subject. Trials un-augmented. Sampled indices saved with the exported models and listed in S5.
- MinMax calibration, static QDQ, per-channel symmetric weights, as registered.

## 9. Condition T scoring and compute cuts (restated from Amendment 1 for completeness)

- T scoring: mean validation accuracy over seeds 0 and 1 per configuration; seed 0 only if budget cut (reported).
- T′: same 24-configuration search run on the PhysioNet-MI left/right subset before freezing; selected configurations fixed in Amendment 1 §3.10.
- Run order: F1 R1–R4 → F2 R1–R4 → T (F1) → T′ (F1) → T (F2) → T′ (F2).
- Cut order if budget exhausted: 1. T for F2; 2. T′ for F2; 3. T scoring seeds for F1 from 2 to 1. No cut touches runs deciding H1, H2, H2b or H3.

## 10. Post-download changes

Allowed without counting as deviations only if confined to data loading / IO code and not touching models, recipes, splits, seeds or analysis code; one commit per fix, listed in S4 with hash and reason. Everything else is a deviation.

---

*Change log of this file*

| Date | Commit | Change |
|---|---|---|
| 2026-11-08T09:09:39Z | f3fc1449bede7aabb5888b980d26ecf3912af59f | first commit, before binary pilot (rules + analysis code + tests) |
