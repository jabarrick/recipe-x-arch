# S8 — interval-width simulation

Pilot: 105 subjects, median 45 trials/cell; confirmatory n_trials = 200; SESOI = ±2.0; 2000 simulations per scenario; category precedence: better first.

## Per-pair SD decomposition (points)

| pair                        | recipe   |   mean_diff_pilot |   sd_pilot |   sd_binom_pilot |   sd_between | truncated   |   sd_binom_conf |   sd_conf |
|:----------------------------|:---------|------------------:|-----------:|-----------------:|-------------:|:------------|----------------:|----------:|
| EEGNet - ShallowConvNet     | R1       |             -1.93 |      11.19 |             8.34 |         7.46 | False       |            3.95 |      8.44 |
| EEGNet - ShallowConvNet     | R2       |             -0.64 |       9.91 |             7.81 |         6.1  | False       |            3.69 |      7.13 |
| EEGNet - ShallowConvNet     | R3       |              0.29 |       7.68 |             7.99 |         0    | True        |            3.78 |      3.78 |
| EEGNet - ShallowConvNet     | R4       |             -0.28 |       9.25 |             7.88 |         4.84 | False       |            3.73 |      6.11 |
| EEGNet - MI-BMInet          | R1       |             -3.2  |       9.2  |             8.24 |         4.11 | False       |            3.9  |      5.66 |
| EEGNet - MI-BMInet          | R2       |             -0.48 |       5.88 |             7.83 |         0    | True        |            3.71 |      3.71 |
| EEGNet - MI-BMInet          | R3       |             -0.18 |       5.98 |             7.98 |         0    | True        |            3.77 |      3.77 |
| EEGNet - MI-BMInet          | R4       |              0.36 |       5.61 |             7.91 |         0    | True        |            3.74 |      3.74 |
| EEGNet - CSOANet-v2         | R1       |              1.46 |      10.52 |             8.6  |         6.05 | False       |            4.07 |      7.29 |
| EEGNet - CSOANet-v2         | R2       |              1.75 |      10.06 |             8.01 |         6.09 | False       |            3.79 |      7.18 |
| EEGNet - CSOANet-v2         | R3       |              4.13 |      10.38 |             8.28 |         6.25 | False       |            3.92 |      7.38 |
| EEGNet - CSOANet-v2         | R4       |              3.32 |      10.9  |             8.16 |         7.23 | False       |            3.86 |      8.19 |
| ShallowConvNet - MI-BMInet  | R1       |             -1.27 |       8.53 |             8.1  |         2.68 | False       |            3.83 |      4.67 |
| ShallowConvNet - MI-BMInet  | R2       |              0.15 |       8.99 |             7.78 |         4.5  | False       |            3.68 |      5.82 |
| ShallowConvNet - MI-BMInet  | R3       |             -0.48 |       7.53 |             7.99 |         0    | True        |            3.78 |      3.78 |
| ShallowConvNet - MI-BMInet  | R4       |              0.64 |       9.34 |             7.89 |         5.01 | False       |            3.73 |      6.25 |
| ShallowConvNet - CSOANet-v2 | R1       |              3.39 |       8.55 |             8.47 |         1.2  | False       |            4.01 |      4.18 |
| ShallowConvNet - CSOANet-v2 | R2       |              2.39 |       9.76 |             7.95 |         5.65 | False       |            3.76 |      6.79 |
| ShallowConvNet - CSOANet-v2 | R3       |              3.84 |       9.85 |             8.3  |         5.31 | False       |            3.92 |      6.6  |
| ShallowConvNet - CSOANet-v2 | R4       |              3.61 |       8.69 |             8.13 |         3.07 | False       |            3.85 |      4.92 |
| MI-BMInet - CSOANet-v2      | R1       |              4.67 |       8.89 |             8.37 |         3    | False       |            3.96 |      4.96 |
| MI-BMInet - CSOANet-v2      | R2       |              2.23 |       9.43 |             7.98 |         5.03 | False       |            3.78 |      6.29 |
| MI-BMInet - CSOANet-v2      | R3       |              4.32 |       9.98 |             8.28 |         5.58 | False       |            3.92 |      6.82 |
| MI-BMInet - CSOANet-v2      | R4       |              2.97 |      11.37 |             8.16 |         7.92 | False       |            3.86 |      8.81 |


## Scenario grid

|   D |   G |   n |   P_A |   P_A_sign |   P_A_equiv_only |   P_B1 |   P_B2 |   mean_n_definitive |   mean_changes_def_def |   mean_changes_sign_rev |   mean_changes_inc_def |   median_halfwidth |
|----:|----:|----:|------:|-----------:|-----------------:|-------:|-------:|--------------------:|-----------------------:|------------------------:|-----------------------:|-------------------:|
|   1 |   0 | 106 | 0.028 |      0     |            0.028 |  0.001 |  0.768 |               5.74  |                  0.038 |                   0     |                  9.959 |              1.783 |
|   1 |   1 | 106 | 0.75  |      0.234 |            0.516 |  0     |  0.212 |               7.442 |                  1.731 |                   0.236 |                 15.02  |              1.784 |
|   1 |   2 | 106 | 1     |      0.996 |            0.003 |  0     |  0     |              10.34  |                  5.074 |                   1.917 |                 17.818 |              1.782 |
|   1 |   3 | 106 | 1     |      1     |            0     |  0     |  0     |              12.616 |                  8.11  |                   4.195 |                 18.612 |              1.786 |
|   2 |   0 | 106 | 0.034 |      0     |            0.034 |  0.003 |  0.614 |               6.692 |                  0.044 |                   0     |                  9.98  |              1.786 |
|   2 |   1 | 106 | 0.758 |      0.144 |            0.614 |  0     |  0.104 |               9.234 |                  1.716 |                   0.144 |                 16.794 |              1.783 |
|   2 |   2 | 106 | 1     |      0.996 |            0.004 |  0     |  0     |              10.57  |                  3.557 |                   1.399 |                 18.016 |              1.783 |
|   2 |   3 | 106 | 1     |      1     |            0     |  0     |  0     |              11.634 |                  4.938 |                   3.12  |                 18.634 |              1.784 |
|   3 |   0 | 106 | 0.033 |      0     |            0.033 |  0.052 |  0.108 |              10.981 |                  0.039 |                   0     |                  8.39  |              1.784 |
|   3 |   1 | 106 | 0.686 |      0.068 |            0.617 |  0.016 |  0.012 |              12.252 |                  1.294 |                   0.068 |                 15.301 |              1.783 |
|   3 |   2 | 106 | 0.989 |      0.975 |            0.014 |  0     |  0     |              12.906 |                  3.532 |                   1.238 |                 16.622 |              1.784 |
|   3 |   3 | 106 | 1     |      1     |            0     |  0     |  0     |              13.098 |                  3.66  |                   2.356 |                 17.495 |              1.785 |
|   1 |   0 | 193 | 0.234 |      0     |            0.234 |  0.403 |  0     |              16.052 |                  0.498 |                   0     |                  9.93  |              1.312 |
|   1 |   1 | 193 | 1     |      0.725 |            0.275 |  0     |  0     |              17.809 |                  9.676 |                   0.764 |                 12.905 |              1.312 |
|   1 |   2 | 193 | 1     |      1     |            0     |  0     |  0     |              18.257 |                 14.489 |                   3.232 |                 10.657 |              1.31  |
|   1 |   3 | 193 | 1     |      1     |            0     |  0     |  0     |              19.658 |                 17.076 |                   5.386 |                  9.962 |              1.311 |
|   2 |   0 | 193 | 0.4   |      0     |            0.4   |  0.284 |  0     |              15.994 |                  0.916 |                   0     |                 11.233 |              1.311 |
|   2 |   1 | 193 | 1     |      0.46  |            0.54  |  0     |  0     |              17.488 |                  8.264 |                   0.471 |                 13.582 |              1.311 |
|   2 |   2 | 193 | 1     |      1     |            0     |  0     |  0     |              17.946 |                 11.575 |                   2     |                 11.966 |              1.312 |
|   2 |   3 | 193 | 1     |      1     |            0     |  0     |  0     |              18.183 |                 11.809 |                   5.743 |                 12.825 |              1.312 |
|   3 |   0 | 193 | 0.413 |      0     |            0.413 |  0.475 |  0     |              17.824 |                  0.89  |                   0     |                  8.536 |              1.311 |
|   3 |   1 | 193 | 0.997 |      0.198 |            0.799 |  0.003 |  0     |              18.108 |                  4.998 |                   0.213 |                 11.468 |              1.311 |
|   3 |   2 | 193 | 1     |      1     |            0     |  0     |  0     |              18.698 |                 10.398 |                   1.86  |                 10.808 |              1.31  |
|   3 |   3 | 193 | 1     |      1     |            0     |  0     |  0     |              18.622 |                 10.459 |                   4.346 |                 11.625 |              1.31  |


## Decision quantity (decision_rules.md §4)

P(B2 | G = 0, D ∈ {1,2,3}, n = 106) = **0.496** → do NOT add

P(A | G = 0) = 0.032 (family-wise false-positive rate of the registered H3 rule)

P(A_sign | G = 0) = 0.000 (sign reversals only)

P(A from equivalent ↔ better only | G = 0) = 0.032


Reading rule (Amendment 1 §3.2): if H3 support comes only from equivalent ↔ better changes with no sign reversal, it is reported as 'difference below SESOI with stable direction', not as a change in ranking.
