# S7 — binary pilot (disclosed; not evidence)

PhysioNet-MI, left vs right hand imagery (runs 4/8/12; S088, S092, S100, S104 excluded), 105 subjects, 44.8 scored trials per subject (42–45).
4 architectures × 4 recipes, cross-subject 5-fold (split seed 0, 10 % validation subjects), one training seed (0), 100 epochs max, 1–40 Hz, 0–4 s, fixed per-channel gain.
Driver `train/pilot_binary.py` at commit 0eff2c8; run 9–10 Nov 2026, after the rules were frozen at f3fc1449. Raw rows: `pilot_binary_long.csv`; settings: `pilot_binary_settings.json`.

Training failures: 0 of 1,680 subject rows (0 of 80 runs); retries: 0.

## Subject-level balanced accuracy, cell means (%)

| architecture   |    R1 |    R2 |    R3 |    R4 |
|:---------------|------:|------:|------:|------:|
| EEGNet         | 76.95 | 81.17 | 80.55 | 80.8  |
| ShallowConvNet | 78.88 | 81.81 | 80.26 | 81.08 |
| MI-BMInet      | 80.16 | 81.66 | 80.73 | 80.44 |
| CSOANet-v2     | 75.49 | 79.42 | 76.42 | 77.48 |

## Interaction residuals (points; double-centred cell means)

| | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| EEGNet | −1.21 | **−0.13** | 0.77 | 0.56 |
| ShallowConvNet | 0.09 | −0.13 | −0.16 | 0.21 |
| MI-BMInet | 1.12 | −0.52 | **0.08** | −0.67 |
| CSOANet-v2 | **−0.00** | 0.79 | −0.69 | −0.09 |

Home cells in bold. H2b contrast (EEGNet R2−R1 minus CSOANet-v2 R2−R1): +0.29 points (SD 10.83, n = 105); four-class pilot was +3.95.

## Training diagnostics (mean over folds)

|                          |   train_acc |   val_acc |   epochs |
|:-------------------------|------------:|----------:|---------:|
| ('CSOANet-v2', 'R1')     |        79.3 |      80.4 |     37.8 |
| ('CSOANet-v2', 'R2')     |        88.6 |      81.2 |    100   |
| ('CSOANet-v2', 'R3')     |        82.8 |      77.3 |    100   |
| ('CSOANet-v2', 'R4')     |        85.1 |      81.8 |     42   |
| ('EEGNet', 'R1')         |        78.8 |      78.4 |     30.4 |
| ('EEGNet', 'R2')         |        86   |      83   |    100   |
| ('EEGNet', 'R3')         |        85.6 |      82.2 |    100   |
| ('EEGNet', 'R4')         |        84.1 |      83.7 |     31.4 |
| ('MI-BMInet', 'R1')      |        87.9 |      83.3 |     52.8 |
| ('MI-BMInet', 'R2')      |        89.8 |      84.1 |    100   |
| ('MI-BMInet', 'R3')      |        86.7 |      83.7 |    100   |
| ('MI-BMInet', 'R4')      |        88.6 |      85.2 |     30   |
| ('ShallowConvNet', 'R1') |        90   |      82.6 |     30.6 |
| ('ShallowConvNet', 'R2') |        95.4 |      84.3 |    100   |
| ('ShallowConvNet', 'R3') |        86.4 |      82.9 |    100   |
| ('ShallowConvNet', 'R4') |        96.5 |      85.7 |     34.8 |

R1 and R4 stop early on validation accuracy (patience 15) at about 30–50 epochs; R2 and R3 run all 100 epochs and keep the lowest-validation-loss checkpoint.

## Reading

Descriptive only. R1 (the authors' recipe) was the lowest recipe for every architecture; CSOANet-v2 (the authors' model) was the lowest architecture under every recipe. Both lower the authors' prior for H2b and for any CSOANet-v2 advantage. PhysioNet-MI is the development dataset of CSOANet-v2 and R1, so even this pattern is not independent evidence. No rule was changed after these results.
