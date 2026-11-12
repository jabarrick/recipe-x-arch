# T′ — selected configurations (search run 2026-11-11, before tag)

Search: `train/tprime_search.py` at commit 531c7fa; 24 configurations of S1 (`docs/S1_search_space.md`), same draws for every architecture (verified equal to the S1 draws to 1e-12); PhysioNet-MI left/right, fold 1 of the pilot split (training subjects for training, validation subjects for scoring; fold 1's test subjects unused); score = mean over seeds 0 and 1 of validation balanced accuracy at the selected checkpoint; 192 runs, 0 non-finite.

| Architecture | Config | Optimiser | Learning rate | Schedule | Batch | Augmentations | Label smoothing | Selection rule | Score | Runner-up |
|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet | #10 | Adam | 3.06e-03 | ×0.1 at 40 %/80 % | 64 | noise, shift, channel dropout | 0.0 | all epochs, lowest val loss | 86.41 | 85.86 |
| ShallowConvNet | #1 | Adam | 3.40e-04 | cosine to 0.01×lr | 32 | none | 0.0 | early stop on val bacc (patience 15) | 88.65 | 88.51 |
| MI-BMInet | #13 | AdamW (wd 1e-3) | 2.53e-03 | ×0.1 at 40 %/80 % | 32 | noise | 0.0 | early stop on val bacc (patience 15) | 89.77 | 89.64 |
| CSOANet-v2 | #10 | Adam | 3.06e-03 | ×0.1 at 40 %/80 % | 64 | noise, shift, channel dropout | 0.0 | all epochs, lowest val loss | 87.51 | 85.29 |

These four configurations are condition T′ and are applied unchanged to every confirmatory cell (Amendment 1 §3.10).

## Descriptive notes (not evidence)

- Runner-up within 0.1–0.6 points for EEGNet, ShallowConvNet and MI-BMInet; the median |seed 0 − seed 1| difference of a configuration's score is 1.1–1.3 points, larger than those margins. The selection among the top configurations is therefore largely seed noise (the winner's curse anticipated in Methods §3.4). CSOANet-v2's margin is 2.2 points.
- No selected configuration uses mixup or label smoothing; ShallowConvNet's uses no augmentation at all. EEGNet and CSOANet-v2 selected the same configuration (#10).
- Spread of scores across the 24 configurations: EEGNet 7.0, ShallowConvNet 7.8, MI-BMInet 7.4, **CSOANet-v2 17.1** points. Spearman correlation of configuration rankings between architectures 0.29–0.63.
- PhysioNet-MI is the development dataset of CSOANet-v2 and R1 (disclosed).

Files: `tprime_selected.json`, `tprime_scores.csv` (24 × 4 scores), `tprime_runs.csv` (192 runs), `tprime_configs.json`.
