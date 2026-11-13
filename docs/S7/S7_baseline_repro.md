# S7 — baseline reproduction on BCI IV 2a (run 2026-11-11, before tag)

Driver `train/baseline_repro.py` at commit 531c7fa; reference figures and ±3-point tolerance fixed in decision_rules §5 (558ac92) before any run; protocols made concrete in S4 row 4 before any run. Three seeds per baseline.

| Baseline | Reference | Reproduced (mean of 3 seeds) | Per seed | Difference | Within ±3 | Protocol |
|---|---|---|---|---|---|---|
| EEGNet-8,2 | 60.46 (MOABB benchmark) | 61.8 | 62.14, 61.45, 61.83 | +1.34 | **yes** | MOABB within-session 5-fold, 8–32 Hz, 128 Hz |
| ShallowConvNet | 72.47 (MOABB benchmark) | 66.62 | 66.37, 66.9, 66.6 | -5.85 | **no** | MOABB within-session 5-fold, 8–32 Hz, 250 Hz |
| MI-BMInet | 76.03 (Wang et al. 2024) | 74.05 | 73.69, 74.65, 73.8 | -1.98 | **yes** | session 1 → 2, 0–3 s, 0.5–100 Hz |

Mean epochs run: EEGNet 249 of 300, ShallowConvNet 138 of 300 (early stopping on validation loss, patience 75), MI-BMInet 500 of 500.

## Remaining protocol and definition differences

- EEGNet: our temporal kernel 65 / separable 17 at 128 Hz (MOABB Keras 64 / 16); max-norm 1 (depthwise) and 0.25 (classifier) as in Lawhern et al. (MOABB: 0.5 on the classifier); PyTorch default initialisation (Keras: Glorot uniform).
- ShallowConvNet: **no max-norm** (braindecode / study definition); MOABB's Keras version uses max-norm 2.0 per filter on both convolutions and 0.5 on the classifier. SafeLog clamps at 1e-6 (MOABB: clip to [1e-7, 1e4]). PyTorch default initialisation (Keras: Glorot uniform).
- MI-BMInet: all 288 trials per session used (the paper removes artefact-marked trials; MOABB does not expose the marks); 3 seeds (paper: 25 repetitions); fixed per-channel gain from session 1; dropout 0.5.

## ShallowConvNet outside tolerance — pre-specified handling (decision_rules §5)

1. **Implementation check (done, documented).** `tests/test_shallow.py` shows the study's ShallowConvNet is numerically identical to braindecode 2.0 `ShallowFBCSPNet` (outputs equal to 1e-4 on three input shapes, same parameter count). No implementation error was found.
2. **Implementation kept.** The study's ShallowConvNet is not changed.
3. **Gap reported here.** −5.85 points against the MOABB reference.
4. **Home-residual flag:** not applicable — ShallowConvNet has no home recipe in the set. Because ShallowConvNet serves as descriptive side-evidence in mechanism module (i), its residual pattern will be reported with this gap attached.
5. **Diagnostic (descriptive, does not change the model):** the same protocol with MOABB's max-norm constraints added (`--archs ShallowConvNet+MOABB-maxnorm`, S4 row 5; seed 0 only; run 12–13 Nov 2026). Result: **74.15 %** (+1.68 against 72.47, within ±3), against 66.37 % for the study definition with the same seed. The constraint raised accuracy in all 18 subject × session units (by 3.1 to 17.1 points; median 6.9) and lengthened training (mean 243 vs 138 epochs before early stopping). The gap to the MOABB reference is therefore explained by the max-norm constraints of the Keras implementation that produced the reference figure, not by an implementation error.

**Implication recorded before the confirmatory runs.** The study's ShallowConvNet follows braindecode's definition (no max-norm); on BCI IV 2a within-session under the MOABB protocol this definition is about 8 points below the constrained variant. In the cross-subject binary pilot on PhysioNet-MI, the study definition was among the two best architectures under every recipe (S7_pilot_binary.md), so there is no sign of a comparable handicap in the confirmatory setting, but readers should know the size of the difference in this one.
