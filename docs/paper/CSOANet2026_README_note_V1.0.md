## Note on the random-crop augmentation (added November 2026)

The `augment()` functions in `batch_run_csoanet2026.py` and `LOSO/batch_run_loso.py` implement random cropping as follows:
- One random offset is drawn per batch.
- 90 % of the window from that offset is moved to the start of the window.
- The last 10 % of the window is filled with zeros.

Test windows are never cropped, so training and test inputs differ systematically: the content is shifted in time, and there is a zero tail.

In a later preregistered study on motor-imagery data (Gao & Zhang, *Is a shared training recipe neutral?*, OSF 8st47, https://doi.org/10.17605/OSF.IO/8ST47; code: https://github.com/jabarrick/recipe-x-arch), this cropping implementation lowered cross-subject balanced accuracy by:
- 3.7–5.9 points for EEGNet, ShallowConvNet and MI-BMInet;
- about 1 point for CSOANet-v2.

That analysis was exploratory.

**Scope for the results in this repository.** In the CSOANet2026 paper:
- CSOANet2026 was trained with this augmentation.
- The deep-learning baselines (EEGNet, ShallowConvNet, DeepConvNet) were trained without augmentation in the 5-fold within-subject comparison.
- In the LOSO comparison, the baselines were trained with augmentation but without cropping (`LOSO/batch_run_loso_baselines.py`).

The cropping defect therefore did not handicap the baselines in these results. The proposed model and the baselines were, however, trained with different recipes, so the comparison does not separate architecture from training recipe. The study above examines that confound.

The code is left unchanged so that the published results stay reproducible. Users who reuse `augment()` should set `crop=1.0`, or apply the same crop at test time.
