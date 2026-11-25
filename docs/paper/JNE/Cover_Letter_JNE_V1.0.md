---
title: "Covering letter"
---

To the Editors, *Journal of Neural Engineering*

25 November 2026

Dear Editors,

We submit the manuscript **"Is a shared training recipe neutral? Recipe × architecture interactions in compact motor-imagery EEG decoders"** for consideration as a **Paper** in *Journal of Neural Engineering*.

Compact motor-imagery decoders are routinely compared under one shared training recipe, on the untested assumption that the recipe does not change their ranking. We tested this in a preregistered 4 × 4 design (four architectures × four recipes; OSF 8st47) on two public datasets that neither author had opened before registration. As registered, the recipe was not neutral: a recipe developed with one of the compared models narrowed that model's gap to the strongest baseline. An exploratory analysis traced most of the effect to one augmentation implemented with a train/test mismatch, and a smaller interaction remained with a published recipe. The paper ends with concrete recommendations for shared-recipe comparisons and shows that, in cross-subject benchmarks, the training run must be treated as a unit of uncertainty. We believe this is of direct interest to the journal's readers who design, benchmark or deploy brain–computer interface decoders.

**Conflict of interest.** The authors proposed CSOANet (Gao and Diniz 2026, *Comput. Methods Biomech. Biomed. Eng. Imaging Vis.* 14 2684109) and the R1 training recipe evaluated in this study. To limit the influence of this interest, the direction of the main contrast and every decision threshold were preregistered before data access, and the home-recipe analysis was designed so that the reading unfavourable to our own model needed less evidence than the favourable one. That analysis returned the unfavourable reading. We have no other competing interests.

**Preregistration and data.** The preregistration, its two amendments and all per-subject results are public on OSF; the frozen code is public on GitHub. Deviations from the registration are listed in supplementary material S4.

**Review.** We request single-anonymous review, because the preregistration identifies the authors.

The manuscript is original, has not been published elsewhere and is not under consideration by another journal. Both authors have approved the submission.

Yours sincerely,

Yu Gao (corresponding author), on behalf of both authors
Doctoral Programme in Electrical and Computer Engineering (PDEEC), Faculty of Engineering, University of Porto
Yu_Gao@alumni.baylor.edu
