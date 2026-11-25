---
title: "Covering letter"
---

To the Editors, *Journal of Neural Engineering*

25 November 2026

Dear Editors,

We submit the manuscript **"A train/test-inconsistent augmentation biased a shared-recipe comparison: a preregistered case study in compact motor-imagery EEG decoders"** for consideration as a **Paper** in *Journal of Neural Engineering*.

Compact motor-imagery decoders are routinely compared under one shared training recipe. We report a preregistered case study (four architectures × four recipes; OSF 8st47) of what happens when that recipe was developed together with one of the compared models. As registered, the recipe favoured that model. Exploratory analyses traced most of the effect to a single augmentation that was applied in training but not at test; beyond it, any remaining recipe × architecture interaction was small and uncertain. The paper also shows that, in cross-subject benchmarks, test subjects who share trained networks make subject-level intervals too narrow, and that the one registered change in a pairwise conclusion does not survive this correction. We believe these points are of direct use to readers who design, benchmark or deploy brain–computer interface decoders. Follow-up runs on the mechanism and an independent preregistered test (OSF Amendment 3) are under way.

**Conflict of interest.** The first author developed CSOANet (Gao and Diniz 2026, *Comput. Methods Biomech. Biomed. Eng. Imaging Vis.* 14 2684109) and the R1 training recipe evaluated in this study; the second author was not involved in either. The same statement appears in the manuscript. To limit the influence of this interest, the direction of the main contrast and every decision threshold were preregistered before data access, and the home-recipe analysis was designed so that the reading unfavourable to our own model needed less evidence than the favourable one. That analysis returned the unfavourable reading. We have no other competing interests.

**Preregistration and data.** The preregistration, its two amendments and all per-subject results are public on OSF; the frozen code is public on GitHub. Deviations from the registration are listed in supplementary material S4.

**Review.** We request single-anonymous review, because the preregistration identifies the authors.

The manuscript is original, has not been published elsewhere and is not under consideration by another journal. Both authors have approved the submission.

Yours sincerely,

Yu Gao (corresponding author), on behalf of both authors
Doctoral Programme in Electrical and Computer Engineering (PDEEC), Faculty of Engineering, University of Porto
Yu_Gao@alumni.baylor.edu
