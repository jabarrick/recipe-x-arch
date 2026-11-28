---
title: "Covering letter"
---

To the Editors, *Journal of Neural Engineering*

[Date of submission]

Dear Editors,

We submit the manuscript **"Is a shared training recipe neutral? Architecture rankings of compact motor-imagery EEG decoders under four recipes"** for consideration as a **Paper** in *Journal of Neural Engineering*.

Compact motor-imagery decoders are routinely compared under one shared training recipe. In a preregistered comparison (four architectures × four recipes, 106 subjects; OSF 8st47), one of the recipes had been developed together with one of the compared models. The registered analysis found that the recipe changed the comparison and favoured that model. Post hoc checks traced this to two defects in our own implementation, an augmentation applied in training but not at test and missing weight constraints in one baseline, and to one model's optimisation failure under one recipe. With the defects corrected, no pairwise conclusion changed between recipes. We report both analyses side by side. The paper also shows that, in cross-subject benchmarks, test subjects who share trained networks make subject-level intervals too narrow. We believe these points are of direct use to readers who design or benchmark brain–computer interface decoders. An independent preregistered test on a further dataset (OSF Amendment 3) is under way.

**Conflict of interest.** The first author developed CSOANet (Gao and Diniz 2026, *Comput. Methods Biomech. Biomed. Eng. Imaging Vis.* 14 2684109) and the R1 training recipe evaluated in this study; the other authors were not involved in either. While preparing this study we found that the published Methods of that earlier article did not describe the training of its baselines accurately; we are notifying that journal, and a correction is in preparation (section 5.4). The direction of the main contrast and every decision threshold of the present study were preregistered before data access, and the home-recipe analysis was designed against our own model. We have no other competing interests.

**Preregistration and data.** The preregistration, its three amendments and all per-subject results are public on OSF; the frozen code is public on GitHub. Deviations from the registration are listed in supplementary material S4.

**Review.** We request single-anonymous review, because the preregistration identifies the authors.

The manuscript is original, has not been published elsewhere and is not under consideration by another journal. All authors have approved the submission.

Yours sincerely,

Yu Gao (corresponding author), on behalf of all authors
Doctoral Programme in Electrical and Computer Engineering (PDEEC), Faculty of Engineering, University of Porto
Yu_Gao@alumni.baylor.edu
