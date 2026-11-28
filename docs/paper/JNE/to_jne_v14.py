#!/usr/bin/env python3
"""Manuscript V1.11 -> Journal of Neural Engineering format (Manuscript_JNE_V1.0.md).
- IOP title page (numbered affiliations, corresponding author)
- structured abstract (Objective / Approach / Main results / Significance)
- numbered section headings (1., 2.1.)
- Vancouver numbering in order of first citation; IOP reference style
- end matter: Acknowledgments, Data availability statement, Ethical statement, Author contributions
"""
import re

W = "/home/claude/work/"
s = open(W + "Manuscript_V1.15.md", encoding="utf-8").read()

# ------------------------------------------------------------------ references (IOP style)
D = lambda d: f" (https://doi.org/{d})"
REF = {
 ("Aristimunha", "2025"): "Aristimunha B *et al* 2025 Braindecode: toolbox for decoding raw electrophysiological brain data with deep learning models (Zenodo)" + D("10.5281/zenodo.17699192"),
 ("Aristimunha", "2026"): "Aristimunha B, Carrara I, Guetschel P, Sedlar S *et al* 2026 Mother of all BCI Benchmarks (version 1.7.2) (Zenodo)" + D("10.5281/zenodo.10034223"),
 ("Bouthillier", "2021"): "Bouthillier X *et al* 2021 Accounting for variance in machine learning benchmarks *Proc. Mach. Learn. Syst.* **3** 747–69",
 ("Cameron", "2008"): "Cameron A C, Gelbach J B and Miller D L 2008 Bootstrap-based improvements for inference with clustered errors *Rev. Econ. Stat.* **90** 414–27" + D("10.1162/rest.90.3.414"),
 ("Chevallier", "2024"): "Chevallier S, Carrara I, Aristimunha B, Guetschel P, Sedlar S, Lopes B, Velut S, Khazem S and Moreau T 2024 The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark arXiv:2404.15319",
 ("Cho", "2017"): "Cho H, Ahn M, Ahn S, Kwon M and Jun S C 2017 EEG datasets for motor imagery brain–computer interface *GigaScience* **6** gix034" + D("10.1093/gigascience/gix034"),
 ("Dehghani", "2021"): "Dehghani M, Tay Y, Gritsenko A A, Zhao Z, Houlsby N, Diaz F, Metzler D and Vinyals O 2021 The benchmark lottery arXiv:2107.07002",
 ("Del Pup", "2025"): "Del Pup F, Zanola A, Tshimanga L F, Bertoldo A and Atzori M 2025 The more, the better? Evaluating the role of EEG preprocessing for deep learning applications *IEEE Trans. Neural Syst. Rehabil. Eng.* **33** 1061–70" + D("10.1109/TNSRE.2025.3547616"),
 ("Ferrari Dacrema", "2019"): "Ferrari Dacrema M, Cremonesi P and Jannach D 2019 Are we really making much progress? A worrying analysis of recent neural recommendation approaches *Proc. 13th ACM Conf. on Recommender Systems (RecSys '19)* (New York: ACM)" + D("10.1145/3298689.3347058"),
 ("Gao", "2026"): "Gao Y and Diniz J M 2026 From perception to imagination: a robust deep learning architecture for real-time EEG decoding and neurophysiological validation *Comput. Methods Biomech. Biomed. Eng. Imaging Vis.* **14** 2684109" + D("10.1080/21681163.2026.2684109"),
 ("Henderson", "2018"): "Henderson P, Islam R, Bachman P, Pineau J, Precup D and Meger D 2018 Deep reinforcement learning that matters *Proc. AAAI Conf. on Artificial Intelligence* **32**" + D("10.1609/aaai.v32i1.11694"),
 ("Holm", "1979"): "Holm S 1979 A simple sequentially rejective multiple test procedure *Scand. J. Stat.* **6** 65–70",
 ("Hou", "2026"): "Hou D, Wu Z, Jiang L, Li Z, Lin F and Yamada K D 2026 Same brain, different prediction: how preprocessing choices undermine EEG decoding reliability arXiv:2605.07212",
 ("Ingolfsson", "2020"): "Ingolfsson T M, Hersche M, Wang X, Kobayashi N, Cavigelli L and Benini L 2020 EEG-TCNet: an accurate temporal convolutional network for embedded motor-imagery brain–machine interfaces *Proc. 2020 IEEE Int. Conf. on Systems, Man, and Cybernetics (SMC)* pp 2958–65" + D("10.1109/SMC42975.2020.9283028"),
 ("Jayaram", "2018"): "Jayaram V and Barachant A 2018 MOABB: trustworthy algorithm benchmarking for BCIs *J. Neural Eng.* **15** 066011" + D("10.1088/1741-2552/aadea0"),
 ("Kessler", "2025"): "Kessler R, Enge A and Skeide M A 2025 How EEG preprocessing shapes decoding performance *Commun. Biol.* **8** 1039" + D("10.1038/s42003-025-08464-3"),
 ("Lawhern", "2018"): "Lawhern V J, Solon A J, Waytowich N R, Gordon S M, Hung C P and Lance B J 2018 EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces *J. Neural Eng.* **15** 056013" + D("10.1088/1741-2552/aace8c"),
 ("Lee", "2019"): "Lee M-H, Kwon O-Y, Kim Y-J, Kim H-K, Lee Y-E, Williamson J, Fazli S and Lee S-W 2019 EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy *GigaScience* **8** giz002" + D("10.1093/gigascience/giz002"),
 ("Lucic", "2018"): "Lucic M, Kurach K, Michalski M, Gelly S and Bousquet O 2018 Are GANs created equal? A large-scale study *Advances in Neural Information Processing Systems* **31**",
 ("Melis", "2018"): "Melis G, Dyer C and Blunsom P 2018 On the state of the art of evaluation in neural language models *Int. Conf. on Learning Representations (ICLR)*",
 ("Musgrave", "2020"): "Musgrave K, Belongie S and Lim S-N 2020 A metric learning reality check *Computer Vision – ECCV 2020 (Lecture Notes in Computer Science vol 12370)* (Cham: Springer) pp 681–99" + D("10.1007/978-3-030-58595-2_41"),
 ("Picard", "2021"): "Picard D 2021 torch.manual_seed(3407) is all you need: on the influence of random seeds in deep learning architectures for computer vision arXiv:2109.08203",
 ("Rommel", "2022"): "Rommel C, Paillard J, Moreau T and Gramfort A 2022 Data augmentation for learning predictive models on EEG: a systematic comparison *J. Neural Eng.* **19** 066020" + D("10.1088/1741-2552/aca220"),
 ("Schirrmeister", "2017"): "Schirrmeister R T, Springenberg J T, Fiederer L D J, Glasstetter M, Eggensperger K, Tangermann M, Hutter F, Burgard W and Ball T 2017 Deep learning with convolutional neural networks for EEG decoding and visualization *Hum. Brain Mapp.* **38** 5391–420" + D("10.1002/hbm.23730"),
 ("Schuirmann", "1987"): "Schuirmann D J 1987 A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability *J. Pharmacokinet. Biopharm.* **15** 657–80" + D("10.1007/BF01068419"),
 ("Sculley", "2018"): "Sculley D, Snoek J, Wiltschko A and Rahimi A 2018 Winner's curse? On pace, progress, and empirical rigor *Int. Conf. on Learning Representations (ICLR), Workshop Track*",
 ("Wang", "2024"): "Wang X, Hersche M, Magno M and Benini L 2024 MI-BMInet: an efficient convolutional neural network for motor imagery brain–machine interfaces with EEG channel selection *IEEE Sens. J.* **24** 8835–47" + D("10.1109/JSEN.2024.3353146"),
 ("Wightman", "2021"): "Wightman R, Touvron H and Jégou H 2021 ResNet strikes back: an improved training procedure in timm arXiv:2110.00476",
 ("Wimpff", "2024"): "Wimpff M, Gizzi L, Zerfowski J and Yang B 2024 EEG motor imagery decoding: a framework for comparative analysis with channel attention mechanisms *J. Neural Eng.* **21** 036020" + D("10.1088/1741-2552/ad48b9"),
 ("Wobbrock", "2011"): "Wobbrock J O, Findlater L, Gergle D and Higgins J J 2011 The aligned rank transform for nonparametric factorial analyses using only ANOVA procedures *Proc. SIGCHI Conf. on Human Factors in Computing Systems* (New York: ACM) pp 143–6" + D("10.1145/1978942.1978963"),
}

# ------------------------------------------------------------------ split
head_end = s.index("# 1 Introduction")
body = s[head_end:s.index("# Declarations")]

body = body.replace("| CSOANet 2026 |", "| Gao and Diniz, 2026 |")


NAME = r"(Ferrari Dacrema|Del Pup|[A-Z][a-zé]+)"
CIT = re.compile(NAME + r"(?: et al\.| and [A-Z][a-z]+)?,? (\d{4})")
order = []
def num(a, y):
    k = (a, y)
    assert k in REF, k
    if k not in order: order.append(k)
    return order.index(k) + 1

# pass 1: establish order of first appearance (text order)
for m in re.finditer(NAME + r"(?: et al\.| and [A-Z][a-z]+)?,? \(?(\d{4})\)?", body):
    if (m.group(1), m.group(2)) in REF: num(m.group(1), m.group(2))

def fmt(ns):
    ns = sorted(set(ns)); out = []; i = 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1: j += 1
        out.append(f"{ns[i]}–{ns[j]}" if j - i >= 2 else ", ".join(str(x) for x in ns[i:j + 1]))
        i = j + 1
    return "[" + ", ".join(out) + "]"

# pass 2: parenthetical groups made only of citations
def paren(m):
    parts = [p.strip() for p in m.group(1).split(";")]
    ns = []
    for p in parts:
        mm = CIT.fullmatch(p)
        if not mm or (mm.group(1), mm.group(2)) not in REF: return m.group(0)
        ns.append(num(mm.group(1), mm.group(2)))
    return fmt(ns)
body = re.sub(r"\(([^()]*?\d{4}[^()]*?)\)", paren, body)
# narrative: Name (1979) / Name et al. (2018)
body = re.sub(NAME + r"( et al\.| and [A-Z][a-z]+)? \((\d{4})\)",
              lambda m: (m.group(1) + (m.group(2) or "") + " " + fmt([num(m.group(1), m.group(3))])) if (m.group(1), m.group(3)) in REF else m.group(0), body)
# mixed parentheses, e.g. "(TOST against ±2; Schuirmann, 1987)"
def mixed(m):
    k = (m.group(1), m.group(2))
    return fmt([num(*k)]) if k in REF else m.group(0)
body = re.sub(NAME + r"(?: et al\.| and [A-Z][a-z]+)?, (\d{4})", mixed, body)
left = [x for x in CIT.findall(body) if x in REF]
assert not left, left
assert set(order) == set(REF), set(REF) - set(order)

# ------------------------------------------------------------------ headings: "# 1 Introduction" -> "# 1. Introduction"; "## 3.1 X" -> "## 3.1. X"
body = re.sub(r"^(#{1,3}) (\d+(?:\.\d+)*) ", lambda m: f"{m.group(1)} {m.group(2)}. ", body, flags=re.M)
# confidence intervals: [a, b] -> (a, b) so that square brackets mean citations only
body = re.sub(r"\[(−?-?\d+\.\d+|−?-?\d+), (−?-?\d+\.\d+|−?-?\d+\.?\d*)\]",
              lambda m: f"({m.group(1)}, {m.group(2)})" if ("." in m.group(1) or "." in m.group(2)) else m.group(0), body)
body = re.sub(r"(^|\. |\n)Supplementary (Table )?S", lambda m: m.group(1) + "Supplementary material " + (m.group(2) or "") + "S", body)
body = re.sub(r"Supplementary (Table )?S(?=\d)", lambda m: "supplementary material " + (m.group(1) or "").lower() + "S", body)
body = body.replace("Supplementary material table S", "Supplementary material table S")

# ------------------------------------------------------------------ front matter
front = """---
title: "Is a shared training recipe neutral? Architecture rankings of compact motor-imagery EEG decoders under four recipes"
---

Yu Gao^1,2,\\*^ and Yimin Zhang^1^

^1^ Doctoral Programme in Electrical and Computer Engineering (PDEEC), Faculty of Engineering, University of Porto, Rua Dr. Roberto Frias, 4200-465 Porto, Portugal

^2^ Doctoral Programme in Health Data Science (HEADS), Faculty of Medicine, University of Porto, Porto, Portugal

^\\*^ Author to whom any correspondence should be addressed.

E-mail: Yu_Gao@alumni.baylor.edu

ORCID iDs: Yu Gao [to add]; Yimin Zhang [to add]

# Abstract

*Objective.* Compact motor-imagery EEG decoders are often compared under one shared training recipe. We tested whether the choice of shared recipe changes such a comparison, in a case where one candidate recipe had been developed with one of the compared models. *Approach.* EEGNet, ShallowConvNet, MI-BMInet and the first author's CSOANet-v2 were each trained under four shared recipes: (R1) the recipe developed with CSOANet, which combines several augmentations with label smoothing; (R2) the recipe published with EEGNet; (R3) the recipe published with MI-BMInet; and (R4) R1 without augmentation or label smoothing. Data were Lee2019-MI and Cho2017 (106 subjects, cross-subject, three seeds), and hypotheses and decision rules were preregistered (OSF 8st47). *Main results.* The registered analysis found that the recipe effect differed across architectures and that R1 narrowed EEGNet's lead over CSOANet-v2 by 3.5 points of balanced accuracy (95% CI 1.4–5.6, with training runs as clusters). Post hoc checks traced these results to two defects in our implementation, an augmentation in R1 applied in training but not at test and missing weight constraints in ShallowConvNet, and to CSOANet-v2 failing to train in 20 of 30 runs under R3. With the two defects corrected, no pair of architectures moved between definitive pairwise categories under any recipe, and without R3 neither a recipe effect (p = 0.58) nor an interaction (p = 0.11) remained. EEGNet and MI-BMInet were the most accurate under every recipe, and CSOANet-v2 the least. Treating shared training runs as clusters reduced the definitive pairwise conclusions of the registered analysis from 20 to 13 of 24. *Significance.* Before recipes are compared, augmentations and baseline implementations should be checked for every compared architecture, and the training run should be treated as a unit of uncertainty.

**Keywords:** motor imagery, EEG decoding, training recipe, data augmentation, benchmarking, preregistration, compact neural networks

"""

end = f"""# Acknowledgments

[Funding: state any institutional support (FEUP / FMUP) or 'No specific funding was received for this work.']

# Conflict of interest

The first author developed CSOANet [{num('Gao','2026')}] and the R1 training recipe evaluated in this study; the second author was not involved in either. The direction of the main contrast and all decision thresholds were preregistered before data access, and the home-recipe rules were set against the first author's model (sections 3.3 and 5.4). No other competing interests.

# Data availability statement

The data that support the findings of this study are openly available. Lee2019-MI [{num('Lee','2019')}] and Cho2017 [{num('Cho','2017')}] were accessed through MOABB 1.7.2 [{num('Aristimunha','2026')}]. All per-subject, per-seed results (float32 and int8) are available in the OSF project at https://osf.io/hfnkm. The confirmatory code is available at https://github.com/jabarrick/recipe-x-arch (tag v1.0-frozen, commit a44e5933; confirmatory driver at tag v1.1-frozen, commit 1a7ba7b8) and archived at Zenodo [DOI to be minted at submission]. Trained models from the confirmatory runs were not retained; they can be regenerated with the frozen code and the registered seeds, up to GPU non-determinism. Models from all follow-up runs are retained and will be archived with the code.

# Preregistration

The study was preregistered on OSF (8st47; https://doi.org/10.17605/OSF.IO/8ST47) on 1 November 2026. Amendment 1 (code freeze and analysis-plan addendum) was public on 13 November 2026 and Amendment 2 (freeze of the confirmatory driver) on 15 November 2026, both before the first confirmatory download. Deviations are listed in supplementary material S4.

# Ethical statement

This study used only publicly available, anonymised datasets [{num('Lee','2019')}, {num('Cho','2017')}], collected by their original investigators under their own ethical approvals. No new data were collected and no further approval was required.

# Author contributions

Yu Gao: conceptualisation, methodology, software, investigation, formal analysis, writing – original draft. Yimin Zhang: validation, writing – review and editing. [Adjust to actual contributions before submission.]

# References

"""
refs = "\n\n".join(f"[{i + 1}] {REF[k]}" for i, k in enumerate(order))
out = front + body.rstrip() + "\n\n" + end + refs + "\n"
open(W + "Manuscript_JNE_V1.4.md", "w", encoding="utf-8").write(out)
print(len(order), "references; abstract words:",
      len(out[out.index("*Objective.*"):out.index("# 1. Introduction")].split()))
