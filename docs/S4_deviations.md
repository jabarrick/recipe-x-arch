# S4 — Preregistration deviations and post-freeze changes

Registration: OSF 8st47 (1 Nov 2026). Rules and analysis code frozen at f3fc1449 (2026-11-08T09:09:39Z).
Every change to `docs/decision_rules.md` or to the analysis code after f3fc1449, and every post-download fix to data-loading code, is one row here. "Before results" = made before any confirmatory run.

| # | Date (UTC) | Commit | File(s) | Registered / frozen text | Change | Reason | Before results? | Affects a decision rule? |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-08 | 2312659 | docs/decision_rules.md | change log row `[TO FILL]` | freeze commit hash and time recorded | self-reference only | yes | no |
| 2 | 2026-11-08 | 558ac92 | docs/decision_rules.md §5 | reference figures marked `[TO VERIFY]`; MI-BMInet recipe described only as "published recipe" | reference figures filled: EEGNet-8,2 60.46 % and ShallowConvNet 72.47 % from the MOABB benchmark (within-session 5-fold, BNCI2014-001); MI-BMInet 76.03 % from Wang et al. 2024 (session 1→2). Note added that R3 is MI-BMInet's cross-subject recipe, not its BCI IV 2a recipe | Lawhern 2018 gives no numeric within-subject figure; Schirrmeister 2017 figures are for cropped training; Wang 2024 uses different recipes for 2a and PhysioNet | yes | no — tolerance (±3) and handling rule unchanged; home mapping unchanged (defined on F1) |
