# Confirmatory runs

Code: tag `v1.1-frozen` (1a7ba7b) plus the data-loading fix c804e98 (S4 row 7). Environment: `docs/environment_confirmatory.txt`
(MOABB 1.7.2, MNE 1.11.0, numpy 2.5.3, torch 2.6.0+cu124, onnxruntime 1.30.0).

Data: Lee2019-MI and Cho2017, first downloaded 15 Nov 2026 16:48–20:28 UTC (file timestamps), after Amendment 2 was public
(16:37 UTC). Files live in `C:\Users\Yu Gao\mne_data` (MOABB 1.5.0 had first written them to a mis-sanitised `C-\...` folder
inside the repository; they were moved, not re-downloaded). Lee2019-MI subject 1 was also fetched once from the NEMAR mirror
(nm000338) by MOABB 1.7.2's prefetch; its SHA-256 equals the GigaDB copy and the NEMAR provenance manifest.

**Set `MOABB_DOWNLOAD_PROVIDER=upstream`** before any run that reads raw data, or MOABB 1.7.2 prefetches every subject from
NEMAR again. `tools/run_confirmatory.ps1` sets it.

Caches (built 17 Nov 2026 with MOABB 1.7.2): `results/confirmatory/cache_{Lee2019_MI,Cho2017}_1-40Hz_250Hz.npz`.
Lee2019-MI: 54 subjects, 200 trials each (100 per session), none excluded. Cho2017: 52 subjects, 200 (49) or 240 (3) trials, none excluded.
Both balanced 50/50 per subject. MOABB notes edge effects for Cho2017, whose trials are distributed epoched and are stacked
with a zero buffer before filtering; this applies identically to all models and recipes.

Run everything in the registered order:  `powershell -ExecutionPolicy Bypass -File tools\run_confirmatory.ps1`
