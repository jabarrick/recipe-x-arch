# S8 run record (2026-11-10)

Input: `results/pilot_binary/pilot_binary_s8.csv` (binary pilot, 105 subjects, 4 × 4, 5-fold, seed 0; pilot code at 0eff2c8).
Script: `analysis/s8_interval_simulation.py` as frozen at f3fc1449 (byte-identical), defaults n_sims = 2000, seed = 0.

| Run | Files | Status |
|---|---|---|
| **Primary** — `--cov arch --n-conf 200 --n-subjects 106 193` | `S8_decision.json`, `S8_scenario_grid.csv`, `S8_pair_sd_decomposition.csv`, `S8_halfwidth_by_pair.csv`, `S8_summary.md`, `S8_outcome_probabilities.png`, `S8_cov_conf.csv` | Decides M1′. `--cov arch` is mandated by decision_rules §7 step 4: the full 16 × 16 covariance had 10 of 16 eigenvalues clipped (> 8). |
| Comparison — full covariance | `*_fullcov_comparison.*` | Reported, not used; superseded by §7 step 4. |
| Sensitivity — n_conf = 400, G = 0, n = 106 | `S8_scenario_grid_nconf400_sensitivity.csv` | Reported, not used; n_conf = 200 is the frozen value ([TO VERIFY] trial count). |

Decision quantity (decision_rules §4): P(B2 | G = 0, D ∈ {1,2,3}, n = 106) = **0.4962** (Monte Carlo SE 0.0053; 0.72 SE below the 0.5 threshold).
