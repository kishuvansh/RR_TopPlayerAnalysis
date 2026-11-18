# Pipeline Run Log

This log records execution history, input/output row counts, execution duration, and key validation metrics for every script in `src/`.

| Timestamp (ISO) | Script | Rows In | Rows Out | Duration | Key Numbers / Metrics | Warnings / Notes |
|---|---|---|---|---|---|---|
| 2026-11-14T18:27:24.298062+05:30 | `src/s05_clean_and_match_state.py` | 883 matches / 200,999 balls | 786 matches / 187,420 balls | 0.88s | Gate 0: FAIL, Kept matches: 786, Median extra residual: 1.729 runs | None |
| 2026-11-14T18:29:15.026638+05:30 | `src/s05_clean_and_match_state.py` | 883 matches / 200,999 balls | 780 matches / 186,246 balls | 0.67s | Gate 0: PASS, Kept matches: 780, Median extra residual: 1.656 runs | None |
| 2026-11-14T18:46:16.020357+05:30 | `src/s05b_integrity_checks.py` | 780 matches / 186,246 balls | 780 matches / 186,246 balls | 1.41s | Gate 0 Re-Verification: PASS, 0 matches/balls removed, players_master: 984 | None |
| 2026-11-15T13:32:21.062051+05:30 | `src/s06_win_probability_model.py` | 186,246 balls | 186,246 balls | 5.2s | Gate 1: PASS, Brier: 0.1765 (naive 0.2499), Max calib error: 0.0385 | None |
| 2026-11-15T13:37:35.411345+05:30 | `s07_pressure_definitions.py` | 186,246 | 186,246 | LI normalized mean=1.0000; HIGH overlap: BPI 16.01%, BoPI 7.10%; Gate 2 READY | None |
| 2026-11-15T13:41:13.030698+05:30 | `s07_pressure_definitions.py` | 186,246 | 186,246 | LI normalized mean=1.0000; HIGH overlap: BPI 16.01%, BoPI 7.10%; Gate 2 READY | None |
| 2026-11-15T13:58:07.225217+05:30 | `s07_pressure_definitions.py` | 186,246 | 186,246 | LI normalized mean=1.0000; HIGH overlap: BPI 15.23%, BoPI 6.71%; Gate 2 READY | None |
| 2026-11-15T14:10:26.401802+05:30 | `s08_player_ratings.py` | 186,246 | 85 bats, 74 bowls | Ridge optimal alpha: bat=300.0, bowl=500.0; zero volume bias (r=0.109); Phase 3 COMPLETE | None |
| 2026-11-18T10:36:39.042626+05:30 | `s09_phase4_phase5_master.py` | 186,246 | 200 boot resamples, split-half r_bat=-0.053, r_bowl=0.236; TOP10 CSVs written | None |
