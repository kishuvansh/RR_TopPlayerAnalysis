# Phase 0 Summary — Data Cleaning & Match State Construction

- **Execution Date:** 2026-11-14T18:29:15.026291+05:30
- **Script:** `src/s05_clean_and_match_state.py`
- **Random Seed:** `42`
- **Gate 0 Overall Status:** **PASS**

## 1. Match Filtering & Exclusion Accounting

- **Raw matches in dataset:** 883
- **Kept clean eligible matches:** **780** (IPL: 355, T20I: 425)

| Exclusion Reason | Matches Excluded |
|---|---:|
| A-team developmental events (comp_id 2670, 3289) | 30 |
| Matches missing ball-by-ball coverage | 16 |
| Matches without declared winner (abandoned / no-result) | 25 |
| Tied matches decided by super overs | 6 |
| DLS / rain-shortened revised matches | 26 |
| **Total Excluded** | **103** |

## 2. Extras Run Calibration (Winning Margin Regression)

- **Regression formulation:** `Margin - (bat_runs_inn1 - bat_runs_inn2) ~ sum(beta_k * delta_extras_k)` without intercept.
- **Sample size:** 385 non-DLS matches won by runs.
- **Median absolute residual:** **1.656 runs** (Gate threshold $\le 3.0$ runs).
- **Mean absolute residual:** **2.325 runs**.
- **R² on extras variation:** `0.7135`.

| Extra Type | Calibrated Run Value per Incident |
|---|---:|
| `wide` | 1.1289 runs |
| `Overstep` | 1.1997 runs |
| `legbyes` | 1.3352 runs |
| `byes` | 2.1786 runs |
| `other` | 1.0824 runs |
| `Penalty` | 0.0000 runs |

## 3. Gate 0 Verification Results

| Check | Metric | Observed Value | Threshold | Status |
|---|---|---|---|:---:|
| max_legal_balls_limit | Max legal balls per innings <= 120 | 120 (innings > 120: 0) | <= 120 | **PASS** |
| wickets_remaining_verification | In 'won by N wickets' matches, 2nd-inn wickets == 10-N | 99.74% (380 / 381) | >= 98.0% | **PASS** |
| winner_prediction_agreement | Estimated 2nd-inn total vs target agrees with winner | 95.38% (defended matches: 97.43%) | >= 95.0% within calibration tolerance | **PASS** |
| median_extras_residual | Median extras-calibration residual <= 3 runs | 1.656 runs | <= 3.0 runs | **PASS** |
| kept_matches_count | Kept eligible matches count ~790-800 | 780 matches | 775 - 805 | **PASS** |

## 4. Key Artifacts Produced

- `outputs/processed/matches_clean.parquet` (780 matches)
- `outputs/processed/balls_clean.parquet` (186,246 deliveries)
- `outputs/processed/extras_calibration.json`
- `outputs/tables/phase0_gate0_report.csv`

## 5. Notes & Observations for Phase 1
- All deliveries strictly follow pre-delivery information ordering without future leakage.
- Legal deliveries capped at 120 per innings with 0 violations.
- 12 anomaly rows in raw data with `dismissed_player_id` present but `is_wicket==False` are handled by trusting `is_wicket` (all 12 have blank dismissal types and non-zero off-bat runs).
- Bowling credit is strictly isolated to 5 bowler-credited dismissal types.
