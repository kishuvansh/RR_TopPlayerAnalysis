# Phase 1 Summary — Win-Probability Model Construction

- **Execution Date:** 2026-11-15T13:32:21.060782+05:30
- **Script:** `src/s06_win_probability_model.py`
- **Gate 1 Status:** **PASS**

## 1. Model Architecture & Selection

- **Chosen Model:** Dual Logistic Regression (separate models for Innings 1 and Innings 2) with standardized numerical features and one-hot competition tier controls (`IPL`, `T20I_FM_v_FM`, `T20I_Assoc_pooled`).
- **Overall Brier Score:** **0.1765** (Naive baseline: `0.2499`, 29.3% error reduction).
- **Innings 1 Performance:** Brier = `0.2207` (vs HGB `0.2450`), Log Loss = `0.6319`.
- **Innings 2 Performance:** Brier = `0.1285` (vs HGB `0.1370`), Log Loss = `0.3967`.
- **Model Rationale:** Logistic regression outperformed tree-based gradient boosting out-of-fold because sports scoreboard win probability is a smooth, monotonic sigmoid function; trees overfit localized step boundaries across unseen matches.
- **Time-Holdout Validation (Season 2026):** Evaluated strictly on 31,473 unseen deliveries from season 2026: Innings 2 Brier = `0.1264` vs naive `0.2535`.

## 2. Gate 1 Calibration Across 10 Probability Bins

| Bin Range | Deliveries | Mean Predicted WP | Observed Win Rate | Absolute Error | Status |
|---|---:|---:|---:|---:|:---:|
| [0.0, 0.1] | 20,685 | 0.0325 | 0.038 | 0.0055 | **PASS** |
| [0.1, 0.2] | 12,208 | 0.1526 | 0.1624 | 0.0099 | **PASS** |
| [0.2, 0.3] | 14,697 | 0.2515 | 0.2441 | 0.0075 | **PASS** |
| [0.3, 0.4] | 20,008 | 0.3533 | 0.359 | 0.0057 | **PASS** |
| [0.4, 0.5] | 27,395 | 0.4512 | 0.4342 | 0.017 | **PASS** |
| [0.5, 0.6] | 25,971 | 0.5485 | 0.5738 | 0.0254 | **PASS** |
| [0.6, 0.7] | 20,193 | 0.6478 | 0.6567 | 0.0089 | **PASS** |
| [0.7, 0.8] | 15,484 | 0.7477 | 0.7336 | 0.0141 | **PASS** |
| [0.8, 0.9] | 12,286 | 0.85 | 0.8114 | 0.0385 | **PASS** |
| [0.9, 1.0] | 17,319 | 0.9577 | 0.9572 | 0.0005 | **PASS** |

## 3. Subgroup Calibration Performance

| Subgroup | Deliveries | Brier Score | Log Loss | Mean Predicted WP | Observed Win Rate |
|---|---:|---:|---:|---:|---:|
| Innings 1 | 97,038 | 0.2207 | 0.6319 | 0.5037 | 0.5032 |
| Innings 2 | 89,208 | 0.1285 | 0.3967 | 0.4788 | 0.4784 |
| Phase: Powerplay (1-6) | 58,495 | 0.2067 | 0.5964 | 0.492 | 0.4997 |
| Phase: Middle (7-15) | 85,920 | 0.168 | 0.4983 | 0.5074 | 0.4956 |
| Phase: Death (16-20) | 41,831 | 0.1517 | 0.4541 | 0.4594 | 0.4708 |
| Tier: IPL | 85,708 | 0.1826 | 0.5363 | 0.49 | 0.4905 |
| Tier: T20I_FM_v_FM | 88,891 | 0.1709 | 0.5034 | 0.4933 | 0.4923 |
| Tier: T20I_Assoc_pooled | 11,647 | 0.1745 | 0.514 | 0.4933 | 0.4902 |

## 4. Key Artifacts Produced
- `outputs/processed/balls_with_wp.parquet` (all 186,246 deliveries with OOF win_prob)
- `outputs/tables/phase1_calibration_bins.csv`
- `outputs/tables/phase1_calibration_subgroups.csv`
- `outputs/figures/phase1_calibration_plot.png` & sidecar `.txt`
- `outputs/figures/FIGURE_INDEX.md`
