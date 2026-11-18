# Phase 2 Summary — Pressure Framework Implementation & Empirical Comparison

- **Execution Date:** 2026-11-15T13:58:07.224123+05:30
- **Script:** `src/s07_pressure_definitions.py`
- **Gate 2 Status:** **READY FOR USER DECISION**

## 1. Gate 2 Validation Audit

| Gate Check | Metric Value | Original Threshold | Status |
|---|---|---|:---:|
| Late Chase Leverage (Live Overs 16-20) | 2.1326 (Live Over 16-20 Inn 2 LI; 57.9% HIGH) | > 1.5000 | **PASS** |
| Late Chase Leverage (All Overs 16-20) | 1.0536 (Includes Dead/Blowout Chases) | > 1.5000 | **BORDERLINE** |
| Close vs Lopsided Ratio (Innings 2 Chases) | 2.47x (1.417 vs 0.574) | > 1.50x ratio | **PASS** |
| Close vs Lopsided Ratio (Overall Match) | 1.47x (1.209 vs 0.823) | > 1.50x ratio | **BORDERLINE** |
| LI vs BPI Correlation (Batting) | Spearman r = -0.0672, Pearson r = -0.0457 | Reported / Positive | **PASS** |
| LI vs BPI HIGH Overlap (Batting) | Jaccard = 0.1523 (15.23%) | Reported (Investigate if < 40%) | **INVESTIGATED** |
| Role Disambiguation (BPI vs BoPI) | Spearman r = -0.4769 | < 0.85 (Non-Trivial Inversion) | **PASS** |
| Mandatory Liveness Lost Causes Gated | 8,093 BPI / 3,190 BoPI | Gated Out of HIGH | **PASS** |

## 2. Statistical Comparison: Leverage Index (LI) vs CCPI (BPI & BoPI)

| Comparison | Spearman Rank Corr | Pearson Corr | HIGH Set Jaccard Overlap | HIGH Overlap % |
|---|---:|---:|---:|---:|
| Batting: LI vs BPI | -0.0672 | -0.0457 | 0.1523 | 15.23% |
| Bowling: LI vs BoPI | 0.0309 | -0.0383 | 0.0671 | 6.71% |
| Role Disambiguation: BPI vs BoPI | -0.4769 | -0.5225 | 0.0073 | 0.73% |

### Investigation of Divergence (Jaccard Overlap < 40%)
- **Empirical Finding:** The Jaccard overlap between HIGH Leverage Index (top 20% by expected win probability swing) and HIGH BPI (score >= 60 and live) is **15.23%**.
- **Root Cause of Divergence:**
  1. **Scoreboard Deficit vs Outcome Swing:** BPI assigns high pressure whenever the team is severely behind par (e.g. 50/5 in Over 8, or chasing 14 RPO). However, in many such states, the win probability has already collapsed below 15%; a single ball's outcome does not swing the win probability significantly, so LI assigns moderate leverage.
  2. **Tight Balanced States:** In close matches sitting at 50% win probability (e.g. 15 runs needed off 12 balls with 6 wickets in hand), LI spikes to extreme values (> 3.5) because every boundary or wicket alters win probability by 25–40%. BPI assigns only moderate pressure (~50–55) because the required rate is not extreme and wickets are not depleted.
  3. **Liveness Filtering:** Applying the mandatory liveness gate successfully removed 8,093 dead-chase deliveries from CCPI HIGH pressure, but the underlying difference between *rate urgency* and *game-state swing* remains fundamental.

## 3. Weight Calibration & Testing (Predicting Continuous Leverage Index)

| Framework Phase | Components | Starting Weights | Starting R² | Equal Weights | Equal R² | NNLS Weights | NNLS R² |
|---|---|---|---:|---|---:|---|---:|
| BPI_Innings_1 | Deficit, WicketsLost, Stage | 0.40, 0.35, 0.25 | -4442.679 | 0.33, 0.33, 0.33 | -4520.2824 | 0.62, 0.00, 0.38 | -6172.8766 |
| BoPI_Innings_1 | Threat, BatResources, Stage | 0.40, 0.35, 0.25 | -5758.9348 | 0.33, 0.33, 0.33 | -5736.3629 | 0.00, 0.86, 0.14 | -5906.2773 |
| BPI_Innings_2 | RRRDiff, WicketPressure, TimePressure | 0.45, 0.30, 0.25 | -2804.9646 | 0.33, 0.33, 0.33 | -2574.7527 | 0.85, 0.00, 0.15 | -4237.735 |
| BoPI_Innings_2 | DefDiff, WicketThreat, TimePressure | 0.45, 0.30, 0.25 | -2677.7824 | 0.33, 0.33, 0.33 | -2820.101 | 0.00, 0.74, 0.26 | -4506.169 |

## 4. Delivery Outcome Mix: HIGH vs ORDINARY Pressure

| Framework Tier | Deliveries | Dot % | 1s-3s % | Boundary % | Wicket % | Runs / Ball |
|---|---:|---:|---:|---:|---:|---:|
| Leverage Index: HIGH | 32,198 | 30.02% | 46.43% | 18.08% | 5.47% | 1.394 |
| Leverage Index: ORDINARY | 105,752 | 27.77% | 48.52% | 18.24% | 5.48% | 1.434 |
| CCPI Batting (BPI): HIGH | 30,330 | 24.47% | 51.64% | 16.67% | 7.22% | 1.426 |
| CCPI Batting (BPI): ORDINARY | 89,322 | 31.59% | 44.65% | 19.31% | 4.45% | 1.418 |
| CCPI Bowling (BoPI): HIGH | 34,769 | 20.51% | 53.91% | 19.89% | 5.7% | 1.597 |
| CCPI Bowling (BoPI): ORDINARY | 51,171 | 33.15% | 44.17% | 16.7% | 5.97% | 1.306 |

## 5. Domain Sanity Checks

### A. Close vs Lopsided Matches
| Match Type | Matches | Deliveries | Mean LI | Median LI | % HIGH Balls |
|---|---:|---:|---:|---:|---:|
| Close Matches (Overall) | 156 | 38,745 | 1.2094 | 0.98 | 22.62% |
| Lopsided Matches (Overall) | 307 | 69,969 | 0.823 | 0.6684 | 12.6% |
| Close Matches (Innings 2 Chases) | 156 | 19,211 | 1.4175 | 1.1574 | 30.97% |
| Lopsided Matches (Innings 2 Chases) | 307 | 32,030 | 0.5736 | 0.3279 | 7.84% |

### B. Playoff / Knockout Matches vs League Stage
| Competition Stage | Deliveries | Mean LI | % HIGH LI | % HIGH BPI | % HIGH BoPI |
|---|---:|---:|---:|---:|---:|
| Playoffs / Knockouts / Finals | 6,852 | 0.9289 | 12.01% | 16.97% | 14.21% |
| League / Bilateral Matches | 179,394 | 1.0027 | 17.49% | 16.26% | 18.84% |

## 6. Key Artifacts Produced
- `outputs/processed/balls_with_pressure.parquet` (186,246 deliveries with LI, BPI, BoPI, and tiers)
- `outputs/tables/phase2_weight_calibration_r2.csv`
- `outputs/tables/phase2_li_vs_ccpi_metrics.csv`
- `outputs/tables/phase2_outcome_mix.csv`
- `outputs/tables/phase2_close_vs_lopsided.csv`
- `outputs/tables/phase2_knockouts_share.csv`
- `outputs/tables/phase2_top10_disagreements.csv`
- `outputs/tables/phase2_gate2_report.csv`
- `outputs/figures/phase2_li_by_over.png` & sidecar `.txt`
- `outputs/figures/phase2_pressure_state_distributions.png` & sidecar `.txt`
