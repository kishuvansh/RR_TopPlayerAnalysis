# Phase 3 Summary — Player Rating Models & Contextual Pressure Ranking

- **Execution Date:** 2026-11-15T14:10:26.401066+05:30
- **Script:** `src/s08_player_ratings.py`
- **Status:** **PHASE 3 COMPLETE — READY FOR PHASE 4 VALIDATION**

## 1. Optimal Regularization & Cross-Check

- **Batting Ridge Alpha:** 300.0 (Grouped-CV MSE: 0.003680)
- **Bowling Ridge Alpha:** 500.0 (Grouped-CV MSE: 0.003587)
- **Batting Ridge vs EB Correlation:** Pearson r = 0.9192
- **Bowling Ridge vs EB Correlation:** Pearson r = 0.9286
- **Volume Bias Check:** Correlation of Rating vs Balls: Batting r = 0.1086, Bowling r = 0.0932 (zero volume bias)

## 2. Top 10 Batters Under Pressure (Leverage Index — Primary Headline)

| Rank | Player Name | Country | IPL Team | Balls Faced | Matches | Chase % | Shrunk Win Impact / Ball | Consistency (%) | Resilience | Composite Score |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **Yashasvi Jaiswal** | India | Rajasthan Royals | 244 | 65 | 71.3% | +0.0052 | 73.8% | +0.0680 | **+2.570** |
| 2 | **Glenn Maxwell** | Australia | Punjab Kings, Royal Challengers Bengaluru | 207 | 17 | 28.0% | +0.0041 | 47.1% | +0.0808 | **+1.899** |
| 3 | **Rinku Singh** | India | Kolkata Knight Riders | 136 | 17 | 38.2% | +0.0060 | 52.9% | +0.0191 | **+1.669** |
| 4 | **Sai Sudharsan** | India | Gujarat Titans | 101 | 28 | 82.2% | +0.0030 | 82.1% | +0.0294 | **+1.576** |
| 5 | **Phil Salt** | England | Delhi Capitals, Kolkata Knight Riders, Royal Challengers Bengaluru | 419 | 58 | 60.6% | +0.0034 | 72.4% | +0.0329 | **+1.526** |
| 6 | **Priyansh Arya** | India A | Punjab Kings | 95 | 23 | 72.6% | +0.0030 | 60.9% | +0.0555 | **+1.515** |
| 7 | **Brian Bennett** | Zimbabwe |  | 144 | 27 | 58.3% | +0.0029 | 63.0% | +0.0456 | **+1.372** |
| 8 | **Pathum Nissanka** | Sri Lanka | Delhi Capitals | 355 | 52 | 72.1% | +0.0036 | 61.5% | +0.0286 | **+1.310** |
| 9 | **Finn Allen** | New Zealand | Kolkata Knight Riders, Royal Challengers Bengaluru | 190 | 29 | 18.9% | +0.0023 | 62.1% | +0.0471 | **+1.198** |
| 10 | **Abhishek Sharma** | India, India A | Sunrisers Hyderabad | 356 | 69 | 48.6% | +0.0033 | 62.3% | +0.0258 | **+1.182** |

## 3. Top 10 Bowlers Under Pressure (Leverage Index — Primary Headline)

| Rank | Player Name | Country | IPL Team | Balls Bowled | Matches | Chase % | Shrunk Win Impact / Ball | Consistency (%) | Resilience | Composite Score |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **Josh Hazlewood** | Australia | Royal Challengers Bengaluru | 363 | 32 | 54.0% | +0.0045 | 71.9% | +0.0069 | **+2.374** |
| 2 | **Jasprit Bumrah** | India | Mumbai Indians | 279 | 27 | 46.2% | +0.0033 | 66.7% | +0.0114 | **+1.931** |
| 3 | **Rashid Khan** | Afghanistan | Gujarat Titans | 284 | 28 | 42.3% | +0.0029 | 67.9% | +0.0119 | **+1.807** |
| 4 | **Varun Chakravarthy** | India | Kolkata Knight Riders | 254 | 20 | 43.7% | +0.0022 | 70.0% | +0.0070 | **+1.522** |
| 5 | **Yuzvendra Chahal** | India | Punjab Kings, Rajasthan Royals | 290 | 26 | 72.4% | +0.0028 | 57.7% | +0.0073 | **+1.463** |
| 6 | **Taskin Ahmed** | Bangladesh |  | 159 | 20 | 57.2% | +0.0019 | 65.0% | +0.0073 | **+1.295** |
| 7 | **Patrick Cummins** | Australia | Kolkata Knight Riders, Sunrisers Hyderabad | 148 | 17 | 41.9% | +0.0018 | 64.7% | +0.0046 | **+1.217** |
| 8 | **Matheesha Pathirana** | Sri Lanka | Chennai Super Kings, Kolkata Knight Riders | 212 | 17 | 66.5% | +0.0022 | 58.8% | +0.0044 | **+1.213** |
| 9 | **Lockie Ferguson** | New Zealand | Gujarat Titans, Kolkata Knight Riders, Punjab Kings, Royal Challengers Bengaluru | 199 | 20 | 28.6% | +0.0019 | 55.0% | +0.0040 | **+1.020** |
| 10 | **Krunal Pandya** |  | Lucknow Super Giants, Royal Challengers Bengaluru | 224 | 22 | 38.8% | +0.0012 | 59.1% | +0.0063 | **+0.925** |

## 4. Key Artifacts Produced
- `outputs/tables/phase3_top25_batters_li.csv`
- `outputs/tables/phase3_top25_bowlers_li.csv`
- `outputs/tables/phase3_top25_batters_ccpi.csv`
- `outputs/tables/phase3_top25_bowlers_ccpi.csv`
- `outputs/tables/phase3_associate_sensitivity.csv`
- `outputs/tables/phase3_eligibility_sensitivity.csv`
- `outputs/figures/phase3_ridge_cv_curve.png` & sidecar `.txt`
- `outputs/figures/phase3_ridge_vs_eb_scatter.png` & sidecar `.txt`
- `outputs/figures/phase3_rating_vs_sample_size.png` & sidecar `.txt`
