# SupeRR Selector Season 2 – Round 1
## Pressure Performance Analysis in T20 Cricket (IPL & T20I, 2022–2026)

This repository contains the complete reproducible data science pipeline, evidence pack, interactive dashboard, and methodology report for the **SupeRR Selector Season 2 (Round 1)** challenge.

The goal is to identify and rank the **10 best Batters** and **10 best Bowlers** under pressure using ball-by-ball IPL and men's T20 International data from 2022 onwards.

---

## 🏆 Final Official Rankings

### Top 10 Batters Under Pressure
| Rank | Player Name | Country | Key Franchise / Team | Pressure Balls | Matches | Composite Score | 90% Rank Interval | Top-10 Prob |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **Rinku Singh** | India | Kolkata Knight Riders | 147 | 17 | **2.573** | [1.0, 72.3] | 81.0% |
| **2** | **Glenn Maxwell** | Australia | RCB, Punjab Kings | 203 | 16 | **1.874** | [1.0, 74.2] | 76.0% |
| **3** | **Krunal Pandya** | India | LSG, RCB | 157 | 15 | **1.348** | [2.0, 84.0] | 56.0% |
| **4** | **Rashid Khan** | Afghanistan | Gujarat Titans | 98 | 17 | **1.150** | [1.0, 76.1] | 43.0% |
| **5** | **Tim David** | Australia | Mumbai Indians, RCB | 317 | 29 | **1.084** | [2.0, 45.0] | 55.5% |
| **6** | **Suryakumar Yadav** | India | Mumbai Indians | 485 | 32 | **1.082** | [2.0, 31.0] | 49.5% |
| **7** | **Harry Brook** | England | Sunrisers Hyderabad, DC | 140 | 15 | **0.971** | [2.0, 85.0] | 47.0% |
| **8** | **David Miller** | South Africa | Gujarat Titans, LSG, DC | 259 | 24 | **0.945** | [2.0, 52.1] | 43.0% |
| **9** | **Sam Curran** | England | PBKS, CSK, RR | 210 | 16 | **0.897** | [4.0, 79.0] | 31.5% |
| **10** | **Jos Buttler** | England | Rajasthan Royals, GT | 352 | 31 | **0.740** | [5.0, 46.0] | 26.5% |

### Top 10 Bowlers Under Pressure
| Rank | Player Name | Country | Key Franchise / Team | Pressure Balls | Matches | Composite Score | 90% Rank Interval | Top-10 Prob |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **Josh Hazlewood** | Australia | Royal Challengers Bengaluru | 296 | 27 | **2.206** | [1.0, 19.1] | 86.0% |
| **2** | **Jasprit Bumrah** | India | Mumbai Indians | 264 | 23 | **1.504** | [1.0, 15.0] | 86.0% |
| **3** | **Rashid Khan** | Afghanistan | Gujarat Titans | 304 | 28 | **1.481** | [1.0, 20.0] | 72.0% |
| **4** | **Varun Chakravarthy** | India | Kolkata Knight Riders | 297 | 25 | **1.275** | [2.0, 25.0] | 69.0% |
| **5** | **Yuzvendra Chahal** | India | Rajasthan Royals, PBKS | 323 | 29 | **1.183** | [1.0, 36.0] | 56.5% |
| **6** | **Lockie Ferguson** | New Zealand | GT, KKR, PBKS, RCB | 206 | 20 | **1.114** | [3.0, 41.1] | 40.5% |
| **7** | **Matheesha Pathirana** | Sri Lanka | Chennai Super Kings, KKR | 215 | 16 | **1.114** | [1.0, 78.0] | 47.5% |
| **8** | **Gudakesh Motie** | West Indies | West Indies National Team | 194 | 16 | **1.099** | [4.0, 85.0] | 37.0% |
| **9** | **Krunal Pandya** | India | LSG, RCB | 239 | 22 | **1.005** | [3.0, 32.0] | 43.5% |
| **10** | **Harshit Rana** | India | Kolkata Knight Riders | 190 | 16 | **0.997** | [2.0, 84.0] | 39.0% |

---

## 🔬 Methodology Overview

1. **Phase 0 — Data Standardization & Margin Calibration**:
   - Filtered **780 clean regulation T20 matches** (IPL + T20Is, excluding rain-affected/revised DLS, super overs, and developmental A-team matches).
   - Reconstructed unrecorded extra runs via match-margin linear regression on 385 defended matches (median absolute residual: 1.656 runs).
2. **Phase 1 — Win Probability Model**:
   - Trained dual-innings logistic regressions (5-fold GroupKFold by match) incorporating run rate differentials, resources remaining (balls, wickets), and competition tier baselines.
   - Overall Brier score: **0.1765** (Innings 2: **0.1285**, 2026 holdout: **0.1264**).
3. **Phase 2 — Leverage Index & Liveness Gate**:
   - Simulates win probability transitions across 7 discrete delivery outcomes.
   - **Liveness Gate**: Deliveries in non-competitive states ($WP < 0.05$ or $WP > 0.95$) are excluded from pressure tiers to eliminate garbage-time noise.
   - **Early-Innings Rule**: Excludes first 12 legal balls of each innings to avoid small-sample current run rate distortions.
   - **HIGH pressure tier**: Top 20% of Leverage Index among live states ($LI \ge 1.5385$).
   - **ORDINARY baseline**: Bottom 50% of Leverage Index among live states ($LI < 0.9404$).
4. **Phase 3 — Player Ratings (Ridge Regression)**:
   - Simultaneous estimation of batter and bowler fixed effects on Win Probability Added ($\Delta WP$), controlling for innings, batting order band, pace vs spin, competition tier, and venue par.
   - Ridge regularization tuned via cross-validation ($\alpha = 300$ for batters, $\alpha = 500$ for bowlers).
5. **Phase 4 & 5 — Multi-Dimensional Composite & Uncertainty**:
   - $\text{Composite} = 0.50 \times \text{Pressure Rating} + 0.25 \times \text{Consistency} + 0.25 \times \text{Resilience}$
   - Where **Resilience** $= \text{Pressure Rating} - \text{Ordinary Rating}$ (performance uplift under pressure relative to baseline).
   - 200 match-level block bootstrap iterations yielding 90% rank intervals and top-10 inclusion probabilities.

---

## 📁 Repository Structure

```
.
├── README.md                                 # Project overview and executive summary
├── .gitignore                                # Optimized git exclusion rules
├── Round 1 Problem Statement _ SupeRR S2.docx# Competition problem statement
├── dataset_eda_overview.ipynb                # Exploratory Data Analysis notebook
├── notebook.py                               # Diagnostic script
└── SupeRR_Round1_Work/                       # Core analysis workspace
    ├── SupeRR_S2_Round1_Methodology_Report.docx # Official 12-page methodology report
    ├── config/                               # Central configuration (config.yaml)
    ├── docs/                                 # Complete documentation & audit trail
    │   ├── CCPI_spec.md                      # Specification for CCPI index
    │   ├── DECISIONS.md                      # D01–D18 architectural decision log
    │   ├── RUN_LOG.md                        # Phase execution logs
    │   ├── WEAKNESSES.md                     # Known limitations & sensitivity notes
    │   └── PHASE_*_SUMMARY.md                # Phase-by-phase summary reports
    ├── handoff/                              # Authoritative delivery artifacts
    │   ├── TOP10_BATTERS.csv                 # Final ranked batter list with CIs
    │   ├── TOP10_BOWLERS.csv                 # Final ranked bowler list with CIs
    │   ├── FINAL_NAMES.md                    # Official player name confirmation
    │   ├── WEAKNESSES.md                     # Explicit modeling weaknesses
    │   └── REPRODUCE.md                      # Step-by-step reproduction instructions
    ├── src/                                  # Pipeline execution scripts
    │   ├── s05_clean_and_match_state.py      # Phase 0: Data cleaning & margin calibration
    │   ├── s05b_integrity_checks.py          # Phase 0b: Masters & integrity audit
    │   ├── s06_win_probability_model.py      # Phase 1: Win probability model training
    │   ├── s07_pressure_definitions.py       # Phase 2: LI simulation & CCPI calibration
    │   ├── s08_player_ratings.py             # Phase 3: Ridge ratings & diagnostics
    │   ├── s09_phase4_phase5_master.py       # Phase 4/5: Master bootstrap & evaluation
    │   ├── s10_redefine_ordinary_and_update.py # Production update: live ordinary baseline
    │   ├── s11_final_validation.py           # Verification & data integrity check
    │   └── s12_dashboard_export.py           # Export clean metrics for Streamlit UI
    ├── dashboard/                            # Interactive Streamlit Web Application
    │   ├── app.py                            # Streamlit main entrypoint
    │   ├── requirements.txt                  # Dashboard dependencies
    │   ├── views/                            # 9 interactive tabs (Rankings, WPA, etc.)
    │   ├── components/                       # UI cards, plots, and design tokens
    │   └── screenshots/                      # Pre-rendered visual dashboard preview
    ├── report_builder/                       # Automated Word document generator
    └── outputs/
        ├── figures/                          # Calibration, LI curve, and rank plots
        ├── tables/                           # 40+ audit, diagnostic, and result CSVs
        └── processed/                        # Cleaned match and player master tables
```

---

## ⚡ How to Reproduce

### 1. Requirements
Ensure Python 3.10+ is installed:
```bash
pip install pandas numpy scipy scikit-learn pyyaml pyarrow matplotlib
```

### 2. Place Raw Data
Download the official organizer data pack and place the 6 CSV files in:
`SupeRR Selector Season 2 | Data Pack/`
- `ball_by_ball.csv`
- `matches.csv`
- `competitions.csv`
- `playing_eleven.csv`
- `teams.csv`
- `competition_team_player_mapping.csv`

### 3. Run Pipeline Scripts
Execute each numbered stage sequentially:
```bash
cd SupeRR_Round1_Work

# Phase 0: Data cleaning & calibration
python3 src/s05_clean_and_match_state.py

# Phase 0b: Integrity checks & master files
python3 src/s05b_integrity_checks.py

# Phase 1: Win probability modeling
python3 src/s06_win_probability_model.py

# Phase 2: Pressure definitions & leverage index
python3 src/s07_pressure_definitions.py

# Phase 3/4/5: Master rating, bootstrap & final updates
python3 src/s09_phase4_phase5_master.py
python3 src/s10_redefine_ordinary_and_update.py
```

All final CSVs will be regenerated deterministically into `handoff/` and `outputs/tables/`.

---

## 📊 Interactive Dashboard

To launch the multi-tab interactive analysis dashboard locally:
```bash
cd SupeRR_Round1_Work/dashboard
pip install -r requirements.txt
streamlit run app.py
```
Explore player radar profiles, live Leverage Index curves, win probability transitions, and bootstrap distribution charts.
