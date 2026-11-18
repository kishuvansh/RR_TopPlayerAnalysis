# Phase 0b Summary — Data Integrity & Standardization Audit

- **Execution Date:** 2026-11-14T18:46:16.020026+05:30
- **Script:** `src/s05b_integrity_checks.py`
- **Gate 0 Re-Verification Status:** **PASS**

## 1. Key Audit Highlights & Actions Taken

- **Matches / Balls Removed:** **0 matches / 0 balls removed** during this audit. Dataset remains strictly **780 clean matches** and **186,246 deliveries**.
- **Player Master Table:** Built `outputs/processed/players_master.csv` containing all 984 eligible players with canonical names, IPL franchises, national teams, date ranges, and batting/bowling role hints.
- **Duplicate Player Names Disambiguated:** 7 names mapped to multiple IDs (e.g. Abdul Samad India/SRH vs Pakistan; Rashid Khan Afghanistan vs Nepal; Haider Ali Pakistan vs UAE). Logged in `player_duplicate_names.csv`; IDs kept strictly separate.
- **Batting Integrity Repaired:** Fixed missing `dismissed_player_id` for match 40709 ball 117 (Tilak Varma retired out, set to ID `7897`). Fixed 4 single-ball non-striker recorder typos.
- **Legality of Wide Dismissals Verified:** 38 wickets on wide balls are 100% compliant with cricket law (24 Stumpings, 14 Run Outs; 0 caught/bowled).
- **Standardization Completed:** Dismissal types normalized (`Obstracting` merged into `Obstructing`). Bowling types mapped to standard 12-category taxonomy. Real booleans, integer IDs, and numeric `match_year` / `season_year` generated.
- **Master Entity Tables:** Built `outputs/processed/team_master.csv` (38 teams) and `outputs/processed/venue_master.csv` (98 venues) with 0 duplicate or conflicting IDs.
- **Competition Comparability Quantified:** Tagged matches into 4 tiers. Confirmed significant scoring environment differences (IPL 8.81 RPO, 20.36% boundaries vs Full Member T20I 8.11 RPO vs Associate T20I 7.22 RPO); locked as mandatory model controls.

## 2. Competition Tier Comparability Table

| Competition Tier | Matches | Deliveries | Overall RPO | Powerplay RPO | Middle RPO | Death RPO | Wicket % | Boundary % | Avg 1st Inn Score | Chase Win % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IPL | 355 | 85,708 | 8.81 | 8.62 | 8.34 | 10.06 | 5.28% | 20.36% | 186.4 | 52.11% |
| T20I (FM v FM) | 375 | 88,891 | 8.11 | 7.81 | 7.8 | 9.21 | 5.83% | 17.72% | 172.8 | 50.13% |
| T20I (FM v Assoc) | 36 | 8,384 | 7.48 | 7.13 | 7.04 | 8.93 | 5.79% | 15.25% | 161.1 | 36.11% |
| T20I (Assoc v Assoc) | 14 | 3,263 | 7.22 | 6.98 | 6.71 | 8.71 | 6.54% | 14.23% | 155.6 | 35.71% |

## 3. Gate 0 Re-Verification Metrics

| Gate 0 Check | Required Threshold | Observed Value | Status |
|---|---|---|:---:|
| Max legal balls per innings | $\le 120$ | **120** (0 innings $> 120$) | **PASS** |
| 2nd-inn wickets in chase victories | $\ge 98.0\%$ | **99.74%** (380/381 matches) | **PASS** |
| Estimated 2nd-inn total vs target agreement | $\ge 95.0\%$ | **95.38%** (defended: 97.43%) | **PASS** |
| Median extras residual | $\le 3.0$ runs | **1.656 runs** | **PASS** |
| Kept matches count | $\sim 790-800$ (775-805) | **780 matches** | **PASS** |

## 4. Key Artifacts Produced
- `outputs/processed/balls_clean.parquet` (186,246 deliveries, standardized types and categories)
- `outputs/processed/matches_clean.parquet` (780 matches, comp_tier, match_year, season_year)
- `outputs/processed/players_master.csv` & `outputs/tables/players_master.csv` (984 players)
- `outputs/processed/team_master.csv` & `outputs/tables/team_master.csv` (38 teams)
- `outputs/processed/venue_master.csv` & `outputs/tables/venue_master.csv` (98 venues)
- `outputs/tables/integrity_report.csv` (audit actions and resolutions)
- `outputs/tables/player_duplicate_names.csv` (duplicate name disambiguation table)
- `outputs/tables/competition_tier_comparability.csv` (tier comparison metrics)
- `outputs/tables/missing_values_audit.csv` (feature missingness documentation)
