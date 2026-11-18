# Known Dataset & Analytical Weaknesses Log

This document records methodological limitations, data-collection gaps, and statistical uncertainties identified across the project.

---

### 1. Same-Country Duplicate Player IDs in Developmental Tournaments
- **Issue**: Three player names in Associate / developmental fixtures appear under two separate IDs within the same country:
  - `Basil Hameed` (United Arab Emirates): ID `8638` (9 clean matches) vs ID `39444` (3 matches in raw data, exclusively in excluded developmental tournament `comp_id 2670`).
  - `Muhammad Farooq` (United Arab Emirates): ID `27373` (1 clean match) vs ID `28978` (squad member in raw mapping, 0 matches bowled/batted).
  - `Bashir Ahmad` (Afghanistan): ID `29833` (1 clean match) vs ID `39479` (squad member in raw mapping, 0 matches bowled/batted).
  - `Mohammad Ali` (Pakistan): ID `20229` (2 clean matches) vs ID `3083` (squad member in raw mapping, 0 matches bowled/batted).
- **Hypothesis**: These same-country pairs are likely identical human individuals who were assigned new secondary IDs by the data provider when registering squads for developmental tournaments (e.g. Emerging Teams Asia Cup).
- **Treatment & Impact**: Following Decision D07, IDs were **NOT merged**. Because the secondary IDs (39444, 28978, 39479, 3083) have **zero active deliveries in `balls_clean`**, no performance metrics or sample sizes are split or corrupted in the clean modeling dataset.

---

### 2. Unrecorded Extras Run Amounts
- **Issue**: The provider records `runs` off the bat, but run amounts for extras (`wide`, `Overstep`, `byes`, `legbyes`) are unrecorded (`extras1_runs` is a category identifier enum).
- **Mitigation & Uncertainty**: Estimated via linear regression of match victory margins on extras differentials ($\text{median absolute residual} = 1.656$ runs).
- **Limitation**: While overall targets and team totals are accurate to within $\pm 1.66$ runs, razor-thin match finishes ($\le 3$ runs margin) have occasional misaligned estimated totals relative to the actual winner.

---

### 3. Structural Competition Differences (IPL vs T20I Tiers)
- **Issue**: IPL features significantly higher scoring rates (8.81 RPO, 20.36% boundaries) than Full Member T20Is (8.11 RPO, 17.72% boundaries) and Associate fixtures (7.22 RPO, 14.23% boundaries).
- **Mitigation**: Addressed via `comp_tier` stratification and mandatory control dummy variables in win-probability and ridge regression models.
