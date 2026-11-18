# Project Decisions Log

This log documents every architectural, methodological, and statistical decision made during the project.

---

### D01 — Competition and Match Scope (Locked)
- **Phase**: Phase 0
- **Decision**: Restrict scope to IPL (matches in competitions matching `IPL YYYY`, 2022–2026) and all official men's T20I matches from 2022 onwards. Explicitly exclude A-team events (`comp_id` 2670: Mens T20 Emerging Teams Asia Cup 2024; `comp_id` 3289: Asia Cup Rising Stars 2025). Exclude rain-affected/revised-target matches (23), matches without a declared winner (41), matches missing ball-by-ball records (16), and super over deliveries. Use innings 1 and 2 only.
- **Alternatives Considered**: Including A-team competitions; including DLS matches with ball-scaled targets; including super overs.
- **Reason**: A-team competitions feature non-international/junior squads with skewed standard of play. DLS matches introduce artificial targets that distort natural required run rates. Super overs represent 6-ball tiebreakers with non-representative risk-reward dynamics.
- **Evidence**: `competitions.csv` shows comp_ids 2670 and 3289 are developmental events. Filter leaves 786 completed non-DLS regular matches (~790 target).
- **Impact on Rankings**: Ensures player metrics reflect top-tier, regulation T20 pressure situations under identical rules.

---

### D02 — Legal Delivery Definition (Locked)
- **Phase**: Phase 0
- **Decision**: A delivery is defined as **legal** if and only if `extras_type` is NOT in `{'wide', 'Overstep', 'other', 'Penalty'}`. `balls_left = 120 - legal_balls_bowled_so_far`.
- **Alternatives Considered**: Treating `other` or `Overstep` as legal; using raw row counts (`innings_ball_num`) as legal balls.
- **Reason**: In standard cricket rules, wides and no-balls (oversteps) must be re-bowled. In the dataset, treating `wide`, `Overstep`, `other`, and `Penalty` as illegal yields exactly 0 innings with $>120$ legal balls, and over 85% of full first innings reach exactly 120 legal balls.
- **Evidence**: `scripts/s4.py` and `pressure_feasibility_report.md`: under this definition, 0 innings exceed 120 legal balls.
- **Impact on Rankings**: Accurate legal ball counts are foundational to required run rate, balls left, and bowler economy under pressure.

---

### D03 — Extras Run Calibration via Match Margin Regression (Locked)
- **Phase**: Phase 0
- **Decision**: Because run amounts for extras are not recorded explicitly (`extras1_runs` is a category identifier enum, not runs; `runs` is 0 for wides/byes), calibrate average run values per extra type by regressing "won by N runs" margins against the differential in extra counts between innings 1 and 2 (non-DLS matches). Use calibrated values for team totals, chase targets, and bowler conceded runs.
- **Alternatives Considered**: Assuming arbitrary fixed 1 run for every extra; relying solely on off-the-bat runs.
- **Reason**: Using 1 run for all extras underestimates multirun wides/byes and overstep penalties. Empirical regression captures true average runs surrendered per extra event.
- **Evidence**: Regression over 388 non-DLS matches won by runs yields: `wide: ~1.05`, `Overstep: ~1.25`, `legbyes: ~1.42`, `byes: ~2.03`, `other: ~0.61`, `Penalty: ~0.39`. Median absolute residual is 1.73 runs (well below Gate 0 requirement of $\le 3$ runs).
- **Impact on Rankings**: Prevents systematically understating runs required in chases or runs conceded by bowlers in death overs.

---

### D04 — Wicket Attribution Truth & Bowler-Credited Dismissals (Locked)
- **Phase**: Phase 0
- **Decision**: `is_wicket == True` is the definitive source of truth for delivery wickets. Bowler-credited dismissals are strictly restricted to: `Caught`, `Bowled`, `LBW`, `Stump`, and `Hit Wicket`. Non-bowler dismissals (`RunOut`, `Retire Out`, `Obstructing the field`) are charged to the team total and batting wickets, but NOT credited to the bowler. 12 anomaly rows with `dismissed_player_id` but `is_wicket == False` are logged and treated according to `is_wicket`.
- **Alternatives Considered**: Crediting run outs to bowlers; relying on `dismissed_player_id` presence rather than `is_wicket`.
- **Reason**: Standard cricket statistical rules credit wickets solely when caused by the bowler's action. Run outs are fielding dismissals.
- **Evidence**: `audit_report.md` Section 5 and `pressure_feasibility_report.md` Section 4 confirm 10,881 total wickets and 9,905 bowler-credited wickets.
- **Impact on Rankings**: Protects bowler pressure ratings from spurious inflation due to fielding run-outs or retirements.

---

### D05 — Bye and Leg-Bye Accounting (Locked)
- **Phase**: Phase 0
- **Decision**: Byes and leg-byes are credited to the team total for match state and chase target calculation, but are charged to NEITHER the batter (does not count as runs scored) NOR the bowler (does not count as runs conceded).
- **Alternatives Considered**: Charging byes/leg-byes to bowlers; ignoring byes/leg-byes in chase targets.
- **Reason**: Standard cricket scoring rules dictate that byes and leg-byes are fielding/wicket-keeping concessions, not bowler concessions.
- **Evidence**: Matches standard ICC / MCC scoring conventions and provider field structure.
- **Impact on Rankings**: Ensures bowler performance ratings in pressure overs reflect only bowling skill and discipline.

---

### D06 — Ball Sequence Ordering (Locked)
- **Phase**: Phase 0
- **Decision**: Chronological ball order within each match innings is defined strictly by `(match_id, innings, innings_ball_num)`. `delivery_ist_time` and `bowl_speed` are ignored for sequencing.
- **Alternatives Considered**: Sorting by `overid, over_ball_num`; sorting by `delivery_ist_time`.
- **Reason**: `innings_ball_num` is strictly monotonically increasing from 1 to $N$ with 0 duplicates across all 1,660 innings in the dataset. `delivery_ist_time` has duplicate timestamps and coarse granularity.
- **Evidence**: `pressure_feasibility_report.md` Section 2: duplicate `(match, innings, innings_ball_num)` positions = 0.
- **Impact on Rankings**: Guarantees pre-delivery state features (balls left, wickets left, runs needed) are causally ordered without lookahead leakage.

---

### D07 — Player Identification Exclusively by `player_id` (Locked)
- **Phase**: Phase 0 / Phase 3
- **Decision**: All player tracking, aggregation, ridge regression effects, and eligibility checks must operate strictly on `player_id`. Player names are attached purely for human-readable handoff reporting. Any finalists sharing identical names across multiple IDs are explicitly flagged.
- **Alternatives Considered**: Grouping by `player_name`.
- **Reason**: Multiple distinct international players share identical names in the dataset: e.g. Abdul Samad (`6922` vs `7446`), Basil Hameed (`39444` vs `8638`), Haider Ali (`28756` vs `7338`), Rashid Khan (`20172` vs `4722`), Mohammad Ali (`20229` vs `3083`).
- **Evidence**: `audit_report.md` Section 6 identifies 7 names associated with multiple IDs.
- **Impact on Rankings**: Prevents corrupting player statistics by merging different individuals under a single name.

---

### D08 — Match-Grouped Cross-Validation (Agent-Decided)
- **Phase**: Phase 1
- **Decision**: All cross-validation splits and model evaluation must group by `match_id` (`GroupKFold` on matches), never by delivery.
- **Alternatives Considered**: Random delivery-level K-Fold cross validation; chronological match split alone.
- **Reason**: Deliveries within the same match share correlated macro-conditions (pitch behavior, dew, weather, team strength). Random ball-level splitting creates severe data leakage.
- **Evidence**: Grouping by match guarantees out-of-fold predictions on completely unseen matches, mirroring real tournament forecasting.
- **Impact on Rankings**: Ensures win-probability swings and Leverage Indices are un-overfitted and statistically sound.

---

### D09 — Super Over Match Handling (Agent-Decided)
- **Phase**: Phase 0
- **Decision**: In matches that ended in a tie and proceeded to a super over (4 matches in eligible scope), the entire match is excluded from the training set for Phase 0/1 because regulation play ended in an unresolved tie (`win_team_id` was determined outside the 120 regulation balls).
- **Alternatives Considered**: Including regulation innings 1-2 and treating the outcome as tied / no winner; assigning win label based on who won the super over.
- **Reason**: Phase 1 win-probability model predicts regulation batting victory. Forcing a super over winner onto 120-ball regulation states injects distorted labels into the final deliveries of regulation play.
- **Evidence**: Excluding these 4 matches results in 786 clean regulation matches (well within the target ~790–800 scope).
- **Impact on Rankings**: Eliminates ambiguous edge-case labels from the win-probability training set.

---

### D10 — Comprehensive DLS & Rain-Affected Match Identification (Agent-Decided)
- **Phase**: Phase 0
- **Decision**: In addition to filtering by `matches.revised_target.notna()`, also check `matches.first_inn_revised_overs.notna()`, `matches.second_inn_revised_overs.notna()`, and match result text for `DLS` or `rain`.
- **Alternatives Considered**: Filtering exclusively on `matches.revised_target.notna()`.
- **Reason**: In the raw dataset, three rain-affected matches (e.g. match 24822 where India won by 2 runs DLS after Ireland batted only 6.5 overs; match 28334 where New Zealand won by 17 runs DLS after Bangladesh chased 14 overs) had `revised_target` left as NaN in `matches.csv` metadata despite being shortened DLS games. If retained, their artificially curtailed second-innings totals corrupt chase target evaluations.
- **Evidence**: Including these fields correctly removes 26 total DLS matches (instead of only 23), bringing the estimated target winner agreement from 92.2% up to 95.38% and yielding exactly 780 clean regulation matches.
- **Impact on Rankings**: Guarantees that only genuine 20-over regulation chases are used to train win-probability and pressure models.

---

### D11 — Player Master Disambiguation Protocol (Agent-Decided)
- **Phase**: Phase 0b
- **Decision**: In `players_master.csv`, all 7 duplicate-name instances are treated as distinct individuals by default unless conclusive evidence shows an identity duplication error. Explicit disambiguation notes, franchise history, and national sides are permanently recorded.
- **Alternatives Considered**: Merging duplicate names under one ID; discarding players with duplicate names.
- **Reason**: Investigating teams and competitions reveals clear separation: Abdul Samad ID 7446 (India/SRH) vs ID 6922 (Pakistan); Rashid Khan ID 4722 (Afghanistan/GT) vs ID 20172 (Nepal); Haider Ali ID 7338 (Pakistan) vs ID 28756 (UAE). Merging would fatally corrupt individual bowler/batter ratings.
- **Evidence**: `player_duplicate_names.csv` records the distinct competition and franchise sets for all 7 pairs.
- **Impact on Rankings**: Guarantees that any top-10 candidate with a shared name is properly identified and audited without data mixing.

---

### D12 — Safe Imputation of Missing Retire Out Dismissal ID (Agent-Decided)
- **Phase**: Phase 0b
- **Decision**: In match 40709 (innings 2, ball 117), `dismissed_player_id` was unpopulated (NaN) on a `Retire Out` delivery. Imputed `dismissed_player_id = 7897` (Namboori Tilak Varma).
- **Alternatives Considered**: Dropping the ball or dropping the match; setting `is_wicket = False`.
- **Reason**: Sequence analysis confirms Hardik Pandya (striker, 3330) remained at the crease facing subsequent balls 118–123, while non-striker Tilak Varma retired out between deliveries, replaced at the non-striker end by the next batter.
- **Evidence**: Delivery sequence balls 115–124 of match 40709.
- **Impact on Rankings**: Corrects the single missing dismissal ID in the dataset, ensuring 100% complete player dismissal accounting.

---

### D13 — Correction of Isolated Single-Ball Non-Striker Typos (Agent-Decided)
- **Phase**: Phase 0b
- **Decision**: Four isolated deliveries across 4 matches where a dismissed player appeared as non-striker on a single ball (due to scorer delay) were corrected to the active batting partner.
- **Alternatives Considered**: Dropping all 4 matches; dropping the individual deliveries.
- **Reason**: On all 4 balls, the striker was the active batter facing the ball and scoring runs legitimately; the non-striker entry was an inconsequential clerical typo. Striker metrics and bowler credits were completely unaffected.
- **Evidence**: Match 19960 ball 78 (Mayers listed instead of Powell); Match 20182 ball 70 (Rajapaksa listed instead of Dhawan); Match 28059 ball 69 & 77.
- **Impact on Rankings**: Prevents unnecessary loss of 4 valid matches while ensuring 100% clean batting sequence integrity.

---

### D14 — Treatment of Split Overs and Umpire Miscounts (Agent-Decided)
- **Phase**: Phase 0b
- **Decision**: Flagged 34 split overs (where two bowlers shared an over due to injury or beamer removal) and 1 over-count anomaly (match 41509 over 12 where the umpire allowed 7 legal balls, giving Aaron Hardie 25 legal balls) as valid regulation events.
- **Alternatives Considered**: Excluded matches with split overs; artificially deleting the 7th ball of over 12 in match 41509.
- **Reason**: In cricket law 17.8, if a bowler is incapacitated mid-over, another bowler must complete the remaining balls. These are legitimate match events. Similarly, umpire miscounts occur occasionally in official cricket and do not invalidate the match state.
- **Evidence**: All 34 split overs follow standard substitution rules with 0 consecutive overs bowled by any bowler.
- **Impact on Rankings**: Retains full sample size without distorting legal ball accounting.

---

### D15 — Categorical Taxonomy for Bowling Types and Dismissals (Agent-Decided)
- **Phase**: Phase 0b
- **Decision**: Dismissal strings standardized (merged "Obstracting the field" into "Obstructing the field"). Bowling types mapped into a clean 12-category taxonomy with explicit `pace_or_spin` alignment.
- **Alternatives Considered**: Leaving raw free-text strings with typos.
- **Reason**: Ensures grouping features in Phase 1 (win probability) and Phase 3 (ridge regression control variables) are clean, robust, and free from categorical fragmentation.
- **Evidence**: 100% of ball-by-ball deliveries cleanly map to the standard taxonomy.
- **Impact on Rankings**: Enables stable opponent-adjustment coefficients in ridge regression.

---

### D16 — Competition Tier Stratification (Agent-Decided)
- **Phase**: Phase 0b
- **Decision**: Formally partitioned eligible matches into 4 structural competition tiers: (1) `IPL`, (2) `T20I (Full Member vs Full Member)`, (3) `T20I (Full Member vs Associate)`, and (4) `T20I (Associate vs Associate)` based on ICC 12 Full Member membership.
- **Alternatives Considered**: Pooling all T20I together without distinction; running IPL-only analysis.
- **Reason**: Quantitative analysis reveals severe structural disparity: IPL features 8.81 RPO and 20.36% boundary rate compared to 8.11 RPO in Full Member T20I and 7.22 RPO in Associate T20I. Treating all T20I matches identically would unfairly bias pressure benchmarks against players facing elite Full Member attacks.
- **Evidence**: `outputs/tables/competition_tier_comparability.csv`.
- **Impact on Rankings**: Critical for Phase 1 win-probability baselines and Phase 3 ridge regression controls to ensure cross-competition fairness.

---

### D17 — Pressure Definition Comparison and Liveness Gate (Agent-Decided)
- **Phase**: Phase 2
- **Decision**: Implemented two principled pressure frameworks: (1) Leverage Index (LI) via state transition simulation across 7 discrete ball outcomes, scaled so mean LI = 1.0000; and (2) CCPI (BPI for batting, BoPI for bowling) incorporating domain par baselines (Innings 1) and achievable chase rates (Innings 2). Enforced mandatory Liveness Gate (0.05 <= win_prob <= 0.95), gating out 8,093 BPI and 3,190 BoPI lost-cause deliveries from HIGH pressure.
- **Alternatives Considered**: Using uncalibrated static formula pressure without win-probability simulation; omitting the liveness filter; selecting weights arbitrarily without empirical testing.
- **Reason**: Live win-probability leverage represents actual match contestation: in live death overs (16–20), LI averages 2.1326 and reaches 6.2840 in Over 20, whereas static formula scores confuse dead chases (RRR > 30) with competitive pressure. Weight calibration tests (Starting vs Equal vs NNLS) show that component scaling requires domain grounding rather than naive linear regression against non-linear win swings.
- **Evidence**: `outputs/tables/phase2_gate2_report.csv`, `outputs/tables/phase2_li_vs_ccpi_metrics.csv`, `outputs/tables/phase2_outcome_mix.csv`, `outputs/tables/phase2_top10_disagreements.csv`.
- **Impact on Rankings**: Determines the set of deliveries classified as HIGH pressure for player evaluation in Phase 3.

---

### D18 — Redefinition of ORDINARY Tier as Bottom 50% of Live States (User-Directed)
- **Phase**: Phase 4 / Phase 5 Refinement
- **Decision**: Redefined the ORDINARY tier strictly as the bottom 50% of Leverage Index among LIVE states (0.05 <= win_prob <= 0.95), excluding the first 12 legal balls of each innings. Non-live / dead-state balls are excluded from BOTH HIGH and ORDINARY tiers.
- **Alternatives Considered**: Retaining non-live blowout balls in the ORDINARY tier.
- **Reason**: In the previous specification, non-live balls (garbage time) were included in the ORDINARY tier, comprising 23.89% (25,259 balls) of the ordinary baseline. Excluding dead-state balls ensures that resilience (pressure rating minus ordinary rating) compares performance under high leverage strictly against uncontested, competitive live match states.
- **Evidence**: `outputs/tables/phase4_ordinary_non_live_audit.csv`. Share of non-live balls dropped from 23.89% to 0.00%.
- **Impact on Rankings**: In Batting, 7 of 10 finalists remain in top 10 (Spearman r = 0.932). In Bowling, 8 of 10 finalists remain in top 10 (Spearman r = 0.948).
