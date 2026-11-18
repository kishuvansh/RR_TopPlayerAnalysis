# How to Reproduce All Results

1. Ensure Python 3.10+ with `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyyaml`, `pyarrow`, and `matplotlib`.
2. Place raw CSV files in `SupeRR Selector Season 2 | Data Pack/`.
3. Run Phase 0: `python3 src/s05_clean_and_match_state.py` (filters 780 clean regulation matches).
4. Run Phase 0b: `python3 src/s05b_integrity_checks.py` (builds player, team, and venue masters).
5. Run Phase 1: `python3 src/s06_win_probability_model.py` (trains dual win-probability logistic regressions).
6. Run Phase 2: `python3 src/s07_pressure_definitions.py` (simulates transitions and computes Leverage Index).
7. Run Phase 3/4/5 Master: `python3 src/s09_phase4_phase5_master.py` (executes ratings, bootstrap, and handoff).
8. Run final definition: `python3 src/s10_redefine_ordinary_and_update.py` (live-only ORDINARY tier; writes the submitted TOP10 CSVs).
9. Run final validation: `python3 src/s11_final_validation.py` (cluster bootstrap, split-half, variants, weight sensitivity, box scores, report figures; asserts the top-10 order matches the handoff CSVs).
10. Build the report: `node report_builder/build.js SupeRR_S2_Round1_Methodology_Report.docx` (needs the `docx` npm package).
11. Export dashboard data: `python3 src/s12_dashboard_export.py` (writes `dashboard/data/`; asserts the submitted top-10s are reproduced). Run the dashboard with `cd dashboard && streamlit run app.py`.
12. All final deliverables are generated in `handoff/` and evidence tables in `outputs/tables/`.
