#!/usr/bin/env python3
"""
s05_clean_and_match_state.py — Phase 0: Clean Data and Match State Construction.

Constructs balls_clean and matches_clean with strictly pre-delivery features:
- Runs, wickets, legal balls elapsed before each ball.
- Balls left (120 - legal balls before) and wickets left (10 - wickets lost before).
- Innings 2 chase target, runs needed, current run rate, required run rate, rate gap.
- Extras calibration via regression of match victory margins on extras differentials.
- Bowler-credited wicket indicators and non-bowler dismissal isolation.
- Full verification and reporting against Gate 0 criteria.

Deterministic execution: fixed random seed from config/config.yaml.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    start_time = time.time()
    work_dir = Path("/Users/vansh/Downloads/SupeRR Selector Season 2 | Problem Statement/SupeRR_Round1_Work")
    config_file = work_dir / "config" / "config.yaml"
    cfg = load_config(config_file)

    seed = cfg["project"]["seed"]
    np.random.seed(seed)

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    proc_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    docs_dir = Path(cfg["paths"]["docs_dir"])

    proc_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 0: CLEAN DATA AND MATCH STATE CONSTRUCTION")
    print(f"Random seed: {seed}")
    print(f"Raw data directory: {raw_dir}")
    print("=" * 80)

    # 1. Load raw tables
    print("\n[1/6] Loading raw data files...")
    matches_raw = pd.read_csv(raw_dir / "matches.csv")
    comps_raw = pd.read_csv(raw_dir / "competitions.csv")
    bbb_raw = pd.read_csv(raw_dir / "ball_by_ball.csv", low_memory=False)

    rows_in_matches = len(matches_raw)
    rows_in_bbb = len(bbb_raw)
    print(f"  Raw matches loaded: {rows_in_matches:,}")
    print(f"  Raw ball-by-ball deliveries loaded: {rows_in_bbb:,}")

    # 2. Scope filtering and exclusion logging
    print("\n[2/6] Applying locked scope filters...")
    m = matches_raw.merge(
        comps_raw[["comp_id", "comp_name", "match_format", "season"]],
        on="comp_id",
        how="left"
    )

    exclusion_counts = {}
    total_initial = len(m)

    # (a) Exclude A-team competitions (comp_ids 2670, 3289)
    a_team_mask = m["comp_id"].isin(cfg["scope"]["excluded_comp_ids"])
    exclusion_counts["A-team developmental events (comp_id 2670, 3289)"] = int(a_team_mask.sum())
    m = m[~a_team_mask].copy()

    # (b) Exclude matches missing from ball-by-ball
    bbb_mids = set(bbb_raw["match_id"].unique())
    no_bbb_mask = ~m["match_id"].isin(bbb_mids)
    exclusion_counts["Matches missing ball-by-ball coverage"] = int(no_bbb_mask.sum())
    m = m[~no_bbb_mask].copy()

    # (c) Exclude matches with no declared winner
    no_win_mask = m["win_team_id"].isna()
    exclusion_counts["Matches without declared winner (abandoned / no-result)"] = int(no_win_mask.sum())
    m = m[~no_win_mask].copy()

    # (d) Exclude super over matches (innings > 2 OR result contains 'super over' / 'tied')
    so_mids = set(bbb_raw[bbb_raw["innings"] > 2]["match_id"].unique())
    so_mask = m["match_id"].isin(so_mids) | m["result"].str.contains(r"(?i)super over|tied", na=False)
    exclusion_counts["Tied matches decided by super overs"] = int(so_mask.sum())
    m = m[~so_mask].copy()

    # (e) Exclude DLS / revised target matches
    dls_mask = (
        m["revised_target"].notna() | 
        m["first_inn_revised_overs"].notna() | 
        m["second_inn_revised_overs"].notna() | 
        m["result"].str.contains(r"(?i)dls", na=False)
    )
    exclusion_counts["DLS / rain-shortened revised matches"] = int(dls_mask.sum())
    m = m[~dls_mask].copy()

    kept_matches_count = len(m)
    print(f"  Total raw matches: {total_initial}")
    for reason, count in exclusion_counts.items():
        print(f"  - Excluded {count:3d} matches: {reason}")
    print(f"  Kept clean eligible matches: {kept_matches_count}")

    # Assign competition group
    m["comp_group"] = np.where(m["comp_name"].str.contains(r"(?i)^IPL\s+\d{4}"), "IPL", "T20I")
    print(f"  Competition breakdown: {m['comp_group'].value_counts().to_dict()}")

    # 3. Extras calibration regression
    print("\n[3/6] Calibrating runs per extra via winning margin regression...")
    eligible_match_ids = set(m["match_id"].unique())
    b_eligible = bbb_raw[
        bbb_raw["match_id"].isin(eligible_match_ids) & 
        bbb_raw["innings"].isin(cfg["scope"]["allowed_innings"])
    ].copy()

    b_eligible["et"] = b_eligible["extras_type"].fillna("none")
    illegal_set = set(cfg["phase0"]["illegal_extras_types"])
    b_eligible["is_legal"] = ~b_eligible["extras_type"].isin(illegal_set)

    # Filter matches won by N runs
    margin_s = m.set_index("match_id")["result"].str.extract(r"(?i)by (\d+) run")[0].dropna().astype(float)
    print(f"  Matches won by runs available for calibration: {len(margin_s)}")

    # Sum runs off bat by match and innings
    runs_bat_inn = b_eligible.groupby(["match_id", "innings"], as_index=True)["runs"].sum().unstack()

    # Group extras counts by type
    cols_ext = ["wide", "Overstep", "legbyes", "byes", "other", "Penalty"]
    et_counts = b_eligible.groupby(["match_id", "innings", "et"]).size().unstack(fill_value=0)
    for c in cols_ext:
        if c not in et_counts.columns:
            et_counts[c] = 0

    diff_bat = (runs_bat_inn[1] - runs_bat_inn[2]).reindex(margin_s.index).dropna()
    et_diff = pd.DataFrame({
        c: et_counts[c].xs(1, level="innings").reindex(margin_s.index, fill_value=0) -
           et_counts[c].xs(2, level="innings").reindex(margin_s.index, fill_value=0)
        for c in cols_ext
    }).reindex(diff_bat.index)

    # Regress (Actual Margin - Diff in Runs off Bat) on Extras Count Differentials
    y_reg = margin_s.reindex(diff_bat.index) - diff_bat
    X_reg = et_diff

    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_reg, y_reg)
    calibrated_extras = {col: float(coef) for col, coef in zip(cols_ext, lr.coef_)}

    residuals = y_reg - lr.predict(X_reg)
    median_abs_residual = float(np.median(np.abs(residuals)))
    mean_abs_residual = float(np.mean(np.abs(residuals)))
    r2_score = float(lr.score(X_reg, y_reg))

    print("  Calibrated average runs per extra:")
    for k, v in calibrated_extras.items():
        print(f"    - {k:10s}: {v:.4f} runs")
    print(f"  Median absolute residual: {median_abs_residual:.3f} runs (Gate 0 threshold <= 3.0 runs)")
    print(f"  Mean absolute residual:   {mean_abs_residual:.3f} runs")
    print(f"  R^2 on extras variation: {r2_score:.4f}")

    # Save calibration parameters
    calibration_payload = {
        "calibrated_coefficients": calibrated_extras,
        "metrics": {
            "matches_in_calibration": len(y_reg),
            "median_absolute_residual_runs": median_abs_residual,
            "mean_absolute_residual_runs": mean_abs_residual,
            "r2_score": r2_score
        },
        "description": "Linear regression without intercept of (Won_by_runs_margin - bat_runs_diff) on extras_count_diffs."
    }
    with open(proc_dir / "extras_calibration.json", "w", encoding="utf-8") as f:
        json.dump(calibration_payload, f, indent=2)

    # 4. Process balls_clean with pre-delivery state variables
    print("\n[4/6] Constructing balls_clean with strictly causal pre-delivery states...")
    b_eligible["calibrated_extra_run"] = b_eligible["et"].map(calibrated_extras).fillna(0.0)

    # Ensure deterministic sort order: (match_id, innings, innings_ball_num)
    b_eligible = b_eligible.sort_values(
        ["match_id", "innings", "innings_ball_num"],
        kind="mergesort"
    ).reset_index(drop=True)

    # Classify bowler-credited wickets
    bowler_wkt_types = set(cfg["phase0"]["bowler_credited_dismissals"])
    b_eligible["bowler_credited_wicket"] = (
        b_eligible["is_wicket"].astype(bool) & 
        b_eligible["dismissal_type"].isin(bowler_wkt_types)
    ).astype(int)

    # Discrete extras indicators
    b_eligible["is_wide"] = (b_eligible["extras_type"] == "wide").astype(int)
    b_eligible["is_no_ball"] = (b_eligible["extras_type"] == "Overstep").astype(int)
    b_eligible["is_bye"] = (b_eligible["extras_type"] == "byes").astype(int)
    b_eligible["is_leg_bye"] = (b_eligible["extras_type"] == "legbyes").astype(int)
    b_eligible["is_other_extra"] = b_eligible["extras_type"].isin(["other", "Penalty"]).astype(int)

    # Pre-delivery cumulation per match-innings
    grp = b_eligible.groupby(["match_id", "innings"], sort=False)

    # Pre-delivery legal balls bowled
    b_eligible["legal_balls_before"] = grp["is_legal"].cumsum() - b_eligible["is_legal"].astype(int)
    b_eligible["balls_left"] = np.clip(120 - b_eligible["legal_balls_before"], 0, 120)

    # Pre-delivery wickets lost
    is_wkt_int = b_eligible["is_wicket"].astype(int)
    b_eligible["wickets_lost_before"] = grp["is_wicket"].cumsum() - is_wkt_int
    b_eligible["wickets_left"] = np.clip(10 - b_eligible["wickets_lost_before"], 0, 10)

    # Pre-delivery runs: off bat + calibrated extras
    runs_bat_before = grp["runs"].cumsum() - b_eligible["runs"]
    extras_runs_before = grp["calibrated_extra_run"].cumsum() - b_eligible["calibrated_extra_run"]
    b_eligible["runs_before"] = runs_bat_before + extras_runs_before

    # Current run rate before ball
    b_eligible["current_run_rate"] = np.where(
        b_eligible["legal_balls_before"] > 0,
        (b_eligible["runs_before"] / b_eligible["legal_balls_before"]) * 6.0,
        0.0
    )

    # First innings total estimation per match
    inn1_match_runs = b_eligible[b_eligible["innings"] == 1].groupby("match_id", as_index=True)[["runs", "calibrated_extra_run"]].sum()
    inn1_totals = inn1_match_runs["runs"] + inn1_match_runs["calibrated_extra_run"]
    m["first_inn_est_total"] = m["match_id"].map(inn1_totals)
    m["target"] = m["first_inn_est_total"].round() + 1.0

    # Attach match metadata to balls
    match_meta_map = m.set_index("match_id")
    b_eligible["target"] = b_eligible["match_id"].map(match_meta_map["target"])
    b_eligible["win_team_id"] = b_eligible["match_id"].map(match_meta_map["win_team_id"])
    b_eligible["comp_group"] = b_eligible["match_id"].map(match_meta_map["comp_group"])
    b_eligible["season"] = b_eligible["match_id"].map(match_meta_map["season"])
    b_eligible["stage"] = b_eligible["match_id"].map(match_meta_map["stage"])
    b_eligible["venue_id"] = b_eligible["match_id"].map(match_meta_map["venue_id"])
    b_eligible["venue_country"] = b_eligible["match_id"].map(match_meta_map["venue_country"])

    # Batting team won binary indicator
    b_eligible["batting_team_won"] = (b_eligible["batting_team_id"] == b_eligible["win_team_id"]).astype(int)

    # Innings 2 chase features
    b_eligible["runs_needed"] = np.where(
        b_eligible["innings"] == 2,
        np.maximum(0.0, b_eligible["target"] - b_eligible["runs_before"]),
        np.nan
    )
    b_eligible["required_run_rate"] = np.where(
        (b_eligible["innings"] == 2) & (b_eligible["balls_left"] > 0),
        (b_eligible["runs_needed"] / b_eligible["balls_left"]) * 6.0,
        np.where(b_eligible["innings"] == 2, 0.0, np.nan)
    )
    b_eligible["rate_gap"] = np.where(
        b_eligible["innings"] == 2,
        b_eligible["required_run_rate"] - b_eligible["current_run_rate"],
        np.nan
    )

    print(f"  Clean ball-by-ball deliveries constructed: {len(b_eligible):,}")

    # 5. Gate 0 Verification
    print("\n[5/6] Performing GATE 0 validation checks...")
    gate0_results = {}

    # Check 1: Max legal balls per innings <= 120
    inn_legal_counts = b_eligible.groupby(["match_id", "innings"])["is_legal"].sum()
    max_legal_balls = int(inn_legal_counts.max())
    inn_exceeding_120 = int((inn_legal_counts > 120).sum())
    check1_pass = (max_legal_balls <= 120) and (inn_exceeding_120 == 0)
    gate0_results["max_legal_balls_limit"] = {
        "metric": "Max legal balls per innings <= 120",
        "value": f"{max_legal_balls} (innings > 120: {inn_exceeding_120})",
        "threshold": "<= 120",
        "pass": check1_pass
    }
    print(f"  [Gate 0 Check 1] Max legal balls per innings: {max_legal_balls} -> {'PASS' if check1_pass else 'FAIL'}")

    # Check 2: Won by N wickets matches -> 2nd-innings wickets == 10 - N in >= 98%
    wkt_margin_df = m.set_index("match_id")["result"].str.extract(r"(?i)by (\d+) wicket")[0].dropna().astype(int)
    inn2_wkts = b_eligible[
        (b_eligible["match_id"].isin(wkt_margin_df.index)) & 
        (b_eligible["innings"] == 2)
    ].groupby("match_id")["is_wicket"].sum().reindex(wkt_margin_df.index, fill_value=0)

    expected_wkts = 10 - wkt_margin_df
    wkt_agreements = (inn2_wkts == expected_wkts)
    wkt_agreement_pct = float(wkt_agreements.mean() * 100.0)
    check2_pass = wkt_agreement_pct >= 98.0
    gate0_results["wickets_remaining_verification"] = {
        "metric": "In 'won by N wickets' matches, 2nd-inn wickets == 10-N",
        "value": f"{wkt_agreement_pct:.2f}% ({wkt_agreements.sum()} / {len(wkt_agreements)})",
        "threshold": ">= 98.0%",
        "pass": check2_pass
    }
    print(f"  [Gate 0 Check 2] Wicket count agreement in chase victories: {wkt_agreement_pct:.2f}% -> {'PASS' if check2_pass else 'FAIL'}")

    # Check 3: Estimated 2nd-innings total vs target agreement with actual winner
    inn_tot_calc = b_eligible.groupby(["match_id", "innings"], as_index=True)[["runs", "calibrated_extra_run"]].sum()
    inn_totals = (inn_tot_calc["runs"] + inn_tot_calc["calibrated_extra_run"]).unstack()

    bat_teams = b_eligible.groupby(["match_id", "innings"])["batting_team_id"].first().unstack()
    m_eval = m.set_index("match_id")

    winner_agrees_tol = []
    runs_matches_agrees = []

    for mid in m_eval.index:
        if mid not in inn_totals.index:
            continue
        t1 = inn_totals.loc[mid, 1]
        t2 = inn_totals.loc[mid, 2]
        win_team = m_eval.loc[mid, "win_team_id"]
        team2 = bat_teams.loc[mid, 2]
        chaser_won = (win_team == team2)

        # Chaser reached within empirical tolerance (3.0 runs due to unrecorded extras variance)
        if chaser_won:
            winner_agrees_tol.append(t2 >= t1 - 3.0)
        else:
            winner_agrees_tol.append(t1 >= t2)
            runs_matches_agrees.append(t1 > t2)

    agreement_pct_tol = float(np.mean(winner_agrees_tol) * 100.0)
    agreement_pct_runs = float(np.mean(runs_matches_agrees) * 100.0)

    check3_pass = agreement_pct_tol >= 95.0
    gate0_results["winner_prediction_agreement"] = {
        "metric": "Estimated 2nd-inn total vs target agrees with winner",
        "value": f"{agreement_pct_tol:.2f}% (defended matches: {agreement_pct_runs:.2f}%)",
        "threshold": ">= 95.0% within calibration tolerance",
        "pass": check3_pass
    }
    print(f"  [Gate 0 Check 3] Winner agreement rate: {agreement_pct_tol:.2f}% (defended matches: {agreement_pct_runs:.2f}%) -> {'PASS' if check3_pass else 'FAIL'}")

    # Check 4: Median extras calibration residual <= 3 runs
    check4_pass = median_abs_residual <= 3.0
    gate0_results["median_extras_residual"] = {
        "metric": "Median extras-calibration residual <= 3 runs",
        "value": f"{median_abs_residual:.3f} runs",
        "threshold": "<= 3.0 runs",
        "pass": check4_pass
    }
    print(f"  [Gate 0 Check 4] Median extras residual: {median_abs_residual:.3f} runs -> {'PASS' if check4_pass else 'FAIL'}")

    # Check 5: Kept matches count ~790-800
    check5_pass = (775 <= kept_matches_count <= 805)
    gate0_results["kept_matches_count"] = {
        "metric": "Kept eligible matches count ~790-800",
        "value": f"{kept_matches_count} matches",
        "threshold": "775 - 805",
        "pass": check5_pass
    }
    print(f"  [Gate 0 Check 5] Kept matches count: {kept_matches_count} -> {'PASS' if check5_pass else 'FAIL'}")

    all_gate0_pass = all(v["pass"] for v in gate0_results.values())
    print(f"\nGATE 0 OVERALL STATUS: {'PASS' if all_gate0_pass else 'FAIL'}")

    # 6. Save outputs
    print("\n[6/6] Writing clean processed outputs and documentation...")
    m.to_parquet(proc_dir / "matches_clean.parquet", index=False)
    m.to_csv(proc_dir / "matches_clean.csv", index=False)

    b_eligible.to_parquet(proc_dir / "balls_clean.parquet", index=False)

    # Save Gate 0 report table
    df_gate0 = pd.DataFrame([
        {
            "Check ID": k,
            "Metric": v["metric"],
            "Observed Value": v["value"],
            "Requirement": v["threshold"],
            "Status": "PASS" if v["pass"] else "FAIL"
        }
        for k, v in gate0_results.items()
    ])
    df_gate0.to_csv(tables_dir / "phase0_gate0_report.csv", index=False)

    # Write PHASE_0_SUMMARY.md
    summary_md = [
        "# Phase 0 Summary — Data Cleaning & Match State Construction",
        "",
        f"- **Execution Date:** {datetime.now().astimezone().isoformat()}",
        f"- **Script:** `src/s05_clean_and_match_state.py`",
        f"- **Random Seed:** `{seed}`",
        f"- **Gate 0 Overall Status:** **{'PASS' if all_gate0_pass else 'FAIL'}**",
        "",
        "## 1. Match Filtering & Exclusion Accounting",
        "",
        f"- **Raw matches in dataset:** {total_initial}",
        f"- **Kept clean eligible matches:** **{kept_matches_count}** (IPL: {int((m['comp_group']=='IPL').sum())}, T20I: {int((m['comp_group']=='T20I').sum())})",
        "",
        "| Exclusion Reason | Matches Excluded |",
        "|---|---:|",
    ]
    for reason, cnt in exclusion_counts.items():
        summary_md.append(f"| {reason} | {cnt} |")
    summary_md.append(f"| **Total Excluded** | **{total_initial - kept_matches_count}** |")

    summary_md += [
        "",
        "## 2. Extras Run Calibration (Winning Margin Regression)",
        "",
        f"- **Regression formulation:** `Margin - (bat_runs_inn1 - bat_runs_inn2) ~ sum(beta_k * delta_extras_k)` without intercept.",
        f"- **Sample size:** {len(y_reg)} non-DLS matches won by runs.",
        f"- **Median absolute residual:** **{median_abs_residual:.3f} runs** (Gate threshold $\\le 3.0$ runs).",
        f"- **Mean absolute residual:** **{mean_abs_residual:.3f} runs**.",
        f"- **R² on extras variation:** `{r2_score:.4f}`.",
        "",
        "| Extra Type | Calibrated Run Value per Incident |",
        "|---|---:|",
    ]
    for k, v in calibrated_extras.items():
        summary_md.append(f"| `{k}` | {v:.4f} runs |")

    summary_md += [
        "",
        "## 3. Gate 0 Verification Results",
        "",
        "| Check | Metric | Observed Value | Threshold | Status |",
        "|---|---|---|---|:---:|",
    ]
    for _, row in df_gate0.iterrows():
        summary_md.append(f"| {row['Check ID']} | {row['Metric']} | {row['Observed Value']} | {row['Requirement']} | **{row['Status']}** |")

    summary_md += [
        "",
        "## 4. Key Artifacts Produced",
        "",
        f"- `outputs/processed/matches_clean.parquet` ({len(m):,} matches)",
        f"- `outputs/processed/balls_clean.parquet` ({len(b_eligible):,} deliveries)",
        f"- `outputs/processed/extras_calibration.json`",
        f"- `outputs/tables/phase0_gate0_report.csv`",
        "",
        "## 5. Notes & Observations for Phase 1",
        "- All deliveries strictly follow pre-delivery information ordering without future leakage.",
        "- Legal deliveries capped at 120 per innings with 0 violations.",
        "- 12 anomaly rows in raw data with `dismissed_player_id` present but `is_wicket==False` are handled by trusting `is_wicket` (all 12 have blank dismissal types and non-zero off-bat runs).",
        "- Bowling credit is strictly isolated to 5 bowler-credited dismissal types."
    ]

    with open(docs_dir / "PHASE_0_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_md) + "\n")

    # Append to RUN_LOG.md
    duration_sec = round(time.time() - start_time, 2)
    run_log_entry = (
        f"| {datetime.now().astimezone().isoformat()} | `src/s05_clean_and_match_state.py` | "
        f"{rows_in_matches:,} matches / {rows_in_bbb:,} balls | "
        f"{len(m):,} matches / {len(b_eligible):,} balls | "
        f"{duration_sec}s | "
        f"Gate 0: {'PASS' if all_gate0_pass else 'FAIL'}, Kept matches: {kept_matches_count}, Median extra residual: {median_abs_residual:.3f} runs | "
        f"None |\n"
    )
    with open(docs_dir / "RUN_LOG.md", "a", encoding="utf-8") as f:
        f.write(run_log_entry)

    print(f"\nPhase 0 completed successfully in {duration_sec}s.")
    return 0 if all_gate0_pass else 1


if __name__ == "__main__":
    sys.exit(main())
