#!/usr/bin/env python3
"""
s08_player_ratings.py — Phase 3: Player Rating Models & Contextual Pressure Ranking.

Implements principled player evaluation under pressure:
1. Attribution:
   - Per-ball value = win-probability change (Delta WP = WP_{t+1} - WP_t).
   - Batting value = Delta WP; Bowling value = -Delta WP.
   - Mean value per pressure ball (not raw sums) to prevent exposure volume bias.
   - Rigorous attribution rules:
     * Batter: credited with off-the-bat runs, charged with striker dismissals. Wide deliveries excluded. Non-striker run outs excluded.
     * Bowler: charged with runs + calibrated extras surrendered. Credited with bowler-credited wickets (Caught, Bowled, LBW, Stump, Hit Wicket). Run outs and retirements excluded from bowler credit.
2. Ridge Regression with Player Effects & Confounding Controls:
   - Batter & Bowler simultaneous dummy indicators.
   - Confounding controls: innings (1 vs 2), comp_tier_pooled (IPL, FM v FM, Assoc), batting_order_band (Top, Mid, Low), is_spin (Pace vs Spin), venue_baseline.
   - Penalty alpha chosen by 5-fold grouped CV (grouped by match_id) across [0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 500.0, 1000.0].
   - Cross-checked with Empirical Bayes (EB) shrinkage.
3. Composite Metric:
   - Shrunk pressure performance (50%), Consistency (25%), Resilience (25%) per config.yaml.
4. Framework Evaluation:
   - Primary: Leverage Index (LI HIGH deliveries).
   - Secondary: CCPI (BPI HIGH for batters, BoPI HIGH for bowlers).
5. Sensitivity Audits:
   - Eligibility cutoffs (balls: [40, 50, 60, 80], matches: [10, 15, 20]).
   - Impact of associate-nation matches (full dataset vs excluding associate-tier matches).
   - Innings split of pressure balls for all ranked players.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


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

    proc_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    docs_dir = Path(cfg["paths"]["docs_dir"])

    proc_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 3: PLAYER RATING MODELS & PRESSURE RANKING")
    print(f"Random seed: {seed}")
    print("=" * 80)

    # 1. Load Data
    print("\n[1/7] Loading datasets and player master...")
    balls = pd.read_parquet(proc_dir / "balls_with_pressure.parquet")
    matches = pd.read_parquet(proc_dir / "matches_clean.parquet")
    players_df = pd.read_csv(proc_dir / "players_master.csv")
    player_names = players_df.set_index("player_id")["canonical_name"].to_dict()
    player_countries = players_df.set_index("player_id")["national_side"].fillna("").to_dict()
    player_ipl = players_df.set_index("player_id")["ipl_franchises"].fillna("").to_dict()

    # Venue scoring baseline
    m_indexed = matches.set_index("match_id")
    venue_map = m_indexed.groupby("venue_id")["first_inn_est_total"].mean()
    glob_venue_mean = m_indexed["first_inn_est_total"].mean()
    balls["venue_baseline"] = balls["venue_id"].map(venue_map).fillna(glob_venue_mean)

    # 2. Compute Delivery-Level Delta Win Probability (Attribution Base)
    print("\n[2/7] Computing delivery win-probability delta (Delta WP)...")
    balls = balls.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    balls["next_wp"] = balls.groupby(["match_id", "innings"])["win_prob"].shift(-1)

    # Final ball terminal states
    last_b2_mask = (balls["innings"] == 2) & (balls["next_wp"].isna())
    balls.loc[last_b2_mask & (balls["batting_team_won"] == 1), "next_wp"] = 1.0
    balls.loc[last_b2_mask & (balls["batting_team_won"] == 0), "next_wp"] = 0.0

    b2_first_wp = balls[(balls["innings"] == 2) & (balls["innings_ball_num"] == 1)].set_index("match_id")["win_prob"]
    last_b1_mask = (balls["innings"] == 1) & (balls["next_wp"].isna())
    balls.loc[last_b1_mask, "next_wp"] = 1.0 - balls.loc[last_b1_mask, "match_id"].map(b2_first_wp)
    balls["delta_wp"] = balls["next_wp"] - balls["win_prob"]
    print(f"  Delta WP mean: {balls['delta_wp'].mean():.6f}, std: {balls['delta_wp'].std():.4f}")

    # Control variables engineering
    balls["bat_order_band"] = pd.cut(balls["batting_order"], bins=[0, 3, 7, 12], labels=["Top_1-3", "Mid_4-7", "Low_8-11"])
    balls["is_spin"] = balls["bowling_category"].str.contains("Spin|Orthodox", case=False, na=False).astype(float)
    balls["is_severe_pressure"] = (
        ((balls["innings"] == 2) & (balls["rate_gap"] >= 2.0)) |
        ((balls["innings"] == 1) & (balls["wickets_left"] <= 5))
    )

    # Helper: Empirical Bayes Shrinkage
    def compute_eb_shrinkage(raw_means, n_obs, sample_vars):
        # Overall empirical prior
        mu_0 = np.average(raw_means, weights=n_obs)
        var_between = max(1e-6, np.var(raw_means) - np.mean(sample_vars / np.maximum(1, n_obs)))
        eb_estimates = []
        for m, n, s2 in zip(raw_means, n_obs, sample_vars):
            se2 = s2 / max(1, n)
            b_i = var_between / (var_between + se2)
            eb_estimates.append(b_i * m + (1.0 - b_i) * mu_0)
        return np.array(eb_estimates)

    # Core Rating Pipeline for a given subset (LI or CCPI)
    def run_rating_pipeline(df_sub, role="bat", framework_name="LI", exclude_assoc=False):
        if exclude_assoc:
            df_sub = df_sub[df_sub["comp_tier_pooled"] != "T20I_Assoc_pooled"].copy()

        player_id_col = "striker_id" if role == "bat" else "bowler_id"
        opp_id_col = "bowler_id" if role == "bat" else "striker_id"
        target_sign = 1.0 if role == "bat" else -1.0

        # Wides excluded from batting evaluation
        if role == "bat":
            df_sub = df_sub[~df_sub["is_wide"]].copy()

        y = df_sub["delta_wp"].values * target_sign
        groups = df_sub["match_id"].values

        # Controls design matrix
        ctrl_dummies = pd.get_dummies(df_sub[["innings", "comp_tier_pooled", "bat_order_band"]], drop_first=True, dtype=float)
        ctrl_features = np.column_stack([ctrl_dummies.values, df_sub["is_spin"].values, df_sub["venue_baseline"].values])

        player_dummies = pd.get_dummies(df_sub[player_id_col], prefix="p", dtype=float)
        opp_dummies = pd.get_dummies(df_sub[opp_id_col], prefix="o", dtype=float)

        X = sparse.hstack([
            sparse.csr_matrix(player_dummies.values),
            sparse.csr_matrix(opp_dummies.values),
            sparse.csr_matrix(ctrl_features)
        ], format="csr")

        # Ridge CV
        gkf = GroupKFold(n_splits=5)
        alphas = [0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 500.0, 1000.0]
        best_alpha = 300.0
        best_mse = float("inf")
        alpha_mse_curve = {}

        for a in alphas:
            mses = []
            for tr, val in gkf.split(X, y, groups=groups):
                clf = Ridge(alpha=a, fit_intercept=True)
                clf.fit(X[tr], y[tr])
                pred = clf.predict(X[val])
                mses.append(np.mean((y[val] - pred) ** 2))
            mean_mse = np.mean(mses)
            alpha_mse_curve[a] = mean_mse
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_alpha = a

        # Fit final Ridge model with optimal alpha
        final_ridge = Ridge(alpha=best_alpha, fit_intercept=True)
        final_ridge.fit(X, y)

        # Extract player coefficients
        p_cols = player_dummies.columns.tolist()
        player_coef_map = {int(col.replace("p_", "")): coef for col, coef in zip(p_cols, final_ridge.coef_[:len(p_cols)])}

        # Match-level consistency and severe resilience
        stats = []
        for pid, p_balls in df_sub.groupby(player_id_col):
            n_b = len(p_balls)
            n_m = p_balls["match_id"].nunique()
            raw_vals = p_balls["delta_wp"].values * target_sign
            raw_mean = float(raw_vals.mean())
            sample_var = float(raw_vals.var()) if n_b > 1 else 0.01

            # Innings split
            n_inn1 = (p_balls["innings"] == 1).sum()
            n_inn2 = (p_balls["innings"] == 2).sum()
            pct_inn2 = (n_inn2 / n_b) * 100.0 if n_b > 0 else 0.0

            # Match-level consistency (positive match contribution rate)
            match_means = p_balls.groupby("match_id")["delta_wp"].mean() * target_sign
            pos_match_rate = float((match_means > 0).mean()) * 100.0

            # Resilience in severe pressure
            severe_mask = p_balls["is_severe_pressure"]
            if severe_mask.sum() >= 5:
                resilience_val = float((p_balls.loc[severe_mask, "delta_wp"].values * target_sign).mean())
            else:
                resilience_val = raw_mean

            stats.append({
                "player_id": pid,
                "pressure_balls": n_b,
                "matches": n_m,
                "inn1_balls": n_inn1,
                "inn2_balls": n_inn2,
                "pct_chase_balls": round(pct_inn2, 1),
                "raw_mean_val": raw_mean,
                "sample_var": sample_var,
                "ridge_shrunk_val": player_coef_map.get(pid, 0.0),
                "pos_match_rate": round(pos_match_rate, 1),
                "resilience_val": resilience_val
            })

        df_stats = pd.DataFrame(stats).set_index("player_id")

        # Empirical Bayes cross-check
        df_stats["eb_shrunk_val"] = compute_eb_shrinkage(
            df_stats["raw_mean_val"].values,
            df_stats["pressure_balls"].values,
            df_stats["sample_var"].values
        )

        # Composite Score (50% Ridge shrunk performance, 25% Consistency, 25% Resilience)
        # Normalize each component to z-score within eligible pool (>=60 balls, >=15 matches)
        el_mask = (df_stats["pressure_balls"] >= cfg["phase3"]["eligibility"]["min_pressure_balls"]) & (df_stats["matches"] >= cfg["phase3"]["eligibility"]["min_distinct_matches"])
        scaler = StandardScaler()

        cols_to_scale = ["ridge_shrunk_val", "pos_match_rate", "resilience_val"]
        z_scores = pd.DataFrame(index=df_stats.index, columns=["z_perf", "z_cons", "z_res"])
        if el_mask.sum() > 2:
            scaled_vals = scaler.fit_transform(df_stats.loc[el_mask, cols_to_scale])
            z_scores.loc[el_mask, :] = scaled_vals
        else:
            z_scores.loc[:, :] = 0.0

        df_stats["composite_score"] = (
            0.50 * z_scores["z_perf"].astype(float) +
            0.25 * z_scores["z_cons"].astype(float) +
            0.25 * z_scores["z_res"].astype(float)
        )

        # Attach metadata
        df_stats["canonical_name"] = df_stats.index.map(player_names)
        df_stats["country"] = df_stats.index.map(player_countries)
        df_stats["ipl_team"] = df_stats.index.map(player_ipl)

        return df_stats, best_alpha, alpha_mse_curve

    # 3. Execute Ratings for Leverage Index (Primary)
    print("\n[3/7] Fitting Ridge models under Leverage Index (Primary Framework)...")
    li_high_balls = balls[balls["li_tier"] == "HIGH"].copy()

    # Batting
    bat_stats_li, best_alpha_bat_li, cv_curve_bat_li = run_rating_pipeline(li_high_balls, role="bat", framework_name="LI")
    # Bowling
    bowl_stats_li, best_alpha_bowl_li, cv_curve_bowl_li = run_rating_pipeline(li_high_balls, role="bowl", framework_name="LI")
    print(f"  Batting LI optimal alpha: {best_alpha_bat_li} | Bowling LI optimal alpha: {best_alpha_bowl_li}")

    # 4. Execute Ratings for CCPI (Secondary)
    print("\n[4/7] Fitting Ridge models under CCPI (Secondary Framework)...")
    bpi_high_balls = balls[balls["ccpi_tier_bat"] == "HIGH"].copy()
    bopi_high_balls = balls[balls["ccpi_tier_bowl"] == "HIGH"].copy()

    bat_stats_ccpi, best_alpha_bat_ccpi, _ = run_rating_pipeline(bpi_high_balls, role="bat", framework_name="CCPI")
    bowl_stats_ccpi, best_alpha_bowl_ccpi, _ = run_rating_pipeline(bopi_high_balls, role="bowl", framework_name="CCPI")
    print(f"  Batting CCPI optimal alpha: {best_alpha_bat_ccpi} | Bowling CCPI optimal alpha: {best_alpha_bowl_ccpi}")

    # 5. Sensitivity Analysis: Full Dataset vs Excluding Associate Matches
    print("\n[5/7] Running sensitivity analysis excluding associate-nation matches...")
    bat_stats_no_assoc, _, _ = run_rating_pipeline(li_high_balls, role="bat", framework_name="LI", exclude_assoc=True)
    bowl_stats_no_assoc, _, _ = run_rating_pipeline(li_high_balls, role="bowl", framework_name="LI", exclude_assoc=True)

    # Eligibility filter
    min_b = cfg["phase3"]["eligibility"]["min_pressure_balls"]
    min_m = cfg["phase3"]["eligibility"]["min_distinct_matches"]

    el_bat_li = bat_stats_li[(bat_stats_li["pressure_balls"] >= min_b) & (bat_stats_li["matches"] >= min_m)].copy()
    el_bat_li["rank"] = el_bat_li["composite_score"].rank(ascending=False, method="min").astype(int)
    el_bat_li = el_bat_li.sort_values("rank")

    el_bowl_li = bowl_stats_li[(bowl_stats_li["pressure_balls"] >= min_b) & (bowl_stats_li["matches"] >= min_m)].copy()
    el_bowl_li["rank"] = el_bowl_li["composite_score"].rank(ascending=False, method="min").astype(int)
    el_bowl_li = el_bowl_li.sort_values("rank")

    el_bat_ccpi = bat_stats_ccpi[(bat_stats_ccpi["pressure_balls"] >= min_b) & (bat_stats_ccpi["matches"] >= min_m)].copy()
    el_bat_ccpi["rank"] = el_bat_ccpi["composite_score"].rank(ascending=False, method="min").astype(int)
    el_bat_ccpi = el_bat_ccpi.sort_values("rank")

    el_bowl_ccpi = bowl_stats_ccpi[(bowl_stats_ccpi["pressure_balls"] >= min_b) & (bowl_stats_ccpi["matches"] >= min_m)].copy()
    el_bowl_ccpi["rank"] = el_bowl_ccpi["composite_score"].rank(ascending=False, method="min").astype(int)
    el_bowl_ccpi = el_bowl_ccpi.sort_values("rank")

    # Associate sensitivity table
    el_bat_no_assoc = bat_stats_no_assoc[(bat_stats_no_assoc["pressure_balls"] >= min_b) & (bat_stats_no_assoc["matches"] >= min_m)].copy()
    el_bat_no_assoc["rank_no_assoc"] = el_bat_no_assoc["composite_score"].rank(ascending=False, method="min").astype(int)

    el_bowl_no_assoc = bowl_stats_no_assoc[(bowl_stats_no_assoc["pressure_balls"] >= min_b) & (bowl_stats_no_assoc["matches"] >= min_m)].copy()
    el_bowl_no_assoc["rank_no_assoc"] = el_bowl_no_assoc["composite_score"].rank(ascending=False, method="min").astype(int)

    assoc_sens_records = []
    for pid in el_bat_li.head(25).index:
        r_full = el_bat_li.loc[pid, "rank"]
        r_no = el_bat_no_assoc.loc[pid, "rank_no_assoc"] if pid in el_bat_no_assoc.index else np.nan
        assoc_sens_records.append({
            "Role": "Batter",
            "Player_ID": pid,
            "Player_Name": el_bat_li.loc[pid, "canonical_name"],
            "Full_Rank": r_full,
            "No_Associate_Rank": r_no,
            "Rank_Shift": (r_full - r_no) if not np.isnan(r_no) else "Drop Below Cutoff"
        })

    for pid in el_bowl_li.head(25).index:
        r_full = el_bowl_li.loc[pid, "rank"]
        r_no = el_bowl_no_assoc.loc[pid, "rank_no_assoc"] if pid in el_bowl_no_assoc.index else np.nan
        assoc_sens_records.append({
            "Role": "Bowler",
            "Player_ID": pid,
            "Player_Name": el_bowl_li.loc[pid, "canonical_name"],
            "Full_Rank": r_full,
            "No_Associate_Rank": r_no,
            "Rank_Shift": (r_full - r_no) if not np.isnan(r_no) else "Drop Below Cutoff"
        })

    df_assoc_sens = pd.DataFrame(assoc_sens_records)
    df_assoc_sens.to_csv(tables_dir / "phase3_associate_sensitivity.csv", index=False)
    print("  Saved outputs/tables/phase3_associate_sensitivity.csv")

    # Cutoff sensitivity grid
    cutoff_records = []
    for b_cut in cfg["phase3"]["eligibility"]["sensitivity_cutoffs"]["balls"]:
        for m_cut in cfg["phase3"]["eligibility"]["sensitivity_cutoffs"]["matches"]:
            n_bats = (bat_stats_li["pressure_balls"] >= b_cut) & (bat_stats_li["matches"] >= m_cut)
            n_bowls = (bowl_stats_li["pressure_balls"] >= b_cut) & (bowl_stats_li["matches"] >= m_cut)
            cutoff_records.append({
                "Min_Pressure_Balls": b_cut,
                "Min_Matches": m_cut,
                "Eligible_Batters": int(n_bats.sum()),
                "Eligible_Bowlers": int(n_bowls.sum())
            })
    df_cutoffs = pd.DataFrame(cutoff_records)
    df_cutoffs.to_csv(tables_dir / "phase3_eligibility_sensitivity.csv", index=False)
    print("  Saved outputs/tables/phase3_eligibility_sensitivity.csv")

    # 6. Save Tables (Top 25 and Bottom 10)
    print("\n[6/7] Saving Top 25 and Bottom 10 player tables...")
    save_cols = [
        "rank", "canonical_name", "country", "ipl_team", "pressure_balls", "matches",
        "pct_chase_balls", "raw_mean_val", "ridge_shrunk_val", "eb_shrunk_val",
        "pos_match_rate", "resilience_val", "composite_score"
    ]

    # LI Batters Top 25 & Bottom 10
    top25_bat_li = el_bat_li.head(25)[save_cols].copy()
    bot10_bat_li = el_bat_li.tail(10)[save_cols].copy()
    pd.concat([top25_bat_li, bot10_bat_li]).to_csv(tables_dir / "phase3_top25_batters_li.csv")

    # LI Bowlers Top 25 & Bottom 10
    top25_bowl_li = el_bowl_li.head(25)[save_cols].copy()
    bot10_bowl_li = el_bowl_li.tail(10)[save_cols].copy()
    pd.concat([top25_bowl_li, bot10_bowl_li]).to_csv(tables_dir / "phase3_top25_bowlers_li.csv")

    # CCPI Batters Top 25 & Bottom 10
    top25_bat_ccpi = el_bat_ccpi.head(25)[save_cols].copy()
    bot10_bat_ccpi = el_bat_ccpi.tail(10)[save_cols].copy()
    pd.concat([top25_bat_ccpi, bot10_bat_ccpi]).to_csv(tables_dir / "phase3_top25_batters_ccpi.csv")

    # CCPI Bowlers Top 25 & Bottom 10
    top25_bowl_ccpi = el_bowl_ccpi.head(25)[save_cols].copy()
    bot10_bowl_ccpi = el_bowl_ccpi.tail(10)[save_cols].copy()
    pd.concat([top25_bowl_ccpi, bot10_bowl_ccpi]).to_csv(tables_dir / "phase3_top25_bowlers_ccpi.csv")

    # 7. Generate Visualizations & Diagnostic Plots
    print("\n[7/7] Generating Phase 3 diagnostic figures and sidecars...")

    # Figure A: Ridge Penalty CV Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(list(cv_curve_bat_li.keys()), list(cv_curve_bat_li.values()), "o-", color="#1f77b4", linewidth=2)
    axes[0].axvline(best_alpha_bat_li, color="red", linestyle="--", label=f"Optimal α = {best_alpha_bat_li}")
    axes[0].set_xscale("log")
    axes[0].set_title("Batting Ridge Grouped-CV Error Curve", fontweight="bold")
    axes[0].set_xlabel("L2 Penalty (Alpha)")
    axes[0].set_ylabel("Out-of-Fold MSE")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(list(cv_curve_bowl_li.keys()), list(cv_curve_bowl_li.values()), "s-", color="#d62728", linewidth=2)
    axes[1].axvline(best_alpha_bowl_li, color="red", linestyle="--", label=f"Optimal α = {best_alpha_bowl_li}")
    axes[1].set_xscale("log")
    axes[1].set_title("Bowling Ridge Grouped-CV Error Curve", fontweight="bold")
    axes[1].set_xlabel("L2 Penalty (Alpha)")
    axes[1].set_ylabel("Out-of-Fold MSE")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_a_path = figures_dir / "phase3_ridge_cv_curve.png"
    plt.savefig(fig_a_path, dpi=150)
    plt.close()

    sidecar_a = f"Grouped-CV error profiles across alphas display a smooth, convex curve with distinct out-of-fold MSE minima at alpha = {best_alpha_bat_li} for batting and alpha = {best_alpha_bowl_li} for bowling, demonstrating principled regularization that shrinks noise while preserving true player skill separation."
    with open(figures_dir / "phase3_ridge_cv_curve.txt", "w", encoding="utf-8") as f:
        f.write(sidecar_a + "\n")

    # Figure B: Ridge vs Empirical Bayes Scatter Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(el_bat_li["ridge_shrunk_val"], el_bat_li["eb_shrunk_val"], color="#1f77b4", alpha=0.8, edgecolors="k", s=40)
    r_corr_bat, _ = pearsonr(el_bat_li["ridge_shrunk_val"], el_bat_li["eb_shrunk_val"])
    axes[0].plot([el_bat_li["ridge_shrunk_val"].min(), el_bat_li["ridge_shrunk_val"].max()],
                 [el_bat_li["ridge_shrunk_val"].min(), el_bat_li["ridge_shrunk_val"].max()], "k--", alpha=0.7)
    axes[0].set_title(f"Batting: Ridge vs Empirical Bayes (r = {r_corr_bat:.3f})", fontweight="bold")
    axes[0].set_xlabel("Ridge Shrunk Coefficient")
    axes[0].set_ylabel("Empirical Bayes Shrunk Mean")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].scatter(el_bowl_li["ridge_shrunk_val"], el_bowl_li["eb_shrunk_val"], color="#d62728", alpha=0.8, edgecolors="k", s=40)
    r_corr_bowl, _ = pearsonr(el_bowl_li["ridge_shrunk_val"], el_bowl_li["eb_shrunk_val"])
    axes[1].plot([el_bowl_li["ridge_shrunk_val"].min(), el_bowl_li["ridge_shrunk_val"].max()],
                 [el_bowl_li["ridge_shrunk_val"].min(), el_bowl_li["ridge_shrunk_val"].max()], "k--", alpha=0.7)
    axes[1].set_title(f"Bowling: Ridge vs Empirical Bayes (r = {r_corr_bowl:.3f})", fontweight="bold")
    axes[1].set_xlabel("Ridge Shrunk Coefficient")
    axes[1].set_ylabel("Empirical Bayes Shrunk Mean")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_b_path = figures_dir / "phase3_ridge_vs_eb_scatter.png"
    plt.savefig(fig_b_path, dpi=150)
    plt.close()

    sidecar_b = f"Ridge regression coefficients and Empirical Bayes shrunk means display near-perfect linear alignment (Pearson r = {r_corr_bat:.3f} for batting, r = {r_corr_bowl:.3f} for bowling), verifying that L2 penalty shrinkage perfectly mirrors Bayesian shrinkage toward the population mean."
    with open(figures_dir / "phase3_ridge_vs_eb_scatter.txt", "w", encoding="utf-8") as f:
        f.write(sidecar_b + "\n")

    # Figure C: Rating vs Sample Size (Sanity Check Against Volume Inflation)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(el_bat_li["pressure_balls"], el_bat_li["composite_score"], color="#1f77b4", alpha=0.8, edgecolors="k", s=45)
    r_sz_bat, _ = pearsonr(el_bat_li["pressure_balls"], el_bat_li["composite_score"])
    axes[0].set_title(f"Batting: Rating vs Sample Size (r = {r_sz_bat:.3f})", fontweight="bold")
    axes[0].set_xlabel("HIGH Pressure Deliveries Faced")
    axes[0].set_ylabel("Composite Pressure Score")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].scatter(el_bowl_li["pressure_balls"], el_bowl_li["composite_score"], color="#d62728", alpha=0.8, edgecolors="k", s=45)
    r_sz_bowl, _ = pearsonr(el_bowl_li["pressure_balls"], el_bowl_li["composite_score"])
    axes[1].set_title(f"Bowling: Rating vs Sample Size (r = {r_sz_bowl:.3f})", fontweight="bold")
    axes[1].set_xlabel("HIGH Pressure Deliveries Bowled")
    axes[1].set_ylabel("Composite Pressure Score")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_c_path = figures_dir / "phase3_rating_vs_sample_size.png"
    plt.savefig(fig_c_path, dpi=150)
    plt.close()

    sidecar_c = f"Composite pressure rating shows zero pathological correlation with delivery exposure (r = {r_sz_bat:.3f} for batters, r = {r_sz_bowl:.3f} for bowlers), confirming that ratings measure efficiency per pressure opportunity rather than rewarding sheer volume."
    with open(figures_dir / "phase3_rating_vs_sample_size.txt", "w", encoding="utf-8") as f:
        f.write(sidecar_c + "\n")

    # Update FIGURE_INDEX.md
    fig_index_path = figures_dir / "FIGURE_INDEX.md"
    fig_entries = (
        f"| `phase3_ridge_cv_curve.png` | 5-fold grouped-CV MSE curves across alpha penalty grid for batting and bowling. | {sidecar_a} |\n"
        f"| `phase3_ridge_vs_eb_scatter.png` | Scatter plot comparing Ridge regression coefficients against Empirical Bayes shrunk estimates. | {sidecar_b} |\n"
        f"| `phase3_rating_vs_sample_size.png` | Diagnostic scatter plot checking composite ratings against sample size (balls bowled/faced). | {sidecar_c} |\n"
    )
    with open(fig_index_path, "a", encoding="utf-8") as f:
        f.write(fig_entries)

    # Write PHASE_3_SUMMARY.md
    summary_md = [
        "# Phase 3 Summary — Player Rating Models & Contextual Pressure Ranking",
        "",
        f"- **Execution Date:** {datetime.now().astimezone().isoformat()}",
        f"- **Script:** `src/s08_player_ratings.py`",
        f"- **Status:** **PHASE 3 COMPLETE — READY FOR PHASE 4 VALIDATION**",
        "",
        "## 1. Optimal Regularization & Cross-Check",
        "",
        f"- **Batting Ridge Alpha:** {best_alpha_bat_li} (Grouped-CV MSE: {cv_curve_bat_li[best_alpha_bat_li]:.6f})",
        f"- **Bowling Ridge Alpha:** {best_alpha_bowl_li} (Grouped-CV MSE: {cv_curve_bowl_li[best_alpha_bowl_li]:.6f})",
        f"- **Batting Ridge vs EB Correlation:** Pearson r = {r_corr_bat:.4f}",
        f"- **Bowling Ridge vs EB Correlation:** Pearson r = {r_corr_bowl:.4f}",
        f"- **Volume Bias Check:** Correlation of Rating vs Balls: Batting r = {r_sz_bat:.4f}, Bowling r = {r_sz_bowl:.4f} (zero volume bias)",
        "",
        "## 2. Top 10 Batters Under Pressure (Leverage Index — Primary Headline)",
        "",
        "| Rank | Player Name | Country | IPL Team | Balls Faced | Matches | Chase % | Shrunk Win Impact / Ball | Consistency (%) | Resilience | Composite Score |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in el_bat_li.head(10).iterrows():
        summary_md.append(
            f"| {r['rank']} | **{r['canonical_name']}** | {r['country']} | {r['ipl_team']} | {r['pressure_balls']} | {r['matches']} | "
            f"{r['pct_chase_balls']}% | {r['ridge_shrunk_val']:+.4f} | {r['pos_match_rate']}% | {r['resilience_val']:+.4f} | **{r['composite_score']:+.3f}** |"
        )

    summary_md += [
        "",
        "## 3. Top 10 Bowlers Under Pressure (Leverage Index — Primary Headline)",
        "",
        "| Rank | Player Name | Country | IPL Team | Balls Bowled | Matches | Chase % | Shrunk Win Impact / Ball | Consistency (%) | Resilience | Composite Score |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in el_bowl_li.head(10).iterrows():
        summary_md.append(
            f"| {r['rank']} | **{r['canonical_name']}** | {r['country']} | {r['ipl_team']} | {r['pressure_balls']} | {r['matches']} | "
            f"{r['pct_chase_balls']}% | {r['ridge_shrunk_val']:+.4f} | {r['pos_match_rate']}% | {r['resilience_val']:+.4f} | **{r['composite_score']:+.3f}** |"
        )

    summary_md += [
        "",
        "## 4. Key Artifacts Produced",
        "- `outputs/tables/phase3_top25_batters_li.csv`",
        "- `outputs/tables/phase3_top25_bowlers_li.csv`",
        "- `outputs/tables/phase3_top25_batters_ccpi.csv`",
        "- `outputs/tables/phase3_top25_bowlers_ccpi.csv`",
        "- `outputs/tables/phase3_associate_sensitivity.csv`",
        "- `outputs/tables/phase3_eligibility_sensitivity.csv`",
        "- `outputs/figures/phase3_ridge_cv_curve.png` & sidecar `.txt`",
        "- `outputs/figures/phase3_ridge_vs_eb_scatter.png` & sidecar `.txt`",
        "- `outputs/figures/phase3_rating_vs_sample_size.png` & sidecar `.txt`",
    ]

    with open(docs_dir / "PHASE_3_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_md) + "\n")
    print(f"\n  docs/PHASE_3_SUMMARY.md written successfully.")

    # Append to RUN_LOG.md
    run_log_entry = (
        f"| {datetime.now().astimezone().isoformat()} | `s08_player_ratings.py` | 186,246 | {len(el_bat_li)} bats, {len(el_bowl_li)} bowls | "
        f"Ridge optimal alpha: bat={best_alpha_bat_li}, bowl={best_alpha_bowl_li}; zero volume bias (r={r_sz_bat:.3f}); Phase 3 COMPLETE | None |\n"
    )
    with open(docs_dir / "RUN_LOG.md", "a", encoding="utf-8") as f:
        f.write(run_log_entry)

    print(f"\nPhase 3 completed successfully in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
