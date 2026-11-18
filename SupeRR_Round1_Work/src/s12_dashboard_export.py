#!/usr/bin/env python3
"""
s12_dashboard_export.py — Export the small, precomputed data files used by the Streamlit dashboard.

No new methodology: every rating is produced by the FINAL definition implemented in
s11_final_validation.py (itself identical to s10). The script asserts that the submitted top-10
order in handoff/TOP10_*.csv is reproduced before writing anything.

Output: dashboard/data/*.csv (+ manifest.json). Validation tables are copied verbatim from outputs/tables.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import s11_final_validation as s11

WORK = Path(__file__).resolve().parents[1]
TABLES = WORK / "outputs" / "tables"
OUT = WORK / "dashboard" / "data"
OUT.mkdir(parents=True, exist_ok=True)

COUNTRY_FIX = {4219: "India"}  # Krunal Pandya: national_side blank in players_master; India international


def clean_country(x: str) -> str:
    if not isinstance(x, str) or not x.strip():
        return ""
    parts = [p.strip() for p in x.split(",")]
    senior = [p for p in parts if not p.endswith(" A")]
    return (senior or parts)[0]


def main():
    balls, matches = s11.load_balls()
    players = pd.read_csv(WORK / "outputs" / "processed" / "players_master.csv").set_index("player_id")
    high, ordy, hi_cut, ord_cut = s11.tiers(balls)

    sub = {"bat": pd.read_csv(WORK / "handoff" / "TOP10_BATTERS.csv"), "bowl": pd.read_csv(WORK / "handoff" / "TOP10_BOWLERS.csv")}
    boot = pd.read_csv(TABLES / "final_bootstrap_rank_intervals.csv")
    match_info = matches.set_index("match_id")

    for role, label in (("bat", "batters"), ("bowl", "bowlers")):
        res = s11.fit_player_models(high, ordy, role)
        assert res.head(10).index.tolist() == sub[role]["player_id"].tolist(), f"{role} order mismatch"
        # cross-check numeric values against handoff
        h = sub[role].set_index("player_id")
        for col in ("pressure_rating", "ordinary_rating", "resilience", "consistency", "composite", "pressure_balls", "matches"):
            assert np.allclose(res.loc[h.index, col].astype(float), h[col].astype(float), rtol=1e-6, atol=1e-9), f"{role} {col} mismatch"

        df = res.copy()
        df.index.name = "player_id"
        df["rank"] = np.arange(1, len(df) + 1)
        df["submitted_top10"] = df["rank"] <= 10
        df["name"] = df.index.map(players["canonical_name"])
        df["country"] = [COUNTRY_FIX.get(p, clean_country(players.loc[p, "national_side"]) if p in players.index else "") for p in df.index]
        df["ipl_teams"] = df.index.map(players["ipl_franchises"]).fillna("")
        df["pace_or_spin"] = df.index.map(players["pace_or_spin"]).fillna("")
        b = boot[boot["role"] == role].set_index("player_id")
        for c in ("median_boot_rank", "p5", "p95", "top10_prob_pct", "top20_prob_pct"):
            df[f"boot_{c}"] = df.index.map(b[c]) if len(b) else np.nan

        # box scores, HIGH vs ORDINARY (same definitions as s11)
        rows = []
        for pid in df.index:
            rec = {"player_id": pid}
            for tier_name, t in (("high", high), ("ord", ordy)):
                if role == "bat":
                    d = t[(t["striker_id"] == pid) & ~t["is_wide"]]
                    outs = int((d["dismissed_player_id"] == pid).sum())
                    n = len(d)
                    rec[f"{tier_name}_balls"] = n
                    rec[f"{tier_name}_runs"] = int(d["runs"].sum())
                    rec[f"{tier_name}_dismissals"] = outs
                    rec[f"{tier_name}_sr"] = d["runs"].sum() / n * 100 if n else np.nan
                    rec[f"{tier_name}_balls_per_dismissal"] = n / outs if outs else np.nan
                    rec[f"{tier_name}_boundary_pct"] = d["is_boundary"].mean() * 100 if n else np.nan
                    rec[f"{tier_name}_dot_pct"] = (d["runs"] == 0).mean() * 100 if n else np.nan
                else:
                    d = t[t["bowler_id"] == pid]
                    legal = int(d["is_legal"].sum())
                    w = int(d["bowler_credited_wicket"].sum())
                    rec[f"{tier_name}_balls"] = legal
                    rec[f"{tier_name}_runs"] = float(d["bowler_runs"].sum())
                    rec[f"{tier_name}_wickets"] = w
                    rec[f"{tier_name}_economy"] = d["bowler_runs"].sum() / legal * 6 if legal else np.nan
                    rec[f"{tier_name}_balls_per_wicket"] = legal / w if w else np.nan
                    rec[f"{tier_name}_dot_pct"] = ((d["runs"] == 0) & d["is_legal"] & ~d["is_wicket"]).sum() / legal * 100 if legal else np.nan
            rows.append(rec)
        box = pd.DataFrame(rows).set_index("player_id")
        df = df.join(box)
        cols = ["rank", "name", "country", "ipl_teams", "pace_or_spin", "submitted_top10", "pressure_balls", "matches", "pct_chase",
                "pressure_rating", "ordinary_rating", "resilience", "consistency", "composite"] + \
               [c for c in df.columns if c.startswith("boot_")] + list(box.columns)
        df[cols].reset_index().to_csv(OUT / f"ratings_{label}.csv", index=False)

        # per-match net pressure WPA (drives the "consistency" component) for every eligible player
        pid_col = "striker_id" if role == "bat" else "bowler_id"
        sign = 1.0 if role == "bat" else -1.0
        src = high[~high["is_wide"]] if role == "bat" else high
        pm = (src[src[pid_col].isin(df.index)].assign(wpa=lambda x: x["delta_wp"] * sign)
              .groupby([pid_col, "match_id"]).agg(balls=("wpa", "size"), net_wpa=("wpa", "sum"), mean_wpa=("wpa", "mean")).reset_index()
              .rename(columns={pid_col: "player_id"}))
        pm["match_date"] = pm["match_id"].map(match_info["match_date"]).astype(str).str[:10]
        pm["competition"] = pm["match_id"].map(match_info["comp_name"])
        pm["positive"] = pm["mean_wpa"] > 0
        pm.to_csv(OUT / f"match_pressure_{label}.csv", index=False)

    # --- Pressure tier profile and LI distributions ---
    shutil.copy(TABLES / "final_tier_profile.csv", OUT / "tier_profile.csv")
    live = balls[balls["is_live"]].copy()
    live["over"] = (live["legal_balls_before"] // 6 + 1).clip(upper=20)
    by_over = (live.assign(early=live["legal_balls_before"] < 12, high=live["leverage_index"] >= hi_cut)
               .groupby(["innings", "over"]).agg(deliveries=("leverage_index", "size"), mean_li=("leverage_index", "mean"),
                                                  median_li=("leverage_index", "median"), share_high_pct=("high", "mean"), early_excluded=("early", "all"))
               .reset_index())
    by_over["share_high_pct"] *= 100
    by_over.to_csv(OUT / "li_by_over.csv", index=False)

    cat = np.select(
        [~balls["is_live"], balls["legal_balls_before"] < 12, balls["leverage_index"] >= hi_cut, balls["leverage_index"] < ord_cut],
        ["Dead / decided", "Early innings (first 12 balls)", "HIGH", "ORDINARY"], default="Middle (unused)")
    edges = np.round(np.arange(0, 6.01, 0.1), 2)
    li_c = balls["leverage_index"].clip(upper=5.999)
    hist = (pd.DataFrame({"category": cat, "bin": pd.cut(li_c, edges, right=False, labels=edges[:-1])})
            .groupby(["category", "bin"], observed=False).size().reset_index(name="deliveries"))
    hist.to_csv(OUT / "li_histogram.csv", index=False)
    pd.DataFrame({"category": cat}).value_counts().reset_index(name="deliveries").to_csv(OUT / "delivery_categories.csv", index=False)

    # Real observed example states (chase, 13-17 needed off 11-13 balls)
    ex = balls[(balls["innings"] == 2) & balls["runs_needed"].between(13, 17) & balls["balls_left"].between(11, 13)]
    ex.groupby("wickets_left").agg(deliveries=("leverage_index", "size"), mean_li=("leverage_index", "mean"),
                                   mean_win_prob=("win_prob", "mean")).reset_index().to_csv(OUT / "example_states.csv", index=False)

    # Real WPA examples: biggest HIGH-pressure ball for the #1 batter and #1 bowler
    exs = []
    for role, pid in (("bat", int(sub["bat"].iloc[0]["player_id"])), ("bowl", int(sub["bowl"].iloc[0]["player_id"]))):
        col = "striker_id" if role == "bat" else "bowler_id"
        sign = 1.0 if role == "bat" else -1.0
        d = high[high[col] == pid].assign(credit=lambda x: x["delta_wp"] * sign).sort_values("credit", ascending=False).iloc[0]
        m = match_info.loc[d["match_id"]]
        exs.append({"role": role, "player_id": pid, "player": players.loc[pid, "canonical_name"], "match_id": int(d["match_id"]),
                    "match_date": str(m["match_date"])[:10], "competition": m["comp_name"], "match": f"{m['home_team']} v {m['away_team']}",
                    "innings": int(d["innings"]), "over": int(d["legal_balls_before"] // 6 + 1), "ball_in_over": int(d["legal_balls_before"] % 6 + 1),
                    "striker": d["striker_name"], "bowler": d["bowler_name"],
                    "runs_needed": d["runs_needed"] if d["innings"] == 2 else np.nan, "balls_left": int(d["balls_left"]), "wickets_left": int(d["wickets_left"]),
                    "outcome": "Wicket" if d["is_wicket"] else f"{int(d['runs'])} run(s)" + (" (extra)" if d["extras_type"] in ("wide", "Overstep") else ""),
                    "leverage_index": d["leverage_index"], "wp_before": d["win_prob"], "wp_after": d["next_wp"], "delta_wp_batting": d["delta_wp"]})
    pd.DataFrame(exs).to_csv(OUT / "wpa_examples.csv", index=False)

    # --- Verbatim copies of validated tables ---
    copies = {
        "phase1_calibration_bins.csv": "calibration_bins.csv",
        "phase1_calibration_subgroups.csv": "calibration_subgroups.csv",
        "final_bootstrap_rank_intervals.csv": "bootstrap_rank_intervals.csv",
        "final_variants_comparison.csv": "variants_comparison.csv",
        "final_weight_sensitivity.csv": "weight_sensitivity.csv",
        "final_split_half_reliability.csv": "split_half_reliability.csv",
        "final_population_reference.csv": "population_reference.csv",
        "competition_tier_comparability.csv": "competition_tiers.csv",
        "phase2_li_vs_ccpi_metrics.csv": "ccpi_comparison.csv",
        "phase2_close_vs_lopsided.csv": "close_vs_lopsided.csv",
        "phase2_weight_calibration_r2.csv": "ccpi_weight_r2.csv",
        "phase0_gate0_report.csv": "data_quality_checks.csv",
        "data_edits_log.csv": "data_edits_log.csv",
        "player_duplicate_names.csv": "duplicate_names.csv",
    }
    for src, dst in copies.items():
        shutil.copy(TABLES / src, OUT / dst)
    shutil.copy(WORK / "outputs" / "processed" / "extras_calibration.json", OUT / "extras_calibration.json")

    # Match exclusions: documented in docs/PHASE_0_SUMMARY.md (s05 run of 2026-11-14 18:29)
    pd.DataFrame([
        {"reason": "A-team / development events", "matches": 30},
        {"reason": "Rain-affected / DLS / revised target", "matches": 26},
        {"reason": "No result / abandoned", "matches": 25},
        {"reason": "Ball-by-ball records missing", "matches": 16},
        {"reason": "Tied and decided by Super Over", "matches": 6},
    ]).to_csv(OUT / "match_exclusions.csv", index=False)
    tiers_count = matches.groupby("comp_tier").size().rename("matches").reset_index()
    tiers_count.to_csv(OUT / "matches_by_tier.csv", index=False)
    matches.groupby("season_year").size().rename("matches").reset_index().to_csv(OUT / "matches_by_season.csv", index=False)

    manifest = {
        "generated_by": "src/s12_dashboard_export.py",
        "definition": "Live (5%<=WP<=95%), legal_balls_before>=12; HIGH = top 20% LI, ORDINARY = bottom 50% LI",
        "high_cut": float(hi_cut), "ordinary_cut": float(ord_cut),
        "high_deliveries": int(len(high)), "ordinary_deliveries": int(len(ordy)),
        "matches": int(len(matches)), "deliveries": int(len(balls)),
        "ipl_matches": int((matches["comp_tier"] == "IPL").sum()), "t20i_matches": int((matches["comp_tier"] != "IPL").sum()),
        "date_min": str(matches["match_date"].min())[:10], "date_max": str(matches["match_date"].max())[:10],
        "eligible_batters": int(pd.read_csv(OUT / "ratings_batters.csv").shape[0]),
        "eligible_bowlers": int(pd.read_csv(OUT / "ratings_bowlers.csv").shape[0]),
        "brier_overall": 0.1765, "brier_naive": 0.2499,
        "note_brier": "From docs/PHASE_1_SUMMARY.md (s06 out-of-fold evaluation); per-innings Brier in calibration_subgroups.csv",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print("files:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
