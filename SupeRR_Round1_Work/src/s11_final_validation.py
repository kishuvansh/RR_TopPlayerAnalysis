#!/usr/bin/env python3
"""
s11_final_validation.py — Re-run all Phase 4 validation on the FINAL (s10 / D18) definition.

Final definition (identical to s10):
  - Eligible states: live (0.05 <= win_prob <= 0.95) AND legal_balls_before >= 12.
  - HIGH     = top 20% of Leverage Index within eligible states.
  - ORDINARY = bottom 50% of Leverage Index within eligible states.
  - Ratings  = ridge (player + opponent + context controls) on signed delta win-probability.
  - Composite = 0.50 z(pressure) + 0.25 z(consistency) + 0.25 z(resilience); eligibility 60 balls / 15 matches.

This script does NOT change the submitted top-10 order (it re-derives it and asserts it matches
handoff/TOP10_*.csv). It produces:
  1. Proper match-level cluster bootstrap (resampled WITH replacement, duplicated matches kept).
  2. Split-half reliability (odd/even match id, 2022-23 vs 2024-26) and predictive validity.
  3. Definition variants (unrestricted early balls, chase-only, HIGH cut at top 15% / 25%).
  4. Composite weight sensitivity (200 random weight draws).
  5. Raw box-score evidence for each finalist, HIGH vs ORDINARY.
  6. Figures for the methodology report.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

WORK = Path(__file__).resolve().parents[1]
PROC = WORK / "outputs" / "processed"
TABLES = WORK / "outputs" / "tables"
FIGS = WORK / "outputs" / "figures"
HANDOFF = WORK / "handoff"
SEED = 42
N_BOOT = 200
ALPHA = {"bat": 300.0, "bowl": 500.0}


def load_balls():
    balls = pd.read_parquet(PROC / "balls_with_pressure.parquet")
    matches = pd.read_parquet(PROC / "matches_clean.parquet")
    m = matches.set_index("match_id")
    venue_map = m.groupby("venue_id")["first_inn_est_total"].mean()
    balls["venue_baseline"] = balls["venue_id"].map(venue_map).fillna(m["first_inn_est_total"].mean())

    balls = balls.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    balls["next_wp"] = balls.groupby(["match_id", "innings"])["win_prob"].shift(-1)
    last2 = (balls["innings"] == 2) & balls["next_wp"].isna()
    balls.loc[last2, "next_wp"] = balls.loc[last2, "batting_team_won"].astype(float)
    b2_first = balls[(balls["innings"] == 2) & (balls["innings_ball_num"] == 1)].set_index("match_id")["win_prob"]
    last1 = (balls["innings"] == 1) & balls["next_wp"].isna()
    balls.loc[last1, "next_wp"] = 1.0 - balls.loc[last1, "match_id"].map(b2_first)
    balls["delta_wp"] = balls["next_wp"] - balls["win_prob"]

    balls["bat_order_band"] = pd.cut(balls["batting_order"], bins=[0, 3, 7, 12], labels=["Top_1-3", "Mid_4-7", "Low_8-11"])
    balls["is_spin"] = balls["bowling_category"].str.contains("Spin|Orthodox", case=False, na=False).astype(float)
    # Box-score helpers
    balls["is_legal"] = ~(balls["is_wide"] | balls["is_no_ball"] | balls["extras_type"].isin(["other", "Penalty"]))
    pen = balls["extras_type"].isin(["wide", "Overstep"])
    balls["bowler_runs"] = balls["runs"] + np.where(pen, balls["calibrated_extra_run"].fillna(0), 0.0)
    return balls, matches


def tiers(balls, high_q=0.8, ord_q=0.5, min_legal_before=12, innings=None):
    elig = balls[balls["is_live"] & (balls["legal_balls_before"] >= min_legal_before)]
    if innings is not None:
        elig = elig[elig["innings"] == innings]
    # Cutoffs always from the full eligible live pool (as in s10)
    pool = balls[balls["is_live"] & (balls["legal_balls_before"] >= min_legal_before)]["leverage_index"]
    hi_cut, ord_cut = pool.quantile(high_q), pool.quantile(ord_q)
    return elig[elig["leverage_index"] >= hi_cut].copy(), elig[elig["leverage_index"] < ord_cut].copy(), hi_cut, ord_cut


def fit_player_models(h_df, o_df, role, min_balls=60, min_matches=15, weights=(0.5, 0.25, 0.25), return_all=False):
    pid_col = "striker_id" if role == "bat" else "bowler_id"
    opp_col = "bowler_id" if role == "bat" else "striker_id"
    sign = 1.0 if role == "bat" else -1.0
    if role == "bat":
        h_df = h_df[~h_df["is_wide"]]
        o_df = o_df[~o_df["is_wide"]]

    def fit_ridge(df):
        ctrl = pd.get_dummies(df[["innings", "comp_tier_pooled", "bat_order_band"]], drop_first=True, dtype=float)
        ctrl_f = np.column_stack([ctrl.values, df["is_spin"].values, df["venue_baseline"].values])
        p_d = pd.get_dummies(df[pid_col], prefix="p", dtype=float)
        o_d = pd.get_dummies(df[opp_col], prefix="o", dtype=float)
        X = sparse.hstack([sparse.csr_matrix(p_d.values), sparse.csr_matrix(o_d.values), sparse.csr_matrix(ctrl_f)], format="csr")
        clf = Ridge(alpha=ALPHA[role], fit_intercept=True)
        clf.fit(X, df["delta_wp"].values * sign)
        cols = p_d.columns.tolist()
        return {int(c.replace("p_", "")): v for c, v in zip(cols, clf.coef_[: len(cols)])}

    p_coefs, o_coefs = fit_ridge(h_df), fit_ridge(o_df)
    g = h_df.groupby(pid_col)
    match_key = "boot_mid" if "boot_mid" in h_df.columns else "match_id"
    mm = h_df.assign(s=h_df["delta_wp"] * sign).groupby([pid_col, match_key])["s"].mean()
    df_s = pd.DataFrame({
        "pressure_balls": g.size(),
        "matches": g[match_key].nunique(),
        "pct_chase": g["innings"].apply(lambda x: (x == 2).mean() * 100).round(1),
        "consistency": (mm > 0).groupby(level=0).mean() * 100.0,
    })
    df_s["pressure_rating"] = df_s.index.map(lambda p: p_coefs.get(p, 0.0))
    df_s["ordinary_rating"] = df_s.index.map(lambda p: o_coefs.get(p, 0.0))
    df_s["resilience"] = df_s["pressure_rating"] - df_s["ordinary_rating"]
    el = (df_s["pressure_balls"] >= min_balls) & (df_s["matches"] >= min_matches)
    sc = StandardScaler().fit_transform(df_s.loc[el, ["pressure_rating", "consistency", "resilience"]])
    df_s.loc[el, "composite"] = weights[0] * sc[:, 0] + weights[1] * sc[:, 1] + weights[2] * sc[:, 2]
    out = df_s[el].sort_values("composite", ascending=False)
    return (out, df_s) if return_all else out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    balls, matches = load_balls()
    players = pd.read_csv(PROC / "players_master.csv").set_index("player_id")
    name = players["canonical_name"].to_dict()

    high, ordy, hi_cut, ord_cut = tiers(balls)
    print(f"HIGH cut LI>={hi_cut:.3f}  ORDINARY cut LI<{ord_cut:.3f}  HIGH balls={len(high):,}  ORD balls={len(ordy):,}")

    res = {r: fit_player_models(high, ordy, r) for r in ("bat", "bowl")}
    sub = {"bat": pd.read_csv(HANDOFF / "TOP10_BATTERS.csv"), "bowl": pd.read_csv(HANDOFF / "TOP10_BOWLERS.csv")}
    for r in ("bat", "bowl"):
        got = res[r].head(10).index.tolist()
        exp = sub[r]["player_id"].tolist()
        assert got == exp, f"{r}: re-derived top-10 {got} != submitted {exp}"
    print("Submitted top-10 order reproduced exactly for batting and bowling.")

    # --- Tier summary table (pressure vs ordinary) ---
    def mix(df, lab):
        legal = df[df["is_legal"]]
        return {"Tier": lab, "Deliveries": len(df), "Mean_LI": round(df["leverage_index"].mean(), 3),
                "Share_Innings2_pct": round((df["innings"] == 2).mean() * 100, 1),
                "Share_Death_Overs_pct": round((df["phase"].astype(str).str.contains("Death", case=False)).mean() * 100, 1),
                "Runs_per_Over": round(df["bowler_runs"].sum() / len(legal) * 6, 2),
                "Dot_pct": round(((legal["runs"] == 0) & ~legal["is_wicket"]).mean() * 100, 1),
                "Boundary_pct": round(legal["is_boundary"].mean() * 100, 1),
                "Wicket_pct": round(legal["is_wicket"].mean() * 100, 2),
                "Mean_abs_dWP": round(df["delta_wp"].abs().mean(), 4)}
    mid = balls[balls["is_live"] & (balls["legal_balls_before"] >= 12) & (balls["leverage_index"] >= ord_cut) & (balls["leverage_index"] < hi_cut)]
    dead = balls[~balls["is_live"]]
    tier_tbl = pd.DataFrame([mix(high, "HIGH pressure (top 20% live LI)"), mix(mid, "Middle (50th-80th pct, unused)"),
                             mix(ordy, "ORDINARY (bottom 50% live LI)"), mix(dead, "Dead / decided (WP<5% or >95%)")])
    tier_tbl.to_csv(TABLES / "final_tier_profile.csv", index=False)
    print(tier_tbl.to_string(index=False))

    # --- Box-score evidence per finalist ---
    def bat_box(df, pid):
        d = df[(df["striker_id"] == pid) & ~df["is_wide"]]
        outs = (d["dismissed_player_id"] == pid).sum()
        return len(d), d["runs"].sum() / max(len(d), 1) * 100, len(d) / max(outs, 1) if outs else np.nan, d["is_boundary"].mean() * 100, (d["runs"] == 0).mean() * 100
    def bowl_box(df, pid):
        d = df[df["bowler_id"] == pid]
        legal = d["is_legal"].sum()
        w = d["bowler_credited_wicket"].sum()
        return legal, d["bowler_runs"].sum() / max(legal, 1) * 6, legal / w if w else np.nan, ((d["runs"] == 0) & d["is_legal"] & ~d["is_wicket"]).sum() / max(legal, 1) * 100
    rows = []
    for _, r in sub["bat"].iterrows():
        pid = int(r["player_id"])
        hb, hsr, hbpd, hbd, hdot = bat_box(high, pid)
        ob, osr, obpd, obd, odot = bat_box(ordy, pid)
        rows.append({"rank": r["rank"], "name": r["name"], "player_id": pid, "HIGH_balls": hb, "HIGH_SR": round(hsr, 1),
                     "HIGH_balls_per_dismissal": round(hbpd, 1) if pd.notna(hbpd) else None, "HIGH_boundary_pct": round(hbd, 1), "HIGH_dot_pct": round(hdot, 1),
                     "ORD_balls": ob, "ORD_SR": round(osr, 1), "ORD_balls_per_dismissal": round(obpd, 1) if pd.notna(obpd) else None,
                     "ORD_boundary_pct": round(obd, 1), "ORD_dot_pct": round(odot, 1)})
    bat_box_df = pd.DataFrame(rows)
    bat_box_df.to_csv(TABLES / "final_top10_batters_boxscore.csv", index=False)
    rows = []
    for _, r in sub["bowl"].iterrows():
        pid = int(r["player_id"])
        hl, hec, hsr, hdot = bowl_box(high, pid)
        ol, oec, osr, odot = bowl_box(ordy, pid)
        rows.append({"rank": r["rank"], "name": r["name"], "player_id": pid, "HIGH_legal_balls": hl, "HIGH_economy": round(hec, 2),
                     "HIGH_balls_per_wicket": round(hsr, 1) if pd.notna(hsr) else None, "HIGH_dot_pct": round(hdot, 1),
                     "ORD_legal_balls": ol, "ORD_economy": round(oec, 2), "ORD_balls_per_wicket": round(osr, 1) if pd.notna(osr) else None,
                     "ORD_dot_pct": round(odot, 1)})
    bowl_box_df = pd.DataFrame(rows)
    bowl_box_df.to_csv(TABLES / "final_top10_bowlers_boxscore.csv", index=False)
    # Population reference
    pop = {"bat_HIGH_SR": high[~high["is_wide"]]["runs"].mean() * 100, "bat_ORD_SR": ordy[~ordy["is_wide"]]["runs"].mean() * 100,
           "bowl_HIGH_econ": high["bowler_runs"].sum() / high["is_legal"].sum() * 6, "bowl_ORD_econ": ordy["bowler_runs"].sum() / ordy["is_legal"].sum() * 6}
    pd.DataFrame([pop]).round(2).to_csv(TABLES / "final_population_reference.csv", index=False)
    print(bat_box_df.to_string(index=False)); print(bowl_box_df.to_string(index=False)); print(pop)

    # --- 1. Cluster bootstrap with replacement ---
    mids = np.array(sorted(set(high["match_id"]) | set(ordy["match_id"])))
    hi_idx = high.groupby("match_id").indices
    or_idx = ordy.groupby("match_id").indices
    top20 = {r: res[r].head(20).index.tolist() for r in ("bat", "bowl")}
    boot = {r: {p: [] for p in top20[r]} for r in ("bat", "bowl")}
    for b in range(N_BOOT):
        draw = rng.choice(mids, size=len(mids), replace=True)
        hi_parts, or_parts, hi_keys, or_keys = [], [], [], []
        for k, m in enumerate(draw):
            if m in hi_idx:
                hi_parts.append(hi_idx[m]); hi_keys.append(np.full(len(hi_idx[m]), k))
            if m in or_idx:
                or_parts.append(or_idx[m]); or_keys.append(np.full(len(or_idx[m]), k))
        hb = high.iloc[np.concatenate(hi_parts)].assign(boot_mid=np.concatenate(hi_keys))
        ob = ordy.iloc[np.concatenate(or_parts)].assign(boot_mid=np.concatenate(or_keys))
        for r in ("bat", "bowl"):
            fit = fit_player_models(hb, ob, r)
            rk = fit["composite"].rank(ascending=False)
            for p in top20[r]:
                boot[r][p].append(rk.get(p, len(fit) + 10))
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b+1}/{N_BOOT}  ({time.time()-t0:.0f}s)")
    boot_rows = []
    for r in ("bat", "bowl"):
        for i, p in enumerate(top20[r], 1):
            a = np.array(boot[r][p])
            boot_rows.append({"role": r, "rank": i, "player_id": p, "name": name.get(p), "median_boot_rank": float(np.median(a)),
                              "p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
                              "top10_prob_pct": round(float((a <= 10).mean() * 100), 1), "top20_prob_pct": round(float((a <= 20).mean() * 100), 1)})
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(TABLES / "final_bootstrap_rank_intervals.csv", index=False)
    print(boot_df.to_string(index=False))

    # --- 2. Split-half reliability & predictive validity ---
    def split_corr(a_ids, b_ids, mb, mm_):
        out = {}
        for r in ("bat", "bowl"):
            A = fit_player_models(high[high["match_id"].isin(a_ids)], ordy[ordy["match_id"].isin(a_ids)], r, mb, mm_)
            B = fit_player_models(high[high["match_id"].isin(b_ids)], ordy[ordy["match_id"].isin(b_ids)], r, mb, mm_)
            com = A.index.intersection(B.index)
            out[r] = {"n": len(com),
                      "composite": spearmanr(A.loc[com, "composite"], B.loc[com, "composite"])[0],
                      "pressure_rating": spearmanr(A.loc[com, "pressure_rating"], B.loc[com, "pressure_rating"])[0],
                      "ordinary_rating": None, "A": A, "B": B, "com": com}
        return out
    odd = set(matches.loc[matches["match_id"] % 2 == 1, "match_id"]); even = set(matches["match_id"]) - odd
    early = set(matches.loc[matches["season_year"] <= 2023, "match_id"]); late = set(matches.loc[matches["season_year"] >= 2024, "match_id"])
    oe = split_corr(odd, even, 25, 6)
    tm = split_corr(early, late, 20, 5)
    # Predictive validity: early pressure rating vs early all-ball raw rating, predicting late pressure rating
    eb = balls[balls["match_id"].isin(early) & balls["is_live"] & (balls["legal_balls_before"] >= 12)]
    pv = {}
    for r in ("bat", "bowl"):
        pid_col = "striker_id" if r == "bat" else "bowler_id"
        sign = 1.0 if r == "bat" else -1.0
        src = eb[~eb["is_wide"]] if r == "bat" else eb
        allball = src.groupby(pid_col)["delta_wp"].mean() * sign
        A, B, com = tm[r]["A"], tm[r]["B"], tm[r]["com"]
        pv[r] = (spearmanr(A.loc[com, "pressure_rating"], B.loc[com, "pressure_rating"])[0],
                 spearmanr(allball.reindex(com).fillna(0), B.loc[com, "pressure_rating"])[0])
    rel = pd.DataFrame([
        {"Check": "Odd vs even match id (composite)", "Batting_r": oe["bat"]["composite"], "Batting_n": oe["bat"]["n"], "Bowling_r": oe["bowl"]["composite"], "Bowling_n": oe["bowl"]["n"]},
        {"Check": "Odd vs even match id (pressure rating only)", "Batting_r": oe["bat"]["pressure_rating"], "Batting_n": oe["bat"]["n"], "Bowling_r": oe["bowl"]["pressure_rating"], "Bowling_n": oe["bowl"]["n"]},
        {"Check": "2022-23 vs 2024-26 (composite)", "Batting_r": tm["bat"]["composite"], "Batting_n": tm["bat"]["n"], "Bowling_r": tm["bowl"]["composite"], "Bowling_n": tm["bowl"]["n"]},
        {"Check": "Predict 2024-26 pressure rating from 2022-23 pressure rating", "Batting_r": pv["bat"][0], "Batting_n": tm["bat"]["n"], "Bowling_r": pv["bowl"][0], "Bowling_n": tm["bowl"]["n"]},
        {"Check": "Predict 2024-26 pressure rating from 2022-23 all-ball rating (baseline)", "Batting_r": pv["bat"][1], "Batting_n": tm["bat"]["n"], "Bowling_r": pv["bowl"][1], "Bowling_n": tm["bowl"]["n"]},
    ])
    rel[["Batting_r", "Bowling_r"]] = rel[["Batting_r", "Bowling_r"]].round(3)
    rel.to_csv(TABLES / "final_split_half_reliability.csv", index=False)
    print(rel.to_string(index=False))

    # --- 3. Definition variants ---
    var_rows = []
    variants = {
        "No early-innings exclusion": dict(min_legal_before=0),
        "Chase innings only": dict(innings=2),
        "HIGH = top 15% LI": dict(high_q=0.85),
        "HIGH = top 25% LI": dict(high_q=0.75),
    }
    for vname, kw in variants.items():
        h2, o2, _, _ = tiers(balls, **kw)
        mb, mm_ = (40, 10) if "Chase" in vname else (60, 15)
        for r in ("bat", "bowl"):
            v = fit_player_models(h2, o2, r, mb, mm_)
            base = res[r]
            com = base.index.intersection(v.index)
            top_b = base.head(10).index
            var_rows.append({"Variant": vname, "Role": "Batting" if r == "bat" else "Bowling",
                             "Top10_overlap": int(len(set(top_b) & set(v.head(10).index))),
                             "Top10_in_variant_top20": int(len(set(top_b) & set(v.head(20).index))),
                             "Spearman_all_common": round(spearmanr(base.loc[com, "composite"], v.loc[com, "composite"])[0], 3),
                             "N_common": len(com)})
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(TABLES / "final_variants_comparison.csv", index=False)
    print(var_df.to_string(index=False))

    # --- 4. Weight sensitivity ---
    w_rows = []
    for r in ("bat", "bowl"):
        _, full = fit_player_models(high, ordy, r, return_all=True)
        el = full[(full["pressure_balls"] >= 60) & (full["matches"] >= 15)]
        Z = StandardScaler().fit_transform(el[["pressure_rating", "consistency", "resilience"]])
        ranks = {p: [] for p in res[r].head(10).index}
        overl = []
        for _ in range(200):
            w = rng.dirichlet([2.0, 1.0, 1.0])  # centred on 50/25/25
            comp = pd.Series(Z @ w, index=el.index).rank(ascending=False)
            for p in ranks:
                ranks[p].append(comp[p])
            overl.append(len(set(comp.nsmallest(10).index) & set(ranks)))
        for i, (p, a) in enumerate(ranks.items(), 1):
            a = np.array(a)
            w_rows.append({"role": r, "rank": i, "name": name.get(p), "median_rank": float(np.median(a)), "best": int(a.min()), "worst": int(a.max()),
                           "pct_draws_in_top10": round((a <= 10).mean() * 100, 1)})
        w_rows.append({"role": r, "rank": None, "name": "MEAN TOP-10 OVERLAP ACROSS DRAWS", "median_rank": float(np.mean(overl)), "best": None, "worst": None, "pct_draws_in_top10": None})
    w_df = pd.DataFrame(w_rows)
    w_df.to_csv(TABLES / "final_weight_sensitivity.csv", index=False)
    print(w_df.to_string(index=False))

    # --- Eligible pool sizes ---
    pd.DataFrame([{"role": "Batting", "eligible_players": len(res["bat"])}, {"role": "Bowling", "eligible_players": len(res["bowl"])}]).to_csv(TABLES / "final_eligible_pool.csv", index=False)
    res["bat"].head(25).assign(name=lambda d: d.index.map(name)).to_csv(TABLES / "final_top25_batters.csv")
    res["bowl"].head(25).assign(name=lambda d: d.index.map(name)).to_csv(TABLES / "final_top25_bowlers.csv")

    # --- 6. Figures ---
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    INK, MUTED, ACC, ACC2 = "#1f2933", "#7b8794", "#c2410c", "#1d4ed8"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, r, ttl, col in zip(axes, ("bat", "bowl"), ("Batters", "Bowlers"), (ACC, ACC2)):
        d = boot_df[(boot_df["role"] == r) & (boot_df["rank"] <= 10)].iloc[::-1]
        y = np.arange(len(d))
        ax.hlines(y, d["p5"], d["p95"], color=col, alpha=0.35, lw=6)
        ax.plot(d["median_boot_rank"], y, "o", color=col, ms=5, label="Bootstrap median rank")
        ax.plot(d["rank"], y, "D", color=INK, ms=4, label="Submitted rank")
        ax.set_yticks(y, [f"{int(a)}. {n}" for a, n in zip(d["rank"], d["name"])])
        ax.set_xlabel("Rank in 200 bootstrap resamples (bar = 5th–95th pct)")
        ax.axvline(10.5, color=MUTED, ls="--", lw=0.8)
        ax.set_title(ttl, loc="left", fontweight="bold", color=INK)
        ax.set_xlim(0, max(60, d["p95"].max() + 2))
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "final_bootstrap_rank_intervals.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, r, ttl, col in zip(axes, ("bat", "bowl"), ("Batters", "Bowlers"), (ACC, ACC2)):
        d = res[r]
        ax.scatter(d["ordinary_rating"] * 1000, d["pressure_rating"] * 1000, s=12, color=MUTED, alpha=0.5)
        t = d.head(10)
        ax.scatter(t["ordinary_rating"] * 1000, t["pressure_rating"] * 1000, s=26, color=col)
        offs = {"Bumrah": (-34, 6), "Khan": (5, -10), "Rana": (5, -10), "Ferguson": (-44, 4), "Brook": (4, 6),
                "Miller": (5, -10), "Yadav": (6, 2), "Pathirana": (5, 2), "Chahal": (5, 3)}
        for p, rr in t.iterrows():
            lab = name.get(p, str(p)).split()[-1]
            ax.annotate(lab, (rr["ordinary_rating"] * 1000, rr["pressure_rating"] * 1000), fontsize=7,
                        xytext=offs.get(lab, (4, 2)), textcoords="offset points", color=INK)
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
        ax.axhline(0, color=MUTED, lw=0.5); ax.axvline(0, color=MUTED, lw=0.5)
        ax.set_xlabel("Ordinary rating (WP added per ball ×1000)"); ax.set_ylabel("Pressure rating (WP added per ball ×1000)")
        ax.set_title(f"{ttl}: above the dashed line = better under pressure", loc="left", fontweight="bold", color=INK, fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "final_pressure_vs_ordinary.png", dpi=200); plt.close(fig)

    # LI by over (live, eligible) to illustrate the definition
    elig = balls[balls["is_live"] & (balls["legal_balls_before"] >= 12)].copy()
    elig["over"] = (elig["legal_balls_before"] // 6 + 1).clip(upper=20)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for inn, col in ((1, ACC2), (2, ACC)):
        s = elig[elig["innings"] == inn].groupby("over")["leverage_index"].mean()
        h = elig[elig["innings"] == inn].assign(hi=lambda d: d["leverage_index"] >= hi_cut).groupby("over")["hi"].mean() * 100
        ax.plot(s.index, s.values, "-o", ms=3, color=col, label=f"Innings {inn}: mean LI")
    ax.axhline(hi_cut, color=MUTED, ls="--", lw=0.8); ax.text(3.1, hi_cut + 0.05, f"HIGH cut-off (LI ≥ {hi_cut:.2f})", fontsize=7, color=MUTED)
    ax.set_xlabel("Over"); ax.set_ylabel("Mean Leverage Index (live balls)"); ax.set_xticks(range(3, 21))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "final_li_by_over.png", dpi=200); plt.close(fig)

    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
