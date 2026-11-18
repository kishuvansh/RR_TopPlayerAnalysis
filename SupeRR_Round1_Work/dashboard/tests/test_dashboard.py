"""Smoke + correctness tests. Run from the dashboard folder:  python -m pytest tests -q   (or: python tests/test_dashboard.py)

1. Every page renders without exceptions (default state and after interacting with widgets).
2. Dashboard data reproduces the submitted rankings and the validated tables exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANALYSIS = ROOT.parent  # SupeRR_Round1_Work (present locally, absent in the deployed repo)
sys.path.insert(0, str(ROOT))

PAGES = ["views/1_overview.py", "views/2_rankings.py", "views/3_pressure.py", "views/4_win_probability.py",
         "views/5_rating_method.py", "views/6_comparison.py", "views/7_reliability.py", "views/8_dataset.py", "views/9_conclusions.py"]

BATTERS = ["Rinku Singh", "Glenn Maxwell", "Krunal Pandya", "Rashid Khan", "Tim David",
           "Suryakumar Yadav", "Harry Brook", "David Miller", "Sam Curran", "Jos Buttler"]
BOWLERS = ["Josh Hazlewood", "Jasprit Bumrah", "Rashid Khan", "Varun Chakravarthy", "Yuzvendra Chahal",
           "Lockie Ferguson", "Matheesha Pathirana", "Gudakesh Motie", "Krunal Pandya", "Harshit Rana"]


def _app(page: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.switch_page(page).run()
    return at


def _ok(at: AppTest, ctx: str):
    assert not at.exception, f"{ctx}: {[e.value for e in at.exception]}"
    errs = [e.value for e in at.error]
    assert not errs, f"{ctx}: st.error shown: {errs}"


def test_pages_render():
    for p in PAGES:
        _ok(_app(p), p)


def test_interactions():
    at = _app("views/2_rankings.py")
    at.toggle[0].set_value(True).run(); _ok(at, "rankings show-all")
    for lab in at.selectbox[0].options[:: max(1, len(at.selectbox[0].options) // 8)]:
        at.selectbox[0].set_value(lab).run(); _ok(at, f"rankings profile {lab}")
    at.segmented_control[0].set_value("Bowlers").run(); _ok(at, "rankings bowlers")
    for lab in at.selectbox[0].options:
        at.selectbox[0].set_value(lab).run(); _ok(at, f"bowler profile {lab}")

    at = _app("views/6_comparison.py")
    for role in ("Batters", "Bowlers"):
        at.segmented_control[0].set_value(role).run(); _ok(at, f"compare {role}")
        opts = at.selectbox[0].options
        at.selectbox[0].set_value(opts[-1]).run(); _ok(at, f"compare {role} last vs 2nd")
        at.selectbox[1].set_value(opts[-1]).run(); _ok(at, f"compare {role} same player")
        at.selectbox[1].set_value(opts[25]).run(); _ok(at, f"compare {role} outside top 20")

    at = _app("views/5_rating_method.py")
    for role in ("Batters", "Bowlers"):
        at.segmented_control[0].set_value(role).run()
        for n in at.selectbox[0].options:
            at.selectbox[0].set_value(n).run(); _ok(at, f"worked example {role} {n}")
            assert any("matches the saved pipeline score exactly" in m.value for m in at.markdown), f"composite recompute mismatch for {n}"

    at = _app("views/7_reliability.py")
    at.segmented_control[0].set_value("Bowlers").run(); _ok(at, "reliability bowlers")
    at = _app("views/3_pressure.py")
    at.multiselect[0].set_value(["Dead / decided", "Early innings (first 12 balls)"]).run(); _ok(at, "pressure histogram")


def test_rankings_exact():
    bat = pd.read_csv(DATA / "ratings_batters.csv")
    bowl = pd.read_csv(DATA / "ratings_bowlers.csv")
    assert bat[bat.submitted_top10].sort_values("rank")["name"].tolist() == BATTERS
    assert bowl[bowl.submitted_top10].sort_values("rank")["name"].tolist() == BOWLERS
    assert bat["rank"].tolist() == list(range(1, len(bat) + 1)) and bat["composite"].is_monotonic_decreasing
    assert bowl["rank"].tolist() == list(range(1, len(bowl) + 1)) and bowl["composite"].is_monotonic_decreasing


def test_matches_analysis_outputs():
    """Cross-check against the analysis repo (skipped when only the dashboard is deployed)."""
    if not (ANALYSIS / "handoff").exists():
        print("analysis repo not present: skipping cross-checks")
        return
    T = ANALYSIS / "outputs" / "tables"
    for role, f in (("bat", "TOP10_BATTERS.csv"), ("bowl", "TOP10_BOWLERS.csv")):
        h = pd.read_csv(ANALYSIS / "handoff" / f).set_index("player_id")
        d = pd.read_csv(DATA / f"ratings_{'batters' if role == 'bat' else 'bowlers'}.csv").set_index("player_id").loc[h.index]
        for c in ("rank", "pressure_balls", "matches", "pct_chase", "pressure_rating", "ordinary_rating", "resilience", "consistency", "composite"):
            assert np.allclose(d[c].astype(float), h[c].astype(float)), f"{role} {c}"
        b = pd.read_csv(T / "final_bootstrap_rank_intervals.csv").query("role == @role").set_index("player_id").loc[h.index]
        assert np.allclose(d["boot_top10_prob_pct"], b["top10_prob_pct"]), f"{role} bootstrap"
        bx = pd.read_csv(T / f"final_top10_{'batters' if role == 'bat' else 'bowlers'}_boxscore.csv").set_index("player_id").loc[h.index]
        pairs = ([("high_sr", "HIGH_SR"), ("ord_sr", "ORD_SR"), ("high_boundary_pct", "HIGH_boundary_pct"), ("high_balls_per_dismissal", "HIGH_balls_per_dismissal")]
                 if role == "bat" else
                 [("high_economy", "HIGH_economy"), ("ord_economy", "ORD_economy"), ("high_balls_per_wicket", "HIGH_balls_per_wicket"), ("high_dot_pct", "HIGH_dot_pct")])
        for a, b_ in pairs:
            assert np.allclose(d[a].round(2), bx[b_], atol=0.051), f"{role} box {a}"
    for dash, src in (("tier_profile.csv", "final_tier_profile.csv"), ("variants_comparison.csv", "final_variants_comparison.csv"),
                      ("split_half_reliability.csv", "final_split_half_reliability.csv"), ("weight_sensitivity.csv", "final_weight_sensitivity.csv"),
                      ("calibration_bins.csv", "phase1_calibration_bins.csv"), ("calibration_subgroups.csv", "phase1_calibration_subgroups.csv")):
        pd.testing.assert_frame_equal(pd.read_csv(DATA / dash), pd.read_csv(T / src))
    tp = pd.read_csv(DATA / "tier_profile.csv")
    assert tp["Deliveries"].tolist() == [28439, 42659, 71097, 25259]
    cats = pd.read_csv(DATA / "delivery_categories.csv")["deliveries"].sum()
    assert cats == 186246


if __name__ == "__main__":
    for t in (test_rankings_exact, test_matches_analysis_outputs, test_pages_render, test_interactions):
        t(); print("PASS", t.__name__)
