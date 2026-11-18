import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components import ui
from utils import data as D

m = D.manifest()
ui.page_header("Trust", "About the Data",
               "IPL and men's T20 International ball-by-ball data from the SupeRR Selector data pack, cleaned so that every delivery "
               "comes from a full, regulation 20-over contest.")

ui.kpi_row([
    ("Jan 2022 – Nov 2026", f"data period (first match {m.get('date_min', '')}, last {m.get('date_max', '')})"),
    ("IPL + men's T20I", "competitions"),
    (f"{m['matches']:,}", "clean matches (of 883)"),
    (f"{m['deliveries']:,}", "deliveries"),
], accent_idx=(2,))

c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("Matches by competition tier")
    t = D.csv("matches_by_tier.csv")
    label = {"IPL": "IPL", "T20I (FM v FM)": "T20I · Full Member v Full Member", "T20I (FM v Assoc)": "T20I · Full Member v Associate",
             "T20I (Assoc v Assoc)": "T20I · Associate v Associate"}
    t = t.assign(label=t["comp_tier"].map(label)).sort_values("matches", ascending=False)
    st.plotly_chart(C.donut(t["label"], t["matches"], [ui.GOLD, ui.BOWL, "#8FD694", "#C9A7F5"], center=f"{t['matches'].sum()}<br>matches"),
                    config=C.CONFIG)
with c2:
    st.subheader("Matches excluded")
    ex = D.csv("match_exclusions.csv")
    st.plotly_chart(C.hbar(ex["reason"], ex["matches"], "#7B879C", fmt="{:.0f}", height=300, hover="%{y}: %{x} matches<extra></extra>"),
                    config=C.CONFIG)
    st.caption(f"{ex['matches'].sum()} of 883 raw matches excluded. Super-over deliveries are also dropped; only innings 1 and 2 are used. "
               "Rain-affected games are removed because DLS targets distort the required rate.")
    ui.source("docs/PHASE_0_SUMMARY.md (src/s05_clean_and_match_state.py)")

st.subheader("The scoring environment differs by tier")
ct = D.csv("competition_tiers.csv")
st.dataframe(ct[["Competition Tier", "Matches", "Deliveries", "Overall RPO", "Powerplay RPO", "Middle RPO", "Death RPO", "Boundary %",
                 "Wicket %", "Avg 1st Inn Score", "Chase Win %"]], hide_index=True)
st.caption("IPL scores much faster than T20Is, and Associate games slower still. That is why competition tier is a control in both "
           "the win-probability model and the player ratings, and why pressure is measured relative to each game's own context.")

s = D.csv("matches_by_season.csv")
fig = go.Figure(go.Bar(x=s["season_year"].astype(str), y=s["matches"], marker_color=ui.GOLD, text=s["matches"], textposition="outside", cliponaxis=False))
fig.update_layout(title="Clean matches by season")
st.plotly_chart(C.style(fig, 260, legend=False), config=C.CONFIG)

st.divider()
st.subheader("Reconstruction decisions that affect the ratings")
e = D.extras()
coef = e.get("calibrated_coefficients", {})
met = e.get("metrics", {})
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("""
**Legal deliveries.** Wides, no-balls and the dataset's *other* and *Penalty* extras are not legal balls. Under this rule no innings
exceeds 120 legal balls, which validates it.

**Wicket credit.** Bowlers are credited only for bowled, caught, LBW, stumped and hit-wicket. Run-outs and retirements reduce the
batting side's wickets but are not credited to the bowler. Byes and leg-byes count towards the team score but are charged to neither
batter nor bowler.

**Player identity.** Everything is keyed on `player_id`, never names. Seven names map to more than one ID; e.g. **Rashid Khan
(ID 4722, Afghanistan)** is a different player from Rashid Khan (ID 20172, Nepal). Five clerical errors (one missing dismissal ID, four
non-striker typos) were corrected and logged.
""")
with c2:
    st.markdown("**Extras were estimated, not recorded.** The data says *which* extra occurred but not *how many runs* it cost. We estimated "
                f"runs per extra type by regressing 'won by N runs' margins on extras counts ({met.get('matches_in_calibration', 385)} matches).")
    if coef:
        st.dataframe(pd.DataFrame({"Extra type": list(coef), "Estimated runs per incident": [round(v, 2) for v in coef.values()]}), hide_index=True)
    ui.callout(f"<b>Limitation:</b> rebuilt totals are accurate to about ±{met.get('median_absolute_residual_runs', 1.66):.1f} runs "
               "(median error). They agree with the recorded winner in 95.4% of matches, but a handful of razor-thin finishes may have "
               "slightly misplaced win probabilities.", "red")

with st.expander("Data quality checks (all passed)"):
    st.dataframe(D.csv("data_quality_checks.csv"), hide_index=True)
with st.expander("Logged data corrections"):
    st.dataframe(D.csv("data_edits_log.csv"), hide_index=True)
with st.expander("Duplicate player names kept separate"):
    st.dataframe(D.csv("duplicate_names.csv"), hide_index=True)
