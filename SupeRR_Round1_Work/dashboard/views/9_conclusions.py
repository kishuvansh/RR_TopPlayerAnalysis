import streamlit as st

from components import ui
from utils import data as D

ui.page_header("Trust", "What Should We Take Away?")

st.subheader("Limitations")
LIMITS = [
    ("Extras are estimated, not recorded",
     "Run values for wides, no-balls, byes and leg-byes were reconstructed by regression. Totals are accurate to about ±1.7 runs, so a few razor-thin finishes may be slightly mis-scored."),
    ("First-innings leverage is less precise",
     "Without a target, the innings-1 win-probability model is weaker (Brier 0.221 vs 0.129 in chases), so pressure on the side batting first is measured less sharply. This favours players with more chase exposure."),
    ("WPA cannot separate every contributor",
     "Each ball is credited to one batter and one bowler. Fielding, captaincy and the non-striker's role are not separated, and run-outs fall on the batting side only."),
    ("No ball-tracking or pitch data",
     "Line, length, pitch, dew and toss effects are captured only partly, through the venue baseline and the pace/spin control."),
    ("Associate cricket is thin",
     "Only 50 matches involve Associate teams. They are controlled for by tier, but players whose pressure balls come mostly from those games carry wider uncertainty."),
    ("Small samples near the cut-off",
     "Rashid Khan (batting, 98 HIGH balls), Harry Brook (140) and Krunal Pandya (157) are ranked on less than a season's worth of high-pressure balls."),
    ("Batting pressure performance is weakly repeatable",
     "Split-half reliability for batters is close to zero. Bowling is better but still modest. The rankings describe 2022–26, not a forecast."),
]
for i in range(0, len(LIMITS), 2):
    cols = st.columns(2, gap="medium")
    for col, (title, body) in zip(cols, LIMITS[i:i + 2]):
        with col:
            st.markdown(f'<div class="kpi"><div style="font-weight:700;color:{ui.INK};margin-bottom:4px">{title}</div>'
                        f'<div class="l" style="font-size:0.9rem">{body}</div></div>', unsafe_allow_html=True)
    st.write("")

st.subheader("Conclusion")
ui.callout("The project provides a transparent, outcome-based framework for identifying players who deliver match-winning value in "
           "high-pressure T20 situations. Bowling rankings are more stable than batting rankings, and the results should be interpreted "
           "with the uncertainty documented in the analysis.")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**What the method gets right**")
    st.markdown("""
- Pressure is measured from **real match outcomes**, not a hand-weighted formula.
- Dead games and early-over noise are **excluded** from both the pressure and the baseline groups.
- Ratings are **adjusted** for opponent, innings, competition, batting position, bowler type and venue.
- Every rank comes with a **bootstrap probability**, and every modelling choice was stress-tested.
""")
with c2:
    st.markdown("**Reproduce it**")
    st.markdown("""
The full pipeline (Python, seed 42) lives in the analysis repository:

`s05` clean & match state → `s05b` integrity → `s06` win probability → `s07` Leverage Index → `s08/s09` ratings →
`s10` final definition (submitted top 10s) → `s11` final validation → `s12` dashboard export.

This dashboard only **reads** the exported results; it never re-runs the analysis.
""")
st.page_link("views/2_rankings.py", label="Back to the rankings", icon=":material/leaderboard:")
