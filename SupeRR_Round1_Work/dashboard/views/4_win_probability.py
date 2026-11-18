import pandas as pd
import streamlit as st

from components import charts as C
from components import ui
from utils import data as D

m = D.manifest()
sub = D.csv("calibration_subgroups.csv").set_index("Subgroup")
bins = D.csv("calibration_bins.csv")

ui.page_header("Method", "How Do We Estimate the Chance of Winning?",
               "Everything in this project rests on one question asked before every ball: <i>how likely is the batting side to win from here?</i> "
               "We answer it with two simple, well-calibrated logistic regression models, one for each innings.")

ui.kpi_row([
    (f"{m.get('brier_overall', 0.1765):.3f}", "Overall Brier score (lower is better)"),
    (f"{m.get('brier_naive', 0.2499):.3f}", "Coin-flip baseline"),
    (f"{sub.loc['Innings 1', 'Brier Score']:.3f}", "Innings 1 Brier"),
    (f"{sub.loc['Innings 2', 'Brier Score']:.3f}", "Chase (innings 2) Brier"),
], accent_idx=(0,))
st.caption(f"The Brier score is the average squared gap between predicted probability and what happened. "
           f"Ours is {(1 - m.get('brier_overall', 0.1765) / m.get('brier_naive', 0.2499)):.0%} lower than always predicting 50/50.")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("#### Innings 1: setting a total")
    st.markdown("""
The side batting first has no target yet, so the model looks at:
- runs scored so far
- wickets in hand
- balls left
- current run rate
- **venue scoring baseline** (average first-innings total at that ground)
- competition tier (IPL / Full-Member T20I / Associate T20I)
""")
    st.caption("Harder to predict, because the chase has not happened yet (Brier 0.221).")
with c2:
    st.markdown("#### Innings 2: the chase")
    st.markdown("""
With a target known, the model uses:
- runs needed
- balls left
- wickets in hand
- required run rate
- current run rate
- the **gap** between required and current rate
- competition tier
""")
    st.caption("Much sharper, because the equation is explicit (Brier 0.129).")

st.divider()
st.subheader("Is the model honest about its probabilities?")
c1, c2 = st.columns([1.3, 1], gap="large")
with c1:
    st.plotly_chart(C.calibration(bins), config=C.CONFIG)
with c2:
    st.markdown("**Calibration** means: when the model says *70%*, the batting side should win about 70% of the time across "
                "all such situations.")
    st.markdown(f"We sorted all {int(bins['Deliveries'].sum()):,} deliveries into ten bands of predicted probability and compared "
                f"them with what actually happened. The largest gap in any band is **{bins['Absolute Error'].max():.3f}** "
                "(under 4 percentage points), so the points sit on the diagonal.")
    ui.callout("Why this matters: the Leverage Index and every player rating are built from these probabilities. A biased model "
               "would bias every rating.")
ui.source("outputs/tables/phase1_calibration_bins.csv · docs/PHASE_1_SUMMARY.md (src/s06_win_probability_model.py)")

st.subheader("No peeking: out-of-fold, match-grouped validation")
st.markdown("""
Every win probability used in the analysis is **out-of-fold**: it comes from a model that never saw that match.

1. The 780 matches are split into five groups (**folds**).
2. The model is trained on four groups and predicts the fifth, five times over, so every ball gets a prediction.
3. Crucially, the split is **by match, not by ball**. Balls from the same game share a pitch, weather, teams and momentum. If some
   balls of a match were used for training and others for testing, the model would partly be "remembering" the game, and its
   accuracy would look better than it really is.
""")
st.markdown("As an extra check, a model trained without the **2026 season** predicted those unseen chases with a Brier score of "
            "**0.126**. Logistic regression also beat gradient-boosted trees out-of-fold: win probability is a smooth, "
            "monotonic function of the scoreboard, and trees overfit step-like boundaries.")

with st.expander("Calibration by subgroup"):
    st.dataframe(D.csv("calibration_subgroups.csv"), hide_index=True)
    st.caption("Mean predicted vs observed win rate agree within about one percentage point in every innings, phase and competition tier.")
