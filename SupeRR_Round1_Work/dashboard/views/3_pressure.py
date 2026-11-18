import pandas as pd
import streamlit as st

from components import charts as C
from components import ui
from utils import data as D

m = D.manifest()
hi_cut, ord_cut = m.get("high_cut", 1.47), m.get("ordinary_cut", 0.90)

ui.page_header("Method", "How Do We Measure Pressure?",
               "Pressure is <b>how much a single delivery can change who wins the match</b>. Instead of hand-weighting required rate, "
               "wickets and balls left, we let real match outcomes decide how much each situation matters.")

ui.flow([
    ("Step 1", "A match situation", "Score, wickets in hand, balls left, target, venue, competition"),
    ("Step 2", "Win probability", "A calibrated model estimates the batting side's chance of winning"),
    ("Step 3", "Seven outcomes", "Dot, 1, 2, 3, 4, 6 or wicket, each with its real-world frequency in that situation"),
    ("Step 4", "Measure the swing", "How far would win probability move after each outcome?"),
    ("Step 5", "Leverage Index", "Expected swing ÷ average swing. 1.0 = an average ball"),
])
ui.formula("Swing(t) = Σ<sub>outcomes o</sub>  P(o | situation) × | WP(after o) − WP(before) |<br>"
           "LI(t) = Swing(t) ÷ mean Swing over all 186,246 deliveries")
st.markdown("**LI = 1** is an average delivery. **LI = 2** carries about twice the average expected swing in win probability. "
            "Higher LI means the ball has more power to decide the result, for batter and bowler alike.")

st.divider()
st.subheader("Same equation, different pressure")
ex = D.csv("example_states.csv")
c1, c2 = st.columns([1, 1.2], gap="large")
with c1:
    r6 = ex[ex["wickets_left"] == 6].iloc[0]
    r2 = ex[ex["wickets_left"] == 2].iloc[0]
    st.markdown(f"""
**Situation A — 15 needed off 12, six wickets in hand.**
The chasing side is a heavy favourite (observed win probability ≈ {r6['mean_win_prob']:.0%}). A dot ball or even a wicket moves the
needle only a little. Observed mean LI: **{r6['mean_li']:.2f}** ({int(r6['deliveries'])} deliveries).

**Situation B — 15 needed off 12, two wickets in hand.**
Now it is close to a coin flip (≈ {r2['mean_win_prob']:.0%}). A boundary or a wicket swings the game hard. Observed mean LI:
**{r2['mean_li']:.2f}** ({int(r2['deliveries'])} deliveries).
""")
    st.caption("Values are averages of real chase deliveries needing 13–17 runs off 11–13 balls, grouped by wickets in hand. "
               "Groups with fewer than 3 deliveries are hidden.")
with c2:
    st.plotly_chart(C.example_states(ex), config=C.CONFIG)
    st.caption("Dotted line = an average ball (LI = 1). Leverage peaks when the game is closest (3–4 wickets in hand here) and falls away as one side takes control.")

st.divider()
st.subheader("Where pressure lives in a T20 innings")
lo = D.csv("li_by_over.csv")
st.plotly_chart(C.li_by_over(lo, hi_cut), config=C.CONFIG)
st.caption("Mean LI of live deliveries by over. In chases, leverage climbs steeply from over 16 and peaks in the 20th. "
           "First-innings leverage is flatter because the eventual target is still unknown.")
ui.source("dashboard/data/li_by_over.csv (from outputs/processed/balls_with_pressure.parquet)")

st.subheader("From LI to pressure tiers")
c1, c2 = st.columns([1, 1.25], gap="large")
with c1:
    st.markdown(f"""
1. **Liveness gate.** Only balls where the batting side's win probability is **between 5% and 95%** can count. Garbage time
   (e.g. 80 needed off 15) is excluded, however "desperate" the scoreboard looks.
2. **Early-innings exclusion.** The **first 12 legal balls** of each innings are excluded. Run rates there rest on one or two
   overs and create artificial leverage spikes.
3. **HIGH pressure** = top 20% of LI among the remaining live balls: **LI ≥ {hi_cut:.2f}**.
4. **ORDINARY** = bottom 50%: **LI < {ord_cut:.2f}**. This is competitive but low-stakes cricket, the player's baseline.
5. Live balls in between (50th–80th percentile) are **left out of both groups** to keep a clear gap.
""")
with c2:
    cats = D.csv("delivery_categories.csv").set_index("category")["deliveries"]
    order = ["HIGH", "Middle (unused)", "ORDINARY", "Dead / decided", "Early innings (first 12 balls)"]
    colors = [ui.BAT, "#8C7A55", ui.BOWL, "#4A5468", "#6E5A8A"]
    st.plotly_chart(C.donut(order, [int(cats[o]) for o in order], colors, center=f"{int(cats.sum()):,}<br>deliveries"), config=C.CONFIG)

tiers = D.csv("tier_profile.csv")
show = pd.DataFrame({
    "Tier": tiers["Tier"], "Deliveries": tiers["Deliveries"], "Mean LI": tiers["Mean_LI"],
    "% in chases": tiers["Share_Innings2_pct"], "% in death overs": tiers["Share_Death_Overs_pct"],
    "Runs/over": tiers["Runs_per_Over"], "Wicket %": tiers["Wicket_pct"], "Mean |ΔWP| per ball": tiers["Mean_abs_dWP"],
})
st.dataframe(show, hide_index=True, column_config={
    "Mean LI": st.column_config.NumberColumn(format="%.2f"),
    "Mean |ΔWP| per ball": st.column_config.NumberColumn(format="%.4f", help="Average absolute change in win probability actually caused by the ball"),
})
hi_row, ord_row = tiers.iloc[0], tiers.iloc[2]
st.caption(f"HIGH balls actually moved win probability {hi_row['Mean_abs_dWP'] / ord_row['Mean_abs_dWP']:.1f}× more than ORDINARY balls "
           f"({hi_row['Mean_abs_dWP']:.4f} vs {ord_row['Mean_abs_dWP']:.4f}), confirming the tiers separate real stakes. "
           "Note: the Early-innings category (18,792 balls) is not shown in this table.")
ui.source("outputs/tables/final_tier_profile.csv (s11_final_validation.py)")

with st.expander("LI distribution by tier"):
    h = D.csv("li_histogram.csv")
    pick = st.multiselect("Show", order, default=["HIGH", "Middle (unused)", "ORDINARY"])
    st.plotly_chart(C.li_histogram(h, pick or order), config=C.CONFIG)
    st.caption("Dead balls sit near LI = 0: nothing can change the result. LI values above 6 are grouped in the last bar.")

st.divider()
st.subheader("Why not a conventional pressure formula?")
cc = D.csv("ccpi_comparison.csv")
bat_overlap = cc.loc[cc["Comparison"].str.startswith("Batting"), "HIGH_Overlap_Pct"].iloc[0]
close = D.csv("close_vs_lopsided.csv").set_index("Match_Type")
st.markdown(f"""
We also built a conventional, hand-weighted **Combined Contextual Pressure Index (CCPI)**: 0–100 scores mixing required-rate
difficulty, wickets lost and innings stage, with separate batting and bowling versions. It is **not** the final model. We tested it
against LI, and the two agreed on only **{bat_overlap}** of high-pressure batting deliveries. The disagreements explain the choice:

- The formula gives *maximum* pressure to hopeless chases (e.g. a required rate of 588 off the last ball). LI correctly gives them zero.
- In a genuinely balanced finish (e.g. 15 off 12 with three or four wickets in hand, where our data shows mean LI of 2.2–3.2),
  every boundary or wicket swings the result hard. The formula rates such states only moderately, because no single ingredient looks alarming.
- Fitting the formula's weights to reproduce LI gave a **negative R²** for every innings and role: no linear mix of the usual
  ingredients reproduces real match leverage.
""")
cl, lp = close.loc["Close Matches (Innings 2 Chases)"], close.loc["Lopsided Matches (Innings 2 Chases)"]
ui.callout(f"Sanity check: chases that ended close have <b>{cl['Mean_LI'] / lp['Mean_LI']:.1f}×</b> the mean LI of lopsided ones "
           f"({cl['Mean_LI']:.2f} vs {lp['Mean_LI']:.2f}), as a good pressure measure should.", "blue")
with st.expander("CCPI comparison details"):
    st.dataframe(cc, hide_index=True)
    st.dataframe(D.csv("ccpi_weight_r2.csv"), hide_index=True)
    ui.source("outputs/tables/phase2_li_vs_ccpi_metrics.csv, phase2_weight_calibration_r2.csv, phase2_close_vs_lopsided.csv (Phase 2, all deliveries)")
