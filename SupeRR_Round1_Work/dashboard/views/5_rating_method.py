import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components import ui
from utils import data as D

ui.page_header("Method", "How Are Players Rated?",
               "Once every ball has a pressure level, we measure what each player actually did with it, in the currency that decides "
               "matches: win probability.")

st.subheader("1 · Win Probability Added (WPA)")
ui.formula("WPA = Win probability AFTER the ball − Win probability BEFORE the ball   (batting side's view)")
c1, c2, c3 = st.columns(3)
with c1:
    ui.kpi("+WPA → batter", "The batter is credited with the change in the batting side's win probability.")
with c2:
    ui.kpi("−WPA → bowler", "The bowler gets the opposite. A positive bowling rating = win probability <b>denied</b>.", accent=True)
with c3:
    ui.kpi("Context built in", "A single when 2 are needed off 1 is worth far more than a single in the 8th over.")
st.write("")

ex = D.csv("wpa_examples.csv")
if ex is not None and len(ex):
    st.markdown("**Two real deliveries from the data**: the biggest HIGH-pressure moments of our #1 batter and #1 bowler.")
    cols = st.columns(len(ex), gap="large")
    for col, (_, r) in zip(cols, ex.iterrows()):
        with col:
            role = r["role"]
            credit = r["delta_wp_batting"] * (1 if role == "bat" else -1)
            color = ui.ROLE_COLOR[role]
            need = f"{r['runs_needed']:.1f} needed, " if not pd.isna(r["runs_needed"]) else ""
            st.markdown(f"<span class='pill' style='background:{color}22;color:{color}'>{'Batter' if role == 'bat' else 'Bowler'}: {r['player']}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"**{r['match']}**, {r['competition']} ({r['match_date']})  \n"
                        f"Innings {int(r['innings'])}, over {int(r['over'])} ball {int(r['ball_in_over'])}: {r['bowler']} to {r['striker']}  \n"
                        f"State before: {need}{int(r['balls_left'])} balls left, {int(r['wickets_left'])} wickets in hand · LI **{r['leverage_index']:.1f}**")
            fig = go.Figure(go.Bar(x=["Before", "After"], y=[r["wp_before"] * 100, r["wp_after"] * 100], marker_color=["#55627A", color],
                                   text=[f"{r['wp_before']:.0%}", f"{r['wp_after']:.0%}"], textposition="outside", cliponaxis=False))
            fig.update_yaxes(title="Batting side win prob. (%)", range=[0, 115])
            fig.update_layout(title=f"Outcome: {r['outcome']} → {r['player'].split()[-1]} credited {credit * 100:+.0f} pp")
            st.plotly_chart(C.style(fig, 280, legend=False), config=C.CONFIG, key=f"wpa_{role}")
    st.caption("pp = percentage points of win probability. Run equations use reconstructed totals: the data does not record "
               "how many runs each extra cost, so targets are estimated to within about ±1.7 runs (see About the Data). "
               "This is why a run requirement can show a decimal.")

st.divider()
st.subheader("2 · Adjusting for opponent and context")
st.markdown("""
Raw averages flatter players who faced weak attacks or batted on flat grounds. Each rating is therefore estimated with a
**ridge regression** on every delivery in the tier, which credits the player and the opponent **at the same time**, plus:
""")
c1, c2 = st.columns(2)
with c1:
    st.markdown("- **Opponent** (the bowler for batters, the batter for bowlers)\n- **Innings** (setting vs chasing)\n- **Competition tier** (IPL, Full-Member T20I, Associate T20I)")
with c2:
    st.markdown("- **Batting position band** (1–3, 4–7, 8–11)\n- **Pace or spin** bowler\n- **Venue scoring baseline**")
ui.formula("± WPA<sub>ball</sub> = β₀ + θ<sub>player</sub> + γ<sub>opponent</sub> + controls + ε      → the player's rating is θ")
ui.callout("<b>Why ridge?</b> It gently pulls every rating towards average, more strongly for players with few balls. A batter who "
           "hit two lucky sixes in 20 balls cannot top the list. The penalty strength was chosen by match-grouped cross-validation "
           "(α = 300 batting, 500 bowling), and the ratings agree closely with an independent empirical-Bayes method (r ≈ 0.92–0.93).", "blue")
st.markdown("Wides are excluded from batters' ratings (the batter cannot score from them) but count against bowlers. Byes and leg-byes "
            "are charged to neither. Run-outs are not credited to the bowler.")

st.divider()
st.subheader("3 · Pressure vs the player's own baseline")
st.markdown("We fit the model **twice**: once on HIGH-pressure balls (→ **pressure rating**) and once on ORDINARY balls "
            "(→ **ordinary rating**). Comparing the two answers the brief's key question: does the player *rise* when it matters, "
            "relative to their usual level?")

st.subheader("4 · The composite score")
c1, c2 = st.columns([1, 1.3], gap="large")
with c1:
    st.plotly_chart(C.donut(["Pressure rating", "Consistency", "Resilience"], [50, 25, 25], [ui.GOLD, ui.BOWL, "#8FD694"], center="Composite"),
                    config=C.CONFIG)
with c2:
    ui.formula("Composite = 0.50 × z(Pressure rating)<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 0.25 × z(Consistency)<br>"
               "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 0.25 × z(Resilience)")
    st.markdown("""
- **Pressure rating (50%)**: adjusted WPA per HIGH-pressure ball. What the player *delivers* when it matters.
- **Consistency (25%)**: % of the player's matches in which their HIGH-pressure WPA was positive. Rewards doing it repeatedly, not once.
- **Resilience (25%)**: pressure rating − ordinary rating. Being *better than your usual self* under pressure.

*z( )* standardises each component within the eligible pool (mean 0, spread 1), so no component dominates because of its units.
""")

st.markdown("##### Worked example from the final table")
role_label = st.segmented_control("Role", ["Batters", "Bowlers"], default="Batters", required=True, key="ex_role")
role = "bat" if role_label == "Batters" else "bowl"
pool = D.ratings(role)
name = st.selectbox("Player", pool[pool["submitted_top10"]]["name"], key="ex_player")
p = pool[pool["name"] == name].iloc[0]
z = lambda c: (p[c] - pool[c].mean()) / pool[c].std(ddof=0)
zp, zc, zr = z("pressure_rating"), z("consistency"), z("resilience")
recomputed = 0.5 * zp + 0.25 * zc + 0.25 * zr
tbl = pd.DataFrame({
    "Component": ["Pressure rating ×1000", "Consistency %", "Resilience ×1000"],
    "Player value": [f"{p['pressure_rating_k']:+.2f}", f"{p['consistency']:.1f}", f"{p['resilience_k']:+.2f}"],
    f"Pool average ({len(pool)} players)": [f"{pool['pressure_rating_k'].mean():+.2f}", f"{pool['consistency'].mean():.1f}", f"{pool['resilience_k'].mean():+.2f}"],
    "z-score": [f"{zp:+.2f}", f"{zc:+.2f}", f"{zr:+.2f}"],
    "Weight": ["0.50", "0.25", "0.25"],
    "Contribution": [f"{0.5 * zp:+.2f}", f"{0.25 * zc:+.2f}", f"{0.25 * zr:+.2f}"],
})
st.dataframe(tbl, hide_index=True)
ok = np.isclose(recomputed, p["composite"], atol=1e-6)
st.markdown(f"**Composite = {recomputed:.3f}** → rank **#{int(p['rank'])}** "
            + ("(matches the saved pipeline score exactly)." if ok else f"(saved score {p['composite']:.3f})."))

st.divider()
st.subheader("5 · Who is eligible?")
c1, c2 = st.columns(2)
with c1:
    ui.kpi("≥ 60", "HIGH-pressure balls faced or bowled")
with c2:
    ui.kpi("≥ 15", "different matches with HIGH-pressure involvement")
st.write("")
st.markdown(f"That leaves **{len(D.ratings('bat'))} batters** and **{len(D.ratings('bowl'))} bowlers**. The match requirement guards against "
            "one-series wonders; ridge shrinkage handles the rest. **Batting and bowling are rated completely separately**, so all-rounders "
            "(Rashid Khan, Krunal Pandya) can, and do, appear in both lists on separate evidence.")
