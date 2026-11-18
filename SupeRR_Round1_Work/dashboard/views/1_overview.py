import streamlit as st

from components import charts as C
from components import ui
from utils import data as D

m = D.manifest()
tiers = D.csv("tier_profile.csv")
mid = int(tiers.loc[tiers["Tier"].str.startswith("Middle"), "Deliveries"].iloc[0])

st.markdown('<div class="eyebrow">SupeRR Selector Season 2 – Round 1</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Who Delivers When It Matters</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="lede">This project identifies the T20 batters and bowlers who contribute the most match-winning value in '
    "high-pressure situations, using ball-by-ball data and win-probability modelling.</p>",
    unsafe_allow_html=True,
)
st.write("")

ui.kpi_row([
    (f"{m['matches']:,}", "clean matches"),
    (f"{m['deliveries']:,}", "deliveries analysed"),
    (f"{m['ipl_matches']}", "IPL matches"),
    (f"{m['t20i_matches']}", "men's T20I matches"),
    (f"{m['high_deliveries']:,}", "HIGH-pressure deliveries"),
    (f"{m['ordinary_deliveries']:,}", "ORDINARY deliveries"),
], accent_idx=(4,), per_row=3)

st.subheader("The final rankings")
st.caption("Exactly as submitted. Ranked by the composite pressure score (right-hand number); details on the Player Rankings page.")
c1, c2 = st.columns(2, gap="medium")
with c1:
    ui.ranking_card(D.top10("bat"), "bat")
with c2:
    ui.ranking_card(D.top10("bowl"), "bowl")
st.write("")
b1, b2, b3 = st.columns(3)
b1.page_link("views/2_rankings.py", label="Explore player profiles", icon=":material/leaderboard:")
b2.page_link("views/6_comparison.py", label="Compare two players", icon=":material/compare_arrows:")
b3.page_link("views/7_reliability.py", label="How reliable is this?", icon=":material/verified:")

st.divider()
st.subheader("What does performing under pressure mean?")
ex = D.csv("example_states.csv")
left, right = st.columns([1.1, 1], gap="large")
with left:
    st.markdown(
        "Picture a chase: **15 needed off 12 balls**. On a scorecard, the equation is the same whether the batting side has "
        "**six wickets in hand or two**. In reality, the two situations are very different.\n\n"
        "- With **six wickets**, the chasing side is a strong favourite. One dot ball barely changes who wins.\n"
        "- With **two wickets**, the game is on a knife edge. A boundary or a wicket swings it decisively.\n\n"
        "We call that potential swing **leverage**. A player *performs under pressure* when they add the most match-winning value "
        "on the high-leverage balls, compared with an average player in the same situation and with their own ordinary level."
    )
with right:
    if ex is not None:
        r6 = ex[ex["wickets_left"] == 6].iloc[0]
        r2 = ex[ex["wickets_left"] == 2].iloc[0]
        st.markdown("**What our data actually shows** (chases needing 13–17 off 11–13 balls)")
        a, b = st.columns(2)
        with a:
            ui.kpi(f"LI {r6['mean_li']:.2f}", f"6 wickets in hand<br>win prob. {r6['mean_win_prob']:.0%} · {int(r6['deliveries'])} balls")
        with b:
            ui.kpi(f"LI {r2['mean_li']:.2f}", f"2 wickets in hand<br>win prob. {r2['mean_win_prob']:.0%} · {int(r2['deliveries'])} balls", accent=True)
        st.caption(f"LI = Leverage Index; 1.0 is an average ball. Same runs and balls, ~{r2['mean_li'] / r6['mean_li']:.1f}× the leverage "
                   "with two wickets left. Small samples: these are averages of real deliveries in our data.")

st.divider()
st.subheader("How the dashboard works")
ui.flow([
    ("1 · Data", "Match data", "780 clean IPL & T20I matches, ball by ball"),
    ("2 · Model", "Win probability", "Chance the batting side wins, before every ball"),
    ("3 · Leverage", "Leverage Index", "How much the next ball could swing that chance"),
    ("4 · Classify", "Pressure tiers", "HIGH = top 20% of live balls; ORDINARY = bottom 50%"),
    ("5 · Rate", "Player ratings", "Win probability added, adjusted for opponent & context"),
    ("6 · Rank", "Final rankings", "50% pressure · 25% consistency · 25% resilience"),
])
n1, n2, n3 = st.columns(3)
n1.page_link("views/3_pressure.py", label="Steps 2–4: Understanding pressure", icon=":material/speed:")
n2.page_link("views/5_rating_method.py", label="Steps 5–6: How players are rated", icon=":material/functions:")
n3.page_link("views/8_dataset.py", label="Step 1: About the data", icon=":material/database:")

ui.callout("<b>Read the rankings with their uncertainty.</b> The bowling list is the more robust: Hazlewood, Bumrah and Rashid Khan stay in the "
           "top 10 in 72–84% of bootstrap resamples. Pressure batting is a noisier signal: no batter exceeds 61%. "
           "These are evidence-based estimates of match-winning contribution under pressure, not guarantees of future performance.")
