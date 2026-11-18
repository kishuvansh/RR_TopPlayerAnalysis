"""SupeRR Selector Season 2 – Round 1 dashboard. Entry point: `streamlit run app.py`."""
import streamlit as st

from components import ui

st.set_page_config(
    page_title="SupeRR Selector S2 · Who Delivers When It Matters",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="auto",
)
ui.inject_css()
st.session_state["_kpi_n"] = 0  # reset per run so KPI container keys are stable

PAGES = {
    "Results": [
        st.Page("views/1_overview.py", title="Overview", icon=":material/home:", default=True),
        st.Page("views/2_rankings.py", title="Player Rankings", icon=":material/leaderboard:"),
        st.Page("views/6_comparison.py", title="Comparison Lab", icon=":material/compare_arrows:"),
    ],
    "Method": [
        st.Page("views/3_pressure.py", title="Understanding Pressure", icon=":material/speed:"),
        st.Page("views/4_win_probability.py", title="Win Probability Model", icon=":material/percent:"),
        st.Page("views/5_rating_method.py", title="How Players Are Rated", icon=":material/functions:"),
    ],
    "Trust": [
        st.Page("views/7_reliability.py", title="Reliability", icon=":material/verified:"),
        st.Page("views/8_dataset.py", title="About the Data", icon=":material/database:"),
        st.Page("views/9_conclusions.py", title="Limitations & Takeaways", icon=":material/flag:"),
    ],
}

nav = st.navigation(PAGES, position="sidebar")

with st.sidebar:
    st.markdown("---")
    st.markdown(
        '<div class="small-muted">SupeRR Selector Season 2 · Round 1<br>IPL + men\'s T20I, Jan 2022 – Nov 2026<br>'
        "780 matches · 186,246 deliveries</div>",
        unsafe_allow_html=True,
    )

ui.data_guard()
nav.run()
