import pandas as pd
import streamlit as st

from components import charts as C
from components import ui
from utils import data as D

ui.page_header("Results", "Final Player Rankings",
               "The two submitted top-10 lists with every component of the score. Pick a player below the table to see why they rank where they do.")

pop = D.csv("population_reference.csv").iloc[0]
role_label = st.segmented_control("Role", ["Batters", "Bowlers"], default="Batters", required=True, label_visibility="collapsed")
role = "bat" if role_label != "Bowlers" else "bowl"
color = ui.ROLE_COLOR[role]
df_all = D.ratings(role)

if role == "bat":
    ui.callout("<b>Batter ratings = win probability <u>added</u></b> per 1,000 balls. +5.79 means about 5.8 matches' worth of win "
               "probability added per 1,000 HIGH-pressure balls, compared with an average batter facing the same bowler in the same situation.")
else:
    ui.callout("<b>Bowler ratings = win probability <u>denied</u></b> to the batting side per 1,000 balls. Positive is good for the bowler: "
               "+3.82 means about 3.8 matches' worth of win probability taken away per 1,000 HIGH-pressure balls versus an average bowler.", "blue")

show_all = st.toggle(f"Show all {len(df_all)} eligible {role_label.lower()} (≥ 60 HIGH balls in ≥ 15 matches)", value=False)
view = df_all if show_all else df_all[df_all["submitted_top10"]]

table = pd.DataFrame({
    "Rank": view["rank"],
    "Player": view["name"],
    "Country": view["country"],
    "HIGH balls": view["pressure_balls"],
    "Matches": view["matches"],
    "Chase %": view["pct_chase"],
    "Pressure ×1000": view["pressure_rating_k"],
    "Ordinary ×1000": view["ordinary_rating_k"],
    "Resilience ×1000": view["resilience_k"],
    "Consistency %": view["consistency"],
    "Composite": view["composite"],
    "P(top 10) %": view["boot_top10_prob_pct"],
})
st.dataframe(
    table, hide_index=True, height=min(38 * len(table) + 40, 620),
    column_config={
        "Rank": st.column_config.NumberColumn(width="small", help="Final rank by composite score"),
        "Player": st.column_config.TextColumn(width="medium"),
        "HIGH balls": st.column_config.NumberColumn(help="High-pressure deliveries faced (batters, excl. wides) or bowled"),
        "Matches": st.column_config.NumberColumn(help="Distinct matches with HIGH-pressure involvement"),
        "Chase %": st.column_config.NumberColumn(format="%.0f", help="Share of the player's HIGH balls that came in the second innings"),
        "Pressure ×1000": st.column_config.NumberColumn(format="%.2f", help=ui.GLOSSARY["Pressure rating"]),
        "Ordinary ×1000": st.column_config.NumberColumn(format="%.2f", help=ui.GLOSSARY["Ordinary rating"]),
        "Resilience ×1000": st.column_config.NumberColumn(format="%+.2f", help=ui.GLOSSARY["Resilience"]),
        "Consistency %": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100, help=ui.GLOSSARY["Consistency"]),
        "Composite": st.column_config.NumberColumn(format="%.2f", help=ui.GLOSSARY["Composite"]),
        "P(top 10) %": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100,
                                                       help=ui.GLOSSARY["P(top 10)"] + " Computed for the top 20 of each list only; blank otherwise."),
    },
)
st.caption("Click a column header to sort. Composite is a z-score blend, so it is only comparable within one role. "
           "P(top 10) was computed for the top 20 players in each list.")
ui.source("handoff/TOP10_*.csv (submitted) · outputs/tables/final_bootstrap_rank_intervals.csv · src/s12_dashboard_export.py")
ui.glossary(["Pressure rating", "Ordinary rating", "Resilience", "Consistency", "Composite", "P(top 10)"])

# ---------------- profile ----------------
st.divider()
st.subheader("Player profile")
opts = view["label"].tolist()
sel = st.selectbox("Choose a player", opts, index=0, key=f"profile_{role}")
p = view[view["label"] == sel].iloc[0]

st.markdown(f"### {p['name']} <span class='small-muted'>· {p['country'] or '—'} · {p['ipl_teams'] or 'T20I only'}</span>", unsafe_allow_html=True)
rank_txt = f"#{int(p['rank'])} of {len(df_all)} eligible" + (" · submitted top 10" if p["submitted_top10"] else "")
st.markdown(f"<span class='pill' style='background:{color}22;color:{color}'>{rank_txt}</span>", unsafe_allow_html=True)
st.write("")
ui.kpi_row([
    (f"{p['pressure_rating_k']:+.2f}", "Pressure rating ×1000"),
    (f"{p['ordinary_rating_k']:+.2f}", "Ordinary rating ×1000"),
    (f"{p['resilience_k']:+.2f}", "Resilience ×1000"),
    (f"{p['consistency']:.0f}%", "Consistency (positive matches)"),
    (f"{p['composite']:.2f}", "Composite score"),
    (f"{int(p['pressure_balls'])}", "HIGH-pressure balls"),
    (f"{int(p['matches'])}", "Matches involved"),
    ("—" if pd.isna(p["boot_top10_prob_pct"]) else f"{p['boot_top10_prob_pct']:.0f}%", "Bootstrap P(top 10)"),
], accent_idx=(0, 4), per_row=4)

st.markdown("##### Under pressure vs their own ordinary level (raw box score)")
if role == "bat":
    rows = [("Strike rate", p["high_sr"], p["ord_sr"], pop["bat_HIGH_SR"], pop["bat_ORD_SR"]),
            ("Balls per dismissal", p["high_balls_per_dismissal"], p["ord_balls_per_dismissal"], None, None),
            ("Boundary %", p["high_boundary_pct"], p["ord_boundary_pct"], None, None)]
    notes = ["higher is better", "higher is better", "higher is better"]
    league = f"All batters: strike rate {pop['bat_HIGH_SR']:.0f} on HIGH balls vs {pop['bat_ORD_SR']:.0f} on ORDINARY."
    counts = f"{int(p['high_balls'])} HIGH balls ({int(p['high_runs'])} runs, {int(p['high_dismissals'])} outs) · {int(p['ord_balls'])} ORDINARY balls ({int(p['ord_runs'])} runs, {int(p['ord_dismissals'])} outs)"
else:
    rows = [("Economy (RPO)", p["high_economy"], p["ord_economy"], pop["bowl_HIGH_econ"], pop["bowl_ORD_econ"]),
            ("Balls per wicket", p["high_balls_per_wicket"], p["ord_balls_per_wicket"], None, None),
            ("Dot-ball %", p["high_dot_pct"], p["ord_dot_pct"], None, None)]
    notes = ["lower is better", "lower is better", "higher is better"]
    league = f"All bowlers: economy {pop['bowl_HIGH_econ']:.2f} on HIGH balls vs {pop['bowl_ORD_econ']:.2f} on ORDINARY."
    counts = f"{int(p['high_balls'])} legal HIGH balls ({int(p['high_wickets'])} wickets) · {int(p['ord_balls'])} legal ORDINARY balls ({int(p['ord_wickets'])} wickets)"
st.plotly_chart(C.high_vs_ord_bars(rows, color, notes), config=C.CONFIG)
st.caption(f"{counts}. {league} Raw numbers are not adjusted for opponent or match state; the rating above is.")

c1, c2 = st.columns([1.4, 1], gap="large")
with c1:
    st.markdown("##### Match by match: net win probability on HIGH-pressure balls")
    pm = D.match_pressure(role)
    if pm is not None:
        pmp = pm[pm["player_id"] == p["player_id"]]
        st.plotly_chart(C.match_wpa_bars(pmp, color), config=C.CONFIG)
        st.caption(f"Each bar is one match ({len(pmp)} in total). Coloured = positive contribution, red = negative. "
                   f"Consistency = share of positive bars ({p['consistency']:.0f}%). pp = percentage points, unadjusted.")
with c2:
    st.markdown("##### Why this rank?")
    pct = lambda col: (df_all[col] < p[col]).mean() * 100
    st.markdown(
        f"- Pressure rating beats **{pct('pressure_rating'):.0f}%** of eligible {role_label.lower()}.\n"
        f"- Resilience beats **{pct('resilience'):.0f}%** — {'better' if p['resilience'] > 0 else 'worse'} under pressure than in their own ordinary games.\n"
        f"- Consistency beats **{pct('consistency'):.0f}%**.\n"
        f"- {p['pct_chase']:.0f}% of their HIGH-pressure balls came in chases."
    )
    if not pd.isna(p["boot_top10_prob_pct"]):
        st.markdown(f"- In match-level resampling, median rank **{p['boot_median_boot_rank']:.1f}**, 90% range "
                    f"**{p['boot_p5']:.0f}–{p['boot_p95']:.0f}**.")
    if p["pressure_balls"] < 160:
        ui.callout(f"Small sample: {int(p['pressure_balls'])} HIGH-pressure balls. Treat this rank as less certain.", "red")
