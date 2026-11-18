import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components import ui
from utils import data as D

ui.page_header("Results", "Player Comparison Lab",
               "Put any two eligible players of the same role side by side. Defaults are each list's top two.")

role_label = st.segmented_control("Role", ["Batters", "Bowlers"], default="Batters", required=True, key="cmp_role")
role = "bat" if role_label == "Batters" else "bowl"
df = D.ratings(role)
color = ui.ROLE_COLOR[role]
OTHER = "#C9A7F5"

c1, c2 = st.columns(2)
a_lab = c1.selectbox("Player A", df["label"], index=0, key=f"a_{role}")
b_lab = c2.selectbox("Player B", df["label"], index=1, key=f"b_{role}")
A, B = df[df["label"] == a_lab].iloc[0], df[df["label"] == b_lab].iloc[0]
if a_lab == b_lab:
    st.info("Pick two different players to compare.")
    st.stop()

ui.callout("<b>Pressure rating and resilience measure different things.</b> Pressure rating is the <i>level</i>: how much win probability "
           "the player adds (or denies) on HIGH balls. Resilience is the <i>gap</i>: how much better they are under pressure than on their "
           "own ORDINARY balls. A great player can have a high pressure rating but modest resilience, because they are excellent all the time.")

# ---------- headline table ----------
def row(p):
    return {
        "Rank": int(p["rank"]), "Composite": round(p["composite"], 2),
        "Pressure ×1000": round(p["pressure_rating_k"], 2), "Ordinary ×1000": round(p["ordinary_rating_k"], 2),
        "Resilience ×1000": round(p["resilience_k"], 2), "Consistency %": round(p["consistency"], 1),
        "HIGH balls": int(p["pressure_balls"]), "Matches": int(p["matches"]), "Chase %": p["pct_chase"],
        "P(top 10)": "—" if pd.isna(p["boot_top10_prob_pct"]) else f"{p['boot_top10_prob_pct']:.0f}%",
    }
cmp = pd.DataFrame([row(A), row(B)], index=[A["label"], B["label"]]).T
cmp.index.name = "Metric"
st.dataframe(cmp.astype(str))

# ---------- components (z-scores) ----------
st.subheader("Where the composite comes from")
z = lambda p, c: (p[c] - df[c].mean()) / df[c].std(ddof=0)
comps = [("Pressure rating", "pressure_rating", 0.5), ("Consistency", "consistency", 0.25), ("Resilience", "resilience", 0.25)]
fig = go.Figure()
for p, col in ((A, color), (B, OTHER)):
    fig.add_trace(go.Bar(name=p["name"], x=[c[0] for c in comps], y=[w * z(p, c) for _, c, w in comps], marker_color=col,
                         text=[f"{w * z(p, c):+.2f}" for _, c, w in comps], textposition="outside", cliponaxis=False,
                         hovertemplate="%{x}: weighted z %{y:+.2f}<extra>" + p["name"] + "</extra>"))
fig.update_layout(barmode="group")
fig.update_yaxes(title="Weighted contribution to composite (z × weight)", zeroline=True)
st.plotly_chart(C.style(fig, 340), config=C.CONFIG)
st.caption(f"Contributions add up to the composite: {A['name']} {A['composite']:.2f}, {B['name']} {B['composite']:.2f}. "
           "z-scores are relative to all eligible players of the role.")

c1, c2 = st.columns([1.2, 1], gap="large")
with c1:
    st.subheader("Pressure vs ordinary")
    st.plotly_chart(C.pressure_vs_ordinary(df, role, highlight=[A["name"], B["name"]], height=460, label_top=False), config=C.CONFIG)
    st.caption("White rings mark the two selected players; coloured dots are the submitted top 10 (hover for names).")
with c2:
    st.subheader("Sample size")
    fig = go.Figure()
    for p, col in ((A, color), (B, OTHER)):
        fig.add_trace(go.Bar(name=p["name"], x=["HIGH balls", "Matches"], y=[p["pressure_balls"], p["matches"]], marker_color=col,
                             text=[int(p["pressure_balls"]), int(p["matches"])], textposition="outside", cliponaxis=False))
    fig.add_hline(y=60, line=dict(color=ui.MUTED, dash="dot"), annotation_text="min 60 balls", annotation_position="top right",
                  annotation_font=dict(color=ui.MUTED, size=11))
    fig.update_layout(barmode="group")
    st.plotly_chart(C.style(fig, 300), config=C.CONFIG)
    st.subheader("Ranking stability")
    both = pd.DataFrame([A, B])
    if both["boot_top10_prob_pct"].notna().all():
        st.plotly_chart(C.rank_interval(both.sort_values("rank"), color, height=260), config=C.CONFIG)
    else:
        st.info("Bootstrap ranks were computed for the top 20 of each list. At least one selected player is outside that group.")

st.subheader("HIGH vs ORDINARY box score")
st.markdown(f"<span style='color:{color}'>■</span> {A['name']} &nbsp;&nbsp; <span style='color:{OTHER}'>■</span> {B['name']}", unsafe_allow_html=True)
if role == "bat":
    metrics = [("Strike rate", "sr", "higher better"), ("Balls per dismissal", "balls_per_dismissal", "higher better"), ("Boundary %", "boundary_pct", "higher better")]
else:
    metrics = [("Economy (RPO)", "economy", "lower better"), ("Balls per wicket", "balls_per_wicket", "lower better"), ("Dot-ball %", "dot_pct", "higher better")]
cols = st.columns(3)
for col, (lab, key, note) in zip(cols, metrics):
    with col:
        fig = go.Figure()
        for p, c in ((A, color), (B, OTHER)):
            vals = [p[f"high_{key}"], p[f"ord_{key}"]]
            fig.add_trace(go.Bar(name=p["name"].split()[-1], x=["HIGH", "ORDINARY"], y=vals, marker_color=c,
                                 text=["–" if pd.isna(v) else f"{v:.1f}" for v in vals], textposition="outside", cliponaxis=False))
        fig.update_layout(barmode="group", title=f"{lab} <span style='font-size:11px;color:{ui.MUTED}'>({note})</span>")
        st.plotly_chart(C.style(fig, 290, legend=False), config=C.CONFIG, key=f"box_{key}")
st.caption("Raw, unadjusted box-score numbers on the same HIGH and ORDINARY deliveries used for the ratings. "
           "They help explain the ratings but do not replace them (no opponent or match-state adjustment).")
