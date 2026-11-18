import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components import ui
from utils import data as D

ui.page_header("Trust", "How Reliable Are These Rankings?",
               "We stress-tested the rankings in three ways: resampling the matches, changing our own modelling choices, and checking "
               "whether pressure performance repeats. The limitations are shown here, not hidden.")
ui.callout("<b>The rankings are evidence-based estimates of match-winning contribution under pressure, not guarantees of future "
           "performance.</b> The bowling list is materially more stable than the batting list.")

# ---------------- bootstrap ----------------
st.subheader("1 · Match-level bootstrap")
st.markdown("""
Imagine the past five seasons had produced a slightly different set of matches. Would the same players still come out on top?
We test that by **resampling matches with replacement 200 times** (keeping each match whole) and **re-running the entire rating
pipeline** each time. *P(top 10)* is the share of those 200 alternative histories in which a player still makes the top 10.
""")
boot = D.csv("bootstrap_rank_intervals.csv")
c1, c2 = st.columns(2, gap="large")
for col, role in ((c1, "bat"), (c2, "bowl")):
    with col:
        t = D.top10(role)
        st.plotly_chart(C.hbar([f"{int(r)}. {n}" for r, n in zip(t["rank"], t["name"])], t["boot_top10_prob_pct"], ui.ROLE_COLOR[role],
                               title=f"{ui.ROLE_NAME[role]}: P(top 10)", xrange=[0, 100], ref=50, height=470,
                               hover="%{y}: %{x:.1f}% of resamples<extra></extra>"),
                        config=C.CONFIG, key=f"p10_{role}")
bat_max = D.top10("bat")["boot_top10_prob_pct"].max()
bw = D.top10("bowl").set_index("name")["boot_top10_prob_pct"]
st.markdown(f"- **Bowling:** Hazlewood **{bw['Josh Hazlewood']:.0f}%**, Rashid Khan **{bw['Rashid Khan']:.0f}%**, Bumrah **{bw['Jasprit Bumrah']:.0f}%**. "
            "The top five bowlers each stay in the top 10 in at least half the resamples; Pathirana, Motie and Rana are less secure.\n"
            f"- **Batting:** no batter exceeds **{bat_max:.0f}%** (Rinku Singh). Suryakumar Yadav, Tim David, David Miller and Glenn Maxwell "
            "are next most robust. Harry Brook, Rashid Khan and Sam Curran rest on small pressure samples and have wide ranges.")

with st.expander("Full rank intervals (5th–95th percentile)", expanded=False):
    rc = st.segmented_control("Role", ["Batters", "Bowlers"], default="Batters", required=True, key="ri_role")
    role = "bat" if rc == "Batters" else "bowl"
    st.plotly_chart(C.rank_interval(D.top10(role), ui.ROLE_COLOR[role]), config=C.CONFIG)
    st.caption("Medians sit above the submitted ranks because a player at the top of a noisy list tends to slip in a resample "
               "(regression to the mean). We kept the point-estimate order, which uses all the data, rather than re-sorting by noisy medians.")
ui.source("outputs/tables/final_bootstrap_rank_intervals.csv (s11_final_validation.py, 200 resamples, seed 42)")

# ---------------- sensitivity ----------------
st.subheader("2 · Sensitivity to our own choices")
v = D.csv("variants_comparison.csv")
w = D.csv("weight_sensitivity.csv")
wb = w[(w["role"] == "bat") & w["name"].str.startswith("MEAN")]["median_rank"].iloc[0]
ww = w[(w["role"] == "bowl") & w["name"].str.startswith("MEAN")]["median_rank"].iloc[0]
tbl = pd.concat([
    pd.DataFrame({"Variant": v["Variant"], "Role": v["Role"], "Top-10 overlap": v["Top10_overlap"].astype(str) + "/10",
                  "Our top 10 in its top 20": v["Top10_in_variant_top20"].astype(str) + "/10", "Spearman ρ": v["Spearman_all_common"].map("{:.2f}".format)}),
    pd.DataFrame({"Variant": ["Randomised composite weights (200 draws around 50/25/25)"] * 2, "Role": ["Batting", "Bowling"],
                  "Top-10 overlap": [f"{wb:.1f}/10 avg", f"{ww:.1f}/10 avg"], "Our top 10 in its top 20": ["n/a", "n/a"], "Spearman ρ": ["n/a", "n/a"]}),
])
vv = pd.concat([v[["Variant", "Role", "Top10_overlap"]],
                pd.DataFrame({"Variant": ["Randomised weights (avg)"] * 2, "Role": ["Batting", "Bowling"], "Top10_overlap": [wb, ww]})])
fig = go.Figure()
for role_name, col in (("Batting", ui.BAT), ("Bowling", ui.BOWL)):
    d = vv[vv["Role"] == role_name]
    fig.add_trace(go.Bar(name=role_name, y=d["Variant"], x=d["Top10_overlap"], orientation="h", marker_color=col,
                         text=[f"{x:.1f}/10" if x % 1 else f"{x:.0f}/10" for x in d["Top10_overlap"]], textposition="outside", cliponaxis=False))
fig.update_layout(barmode="group", title="How many of our top 10 survive each change?")
fig.update_xaxes(range=[0, 11], title="Players in common with the submitted top 10")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(C.style(fig, 380), config=C.CONFIG)
st.dataframe(tbl, hide_index=True)
st.markdown("""
- **Bowling** is stable everywhere: 8/10 overlap under every variant and rank correlations of 0.83–0.94.
- **Batting** is robust to the score weights (8.6/10 on average) and to a wider HIGH band (7/10). It moves more under a narrower band
  (5/10) or chases only (6/10). It is most sensitive to removing the early-innings exclusion: without it, the top 10 fills with openers
  whose first-two-over balls get inflated leverage. That is the artefact the exclusion is designed to remove.
""")
wt = w[~w["name"].str.startswith("MEAN")]
buttler = wt[(wt["role"] == "bat") & (wt["name"] == "Jos Buttler")]["pct_draws_in_top10"].iloc[0]
st.caption(f"Jos Buttler at #10 is the most weight-dependent selection (top 10 in {buttler:.1f}% of weight draws).")
ui.source("outputs/tables/final_variants_comparison.csv, final_weight_sensitivity.csv")

# ---------------- repeatability ----------------
st.subheader("3 · Is pressure performance a repeatable skill?")
r = D.csv("split_half_reliability.csv")
fig = go.Figure()
for role_name, col, key in (("Batting", ui.BAT, "Batting_r"), ("Bowling", ui.BOWL, "Bowling_r")):
    fig.add_trace(go.Bar(name=role_name, y=r["Check"], x=r[key], orientation="h", marker_color=col, text=r[key].map("{:+.2f}".format),
                         textposition="outside", cliponaxis=False))
fig.add_vline(x=0, line=dict(color=ui.MUTED, width=1))
fig.update_layout(barmode="group")
fig.update_xaxes(title="Spearman rank correlation (1 = identical order, 0 = unrelated)", range=[-0.15, 0.4])
fig.update_yaxes(autorange="reversed")
st.plotly_chart(C.style(fig, 420), config=C.CONFIG)
rr = r.set_index("Check")
pb = rr.loc[rr.index.str.startswith("Predict 2024-26 pressure rating from 2022-23 pressure")].iloc[0]
pa = rr.loc[rr.index.str.contains("all-ball")].iloc[0]
st.markdown(f"""
These are the most sobering numbers in the project. Rating players separately on **odd and even matches** gives almost no agreement for
batters (ρ ≈ {rr.iloc[0]['Batting_r']:.2f}) and weak agreement for bowlers (ρ ≈ {rr.iloc[0]['Bowling_r']:.2f}). Across eras (2022–23 vs 2024–26)
it is about 0.15–0.20. This matches what sports analytics has long found about "clutch" ability: real but small, and buried in
noise over the samples any T20 player accumulates.

One encouraging result: a player's **2022–23 pressure rating predicts their 2024–26 pressure rating better than their all-ball rating
does** (batting {pb['Batting_r']:.2f} vs {pa['Batting_r']:.2f}; bowling {pb['Bowling_r']:.2f} vs {pa['Bowling_r']:.2f}). Pressure-specific
measurement adds information beyond general quality, especially for batters.
""")
ui.callout("<b>Bottom line:</b> batting pressure performance is noisier and less repeatable than bowling pressure performance. "
           "Read the batting list as \"who delivered most in 2022–26\", with wide uncertainty, rather than a forecast.", "red")
ui.source("outputs/tables/final_split_half_reliability.csv (n = 58–62 players per comparison, lower eligibility within each half)")
