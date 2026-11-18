"""Shared visual building blocks: CSS, hero, KPI cards, callouts, ranking lists, glossary."""
from __future__ import annotations

import html

import streamlit as st

GOLD = "#F5A524"
BAT = "#F5A524"     # batting accent (amber)
BOWL = "#5AB0F0"    # bowling accent (sky blue)
MUTED = "#8A97AD"
INK = "#E7ECF4"
CARD = "#141E31"
LINE = "#24324B"

ROLE_NAME = {"bat": "Batters", "bowl": "Bowlers"}
ROLE_COLOR = {"bat": BAT, "bowl": BOWL}

CSS = f"""
<style>
.block-container {{ padding-top: 4rem !important; padding-bottom: 3rem; max-width: 1180px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
h1 {{ font-weight: 800 !important; }}
.eyebrow {{ color: {GOLD}; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.2rem; }}
.hero-title {{ font-size: clamp(2rem, 5vw, 3.1rem); font-weight: 800; line-height: 1.05; margin: 0 0 0.4rem 0; color: {INK}; }}
.hero-sub {{ font-size: clamp(1.05rem, 2.4vw, 1.35rem); color: {GOLD}; font-weight: 600; margin-bottom: 0.8rem; }}
.lede {{ color: #C3CCDB; font-size: 1.02rem; line-height: 1.6; max-width: 820px; }}
.kpi {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; padding: 14px 16px; height: 100%; }}
.kpi .v {{ font-size: 1.65rem; font-weight: 800; color: {INK}; line-height: 1.1; font-variant-numeric: tabular-nums; }}
.kpi .l {{ color: {MUTED}; font-size: 0.82rem; margin-top: 4px; line-height: 1.3; }}
.kpi.accent {{ border-color: {GOLD}55; }}
.kpi.accent .v {{ color: {GOLD}; }}
.callout {{ background: #1A2338; border-left: 4px solid {GOLD}; border-radius: 8px; padding: 14px 18px; margin: 10px 0 18px; color: #D6DDE8; line-height: 1.55; }}
.callout.blue {{ border-left-color: {BOWL}; }}
.callout.red {{ border-left-color: #F0716B; }}
.rank-card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: 16px 18px 8px; }}
.rank-card h4 {{ margin: 0 0 10px 0; font-size: 1.05rem; }}
.rank-row {{ display: flex; align-items: center; gap: 12px; padding: 7px 0; border-top: 1px solid {LINE}; }}
.rank-row:first-of-type {{ border-top: none; }}
.rank-n {{ width: 28px; height: 28px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.85rem; flex: none; }}
.rank-name {{ font-weight: 600; color: {INK}; flex: 1; min-width: 0; }}
.rank-ctry {{ color: {MUTED}; font-size: 0.82rem; font-weight: 400; margin-left: 6px; }}
.rank-val {{ color: {MUTED}; font-size: 0.8rem; font-variant-numeric: tabular-nums; text-align: right; }}
.flow {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: 8px; margin: 6px 0 12px; }}
.flow .step {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 10px; padding: 10px 12px; }}
.flow .step b {{ color: {GOLD}; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; display: block; margin-bottom: 2px; }}
.flow .step span {{ color: {INK}; font-weight: 600; font-size: 0.95rem; }}
.flow .step small {{ display: block; color: {MUTED}; font-size: 0.78rem; margin-top: 3px; line-height: 1.35; }}
.formula {{ background: #0F1829; border: 1px solid {LINE}; border-radius: 10px; padding: 14px 18px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.95rem; color: {INK}; overflow-x: auto; margin: 6px 0 14px; }}
.pill {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }}
.small-muted {{ color: {MUTED}; font-size: 0.85rem; }}
.src {{ color: #6F7D94; font-size: 0.76rem; margin-top: -6px; margin-bottom: 14px; }}
div[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
@media (max-width: 640px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 3.4rem !important; }}
  .kpi .v {{ font-size: 1.25rem; }}
  [class*="st-key-kpis_"] [data-testid="stHorizontalBlock"] {{ flex-direction: row !important; flex-wrap: wrap !important; gap: 8px !important; }}
  [class*="st-key-kpis_"] [data-testid="stColumn"] {{ flex: 1 1 calc(50% - 8px) !important; min-width: calc(50% - 8px) !important; width: calc(50% - 8px) !important; }}
}}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, intro: str | None = None):
    st.markdown(f'<div class="eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    if intro:
        st.markdown(f'<p class="lede">{intro}</p>', unsafe_allow_html=True)


def kpi(value: str, label: str, accent: bool = False):
    cls = "kpi accent" if accent else "kpi"
    st.markdown(f'<div class="{cls}"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


def kpi_row(items: list[tuple[str, str]], accent_idx: tuple[int, ...] = (), per_row: int | None = None):
    per_row = per_row or len(items)
    n = st.session_state.get("_kpi_n", 0) + 1
    st.session_state["_kpi_n"] = n
    with st.container(key=f"kpis_{n}"):  # keyed so the mobile CSS can lay KPI cards out two-up
        for start in range(0, len(items), per_row):
            cols = st.columns(per_row)
            for i, (v, l) in enumerate(items[start:start + per_row]):
                with cols[i]:
                    kpi(v, l, accent=(start + i) in accent_idx)
    st.write("")


def callout(text: str, tone: str = ""):
    st.markdown(f'<div class="callout {tone}">{text}</div>', unsafe_allow_html=True)


def source(text: str):
    st.markdown(f'<div class="src">Source: {text}</div>', unsafe_allow_html=True)


def flow(steps: list[tuple[str, str, str]]):
    parts = "".join(f'<div class="step"><b>{html.escape(a)}</b><span>{html.escape(b)}</span><small>{html.escape(c)}</small></div>' for a, b, c in steps)
    st.markdown(f'<div class="flow">{parts}</div>', unsafe_allow_html=True)


def formula(text: str):
    st.markdown(f'<div class="formula">{text}</div>', unsafe_allow_html=True)


def ranking_card(df, role: str, value_col: str = "composite", value_fmt: str = "{:.2f}", value_label: str = "score"):
    color = ROLE_COLOR[role]
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f'<div class="rank-row"><span class="rank-n" style="background:{color}22;color:{color}">{int(r["rank"])}</span>'
            f'<span class="rank-name">{html.escape(r["name"])}<span class="rank-ctry">{html.escape(str(r["country"] or ""))}</span></span>'
            f'<span class="rank-val">{value_fmt.format(r[value_col])} {value_label}</span></div>')
    st.markdown(f'<div class="rank-card"><h4 style="color:{color}">Top 10 {ROLE_NAME[role]} under pressure</h4>{"".join(rows)}</div>',
                unsafe_allow_html=True)


GLOSSARY = {
    "Leverage Index (LI)": "How much the next ball can swing the result. 1.0 = an average delivery; 2.0 = twice the average expected swing in win probability.",
    "HIGH pressure": "Live balls (win probability 5–95%, after the first 12 legal balls) whose LI is in the top 20% (LI ≥ 1.47).",
    "ORDINARY": "Live balls whose LI is in the bottom 50% (LI < 0.90). Competitive cricket that is simply low-stakes.",
    "WPA": "Win Probability Added: win probability after the ball minus before it, from the batting side's view. Batters get +WPA, bowlers get −WPA.",
    "Pressure rating": "Adjusted WPA per HIGH-pressure ball, shown ×1000. Batters: win probability added. Bowlers: win probability denied. +5 ≈ five matches' worth of win probability per 1,000 pressure balls versus an average player.",
    "Ordinary rating": "The same adjusted measure on ORDINARY balls — the player's own competitive baseline.",
    "Resilience": "Pressure rating minus ordinary rating. Positive = better when it matters than in ordinary moments. It measures the *gap*, not the level.",
    "Consistency": "Share of the player's matches (with HIGH-pressure involvement) in which their HIGH-pressure WPA was positive.",
    "Composite": "0.50·z(pressure rating) + 0.25·z(consistency) + 0.25·z(resilience), with z-scores taken within the eligible pool. Ranks are by this score.",
    "P(top 10)": "Share of 200 match-level bootstrap resamples in which the player still ranks in the top 10. Higher = more robust.",
}


def glossary(keys: list[str] | None = None, expanded: bool = False):
    with st.expander("What do these numbers mean?", expanded=expanded):
        for k in (keys or GLOSSARY.keys()):
            st.markdown(f"**{k}** — {GLOSSARY[k]}")


def data_guard():
    """Stop the page with a clear message if required data is missing or does not match the submission."""
    from utils import data as D
    if D.csv("ratings_batters.csv") is None or D.csv("ratings_bowlers.csv") is None:
        st.error("Dashboard data files are missing from `data/`. Run `python src/s12_dashboard_export.py` in the analysis repo.")
        st.stop()
    probs = D.verify_submitted()
    if probs:
        st.error("Data does not reproduce the submitted rankings:\n\n" + "\n\n".join(probs))
        st.stop()
