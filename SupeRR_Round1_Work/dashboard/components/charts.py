"""Plotly chart builders with one consistent dark theme."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.ui import BAT, BOWL, GOLD, INK, LINE, MUTED, ROLE_COLOR

GREY = "#55627A"
RED = "#F0716B"
CONFIG = {"displayModeBar": False, "responsive": True}


def style(fig: go.Figure, height: int = 380, legend: bool = True, margin=None) -> go.Figure:
    title_text = fig.layout.title.text
    fig.layout.title = None  # an empty title object renders as "undefined" under Streamlit's Plotly theme
    if title_text:
        fig.update_layout(title=dict(text=title_text, font=dict(size=15, color=INK), x=0, xanchor="left"))
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=13, family="Inter, system-ui, -apple-system, Segoe UI, sans-serif"),
        margin=margin or dict(l=10, r=10, t=36, b=10),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=12)),
        hoverlabel=dict(bgcolor="#1C2740", bordercolor=LINE, font=dict(color=INK, size=13)),
    )
    if title_text and legend:  # keep the legend clear of the title
        fig.update_layout(legend=dict(yanchor="top", y=-0.16), margin=dict(b=(margin or {}).get("b", 10) + 50))
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=MUTED), title_font=dict(color=MUTED, size=12))
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=MUTED), title_font=dict(color=MUTED, size=12))
    return fig


# ---------- rankings / players ----------
def pressure_vs_ordinary(df: pd.DataFrame, role: str, highlight: list[str] | None = None, height: int = 460, label_top: bool = True) -> go.Figure:
    """Scatter of every eligible player; submitted top 10 coloured; optional highlighted players ringed."""
    col = ROLE_COLOR[role]
    unit = "added" if role == "bat" else "denied"
    others = df[~df["submitted_top10"]]
    top = df[df["submitted_top10"]]
    fig = go.Figure()
    lo = min(df["ordinary_rating_k"].min(), df["pressure_rating_k"].min()) - 0.5
    hi = max(df["ordinary_rating_k"].max(), df["pressure_rating_k"].max()) + 0.5
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=GREY, dash="dash", width=1), hoverinfo="skip", name="Same as ordinary (above = better under pressure)"))
    tmpl = ("<b>%{customdata[0]}</b> (rank %{customdata[1]})<br>Pressure: %{y:.2f}<br>Ordinary: %{x:.2f}"
            "<br>Resilience: %{customdata[2]:+.2f}<br>HIGH balls: %{customdata[3]}<extra></extra>")
    cd = lambda d: np.stack([d["name"], d["rank"], d["resilience_k"], d["pressure_balls"]], axis=-1)
    fig.add_trace(go.Scatter(x=others["ordinary_rating_k"], y=others["pressure_rating_k"], mode="markers", name="Other eligible players",
                             marker=dict(size=8, color=GREY, opacity=0.65), customdata=cd(others), hovertemplate=tmpl))
    fig.add_trace(go.Scatter(x=top["ordinary_rating_k"], y=top["pressure_rating_k"], mode="markers+text" if label_top else "markers", name="Submitted top 10",
                             marker=dict(size=12, color=col, line=dict(color="#0B1220", width=1)),
                             text=top["name"].str.split().str[-1], textposition="top center", textfont=dict(size=10, color=INK),
                             customdata=cd(top), hovertemplate=tmpl))
    if highlight:
        h = df[df["name"].isin(highlight)]
        fig.add_trace(go.Scatter(x=h["ordinary_rating_k"], y=h["pressure_rating_k"], mode="markers+text", name="Selected",
                                 text=h["name"], textposition="bottom center", textfont=dict(size=11, color="#FFFFFF"),
                                 marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color="#FFFFFF", width=2)),
                                 customdata=cd(h), hovertemplate=tmpl))
    fig.update_xaxes(title=f"Ordinary rating (win prob. {unit} per 1,000 ordinary balls)", range=[lo, hi])
    fig.update_yaxes(title=f"Pressure rating (per 1,000 HIGH balls)", range=[lo, hi])
    return style(fig, height)


def hbar(labels, values, color, title="", fmt="{:.0f}%", height=None, xrange=None, ref=None, ref_label=None, hover=None):
    labels, values = list(labels), list(values)
    fig = go.Figure(go.Bar(y=labels[::-1], x=values[::-1], orientation="h", marker_color=color,
                           text=[fmt.format(v) for v in values[::-1]], textposition="outside", cliponaxis=False,
                           hovertemplate=hover or "%{y}: %{x}<extra></extra>"))
    if ref is not None:
        fig.add_vline(x=ref, line=dict(color=MUTED, dash="dot", width=1))
        if ref_label:
            fig.add_annotation(x=ref, y=1.02, yref="paper", text=ref_label, showarrow=False, font=dict(color=MUTED, size=11))
    fig.update_xaxes(range=xrange, showgrid=True)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title=title)
    return style(fig, height or 40 * len(labels) + 70, legend=False, margin=dict(l=10, r=40, t=40 if title else 20, b=10))


def high_vs_ord_bars(rows: list[tuple[str, float, float, float | None, float | None]], color, note_better: list[str]) -> go.Figure:
    """Small-multiple bars: each row = (metric, player HIGH, player ORD, league HIGH, league ORD)."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=len(rows), subplot_titles=[f"{m}<br><span style='font-size:11px;color:{MUTED}'>{nb}</span>" for (m, *_), nb in zip(rows, note_better)],
                        horizontal_spacing=0.09)
    for i, (m, h, o, lh, lo) in enumerate(rows, start=1):
        fig.add_trace(go.Bar(x=["HIGH", "ORDINARY"], y=[h, o], marker_color=[color, GREY], text=[_fmt(h), _fmt(o)], textposition="outside",
                             cliponaxis=False, showlegend=False, hovertemplate=f"{m} · %{{x}}: %{{y:.2f}}<extra></extra>"), row=1, col=i)
        top = max([v for v in (h, o, lh, lo) if v is not None and not pd.isna(v)] + [1])
        fig.update_yaxes(range=[0, top * 1.25], row=1, col=i, showticklabels=False, showgrid=False)
    fig.update_annotations(font=dict(size=13, color=INK))
    return style(fig, 300, legend=False, margin=dict(l=10, r=10, t=70, b=10))


def _fmt(v):
    if v is None or pd.isna(v):
        return "–"
    return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


def match_wpa_bars(pm: pd.DataFrame, color) -> go.Figure:
    pm = pm.sort_values("match_date")
    colors = [color if v > 0 else RED for v in pm["net_wpa"]]
    fig = go.Figure(go.Bar(x=list(range(1, len(pm) + 1)), y=pm["net_wpa"] * 100, marker_color=colors,
                           customdata=np.stack([pm["match_date"], pm["competition"], pm["balls"]], axis=-1),
                           hovertemplate="%{customdata[0]} · %{customdata[1]}<br>%{customdata[2]} HIGH balls<br>Net WPA: %{y:+.1f} pp<extra></extra>"))
    fig.update_xaxes(title="Matches with HIGH-pressure involvement (chronological)", showgrid=False)
    fig.update_yaxes(title="Net win probability (pp)")
    return style(fig, 300, legend=False)


def rank_interval(df: pd.DataFrame, color, height=None) -> go.Figure:
    """df: rows with name, rank, boot_p5, boot_p95, boot_median_boot_rank (top first)."""
    d = df.iloc[::-1]
    labels = [f"{int(r)}. {n}" for r, n in zip(d["rank"], d["name"])]
    fig = go.Figure()
    for lab, p5, p95 in zip(labels, d["boot_p5"], d["boot_p95"]):
        fig.add_trace(go.Scatter(x=[p5, p95], y=[lab, lab], mode="lines", line=dict(color=color, width=10), opacity=0.35, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["boot_median_boot_rank"], y=labels, mode="markers", name="Bootstrap median rank", marker=dict(color=color, size=10),
                             customdata=np.stack([d["boot_p5"], d["boot_p95"], d["boot_top10_prob_pct"]], axis=-1),
                             hovertemplate="%{y}<br>Median rank %{x:.1f}<br>90% range %{customdata[0]:.0f}–%{customdata[1]:.0f}<br>P(top 10) %{customdata[2]:.0f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["rank"], y=labels, mode="markers", name="Submitted rank", marker=dict(color=INK, size=8, symbol="diamond")))
    fig.add_vline(x=10.5, line=dict(color=MUTED, dash="dot", width=1))
    fig.update_xaxes(title="Bootstrap rank (bar = 5th–95th pct)", range=[0, 75])
    return style(fig, height or 44 * len(df) + 90)


# ---------- pressure ----------
def li_by_over(df: pd.DataFrame, high_cut: float) -> go.Figure:
    fig = go.Figure()
    fig.add_vrect(x0=0.5, x1=2.5, fillcolor="#FFFFFF", opacity=0.05, line_width=0,
                  annotation_text="First 12 balls excluded", annotation_position="top left", annotation_font=dict(color=MUTED, size=11))
    for inn, col, name in ((1, BOWL, "Innings 1 (setting a total)"), (2, BAT, "Innings 2 (chasing)")):
        d = df[df["innings"] == inn]
        fig.add_trace(go.Scatter(x=d["over"], y=d["mean_li"], mode="lines+markers", name=name, line=dict(color=col, width=3), marker=dict(size=6),
                                 customdata=np.stack([d["deliveries"], d["share_high_pct"]], axis=-1),
                                 hovertemplate="Over %{x}<br>Mean LI %{y:.2f}<br>%{customdata[1]:.0f}% of balls HIGH<br>%{customdata[0]:,} live balls<extra></extra>"))
    fig.add_hline(y=high_cut, line=dict(color=MUTED, dash="dash", width=1), annotation_text=f"HIGH threshold (LI ≥ {high_cut:.2f})",
                  annotation_position="top left", annotation_font=dict(color=MUTED, size=11))
    fig.update_xaxes(title="Over", dtick=1)
    fig.update_yaxes(title="Mean Leverage Index (live balls)")
    return style(fig, 400)


def li_histogram(h: pd.DataFrame, show: list[str]) -> go.Figure:
    colors = {"HIGH": BAT, "Middle (unused)": "#8C7A55", "ORDINARY": BOWL, "Dead / decided": "#4A5468", "Early innings (first 12 balls)": "#6E5A8A"}
    fig = go.Figure()
    for c in show:
        d = h[h["category"] == c]
        fig.add_trace(go.Bar(x=d["bin"] + 0.05, y=d["deliveries"], name=c, marker_color=colors.get(c, GREY), width=0.1,
                             hovertemplate=f"{c}<br>LI %{{x:.2f}}<br>%{{y:,}} deliveries<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0)
    fig.update_xaxes(title="Leverage Index (capped at 6)", range=[0, 6])
    fig.update_yaxes(title="Deliveries")
    return style(fig, 380)


def example_states(ex: pd.DataFrame) -> go.Figure:
    ex = ex[ex["deliveries"] >= 3]
    fig = go.Figure(go.Bar(x=ex["wickets_left"].astype(int), y=ex["mean_li"], marker_color=[BAT if w in (2,) else (BOWL if w == 6 else GREY) for w in ex["wickets_left"]],
                           text=[f"LI {v:.2f}" for v in ex["mean_li"]], textposition="outside", cliponaxis=False,
                           customdata=np.stack([ex["deliveries"], ex["mean_win_prob"] * 100], axis=-1),
                           hovertemplate="%{x} wickets in hand<br>Mean LI %{y:.2f}<br>Chasing side win prob. %{customdata[1]:.0f}%<br>%{customdata[0]} deliveries observed<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=MUTED, dash="dot", width=1))
    fig.update_xaxes(title="Wickets in hand (chasing side)", dtick=1)
    fig.update_yaxes(title="Mean Leverage Index", range=[0, ex["mean_li"].max() * 1.25])
    return style(fig, 340, legend=False)


def calibration(bins: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1, 1, 0], y=[-0.05, 0.95, 1.05, 0.05], fill="toself", fillcolor="rgba(255,255,255,0.05)", line=dict(width=0),
                             hoverinfo="skip", name="±0.05 band"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dash"), name="Perfect calibration", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=bins["Mean Predicted WP"], y=bins["Observed Win Rate"], mode="lines+markers", name="Model (out-of-fold)",
                             line=dict(color=GOLD, width=3), marker=dict(size=9),
                             customdata=np.stack([bins["Bin Range"], bins["Deliveries"], bins["Absolute Error"]], axis=-1),
                             hovertemplate="Bin %{customdata[0]}<br>Predicted %{x:.1%}<br>Observed %{y:.1%}<br>Error %{customdata[2]:.3f}<br>%{customdata[1]:,} deliveries<extra></extra>"))
    fig.update_xaxes(title="Predicted chance of winning", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Observed win rate", tickformat=".0%", range=[0, 1])
    return style(fig, 420)


def donut(labels, values, colors, center: str = "", height=300) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62, marker=dict(colors=colors, line=dict(color="#0B1220", width=2)),
                           textinfo="percent", sort=False, hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"))
    fig.add_annotation(text=center, showarrow=False, font=dict(size=14, color=INK))
    fig = style(fig, height, legend=True, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5))
    return fig
