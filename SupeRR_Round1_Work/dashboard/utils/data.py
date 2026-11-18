"""Cached loaders for the precomputed dashboard data (dashboard/data/*). No analysis is re-run here."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data"

ROLE_FILE = {"bat": "batters", "bowl": "bowlers"}

# The final submitted lists (Unstop). The app checks the data against these on load.
SUBMITTED = {
    "bat": ["Rinku Singh", "Glenn Maxwell", "Krunal Pandya", "Rashid Khan", "Tim David",
            "Suryakumar Yadav", "Harry Brook", "David Miller", "Sam Curran", "Jos Buttler"],
    "bowl": ["Josh Hazlewood", "Jasprit Bumrah", "Rashid Khan", "Varun Chakravarthy", "Yuzvendra Chahal",
             "Lockie Ferguson", "Matheesha Pathirana", "Gudakesh Motie", "Krunal Pandya", "Harshit Rana"],
}


@st.cache_data(show_spinner=False)
def csv(name: str) -> pd.DataFrame | None:
    p = DATA / name
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def manifest() -> dict:
    p = DATA / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(show_spinner=False)
def extras() -> dict:
    p = DATA / "extras_calibration.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(show_spinner=False)
def ratings(role: str) -> pd.DataFrame:
    """All eligible players for a role, in final composite order. Ratings converted to x1000 display units."""
    df = csv(f"ratings_{ROLE_FILE[role]}.csv").copy()
    for c in ("pressure_rating", "ordinary_rating", "resilience"):
        df[f"{c}_k"] = df[c] * 1000.0
    df["label"] = df["rank"].astype(str) + ". " + df["name"]
    return df


def top10(role: str) -> pd.DataFrame:
    return ratings(role).query("submitted_top10").sort_values("rank")


def match_pressure(role: str) -> pd.DataFrame | None:
    return csv(f"match_pressure_{ROLE_FILE[role]}.csv")


def verify_submitted() -> list[str]:
    """Return a list of problems if the data does not reproduce the submitted rankings (should be empty)."""
    problems = []
    for role, names in SUBMITTED.items():
        got = top10(role)["name"].tolist()
        if got != names:
            problems.append(f"{role}: data order {got} differs from submitted {names}")
    return problems
