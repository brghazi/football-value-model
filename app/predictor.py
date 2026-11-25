"""Turns the app's form inputs into the exact feature row the LightGBM model was trained on."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PERF_COLS = ["games_12m", "goals_12m", "assists_12m",
             "minutes_played_12m", "yellow_cards_12m", "red_cards_12m"]


def load_assets():
    bundle = joblib.load(ROOT / "models" / "lightgbm_final.joblib")
    with open(ROOT / "app" / "lookup.json", encoding="utf-8") as f:
        lookup = json.load(f)
    return bundle["model"], bundle["features"], bundle["cat_cols"], lookup


def club_context(lookup, club):
    """Context values for one club entry of lookup["clubs"] (average value, league info)."""
    league = lookup["leagues"][club["league"]]
    return {"club_avg_lv_prev": club["club_avg_lv"], "club_n_prev": club["club_n_prev"],
            "league": club["league"], "league_avg_lv_prev": league["avg_lv"],
            "league_covered": league["covered"]}


def build_features(inp, ctx, features, cat_cols, categories):
    """inp: the player's details from the form; ctx: the club/league values from club_context."""
    has_stats = int(inp["games"] > 0)
    perf = [inp["games"], inp["goals"], inp["assists"],
            inp["minutes"], inp["yellow"], inp["red"]]

    # Same rule as training: no games in a league without performance data means "unknown", not zero
    if not has_stats and not ctx["league_covered"]:
        perf = [np.nan] * 6

    # Per-90 rates only exist from 450 minutes, as in training
    enough = inp["minutes"] >= 450
    mins = max(inp["minutes"], 1)

    row = {
        "age": inp["age"], "age_sq": inp["age"] ** 2, "height_in_cm": inp["height_in_cm"],
        "has_stats": has_stats, "league_covered": ctx["league_covered"],
        **dict(zip(PERF_COLS, perf)),
        "goals_per90": inp["goals"] / mins * 90 if enough else np.nan,
        "assists_per90": inp["assists"] / mins * 90 if enough else np.nan,
        "club_n_prev": ctx["club_n_prev"], "club_avg_lv_prev": ctx["club_avg_lv_prev"],
        "league_avg_lv_prev": ctx["league_avg_lv_prev"],
        "position": inp["position"], "sub_position": inp["sub_position"],
        "foot": inp["foot"], "country_of_citizenship": inp["country_of_citizenship"],
        "player_club_domestic_competition_id": ctx["league"],
    }
    X = pd.DataFrame([row])[features]
    for c in cat_cols:  # using the training category lists so the codes match the model
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X


def predict_value(model, X, lookup):
    """Returns (estimate, low, high) in euros; low/high come from the test-set error band."""
    est = float(np.expm1(model.predict(X)[0]))
    band = lookup.get("error_band")
    if band is None:
        return est, None, None
    return est, est * band["low_mult"], est * band["high_mult"]