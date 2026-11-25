"""Checks the app's predictor in a clean process, then saves a typical error band into app/lookup.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "app"))
import predictor

model, features, cat_cols, lk = predictor.load_assets()
model.set_params(n_jobs=1)  # one thread is plenty here, and it matches how the app will predict

df = pd.read_csv(ROOT / "data" / "processed" / "features_v2.csv", parse_dates=["date"])
test = df[df["date"] >= "2025-01-01"].reset_index(drop=True)


def with_categories(X):
    """Applies the training category lists so the codes match what the model learned."""
    X = X.copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=lk["categories"][c])
    return X


def real_inputs(r):
    z = lambda v: 0 if pd.isna(v) else v  # an empty stat box in the app arrives as 0
    return {"age": r.age, "height_in_cm": r.height_in_cm, "position": r.position,
            "sub_position": r.sub_position, "foot": r.foot,
            "country_of_citizenship": r.country_of_citizenship,
            "games": z(r.games_12m), "goals": z(r.goals_12m), "assists": z(r.assists_12m),
            "minutes": z(r.minutes_played_12m), "yellow": z(r.yellow_cards_12m), "red": z(r.red_cards_12m)}


def real_context(r):
    return {"club_avg_lv_prev": r.club_avg_lv_prev, "club_n_prev": r.club_n_prev,
            "league": r.player_club_domestic_competition_id,
            "league_avg_lv_prev": r.league_avg_lv_prev, "league_covered": r.league_covered}


# 1) Predictions from the app's feature builder vs the training rows must match
sample = test.sample(300, random_state=1)
X_app = pd.concat([predictor.build_features(real_inputs(r), real_context(r), features, cat_cols, lk["categories"])
                   for r in sample.itertuples()], ignore_index=True)
X_ref = with_categories(sample[features].reset_index(drop=True))
print("Max prediction difference:", np.abs(model.predict(X_app) - model.predict(X_ref)).max())

# 2) Re-scoring the whole test set, which should reproduce about 0.77 from notebook 03
y = np.log1p(test["market_value_in_eur"]).to_numpy()
pred_log = model.predict(with_categories(test[features]))
r2 = 1 - ((y - pred_log) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print("Test R2 (log):", round(r2, 3))

# 3) The middle half of the errors becomes the "typical range" shown in the app
q25, q75 = np.quantile(y - pred_log, [0.25, 0.75])
lk["error_band"] = {"low_mult": round(float(np.exp(q25)), 3), "high_mult": round(float(np.exp(q75)), 3)}
print("Typical range:", lk["error_band"]["low_mult"], "x to", lk["error_band"]["high_mult"], "x the estimate")

with open(ROOT / "app" / "lookup.json", "w", encoding="utf-8") as f:
    json.dump(lk, f, ensure_ascii=False, indent=1)