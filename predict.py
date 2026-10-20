"""Predict a match from the command line.

    python -m src.predict --team1 "Mumbai Indians" --team2 "Chennai Super Kings" \
        --venue "Wankhede Stadium" --toss-winner "Mumbai Indians" --toss-decision field
"""
import argparse

import joblib
import pandas as pd

from src.data import normalize_team


def main(a):
    art = joblib.load(a.model)
    fb, model, feats = art["builder"], art["model"], art["features"]

    t1, t2, tw = normalize_team(a.team1), normalize_team(a.team2), normalize_team(a.toss_winner)
    for t in (t1, t2):
        if t not in fb.teams:
            raise SystemExit(f"Unknown team '{t}'. Known teams: {', '.join(fb.teams)}")
    if tw not in (t1, t2):
        raise SystemExit("--toss-winner must be team1 or team2")

    x = pd.DataFrame([fb.features(t1, t2, a.venue, tw, a.toss_decision.lower())])[feats]
    p1 = float(model.predict_proba(x)[0, 1])
    print(f"{t1}: {p1:.1%}\n{t2}: {1 - p1:.1%}")
    print(f"Predicted winner: {t1 if p1 >= 0.5 else t2}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--team1", required=True)
    p.add_argument("--team2", required=True)
    p.add_argument("--venue", required=True)
    p.add_argument("--toss-winner", required=True)
    p.add_argument("--toss-decision", choices=["bat", "field"], required=True)
    p.add_argument("--model", default="models/ipl_model.joblib")
    main(p.parse_args())
