"""Generate SYNTHETIC IPL-like matches.csv so the pipeline can be tested without Kaggle data."""
import argparse
import itertools

import numpy as np
import pandas as pd

TEAMS = ["Mumbai Indians", "Chennai Super Kings", "Kolkata Knight Riders", "Royal Challengers Bengaluru",
         "Delhi Capitals", "Punjab Kings", "Rajasthan Royals", "Sunrisers Hyderabad"]
VENUES = ["Wankhede Stadium", "MA Chidambaram Stadium", "Eden Gardens", "M Chinnaswamy Stadium",
          "Arun Jaitley Stadium", "Narendra Modi Stadium"]


def main(out, seed=42):
    rng = np.random.default_rng(seed)
    strength = {t: rng.normal(0, 0.5) for t in TEAMS}
    rows, mid = [], 1
    for year in range(2008, 2025):
        pairs = list(itertools.permutations(TEAMS, 2))
        rng.shuffle(pairs)
        for i, (t1, t2) in enumerate(pairs[:56]):
            venue = rng.choice(VENUES)
            toss = rng.choice([t1, t2])
            dec = rng.choice(["bat", "field"])
            p1 = 1 / (1 + np.exp(-(strength[t1] - strength[t2])))
            winner = t1 if rng.random() < p1 else t2
            date = pd.Timestamp(f"{year}-04-01") + pd.Timedelta(days=i)
            rows.append(dict(id=mid, season=year, date=date.date(), team1=t1, team2=t2, venue=venue,
                             toss_winner=toss, toss_decision=dec, winner=winner, result="normal"))
            mid += 1
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} synthetic matches to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sample_matches.csv")
    main(p.parse_args().out)
