"""Leak-free, chronological feature engineering.

Every feature for a match is computed ONLY from matches played before it.
State (Elo, form, head-to-head, venue record) is updated after the row is built.
"""
from collections import defaultdict, deque

import pandas as pd

BASE_ELO = 1500.0
K_FACTOR = 24.0
FORM_WINDOW = 5

FEATURES = [
    "elo_diff",
    "form_diff",
    "h2h_rate",
    "venue_rate_diff",
    "toss_won_by_team1",
    "team1_bats_first",
]


def _base_elo():
    return BASE_ELO


def _pair():
    return [0, 0]


class FeatureBuilder:
    def __init__(self, form_window: int = FORM_WINDOW):
        self.form_window = form_window
        self.elo = defaultdict(_base_elo)
        self.form = {}
        self.h2h = defaultdict(_pair)      # (a, b) -> [a_wins, games]
        self.venue = defaultdict(_pair)    # (venue, team) -> [wins, games]

    # ---- reading state -------------------------------------------------
    def _form(self, team):
        f = self.form.get(team)
        return sum(f) / len(f) if f else 0.5

    def _venue_rate(self, venue, team):
        w, n = self.venue.get((venue, team), (0, 0))
        return (w + 1) / (n + 2)  # Laplace smoothing

    def features(self, team1, team2, venue, toss_winner, toss_decision) -> dict:
        w, n = self.h2h.get((team1, team2), (0, 0))
        team1_bats_first = int(
            (toss_winner == team1 and toss_decision == "bat")
            or (toss_winner == team2 and toss_decision == "field")
        )
        return {
            "elo_diff": self.elo.get(team1, BASE_ELO) - self.elo.get(team2, BASE_ELO),
            "form_diff": self._form(team1) - self._form(team2),
            "h2h_rate": (w + 1) / (n + 2),
            "venue_rate_diff": self._venue_rate(venue, team1) - self._venue_rate(venue, team2),
            "toss_won_by_team1": int(toss_winner == team1),
            "team1_bats_first": team1_bats_first,
        }

    # ---- updating state ------------------------------------------------
    def update(self, team1, team2, venue, winner):
        t1_won = int(winner == team1)

        e1, e2 = self.elo[team1], self.elo[team2]
        exp1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        self.elo[team1] = e1 + K_FACTOR * (t1_won - exp1)
        self.elo[team2] = e2 + K_FACTOR * ((1 - t1_won) - (1 - exp1))

        for team, won in ((team1, t1_won), (team2, 1 - t1_won)):
            self.form.setdefault(team, deque(maxlen=self.form_window)).append(won)
            self.venue[(venue, team)][0] += won
            self.venue[(venue, team)][1] += 1

        self.h2h[(team1, team2)][0] += t1_won
        self.h2h[(team1, team2)][1] += 1
        self.h2h[(team2, team1)][0] += 1 - t1_won
        self.h2h[(team2, team1)][1] += 1

    @property
    def teams(self):
        return sorted(self.elo.keys())


def build_dataset(df: pd.DataFrame):
    """Return (X, y, years, fitted FeatureBuilder) for a chronologically sorted matches df."""
    fb = FeatureBuilder()
    rows, y = [], []
    for r in df.itertuples(index=False):
        rows.append(fb.features(r.team1, r.team2, r.venue, r.toss_winner, r.toss_decision))
        y.append(int(r.winner == r.team1))
        fb.update(r.team1, r.team2, r.venue, r.winner)
    X = pd.DataFrame(rows)[FEATURES]
    return X, pd.Series(y, name="team1_won"), df["year"].reset_index(drop=True), fb
