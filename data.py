"""Load and clean the IPL matches table."""
import pandas as pd

# Franchises that were renamed / replaced -> canonical current name
TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

REQUIRED = ["date", "team1", "team2", "toss_winner", "toss_decision", "venue", "winner"]


def normalize_team(name: str) -> str:
    name = str(name).strip()
    return TEAM_MAP.get(name, name)


def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    # Matches with no winner (no result / abandoned) can't be used as labels
    df = df.dropna(subset=["winner", "team1", "team2", "toss_winner"]).copy()

    for col in ["team1", "team2", "toss_winner", "winner"]:
        df[col] = df[col].map(normalize_team)
    df["venue"] = df["venue"].astype(str).str.split(",").str[0].str.strip()
    df["toss_decision"] = df["toss_decision"].str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # winner must be one of the two teams (drops odd rows)
    df = df[(df["winner"] == df["team1"]) | (df["winner"] == df["team2"])]
    return df.sort_values("date", kind="stable").reset_index(drop=True)
