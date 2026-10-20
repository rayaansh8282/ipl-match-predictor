# 🏏 IPL Match Winner Predictor

Predict the winner of an IPL match using **pre-match information only**: team strength (Elo), recent form,
head-to-head record, venue record, and toss outcome. Built with pandas + scikit-learn.

## Why this project is different from most Kaggle notebooks
- **No data leakage.** Features are built chronologically; each match only sees matches played before it.
- **Time-based evaluation.** Train on past seasons, test on the latest season (not a random split).
- **Honest baselines.** Compared against a majority-class baseline. IPL is noisy, so expect ~55–65% accuracy, not 90%.
- **Reusable.** Saved model + CLI for predicting any fixture.

## Project structure
```
ipl-match-predictor/
├── data/                 # put matches.csv here (see data/README.md)
├── src/
│   ├── data.py           # loading, cleaning, team-name normalisation
│   ├── features.py       # leak-free Elo / form / H2H / venue features
│   ├── train.py          # train + evaluate + save
│   ├── predict.py        # CLI prediction
│   └── make_sample_data.py  # synthetic data for smoke-testing
├── tests/                # pytest
├── models/  reports/     # outputs
└── requirements.txt
```

## Quick start
```bash
git clone https://github.com/<your-username>/ipl-match-predictor.git
cd ipl-match-predictor
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Get data: download matches.csv from Kaggle (IPL Complete Dataset 2008-2024) into data/
# 2. Train
python -m src.train --data data/matches.csv
# 3. Predict
python -m src.predict --team1 "Mumbai Indians" --team2 "Chennai Super Kings" \
  --venue "Wankhede Stadium" --toss-winner "Mumbai Indians" --toss-decision field
```
No dataset yet? `python -m src.make_sample_data` then train with `--data data/sample_matches.csv`
(synthetic, only to verify the pipeline works).

## Features
| Feature | Meaning |
|---|---|
| `elo_diff` | Elo rating of team1 minus team2 |
| `form_diff` | Win rate in last 5 matches, team1 minus team2 |
| `h2h_rate` | Smoothed head-to-head win rate of team1 vs team2 |
| `venue_rate_diff` | Smoothed win rate at this venue, team1 minus team2 |
| `toss_won_by_team1` | 1 if team1 won the toss |
| `team1_bats_first` | 1 if team1 bats first after toss decision |

## Models
Logistic Regression, Random Forest, Gradient Boosting — best chosen by log-loss on the held-out latest season,
then refit on all data. Results are written to `reports/metrics.json` and `reports/model_comparison.png`.

## Run tests
```bash
pytest
```

## Ideas to extend
- Add player-level features from `deliveries.csv` (batting/bowling strength of the playing XI)
- Live in-match win probability using score, wickets, overs (second innings)
- Streamlit web app for predictions
- Hyperparameter tuning with time-series cross-validation

## License
MIT
