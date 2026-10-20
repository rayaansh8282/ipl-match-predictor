"""Train, evaluate (time-based split) and save the IPL match predictor.

    python -m src.train --data data/matches.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_matches
from src.features import FEATURES, build_dataset


def get_models():
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                                        learning_rate=0.05, random_state=42),
    }


def main(args):
    df = load_matches(args.data)
    X, y, years, fb = build_dataset(df)

    test_year = years.max()
    train_mask = years < test_year
    X_tr, y_tr, X_te, y_te = X[train_mask], y[train_mask], X[~train_mask], y[~train_mask]
    print(f"{len(df)} matches | train: {len(X_tr)} (<{test_year}) | test: {len(X_te)} ({test_year})")

    results = {"baseline_accuracy": float(max(y_te.mean(), 1 - y_te.mean())), "test_season": int(test_year)}
    scores = {}
    for name, model in get_models().items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        scores[name] = {
            "accuracy": float(accuracy_score(y_te, proba > 0.5)),
            "log_loss": float(log_loss(y_te, proba)),
        }
        print(f"{name:22s} acc={scores[name]['accuracy']:.3f}  log_loss={scores[name]['log_loss']:.3f}")
    results["models"] = scores

    best = min(scores, key=lambda n: scores[n]["log_loss"])
    results["best_model"] = best
    print(f"\nBest by log-loss: {best}  (majority-class baseline acc = {results['baseline_accuracy']:.3f})")

    # Refit the winner on ALL matches and save with the feature state
    final = get_models()[best].fit(X, y)
    Path(args.models_dir).mkdir(exist_ok=True)
    joblib.dump({"model": final, "builder": fb, "features": FEATURES}, Path(args.models_dir) / "ipl_model.joblib")

    Path(args.reports_dir).mkdir(exist_ok=True)
    (Path(args.reports_dir) / "metrics.json").write_text(json.dumps(results, indent=2))
    names = list(scores)
    plt.figure(figsize=(7, 4))
    plt.bar(names, [scores[n]["accuracy"] for n in names], color="#4c78a8")
    plt.axhline(results["baseline_accuracy"], color="red", ls="--", label="majority baseline")
    plt.ylim(0.3, 0.8); plt.ylabel(f"Accuracy on {test_year} season"); plt.legend()
    plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(Path(args.reports_dir) / "model_comparison.png", dpi=130)
    print(f"Saved model to {args.models_dir}/ipl_model.joblib and reports to {args.reports_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/matches.csv")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--reports-dir", default="reports")
    main(p.parse_args())
