"""Exploratory unsupervised anomaly scoring for actor-week features."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import joblib

FEATURES = [
    "event_count", "distinct_repositories", "normalized_repo_entropy",
    "repository_transitions", "transitions_per_observed_span_hour",
    "short_horizon_reentry_rate", "active_utc_dates"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/actor_week_features.csv")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    src = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    if len(df) < 20:
        raise SystemExit(
            f"Only {len(df)} actor-week rows. Need at least 20 for this exploratory "
            "anomaly detector; collect more data and do not interpret tiny samples."
        )
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required features: {missing}")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("model", IsolationForest(
            n_estimators=200, contamination=args.contamination,
            random_state=args.seed
        ))
    ])
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    pipe.fit(X)
    # sklearn: lower decision_function values are more anomalous.
    df["anomaly_score"] = -pipe.decision_function(X)
    df["is_flagged"] = (pipe.predict(X) == -1)
    df.sort_values("anomaly_score", ascending=False).to_csv(out / "anomaly_scores.csv", index=False)
    joblib.dump(pipe, out / "isolation_forest_pipeline.joblib")
    summary = [
        f"Rows scored: {len(df)}",
        f"Flagged rows: {int(df['is_flagged'].sum())}",
        f"Flagged fraction: {df['is_flagged'].mean():.4f}",
        f"Random seed: {args.seed}",
        f"Contamination parameter: {args.contamination}",
        "Interpretation: anomaly flags indicate unusual feature combinations only; they are not labels of burnout, stress, productivity, or poor performance."
    ]
    (out / "model_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    print(f"Saved outputs under {out.resolve()}")

if __name__ == "__main__":
    main()
