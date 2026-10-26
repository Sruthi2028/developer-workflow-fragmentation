"""Compare activity-only, temporal-only, and combined Isolation Forest models.

This is a descriptive comparison on unlabeled real GitHub activity data.
It does not measure predictive accuracy or establish that flagged actors have
genuinely fragmented workflows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

INPUT_PATH = Path("data/processed/actor_week_features_filtered.csv")
OUTPUT_DIR = Path("reports/real_data_comparison")
CONTAMINATION = 0.05
RANDOM_SEED = 42

ACTIVITY_FEATURES = [
    "event_count",
    "distinct_repositories",
    "normalized_repo_entropy",
]
TEMPORAL_FEATURES = [
    "repository_transitions",
    "transitions_per_observed_span_hour",
    "short_horizon_reentry_rate",
]
MODEL_FEATURES = {
    "activity_baseline": ACTIVITY_FEATURES,
    "temporal_only": TEMPORAL_FEATURES,
    "combined": ACTIVITY_FEATURES + TEMPORAL_FEATURES,
}
LOG1P_FEATURES = [
    "event_count",
    "distinct_repositories",
    "repository_transitions",
    "transitions_per_observed_span_hour",
]


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.resolve()}. Run this from the project root."
        )

    required = ["actor", "week_start"] + ACTIVITY_FEATURES + TEMPORAL_FEATURES
    data = pd.read_csv(path)
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required).copy()
    data["raw_event_count"] = data["event_count"].astype(float)

    # Match the transformations used by the existing real-data experiment.
    for col in LOG1P_FEATURES:
        data[col] = np.log1p(data[col].clip(lower=0))

    if data.duplicated(["actor", "week_start"]).any():
        raise ValueError("Duplicate actor/week keys found; resolve before comparing.")

    return data.reset_index(drop=True)


def fit_and_score(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = RobustScaler()
    scaled = scaler.fit_transform(matrix)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(scaled)

    # Higher scores mean more unusual. Isolation Forest's decision_function
    # is lower for more anomalous records, so negate it.
    scores = -model.decision_function(scaled)
    flags = model.predict(scaled) == -1
    return scores, flags


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_features(INPUT_PATH)
    print(f"Loaded {len(data):,} actor-week records.")
    print(f"Unique actors: {data['actor'].nunique():,}")
    print(
        "Caution: scores and flags are exploratory because this dataset has "
        "no independently verified fragmentation labels."
    )

    outputs = data[["actor", "week_start", "raw_event_count"]].rename(
        columns={"raw_event_count": "event_count"}
    ).copy()
    metrics = []
    flagged_sets: dict[str, set[tuple[str, str]]] = {}

    for model_name, features in MODEL_FEATURES.items():
        print(f"\nFitting {model_name} ({len(features)} features)...")
        scores, flags = fit_and_score(data[features].astype(float).to_numpy())

        score_col = f"{model_name}_score"
        flag_col = f"{model_name}_flag"
        outputs[score_col] = scores
        outputs[flag_col] = np.where(flags, "anomaly", "inlier")

        keys = list(zip(data["actor"].astype(str), data["week_start"].astype(str)))
        flagged_sets[model_name] = {
            key for key, flagged in zip(keys, flags) if flagged
        }

        metrics.append({
            "model": model_name,
            "feature_count": len(features),
            "records": len(data),
            "flagged_count": int(flags.sum()),
            "flagged_rate_pct": float(flags.mean() * 100),
            "median_score": float(np.median(scores)),
            "score_95th_percentile": float(np.quantile(scores, 0.95)),
            "features": ", ".join(features),
        })

    comparison = []
    names = list(MODEL_FEATURES)
    for i, model_a in enumerate(names):
        for model_b in names[i + 1:]:
            set_a = flagged_sets[model_a]
            set_b = flagged_sets[model_b]
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            comparison.append({
                "model_a": model_a,
                "model_b": model_b,
                "flagged_a": len(set_a),
                "flagged_b": len(set_b),
                "shared_flags": intersection,
                "only_a": len(set_a - set_b),
                "only_b": len(set_b - set_a),
                "jaccard_overlap_pct": (100 * intersection / union) if union else 0.0,
            })

    outputs.to_csv(OUTPUT_DIR / "model_scores_and_flags.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUTPUT_DIR / "model_comparison_summary.csv", index=False)
    pd.DataFrame(comparison).to_csv(OUTPUT_DIR / "flag_overlap_comparison.csv", index=False)

    # Check whether each model's flags are concentrated among high-event records.
    activity_groups = pd.cut(
        data["raw_event_count"],
        bins=[0, 1, 5, 9, 19, float("inf")],
        labels=["1 event", "2-5 events", "6-9 events", "10-19 events", "20+ events"],
        include_lowest=True,
    )
    group_rows = []
    for group_name, row_indices in activity_groups.groupby(activity_groups, observed=False).groups.items():
        group_idx = list(row_indices)
        row = {"activity_group": str(group_name), "records": len(group_idx)}
        for model_name in MODEL_FEATURES:
            flag_col = f"{model_name}_flag"
            flags = outputs.loc[group_idx, flag_col].eq("anomaly")
            row[f"{model_name}_flagged_count"] = int(flags.sum())
            row[f"{model_name}_flagged_rate_pct"] = float(flags.mean() * 100) if len(flags) else 0.0
        group_rows.append(row)
    pd.DataFrame(group_rows).to_csv(OUTPUT_DIR / "flag_rates_by_activity.csv", index=False)

    print("\nModel summary:")
    print(pd.DataFrame(metrics)[
        ["model", "feature_count", "flagged_count", "flagged_rate_pct", "median_score"]
    ].round(4).to_string(index=False))
    print("\nFlag overlap (Jaccard percentage):")
    print(pd.DataFrame(comparison)[
        ["model_a", "model_b", "shared_flags", "jaccard_overlap_pct"]
    ].round(2).to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_DIR.resolve()}")
    print("Files: model_scores_and_flags.csv, model_comparison_summary.csv,")
    print("       flag_overlap_comparison.csv, flag_rates_by_activity.csv")


if __name__ == "__main__":
    main()
