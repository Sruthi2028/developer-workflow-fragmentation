"""Sensitivity analysis for temporal-only Isolation Forest contamination settings.

Runs separate temporal-only models at contamination levels 1%, 2%, 5%, and 10%.
Outputs are descriptive: there are no verified real-data anomaly labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

INPUT_PATH = Path("data/processed/actor_week_features_filtered.csv")
OUTPUT_DIR = Path("reports/sensitivity_analysis")
CONTAMINATION_LEVELS = [0.01, 0.02, 0.05, 0.10]
RANDOM_SEED = 42

TEMPORAL_FEATURES = [
    "repository_transitions",
    "transitions_per_observed_span_hour",
    "short_horizon_reentry_rate",
]
LOG1P_FEATURES = [
    "repository_transitions",
    "transitions_per_observed_span_hour",
]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.resolve()}. Run this command from the project root."
        )

    required = ["actor", "week_start", "event_count"] + TEMPORAL_FEATURES
    data = pd.read_csv(path)
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required).copy()
    data["event_count"] = pd.to_numeric(data["event_count"], errors="coerce")
    data = data.dropna(subset=["event_count"]).copy()

    for col in LOG1P_FEATURES:
        data[col] = np.log1p(data[col].clip(lower=0))

    if data.duplicated(["actor", "week_start"]).any():
        raise ValueError("Duplicate actor/week keys found; resolve before analysis.")

    return data.reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data(INPUT_PATH)
    print(f"Loaded {len(data):,} actor-week records.")
    print(f"Unique actors: {data['actor'].nunique():,}")
    print("Model: temporal-only Isolation Forest")
    print("Contamination settings:", ", ".join(f"{x:.0%}" for x in CONTAMINATION_LEVELS))
    print("Reminder: contamination sets the expected flagged fraction; it is not accuracy.")

    matrix = data[TEMPORAL_FEATURES].astype(float).to_numpy()
    scaler = RobustScaler()
    scaled = scaler.fit_transform(matrix)

    outputs = data[["actor", "week_start", "event_count"]].copy()
    summary_rows = []
    flag_columns = []

    for contamination in CONTAMINATION_LEVELS:
        label = f"{int(contamination * 100):02d}pct"
        print(f"\nFitting model with contamination={contamination:.0%}...", flush=True)
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(scaled)
        scores = -model.decision_function(scaled)
        flags = model.predict(scaled) == -1

        score_col = f"score_{label}"
        flag_col = f"flag_{label}"
        outputs[score_col] = scores
        outputs[flag_col] = flags
        flag_columns.append(flag_col)

        summary_rows.append({
            "contamination": contamination,
            "records": len(data),
            "flagged_count": int(flags.sum()),
            "flagged_rate_pct": float(flags.mean() * 100),
            "median_anomaly_score": float(np.median(scores)),
            "score_95th_percentile": float(np.quantile(scores, 0.95)),
        })

    overlap_rows = []
    for i, cont_a in enumerate(CONTAMINATION_LEVELS):
        for j in range(i + 1, len(CONTAMINATION_LEVELS)):
            cont_b = CONTAMINATION_LEVELS[j]
            col_a = f"flag_{int(cont_a * 100):02d}pct"
            col_b = f"flag_{int(cont_b * 100):02d}pct"
            set_a = set(outputs.loc[outputs[col_a], ["actor", "week_start"]].itertuples(index=False, name=None))
            set_b = set(outputs.loc[outputs[col_b], ["actor", "week_start"]].itertuples(index=False, name=None))
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            overlap_rows.append({
                "contamination_a_pct": int(cont_a * 100),
                "contamination_b_pct": int(cont_b * 100),
                "flagged_a": len(set_a),
                "flagged_b": len(set_b),
                "shared_flags": intersection,
                "jaccard_overlap_pct": 100 * intersection / union if union else 0.0,
                "fraction_of_a_retained_in_b_pct": 100 * intersection / len(set_a) if set_a else 0.0,
            })

    activity_groups = pd.cut(
        data["event_count"],
        bins=[0, 1, 5, 9, 19, float("inf")],
        labels=["1 event", "2-5 events", "6-9 events", "10-19 events", "20+ events"],
        include_lowest=True,
    )
    group_rows = []
    for group_name, row_indices in activity_groups.groupby(activity_groups, observed=False).groups.items():
        row = {"activity_group": str(group_name), "records": len(row_indices)}
        for col in flag_columns:
            flags = outputs.loc[list(row_indices), col]
            row[col.replace("flag_", "flagged_rate_").replace("pct", "pct")] = (
                100 * flags.mean() if len(flags) else 0.0
            )
        group_rows.append(row)

    outputs.to_csv(OUTPUT_DIR / "temporal_sensitivity_scores_and_flags.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "contamination_summary.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(OUTPUT_DIR / "contamination_overlap.csv", index=False)
    pd.DataFrame(group_rows).to_csv(OUTPUT_DIR / "flag_rates_by_activity.csv", index=False)

    print("\nContamination summary:")
    print(pd.DataFrame(summary_rows).round(4).to_string(index=False))
    print("\nOverlap between contamination settings:")
    print(pd.DataFrame(overlap_rows).round(2).to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")
    print("Files: contamination_summary.csv, contamination_overlap.csv,")
    print("       flag_rates_by_activity.csv, temporal_sensitivity_scores_and_flags.csv")


if __name__ == "__main__":
    main()
