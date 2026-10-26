
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = (
    ROOT / "data" / "processed"
    / "actor_week_features_filtered.csv"
)

SCORES_FILE = (
    ROOT / "reports" / "real_data"
    / "anomaly_scores.csv"
)

OUTPUT_DIR = (
    ROOT / "reports"
    / "activity_stratified_feature_analysis"
)

ACTIVITY_FEATURE = "event_count"

TEMPORAL_FEATURES = [
    "repository_transitions",
    "transitions_per_observed_span_hour",
    "short_horizon_reentry_rate",
    "normalized_repo_entropy",
]

ACTIVITY_GROUPS = [
    "Low (2-5)",
    "Moderate (6-20)",
    "High (21-50)",
    "Very high (>50)",
]


def assign_activity_group(event_count):
    if 2 <= event_count <= 5:
        return "Low (2-5)"
    elif 6 <= event_count <= 20:
        return "Moderate (6-20)"
    elif 21 <= event_count <= 50:
        return "High (21-50)"
    elif event_count > 50:
        return "Very high (>50)"
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")

    features = pd.read_csv(FEATURES_FILE)
    scores = pd.read_csv(SCORES_FILE)

    keys = ["actor", "week_start"]

    required_features = keys + [ACTIVITY_FEATURE] + TEMPORAL_FEATURES
    required_scores = keys + ["extended_flag"]

    missing_features = set(required_features) - set(features.columns)
    missing_scores = set(required_scores) - set(scores.columns)

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {sorted(missing_features)}"
        )

    if missing_scores:
        raise ValueError(
            f"Missing score columns: {sorted(missing_scores)}"
        )

    data = features.merge(
        scores[required_scores],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    data["activity_group"] = data[ACTIVITY_FEATURE].apply(
        assign_activity_group
    )

    data = data.dropna(subset=["activity_group"]).copy()

    data["model_status"] = (
        data["extended_flag"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "anomaly": "Flagged",
            "inlier": "Unflagged",
        })
    )

    data = data.dropna(subset=["model_status"]).copy()

    print(f"Matched records analyzed: {len(data):,}")
    print(f"Extended-model flagged: {(data['model_status'] == 'Flagged').sum():,}")

    # Compare feature values within each activity group.
    results = []

    for group in ACTIVITY_GROUPS:
        group_data = data[data["activity_group"] == group]

        for feature in TEMPORAL_FEATURES:
            flagged = group_data.loc[
                group_data["model_status"] == "Flagged", feature
            ].dropna()

            unflagged = group_data.loc[
                group_data["model_status"] == "Unflagged", feature
            ].dropna()

            flagged_mean = flagged.mean()
            unflagged_mean = unflagged.mean()

            results.append({
                "activity_group": group,
                "feature": feature,
                "flagged_records": len(flagged),
                "unflagged_records": len(unflagged),
                "flagged_mean": flagged_mean,
                "unflagged_mean": unflagged_mean,
                "mean_difference": flagged_mean - unflagged_mean,
                "flagged_median": flagged.median(),
                "unflagged_median": unflagged.median(),
            })

    results_df = pd.DataFrame(results)

    results_path = OUTPUT_DIR / "within_group_feature_comparison.csv"
    results_df.to_csv(results_path, index=False)

    # Produce one chart for each temporal feature.
    for feature in TEMPORAL_FEATURES:
        subset = results_df[results_df["feature"] == feature]

        plot_data = subset.set_index("activity_group")[
            ["flagged_mean", "unflagged_mean"]
        ].reindex(ACTIVITY_GROUPS)

        ax = plot_data.plot(
            kind="bar",
            figsize=(10, 6),
        )

        ax.set_title(
            f"{feature}: Flagged vs Unflagged Within Activity Groups"
        )
        ax.set_xlabel("Activity group")
        ax.set_ylabel("Mean feature value")
        ax.legend(["Flagged", "Unflagged"])
        ax.grid(axis="y", alpha=0.3)

        plt.xticks(rotation=15)
        plt.tight_layout()

        safe_name = feature.replace("/", "_")
        plt.savefig(
            OUTPUT_DIR / f"{safe_name}_within_groups.png",
            dpi=200,
        )
        plt.close()

    # Save group counts for interpretation.
    group_counts = (
        data.groupby(
            ["activity_group", "model_status"],
            observed=False,
        )
        .size()
        .unstack(fill_value=0)
        .reindex(ACTIVITY_GROUPS, fill_value=0)
    )

    group_counts.to_csv(
        OUTPUT_DIR / "activity_group_flag_counts.csv"
    )

    print("\nWITHIN-GROUP FEATURE COMPARISON")
    print(
        results_df.round(4).to_string(index=False)
    )

    print("\nFLAG COUNTS BY ACTIVITY GROUP")
    print(group_counts.to_string())

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Comparison CSV: {results_path}")


if __name__ == "__main__":
    main()

