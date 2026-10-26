
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

FEATURES_FILE = (
    ROOT / "data" / "processed"
    / "actor_week_features_filtered.csv"
)

SCORES_FILE = (
    ROOT / "reports" / "real_data"
    / "anomaly_scores.csv"
)

OUTPUT_DIR = (
    ROOT / "reports" / "final_dashboard"
    / "feature_level_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "event_count",
    "distinct_repositories",
    "normalized_repo_entropy",
    "repository_transitions",
    "transitions_per_observed_span_hour",
    "short_horizon_reentry_rate",
]


def save_plot(filename):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path.relative_to(ROOT)}")


def main():
    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------

    features = pd.read_csv(FEATURES_FILE)
    scores = pd.read_csv(SCORES_FILE)

    keys = ["actor", "week_start"]

    required_features = set(keys + FEATURES)
    required_scores = set(
        keys + [
            "baseline_anomaly_score",
            "baseline_flag",
            "extended_anomaly_score",
            "extended_flag",
        ]
    )

    if not required_features.issubset(features.columns):
        missing = required_features - set(features.columns)
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    if not required_scores.issubset(scores.columns):
        missing = required_scores - set(scores.columns)
        raise ValueError(f"Missing score columns: {sorted(missing)}")

    print(f"Feature records loaded: {len(features):,}")
    print(f"Score records loaded: {len(scores):,}")

    # Verify the join keys before merging.
    if features.duplicated(keys).any():
        raise ValueError("Feature dataset contains duplicate actor-week keys.")

    if scores.duplicated(keys).any():
        raise ValueError("Score dataset contains duplicate actor-week keys.")

    merged = features.merge(
        scores,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(features):
        print(
            "WARNING: Some feature records did not match score records. "
            f"Matched records: {len(merged):,}"
        )

    merged.to_csv(
        OUTPUT_DIR / "joined_features_and_anomaly_scores.csv",
        index=False,
    )

    print(f"Matched records: {len(merged):,}")

    # --------------------------------------------------
    # 2. FLAGGED VS UNFLAGGED GROUPS
    # --------------------------------------------------

    merged["extended_is_anomaly"] = (
        merged["extended_flag"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("anomaly")
    )

    merged["baseline_is_anomaly"] = (
        merged["baseline_flag"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("anomaly")
    )

    group_names = {
        False: "Extended model: inlier",
        True: "Extended model: anomaly",
    }

    group_summary = (
        merged.groupby("extended_is_anomaly")[FEATURES]
        .agg(["count", "mean", "median", "std"])
    )

    group_summary.to_csv(
        OUTPUT_DIR / "feature_group_summary.csv"
    )

    # Long-form table for easier comparison.
    rows = []

    for feature in FEATURES:
        flagged = merged.loc[
            merged["extended_is_anomaly"], feature
        ].dropna()

        unflagged = merged.loc[
            ~merged["extended_is_anomaly"], feature
        ].dropna()

        rows.append({
            "feature": feature,
            "flagged_records": len(flagged),
            "unflagged_records": len(unflagged),
            "flagged_mean": flagged.mean(),
            "unflagged_mean": unflagged.mean(),
            "mean_difference_flagged_minus_unflagged": (
                flagged.mean() - unflagged.mean()
            ),
            "flagged_median": flagged.median(),
            "unflagged_median": unflagged.median(),
            "flagged_std": flagged.std(),
            "unflagged_std": unflagged.std(),
        })

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        OUTPUT_DIR / "feature_comparison_flagged_vs_inliers.csv",
        index=False,
    )

    print("\nFEATURE COMPARISON")
    print(
        comparison[
            [
                "feature",
                "flagged_mean",
                "unflagged_mean",
                "mean_difference_flagged_minus_unflagged",
            ]
        ].round(4).to_string(index=False)
    )

    # --------------------------------------------------
    # 3. FEATURE MEAN COMPARISON GRAPH
    # --------------------------------------------------

    means = merged.groupby("extended_is_anomaly")[FEATURES].mean().T

    ax = means.rename(columns=group_names).plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_title("Mean Feature Values: Flagged vs. Unflagged Records")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean feature value")
    ax.tick_params(axis="x", rotation=35)

    save_plot("feature_mean_comparison.png")

    # --------------------------------------------------
    # 4. FEATURE DISTRIBUTION GRAPHS
    # --------------------------------------------------

    for feature in FEATURES:
        flagged = merged.loc[
            merged["extended_is_anomaly"], feature
        ].dropna()

        unflagged = merged.loc[
            ~merged["extended_is_anomaly"], feature
        ].dropna()

        if flagged.empty or unflagged.empty:
            print(f"[SKIP] Insufficient data for {feature}")
            continue

        plt.figure(figsize=(8, 5))

        plt.boxplot(
            [unflagged, flagged],
            tick_labels=[
                "Extended model: inlier",
                "Extended model: anomaly",
            ],
            showfliers=False,
        )

        plt.title(f"{feature}: Flagged vs. Unflagged")
        plt.xlabel("Model-generated group")
        plt.ylabel(feature)

        save_plot(f"distribution_{feature}.png")

    # --------------------------------------------------
    # 5. TOP FLAGGED RECORDS WITH FEATURES
    # --------------------------------------------------

    flagged_records = merged.loc[
        merged["extended_is_anomaly"]
    ].copy()

    # The model's saved score convention should be checked before
    # ranking. Here we retain the original score and use descending
    # order to inspect the largest values, not to assert severity.
    flagged_records = flagged_records.sort_values(
        "extended_anomaly_score",
        ascending=False,
    )

    selected_columns = (
        keys
        + FEATURES
        + [
            "baseline_anomaly_score",
            "baseline_flag",
            "extended_anomaly_score",
            "extended_flag",
        ]
    )

    flagged_records[selected_columns].head(100).to_csv(
        OUTPUT_DIR / "top_100_flagged_with_features.csv",
        index=False,
    )

    # --------------------------------------------------
    # 6. SUMMARY METRICS
    # --------------------------------------------------

    summary = pd.DataFrame([{
        "feature_records": len(features),
        "score_records": len(scores),
        "matched_records": len(merged),
        "unique_actors_matched": merged["actor"].nunique(),
        "extended_flagged_records": int(
            merged["extended_is_anomaly"].sum()
        ),
        "extended_inlier_records": int(
            (~merged["extended_is_anomaly"]).sum()
        ),
        "baseline_flagged_records": int(
            merged["baseline_is_anomaly"].sum()
        ),
        "missing_feature_values": int(
            merged[FEATURES].isna().sum().sum()
        ),
    }])

    summary.to_csv(
        OUTPUT_DIR / "analysis_summary.csv",
        index=False,
    )

    print("\nANALYSIS SUMMARY")
    print(summary.to_string(index=False))

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

