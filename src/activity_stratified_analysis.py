
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

OUTPUT_DIR = ROOT / "reports" / "activity_stratified_analysis"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading feature and anomaly-score datasets...")

    features = pd.read_csv(FEATURES_FILE)
    scores = pd.read_csv(SCORES_FILE)

    print(f"Feature records loaded: {len(features):,}")
    print(f"Score records loaded: {len(scores):,}")

    keys = ["actor", "week_start"]

    required_features = keys + ["event_count"]
    required_scores = keys + ["baseline_flag", "extended_flag"]

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

    # Match feature records to their corresponding model predictions.
    data = features.merge(
        scores[required_scores],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    print(f"Matched records: {len(data):,}")

    if data.empty:
        raise ValueError("No matching feature and score records found.")

    # Define groups using event counts.
    conditions = [
        data["event_count"].between(2, 5),
        data["event_count"].between(6, 20),
        data["event_count"].between(21, 50),
        data["event_count"].gt(50),
    ]

    labels = [
        "Low (2-5)",
        "Moderate (6-20)",
        "High (21-50)",
        "Very high (>50)",
    ]

    data["activity_group"] = pd.Categorical(
        pd.Series(
            pd.NA,
            index=data.index,
            dtype="object",
        ).mask(conditions[0], labels[0])
         .mask(conditions[1], labels[1])
         .mask(conditions[2], labels[2])
         .mask(conditions[3], labels[3]),
        categories=labels,
        ordered=True,
    )

    data = data.dropna(subset=["activity_group"]).copy()

    # Calculate flagged records and rates within each group.
    results = []

    for group in labels:
        subset = data[data["activity_group"] == group]

        if subset.empty:
            continue

        for model, flag_column in [
            ("Activity baseline", "baseline_flag"),
            ("Extended model", "extended_flag"),
        ]:
            flags = (
    subset[flag_column]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("anomaly")
)

            flagged_count = int(flags.sum())

            results.append({
                "activity_group": group,
                "model": model,
                "records": len(subset),
                "flagged_records": flagged_count,
                "unflagged_records": len(subset) - flagged_count,
                "flagged_rate_percent": round(
                    100 * flagged_count / len(subset), 4
                ),
            })

    results_df = pd.DataFrame(results)

    results_path = (
        OUTPUT_DIR / "flag_rates_by_activity_group.csv"
    )
    results_df.to_csv(results_path, index=False)

    group_sizes = (
        data.groupby("activity_group", observed=False)
        .size()
        .rename("records")
        .reset_index()
    )

    group_sizes.to_csv(
        OUTPUT_DIR / "activity_group_sizes.csv",
        index=False,
    )

    # Plot model flag rates by activity group.
    if not results_df.empty:
        pivot = results_df.pivot(
            index="activity_group",
            columns="model",
            values="flagged_rate_percent",
        )

        ax = pivot.plot(
            kind="bar",
            figsize=(10, 6),
            rot=15,
        )

        ax.set_title(
            "Anomaly Flag Rates by Developer Activity Group"
        )
        ax.set_xlabel("Activity group (events per actor-week)")
        ax.set_ylabel("Records flagged (%)")
        ax.legend(title="Model")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "flag_rates_by_activity_group.png",
            dpi=200,
        )
        plt.close()

    summary = pd.DataFrame([{
        "matched_records_before_grouping": len(features.merge(
            scores[keys],
            on=keys,
            how="inner",
            validate="one_to_one",
        )),
        "records_analyzed": len(data),
        "records_excluded_from_groups": (
            len(features.merge(
                scores[keys],
                on=keys,
                how="inner",
                validate="one_to_one",
            )) - len(data)
        ),
        "activity_groups_present": data["activity_group"].nunique(),
        "unique_actors_analyzed": data["actor"].nunique(),
    }])

    summary.to_csv(
        OUTPUT_DIR / "analysis_summary.csv",
        index=False,
    )

    print("\nACTIVITY GROUP SIZES")
    print(group_sizes.to_string(index=False))

    print("\nANOMALY FLAG RATES")
    print(results_df.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"CSV: {results_path}")
    print(
        "Plot:",
        OUTPUT_DIR / "flag_rates_by_activity_group.png",
    )


if __name__ == "__main__":
    main()

