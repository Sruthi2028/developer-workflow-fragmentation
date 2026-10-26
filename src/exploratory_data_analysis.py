
"""Exploratory data analysis for GitHub workflow fragmentation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "reports" / "eda"

EVENTS_FILE = DATA_DIR / "clean_events.csv"
FEATURES_FILE = DATA_DIR / "actor_week_features.csv"
FILTERED_FILE = DATA_DIR / "actor_week_features_filtered.csv"


def save_plot(fig, filename):
    """Save a plot and close its figure."""
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [EVENTS_FILE, FEATURES_FILE, FILTERED_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found: {path}")

    print("Loading official GitHub Archive datasets...")

    events = pd.read_csv(EVENTS_FILE)
    features = pd.read_csv(FEATURES_FILE)
    filtered = pd.read_csv(FILTERED_FILE)

    print(f"Clean events: {len(events):,}")
    print(f"All actor-week records: {len(features):,}")
    print(f"Filtered actor-week records: {len(filtered):,}")

    # 1. Dataset summary
    summary = pd.DataFrame(
        {
            "metric": [
                "Clean event records",
                "Distinct actors in clean events",
                "Distinct repositories in clean events",
                "All actor-week records",
                "Filtered actor-week records",
                "Distinct actors in filtered data",
                "Missing values in all actor-week features",
                "Missing values in filtered features",
            ],
            "value": [
                len(events),
                events["actor"].nunique(),
                events["repo"].nunique(),
                len(features),
                len(filtered),
                filtered["actor"].nunique(),
                int(features.isna().sum().sum()),
                int(filtered.isna().sum().sum()),
            ],
        }
    )

    summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    print("\nDataset summary:")
    print(summary.to_string(index=False))

    # 2. Distribution of event counts
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(features["event_count"], bins=50, edgecolor="black")
    ax.set_title("Distribution of Events per Actor-Week")
    ax.set_xlabel("Number of Events")
    ax.set_ylabel("Number of Actor-Week Records")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, "event_count_distribution.png")

    # 3. Distinct repositories
    fig, ax = plt.subplots(figsize=(9, 5))
    repository_counts = features["distinct_repositories"].value_counts().sort_index()
    ax.bar(
        repository_counts.index,
        repository_counts.values,
        edgecolor="black",
    )
    ax.set_title("Distinct Repositories per Actor-Week")
    ax.set_xlabel("Number of Distinct Repositories")
    ax.set_ylabel("Number of Actor-Week Records")
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, "repository_distribution.png")

    # 4. Repository transitions versus event count
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        features["event_count"],
        features["repository_transitions"],
        alpha=0.25,
        s=10,
    )
    ax.set_title("Repository Transitions versus Event Count")
    ax.set_xlabel("Number of Events")
    ax.set_ylabel("Number of Repository Transitions")
    ax.grid(alpha=0.25)
    save_plot(fig, "transitions_vs_events.png")

    # 5. SHRR distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        features["short_horizon_reentry_rate"],
        bins=40,
        edgecolor="black",
    )
    ax.set_title("Distribution of Short-Horizon Re-entry Rate")
    ax.set_xlabel("SHRR (60-minute window)")
    ax.set_ylabel("Number of Actor-Week Records")
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, "shrr_distribution.png")

    # 6. Feature correlation matrix
    correlation_columns = [
        "event_count",
        "distinct_repositories",
        "normalized_repo_entropy",
        "repository_transitions",
        "transitions_per_observed_span_hour",
        "short_horizon_reentry_rate",
        "active_utc_dates",
    ]

    correlations = features[correlation_columns].corr()
    correlations.to_csv(OUTPUT_DIR / "feature_correlations.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(range(len(correlation_columns)))
    ax.set_yticks(range(len(correlation_columns)))
    ax.set_xticklabels(
        correlation_columns, rotation=45, ha="right", fontsize=8
    )
    ax.set_yticklabels(correlation_columns, fontsize=8)
    ax.set_title("Correlation Between Workflow Features")

    for row in range(len(correlation_columns)):
        for col in range(len(correlation_columns)):
            ax.text(
                col,
                row,
                f"{correlations.iloc[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    fig.colorbar(image, ax=ax, label="Pearson Correlation")
    save_plot(fig, "feature_correlation_matrix.png")

    print("\nEDA completed successfully.")
    print(f"All EDA outputs are saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
