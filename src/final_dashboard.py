
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "final_dashboard"

OUTPUT.mkdir(parents=True, exist_ok=True)


def load_csv(relative_path):
    """Load an existing CSV and report missing files clearly."""
    path = REPORTS / relative_path

    if not path.exists():
        print(f"[SKIP] File not found: {path}")
        return None

    print(f"[LOAD] {path.relative_to(ROOT)}")
    return pd.read_csv(path)


def save_plot(filename):
    """Save a plot and close its figure."""
    destination = OUTPUT / filename
    plt.tight_layout()
    plt.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {destination.relative_to(ROOT)}")


def plot_model_comparison(models):
    """Compare anomaly counts and flagged percentages."""
    if models is None or models.empty:
        return

    required = {"model", "flagged_count", "flagged_rate_pct"}

    if not required.issubset(models.columns):
        print("[SKIP] Model comparison CSV lacks required columns.")
        return

    models = models.copy()

    plt.figure(figsize=(9, 5))
    plt.bar(models["model"], models["flagged_count"])
    plt.title("Anomaly Counts by Model")
    plt.xlabel("Model")
    plt.ylabel("Flagged records")
    plt.xticks(rotation=15)
    save_plot("model_anomaly_counts.png")

    plt.figure(figsize=(9, 5))
    plt.bar(models["model"], models["flagged_rate_pct"])
    plt.title("Anomaly Flag Rate by Model")
    plt.xlabel("Model")
    plt.ylabel("Flagged records (%)")
    plt.xticks(rotation=15)
    save_plot("model_flag_rates.png")


def plot_anomaly_score_distributions(scores):
    """Plot anomaly-score distributions when score columns exist."""
    if scores is None or scores.empty:
        return

    # Inspect columns rather than assuming a fixed naming convention.
    score_columns = [
        col for col in scores.columns
        if "score" in col.lower()
        and pd.api.types.is_numeric_dtype(scores[col])
    ]

    if not score_columns:
        print("[SKIP] No numeric anomaly-score columns found.")
        return

    for column in score_columns:
        values = scores[column].dropna()

        if values.empty:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=50, edgecolor="black")
        plt.title(f"Distribution of {column}")
        plt.xlabel("Anomaly score")
        plt.ylabel("Number of records")
        save_plot(f"score_distribution_{column}.png")


def plot_contamination_sensitivity(sensitivity):
    """Plot flagged counts across contamination settings."""
    if sensitivity is None or sensitivity.empty:
        return

    print("\nContamination sensitivity columns:")
    print(sensitivity.columns.tolist())

    parameter_col = next(
        (
            col for col in sensitivity.columns
            if "contamination" in col.lower()
        ),
        None,
    )

    count_col = next(
        (
            col for col in sensitivity.columns
            if col.lower() in {
                "flagged_count",
                "flagged_records",
                "anomaly_count",
            }
        ),
        None,
    )

    if parameter_col is None or count_col is None:
        print("[SKIP] Could not identify contamination/count columns.")
        return

    data = sensitivity.copy()
    data[parameter_col] = pd.to_numeric(
        data[parameter_col], errors="coerce"
    )
    data[count_col] = pd.to_numeric(
        data[count_col], errors="coerce"
    )
    data = data.dropna(subset=[parameter_col, count_col])

    if data.empty:
        return

    data = data.sort_values(parameter_col)

    plt.figure(figsize=(8, 5))
    plt.plot(
        data[parameter_col],
        data[count_col],
        marker="o",
    )
    plt.title("Contamination Setting vs. Flagged Records")
    plt.xlabel("Contamination setting")
    plt.ylabel("Flagged records")
    save_plot("contamination_sensitivity.png")


def plot_shrr_sensitivity(shrr):
    """Plot the proportion of records with nonzero SHRR."""
    if shrr is None or shrr.empty:
        return

    print("\nSHRR sensitivity columns:")
    print(shrr.columns.tolist())

    window_col = next(
        (
            col for col in shrr.columns
            if "window" in col.lower()
            and "minute" in col.lower()
        ),
        None,
    )

    nonzero_col = next(
        (
            col for col in shrr.columns
            if "nonzero" in col.lower()
        ),
        None,
    )

    if window_col is None or nonzero_col is None:
        print("[SKIP] Could not identify SHRR summary columns.")
        return

    data = shrr.copy()
    data[window_col] = pd.to_numeric(
        data[window_col], errors="coerce"
    )
    data[nonzero_col] = pd.to_numeric(
        data[nonzero_col], errors="coerce"
    )
    data = data.dropna(subset=[window_col, nonzero_col])

    if data.empty:
        return

    data = data.sort_values(window_col)

    plt.figure(figsize=(8, 5))
    plt.plot(
        data[window_col],
        data[nonzero_col],
        marker="o",
    )
    plt.title("SHRR Window Sensitivity")
    plt.xlabel("Re-entry window (minutes)")
    plt.ylabel("Records with nonzero SHRR")
    plt.xticks(data[window_col].tolist())
    save_plot("shrr_window_sensitivity.png")



def plot_feature_correlations(correlations):
    """Plot the feature correlation matrix from a CSV."""
    if correlations is None or correlations.empty:
        print("[SKIP] Correlation data is empty.")
        return

    # The first CSV column contains feature names.
    matrix = correlations.copy()

    first_column = matrix.columns[0]
    matrix = matrix.set_index(first_column)

    # Convert all matrix values to numeric.
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    # Remove rows and columns that contain no correlation values.
    matrix = matrix.dropna(axis=0, how="all")
    matrix = matrix.dropna(axis=1, how="all")

    if matrix.empty:
        print("[SKIP] No valid correlation values found.")
        return

    # Generate the correlation heatmap.
    plt.figure(figsize=(11, 8))

    image = plt.imshow(
        matrix.to_numpy(dtype=float),
        cmap="coolwarm",
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(image, label="Pearson correlation")

    plt.xticks(
        range(len(matrix.columns)),
        matrix.columns,
        rotation=90,
    )

    plt.yticks(
        range(len(matrix.index)),
        matrix.index,
    )

    # Display the correlation values inside the cells.
    for row in range(len(matrix.index)):
        for col in range(len(matrix.columns)):
            value = matrix.iloc[row, col]

            if pd.notna(value):
                plt.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    plt.title("Feature Correlation Matrix")
    plt.xlabel("Features")
    plt.ylabel("Features")

    save_plot("feature_correlation_matrix.png")


def create_metrics_summary(*datasets):
    """Combine compatible summary tables without inventing metrics."""
    valid = [df.copy() for df in datasets if df is not None and not df.empty]

    if not valid:
        print("[SKIP] No summary tables available.")
        return

    # Preserve each source table in a separate section of one CSV.
    output_file = OUTPUT / "consolidated_metrics.csv"

    with output_file.open("w", encoding="utf-8", newline="") as file:
        for index, frame in enumerate(valid, start=1):
            file.write(f"# Source table {index}\n")
            frame.to_csv(file, index=False)
            file.write("\n")

    print(f"[SAVED] {output_file.relative_to(ROOT)}")


def main():
    print("=" * 60)
    print("FINAL DASHBOARD: REAL GITHUB ACTIVITY ANALYSIS")
    print("=" * 60)
    print(f"Output directory: {OUTPUT}")

    model_comparison = load_csv(
        "real_data_comparison/model_comparison_summary.csv"
    )
    real_model_summary = load_csv(
        "real_data/model_summary.csv"
    )
    anomaly_scores = load_csv(
        "real_data/anomaly_scores.csv"
    )
    contamination = load_csv(
        "sensitivity_analysis/contamination_summary.csv"
    )
    shrr = load_csv(
        "shrr_sensitivity/shrr_window_summary.csv"
    )
    correlations = load_csv(
        "eda/feature_correlations.csv"
    )
    dataset_summary = load_csv(
        "eda/dataset_summary.csv"
    )
    overlap = load_csv(
        "real_data_comparison/flag_overlap_comparison.csv"
    )

    plot_model_comparison(model_comparison)
    plot_anomaly_score_distributions(anomaly_scores)
    plot_contamination_sensitivity(contamination)
    plot_shrr_sensitivity(shrr)
    plot_feature_correlations(correlations)

    create_metrics_summary(
        model_comparison,
        real_model_summary,
        contamination,
        shrr,
        dataset_summary,
        overlap,
    )

    print("\nDashboard generation finished.")
    print("Review any [SKIP] messages to identify missing or differently")
    print("formatted input files.")
    print("All outputs are saved under reports/final_dashboard/.")


if __name__ == "__main__":
    main()

