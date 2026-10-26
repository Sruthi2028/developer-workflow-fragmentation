
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "reports" / "real_data" / "top_100_extended_anomalies.csv"
OUTPUT = ROOT / "reports" / "final_dashboard" / "anomaly_analysis"

OUTPUT.mkdir(parents=True, exist_ok=True)


def save_plot(filename):
    path = OUTPUT / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path.relative_to(ROOT)}")


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT)

    required = {
        "actor",
        "baseline_anomaly_score",
        "extended_anomaly_score",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --------------------------------------------------
    # 1. SCORE SUMMARY
    # --------------------------------------------------

    score_columns = [
        "baseline_anomaly_score",
        "extended_anomaly_score",
    ]

    score_summary = df[score_columns].describe().T
    score_summary.to_csv(OUTPUT / "anomaly_score_summary.csv")

    print("\nANOMALY SCORE SUMMARY")
    print(score_summary.round(4))

    # --------------------------------------------------
    # 2. BOT-LIKE NAME INDICATOR
    # --------------------------------------------------

    # This is only a name-based heuristic, not verified
    # classification of GitHub accounts.
    bot_pattern = r"\[bot\]|bot$|dependabot|renovate"

    df["bot_like_name"] = (
        df["actor"]
        .astype(str)
        .str.contains(bot_pattern, case=False, regex=True, na=False)
    )

    bot_summary = (
        df.groupby("bot_like_name")
        .agg(
            records=("actor", "size"),
            baseline_mean_score=("baseline_anomaly_score", "mean"),
            extended_mean_score=("extended_anomaly_score", "mean"),
            baseline_median_score=("baseline_anomaly_score", "median"),
            extended_median_score=("extended_anomaly_score", "median"),
        )
        .reset_index()
    )

    bot_summary["account_name_group"] = bot_summary["bot_like_name"].map(
        {True: "Bot-like name", False: "Other account names"}
    )

    bot_summary.to_csv(
        OUTPUT / "bot_like_name_summary.csv",
        index=False,
    )

    print("\nBOT-LIKE NAME SUMMARY — TOP 100 ONLY")
    print(bot_summary.round(4).to_string(index=False))

    # Graph: number of records by name-based category.
    counts = df["bot_like_name"].value_counts().reindex(
        [False, True], fill_value=0
    )

    plt.figure(figsize=(7, 5))
    plt.bar(
        ["Other account names", "Bot-like names"],
        [counts[False], counts[True]],
    )
    plt.title("Account-Name Categories Among Top 100 Flagged Records")
    plt.xlabel("Name-based category")
    plt.ylabel("Records")
    save_plot("bot_like_name_counts.png")

    # --------------------------------------------------
    # 3. BASELINE VS EXTENDED SCORE COMPARISON
    # --------------------------------------------------

    plt.figure(figsize=(7, 6))
    plt.scatter(
        df["baseline_anomaly_score"],
        df["extended_anomaly_score"],
        alpha=0.75,
    )
    plt.title("Baseline vs. Extended Anomaly Scores")
    plt.xlabel("Baseline anomaly score")
    plt.ylabel("Extended anomaly score")
    save_plot("baseline_vs_extended_scores.png")

    score_correlation = df[score_columns].corr().iloc[0, 1]

    comparison = pd.DataFrame(
        [
            {
                "records_in_top_100_file": len(df),
                "unique_actors": df["actor"].nunique(),
                "bot_like_name_records": int(df["bot_like_name"].sum()),
                "other_name_records": int((~df["bot_like_name"]).sum()),
                "score_pearson_correlation": score_correlation,
                "mean_baseline_score": df["baseline_anomaly_score"].mean(),
                "mean_extended_score": df["extended_anomaly_score"].mean(),
            }
        ]
    )

    comparison.to_csv(
        OUTPUT / "anomaly_analysis_summary.csv",
        index=False,
    )

    # Save an annotated copy without changing the original input.
    df.to_csv(
        OUTPUT / "top_100_anomalies_with_name_indicator.csv",
        index=False,
    )

    print("\nOVERALL SUMMARY")
    print(comparison.round(4).to_string(index=False))

    print(f"\nAnalysis completed. Results saved to:\n{OUTPUT}")


if __name__ == "__main__":
    main()

