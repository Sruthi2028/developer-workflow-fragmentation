"""Train and compare Isolation Forest models on real processed GitHub features.

Important:
- This is unsupervised anomaly detection, not a classifier with ground-truth labels.
- Anomaly scores do not establish burnout, productivity, or cognitive load.
- The comparison measures how feature choices change model outputs; it does not prove
  that one model is more accurate without independent, validated labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

BASIC_FEATURES = [
    "event_count",
    "distinct_repositories",
    "normalized_repo_entropy",
]
EXTENDED_FEATURES = BASIC_FEATURES + [
    "repository_transitions",
    "transitions_per_observed_span_hour",
    "short_horizon_reentry_rate",
]
REQUIRED_COLUMNS = ["actor", "week_start"] + EXTENDED_FEATURES


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path.resolve()}\n"
            "Run the data-processing pipeline first."
        )
    data = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=REQUIRED_COLUMNS).copy()
    # Activity counts are extremely right-skewed. log1p reduces the influence
    # of very large accounts without deleting them.
    for column in [
        "event_count",
        "distinct_repositories",
        "repository_transitions",
        "transitions_per_observed_span_hour",
    ]:
        data[column] = np.log1p(data[column].clip(lower=0))
    data = data.reset_index(drop=True)
    if len(data) < 100:
        raise ValueError(f"Only {len(data)} valid records; too few for this experiment.")
    return data


def make_matrix(data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    matrix = data[columns].astype(float).to_numpy()
    scaler = RobustScaler()
    return scaler.fit_transform(matrix)


def fit_model(matrix: np.ndarray, contamination: float, seed: int) -> tuple[IsolationForest, np.ndarray, np.ndarray]:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(matrix)
    # sklearn decision_function: lower values are more anomalous.
    anomaly_score = -model.decision_function(matrix)
    prediction = model.predict(matrix)  # -1 = model-flagged anomaly, 1 = inlier
    return model, anomaly_score, prediction


def evaluate_flags(data: pd.DataFrame, predictions: np.ndarray, scores: np.ndarray) -> dict:
    flagged = predictions == -1
    return {
        "records": int(len(data)),
        "flagged_count": int(flagged.sum()),
        "flagged_fraction": float(flagged.mean()),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_95th_percentile": float(np.quantile(scores, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/actor_week_features_filtered.csv",
        help="Filtered actor-week feature CSV",
    )
    parser.add_argument("--out", default="reports/real_data", help="Output directory")
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Expected fraction flagged as anomalies; default 0.05")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.contamination < 0.5:
        parser.error("--contamination must be between 0 and 0.5")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(Path(args.input))
    print(f"Loaded {len(data):,} actor-week records from {args.input}", flush=True)
    print(f"Unique actors: {data['actor'].nunique():,}", flush=True)

    all_results = []
    model_outputs = data[["actor", "week_start"]].copy()

    experiments = [
        ("baseline", BASIC_FEATURES),
        ("extended", EXTENDED_FEATURES),
    ]

    for name, columns in experiments:
        print(f"\nTraining {name} Isolation Forest with {len(columns)} features ...", flush=True)
        matrix = make_matrix(data, columns)
        model, scores, predictions = fit_model(matrix, args.contamination, args.seed)
        model_outputs[f"{name}_anomaly_score"] = scores
        model_outputs[f"{name}_flag"] = np.where(predictions == -1, "anomaly", "inlier")
        metrics = evaluate_flags(data, predictions, scores)
        metrics.update({
            "model": f"{name}_isolation_forest",
            "features": ", ".join(columns),
            "contamination_parameter": args.contamination,
        })
        all_results.append(metrics)
        print(
            f"{name}: flagged {metrics['flagged_count']:,} "
            f"({metrics['flagged_fraction']:.2%})",
            flush=True,
        )

    scores_path = output / "anomaly_scores.csv"
    results_path = output / "model_summary.csv"
    model_outputs.to_csv(scores_path, index=False)
    pd.DataFrame(all_results).to_csv(results_path, index=False)

    # Compare score distributions; this is descriptive, not a measure of accuracy.
    plt.figure(figsize=(9, 5))
    plt.hist(model_outputs["baseline_anomaly_score"], bins=60, alpha=0.65,
             label="Baseline features")
    plt.hist(model_outputs["extended_anomaly_score"], bins=60, alpha=0.55,
             label="Extended temporal features")
    plt.xlabel("Anomaly score (higher = more unusual according to model)")
    plt.ylabel("Actor-week record count")
    plt.title("Isolation Forest score distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "score_distributions.png", dpi=160)
    plt.close()

    # Show the highest-scoring records for inspection, not as confirmed problem cases.
    top = model_outputs.sort_values("extended_anomaly_score", ascending=False).head(100)
    top.to_csv(output / "top_100_extended_anomalies.csv", index=False)

    print("\nCompleted. Files saved:", flush=True)
    for filename in [
        "model_summary.csv",
        "anomaly_scores.csv",
        "score_distributions.png",
        "top_100_extended_anomalies.csv",
    ]:
        print(f"  {(output / filename).resolve()}", flush=True)
    print(
        "\nInterpretation: these are unsupervised anomaly scores. No ground-truth labels "
        "are available, so precision/recall/ROC-AUC and claims of predictive accuracy "
        "cannot be calculated honestly from this dataset alone.",
        flush=True,
    )


if __name__ == "__main__":
    main()
