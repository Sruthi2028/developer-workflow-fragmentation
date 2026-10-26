
"""Evaluate SHRR feature sensitivity across multiple re-entry windows."""

import sys
from pathlib import Path

import pandas as pd

# Allow importing the existing feature-generation function from src.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gharchive_pipeline import actor_week_features


ROOT = Path(__file__).resolve().parent.parent
EVENTS_FILE = ROOT / "data" / "processed" / "clean_events.csv"
OUTPUT_DIR = ROOT / "reports" / "shrr_sensitivity"
WINDOWS = [15, 30, 60, 120]


def main() -> None:
    if not EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Clean event data not found: {EVENTS_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading events from: {EVENTS_FILE}")

    events = pd.read_csv(
        EVENTS_FILE,
        parse_dates=["created_at"],
    )

    required = {"actor", "repo", "created_at", "event_type"}
    missing = required - set(events.columns)

    if missing:
        raise ValueError(
            f"Required columns are missing: {sorted(missing)}"
        )

    print(f"Loaded {len(events):,} cleaned events.")
    print(f"Evaluating windows: {WINDOWS} minutes")

    summaries = []
    all_features = []

    for window in WINDOWS:
        print(f"\nCalculating features for {window}-minute window...")

        features = actor_week_features(
            events,
            reentry_minutes=window,
        )

        if features.empty:
            print("No actor-week features generated.")
            continue

        features["reentry_window_minutes"] = window

        # Save each window's features separately.
        feature_path = (
            OUTPUT_DIR / f"actor_week_features_{window}min.csv"
        )
        features.to_csv(feature_path, index=False)

        shrr = features["short_horizon_reentry_rate"]

        summaries.append(
            {
                "window_minutes": window,
                "actor_week_records": len(features),
                "records_with_nonzero_shrr": int((shrr > 0).sum()),
                "nonzero_shrr_rate_pct": float((shrr > 0).mean() * 100),
                "mean_shrr": float(shrr.mean()),
                "median_shrr": float(shrr.median()),
                "shrr_95th_percentile": float(shrr.quantile(0.95)),
                "mean_repository_transitions": float(
                    features["repository_transitions"].mean()
                ),
                "mean_transition_intensity": float(
                    features["transitions_per_observed_span_hour"].mean()
                ),
            }
        )

        selected = features[
            [
                "actor",
                "week_start",
                "event_count",
                "repository_transitions",
                "short_horizon_reentry_rate",
            ]
        ].copy()

        selected["window_minutes"] = window
        all_features.append(selected)

        print(f"Actor-week records: {len(features):,}")
        print(f"Mean SHRR: {shrr.mean():.6f}")
        print(f"Median SHRR: {shrr.median():.6f}")
        print(f"Records with nonzero SHRR: {(shrr > 0).sum():,}")

    if not summaries:
        raise RuntimeError("No results were generated.")

    summary = pd.DataFrame(summaries)
    summary_path = OUTPUT_DIR / "shrr_window_summary.csv"
    summary.to_csv(summary_path, index=False)

    comparison = pd.concat(all_features, ignore_index=True)
    comparison_path = OUTPUT_DIR / "shrr_window_actor_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\nSHRR WINDOW SENSITIVITY SUMMARY")
    print(summary.to_string(index=False))

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved actor-level comparison: {comparison_path}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()



