"""Controlled synthetic sanity check for workflow-fragmentation features.

Important: synthetic labels are constructed from event patterns. Results show whether
features distinguish those constructed patterns; they do not validate burnout,
cognitive load, or real developer context switching.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def make_dataset(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = int(i % 2 == 1)  # balanced constructed classes
        n_events = int(rng.integers(20, 61))
        if label == 0:
            # Lower-fragmentation patterns: one or a few repository blocks.
            k = int(rng.integers(1, 4))
            repos = [f"repo_{j}" for j in range(k)]
            seq = []
            for repo in repos:
                seq.extend([repo] * max(1, n_events // k))
            seq = (seq + [repos[-1]] * n_events)[:n_events]
            rng.shuffle(seq) if False else None
        else:
            # Higher-fragmentation patterns: repeated short A/B/C alternation.
            repos = [f"repo_{j}" for j in range(int(rng.integers(2, 5)))]
            seq = [repos[j % len(repos)] for j in range(n_events)]
            # Occasional runs make the pattern less perfectly regular.
            for _ in range(int(n_events * 0.12)):
                pos = int(rng.integers(0, n_events))
                seq[pos] = repos[int(rng.integers(0, len(repos)))]
        # Event times spaced 5–25 minutes apart to make the window semantics explicit.
        gaps = rng.integers(5, 26, size=n_events)
        times = pd.to_datetime("2026-10-05T08:00:00Z") + pd.to_timedelta(np.cumsum(gaps), unit="m")
        for j, (repo, ts) in enumerate(zip(seq[:n_events], times)):
            rows.append({
                "actor": f"synthetic_{i}",
                "repo": repo,
                "created_at": ts,
                "event_type": "PushEvent",
                "source_file": "synthetic",
                "synthetic_label": label,
            })
    return pd.DataFrame(rows)

def extract_actor_features(events, window=60):
    # Ensure the project root is importable when this file is run as a script.
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Local implementation keeps synthetic labels separate from feature construction.
    from src.gharchive_pipeline import actor_week_features
    f = actor_week_features(events.drop(columns=["synthetic_label"]), reentry_minutes=window)
    labels = events.groupby("actor")["synthetic_label"].first()
    f["synthetic_label"] = f["actor"].map(labels)
    return f

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/synthetic_benchmark.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=1200)
    args = p.parse_args()
    events = make_dataset(args.n, args.seed)
    features = extract_actor_features(events)
    base_cols = ["event_count", "distinct_repositories", "normalized_repo_entropy"]
    extended_cols = base_cols + [
        "repository_transitions", "transitions_per_observed_span_hour",
        "short_horizon_reentry_rate"
    ]
    y = features["synthetic_label"].astype(int)
    train_idx, test_idx = train_test_split(
        np.arange(len(features)), test_size=0.25, random_state=args.seed, stratify=y
    )
    results = []
    for name, cols in [("Baseline counts+entropy", base_cols), ("Extended temporal features", extended_cols)]:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=args.seed)
        )
        model.fit(features.iloc[train_idx][cols], y.iloc[train_idx])
        pred = model.predict(features.iloc[test_idx][cols])
        proba = model.predict_proba(features.iloc[test_idx][cols])[:, 1]
        results.append({
            "experiment": name,
            "n_synthetic_actor_weeks": len(features),
            "test_size": len(test_idx),
            "accuracy": accuracy_score(y.iloc[test_idx], pred),
            "precision": precision_score(y.iloc[test_idx], pred, zero_division=0),
            "recall": recall_score(y.iloc[test_idx], pred, zero_division=0),
            "f1": f1_score(y.iloc[test_idx], pred, zero_division=0),
            "roc_auc": roc_auc_score(y.iloc[test_idx], proba),
            "seed": args.seed,
            "label_note": "Constructed synthetic pattern labels; not real developer labels"
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nSaved: {out}")
    print("\nCAUTION: This benchmark only tests synthetic patterns by construction.")
if __name__ == "__main__":
    main()
