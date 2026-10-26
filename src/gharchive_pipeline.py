"""Download public GitHub activity archives and build actor-week workflow features.

This analyzes observable public GitHub events only. It cannot measure private work,
local branch switches, all task changes, cognitive load, productivity, or burnout.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE_URL = "https://gharchive.open-digger.cn"
RELEVANT_TYPES = {
    "PushEvent", "PullRequestEvent", "PullRequestReviewEvent",
    "IssuesEvent", "IssueCommentEvent", "CreateEvent",
}
EVENT_COLUMNS = ["actor", "repo", "created_at", "event_type", "source_file"]


def download_archives(start: date, end: date, out_dir: Path,
                      hours: list[int] | None = None) -> None:
    """Download hourly JSONL gzip archives, skipping non-empty existing files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hours = hours if hours is not None else list(range(24))
    current = start
    while current <= end:
        for hour in hours:
            if not 0 <= hour <= 23:
                raise ValueError(f"Hour must be between 0 and 23, got {hour}")
            name = f"{current.isoformat()}-{hour}.json.gz"
            target = out_dir / name
            if target.exists() and target.stat().st_size > 0:
                print(f"Exists: {name}", flush=True)
                continue
            url = f"{BASE_URL}/{name}"
            try:
                request = Request(url, headers={"User-Agent": "ML-Theory-Student-Project/1.0"})
                with urlopen(request, timeout=60) as response, target.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                print(f"Downloaded: {name} ({target.stat().st_size:,} bytes)", flush=True)
            except Exception as exc:
                print(f"WARNING: could not download {name}: {exc}", flush=True)
                target.unlink(missing_ok=True)
            time.sleep(0.15)
        current += timedelta(days=1)


def parse_archives(raw_dir: Path) -> pd.DataFrame:
    """Stream archive contents into CSV in chunks to reduce memory use."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Archive directory not found: {raw_dir.resolve()}")
    files = sorted(raw_dir.glob("*.json.gz"))
    if not files:
        print(f"No .json.gz files found in {raw_dir.resolve()}", flush=True)
        return pd.DataFrame(columns=EVENT_COLUMNS)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    temp_path = processed_dir / "clean_events_temp.csv"
    temp_path.unlink(missing_ok=True)
    chunk_size = 50_000
    rows: list[dict] = []
    total_written = 0

    for file_number, archive_path in enumerate(files, start=1):
        print(f"[{file_number}/{len(files)}] Reading {archive_path.name}", flush=True)
        count_in_file = 0
        try:
            with gzip.open(archive_path, "rt", encoding="utf-8") as archive:
                for line in archive:
                    try:
                        event = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    event_type = event.get("type")
                    if event_type not in RELEVANT_TYPES:
                        continue
                    actor = (event.get("actor") or {}).get("login")
                    repo = (event.get("repo") or {}).get("name")
                    timestamp = event.get("created_at")
                    if not actor or not repo or not timestamp:
                        continue
                    rows.append({
                        "actor": str(actor),
                        "repo": str(repo),
                        "created_at": timestamp,
                        "event_type": str(event_type),
                        "source_file": archive_path.name,
                    })
                    count_in_file += 1

                    if len(rows) >= chunk_size:
                        _append_chunk(rows, temp_path)
                        total_written += len(rows)
                        rows.clear()
                        if total_written and total_written % 250_000 == 0:
                            print(f"  Parsed about {total_written:,} events so far", flush=True)
        except (OSError, EOFError) as exc:
            print(f"WARNING: skipping unreadable archive {archive_path.name}: {exc}", flush=True)
        print(f"  Relevant events found in this archive: {count_in_file:,}", flush=True)

    if rows:
        _append_chunk(rows, temp_path)
        total_written += len(rows)
        rows.clear()

    if not temp_path.exists() or temp_path.stat().st_size == 0:
        print("No usable events found. Check archive files and date range.", flush=True)
        return pd.DataFrame(columns=EVENT_COLUMNS)

    print(f"Loading {total_written:,} extracted events for deduplication ...", flush=True)
    events = pd.read_csv(temp_path, dtype={"actor": "string", "repo": "string",
                                           "event_type": "string", "source_file": "string"})
    events["created_at"] = pd.to_datetime(events["created_at"], utc=True, errors="coerce")
    events = events.dropna(subset=["created_at"])
    events = events.drop_duplicates(subset=["actor", "repo", "created_at", "event_type"])
    events = events.sort_values(["actor", "created_at"]).reset_index(drop=True)
    temp_path.unlink(missing_ok=True)
    print(f"Finished parsing: {len(events):,} unique events", flush=True)
    return events


def _append_chunk(rows: list[dict], output_path: Path) -> None:
    chunk = pd.DataFrame.from_records(rows, columns=EVENT_COLUMNS)
    # Batch conversion is much faster than calling pandas once per event.
    chunk["created_at"] = pd.to_datetime(chunk["created_at"], utc=True, errors="coerce")
    chunk = chunk.dropna(subset=["created_at"])
    if chunk.empty:
        return
    chunk.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)


def _entropy(repos: pd.Series) -> float:
    counts = repos.value_counts().to_numpy(dtype=float)
    if len(counts) <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum() / np.log2(len(counts)))


def actor_week_features(events: pd.DataFrame, reentry_minutes: int = 60) -> pd.DataFrame:
    """Create actor-week features and an exploratory Workflow Fragmentation Score."""
    if events.empty:
        return pd.DataFrame()
    x = events.copy()
    x["created_at"] = pd.to_datetime(x["created_at"], utc=True, errors="coerce")
    x = x.dropna(subset=["created_at"])
    x["week_start"] = (
        x["created_at"].dt.tz_convert(None).dt.to_period("W-SUN")
        .dt.start_time.dt.date.astype(str)
    )

    feature_rows = []
    for (actor, week), group in x.groupby(["actor", "week_start"], sort=False):
        group = group.sort_values("created_at").reset_index(drop=True)
        repos = group["repo"].astype(str)
        transitions = 0
        reentries = 0
        previous_repo = None
        last_seen: dict[str, object] = {}
        first_timestamp = group["created_at"].iloc[0]
        last_timestamp = group["created_at"].iloc[-1]

        for row in group.itertuples(index=False):
            repo = str(row.repo)
            timestamp = row.created_at.to_pydatetime()
            if previous_repo is not None and repo != previous_repo:
                transitions += 1
                prior_timestamp = last_seen.get(repo)
                if prior_timestamp is not None:
                    gap = (timestamp - prior_timestamp).total_seconds() / 60
                    if 0 <= gap <= reentry_minutes:
                        reentries += 1
            last_seen[repo] = timestamp
            previous_repo = repo

        span_hours = max((last_timestamp - first_timestamp).total_seconds() / 3600, 1.0)
        feature_rows.append({
            "actor": actor,
            "week_start": week,
            "event_count": len(group),
            "distinct_repositories": int(repos.nunique()),
            "normalized_repo_entropy": _entropy(repos),
            "repository_transitions": int(transitions),
            "transitions_per_observed_span_hour": float(transitions / span_hours),
            "short_horizon_reentry_rate": float(reentries / transitions) if transitions else 0.0,
            "active_utc_dates": int(group["created_at"].dt.date.nunique()),
            "reentry_window_minutes": int(reentry_minutes),
        })

    features = pd.DataFrame(feature_rows)
    score_columns = ["normalized_repo_entropy",
                     "transitions_per_observed_span_hour",
                     "short_horizon_reentry_rate"]
    for column in score_columns:
        standard_deviation = features[column].std(ddof=0)
        features[column + "_z"] = (
            (features[column] - features[column].mean()) / standard_deviation
            if standard_deviation and np.isfinite(standard_deviation) else 0.0
        )
    # Descriptive exploratory score, not a validated productivity or health measure.
    features["exploratory_wfs"] = features[[c + "_z" for c in score_columns]].mean(axis=1)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Download archives or process local GitHub event archives.")
    parser.add_argument("--start", help="Start date YYYY-MM-DD inclusive")
    parser.add_argument("--end", help="End date YYYY-MM-DD inclusive")
    parser.add_argument("--hours", help="Optional comma-separated UTC hours, e.g. 0,6,12,18")
    parser.add_argument("--out", default="data/raw", help="Archive output directory")
    parser.add_argument("--features", help="Directory of downloaded .json.gz archives to process")
    parser.add_argument("--window", type=int, default=60, help="SHRR time window in minutes")
    args = parser.parse_args()

    if args.features:
        events = parse_archives(Path(args.features))
        if events.empty:
            return
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        clean_path = processed_dir / "clean_events.csv"
        feature_path = processed_dir / "actor_week_features.csv"
        events.to_csv(clean_path, index=False)
        print(f"Saved cleaned events: {clean_path.resolve()} ({len(events):,} rows)", flush=True)
        features = actor_week_features(events, reentry_minutes=args.window)
        features.to_csv(feature_path, index=False)
        print(f"Saved actor-week features: {feature_path.resolve()} ({len(features):,} rows)", flush=True)
        return

    if not args.start or not args.end:
        parser.error("Provide --start and --end to download, or --features to process local archives.")
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    if end_date < start_date:
        parser.error("--end must be on or after --start")
    hours = [int(value.strip()) for value in args.hours.split(",")] if args.hours else None
    download_archives(start_date, end_date, Path(args.out), hours)


if __name__ == "__main__":
    main()
