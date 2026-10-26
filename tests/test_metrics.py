"""Small deterministic sanity checks; these do not validate burnout or real work context."""
import pandas as pd
from src.gharchive_pipeline import actor_week_features

def test_single_repository_has_zero_entropy_and_transitions():
    df = pd.DataFrame({
        "actor": ["dev"] * 3,
        "repo": ["A/repo"] * 3,
        "created_at": pd.to_datetime([
            "2026-10-20T09:00:00Z", "2026-10-20T09:10:00Z", "2026-10-20T09:20:00Z"
        ], utc=True),
        "event_type": ["PushEvent"] * 3,
        "source_file": ["test"] * 3
    })
    result = actor_week_features(df)
    assert result.iloc[0]["normalized_repo_entropy"] == 0
    assert result.iloc[0]["repository_transitions"] == 0
    assert result.iloc[0]["short_horizon_reentry_rate"] == 0

def test_a_b_a_sequence_has_one_short_horizon_reentry():
    df = pd.DataFrame({
        "actor": ["dev"] * 3,
        "repo": ["A/repo", "B/repo", "A/repo"],
        "created_at": pd.to_datetime([
            "2026-10-20T09:00:00Z", "2026-10-20T09:10:00Z", "2026-10-20T09:20:00Z"
        ], utc=True),
        "event_type": ["PushEvent"] * 3,
        "source_file": ["test"] * 3
    })
    result = actor_week_features(df, reentry_minutes=60)
    assert result.iloc[0]["repository_transitions"] == 2
    assert result.iloc[0]["short_horizon_reentry_rate"] == 0.5
