"""
Listening Session Partitioning & Aggregation Module

WHAT: Groups sequential timestamped listening events into discrete listening sessions
using a 30-minute inactivity-gap threshold rule.

WHY: Human music consumption occurs in temporal episodes (e.g. study sessions, workouts, morning commutes).
Treating every track play as an independent, uncorrelated observation ignores sequential continuity and behavioral context.

FIRST-PRINCIPLES INTUITION:
If a listener finishes a song and plays another 3 minutes later, they are inside the same listening state/session.
If they pause for 45 minutes and start music again, a new listening episode has begun.
"""

import pandas as pd
import numpy as np
from typing import Optional

SESSION_GAP_MINUTES = 30


def assign_listening_sessions(
    history_df: pd.DataFrame,
    gap_minutes: int = SESSION_GAP_MINUTES
) -> pd.DataFrame:
    """
    Partition chronological listening logs into discrete session IDs.

    WHAT: Iterates over sorted timestamps per user. Whenever consecutive plays are separated by
    more than `gap_minutes`, increments the session_id counter. Adds session positional features.

    WHY: Grouping tracks into sessions enables intra-session position tracking, session duration
    calculation, and sequential behavioral modeling.
    """
    df = history_df.sort_values(["user_id", "played_at"]).copy()

    if not pd.api.types.is_datetime64_any_dtype(df["played_at"]):
        df["played_at"] = pd.to_datetime(df["played_at"], utc=True)

    # Compute time difference between consecutive tracks in minutes
    df["time_diff_min"] = df.groupby("user_id")["played_at"].diff().dt.total_seconds() / 60.0

    # New session flag: True if first track or time_diff > gap_minutes
    df["is_new_session"] = (df["time_diff_min"].isna()) | (df["time_diff_min"] > gap_minutes)

    # Cumulatively sum new session flags per user to generate session_id
    df["session_number"] = df.groupby("user_id")["is_new_session"].cumsum()
    df["session_id"] = df["user_id"] + "_SESS_" + df["session_number"].astype(str).str.zfill(4)

    # Positional features inside each session
    df["session_position"] = df.groupby("session_id").cumcount() + 1
    df["session_start_time"] = df.groupby("session_id")["played_at"].transform("min")
    df["time_since_session_start_min"] = (df["played_at"] - df["session_start_time"]).dt.total_seconds() / 60.0

    # Clean up intermediate helper columns
    df = df.drop(columns=["is_new_session", "session_number", "time_diff_min"])

    return df


def compute_session_summary(
    session_history_df: pd.DataFrame,
    catalog_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate session-level metadata and average audio features.

    WHAT: Merges history logs with catalog track features and aggregates metrics per session.

    WHY: Session-level summaries represent user behavioral episodes, providing input for habit profiling.
    """
    if "session_id" not in session_history_df.columns:
        session_history_df = assign_listening_sessions(session_history_df)

    merged = session_history_df.merge(catalog_df, on="track_id", how="inner")

    feature_cols = ["energy", "valence", "danceability", "acousticness", "instrumentalness", "tempo"]
    available_features = [c for c in feature_cols if c in merged.columns]

    agg_dict = {
        "track_id": "count",
        "played_at": ["min", "max"],
    }
    for f in available_features:
        agg_dict[f] = "mean"

    session_summary = merged.groupby("session_id").agg(agg_dict)

    # Flatten multi-level columns
    session_summary.columns = [
        "track_count" if c[0] == "track_id" else
        "start_time" if c[1] == "min" else
        "end_time" if c[1] == "max" else
        f"avg_{c[0]}"
        for c in session_summary.columns
    ]

    session_summary["duration_minutes"] = (
        session_summary["end_time"] - session_summary["start_time"]
    ).dt.total_seconds() / 60.0

    return session_summary.reset_index()
