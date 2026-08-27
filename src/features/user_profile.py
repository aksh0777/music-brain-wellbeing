"""
User Music Intelligence Profile Construction Module

WHAT: Synthesizes track-level listening events and session-level behavioral data into a quantitative
user-level music profile.

WHY: Downstream recommendation systems and AI explanation models require user-level preference aggregations
rather than raw, unaggregated event logs.

TRANSFORMATION HIERARCHY:
    Track-Level Listening Events
               ↓
    Session-Level Behavioral Episodes
               ↓
    Quantitative User Music Profile Vector
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from .sessions import assign_listening_sessions
from .temporal import generate_temporal_features


def build_user_music_profile(
    history_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    cluster_labels: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Construct a quantitative user music habit profile.

    WHAT: Merges history with track catalog features, computes mean and standard deviation of audio
    descriptors, calculates session duration averages, peak listening hours, and acoustic cluster shares.

    WHY: Synthesizes granular listening history into a structured user feature dictionary for recommendation engines.
    """
    df_hist = history_df.copy()

    # Ensure sessions and temporal features exist
    if "session_id" not in df_hist.columns:
        df_hist = assign_listening_sessions(df_hist)

    if "hour" not in df_hist.columns:
        df_hist = generate_temporal_features(df_hist)

    catalog_target = catalog_df.copy()
    if cluster_labels is not None and len(cluster_labels) == len(catalog_target):
        catalog_target["cluster_id"] = cluster_labels

    # Prevent suffix collision (_x, _y) if df_hist already contains catalog columns
    overlapping = [c for c in df_hist.columns if c in catalog_target.columns and c != "track_id"]
    df_hist_clean = df_hist.drop(columns=overlapping) if overlapping else df_hist

    merged = df_hist_clean.merge(catalog_target, on="track_id", how="inner")

    user_id = df_hist["user_id"].iloc[0] if "user_id" in df_hist.columns and not df_hist.empty else "UNKNOWN"
    total_tracks = len(merged)

    # Session metrics
    session_counts = merged.groupby("session_id")["track_id"].count()
    total_sessions = len(session_counts)
    avg_session_length = float(session_counts.mean()) if total_sessions > 0 else 0.0

    # Session durations
    session_times = merged.groupby("session_id")["played_at"].agg(["min", "max"])
    session_durations = (session_times["max"] - session_times["min"]).dt.total_seconds() / 60.0
    avg_session_duration = float(session_durations.mean()) if total_sessions > 0 else 0.0

    # Audio feature aggregations (Mean & Std)
    audio_features = ["energy", "valence", "danceability", "acousticness", "instrumentalness", "tempo"]
    audio_profile = {}

    for f in audio_features:
        if f in merged.columns:
            audio_profile[f"{f}_mean"] = round(float(merged[f].mean()), 4)
            audio_profile[f"{f}_std"] = round(float(merged[f].std()), 4) if total_tracks > 1 else 0.0

    # Cluster distribution
    cluster_dist = {}
    if "cluster_id" in merged.columns:
        counts = merged["cluster_id"].value_counts(normalize=True)
        for cluster_id, proportion in counts.items():
            cluster_dist[f"cluster_{cluster_id}"] = round(float(proportion), 4)

    # Peak listening hours & time of day distribution
    # Night (0-5), Morning (6-11), Afternoon (12-17), Evening (18-23)
    def categorize_hour(h):
        if 0 <= h <= 5:
            return "Night"
        elif 6 <= h <= 11:
            return "Morning"
        elif 12 <= h <= 17:
            return "Afternoon"
        else:
            return "Evening"

    time_of_day = merged["hour"].apply(categorize_hour).value_counts(normalize=True)
    time_of_day_dist = {tod: round(float(prop), 4) for tod, prop in time_of_day.items()}

    # Weekend listening ratio
    weekend_ratio = float((merged["day_of_week"] >= 5).mean()) if "day_of_week" in merged.columns else 0.0

    # Synthesize complete user profile dictionary
    profile = {
        "user_id": user_id,
        "total_tracks_listened": total_tracks,
        "total_sessions": total_sessions,
        "avg_tracks_per_session": round(avg_session_length, 2),
        "avg_session_duration_minutes": round(avg_session_duration, 2),
        "audio_feature_summary": audio_profile,
        "cluster_distribution": cluster_dist,
        "time_of_day_distribution": time_of_day_dist,
        "weekend_listening_ratio": round(weekend_ratio, 4),
        "data_provenance": "synthetic/demo" if "data_type" in merged.columns and (merged["data_type"] == "synthetic/demo").any() else "real"
    }

    return profile
