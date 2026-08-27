"""
Music Loader Module

WHAT: Ingestion and validation helper functions for loading Spotify track catalogs
and timestamped user listening history files.

WHY: Raw CSV datasets can contain missing values, invalid schemas, duplicate track IDs,
or unparsed timestamp strings. Validating and cleaning data at ingestion prevents silent downstream
errors in feature scaling, sessionization, and clustering.
"""

import os
import pandas as pd
from typing import List, Optional

REQUIRED_CATALOG_COLUMNS = [
    "track_id",
    "track_name",
    "artist_name",
    "genre",
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness"
]

REQUIRED_HISTORY_COLUMNS = [
    "event_id",
    "user_id",
    "track_id",
    "played_at",
    "data_type"
]


def load_music_catalog(filepath: str) -> pd.DataFrame:
    """
    Load, validate, and clean the Spotify music track catalog.

    WHAT: Reads track catalog CSV, verifies required columns, checks for duplicates and nulls,
    and returns a clean DataFrame.

    WHY: Music features are inputs to clustering and user profiling; missing or malformed audio
    features would crash numeric scaling or yield invalid vector calculations.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Track catalog file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Check required columns
    missing_cols = [col for col in REQUIRED_CATALOG_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Catalog missing required columns: {missing_cols}")

    # Check for duplicate track_ids
    duplicate_count = df["track_id"].duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates(subset=["track_id"], keep="first")

    # Drop rows with null audio features
    feature_cols = ["danceability", "energy", "valence", "tempo", "acousticness", "instrumentalness"]
    df = df.dropna(subset=feature_cols)

    return df.reset_index(drop=True)


def load_listening_history(filepath: str) -> pd.DataFrame:
    """
    Load and validate timestamped user listening history.

    WHAT: Reads listening event logs, parses played_at strings to datetime objects, verifies required
    columns, and ensures chronological sorting.

    WHY: Sessionization and temporal feature encodings depend strictly on chronologically ordered,
    valid ISO 8601 timestamps.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Listening history file not found: {filepath}")

    df = pd.read_csv(filepath)

    missing_cols = [col for col in REQUIRED_HISTORY_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"History log missing required columns: {missing_cols}")

    # Parse timestamps to pandas datetime
    df["played_at"] = pd.to_datetime(df["played_at"], utc=True)

    # Deduplicate composite primary key (track_id, played_at)
    df = df.drop_duplicates(subset=["track_id", "played_at"], keep="first")

    # Sort chronologically by played_at
    df = df.sort_values("played_at").reset_index(drop=True)

    return df
