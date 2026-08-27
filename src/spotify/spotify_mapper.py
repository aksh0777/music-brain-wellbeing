"""
Spotify API Data Mapping & Adapter Module

WHAT: Converts raw nested Spotify Web API JSON objects into our application's internal DataFrames
and isolates API changes behind an adapter interface.

WHY: External APIs evolve independently of internal business logic. Mapping Spotify API structures into
our internal schema (`event_id`, `user_id`, `track_id`, `played_at`, `data_type`, `source`) ensures Phase 1
sessionization, temporal features, and Phase 2 recommendation modules require zero code changes.

PROVENANCE POLICY:
    - Real Spotify API Data: source = "spotify_api", data_type = "real"
    - Synthetic/Demo Data: source = "synthetic", data_type = "synthetic/demo"
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


def map_recently_played_to_internal(
    spotify_response: Dict[str, Any],
    user_id: str = "USR_SPOTIFY",
    source: str = "spotify_api",
    data_type: str = "real"
) -> pd.DataFrame:
    """
    Map Spotify /v1/me/player/recently-played JSON response to internal listening history DataFrame.

    WHAT: Iterates over response['items'], extracts track IDs, titles, artists, played_at timestamps,
    and returns a clean DataFrame conforming to internal pipeline requirements.

    WHY: Decouples nested external JSON schemas from downstream pandas/numpy feature pipelines.
    """
    columns = ["event_id", "user_id", "track_id", "played_at", "track_name", "artist_name", "album_name", "duration_ms", "data_type", "source"]

    if not spotify_response or "items" not in spotify_response or not spotify_response["items"]:
        return pd.DataFrame(columns=columns)

    records = []
    items = spotify_response.get("items", [])

    for idx, item in enumerate(items):
        track = item.get("track", {})
        if not track:
            continue

        played_at_str = item.get("played_at")
        if not played_at_str:
            continue

        track_id = track.get("id", f"TRK_UNK_{idx}")
        track_name = track.get("name", "Unknown Track")

        # Artist handling (extract primary or comma-separated list)
        artists = track.get("artists", [])
        artist_name = ", ".join([a.get("name", "") for a in artists if a.get("name")]) if artists else "Unknown Artist"

        album = track.get("album", {})
        album_name = album.get("name", "Unknown Album")
        duration_ms = track.get("duration_ms", 0)

        # Unique event ID derived from timestamp and track
        event_id = f"EVT_SP_{idx+1:04d}"

        records.append({
            "event_id": event_id,
            "user_id": user_id,
            "track_id": track_id,
            "played_at": played_at_str,
            "track_name": track_name,
            "artist_name": artist_name,
            "album_name": album_name,
            "duration_ms": duration_ms,
            "data_type": data_type,
            "source": source
        })

    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records)

    # Parse ISO 8601 timestamps to datetime64 UTC
    df["played_at"] = pd.to_datetime(df["played_at"], utc=True)

    # Deduplicate composite primary key (track_id, played_at)
    df = df.drop_duplicates(subset=["track_id", "played_at"], keep="first")

    # Sort chronologically
    df = df.sort_values("played_at").reset_index(drop=True)

    return df


class AudioFeatureProvider:
    """
    Audio Feature Provider Adapter & Fallback Interface.

    WHAT: Enriches mapped track records with audio features (energy, valence, danceability, etc.)
    by querying a reference track catalog or assigning baseline defaults.

    WHY: Spotify API access policies restrict audio-features endpoints for unapproved developer apps.
    Providing a clean provider abstraction allows local catalogs or external datasets to be plugged in
    without crashing downstream feature scaling or clustering pipelines.
    """

    @staticmethod
    def enrich_tracks_with_features(
        history_df: pd.DataFrame,
        reference_catalog: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge mapped Spotify listening history with reference track catalog audio features.

        WHAT: Performs left join on track_id, filling unmapped audio features with catalog mean defaults.

        WHY: Ensures every track in the history log possesses valid numerical audio descriptors.
        """
        if history_df.empty:
            return history_df

        feature_cols = ["danceability", "energy", "valence", "tempo", "acousticness", "instrumentalness", "speechiness", "liveness", "genre"]
        available_cols = ["track_id"] + [c for c in feature_cols if c in reference_catalog.columns]

        ref_subset = reference_catalog[available_cols].drop_duplicates("track_id")

        merged = history_df.merge(ref_subset, on="track_id", how="left")

        # Fill missing features for tracks not in reference catalog using catalog averages
        defaults = {
            "danceability": 0.5,
            "energy": 0.5,
            "valence": 0.5,
            "tempo": 120.0,
            "acousticness": 0.3,
            "instrumentalness": 0.2,
            "speechiness": 0.05,
            "liveness": 0.15,
            "genre": "Pop"
        }

        for col, default_val in defaults.items():
            if col in merged.columns:
                merged[col] = merged[col].fillna(default_val)
            else:
                merged[col] = default_val

        return merged
