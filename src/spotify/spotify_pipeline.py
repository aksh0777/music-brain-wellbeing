"""
Spotify End-to-End Pipeline Coordinator Module

WHAT: Orchestrates Spotify data ingestion (REAL API or DEMO mock mode), data mapping,
Phase 1 feature engineering (sessionization, temporal encoding, user profiling), and Phase 2 recommendation execution.

WHY: Connects Spotify user stream data to our completed Music Intelligence and Recommendation Engine layers,
providing a clean unified interface supporting both offline testing (`DEMO` mode) and live API deployment (`REAL` mode).
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.data.music_loader import load_music_catalog
from src.features.music_features import validate_feature_ranges, normalize_tempo, extract_feature_matrix, scale_feature_matrix
from src.features.sessions import assign_listening_sessions
from src.features.temporal import generate_temporal_features
from src.features.clustering import train_kmeans_clustering
from src.features.user_profile import build_user_music_profile
from src.recommendation.recommender import recommend_tracks

from .spotify_auth import SpotifyAuth
from .spotify_client import SpotifyClient
from .spotify_mapper import map_recently_played_to_internal, AudioFeatureProvider


def generate_mock_spotify_recently_played(catalog_df: pd.DataFrame, n_tracks: int = 20) -> Dict[str, Any]:
    """
    Generate realistic mock Spotify Web API recently-played JSON response fixture.

    WHAT: Constructs a JSON structure matching Spotify's GET /v1/me/player/recently-played payload.

    WHY: Enables full offline pipeline testing without requiring live Spotify OAuth credentials.
    """
    items = []
    base_time = pd.Timestamp("2026-08-27T08:00:00Z")

    for i in range(n_tracks):
        track_row = catalog_df.iloc[i % len(catalog_df)]
        play_time = (base_time + pd.Timedelta(minutes=i * 12)).isoformat()

        items.append({
            "track": {
                "id": str(track_row["track_id"]),
                "name": str(track_row["track_name"]),
                "artists": [{"name": str(track_row["artist_name"])}],
                "album": {"name": str(track_row["album_name"])},
                "duration_ms": int(track_row["duration_ms"])
            },
            "played_at": play_time
        })

    return {"items": items}


def run_spotify_pipeline(
    mode: str = "DEMO",
    catalog_path: str = "data/raw/spotify/tracks.csv",
    mock_response: Optional[Dict[str, Any]] = None,
    top_n_recs: int = 5
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Execute end-to-end Spotify Integration Pipeline.

    WHAT: Ingests Spotify data (REAL API or DEMO mock), maps to internal schema, computes Phase 1 user profile,
    and generates Phase 2 recommendations.

    WHY: Demonstrates complete integration across Spotify API, Phase 1 Music Intelligence, and Phase 2 Recommendation Engine.
    """
    mode_upper = mode.upper()
    catalog_df = load_music_catalog(catalog_path)

    # 1. Fetch raw Spotify JSON response (REAL or DEMO)
    if mode_upper == "REAL":
        auth = SpotifyAuth()
        client = SpotifyClient(auth=auth)
        spotify_json = client.get_recently_played(limit=50)
        source_label = "spotify_api"
        data_type_label = "real"
    else:
        # DEMO mode
        if mock_response is not None:
            spotify_json = mock_response
        else:
            spotify_json = generate_mock_spotify_recently_played(catalog_df, n_tracks=20)
        source_label = "spotify_api_mock"
        data_type_label = "synthetic/demo"

    # 2. Map Spotify JSON to internal DataFrame schema
    history_df = map_recently_played_to_internal(
        spotify_json,
        user_id="USR_SPOTIFY_001",
        source=source_label,
        data_type=data_type_label
    )

    if history_df.empty:
        return pd.DataFrame(), {}, pd.DataFrame()

    # 3. Enrich history with audio features via AudioFeatureProvider
    enriched_history = AudioFeatureProvider.enrich_tracks_with_features(history_df, catalog_df)

    # 4. Connect to Phase 1 Feature Pipeline (Sessions, Temporal, Clustering, User Profile)
    session_df = assign_listening_sessions(enriched_history, gap_minutes=30)
    temporal_df = generate_temporal_features(session_df)

    # Prepare catalog clustering baseline for profile matching (K=4)
    catalog_clean = validate_feature_ranges(catalog_df)
    catalog_clean = normalize_tempo(catalog_clean)
    X_raw, feat_cols = extract_feature_matrix(catalog_clean)
    X_scaled, _ = scale_feature_matrix(X_raw)
    n_clusters = min(4, max(1, len(catalog_clean)))
    kmeans_model, cluster_labels = train_kmeans_clustering(X_scaled, n_clusters=n_clusters)
    catalog_clean["cluster_id"] = cluster_labels

    user_profile = build_user_music_profile(temporal_df, catalog_clean, cluster_labels=cluster_labels)

    # 5. Connect to Phase 2 Recommendation Engine
    recommendations = recommend_tracks(
        user_profile=user_profile,
        catalog_df=catalog_clean,
        top_n=top_n_recs
    )

    return temporal_df, user_profile, recommendations
