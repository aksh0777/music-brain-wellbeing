"""
Audio Vector Feature Alignment & Similarity Calculation Module

WHAT: Extracts aligned user and track numerical feature vectors, computes Euclidean distances,
and converts distances into bounded similarity scores.

WHY: Content-based recommendation measures geometric proximity between a user's quantitative feature preferences
and candidate track audio features in standardized vector space.

FEATURE ALIGNMENT REQUIREMENT:
A vector [energy, valence, tempo_norm] is NOT equivalent to [tempo_norm, energy, valence].
Strict feature ordering prevents dimensional misalignment errors during matrix operations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from src.features.music_features import normalize_tempo

RECOMMENDATION_FEATURES = [
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
    "tempo_norm"
]


def extract_user_vector(
    user_profile: Dict[str, Any],
    feature_names: List[str] = RECOMMENDATION_FEATURES
) -> np.ndarray:
    """
    Extract a 1D numerical feature vector from a user music profile dictionary.

    WHAT: Converts user profile mean feature values into an aligned 1D numpy array.

    WHY: Aligns user preferences with the exact feature vector structure used by candidate tracks.
    """
    summary = user_profile.get("audio_feature_summary", {})
    vector = []

    for feat in feature_names:
        if feat == "tempo_norm":
            tempo_mean = summary.get("tempo_mean", 120.0)
            val = min(1.0, tempo_mean / 250.0)
        else:
            val = summary.get(f"{feat}_mean", 0.5)
        vector.append(float(val))

    return np.array(vector, dtype=float)


def extract_track_matrix(
    catalog_df: pd.DataFrame,
    feature_names: List[str] = RECOMMENDATION_FEATURES
) -> np.ndarray:
    """
    Extract a 2D numerical feature matrix from a track catalog DataFrame.

    WHAT: Ensures tempo_norm exists and extracts rows into a 2D numpy array in feature_names order.

    WHY: Guarantees exact feature alignment with the user profile vector.
    """
    df = catalog_df.copy()
    if "tempo_norm" in feature_names and "tempo_norm" not in df.columns:
        df = normalize_tempo(df)

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Catalog missing required recommendation features: {missing}")

    return df[feature_names].astype(float).values


def compute_euclidean_distance(
    user_vector: np.ndarray,
    candidate_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute Euclidean distance between a user profile vector and all candidate track vectors.

    WHAT: Calculates d(u, c_j) = sqrt( sum( (u_k - c_{jk})^2 ) ) for each candidate row j.

    WHY: Euclidean distance measures absolute geometric proximity in feature space.
    """
    if candidate_matrix.ndim == 1:
        candidate_matrix = candidate_matrix.reshape(1, -1)

    diff = candidate_matrix - user_vector
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    return distances


def compute_similarity_score(distances: np.ndarray) -> np.ndarray:
    """
    Convert Euclidean distances to similarity scores bounded in (0, 1].

    WHAT: Computes similarity = 1.0 / (1.0 + distance).

    WHY: Transforms unbounded distance metrics (0 = identical, large = distant) into an intuitive,
    monotonically decreasing similarity score where 1.0 represents perfect match.
    """
    return 1.0 / (1.0 + distances)
