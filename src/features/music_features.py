"""
Music Audio Feature Processing Module

WHAT: Audio feature range validation, tempo normalization, feature matrix extraction,
and standardized scaling for track audio descriptors.

WHY: Raw audio descriptors exist on mixed scales (e.g. tempo in 40-220 BPM vs acousticness in 0.0-1.0).
Standardization and range checks ensure distance metrics in clustering and similarity algorithms
are not artificially dominated by large-magnitude features.

SCIENTIFIC BOUNDARY:
These features describe physical and perceptual audio properties of music (acoustics, rhythm, spectral energy).
They are NOT direct measurements of clinical anxiety, mood disorders, or psychological diagnoses.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler

CORE_AUDIO_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness"
]

BOUNDED_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness"
]


def validate_feature_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that numerical audio features conform to expected physiological/audio bounds.

    WHAT: Ensures bounded audio features lie in [0.0, 1.0] and tempo lies in [30.0, 250.0] BPM.
    Clipped out-of-bounds values to legal limits.

    WHY: Out-of-bounds inputs distort standard scalers and clustering centroids.
    """
    df_clean = df.copy()

    for col in BOUNDED_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=0.0, upper=1.0)

    if "tempo" in df_clean.columns:
        df_clean["tempo"] = df_clean["tempo"].clip(lower=30.0, upper=250.0)

    return df_clean


def normalize_tempo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized tempo column scaled into [0.0, 1.0].

    WHAT: Adds 'tempo_norm' column computed as tempo / 250.0.

    WHY: Division by 250.0 rescales tempo to match the [0, 1] range of Spotify's bounded audio features.
    """
    df_out = df.copy()
    if "tempo" in df_out.columns:
        df_out["tempo_norm"] = (df_out["tempo"] / 250.0).clip(upper=1.0)
    return df_out


def extract_feature_matrix(
    df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract a numerical feature matrix from a track catalog DataFrame.

    WHAT: Selects numerical feature columns and returns a 2D numpy array.

    WHY: Scikit-learn algorithms (K-Means, PCA, Scaler) require clean 2D NumPy arrays as inputs.
    """
    if features is None:
        features = ["valence", "energy", "danceability", "acousticness", "instrumentalness", "tempo_norm"]

    missing = [f for f in features if f not in df.columns]
    if missing:
        # If tempo_norm requested but not present, attempt normalize_tempo
        if "tempo_norm" in missing and "tempo" in df.columns:
            df = normalize_tempo(df)
            missing = [f for f in features if f not in df.columns]

    if missing:
        raise ValueError(f"Requested features missing from DataFrame: {missing}")

    X = df[features].astype(float).values
    return X, features


def scale_feature_matrix(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize features by subtracting mean and scaling to unit variance.

    WHAT: Applies StandardScaler to feature matrix X.

    WHY: K-Means uses Euclidean distance. Standardization gives zero mean and unit variance to all features,
    preventing any single dimension from dominating distance calculations.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
