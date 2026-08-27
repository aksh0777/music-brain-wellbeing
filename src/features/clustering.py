"""
Music Acoustic Profile Clustering Engine

WHAT: K-Means clustering workflow for discovering natural acoustic feature groups in track catalogs,
including candidate K evaluation (Inertia/Elbow & Silhouette Score) and centroid interpretation.

WHY: Music tracks can be grouped by audio properties (tempo, energy, acousticness, valence) without relying
on manual genre labels. Unsupervised clustering builds an acoustic feature baseline for recommendation.

SCIENTIFIC BOUNDARY:
Cluster labels are strictly **"Acoustic Profiles"** describing physical audio structure.
They must NOT be interpreted as clinical diagnoses, psychological mood states, or medical classifications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluate_k_candidates(
    X_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42
) -> Tuple[List[int], List[float], List[float]]:
    """
    Evaluate candidate K values using Inertia (Elbow method) and Silhouette Scores.

    WHAT: Fits K-Means for each K in [k_min, k_max], recording inertia and silhouette score.

    WHY: Silhouette score measures how similar an instance is to its own cluster compared to other
    clusters (range -1 to +1). Inertia measures total intra-cluster sum of squared distances.
    Comparing these metrics identifies an optimal K value.
    """
    k_values = list(range(k_min, k_max + 1))
    inertias = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        inertias.append(float(kmeans.inertia_))

        if len(np.unique(labels)) > 1:
            score = float(silhouette_score(X_scaled, labels))
        else:
            score = -1.0
        silhouette_scores.append(score)

    return k_values, inertias, silhouette_scores


def train_kmeans_clustering(
    X_scaled: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray]:
    """
    Fit K-Means clustering model with chosen n_clusters.

    WHAT: Fits KMeans model and returns (model, cluster_labels).

    WHY: Assigns each track in the dataset to its nearest acoustic cluster centroid.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels


def generate_cluster_labels(
    centroids_unscaled: np.ndarray,
    feature_names: List[str]
) -> List[str]:
    """
    Generate human-readable Acoustic Profile descriptions based on centroid coordinates.

    WHAT: Translates raw centroid feature averages into acoustic descriptor labels.

    WHY: Provides clear human-interpretable audio structural descriptions (e.g. 'High Energy / Acoustic').
    """
    labels = []
    feat_map = {name: idx for idx, name in enumerate(feature_names)}

    for i, centroid in enumerate(centroids_unscaled):
        energy = centroid[feat_map["energy"]] if "energy" in feat_map else 0.5
        valence = centroid[feat_map["valence"]] if "valence" in feat_map else 0.5
        acousticness = centroid[feat_map["acousticness"]] if "acousticness" in feat_map else 0.5
        danceability = centroid[feat_map["danceability"]] if "danceability" in feat_map else 0.5
        instrumentalness = centroid[feat_map["instrumentalness"]] if "instrumentalness" in feat_map else 0.5

        descriptors = []

        # Energy descriptor
        if energy >= 0.65:
            descriptors.append("High Energy")
        elif energy <= 0.35:
            descriptors.append("Low Energy")

        # Valence descriptor
        if valence >= 0.6:
            descriptors.append("Bright/Upbeat")
        elif valence <= 0.35:
            descriptors.append("Subdued/Mellow")

        # Acousticness / Instrumentalness
        if acousticness >= 0.6:
            descriptors.append("Acoustic")
        elif instrumentalness >= 0.6:
            descriptors.append("Instrumental")
        elif danceability >= 0.65:
            descriptors.append("Rhythmic")

        if not descriptors:
            descriptors.append("Balanced Acoustic")

        profile_name = f"Cluster {i}: " + ", ".join(descriptors)
        labels.append(profile_name)

    return labels


def assign_nearest_cluster(
    X_new_scaled: np.ndarray,
    kmeans_model: KMeans
) -> np.ndarray:
    """
    Assign new track feature vectors to the nearest existing cluster centroid.

    WHAT: Calculates Euclidean distance to fitted cluster centroids and returns cluster index.

    WHY: Enables incremental track labeling without re-fitting K-Means on the entire historical dataset.
    """
    return kmeans_model.predict(X_new_scaled)
