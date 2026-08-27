"""
Recommendation Ranking & Acoustic Profile Compatibility Module

WHAT: Combines audio similarity scores and acoustic profile compatibility shares to calculate
final recommendation scores and rank candidate tracks.

WHY: Content-based recommendation is enhanced when incorporating cluster-level user habit distribution
alongside continuous audio feature similarity.

RANKING SCORE FORMULA:
    final_score = (similarity_weight * similarity_score) + (profile_weight * profile_compatibility)
Default weights: similarity_weight = 0.7, profile_weight = 0.3.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List


def compute_profile_compatibility(
    catalog_df: pd.DataFrame,
    user_profile: Dict[str, Any]
) -> np.ndarray:
    """
    Compute acoustic profile compatibility score for each track based on user's cluster listening shares.

    WHAT: Checks track's cluster_id against user's cluster_distribution dictionary.

    WHY: Tracks belonging to acoustic clusters that the user frequently listens to receive a compatibility bonus.
    """
    n_tracks = len(catalog_df)
    if n_tracks == 0:
        return np.array([])

    cluster_dist = user_profile.get("cluster_distribution", {})

    if "cluster_id" not in catalog_df.columns or not cluster_dist:
        # Default neutral compatibility if cluster data is unassigned
        return np.full(n_tracks, 0.5)

    scores = []
    for _, row in catalog_df.iterrows():
        c_id = row.get("cluster_id")
        key = f"cluster_{c_id}"
        score = cluster_dist.get(key, 0.0)
        scores.append(float(score))

    return np.array(scores)


def rank_candidates(
    catalog_df: pd.DataFrame,
    similarity_scores: np.ndarray,
    profile_scores: np.ndarray,
    similarity_weight: float = 0.7,
    profile_weight: float = 0.3
) -> pd.DataFrame:
    """
    Combine similarity and profile scores into a final recommendation score and rank tracks.

    WHAT: Computes final_score = (w_sim * similarity) + (w_prof * profile_comp) and sorts descending.

    WHY: Balances continuous audio feature similarity with discrete acoustic habit preference.
    """
    if len(catalog_df) == 0:
        return catalog_df.copy()

    df_ranked = catalog_df.copy()
    df_ranked["similarity_score"] = np.round(similarity_scores, 4)
    df_ranked["profile_score"] = np.round(profile_scores, 4)

    # Weighted recommendation score
    final_scores = (similarity_weight * similarity_scores) + (profile_weight * profile_scores)
    df_ranked["final_score"] = np.round(final_scores, 4)

    # Sort descending by final score
    df_ranked = df_ranked.sort_values("final_score", ascending=False).reset_index(drop=True)
    return df_ranked


def apply_diversity_filter(
    ranked_df: pd.DataFrame,
    top_n: int = 10,
    max_per_cluster: int = 4
) -> pd.DataFrame:
    """
    Apply a simple post-ranking diversity filter to cap maximum tracks per acoustic cluster.

    WHAT: Iterates through ranked tracks and limits tracks from the same cluster_id to max_per_cluster.

    WHY: Pure top-N sorting can recommend 10 near-identical tracks from a single cluster.
    A diversity constraint ensures variety across recommendations.
    """
    if "cluster_id" not in ranked_df.columns or len(ranked_df) <= top_n:
        return ranked_df.head(top_n).copy()

    selected_rows = []
    cluster_counts: Dict[Any, int] = {}

    for _, row in ranked_df.iterrows():
        c_id = row["cluster_id"]
        current_count = cluster_counts.get(c_id, 0)

        if current_count < max_per_cluster:
            selected_rows.append(row)
            cluster_counts[c_id] = current_count + 1

        if len(selected_rows) == top_n:
            break

    # If diversity filter produces fewer than top_n, backfill from remaining ranked tracks
    if len(selected_rows) < top_n:
        selected_ids = {r["track_id"] for r in selected_rows}
        for _, row in ranked_df.iterrows():
            if row["track_id"] not in selected_ids:
                selected_rows.append(row)
                if len(selected_rows) == top_n:
                    break

    return pd.DataFrame(selected_rows).reset_index(drop=True)
