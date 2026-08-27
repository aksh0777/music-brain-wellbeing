"""
Top-N Personalized Music Recommender Engine

WHAT: End-to-end recommendation pipeline combining candidate retrieval, standardized feature vector similarity,
acoustic profile compatibility, weighted ranking, diversity filtering, and deterministic machine-readable explanations.

WHY: Provides an explainable, content-based recommendation service that powers downstream AI/LLM explanations
and FastAPI endpoints.

DECOUPLED DESIGN:
    Recommendation Computation (Deterministic Math)
                       ↓
    Machine-Readable Explanation (Python Formatting)
                       ↓
    [Future] LLM Natural Language Generation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.features.music_features import scale_feature_matrix, validate_feature_ranges, normalize_tempo
from .candidate_retrieval import retrieve_candidates
from .similarity import (
    RECOMMENDATION_FEATURES,
    extract_user_vector,
    extract_track_matrix,
    compute_euclidean_distance,
    compute_similarity_score
)
from .ranking import compute_profile_compatibility, rank_candidates, apply_diversity_filter


def generate_deterministic_explanation(
    track_row: pd.Series,
    user_profile: Dict[str, Any]
) -> str:
    """
    Generate a deterministic machine-readable explanation string for a recommended track.

    WHAT: Formats track metrics and similarity scores into a human-readable explanation sentence.

    WHY: Separates deterministic recommendation logic from natural language LLM generation.
    """
    sim_score = track_row.get("similarity_score", 0.0)
    final_score = track_row.get("final_score", 0.0)
    genre = track_row.get("genre", "Music")
    energy = track_row.get("energy", 0.5)

    energy_desc = "high energy" if energy >= 0.65 else ("low energy" if energy <= 0.35 else "moderate energy")

    explanation = (
        f"Recommended {genre} track with {energy_desc} acoustics. "
        f"Matches your historical profile with similarity score {sim_score:.2f} "
        f"and combined score {final_score:.2f}."
    )
    return explanation


def recommend_tracks(
    user_profile: Dict[str, Any],
    catalog_df: pd.DataFrame,
    context: Optional[Dict[str, Any]] = None,
    top_n: int = 10,
    similarity_weight: float = 0.7,
    profile_weight: float = 0.3,
    enable_diversity: bool = True,
    max_per_cluster: int = 4
) -> pd.DataFrame:
    """
    Generate top-N personalized music recommendations for a given user profile.

    WHAT: Full recommendation pipeline: Candidate Retrieval -> Vector Extraction -> Euclidean Similarity ->
    Profile Compatibility -> Weighted Ranking -> Diversity Filtering -> Explanation Formatting.

    WHY: Serves as the primary recommendation entry point for the Music Brain Wellbeing System.
    """
    if catalog_df.empty:
        return pd.DataFrame()

    # Step 1: Candidate Retrieval (Apply optional context filter)
    candidates = retrieve_candidates(catalog_df, filter_criteria=context)
    if candidates.empty:
        return pd.DataFrame()

    # Step 2: Validate ranges & normalize tempo
    candidates_clean = validate_feature_ranges(candidates)
    candidates_clean = normalize_tempo(candidates_clean)

    # Step 3: Extract aligned user vector and track matrix
    user_vec = extract_user_vector(user_profile, feature_names=RECOMMENDATION_FEATURES)
    track_mat = extract_track_matrix(candidates_clean, feature_names=RECOMMENDATION_FEATURES)

    # Step 4: Scale feature matrix and user vector using common StandardScaler
    combined_mat = np.vstack([user_vec, track_mat])
    scaled_combined, _ = scale_feature_matrix(combined_mat)

    user_vec_scaled = scaled_combined[0]
    track_mat_scaled = scaled_combined[1:]

    # Step 5: Compute Euclidean Distance & Similarity Score
    distances = compute_euclidean_distance(user_vec_scaled, track_mat_scaled)
    similarity_scores = compute_similarity_score(distances)

    # Step 6: Compute Acoustic Profile Compatibility
    profile_scores = compute_profile_compatibility(candidates_clean, user_profile)

    # Step 7: Weighted Candidate Ranking
    ranked_df = rank_candidates(
        candidates_clean,
        similarity_scores=similarity_scores,
        profile_scores=profile_scores,
        similarity_weight=similarity_weight,
        profile_weight=profile_weight
    )

    # Step 8: Apply Post-Ranking Diversity Filter if enabled
    if enable_diversity and "cluster_id" in ranked_df.columns:
        final_recs = apply_diversity_filter(ranked_df, top_n=top_n, max_per_cluster=max_per_cluster)
    else:
        final_recs = ranked_df.head(top_n).copy()

    # Step 9: Add Deterministic Machine-Readable Explanations
    explanations = [
        generate_deterministic_explanation(row, user_profile)
        for _, row in final_recs.iterrows()
    ]
    final_recs["recommendation_reason"] = explanations

    return final_recs.reset_index(drop=True)
