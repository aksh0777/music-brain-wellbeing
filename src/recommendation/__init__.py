"""
Personalized Music Recommendation Package

WHAT: Provides candidate retrieval, feature vector similarity computation,
acoustic profile compatibility scoring, and deterministic recommendation generation.

WHY: Forms the core recommendation engine layer of the Music Brain Wellbeing Intelligence System,
enabling content-based track recommendation grounded in quantitative user listening profiles.
"""

from .candidate_retrieval import retrieve_candidates
from .similarity import compute_euclidean_distance, compute_similarity_score
from .ranking import rank_candidates, compute_profile_compatibility
from .recommender import recommend_tracks

__all__ = [
    "retrieve_candidates",
    "compute_euclidean_distance",
    "compute_similarity_score",
    "compute_profile_compatibility",
    "rank_candidates",
    "recommend_tracks"
]
