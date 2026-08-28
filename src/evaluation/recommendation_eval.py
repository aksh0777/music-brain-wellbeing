"""
Recommendation Engine Evaluator Module

WHAT: Systematically evaluates Phase 2 & Phase 3 personalized recommendation engine performance.

WHY: Measures quantitative vector alignment, score monotonicity, intra-list diversity, cluster coverage,
duplicate rate, and baseline comparison against random recommendations without fabricating ground-truth user feedback.

EXCLUDED METRICS & JUSTIFICATION:
    - Precision / Recall / F1-Score: Excluded because Spotify listening logs lack user relevance feedback annotations.
    - NDCG / MAP: Excluded because user preference targets are continuous audio feature vectors, not ground-truth relevancy ranks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.recommendation.recommender import recommend_tracks
from .metrics import (
    mean_vector_distance,
    ranking_monotonicity_score,
    intra_list_diversity,
    cluster_coverage
)


class RecommendationEvaluator:
    """
    Evaluator executing quantitative performance measurements on recommendation outputs.
    """

    FEATURE_KEYS = ["energy", "tempo", "acousticness", "danceability", "valence", "instrumentalness"]

    def evaluate_recommendations(
        self,
        user_profile: Dict[str, Any],
        catalog_df: pd.DataFrame,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate recommendation performance for a given user profile.

        Args:
            user_profile: Quantitative user profile dictionary from Phase 1.
            catalog_df: Spotify track catalog DataFrame.
            top_n: Number of recommendations to retrieve.

        Returns:
            Dictionary containing evaluation metrics and random baseline comparison.
        """
        # Generate model recommendations
        recs_df = recommend_tracks(user_profile, catalog_df, top_n=top_n)
        recs_list = recs_df.to_dict(orient="records")

        # Target feature vector
        target_features = user_profile.get("audio_feature_summary", {})

        # 1. Mean Vector Distance (Model)
        model_distance = mean_vector_distance(recs_list, target_features, self.FEATURE_KEYS)

        # 2. Ranking Monotonicity Score
        monotonicity = ranking_monotonicity_score(recs_list, score_key="final_score")

        # 3. Intra-List Diversity
        diversity = intra_list_diversity(recs_list, self.FEATURE_KEYS)

        # 4. Cluster Coverage (K=4)
        coverage = cluster_coverage(recs_list, cluster_key="cluster_label", n_total_clusters=4)

        # 5. Duplicate Rate
        track_ids = [r.get("track_id") for r in recs_list if r.get("track_id")]
        dup_rate = round(1.0 - (len(set(track_ids)) / float(len(track_ids))), 4) if track_ids else 0.0

        # 6. Random Baseline Comparison
        np.random.seed(42)
        sample_n = min(top_n, len(catalog_df))
        random_sample_df = catalog_df.sample(n=sample_n, random_state=42)
        random_list = random_sample_df.to_dict(orient="records")
        random_distance = mean_vector_distance(random_list, target_features, self.FEATURE_KEYS)

        distance_improvement = round(random_distance - model_distance, 4)

        return {
            "top_n": len(recs_list),
            "mean_feature_distance": model_distance,
            "ranking_monotonicity_score": monotonicity,
            "intra_list_diversity": diversity,
            "cluster_coverage": coverage,
            "duplicate_rate": dup_rate,
            "random_baseline_distance": random_distance,
            "distance_improvement_over_random": distance_improvement,
            "is_outperforming_random": (distance_improvement > 0.0)
        }
