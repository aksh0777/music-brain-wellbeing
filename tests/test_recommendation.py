import unittest
import numpy as np
import pandas as pd

from src.recommendation.candidate_retrieval import retrieve_candidates
from src.recommendation.similarity import (
    RECOMMENDATION_FEATURES,
    extract_user_vector,
    extract_track_matrix,
    compute_euclidean_distance,
    compute_similarity_score
)
from src.recommendation.ranking import (
    compute_profile_compatibility,
    rank_candidates,
    apply_diversity_filter
)
from src.recommendation.recommender import recommend_tracks, generate_deterministic_explanation


class TestRecommendationEngine(unittest.TestCase):

    def setUp(self):
        self.catalog_df = pd.DataFrame({
            "track_id": ["T1", "T2", "T3", "T4"],
            "track_name": ["Track 1", "Track 2", "Track 3", "Track 4"],
            "artist_name": ["Artist 1", "Artist 2", "Artist 3", "Artist 4"],
            "genre": ["Pop", "Rock", "Ambient", "Jazz"],
            "explicit": [False, True, False, False],
            "cluster_id": [0, 0, 1, 2],
            "energy": [0.8, 0.9, 0.2, 0.5],
            "valence": [0.7, 0.4, 0.3, 0.6],
            "danceability": [0.8, 0.5, 0.2, 0.6],
            "acousticness": [0.1, 0.05, 0.9, 0.6],
            "instrumentalness": [0.0, 0.1, 0.8, 0.4],
            "tempo": [120.0, 140.0, 80.0, 110.0]
        })

        self.user_profile = {
            "user_id": "USR001",
            "audio_feature_summary": {
                "energy_mean": 0.8,
                "valence_mean": 0.7,
                "danceability_mean": 0.8,
                "acousticness_mean": 0.1,
                "instrumentalness_mean": 0.0,
                "tempo_mean": 120.0
            },
            "cluster_distribution": {
                "cluster_0": 0.6,
                "cluster_1": 0.3,
                "cluster_2": 0.1
            }
        }

    def test_candidate_retrieval(self):
        candidates = retrieve_candidates(self.catalog_df, filter_criteria={"genres": ["Pop", "Rock"]})
        self.assertEqual(len(candidates), 2)

        candidates_no_explicit = retrieve_candidates(self.catalog_df, filter_criteria={"exclude_explicit": True})
        self.assertEqual(len(candidates_no_explicit), 3)

    def test_feature_vector_alignment_and_dimensions(self):
        u_vec = extract_user_vector(self.user_profile)
        t_mat = extract_track_matrix(self.catalog_df)

        self.assertEqual(u_vec.shape, (len(RECOMMENDATION_FEATURES),))
        self.assertEqual(t_mat.shape, (4, len(RECOMMENDATION_FEATURES)))

    def test_similarity_identical_and_distant(self):
        # Identical vector should have distance 0.0 and similarity 1.0
        v1 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        v2 = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]])

        dists = compute_euclidean_distance(v1, v2)
        sims = compute_similarity_score(dists)

        self.assertAlmostEqual(dists[0], 0.0)
        self.assertAlmostEqual(sims[0], 1.0)
        self.assertTrue(sims[0] > sims[1]) # Nearer vector has higher similarity

    def test_ranking_sorts_descending(self):
        sims = np.array([0.9, 0.4, 0.6, 0.8])
        profs = np.array([0.6, 0.6, 0.3, 0.1])

        ranked = rank_candidates(self.catalog_df, sims, profs, similarity_weight=0.7, profile_weight=0.3)
        self.assertEqual(len(ranked), 4)

        # Check descending order of final_score
        scores = list(ranked["final_score"])
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_n_and_explanations(self):
        recs = recommend_tracks(self.user_profile, self.catalog_df, top_n=2)
        self.assertEqual(len(recs), 2)
        self.assertIn("recommendation_reason", recs.columns)
        self.assertIn("final_score", recs.columns)

        # T1 (Pop, Cluster 0) matches user profile perfectly (energy=0.8, valence=0.7, tempo=120)
        self.assertEqual(recs["track_id"].iloc[0], "T1")

    def test_empty_catalog_handling(self):
        empty_df = pd.DataFrame()
        recs = recommend_tracks(self.user_profile, empty_df)
        self.assertTrue(recs.empty)

    def test_missing_feature_error_handling(self):
        invalid_df = pd.DataFrame({"track_id": ["T1"], "tempo": [120.0]})
        with self.assertRaises(ValueError):
            extract_track_matrix(invalid_df)


if __name__ == "__main__":
    unittest.main()
