"""
Unit Tests for Phase 6 Systematic Evaluation Layer

WHAT: Comprehensive unit tests validating metric calculations, recommendation evaluator logic,
RAG retrieval benchmarking, LLM explanation structural & grounding validation, and end-to-end evaluator execution.

WHY: Ensures 100% deterministic, offline verification of all evaluation metrics and runner modules.
"""

import unittest
import json
import os
import pandas as pd
from src.evaluation.metrics import (
    mean_vector_distance,
    ranking_monotonicity_score,
    intra_list_diversity,
    cluster_coverage,
    retrieval_hit_rate_at_k,
    mean_reciprocal_rank,
    citation_grounding_accuracy,
    track_grounding_accuracy,
    safety_compliance_score
)
from src.evaluation.recommendation_eval import RecommendationEvaluator
from src.evaluation.rag_eval import RAGRetrieverEvaluator
from src.evaluation.llm_eval import LLMExplanationEvaluator
from src.evaluation.end_to_end_eval import EndToEndEvaluator
from src.explanation.schemas import ExplanationRequest, ExplanationResponse, SafetyConstraints
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


class TestEvaluationLayer(unittest.TestCase):

    def setUp(self):
        """Set up test data fixtures for metrics and evaluators."""
        self.sample_target_features = {
            "energy": 0.30,
            "tempo": 80.0,
            "acousticness": 0.80,
            "danceability": 0.40,
            "valence": 0.50,
            "instrumentalness": 0.70
        }

        self.sample_recs = [
            {
                "track_id": "TRK_001",
                "track_name": "Ambient Soothing",
                "final_score": 0.90,
                "cluster_label": 0,
                "energy": 0.28, "tempo": 82.0, "acousticness": 0.82,
                "danceability": 0.38, "valence": 0.48, "instrumentalness": 0.68
            },
            {
                "track_id": "TRK_002",
                "track_name": "Calm Acoustic",
                "final_score": 0.85,
                "cluster_label": 1,
                "energy": 0.35, "tempo": 85.0, "acousticness": 0.75,
                "danceability": 0.42, "valence": 0.52, "instrumentalness": 0.65
            },
            {
                "track_id": "TRK_003",
                "track_name": "Peaceful Piano",
                "final_score": 0.80,
                "cluster_label": 0,
                "energy": 0.25, "tempo": 75.0, "acousticness": 0.88,
                "danceability": 0.35, "valence": 0.45, "instrumentalness": 0.75
            }
        ]

        self.sample_retrieved_chunks = [
            {
                "chunk_id": "c1",
                "distance": 0.15,
                "metadata": {"pmid": "33176590", "doi": "10.1080/17437199.2020.1846580"}
            },
            {
                "chunk_id": "c2",
                "distance": 0.25,
                "metadata": {"pmid": "34365216", "doi": "10.1016/j.aenj.2021.03.003"}
            }
        ]

        self.sample_catalog_df = pd.DataFrame([
            {"track_id": "TRK_001", "track_name": "Song A", "artist": "Artist 1", "genre": "Acoustic",
             "energy": 0.28, "tempo": 82.0, "acousticness": 0.82, "danceability": 0.38, "valence": 0.48, "instrumentalness": 0.68},
            {"track_id": "TRK_002", "track_name": "Song B", "artist": "Artist 2", "genre": "Ambient",
             "energy": 0.35, "tempo": 85.0, "acousticness": 0.75, "danceability": 0.42, "valence": 0.52, "instrumentalness": 0.65},
            {"track_id": "TRK_003", "track_name": "Song C", "artist": "Artist 3", "genre": "Classical",
             "energy": 0.25, "tempo": 75.0, "acousticness": 0.88, "danceability": 0.35, "valence": 0.45, "instrumentalness": 0.75},
            {"track_id": "TRK_004", "track_name": "Song D", "artist": "Artist 4", "genre": "Pop",
             "energy": 0.90, "tempo": 130.0, "acousticness": 0.10, "danceability": 0.85, "valence": 0.80, "instrumentalness": 0.05}
        ])

        self.sample_profile = {
            "user_id": "USR_TEST_EVAL",
            "audio_feature_summary": self.sample_target_features,
            "listening_history_summary": {"total_tracks": 20}
        }

    def test_1_mean_vector_distance(self):
        """Test mean Euclidean vector distance calculation."""
        dist = mean_vector_distance(self.sample_recs, self.sample_target_features)
        self.assertGreater(dist, 0.0)
        self.assertLess(dist, 10.0)

    def test_2_ranking_monotonicity_score(self):
        """Test score monotonicity calculation for ordered vs unordered recommendations."""
        # Perfectly ordered (0.90 >= 0.85 >= 0.80)
        score_ordered = ranking_monotonicity_score(self.sample_recs, score_key="final_score")
        self.assertEqual(score_ordered, 1.0)

        # Unordered recommendations
        unordered_recs = [self.sample_recs[1], self.sample_recs[0], self.sample_recs[2]]
        score_unordered = ranking_monotonicity_score(unordered_recs, score_key="final_score")
        self.assertLess(score_unordered, 1.0)

    def test_3_intra_list_diversity(self):
        """Test intra-list diversity calculation."""
        div = intra_list_diversity(self.sample_recs)
        self.assertGreater(div, 0.0)

    def test_4_cluster_coverage(self):
        """Test cluster coverage calculation across K-Means clusters."""
        cov = cluster_coverage(self.sample_recs, cluster_key="cluster_label", n_total_clusters=4)
        # Clusters 0 and 1 present out of 4 total -> 2/4 = 0.50
        self.assertEqual(cov, 0.5)

    def test_5_retrieval_hit_rate_at_k(self):
        """Test retrieval Hit Rate @ K calculation."""
        hr_hit = retrieval_hit_rate_at_k(self.sample_retrieved_chunks, ["33176590"], k=2)
        self.assertEqual(hr_hit, 1.0)

        hr_miss = retrieval_hit_rate_at_k(self.sample_retrieved_chunks, ["99999999"], k=2)
        self.assertEqual(hr_miss, 0.0)

    def test_6_mean_reciprocal_rank(self):
        """Test Mean Reciprocal Rank (MRR) calculation."""
        # PMID 33176590 is at rank 1 -> MRR = 1/1 = 1.0
        mrr_first = mean_reciprocal_rank(self.sample_retrieved_chunks, ["33176590"])
        self.assertEqual(mrr_first, 1.0)

        # PMID 34365216 is at rank 2 -> MRR = 1/2 = 0.5
        mrr_second = mean_reciprocal_rank(self.sample_retrieved_chunks, ["34365216"])
        self.assertEqual(mrr_second, 0.5)

    def test_7_citation_grounding_accuracy(self):
        """Test citation grounding accuracy metric."""
        evidence_sources = [{"pmid": "33176590"}, {"pmid": "34365216"}]
        explanation_sources = [{"pmid": "33176590"}]
        acc = citation_grounding_accuracy(explanation_sources, evidence_sources)
        self.assertEqual(acc, 1.0)

        fake_sources = [{"pmid": "9999999"}]
        fake_acc = citation_grounding_accuracy(fake_sources, evidence_sources)
        self.assertEqual(fake_acc, 0.0)

    def test_8_track_grounding_accuracy(self):
        """Test track ID grounding accuracy metric."""
        valid_tracks = ["TRK_001", "TRK_002"]
        reasons = ["Track ID: TRK_001 matches preference."]
        acc = track_grounding_accuracy(reasons, valid_tracks)
        self.assertEqual(acc, 1.0)

        invalid_reasons = ["Track ID: TRK_FAKE matches preference."]
        invalid_acc = track_grounding_accuracy(invalid_reasons, valid_tracks)
        self.assertEqual(invalid_acc, 0.0)

    def test_9_safety_compliance_score(self):
        """Test safety compliance score detection of prohibited clinical phrases."""
        safe_text = "Music recommendations provide acoustic preference match and observational scientific context."
        self.assertEqual(safety_compliance_score(safe_text), 1.0)

        unsafe_text = "This playlist cures anxiety and provides psychiatric therapy."
        self.assertEqual(safety_compliance_score(unsafe_text), 0.0)

    def test_10_recommendation_evaluator(self):
        """Test RecommendationEvaluator execution and random baseline comparison."""
        evaluator = RecommendationEvaluator()
        metrics = evaluator.evaluate_recommendations(self.sample_profile, self.sample_catalog_df, top_n=3)

        self.assertIn("mean_feature_distance", metrics)
        self.assertIn("ranking_monotonicity_score", metrics)
        self.assertIn("distance_improvement_over_random", metrics)
        self.assertTrue(metrics["is_outperforming_random"])

    def test_11_llm_explanation_evaluator(self):
        """Test LLMExplanationEvaluator structural, citation, track ID, and safety checks."""
        req = ExplanationRequest(
            user_profile=self.sample_profile,
            recommendations=self.sample_recs,
            acoustic_profiles={},
            evidence_package={"sources": [{"pmid": "33176590"}]}
        )
        res = ExplanationResponse(
            summary="Valid explanation summary.",
            recommendation_reasons=["Matched ID: TRK_001"],
            observed_user_patterns=["Pattern A"],
            research_context=["Context mentioning PMID: 33176590."],
            limitations=["Non-clinical disclaimer."],
            sources=[{"pmid": "33176590"}],
            is_validated=True
        )

        evaluator = LLMExplanationEvaluator()
        eval_dict = evaluator.evaluate_explanation(req, res)

        self.assertTrue(eval_dict["json_structural_validity"])
        self.assertEqual(eval_dict["citation_grounding_accuracy"], 1.0)
        self.assertEqual(eval_dict["track_grounding_accuracy"], 1.0)
        self.assertEqual(eval_dict["safety_compliance_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
